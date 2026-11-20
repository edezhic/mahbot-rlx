// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One copy of a weight across CPU executables.
//!
//! Each executable owns one arena and every kernel addresses it by offset, so
//! two executables of one model (a prefill graph, a decode graph per KV
//! bucket) each held a full copy of the weights. Under a sharing scope
//! ([`set_scope`], one label per model's weights) the arena is a page-aligned
//! anonymous mapping, and each read-only weight's page-aligned slot is backed by a
//! file of its own, kept in a process-wide pool by `(scope, name, bytes)`: the
//! first executable to place a weight makes the file, every later one maps the
//! same pages at its own offset. Kernels still see one buffer and the same offsets.
//!
//! The scope is the compiling **thread's**, while the pool it names is the process's.
//! A scope is what a compile is told its weights belong to, and a compile runs on the thread
//! that declared it (`with_param_sharing_scope` and `Runner::compile` both set it there), so
//! reading it per thread is exact — and a process-wide label is not. Under one, a graph
//! compiled for no model at all inherited whatever label was set last, put its parameters in
//! that model's pool by name and length, and two such graphs overwrote each other's weights:
//! measured on `rlx-runtime`'s own suite, the fused-optimizer tests disagreed with their host
//! reference in 1-2 of 362 cases per run with the label shared, and never with
//! `RLX_CPU_SHARED_WEIGHTS=0`.
//!
//! The weights live in files rather than in shared-memory objects because of what a
//! page costs a machine that is short of memory: a shared-memory page is
//! swap-backed, so a model's bytes are written to the swap file when the kernel
//! wants the memory back, while a file's pages are written back once and are then
//! clean — reclaimable without touching swap. The arena stays anonymous, so the
//! footprint a row reports is the memory it really holds.
//!
//! The pool is held by the process, not by the executables that map it: a
//! weight the last graph of a scope dropped is still the model's bytes, and the
//! next graph of that scope would otherwise read them off disk again. That is
//! what the graphics backend's shared weight pool buys, and on the processor it
//! is the difference between a second call that reads the model and one that
//! does not. `POOL_BYTES` bounds it to one model's weights; [`clear`] gives the
//! pages back.
//!
//! Each pooled weight holds one open descriptor for as long as the pool does — the file is
//! unlinked at birth, so the descriptor is what keeps it alive for the next mapping. That is
//! one per tensor of one model, a few hundred, and whether that fits `RLIMIT_NOFILE` is the
//! host's business, not this module's: an interactive shell on this platform raises the soft
//! limit into the thousands, while a daemon started by launchd inherits 256. Past whatever the
//! process may hold, `create_new` fails, the weight is not pooled, and it is private to its
//! executable instead — correct, and costing what the pool exists to save: the next graph of
//! the same scope reads those bytes again and holds its own copy of them. A daemon that keeps
//! several model families at once and wants them shared raises its own soft limit; this module
//! does not raise a process-wide resource limit on a caller's behalf.
//!
//! The Metal backend does the same with Mach memory entries
//! (`rlx_metal::shared_weights`); this is the POSIX version. Elsewhere (wasm,
//! Windows) [`SUPPORTED`] is false and the pool is compiled but inert.

#![cfg_attr(not(unix), allow(dead_code))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

thread_local! {
    /// The scope the current thread's compiles run under — see the module doc for why this is
    /// per thread while the pool it names is per process.
    static SCOPE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Set the sharing scope for executables this thread compiles from now on (`None`, and the
/// empty label, turn sharing off). Every executable compiled under one label must load the
/// same model's weights: a weight is matched by name and storage length, and a name inside
/// one model file has one dtype.
pub fn set_scope(scope: Option<String>) {
    // An empty label is no label: it comes from a build that declares no scope, and treating
    // it as one would put every unscoped graph in a single pool keyed by name alone.
    let scope = scope.filter(|label| !label.is_empty());
    SCOPE.with(|current| *current.borrow_mut() = scope);
}

/// The scope this thread's compiles run under.
pub fn scope() -> Option<String> {
    SCOPE.with(|current| current.borrow().clone())
}

/// Whether this platform can share weight pages.
pub const SUPPORTED: bool = cfg!(unix);

/// The VM page size; slots shared through [`MappedArena::map_shared`] must
/// start on one.
pub fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            return p as usize;
        }
    }
    16 * 1024
}

fn page_round(n: usize) -> usize {
    let p = page_size();
    n.div_ceil(p) * p
}

/// What one shared weight is identified by: the scope it belongs to, the tensor's name, and
/// its storage length. The dtype is implied by the name inside a model file, and the length
/// is what a caller that holds only a copy can state — which is what makes the question
/// answerable without the tensor itself.
type Key = (String, String, usize);

/// One file holding one weight; closed when the pool releases it, and the mappings that are
/// still live keep their pages until they drop.
struct SharedWeight {
    #[cfg(unix)]
    fd: libc::c_int,
    /// Set when an executable has written the weight's bytes into the file. Until it is, the
    /// pages are zero-filled and a reader that skips its write would run the model on zeros.
    written: std::sync::atomic::AtomicBool,
}

impl Drop for SharedWeight {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: `fd` is owned by this value and closed exactly once.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// What the pool holds, and how many bytes of weight that is.
#[derive(Default)]
struct Pool {
    objects: HashMap<Key, Arc<SharedWeight>>,
    bytes: usize,
}

/// The pool's ceiling: one model's weights and no more. The recognition pair is 1.35 GB of
/// packed weights, so a second model's arrival is what empties it.
///
/// A process that holds more than one family — the daemon loads ASR, TTS and CED in one
/// process, about 3.3 GB of packed weights between them — therefore thrashes: each family's
/// first compile clears the pool and the next family's clears it again, so every compile after
/// the first of a family re-reads that family's weights. That is a cost the stand cannot show,
/// because the stand gives every row its own process and never holds two families at once. It
/// is the correct trade for the measured product and a known one for the daemon; raising the
/// ceiling is what would fix it, at a resident cost the stand's rows would pay.
const POOL_BYTES: usize = 2560 << 20;

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// Give the pool's pages back: every object it holds is closed, and the mappings that are
/// still live keep their own pages until they drop. What a caller between two models asks
/// for, and what [`crate::shared_params`]'s users do when they release the process's memory.
pub fn clear() {
    let _ = POOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

/// A page-aligned, zero-filled arena whose ranges can be backed by a shared weight file.
pub struct MappedArena {
    ptr: *mut u8,
    len: usize,
    mapped_len: usize,
    /// The shared objects this arena mapped, at the offsets they were mapped at, so residency
    /// is recorded against the object that was mapped and never re-derived by key.
    held: Vec<(usize, Arc<SharedWeight>)>,
}

// SAFETY: the mapping is owned by this value; access is through `&`/`&mut`.
unsafe impl Send for MappedArena {}
unsafe impl Sync for MappedArena {}

impl MappedArena {
    /// A zeroed anonymous mapping of at least `len` bytes. `None` when the platform has no mmap
    /// or the call fails.
    ///
    /// Anonymous — the weights are the parts moved into files, which the module doc explains —
    /// so the memory a row allocates is the memory the platform's footprint reports.
    pub fn new(len: usize) -> Option<Self> {
        #[cfg(unix)]
        {
            let mapped_len = page_round(len.max(1));
            // SAFETY: an anonymous private mapping with no address hint.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    mapped_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return None;
            }
            Some(Self {
                ptr: ptr.cast(),
                len,
                mapped_len,
                held: Vec::new(),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = len;
            None
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `mapped_len >= len` readable bytes for our lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`; `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Back `[offset, offset + page_round(nbytes))` with the shared copy of
    /// weight `name`. `Some(true)` when the copy's values are already there,
    /// `Some(false)` when the caller must write them — this call made the object, or another
    /// executable made it and has not written it yet — and `None` when the range stays private
    /// (a failed system call). `offset` must be page-aligned and the rounded range must lie
    /// inside the arena.
    pub fn map_shared(
        &mut self,
        scope: &str,
        name: &str,
        nbytes: usize,
        offset: usize,
    ) -> Option<bool> {
        #[cfg(unix)]
        {
            let size = page_round(nbytes);
            // No label, no sharing: two graphs that declare no scope are not known to be the same
            // model, and a weight matched by name and size alone would hand one model's bytes to
            // another. A private slot is uploaded, which is always correct.
            if scope.is_empty() {
                return None;
            }
            if !offset.is_multiple_of(page_size()) || offset + size > self.mapped_len {
                return None;
            }
            let key: Key = (scope.to_string(), name.to_string(), nbytes);
            let mut pool = POOL
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let pool = pool.get_or_insert_with(Pool::default);
            let (weight, created) = match pool.objects.get(&key) {
                Some(w) => (Arc::clone(w), false),
                None => (Arc::new(create_object(size)?), true),
            };
            // SAFETY: the target range lies inside our own mapping (checked
            // above), so MAP_FIXED only replaces pages this arena owns.
            let p = unsafe {
                libc::mmap(
                    self.ptr.add(offset).cast(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    weight.fd,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return None;
            }
            // The object joins the pool only once a mapping of it has succeeded, and a weight
            // answers "already there" only once its bytes have been written: an entry made
            // before either would let a later compile skip the write and run on zeros. The
            // ceiling is applied here, after the mapping, so a refused one leaves the pool as
            // it found it rather than emptied for nothing.
            if created {
                // One model's weights and no more: a second model starts the pool over rather
                // than growing it. Nothing live is disturbed — an arena that already mapped an
                // object keeps its pages until it drops, and its accounting entry is only what
                // a later arrival may reuse.
                if pool.bytes + size > POOL_BYTES {
                    pool.objects.clear();
                    pool.bytes = 0;
                }
                pool.objects.insert(key, Arc::clone(&weight));
                pool.bytes += size;
            }
            let written = weight.written.load(std::sync::atomic::Ordering::Acquire);
            self.held.push((offset, weight));
            Some(!created && written)
        }
        #[cfg(not(unix))]
        {
            let _ = (scope, name, nbytes, offset);
            None
        }
    }

    /// Record that the weight mapped at `offset` holds its bytes now. Called by the executable
    /// that wrote them, after they are in the file. Answers `false` when no shared object backs
    /// that offset, so a private slot cannot be marked.
    pub fn mark_written(&mut self, offset: usize) -> bool {
        let Some((_, weight)) = self.held.iter().find(|(at, _)| *at == offset) else {
            return false;
        };
        weight
            .written
            .store(true, std::sync::atomic::Ordering::Release);
        true
    }
}

impl Drop for MappedArena {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: unmaps exactly the region `new` mapped (shared sub-ranges
        // included); nothing borrows it past `self`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.mapped_len);
        }
    }
}

/// An unnamed file of `size` bytes holding one weight: a file of the temporary directory
/// rather than a shared-memory object, for the reason the module doc gives. The name is unique
/// per process and unlinked straight away: the object lives on through the descriptor and the
/// mappings, and nothing else can open it.
#[cfg(unix)]
fn create_object(size: usize) -> Option<SharedWeight> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "rlx-w-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .ok()?;
    let _ = std::fs::remove_file(&path);
    // SAFETY: `file` is open read-write and owned by this function until the descriptor is
    // handed over; the size is the mapping's own extent.
    unsafe {
        let fd = std::os::fd::IntoRawFd::into_raw_fd(file);
        if libc::ftruncate(fd, size as libc::off_t) != 0 {
            libc::close(fd);
            return None;
        }
        Some(SharedWeight {
            fd,
            written: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn a_weight_mapped_twice_is_one_copy() {
        let page = page_size();
        let mut a = MappedArena::new(page * 4).unwrap();
        let mut b = MappedArena::new(page * 3).unwrap();
        assert_eq!(a.map_shared("t", "w", page + 8, page), Some(false));
        // Placed but not yet written: a second executable must be told to write its own
        // bytes, or it would skip the write and run on the zeros.
        assert_eq!(b.map_shared("t", "w", page + 8, 0), Some(false));
        a.as_mut_slice()[page..page + 4].copy_from_slice(&[1, 2, 3, 4]);
        assert!(a.mark_written(page));
        // An offset no shared object backs cannot be marked: a private slot has nothing to
        // record, and a later executable must not be told to skip its write on its word.
        assert!(!a.mark_written(0));
        let mut d = MappedArena::new(page * 2).unwrap();
        assert_eq!(d.map_shared("t", "w", page + 8, 0), Some(true));
        assert_eq!(&d.as_slice()[..4], &[1, 2, 3, 4]);
        // A write through either is seen by both; private pages stay private.
        b.as_mut_slice()[1] = 9;
        assert_eq!(a.as_slice()[page + 1], 9);
        a.as_mut_slice()[0] = 7;
        assert_eq!(b.as_slice()[page * 2 + 1], 0);
        // Another scope or size is another weight.
        let mut c = MappedArena::new(page * 2).unwrap();
        assert_eq!(c.map_shared("u", "w", page + 8, 0), Some(false));
        assert_eq!(c.as_slice()[0], 0);
        // Misaligned or out-of-range requests stay private.
        assert_eq!(c.map_shared("t", "w", 8, 1), None);
        assert_eq!(c.map_shared("t", "w", page * 4, 0), None);
    }
}
