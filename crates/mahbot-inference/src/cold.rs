// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The page cache, established rather than assumed.
//!
//! The stand's cold column is the short-lived command-line path on a machine that has not
//! read the models. That is a property of the machine, not of the process, so it has to be
//! made — and asserted — immediately before the timed call; otherwise the figure is a warm
//! one wearing a cold label.
//!
//! The method is the one `tests/cold_read.rs` proved here and the only no-root one: an
//! invalidation over a read-only mapping of the whole file, with residency read back
//! through a second mapping that is never faulted, so a zero is the page cache's answer
//! rather than the invalidated mapping's own page-table entries.
//!
//! The warm column is that reading the other way round: the cold arm of a row has just read
//! the files the warm arm needs, so [`Cache::reading`] records what the page cache holds of
//! them and [`Cache::grown_during_arm`] says afterwards what the arm had to fetch from disk.
//! That reading is taken once the cache has stopped changing ([`Cache::settled`]), because the
//! upload's request is asynchronous: a page it lands after the reading would be read as the
//! warm arm faulting it in.
//!
//! The reading is an observation, not a repair: whatever the cold arm left out of the cache is
//! exactly what the warm arm will have to fetch, and completing those pages first would leave
//! a check that can only catch eviction during the arm — the instrument's whole meaning is that
//! cold is not in the cache and warm is. What a count of fetched pages means for a warm figure,
//! and the ratio that refuses a take over it, is stated where the count is read: the stand's
//! warm arm check.
//!
//! A [`Cache`] is the models root and a page size, and nothing live: [`Cache::evict`] walks
//! the root, and every operation maps one file at a time for its own duration, so nothing the
//! instrument allocates outlives the operation — a row's resting figure is taken with it
//! holding one path, not a list of every file it emptied.
//!
//! What is emptied is every file under the models root rather than a list of the files a row
//! is expected to read: every input lives under it, so there is no per-row list to fall out
//! of date as the product gains one. What the warm arm is checked against is the files the
//! cold arm opened, which the loader records — so a model the row does not read cannot
//! refuse a take, and an input that is new to a row is covered without anybody listing it.
//! The row's recording is emptied too, so a cold machine does not serve it from the cache
//! either, but the warm check is not gated on it: the product reads its input on every call,
//! and the OS may reclaim a ten-megabyte recording under the working set a long row
//! allocates.
//!
//! The invalidation is this platform's — `msync(MS_INVALIDATE)` drops a mapped file's
//! clean pages where `madvise(MADV_DONTNEED)` and `MADV_FREE` leave them (measured on
//! macOS 26.5/arm64), and `posix_fadvise(POSIX_FADV_DONTNEED)` skips pages that are
//! mapped — so a platform without the method is refused before a run starts
//! ([`supported`]) rather than given a column that means nothing.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

/// Why a platform without the method is refused: [`supported`] and the arms it makes
/// unreachable all state it.
#[cfg(not(target_os = "macos"))]
const UNSUPPORTED: &str = "the stand's cold column means a page cache that does not hold the \
     models, and this platform has no proven no-root way to empty it (`tests/cold_read.rs` is \
     macOS-only)";

/// Whether this platform has the method [`Cache`] is built on. Asked once, before a run
/// starts: the cold column is on every row's path, so a per-row refusal would abort a run
/// that had already measured part of itself.
#[cfg(target_os = "macos")]
pub(crate) fn supported() -> Result<()> {
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn supported() -> Result<()> {
    bail!("{UNSUPPORTED}")
}

/// One file under the models root, opened read-only and mapped for the duration of one
/// operation: the mapping an invalidation walks, or the second one residency is read through.
struct Mapped {
    file: std::fs::File,
    map: memmap2::Mmap,
}

/// The page cache over the models root: what to establish it over, and the page size its
/// readings are indexed by. Nothing is held between operations, so a row's resting figure
/// carries one path rather than every file that was emptied.
pub(crate) struct Cache {
    root: PathBuf,
    page: usize,
}

/// What the page cache holds of one file: one byte per page from `mincore`, bit 0 set
/// where the page is resident.
struct Share {
    resident: Vec<u8>,
}

impl Share {
    /// The file's pages.
    fn pages(&self) -> usize {
        self.resident.len()
    }

    /// How many of them are resident.
    fn count(&self) -> usize {
        self.resident.iter().filter(|byte| **byte & 1 != 0).count()
    }

    /// What the page cache holds of the file `map` maps: one byte per page, bit 0 set where
    /// the page is resident. `map` is never faulted, so a zero is the page cache's answer.
    #[cfg(target_os = "macos")]
    fn read(map: &memmap2::Mmap, path: &Path, page: usize) -> Result<Self> {
        let mut scratch = vec![0u8; map.len().div_ceil(page)];
        // SAFETY: the mapping is read-only, page-aligned and live; `scratch` holds one byte
        // per page of it, which `mincore` writes.
        let rc =
            unsafe { libc::mincore(map.as_ptr().cast(), map.len(), scratch.as_mut_ptr().cast()) };
        if rc != 0 {
            bail!(
                "`{}`: mincore: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        Ok(Self { resident: scratch })
    }

    #[cfg(not(target_os = "macos"))]
    fn read(_: &memmap2::Mmap, _: &Path, _: usize) -> Result<Self> {
        bail!("{UNSUPPORTED}")
    }
}

/// What the page cache held of the files the cold arm read, in the order it read them: the
/// state the warm arm's assertion is made against.
pub(crate) struct Held {
    shares: Vec<(PathBuf, Share)>,
}

impl Held {
    /// The pages of every file the cold arm read: what the warm arm's re-read is a fraction of.
    pub(crate) fn pages(&self) -> usize {
        self.shares.iter().map(|(_, share)| share.pages()).sum()
    }
}

impl Cache {
    /// The models root, whose files the cold state is established over.
    pub(crate) fn over_models_root() -> Result<Self> {
        Ok(Self {
            root: crate::loader::models_dir()?,
            page: page_size()?,
        })
    }

    /// Drop every file's clean pages and assert that not one of them is resident: the state
    /// the cold column is taken in. An entry this process cannot open and map is counted and
    /// left out rather than named — it cannot make a column warm, since the arms read with the
    /// same permissions — so a permissions quirk neither buries the run in one long line nor
    /// kills it before its first row. Everything after the open is the instrument's own work on
    /// a file it holds, so it refuses the run; and a file still resident after the invalidation
    /// is a warm column, which refuses it too.
    pub(crate) fn evict(&self) -> Result<()> {
        let mut files = Vec::new();
        walk(&self.root, &mut files)?;
        let mut left_out = 0;
        for path in &files {
            let mapped = match Mapped::open(path) {
                Ok(mapped) => mapped,
                Err(_) => {
                    left_out += 1;
                    continue;
                }
            };
            invalidate(&mapped.map).with_context(|| format!("`{}`", path.display()))?;
            let share = Share::read(&mapped.remap()?, path, self.page)
                .with_context(|| format!("`{}`", path.display()))?;
            let resident = share.count();
            if resident != 0 {
                bail!(
                    "`{}`: {resident} of {} pages are still resident after the invalidation, so \
                     the cold column would be a warm one",
                    path.display(),
                    share.pages()
                );
            }
        }
        if left_out > 0 {
            eprintln!("stand: cold: {left_out} file(s) left out of the cold state");
        }
        Ok(())
    }

    /// What the page cache holds of `opened` — the files the cold arm read, taken rather than
    /// borrowed — in the order they were given: the state the warm arm's reading is made
    /// against. A file outside the root the cold state was established over is refused, not
    /// silently left out, and so is an empty set: a row that opened no model has no warm state
    /// to be read.
    pub(crate) fn reading(&self, opened: Vec<PathBuf>) -> Result<Held> {
        if opened.is_empty() {
            bail!("the cold arm opened no model file, so there is no warm state to check");
        }
        let mut shares = Vec::with_capacity(opened.len());
        for path in opened {
            if !path.starts_with(&self.root) {
                bail!(
                    "`{}` was read but is not under the models root, so the cold column was \
                     never established over it",
                    path.display()
                );
            }
            shares.push((path.clone(), self.settled(&path)?));
        }
        Ok(Held { shares })
    }

    /// How many pages of the files the cold arm read became resident during the arm: the
    /// pages it faulted in from disk, which a warm column claims it did not have to. Per
    /// page, not net — a page dropped and another faulted in would balance a count — and
    /// taken after the arm, where its page-table walk cannot reach the figure (the same walk
    /// before the arm costs it about eight milliseconds). A reading taken before and after
    /// cannot see a page that leaves the cache and comes back inside the arm; nothing that
    /// reads the cache afterwards can.
    ///
    /// One entry per file that was short, as `(path, pages the arm fetched, pages the file
    /// has)`, in the order the cold arm opened them: the row states each, because a total
    /// that named only one of them would hide the others. What the stand refuses over this
    /// count is stated where it reads it.
    pub(crate) fn grown_during_arm(&self, held: &Held) -> Result<Vec<(PathBuf, usize, usize)>> {
        let mut short = Vec::new();
        for (path, before) in &held.shares {
            let now = self.residency(path)?;
            let fresh = grown(before, &now);
            if fresh != 0 {
                short.push((path.clone(), fresh, now.pages()));
            }
        }
        Ok(short)
    }

    /// What the page cache holds of one file, once it has stopped changing. The upload's request
    /// is asynchronous, so a page it lands after this reading is growth the warm arm is blamed
    /// for; two identical readings in a row mean nothing is still arriving. Bounded, so a cache
    /// that never settles is read as it is rather than refusing the run.
    ///
    /// Each poll is a whole-file `mincore` — about eight milliseconds for a body, the walk
    /// [`Self::grown_during_arm`] measures — so a settled cache costs two of them and the
    /// hundred and twenty readings the bound allows cost about two seconds per file
    /// (a hundred and nineteen 10 ms sleeps and a hundred and twenty walks). The wait is here
    /// rather than in the product because the doubt is the instrument's: the request may
    /// legitimately still be in flight when the cold arm returns, and only a reader of the
    /// page cache can tell. A reader outside this process is the one thing it cannot wait out.
    /// The bound is generous because a processor arm asks for far more bytes than a graphics
    /// one and its request can still be arriving a second later.
    fn settled(&self, path: &Path) -> Result<Share> {
        // Whole-file readings taken before the cache is called settled, the first included.
        const POLLS: usize = 120;
        let mut last = self.residency(path)?;
        for _ in 1..POLLS {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let now = self.residency(path)?;
            if now.resident == last.resident {
                return Ok(now);
            }
            last = now;
        }
        Ok(last)
    }

    /// What the page cache holds of one file, through a mapping taken for the reading and
    /// never faulted.
    fn residency(&self, path: &Path) -> Result<Share> {
        Share::read(&Mapped::open(path)?.map, path, self.page)
    }
}

/// Collect every file below `dir`, the models root. An empty file holds no page and is
/// skipped; a symlinked file is collected, because the product maps through one. A symlinked
/// directory is followed, so a model installed behind one is still covered — a link that
/// points outside the root is followed too, which is the trade: emptying the models empties
/// whatever a link under it names. Each real directory is walked once however many links name
/// it, which is what keeps a link from looping.
///
/// What the walk cannot read refuses the run: the cold state is claimed over everything an
/// arm could open, and a directory this process cannot list may still be opened by exact path
/// through its search bit, so its contents are unknown rather than unreachable. A *file* that
/// cannot be opened is a leaf with no such doubt, and `evict` counts and leaves that one out.
fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    walk_into(dir, files, &mut Vec::new())
}

fn walk_into(dir: &Path, files: &mut Vec<PathBuf>, seen: &mut Vec<PathBuf>) -> Result<()> {
    let real = dir
        .canonicalize()
        .with_context(|| format!("resolving `{}`", dir.display()))?;
    if seen.contains(&real) {
        return Ok(());
    }
    seen.push(real);
    let entries = std::fs::read_dir(dir).with_context(|| format!("reading `{}`", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("reading `{}`", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            walk_into(&path, files, seen)?;
        } else if path.is_file() {
            let empty = entry
                .metadata()
                .with_context(|| format!("`{}`", path.display()))?
                .len()
                == 0;
            if !empty {
                files.push(path);
            }
        }
    }
    Ok(())
}

/// The pages resident in `now` that were not resident in `before`.
fn grown(before: &Share, now: &Share) -> usize {
    now.resident
        .iter()
        .zip(&before.resident)
        .filter(|(now, before)| **now & 1 != 0 && **before & 1 == 0)
        .count()
}

impl Mapped {
    fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening `{}`", path.display()))?;
        let map = map_of(&file).with_context(|| format!("mapping `{}`", path.display()))?;
        Ok(Self { file, map })
    }

    /// A second mapping of the same open file, taken after the invalidation and never
    /// faulted, so a zero read through it is the page cache's answer rather than the
    /// invalidated mapping's own page-table entries.
    fn remap(&self) -> Result<memmap2::Mmap> {
        Ok(map_of(&self.file)?)
    }
}

fn map_of(file: &std::fs::File) -> std::io::Result<memmap2::Mmap> {
    // SAFETY: a read-only mapping of an input file, which nothing in this tree writes.
    unsafe { memmap2::MmapOptions::new().map(file) }
}

#[cfg(target_os = "macos")]
fn page_size() -> Result<usize> {
    // SAFETY: `sysconf` with a valid name reads a process constant.
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        size if size > 0 => Ok(size as usize),
        _ => bail!("sysconf(_SC_PAGESIZE): {}", std::io::Error::last_os_error()),
    }
}

#[cfg(not(target_os = "macos"))]
fn page_size() -> Result<usize> {
    bail!("{UNSUPPORTED}")
}

/// Drop the file's clean pages from the page cache.
#[cfg(target_os = "macos")]
fn invalidate(map: &memmap2::Mmap) -> Result<()> {
    // SAFETY: the mapping is read-only and live; the kernel only walks it.
    let rc = unsafe {
        libc::msync(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MS_INVALIDATE,
        )
    };
    if rc != 0 {
        bail!("msync(MS_INVALIDATE): {}", std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn invalidate(_: &memmap2::Mmap) -> Result<()> {
    bail!("{UNSUPPORTED}")
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// A unique path in the temp directory: a file in one test, a directory in another.
    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mahbot-cold-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ))
    }

    /// The state a row is measured in is a reading, not a claim: a file this test has
    /// just written is resident, after the invalidation it holds nothing, and a read of it
    /// puts every page back — which the warm arm's check sees page by page.
    #[test]
    fn a_file_is_evicted_and_read_back() {
        let path = scratch("readback");
        std::fs::write(&path, vec![0x5au8; 8 << 20]).expect("a scratch file");
        let page = page_size().expect("a page size");
        let mapped = Mapped::open(&path).expect("a mapping of it");
        let written = Share::read(&mapped.map, &path, page).expect("the file's pages");
        assert_eq!(
            written.count(),
            written.pages(),
            "a file this test just wrote is resident"
        );
        invalidate(&mapped.map).expect("the invalidation empties it");
        let empty = Share::read(&mapped.map, &path, page).expect("the file's pages");
        assert_eq!(empty.count(), 0, "an invalidated file holds no page");

        // Any read is growth against the empty reading, and none against the file as it
        // was written: that is the check the warm column rests on.
        let mut half = vec![0u8; 4 << 20];
        std::fs::File::open(&path)
            .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut half))
            .expect("a read of half the file");
        let after_half = Share::read(&mapped.map, &path, page).expect("the file's pages");
        assert!(
            grown(&empty, &after_half) > 0,
            "half the file read back is growth"
        );
        std::fs::read(&path).expect("a read of the whole file");
        let after_all = Share::read(&mapped.map, &path, page).expect("the file's pages");
        assert!(
            grown(&empty, &after_all) > 0,
            "the whole file read back is growth"
        );
        assert_eq!(
            grown(&written, &after_all),
            0,
            "nothing is resident that was not when it was written"
        );
        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    /// The cold state is established over the root itself, so what a row reads is not a list
    /// anybody maintains: every file below it is collected — through a symlink too, because
    /// the product maps through one — and an empty file holds no page to drop. A symlinked
    /// directory is followed, so a model installed behind one is emptied; a directory two
    /// links name is walked once.
    #[test]
    fn the_root_walk_collects_every_file() {
        let dir = scratch("tree");
        let outside = scratch("outside");
        std::fs::create_dir_all(dir.join("corpus/speech")).expect("a scratch tree");
        std::fs::create_dir_all(&outside).expect("a tree beside it, named by a link");
        std::fs::write(dir.join("body.gguf"), b"x").expect("a model");
        std::fs::write(dir.join("corpus/mmproj.gguf"), b"x").expect("a model below the root");
        std::fs::write(dir.join("corpus/speech/recording.wav"), b"x").expect("a recording");
        std::fs::write(dir.join("notes.txt"), b"").expect("an empty file");
        std::fs::write(outside.join("installed.gguf"), b"x").expect("a model behind a link");
        std::os::unix::fs::symlink(dir.join("body.gguf"), dir.join("alias.gguf"))
            .expect("a symlinked model");
        std::os::unix::fs::symlink(&outside, dir.join("installed")).expect("a symlinked directory");
        std::os::unix::fs::symlink(dir.join("corpus"), dir.join("linked"))
            .expect("a second link to a directory already below the root");
        let mut files = Vec::new();
        walk(&dir, &mut files).expect("the walk");
        assert_eq!(
            files.len(),
            5,
            "four files below the root plus the one only the link outside it names, with the \
             empty file and the second copy of `corpus` left out"
        );
        assert!(
            files
                .iter()
                .any(|path| path.ends_with("installed/installed.gguf")),
            "a directory a link outside the root names is followed"
        );
        assert_eq!(
            files
                .iter()
                .filter(|path| path.ends_with("mmproj.gguf"))
                .count(),
            1,
            "a directory two links name is walked once"
        );
        std::fs::remove_dir_all(&dir).expect("the tree goes");
        std::fs::remove_dir_all(&outside).expect("the linked tree goes");
    }

    /// A row that opened no model has no warm state to be checked for, so the reading
    /// refuses it: an assertion that passes on an empty set is no assertion at all.
    #[test]
    fn an_empty_opened_set_is_refused() {
        let dir = scratch("empty");
        std::fs::create_dir_all(&dir).expect("a scratch tree");
        std::fs::write(dir.join("body.gguf"), b"x").expect("a model");
        let cache = Cache {
            root: dir.clone(),
            page: page_size().expect("a page size"),
        };
        cache.evict().expect("the eviction");
        assert!(
            cache.reading(Vec::new()).is_err(),
            "an empty set has no warm state to check"
        );
        std::fs::remove_dir_all(&dir).expect("the tree goes");
    }
}
