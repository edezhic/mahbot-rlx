// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One open hidden under another, when only one of them compiles.
//!
//! A recognition call opens three things before it can do any work: the model pair, the
//! tokenizer's vocabulary and the sound tagger. Nothing requires their order, and only the tagger
//! touches the device. The pair and the vocabulary are a chain — the vocabulary is built from the
//! pair's body — and that chain hides under the tagger, which is the critical path and cannot
//! overlap itself.
//!
//! Serially a recognition open costs 63.9 ms on this machine (M3 Pro; pair 8.0, vocabulary 8.9,
//! tagger 47.0, the median of nine paired rounds `tests/open_overlap.rs` takes); overlapped, 48.3 ms
//! — a paired saving of 15.7 ms, the chain's own 16.9 ms and no more. That harness is the record and
//! re-derives both arms. On the stand's shipped take that is 0.8 % of the 10 s recognition cold row,
//! 0.5 % of the 30 s row and 0.06 % of the 318 s one. Synthesis overlaps the same way — its
//! vocabulary (15.6 ms) under the code predictor's frame compile (58.9 ms) — for a paired 14.7 ms,
//! 2.2 % of the 25-char row's cold-minus-warm delta and 2.9 % of the 129-char one. A wake row saves
//! nothing: its pair open is a hard predecessor of the listener's bring-up compile, which has nowhere
//! to hide.
//!
//! **Only one side of an overlap may compile, and that side runs whole on one thread.** A compile
//! in this tree installs a process-wide fusion override ([`crate::graphs::compile_encoder`]) and
//! carries thread-local build records ([`crate::emit`]), so a build and its compile may not be
//! split across threads and may not run beside another compile. That is a convention here, not an
//! enforced invariant — no lock would refuse it. So the compiling side is the worker for the
//! tagger's open and this thread for the predictor frame's compile, and the side beside it compiles
//! nothing and touches no process-global compile state either: `release_process_memory()` clears
//! the shared weight pool with no lock held across a compile, so releasing memory beside one would
//! strip its weights.
//!
//! The loader's record of opened files is the calling thread's ([`crate::loader`]), and the
//! cold/warm gate reads only that thread's list: the worker's record is folded back here, or a
//! file the worker read drops out of the gate silently and the gate narrows without failing. A
//! panic on the worker is this call's error, not a swallowed one — the worker is joined on
//! every path out, including the error path where this thread's own chain failed first.

use anyhow::{Result, anyhow};

use crate::loader;

/// The stack a worker's open is given: 8 MiB, four times a spawned thread's 2 MiB default, so a
/// compiling open is not given less stack than it had. A reservation, not a resident figure: the
/// frames it touches are the ones the same open touches on this thread.
const OPEN_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Run `worker` on a thread of its own while this one runs `alongside`, and hand both back
/// as `(worker, alongside)`.
///
/// `worker` is a `Fn` so the work can also run here when the thread cannot be spawned. Only one
/// side of the call may compile, and it runs whole on one thread: see the module doc.
pub(crate) fn opens<T, U>(
    name: &str,
    worker: impl Fn() -> Result<T> + Send + Sync,
    alongside: impl FnOnce() -> Result<U>,
) -> Result<(T, U)>
where
    T: Send,
{
    std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(OPEN_STACK_BYTES)
            .spawn_scoped(scope, || {
                let out = worker();
                // Taken on the thread that wrote it: the record is a thread-local.
                (out, loader::take_opened())
            });
        // This thread's own chain, beside the worker rather than after it.
        let alongside = alongside();
        let (out, opened) = match spawned {
            Ok(handle) => match handle.join() {
                Ok(out) => out,
                Err(payload) => return Err(panicked(name, payload)),
            },
            Err(error) => {
                eprintln!(
                    "overlap: cannot start the {name} thread ({error}) — running it here instead"
                );
                (worker(), Vec::new())
            }
        };
        // The record is the calling thread's, and the gate reads only that one.
        for path in opened {
            loader::opened(&path);
        }
        Ok((out?, alongside?))
    })
}

/// A worker's panic as this call's error: the payload the thread unwound with, or the fact
/// of it when the payload carries no message.
fn panicked(name: &str, payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    let what = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message");
    anyhow!("the {name} thread panicked: {what}")
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// The worker's record is this thread's once the call returns: the cold/warm gate reads
    /// only this thread's list, so a file the worker opened would otherwise drop out of it.
    #[test]
    fn a_file_the_worker_opened_is_this_threads_record() {
        opens::<(), ()>(
            "test-open",
            || {
                loader::opened(Path::new("/worker/model.gguf"));
                Ok(())
            },
            || Ok(()),
        )
        .expect("both arms");
        assert_eq!(
            loader::take_opened(),
            vec![std::path::PathBuf::from("/worker/model.gguf")]
        );
    }

    /// A worker's panic is an error the caller sees, not one that escapes the call.
    #[test]
    fn a_worker_panic_is_the_calls_error() {
        let error = opens::<(), ()>("test-open", || panic!("the model file is gone"), || Ok(()))
            .expect_err("a panicked worker");
        assert!(
            format!("{error:#}").contains("the model file is gone"),
            "{error:#}"
        );
    }

    /// The worker is joined even when this thread's own chain failed first: nothing it opened
    /// is still running when the error comes back.
    #[test]
    fn the_worker_is_joined_when_this_thread_fails_first() {
        let done = AtomicBool::new(false);
        let result = opens::<(), ()>(
            "test-open",
            || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                done.store(true, Ordering::Release);
                Ok(())
            },
            || anyhow::bail!("this thread's own chain failed"),
        );
        assert!(result.is_err());
        assert!(
            done.load(Ordering::Acquire),
            "the worker outlived the error"
        );
    }
}
