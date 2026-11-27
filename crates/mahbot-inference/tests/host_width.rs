// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recognition encoder's two graphs built under both host-width arms, in one session:
//! the file's width on the host (`MAHBOT_HOST_F16` unset) against the f32 image it replaced
//! (`MAHBOT_HOST_F16=0`).
//!
//! The two arms are one process each — the parent re-runs this binary with the switch in the
//! child's environment, because the f32 image the old arm keeps is an anonymous allocation
//! this platform never gives back, so the two cannot share a footprint. What the parent
//! compares is what the change is about: the answers must be **identical** (the widening is
//! the loaders' own, so any difference is a defect, not a drift), the bytes handed to the
//! device must be no more than the f32 image's, and the resting system memory must be lower
//! on the new arm.
//!
//! Needs the installed model directory; a machine without it skips. Run with
//! `cargo test -p mahbot-inference --release --features hand-run --test host_width --
//! --ignored --nocapture` (add `MAHBOT_DEVICE=cpu` for the processor's own record).

use std::process::Stdio;

use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::report;

/// The child's marker: without it the parent's own test body runs the comparison.
const CHILD: &str = "MAHBOT_HOST_WIDTH_CHILD";
/// How long one child may run: two graphs of the encoder's front and tail, at three rungs.
const CHILD_BUDGET: std::time::Duration = std::time::Duration::from_secs(900);

/// The rungs the tail is built at — the ladder's own arrangement, where a tensor the tail binds
/// is asked for once per rung. Three rungs make the two arms' widening counts differ: the f32
/// image is made on the first ask and every later rung uploads the copy, while the file's width
/// is widened on every compile — one materialisation against three widens. On the graphics
/// device the upload figure is then equal on both arms and only the memory column moves; on the
/// processor the new arm's upload also falls, because a later rung finds the parameter resident.
const RUNGS: [usize; 3] = [128, 256, 384];

#[test]
#[ignore = "measurement: the host's width against the f32 image, two arms in one session"]
fn the_host_keeps_the_files_width() {
    if loader::models_dir().is_err() {
        eprintln!("skipped: the model directory is not installed — the arms were not compared");
        return;
    }
    let (off, on) = (child("0"), child("1"));
    for arm in [&off, &on] {
        eprintln!(
            "MAHBOT_HOST_F16={}: widened {:?}, upload {} bytes, digest {:#018x}, resting system \
             {} MiB",
            arm.switch,
            arm.widened,
            arm.upload,
            arm.digest,
            mib(arm.resting),
        );
    }
    assert!(
        off.widened.is_empty(),
        "the arm that materialises the f32 image keeps no file-width record: {:?}",
        off.widened
    );
    assert!(
        !on.widened.is_empty(),
        "the file's width is not what the new arm recorded"
    );
    assert_eq!(
        on.digest, off.digest,
        "the two arms must produce the same answer, bit for bit"
    );
    // The new route asks whether the device already holds a parameter before reading it at all,
    // which the f32 image never did: the bytes it hands over are never more than that arm's.
    assert!(
        on.upload <= off.upload,
        "the new arm uploaded {} bytes against the f32 image's {}",
        on.upload,
        off.upload,
    );
    // The columns the change exists to move. A platform without the footprint basis reports
    // none, and then there is nothing here to compare.
    if off.resting == 0 {
        eprintln!("this platform reports no footprint — the memory columns were not compared");
        return;
    }
    assert!(
        on.resting < off.resting,
        "the new arm's resting system memory ({} MiB) must be below the f32 image's ({} MiB)",
        mib(on.resting),
        mib(off.resting),
    );
}

/// One arm: what the child printed, parsed.
struct Arm {
    switch: String,
    widened: Vec<String>,
    upload: u64,
    digest: u64,
    /// The footprint at rest less the graphics-tagged pages, as `STAND.md`'s system column is.
    resting: u64,
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

/// The names the build being made handed over at the file's width.
fn file_width_names() -> Vec<String> {
    mahbot_inference::emit::host_width_reads()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect()
}

/// Run one arm in a child process and read back what it printed.
fn child(arm: &str) -> Arm {
    let path = std::env::temp_dir().join(format!(
        "mahbot-host-width-{}-{arm}.stdout",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("child stdout file");
    let mut child = std::process::Command::new(std::env::current_exe().expect("current exe"))
        .args([
            "--exact",
            "the_host_keeps_the_files_width_child",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("MAHBOT_HOST_F16", arm)
        .stdout(file)
        .stderr(Stdio::inherit())
        .spawn()
        .expect("re-run this test in a child process");
    let deadline = std::time::Instant::now() + CHILD_BUDGET;
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait on the child") {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the MAHBOT_HOST_F16={arm} child did not finish in time");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let text = std::fs::read_to_string(&path).expect("the child's stdout");
    let _ = std::fs::remove_file(&path);
    assert!(
        status.success(),
        "the MAHBOT_HOST_F16={arm} child failed:\n{text}"
    );
    // The child's output shares its stream with the harness's own progress line, so a field is
    // found wherever its `key=` appears rather than at the start of a line.
    let field = |key: &str| -> String {
        let needle = format!("{key}=");
        text.lines()
            .find_map(|line| {
                line.split_once(&needle)
                    .map(|(_, rest)| rest.trim().to_string())
            })
            .unwrap_or_else(|| panic!("the child printed no `{key}` line:\n{text}"))
    };
    Arm {
        switch: arm.to_string(),
        widened: field("widened")
            .split_whitespace()
            .map(str::to_string)
            .collect(),
        upload: field("upload").parse().expect("an upload figure"),
        digest: u64::from_str_radix(field("digest").trim_start_matches("0x"), 16)
            .expect("a digest"),
        resting: field("resting").parse().expect("a footprint"),
    }
}

/// The child's body: build, compile and run the encoder's front and tail under this arm, and
/// print what the parent compares — one `key=value` line each.
#[test]
#[ignore = "the child half of the_host_keeps_the_files_width"]
fn the_host_keeps_the_files_width_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("skipped: {error:#}");
            return;
        }
    };
    let model = pair.sound.clone();
    let mut widened: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut upload = 0u64;

    // The front's three convolution kernels, then the tail's `conv_out` — the four tensors
    // the file stores at half width — at each rung the ladder builds.
    let mel_bins = model.clip().mel_bins;
    let built = graphs::asr_encoder_front(model.clone(), RUNGS[0]).expect("the front builds");
    widened.extend(file_width_names());
    let mut front = graphs::Compiled::new(built).expect("the front compiles");
    upload += front.upload().bytes;
    let mel = report::pseudo_rand(mel_bins * RUNGS[0], 7);
    let flat = front.run(&[("mel".to_string(), mel)]);
    let mut digest = report::digest(&flat[0]);
    for rung in RUNGS {
        let built = graphs::asr_encoder_tail(model.clone(), rung, 1).expect("the tail builds");
        widened.extend(file_width_names());
        let mut tail = graphs::Compiled::new(built).expect("the tail compiles");
        upload += tail.upload().bytes;
        let rows = graphs::asr_tokens(rung);
        let input = report::pseudo_rand(rows * graphs::flat_width(mel_bins), 11);
        let tokens = tail.run(&[("flat".to_string(), input)]);
        digest = digest.wrapping_mul(31) ^ report::digest(&tokens[0]);
    }

    // The stand's resting-system column: the footprint at rest less the graphics-tagged
    // pages, so what is compared is the host image rather than the device's own allocations.
    let memory = report::Memory::take();
    let resting = memory
        .now
        .unwrap_or(0)
        .saturating_sub(memory.gpu_now.unwrap_or(0));
    let widened: Vec<String> = widened.into_iter().collect();
    println!("widened={}", widened.join(" "));
    println!("upload={upload}");
    println!("digest=0x{digest:016x}");
    println!("resting={resting}");
}
