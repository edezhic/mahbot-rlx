// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and ignored: the time the live wake-word pair pass spends
//! inside quantized weight multiplies, for the two readers a 34-byte Q8_0 block
//! can be read with — nine whole words (shipped) against the per-word
//! reassembly it replaced (`RLX_WGPU_Q8_0_BLOCK_WORDS=0`).
//!
//! Needs the installed model directory and the pinned corpus recording; a machine
//! without them skips. Run with `RLX_ALLOW_THROTTLE=1 cargo test -p mahbot-inference
//! --release --features hand-run --test wake_multiply_share -- --ignored --nocapture`,
//! after `just throttle`.
//!
//! **The instrument is the backend's own device clock**, never a wall clock: the
//! child runs the product's paired pass ([`Listener::embed_pair`] — two scoring
//! windows through one graph) under `RLX_WGPU_DBG_COUNTERS=1` and
//! `RLX_WGPU_ONE_OP_PER_PASS=1`. The second switch makes a packed multiply open a
//! pass of its own, so the run report's `device_pass` records carry one timestamp
//! pair per multiply: the 96 a pass dispatches are the records with no schedule
//! census (`interval=host kinds=`), and their `ms` sum is the part of the pass
//! spent inside the packed multiplies. Both switches change how work is
//! submitted, not what it computes, and both arms run under them — which is what
//! makes that pair comparable, and what a device figure here is a claim about:
//! the multiplies' own time under the instrument's shape. The plain round set at
//! the end is what the pass costs as the product runs it.
//!
//! A round runs the two arms back to back and alternates which goes first; the
//! medians, the median of the per-round differences and the rounds the whole-word
//! arm won are the reading. The bar is the ratio of the two medians, and it is the
//! multiplies' own time that is compared — not their fraction of the pass, which
//! moves by less than the multiplies do because the rest of the pass does not.
//!
//! The last round set repeats the comparison **without the instrument**, where the
//! multiplies share the batch's passes with the rest of the schedule: that is the
//! product's own submission shape, and its wall clock is the other half of what the
//! reader is worth. The two readings answer different questions — the device one
//! prices the multiplies, the plain one prices the pass — and a run prints both.
//!
//! The arms are one kernel body over the same bytes, so the child also prints what
//! they produce against each other — the two slots' cosine, largest element
//! difference and norm — as a reading of the output, not as a gate on a signature.

use std::process::Stdio;

use mahbot_inference::host::wav_mono_16k;
use mahbot_inference::listen::{self, Listener};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::stand::WAKE_OFFSET;
use mahbot_inference::wake::cosine;

mod support;

use support::median;

/// The switch the two arms differ in, and the two values it takes.
const ARM_SWITCH: &str = "RLX_WGPU_Q8_0_BLOCK_WORDS";
/// Rounds the two arms alternate in. One run's own spread is a sizeable share of
/// the effect, so the paired difference is the reading and this many rounds are
/// what separate it.
const ROUNDS: usize = 9;
/// How long the child may run before it is killed and the case reported: model
/// loading, both arms' warm-up and every round, with room to spare.
const CHILD_BUDGET: std::time::Duration = std::time::Duration::from_secs(900);
/// The packed multiplies one pass of the encoder dispatches: 24 blocks of four
/// (the fused projection, `attn_out`, `ffn_up`, `ffn_down`).
const MULTIPLIES_PER_PASS: usize = 96;

#[test]
#[ignore = "measurement: prices the Q8_0 block reader inside the live pair pass"]
fn the_packed_multiply_share_of_the_pair_pass() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    if loader::models_dir().is_err() || mahbot_inference::corpus::dir().is_err() {
        eprintln!(
            "skipped: the model directory or the pinned corpus is not installed — the packed \
             multiply was not priced"
        );
        return;
    }
    let path =
        std::env::temp_dir().join(format!("mahbot-wake-share-{}.stderr", std::process::id()));
    let file = std::fs::File::create(&path).expect("child stderr file");
    let mut child = std::process::Command::new(std::env::current_exe().expect("current exe"))
        .args([
            "--exact",
            "the_packed_multiply_share_child",
            "--ignored",
            "--nocapture",
        ])
        .env("MAHBOT_WAKE_SHARE_CHILD", "1")
        .stdout(Stdio::null())
        .stderr(file)
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
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let _ = std::fs::remove_file(&path);
            panic!(
                "the child did not finish within {}s; its stderr so far:\n{text}",
                CHILD_BUDGET.as_secs()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let text = std::fs::read_to_string(&path).expect("the child's stderr");
    let _ = std::fs::remove_file(&path);
    assert!(
        status.success(),
        "the child exited with {status}; its stderr:\n{text}"
    );
    report(&text);
}

/// The two arms, as the switch's value and the name they are printed under.
const ARMS: [(&str, &str); 2] = [("1", "nine-words"), ("0", "reassembly")];

/// One arm installed: the switch plus the two instrument switches, restored on
/// drop so a round's marker cannot leak into the next arm's run.
fn arm(arm: &str) -> rlx_ir::env::RuntimeOverrides {
    rlx_ir::env::RuntimeOverrides::install([
        ("RLX_WGPU_DBG_COUNTERS", "1"),
        ("RLX_WGPU_ONE_OP_PER_PASS", "1"),
        (ARM_SWITCH, arm),
    ])
}

/// The child: the product's paired pass, both arms, alternating, each round's
/// report left on stderr for the parent to read.
#[test]
#[ignore = "child of the harness above: runs both arms and prints its markers"]
fn the_packed_multiply_share_child() {
    if std::env::var_os("MAHBOT_WAKE_SHARE_CHILD").is_none() {
        return;
    }
    let pair =
        std::sync::Arc::new(Pair::of_role(Role::Recognition).expect("the recognition pair opens"));
    let recording =
        mahbot_inference::corpus::pinned(mahbot_inference::stand::RECOGNITION_RECORDING)
            .expect("the pinned corpus recording");
    let read = |offset: usize| {
        wav_mono_16k(
            &recording.display().to_string(),
            offset,
            listen::WINDOW_SAMPLES,
        )
        .expect("the pinned corpus recording")
    };
    let first = read(WAKE_OFFSET);
    let second = read(WAKE_OFFSET + listen::SCORE_STRIDE_SAMPLES);
    let mut listener = Listener::new(pair);

    // Each arm's first run compiles its kernel and warms the device, and the
    // instrument's submission shape is not the plain one, so both are warmed
    // under the instrument they are measured with.
    for (value, _) in ARMS {
        let _warm = arm(value);
        let _ = listener.embed_pair(&first, &second).expect("the pair pass");
    }

    // What the arms produce: the same two embeddings, read against each other in
    // the quiet configuration, since a run under the instrument is only a
    // different submission shape.
    let mut produced = Vec::new();
    for (value, name) in ARMS {
        let _quiet = rlx_ir::env::RuntimeOverrides::install([(ARM_SWITCH, value)]);
        let pair = listener.embed_pair(&first, &second).expect("the pair pass");
        produced.push((name, pair));
    }
    for slot in 0..listen::PAIR {
        let (name_a, a) = &produced[0];
        let (name_b, b) = &produced[1];
        let (ea, eb) = (&a[slot], &b[slot]);
        let delta = ea
            .iter()
            .zip(eb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[wake-share] parity slot {slot}: {name_a} vs {name_b} cosine {:.9}, max|Δ| {delta:e}, \
             norms {:.6} / {:.6} over {} dimensions",
            cosine(ea, eb),
            norm(ea),
            norm(eb),
            ea.len()
        );
    }

    for round in 0..ROUNDS {
        let order = if round % 2 == 0 {
            ARMS
        } else {
            [ARMS[1], ARMS[0]]
        };
        for (value, name) in order {
            let _instrument = arm(value);
            eprintln!("[wake-share] round {round} arm {name}");
            let _ = listener.embed_pair(&first, &second).expect("the pair pass");
            eprintln!("[wake-share] end {round}");
        }
    }

    // The same pass as the product runs it: no instrument, so the multiplies share
    // the batch's passes with everything else. What the reader is worth there is a
    // reading of its own — the device figures above are the multiplies alone, and
    // the two need not move by the same amount.
    for round in 0..ROUNDS {
        let order = if round % 2 == 0 {
            ARMS
        } else {
            [ARMS[1], ARMS[0]]
        };
        for (value, name) in order {
            let _quiet = rlx_ir::env::RuntimeOverrides::install([(ARM_SWITCH, value)]);
            let t = rlx_ir::Tick::now();
            let _ = listener.embed_pair(&first, &second).expect("the pair pass");
            let ms = rlx_ir::Tick::now().elapsed_ns(t) as f64 / 1e6;
            eprintln!("[wake-share] quiet {round} arm {name} ms {ms:.4}");
        }
    }
}

/// An embedding's L2 norm, for the parity line.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// One `device_pass` record, as the report printed it.
struct Pass {
    interval: String,
    ms: String,
    kinds: String,
}

impl Pass {
    /// The instrument makes each packed multiply open a pass of its own, and that
    /// pass covers no schedule step, so its census is empty and it is not charged to
    /// the run's own intervals.
    fn is_multiply(&self) -> bool {
        self.interval == "host" && self.kinds.is_empty()
    }
}

/// One round's figures: the packed multiplies' device time, and the encoder pass's
/// own device time for the share they are a part of.
struct Figure {
    arm: String,
    multiply_ms: f64,
    pass_ms: f64,
}

/// Read the child's stderr: the marker lines say which arm and round each report
/// block belongs to, and the block with [`MULTIPLIES_PER_PASS`] census-free passes
/// is the encoder pass — that many packed multiplies, each on its own pass.
fn report(text: &str) {
    let mut figures: Vec<Figure> = Vec::new();
    let mut parity = Vec::new();
    let mut arm_now: Option<String> = None;
    let mut quiet: Vec<(String, f64)> = Vec::new();
    // One block is one arm's round, which runs more than one graph: the parent
    // keeps every run's passes apart, and the encoder's run is the one with a
    // packed multiply on each of its census-free passes.
    let mut runs: Vec<Vec<Pass>> = Vec::new();
    for line in text.lines() {
        if line.starts_with("[wake-share] parity") {
            parity.push(line.trim_start_matches("[wake-share] ").to_string());
            continue;
        }
        if line.starts_with("[wgpu-run-report] run total_ms=") {
            runs.push(Vec::new());
            continue;
        }
        if let Some(rest) = line.strip_prefix("[wake-share] quiet ") {
            // `quiet <round> arm <name> ms <figure>`
            let fields: Vec<&str> = rest.split(' ').collect();
            quiet.push((fields[2].to_string(), fields[4].parse().expect("a figure")));
            continue;
        }
        if let Some(rest) = line.strip_prefix("[wake-share] round ") {
            let mut parts = rest.splitn(3, ' ');
            let _round = parts.next();
            let _arm = parts.next();
            arm_now = parts.next().map(str::to_string);
            runs.clear();
            continue;
        }
        if line.starts_with("[wake-share] end ") {
            let Some(arm) = arm_now.take() else { continue };
            // Per run: how many census-free passes it held — the count names the
            // encoder's — and their summed device time, the figure.
            let multiplies: Vec<(usize, f64)> = runs
                .iter()
                .map(|passes| {
                    let mut count = 0;
                    let mut ms = 0.0;
                    for p in passes.iter().filter(|p| p.is_multiply()) {
                        count += 1;
                        ms += p.ms.parse::<f64>().expect("a device figure");
                    }
                    (count, ms)
                })
                .collect();
            let run = multiplies
                .iter()
                .position(|(count, _)| *count == MULTIPLIES_PER_PASS)
                .unwrap_or_else(|| {
                    panic!(
                        "one pass dispatches {MULTIPLIES_PER_PASS} packed multiplies, each on its \
                         own pass under the instrument; this block's runs held {:?} census-free \
                         passes — the encoder's was not among them",
                        multiplies
                            .iter()
                            .map(|(count, _)| *count)
                            .collect::<Vec<_>>()
                    )
                });
            let pass_ms = |passes: &Vec<Pass>| -> f64 {
                passes
                    .iter()
                    .map(|p| p.ms.parse::<f64>().expect("a device figure"))
                    .sum()
            };
            figures.push(Figure {
                arm,
                multiply_ms: multiplies[run].1,
                pass_ms: pass_ms(&runs[run]),
            });
            continue;
        }
        if let Some(rest) = line.strip_prefix("[wgpu-run-report] device_pass index=") {
            // `device_pass index=<i> interval=<phase> ms=<figure> kinds=<census>`
            let mut fields = rest.splitn(4, ' ');
            let _index = fields.next();
            let Some(interval) = fields.next().and_then(|f| f.strip_prefix("interval=")) else {
                continue;
            };
            let Some(ms) = fields.next().and_then(|f| f.strip_prefix("ms=")) else {
                continue;
            };
            let kinds = fields
                .next()
                .and_then(|f| f.strip_prefix("kinds="))
                .unwrap_or("")
                .to_string();
            if let Some(passes) = runs.last_mut() {
                passes.push(Pass {
                    interval: interval.to_string(),
                    ms: ms.to_string(),
                    kinds,
                });
            }
        }
    }

    for line in &parity {
        eprintln!("{line}");
    }
    assert_eq!(
        figures.len(),
        ROUNDS * ARMS.len(),
        "every round runs both arms: {} figures",
        figures.len()
    );
    let arm_figures =
        |name: &str| -> Vec<&Figure> { figures.iter().filter(|f| f.arm == name).collect() };
    let (_, words) = ARMS[0];
    let (_, narrow) = ARMS[1];
    let (a, b) = (arm_figures(words), arm_figures(narrow));
    let (m_a, m_b) = (
        median(&a.iter().map(|f| f.multiply_ms).collect::<Vec<_>>()),
        median(&b.iter().map(|f| f.multiply_ms).collect::<Vec<_>>()),
    );
    let (p_a, p_b) = (
        median(&a.iter().map(|f| f.pass_ms).collect::<Vec<_>>()),
        median(&b.iter().map(|f| f.pass_ms).collect::<Vec<_>>()),
    );
    let deltas: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(a, b)| a.multiply_ms - b.multiply_ms)
        .collect();
    let won = deltas.iter().filter(|d| **d < 0.0).count();
    eprintln!(
        "{MULTIPLIES_PER_PASS} packed multiplies, device time per pass: {words} {m_a:.3} ms, \
         {narrow} {m_b:.3} ms · the whole pass {p_a:.3} / {p_b:.3} ms of device time"
    );
    eprintln!(
        "median paired difference {:+.3} ms · {words} won {won} of {ROUNDS} rounds · ratio \
         {:.4} ({:.1} % off the multiply's time)",
        median(&deltas),
        m_b / m_a,
        (1.0 - m_a / m_b) * 100.0
    );

    // The plain pass, arms alternating in the same process: no instrument, so this
    // is the product's own submission shape.
    let of_arm = |name: &str| -> Vec<f64> {
        quiet
            .iter()
            .filter(|(arm, _)| arm == name)
            .map(|(_, ms)| *ms)
            .collect()
    };
    let (q_a, q_b) = (of_arm(words), of_arm(narrow));
    assert_eq!(
        q_a.len(),
        ROUNDS,
        "every round times the plain pass on both arms"
    );
    let quiet_deltas: Vec<f64> = q_a.iter().zip(&q_b).map(|(a, b)| a - b).collect();
    let quiet_won = quiet_deltas.iter().filter(|d| **d < 0.0).count();
    eprintln!(
        "plain pair pass (no instrument): {words} {:.3} ms, {narrow} {:.3} ms · median \
         paired difference {:+.3} ms · {words} won {quiet_won} of {ROUNDS} rounds",
        median(&q_a),
        median(&q_b),
        median(&quiet_deltas),
    );
}
