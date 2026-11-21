// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu device discovery + capabilities.
//!
//! [`wgpu_device`] returns a process-global singleton. [`select_vulkan_backend`]
//! switches that singleton to a Vulkan-only instance; it is not how the
//! runtime's `Device::Vulkan` is served — `rlx-runtime` maps that device to the
//! separate `rlx-vulkan` (ash/SPIR-V) backend.

use std::sync::OnceLock;
#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(not(target_family = "wasm"))]
const PREF_DEFAULT: u8 = 0;
#[cfg(not(target_family = "wasm"))]
const PREF_VULKAN: u8 = 1;

#[cfg(not(target_family = "wasm"))]
static BACKEND_PREF: AtomicU8 = AtomicU8::new(PREF_DEFAULT);

/// Detected wgpu adapter + device. We hold them together because
/// every command submission needs both the device (for encoding) and
/// the queue (for committing).
pub struct WgpuDevice {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pub backend: wgpu::Backend,
    /// The adapter's driver name, as `AdapterInfo` reports it, `None` when the
    /// HAL leaves it empty: only the Vulkan HAL fills `driver`/`driver_info`
    /// (the Vulkan driver's name and its version), and wgpu's Metal HAL reports
    /// neither. Kept rather than dropped so a caller can say the reading is
    /// absent instead of presenting a placeholder as a driver.
    pub driver: Option<String>,
    /// The driver's own version string, in the same `None`-when-empty form:
    /// wgpu's Vulkan HAL fills it from the Vulkan driver.
    pub driver_info: Option<String>,
}

/// Pick the subset of adapter features we want to enable on the device.
/// Shared by the native (blocking) and wasm (async) constructors so the
/// capability policy stays in one place.
fn required_features_for(adapter: &wgpu::Adapter) -> wgpu::Features {
    let adapter_feats = adapter.features();
    let mut required_features = wgpu::Features::empty();
    if adapter_feats.contains(wgpu::Features::SHADER_F16) {
        required_features |= wgpu::Features::SHADER_F16;
    }
    if adapter_feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    }
    if adapter_feats.contains(wgpu::Features::SUBGROUP) {
        required_features |= wgpu::Features::SUBGROUP;
    }
    // The run report's device-time view asks for pass timestamps, so the device
    // has to carry the capability even on runs that never print one — a
    // capability is not a behaviour, and asking for one the adapter does not
    // advertise is fatal here. This also covers `step_profile`, which writes
    // marks only when asked but needs the same feature to be present.
    if adapter_feats.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    required_features
}

fn make_instance(backends: wgpu::Backends) -> wgpu::Instance {
    // Release defaults set `VALIDATION_INDIRECT_CALL`, which clamps
    // `max_buffer_size` to `u32::MAX` (~4 GiB). That forces F5-scale arenas
    // onto sharded stripes. Opt into native Metal-sized buffers with
    // `RLX_WGPU_LARGE_BUFFERS=1` (or `WGPU_VALIDATION_INDIRECT_CALL=0`).
    // Oversized unsharded arenas still stage via the dedicated scratch tail
    // (not bind-window end) so live params are not clobbered.
    let mut flags = wgpu::InstanceFlags::default().with_env();
    if rlx_ir::env::flag("RLX_WGPU_LARGE_BUFFERS") {
        flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags,
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    })
}

const ADAPTER_OPTIONS: wgpu::RequestAdapterOptions = wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    compatible_surface: None,
    force_fallback_adapter: false,
    apply_limit_buckets: false,
};

/// The bytes of the largest single buffer this process will ask the device for, as
/// [`require_buffer_bytes`] declares them; `0` — no requirement — until it does.
#[cfg(not(target_family = "wasm"))]
static REQUIRED_BUFFER_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Declare the largest single buffer this process will ask the device for, before the
/// device exists.
///
/// The figure is the caller's and not this crate's: only the caller knows what it will
/// build, and `mahbot-inference`'s `LARGEST_BUFFER_BYTES` carries this fork's with its
/// provenance. Declaring nothing is the same as declaring zero — no requirement — and
/// leaves the chooser with its capability term; what the number then decides is
/// `better_candidate`'s rule. A call after the device exists is inert by design:
/// `mahbot-inference` declares from both its pre-device funnel and its compile path, and
/// only the first can move the choice.
#[cfg(not(target_family = "wasm"))]
pub fn require_buffer_bytes(bytes: u64) {
    REQUIRED_BUFFER_BYTES.store(bytes, std::sync::atomic::Ordering::Release);
}

/// How many device features an adapter advertises.
///
/// The count, not a name and not a family guess: a name says nothing about what a driver
/// will accept, and the vendor field is zero on Metal. Two adapters of one machine that
/// report the same features are indistinguishable to this rule, which is what makes the
/// tie-break "keep the power preference's own" well defined.
#[cfg(not(target_family = "wasm"))]
fn feature_count(features: wgpu::Features) -> u32 {
    features.bits().0.into_iter().map(u64::count_ones).sum()
}

/// One adapter as the chooser sees it: what it is, what it advertises, and the largest
/// buffer it can hold. A struct rather than a `wgpu::Adapter` so the rule below is
/// device-free and its tests need no GPU.
#[cfg(not(target_family = "wasm"))]
struct Candidate {
    info: wgpu::AdapterInfo,
    features: u32,
    max_buffer_size: u64,
}

#[cfg(not(target_family = "wasm"))]
impl Candidate {
    fn of(adapter: &wgpu::Adapter) -> Self {
        Self {
            info: adapter.get_info(),
            features: feature_count(adapter.features()),
            max_buffer_size: adapter.limits().max_buffer_size,
        }
    }

    /// Whether this candidate holds `required` bytes in one buffer: the shipped route, and
    /// what the run's largest allocation asks for.
    fn holds(&self, required: u64) -> bool {
        self.max_buffer_size >= required
    }

    /// Whether this candidate can hold `required` bytes at all: in one buffer, or striped
    /// across at most [`crate::buffer::MAX_SHARDS`] of them.
    ///
    /// The bound is the parked weight buffer's alone: it stripes at the adapter's own cap
    /// with nothing taken off it (the stage reserve is the activation arena's staging tail,
    /// not the weights'). Holding `required` in one buffer therefore means that buffer needs
    /// no stripe — not that the whole plan fits, since the activation arena is planned
    /// separately and may still be refused at the first compile, where the refusal steps down
    /// to the processor through `mahbot-inference`'s device choice.
    fn can_hold_striped(&self, required: u64) -> bool {
        crate::buffer::shard_cap_for(self.max_buffer_size)
            .saturating_mul(crate::buffer::MAX_SHARDS as u64)
            >= required
    }
}

/// The pick among `candidates`, or `None` to keep `today` — the adapter [`ADAPTER_OPTIONS`]
/// asks for.
///
/// Three terms, in this order. **Room** first: a candidate that cannot hold the run's
/// largest buffer even striped is out of the running, and one that holds it in a single
/// buffer beats one that would have to stripe it — the shipped route, and the point of the
/// rule, because `HighPerformance` on a dual-GPU Mac of the Intel era names the older
/// discrete part. **Capability** next: among what is left, the most advertised device
/// features wins, which is how the integrated GPU of such a machine beats the discrete one
/// that lacks the newer Metal feature set. **Tie**: equal keeps `today`, so a machine whose
/// GPUs this rule cannot tell apart keeps the adapter it has always used.
///
/// `today` is a candidate too, so a machine where nothing else qualifies — including one
/// where the power preference's own adapter is the only one with room — answers `None` and
/// changes nothing.
#[cfg(not(target_family = "wasm"))]
fn better_candidate(required: u64, today: &Candidate, candidates: &[Candidate]) -> Option<usize> {
    let score = |candidate: &Candidate| (candidate.holds(required), candidate.features);
    let mut best = today
        .can_hold_striped(required)
        .then(|| (score(today), None));
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.can_hold_striped(required) {
            continue;
        }
        let candidate_score = score(candidate);
        if best.is_none_or(|(best_score, _)| candidate_score > best_score) {
            best = Some((candidate_score, Some(index)));
        }
    }
    best.and_then(|(_, index)| index)
}

/// Whether this crate will run on an adapter of this description.
///
/// Software rasterisers (WARP on Windows, llvmpipe/lavapipe on Linux) report
/// `DeviceType::Cpu`. They only get picked when there is no real GPU, and they run the
/// shaders on the CPU: slower than rlx-cpu, with a third set of numerics. Report no wgpu
/// device instead so callers fall back to the CPU backend (rlx-hwprofile skips them for the
/// same reason). `RLX_WGPU_ALLOW_SOFTWARE=1` keeps them, for exercising the GPU path on a
/// machine without a GPU.
#[cfg(not(target_family = "wasm"))]
fn usable(info: &wgpu::AdapterInfo) -> bool {
    info.device_type != wgpu::DeviceType::Cpu || rlx_ir::env::flag("RLX_WGPU_ALLOW_SOFTWARE")
}

/// [`usable`], saying why not: the refusal is the only trace a software adapter leaves.
#[cfg(not(target_family = "wasm"))]
fn usable_or_say(info: &wgpu::AdapterInfo) -> bool {
    if usable(info) {
        return true;
    }
    eprintln!(
        "[rlx-wgpu] skipping software adapter {:?} ({:?}); set RLX_WGPU_ALLOW_SOFTWARE=1 to use it",
        info.name, info.backend
    );
    false
}

/// The limits this crate asks a device for: the adapter's own, with the two windowing
/// clamps its ops need.
#[cfg(not(target_family = "wasm"))]
fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let mut limits = adapter.limits();
    // When large buffers are enabled, keep storage *bindings* ≤ ~4 GiB so
    // ops window into the single physical buffer (whole-buffer binds of
    // >4 GiB returned all-zero F5 DiT outputs on wgpu-Metal).
    if rlx_ir::env::flag("RLX_WGPU_LARGE_BUFFERS") {
        const BIND_CAP: u64 = (4u64 << 30) - 256;
        if limits.max_buffer_size > BIND_CAP {
            limits.max_storage_buffer_binding_size =
                limits.max_storage_buffer_binding_size.min(BIND_CAP);
            limits.max_uniform_buffer_binding_size =
                limits.max_uniform_buffer_binding_size.min(BIND_CAP);
        }
    }
    // `RLX_WGPU_MAX_BIND_MB` caps the storage-binding size the device is
    // created with, to exercise the over-the-binding-limit paths (arena
    // windows, parked weights, staging) on an adapter whose real limit is
    // larger — e.g. NVIDIA's 2 GiB on a Mac. It already capped the GGUF
    // path; capping the device makes every path see the same limit.
    if let Some(mb) = rlx_ir::env::var("RLX_WGPU_MAX_BIND_MB").and_then(|v| v.parse::<u64>().ok()) {
        let cap = (mb << 20) / 256 * 256;
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.min(cap);
    }
    if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
        eprintln!(
            "[rlx-wgpu] adapter limits: max_buffer_size={:.3}GiB max_storage_binding={:.3}GiB",
            limits.max_buffer_size as f64 / ((1u64 << 30) as f64),
            limits.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
        );
    }
    limits
}

/// Create the device on `adapter`, with this crate's limits and feature policy; `None` when
/// the adapter cannot produce one. The adapter is the caller's to keep.
#[cfg(not(target_family = "wasm"))]
fn open_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    let limits = device_limits(adapter);
    let required_features = required_features_for(adapter);
    match pollster::block_on(adapter.request_device(&device_descriptor(required_features, limits)))
    {
        Ok((device, queue)) => {
            if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
                let dl = device.limits();
                eprintln!(
                    "[rlx-wgpu] device limits: max_buffer_size={:.3}GiB \
                     max_storage_binding={:.3}GiB",
                    dl.max_buffer_size as f64 / ((1u64 << 30) as f64),
                    dl.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
                );
            }
            Some((device, queue))
        }
        Err(e) => {
            eprintln!(
                "rlx-wgpu request_device failed on {}: {e}",
                adapter.get_info().name
            );
            None
        }
    }
}

/// The one line [`better_adapter`] prints when there is a choice to make: which adapter was
/// taken, how many of the machine's adapters can hold the run's buffer, and the term that
/// decided the pick — room, features, or the power preference's own kept. On a machine nobody
/// here can test this line is the whole diagnosis, so it is a pure function of the decision
/// and tested as one.
#[cfg(not(target_family = "wasm"))]
fn choice_line(
    today: &Candidate,
    chosen: Option<&Candidate>,
    candidates: &[Candidate],
    required: u64,
) -> String {
    let (name, backend, features) = match chosen {
        Some(candidate) => (
            &candidate.info.name,
            candidate.info.backend,
            candidate.features,
        ),
        None => (&today.info.name, today.info.backend, today.features),
    };
    let seen = candidates.len() + 1;
    let kept = candidates.iter().filter(|c| c.holds(required)).count()
        + usize::from(today.holds(required));
    let capacity = if required > 0 {
        format!(
            ", {kept} of {seen} can hold the {} MiB the run needs in one buffer",
            required.div_ceil(1 << 20)
        )
    } else {
        format!(", {seen} candidates")
    };
    // The reason names the term that decided the pick. Room first: the power preference's own
    // may not hold the buffer at all — not even striped, which the score cannot express, both
    // then reading `holds = false` — or may hold it only by striping. Features otherwise: two
    // adapters on the same side of that line were compared on features, and the line says so.
    let reason = match chosen {
        None => ", the power preference's own",
        Some(_) if !today.can_hold_striped(required) => {
            ", the power preference's own cannot hold the run's buffer even striped"
        }
        Some(c) if c.holds(required) && !today.holds(required) => {
            ", the power preference's own cannot hold the run's buffer in one buffer"
        }
        Some(_) => ", more features than the power preference's",
    };
    format!("[rlx-wgpu] adapter {name} ({backend:?}): {features} features{capacity}{reason}")
}

/// The adapter to run on *instead of* [`ADAPTER_OPTIONS`]'s own, or `None` to keep it: the
/// candidate [`better_candidate`] picks among the adapters of the same API.
///
/// Scoped to the power preference's own API on purpose: on Apple the request spans Metal and
/// MoltenVK, and a cross-API pick would move the record's device line and the calibration
/// cache key on a machine whose API order is a separate decision ([`default_backends`]). The
/// enumeration is scoped with it, so a machine that has a second backend installed does not
/// pay to enumerate it. A software rasteriser is not a candidate unless the operator asked
/// for one ([`usable`]).
///
/// A machine with no other usable adapter of this API answers `None` without printing: there
/// is no choice to explain, which is also what a second adapter of identical info or a
/// software rasteriser ([`usable`]) leaves it with.
#[cfg(not(target_family = "wasm"))]
fn better_adapter(instance: &wgpu::Instance, today: &wgpu::Adapter) -> Option<wgpu::Adapter> {
    let required = REQUIRED_BUFFER_BYTES.load(std::sync::atomic::Ordering::Acquire);
    let today_view = Candidate::of(today);
    let api = wgpu::Backends::from(today_view.info.backend);
    let mut candidates = Vec::new();
    let mut adapters = Vec::new();
    for adapter in pollster::block_on(instance.enumerate_adapters(api)) {
        let view = Candidate::of(&adapter);
        if view.info == today_view.info || !usable(&view.info) {
            continue;
        }
        candidates.push(view);
        adapters.push(adapter);
    }
    if candidates.is_empty() {
        // One adapter of this API: there is no choice to make and nothing to print, so the
        // machine keeps exactly the behaviour it had.
        return None;
    }
    let chosen = better_candidate(required, &today_view, &candidates);
    eprintln!(
        "{}",
        choice_line(
            &today_view,
            chosen.map(|index| &candidates[index]),
            &candidates,
            required
        )
    );
    chosen.map(|index| adapters.swap_remove(index))
}

/// Whether the wide (subgroup) reduction may be taken on `device`.
///
/// The one wide reduction in this crate is the packed one-row GEMV's subgroup arm
/// (`gguf_gpu`'s `dequant_gemv_wide`, admitted by [`gguf_gpu::gemv_subgroup_probe`]'s
/// width). On Metal it is taken on Apple silicon and nowhere else, because on the Intel
/// and AMD Macs it produces wrong numbers rather than failing: llama.cpp's Metal backend
/// disables its simdgroup reduction on all Intel Macs and forces a narrower one on the AMD
/// parts of this era for exactly that reason. Vega is the same generation and the same
/// doubt, so the boundary is not "older than Vega" but the GPU family itself — the fork's
/// rule is that an uncertain signal resolves to the slow-but-correct path, and the
/// portable arm (`dequant_gemv_tiled`, a workgroup-memory reduction) is that path.
///
/// Scoped to the Metal backend explicitly. The same reduction is measured and correct on a
/// discrete Vulkan adapter (see `GEMV_WIDE_DEFAULT`'s doc), so a name-based test for
/// "Intel" would switch off a correct arm on an Intel Arc under Vulkan.
///
/// The backend is the one `device` reports, not the process-global device's: the probe is
/// handed a device by its caller, and this crate can hold a second one
/// (`vulkan_device`), which would otherwise get the answer for the first.
///
/// The family signal is the process's own architecture, which is a compile-time fact:
/// every Apple-silicon Mac runs an `aarch64` build and every Intel Mac an `x86_64` one.
/// The one case it gets wrong is an `x86_64` build under Rosetta on an Apple-silicon Mac,
/// where it answers "not Apple silicon" — slower and correct, which is the side the rule
/// resolves uncertainty to.
pub(crate) fn wide_reduction_admitted(device: &wgpu::Device) -> bool {
    let info = device.adapter_info();
    let admitted = wide_reduction_admitted_on(info.backend, cfg!(target_arch = "aarch64"));
    if !admitted {
        // Reached once per process: its only caller fills the probe cache, so the line
        // cannot repeat.
        eprintln!(
            "[rlx-wgpu] adapter {} ({:?}): this binary does not take the wide one-row \
             reduction on Metal — the portable reduction serves it",
            info.name, info.backend
        );
    }
    admitted
}

/// [`wide_reduction_admitted`]'s rule, without a device.
fn wide_reduction_admitted_on(backend: wgpu::Backend, apple_silicon: bool) -> bool {
    backend != wgpu::Backend::Metal || apple_silicon
}

fn device_descriptor(
    required_features: wgpu::Features,
    limits: wgpu::Limits,
) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("rlx-wgpu device"),
        required_features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }
}

impl WgpuDevice {
    /// Native: block on adapter + device discovery via `pollster`.
    #[cfg(not(target_family = "wasm"))]
    fn new_with_backends(backends: wgpu::Backends) -> Option<Self> {
        let instance = make_instance(backends);
        let today = pollster::block_on(instance.request_adapter(&ADAPTER_OPTIONS)).ok()?;
        let (mut adapter, chose_another) = match better_adapter(&instance, &today) {
            Some(picked) => (picked, true),
            None => (today.clone(), false),
        };
        if !usable_or_say(&adapter.get_info()) {
            return None;
        }
        let (device, queue) = match open_device(&adapter) {
            Some(opened) => opened,
            // The adapter this rule picked could not produce a device. The power
            // preference's own is asked once more, so choosing between two GPUs can never
            // cost a machine the one that worked before the rule existed.
            None if chose_another && usable(&today.get_info()) => {
                eprintln!(
                    "[rlx-wgpu] the adapter this rule picked could not produce a device; \
                     asking the power preference's own"
                );
                adapter = today;
                open_device(&adapter)?
            }
            None => return None,
        };
        let info = adapter.get_info();

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            name: info.name,
            backend: info.backend,
            // `AdapterInfo` carries both as empty strings on a HAL that does not
            // report a driver; `None` is what a record's provenance line then
            // presents as `unknown`.
            driver: Some(info.driver).filter(|d| !d.is_empty()),
            driver_info: Some(info.driver_info).filter(|d| !d.is_empty()),
        })
    }

    /// WASM: the browser cannot block its event loop, so adapter + device
    /// discovery is genuinely async — callers `.await` [`init_wgpu_device`]
    /// once at startup, after which [`wgpu_device`] returns synchronously.
    #[cfg(target_family = "wasm")]
    async fn new_with_backends_async(backends: wgpu::Backends) -> Option<Self> {
        let instance = make_instance(backends);
        let adapter = instance.request_adapter(&ADAPTER_OPTIONS).await.ok()?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let required_features = required_features_for(&adapter);

        let (device, queue) = adapter
            .request_device(&device_descriptor(required_features, limits))
            .await
            .ok()?;
        // A lost device (iOS Safari loses it when the tab's GPU memory runs
        // out) and uncaptured errors (out-of-memory, validation) are recorded
        // rather than panicking, so a run can return them as its error and a
        // caller can fall back to another backend.
        device.set_device_lost_callback(|reason, msg| {
            DEVICE_LOST.store(true, std::sync::atomic::Ordering::SeqCst);
            record_gpu_error(format!("WebGPU device lost ({reason:?}): {msg}"));
        });
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            record_gpu_error(format!("WebGPU error: {e}"));
        }));

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            name: info.name,
            backend: info.backend,
            driver: Some(info.driver).filter(|d| !d.is_empty()),
            driver_info: Some(info.driver_info).filter(|d| !d.is_empty()),
        })
    }

    #[cfg(not(target_family = "wasm"))]
    fn new_default() -> Option<Self> {
        // An explicit `WGPU_BACKEND` names one API and is honoured exactly: the
        // second attempt is for a machine whose preference is missing, not for a
        // caller who asked for a HAL.
        if let Some(backends) = wgpu::Backends::from_env() {
            return Self::new_with_backends(backends);
        }
        first_usable(
            default_backends(),
            fallback_backends(),
            Self::new_with_backends,
        )
    }
}

/// Ask `first`, then — only where a fallback is given — the fallback, and return
/// the first attempt that yields something.
#[cfg(not(target_family = "wasm"))]
fn first_usable<T>(
    first: wgpu::Backends,
    fallback: Option<wgpu::Backends>,
    mut attempt: impl FnMut(wgpu::Backends) -> Option<T>,
) -> Option<T> {
    if let Some(found) = attempt(first) {
        return Some(found);
    }
    let second = fallback?;
    eprintln!("[rlx-wgpu] no usable {first:?} device; asking {second:?}");
    attempt(second)
}

/// The API this platform asks for first.
#[cfg(not(target_family = "wasm"))]
fn default_backends() -> wgpu::Backends {
    #[cfg(target_os = "windows")]
    {
        // DX12 first: the preference this fork has always had on Windows, and
        // nothing here is measured there to justify reversing it.
        wgpu::Backends::DX12
    }
    #[cfg(target_os = "linux")]
    {
        // WSL Ubuntu + native Linux: prefer Vulkan (NVIDIA passthrough).
        wgpu::Backends::VULKAN
    }
    #[cfg(target_vendor = "apple")]
    {
        // Every Apple platform (macOS, iOS, tvOS, visionOS): native Metal,
        // plus MoltenVK-translated Vulkan where the user installed it.
        wgpu::Backends::METAL | wgpu::Backends::VULKAN
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_vendor = "apple")))]
    {
        wgpu::Backends::all()
    }
}

/// The second attempt, where a platform has one: Windows asks Vulkan, already
/// compiled into the binary, so a machine whose DX12 adapter is missing or cannot
/// produce a device runs instead of reporting no adapter at all.
///
/// Sequential, not a `DX12 | VULKAN` mask: a mask builds the Vulkan instance and
/// loads its driver stack on every Windows machine, including the ones DX12
/// already serves, and lets wgpu's own ordering pick between them. No other
/// platform has a second attempt — Linux asks Vulkan alone, and Apple asks Metal
/// plus MoltenVK in one instance.
#[cfg(not(target_family = "wasm"))]
fn fallback_backends() -> Option<wgpu::Backends> {
    cfg!(target_os = "windows").then_some(wgpu::Backends::VULKAN)
}

// SAFETY: wgpu's Device + Queue are documented thread-safe.
unsafe impl Send for WgpuDevice {}
unsafe impl Sync for WgpuDevice {}

#[cfg(not(target_family = "wasm"))]
fn default_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE.get_or_init(WgpuDevice::new_default).as_ref()
}

#[cfg(not(target_family = "wasm"))]
fn vulkan_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::new_with_backends(wgpu::Backends::VULKAN))
        .as_ref()
}

/// Point this crate's own singleton at a Vulkan-only instance. Call before the
/// first [`wgpu_device`] use in that process. Nothing in the runtime calls it:
/// `Device::Vulkan` there is served by `rlx-vulkan`, not by this crate.
#[cfg(not(target_family = "wasm"))]
pub fn select_vulkan_backend() {
    BACKEND_PREF.store(PREF_VULKAN, Ordering::SeqCst);
}

/// No Vulkan path in the browser — the only wasm adapter is WebGPU.
#[cfg(target_family = "wasm")]
pub fn select_vulkan_backend() {}

/// True when a Vulkan adapter is reachable (MoltenVK on macOS, native on Linux/Windows).
#[cfg(not(target_family = "wasm"))]
pub fn is_vulkan_available() -> bool {
    vulkan_device().is_some()
}

/// No Vulkan in the browser.
#[cfg(target_family = "wasm")]
pub fn is_vulkan_available() -> bool {
    false
}

/// Get or initialize the global wgpu device singleton. Returns None
/// on systems with no compatible adapter.
#[cfg(not(target_family = "wasm"))]
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    if BACKEND_PREF.load(Ordering::SeqCst) == PREF_VULKAN {
        vulkan_device()
    } else {
        default_device()
    }
}

// ── WASM device singleton ──────────────────────────────────────────────
//
// The browser cannot block on adapter/device discovery, so the singleton
// is populated asynchronously by [`init_wgpu_device`] (called once at
// startup from JS via `wasm-bindgen-futures`). After that completes,
// [`wgpu_device`] returns the cached device synchronously, so the rest of
// the (synchronous) backend is unchanged.
#[cfg(target_family = "wasm")]
static WASM_DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();

/// Initialize the WebGPU device. Idempotent; returns `true` when a device
/// is available. Must be awaited before any [`wgpu_device`] call on wasm.
#[cfg(target_family = "wasm")]
pub async fn init_wgpu_device() -> bool {
    if let Some(slot) = WASM_DEVICE.get() {
        return slot.is_some();
    }
    let dev = WgpuDevice::new_with_backends_async(wgpu::Backends::BROWSER_WEBGPU).await;
    let available = dev.is_some();
    let _ = WASM_DEVICE.set(dev);
    available
}

#[cfg(target_family = "wasm")]
static DEVICE_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(target_family = "wasm")]
static GPU_ERROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Keep the first error since the last [`take_gpu_error`]: later ones are
/// usually its consequences.
#[cfg(target_family = "wasm")]
fn record_gpu_error(msg: String) {
    if let Ok(mut e) = GPU_ERROR.lock() {
        e.get_or_insert(msg);
    }
}

/// The first device-lost or uncaptured WebGPU error since the last call.
#[cfg(target_family = "wasm")]
pub fn take_gpu_error() -> Option<String> {
    GPU_ERROR.lock().ok().and_then(|mut e| e.take())
}

/// Whether the browser lost the WebGPU device: nothing more runs on it.
#[cfg(target_family = "wasm")]
pub fn device_lost() -> bool {
    DEVICE_LOST.load(std::sync::atomic::Ordering::SeqCst)
}

/// No browser device natively.
#[cfg(not(target_family = "wasm"))]
pub fn device_lost() -> bool {
    false
}

/// Get the wgpu device. On wasm this returns `None` until
/// [`init_wgpu_device`] has been awaited.
#[cfg(target_family = "wasm")]
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    WASM_DEVICE.get().and_then(|d| d.as_ref())
}

/// The adapter's name, unqualified: what a console line or a diagnostic prints.
/// On-disk caches and record columns take [`adapter_label`] instead.
pub fn adapter_name() -> Option<String> {
    wgpu_device().map(|d| d.name.clone())
}

/// The adapter as one label — its name qualified by the backend that reported
/// it — for record columns and on-disk caches.
pub fn adapter_label() -> Option<String> {
    wgpu_device().map(|d| format!("{} ({:?})", d.name, d.backend))
}

/// The queue paired with the process-global device.
///
/// For one-shot probes that are handed only a `&wgpu::Device` and still have to
/// submit (`gguf_gpu::gemv_subgroup_probe`). This crate has one device per
/// process — [`wgpu_device`] — and every encoder in it is handed that device, so
/// the queue returned here is the one that device was requested with.
pub(crate) fn device_queue() -> Option<&'static wgpu::Queue> {
    wgpu_device().map(|d| &d.queue)
}

/// Whether the shader compiler admits 16-bit floats inside 32-bit arithmetic —
/// naga's `SHADER_FLOAT16_IN_FLOAT32`, the capability `unpack2x16float` needs.
///
/// Not the same bit as `Features::SHADER_F16`, which gates the `f16` *type*: wgpu
/// maps the unpack builtin's capability from the adapter's
/// `DownlevelFlags::SHADER_F16_IN_F32` (`wgpu-naga-bridge`'s
/// `features_to_naga_capabilities`), and `Device` exposes no downlevel flags, so
/// this reads the singleton's adapter — the same one-device-per-process argument
/// [`device_queue`] makes. `false` without a device, which selects the spelling
/// that needs no capability.
pub(crate) fn shader_f16_in_f32_supported() -> bool {
    wgpu_device().is_some_and(|d| {
        d.adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SHADER_F16_IN_F32)
    })
}

static FAST_MATH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Opt into fast WGSL arithmetic: `/`, `sqrt` and `1/sqrt` use the native
/// WGSL operators instead of rlxsl's correctly-rounded helpers. Native `/` may
/// be 2.5 ULP off and Metal's `sqrt` is often 1 ULP off, so results stop
/// matching the CPU bit for bit. `fmod` stays exact (a fast one is wrong by
/// whole multiples, not ULPs). It also lets an `m <= 4` matmul split K across
/// threads (`matmul_gemv`'s 32-column form), which adds K in a different order
/// from every other matmul kernel. Off by default.
///
/// Kernels are built once per process, so call this before the first
/// compile; a later call does not rebuild kernels that already exist.
pub fn set_fast_math(on: bool) {
    FAST_MATH.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_fast_math`] is on.
pub fn fast_math() -> bool {
    FAST_MATH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether discrete Vulkan/DX12 should still run elementwise, norm and
/// structural ops on the host. Off by default: that policy was a workaround
/// for a KittenTTS "NSF collapse" whose real causes (arena-stripe addressing
/// past the 2 GiB binding limit) are fixed, and it cost EEGNet 295 s against
/// 6.4 s on an RTX 3080 Ti with identical score bits. `RLX_WGPU_DISCRETE_HOST=1`
/// restores it.
pub fn discrete_hosts_ops() -> bool {
    coop_discrete_backend() && rlx_ir::env::flag("RLX_WGPU_DISCRETE_HOST")
}

/// True on Vulkan or DX12 — the discrete-adapter predicate behind this
/// backend's host-fallback routing and a term in its cooperative-matrix
/// eligibility checks. On DX12 no cooperative path can follow from it: this
/// engine exposes no cooperative matrices there (the adapter reports an empty
/// property list and its shader compilers cannot emit them).
pub fn coop_discrete_backend() -> bool {
    wgpu_device()
        .map(|d| matches!(d.backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12))
        .unwrap_or(false)
}

/// True when the process-global device advertises `Features::SHADER_F16`.
///
/// The one adapter term the two-sided band kernel
/// (`kernels::attention_band_kernel`) reads: it selects the f16 staging module
/// where the feature is there and the f32 one where it is not, and the route is
/// the same either way. False without a device.
pub fn shader_f16_supported() -> bool {
    wgpu_device()
        .map(|d| d.device.features().contains(wgpu::Features::SHADER_F16))
        .unwrap_or(false)
}

/// True when the adapter reports 8×8×8 f32 cooperative-matrix support, which
/// selects `matmul_coop_f32_portable` automatically on a discrete adapter.
pub fn coop_f32_8x8_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 8
            && p.n_size == 8
            && p.k_size == 8
            && p.ab_type == wgpu::CooperativeScalarType::F32
            && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}

/// True when the adapter reports 8×8×8 f16 cooperative-matrix support with an
/// f16 accumulator — the one shape `matmul_coop16` uses. Apple reports it
/// (`simdgroup_half8x8`); an RTX 3080 Ti reports only 16×16×16, 16×8×16 and
/// 16×8×8, and there `matmul_coop16` failed its self-check under every
/// load/store convention, because the shape itself is unsupported.
pub fn coop_f16_8x8_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 8
            && p.n_size == 8
            && p.k_size == 8
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F16
    })
}

/// True when the adapter reports 16×16×16 f16 cooperative-matrix support
/// (the NVIDIA / AMD tensor-core path on Vulkan; DX12 reports none).
pub fn coop_f16_16x16_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && (p.cr_type == wgpu::CooperativeScalarType::F16
                || p.cr_type == wgpu::CooperativeScalarType::F32)
    })
}

/// True when f16 operands with f32 accumulator are available (preferred on RTX).
pub fn coop_f16_16x16_f32_acc_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    /// A first attempt that yields a device ends the request: the second API's
    /// driver stack is never loaded on a machine the first serves.
    #[test]
    fn a_working_first_attempt_ends_the_request() {
        let mut asked = Vec::new();
        let found = first_usable(wgpu::Backends::DX12, Some(wgpu::Backends::VULKAN), |b| {
            asked.push(b);
            Some(b)
        });
        assert_eq!(found, Some(wgpu::Backends::DX12));
        assert_eq!(asked, [wgpu::Backends::DX12]);
    }

    /// A first attempt that yields nothing reaches the fallback, after it.
    #[test]
    fn a_failed_first_attempt_reaches_the_fallback() {
        let mut asked = Vec::new();
        let found = first_usable(wgpu::Backends::DX12, Some(wgpu::Backends::VULKAN), |b| {
            asked.push(b);
            (b == wgpu::Backends::VULKAN).then_some(b)
        });
        assert_eq!(found, Some(wgpu::Backends::VULKAN));
        assert_eq!(asked, [wgpu::Backends::DX12, wgpu::Backends::VULKAN]);
    }

    /// The pair Windows ships, driven through the order it ships with: DX12
    /// first, Vulkan second.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_asks_dx12_then_vulkan() {
        let mut asked = Vec::new();
        let found = first_usable(default_backends(), fallback_backends(), |b| {
            asked.push(b);
            (b == wgpu::Backends::VULKAN).then_some(b)
        });
        assert_eq!(found, Some(wgpu::Backends::VULKAN));
        assert_eq!(asked, [wgpu::Backends::DX12, wgpu::Backends::VULKAN]);
    }

    /// The wide reduction is taken on Apple silicon and nowhere else on Metal, and
    /// on every other API it is not this rule's to refuse: the same arm is measured
    /// and correct on a discrete Vulkan adapter, so a family test that also caught
    /// an Intel Arc under Vulkan would switch off a correct route.
    #[test]
    fn the_wide_reduction_is_apple_silicon_only_on_metal() {
        assert!(!wide_reduction_admitted_on(wgpu::Backend::Metal, false));
        assert!(wide_reduction_admitted_on(wgpu::Backend::Metal, true));
        for backend in [
            wgpu::Backend::Vulkan,
            wgpu::Backend::Dx12,
            wgpu::Backend::Gl,
        ] {
            assert!(
                wide_reduction_admitted_on(backend, false),
                "{backend:?} keeps the arm whatever the process architecture is"
            );
        }
    }

    /// A candidate the rule can name, with the two numbers it reads.
    fn candidate(name: &str, device_type: wgpu::DeviceType, features: u32, mib: u64) -> Candidate {
        Candidate {
            info: wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type,
                device_pci_bus_id: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
                subgroup_min_size: 4,
                subgroup_max_size: 64,
                transient_saves_memory: None,
                limit_bucket: None,
            },
            features,
            max_buffer_size: mib << 20,
        }
    }

    /// The dual-GPU Mac this rule exists for: the power preference names the older
    /// discrete part, the integrated one has the newer feature set, and both can hold
    /// the run's buffer in one buffer — so the one with more features is taken, and the
    /// line says that. With nothing declared the same candidate wins: the room term is a
    /// floor, not the rule's reason for existing.
    #[test]
    fn the_adapter_with_more_features_wins_when_both_hold_the_buffer() {
        let today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            2048,
        );
        let integrated = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 1536);
        let others = std::slice::from_ref(&integrated);
        assert_eq!(better_candidate(1_050 << 20, &today, others), Some(0));
        assert_eq!(better_candidate(0, &today, others), Some(0));
        assert_eq!(
            choice_line(&today, Some(&integrated), others, 1_050 << 20),
            "[rlx-wgpu] adapter Intel UHD 630 (Metal): 24 features, 2 of 2 can hold the \
             1050 MiB the run needs in one buffer, more features than the power preference's"
        );
    }

    /// Room is the first term: an adapter that would have to stripe the run's buffer loses
    /// to one that holds it in one buffer, even against more features — and the line has to
    /// say that, not claim a feature win it does not have.
    #[test]
    fn an_adapter_that_would_stripe_the_buffer_loses_to_one_that_holds_it() {
        let today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            1024,
        );
        let integrated = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 16, 1536);
        let others = std::slice::from_ref(&integrated);
        assert_eq!(better_candidate(1_050 << 20, &today, others), Some(0));
        assert_eq!(
            choice_line(&today, Some(&integrated), others, 1_050 << 20),
            "[rlx-wgpu] adapter Intel UHD 630 (Metal): 16 features, 1 of 2 can hold the \
             1050 MiB the run needs in one buffer, the power preference's own cannot hold \
             the run's buffer in one buffer"
        );
    }

    /// An adapter that cannot hold the buffer even striped is out of the running, and
    /// then the power preference's own is kept — or, if it is the one without room, the
    /// candidate that has it is taken.
    #[test]
    fn a_candidate_without_room_even_striped_is_out_of_the_running() {
        let today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            2048,
        );
        // 8 MiB × MAX_SHARDS is 512 MiB: not enough for a 1050 MiB buffer.
        let integrated = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 8);
        assert!(!integrated.can_hold_striped(1_050 << 20));
        assert_eq!(better_candidate(1_050 << 20, &today, &[integrated]), None);
        // The same adapter in the power preference's place: it is not in the running, so
        // the candidate with room is taken whatever it advertises.
        let small = candidate("AMD Radeon Pro 5500M", wgpu::DeviceType::DiscreteGpu, 20, 8);
        let room = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 16, 1536);
        assert_eq!(better_candidate(1_050 << 20, &small, &[room]), Some(0));
        // Neither can hold it, even striped — the case the drop term exists for. Without it
        // the more-featured candidate would be taken for a run neither GPU can carry, which
        // is how a machine ends up moved off its working adapter.
        let other_small = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 8);
        assert_eq!(better_candidate(1_050 << 20, &small, &[other_small]), None);
    }

    /// Two adapters that would both have to stripe the run's buffer are still compared on
    /// features, and the one with more is taken.
    #[test]
    fn two_adapters_that_both_stripe_are_compared_on_features() {
        let today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            1024,
        );
        let integrated = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 1024);
        assert_eq!(
            better_candidate(1_050 << 20, &today, &[integrated]),
            Some(0)
        );
    }

    /// Both terms equal keeps today's choice, and so does a candidate that is worse on
    /// both — a machine whose GPUs this rule cannot tell apart does not move.
    #[test]
    fn a_tie_keeps_the_power_preferences_own() {
        let today = candidate("Apple M3 Pro", wgpu::DeviceType::IntegratedGpu, 24, 4096);
        let same = candidate("Apple M3 Pro", wgpu::DeviceType::IntegratedGpu, 24, 4096);
        let fewer = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 20, 4096);
        assert_eq!(better_candidate(0, &today, &[same]), None);
        assert_eq!(better_candidate(0, &today, &[fewer]), None);
    }

    /// The line a machine leaves behind, in the three shapes the decision has: the power
    /// preference's own kept, a pick made on features between two adapters that both have to
    /// stripe the run's buffer, and a pick made on room when the power preference's own
    /// cannot hold it at all. The wording is the contract — it is all anyone has to diagnose
    /// such a machine with.
    #[test]
    fn the_choice_line_names_the_adapter_and_the_reason() {
        let today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            2048,
        );
        let integrated = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 1536);
        let others = std::slice::from_ref(&integrated);
        // The other candidate cannot hold the run's buffer in one buffer, and a run that
        // declared none: both end on the adapter the machine has always used.
        assert_eq!(
            choice_line(&today, None, others, 2_000 << 20),
            "[rlx-wgpu] adapter AMD Radeon Pro 5500M (Metal): 20 features, 1 of 2 can hold \
             the 2000 MiB the run needs in one buffer, the power preference's own"
        );
        assert_eq!(
            choice_line(&today, None, others, 0),
            "[rlx-wgpu] adapter AMD Radeon Pro 5500M (Metal): 20 features, 2 candidates, \
             the power preference's own"
        );
        // Both would stripe the run's buffer, so the pick was made on features: the line says
        // that rather than naming a capacity term that decided nothing.
        let striped_today = candidate(
            "AMD Radeon Pro 5500M",
            wgpu::DeviceType::DiscreteGpu,
            20,
            1024,
        );
        let striped_other = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 24, 1024);
        let striped = std::slice::from_ref(&striped_other);
        assert_eq!(
            choice_line(&striped_today, Some(&striped_other), striped, 1_050 << 20),
            "[rlx-wgpu] adapter Intel UHD 630 (Metal): 24 features, 0 of 2 can hold the 1050 \
             MiB the run needs in one buffer, more features than the power preference's"
        );
        // The power preference's own cannot carry the buffer even striped, so the pick was made
        // on room — the line must not claim a feature win that decided nothing.
        let cannot = candidate("AMD Radeon Pro 5500M", wgpu::DeviceType::DiscreteGpu, 20, 8);
        let striped_only = candidate("Intel UHD 630", wgpu::DeviceType::IntegratedGpu, 16, 1024);
        assert_eq!(
            choice_line(
                &cannot,
                Some(&striped_only),
                std::slice::from_ref(&striped_only),
                1_050 << 20
            ),
            "[rlx-wgpu] adapter Intel UHD 630 (Metal): 16 features, 0 of 2 can hold the 1050 \
             MiB the run needs in one buffer, the power preference's own cannot hold the run's \
             buffer even striped"
        );
    }
}
