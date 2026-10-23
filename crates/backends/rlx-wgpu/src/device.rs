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
use std::sync::atomic::{AtomicU8, Ordering};

const PREF_DEFAULT: u8 = 0;
const PREF_VULKAN: u8 = 1;

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
    // advertise is fatal here.
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
    /// Block on adapter + device discovery via `pollster`.
    fn new_with_backends(backends: wgpu::Backends) -> Option<Self> {
        let instance = make_instance(backends);
        let adapter = pollster::block_on(instance.request_adapter(&ADAPTER_OPTIONS)).ok()?;

        let info = adapter.get_info();
        let mut limits = adapter.limits();
        // When large buffers are enabled, keep storage *bindings* ≤ ~4 GiB so
        // ops window into the single physical buffer (whole-buffer binds of
        // >4 GiB returned all-zero F5 DiT outputs on wgpu-Metal).
        if rlx_ir::env::flag("RLX_WGPU_LARGE_BUFFERS") {
            const BIND_CAP: u64 = (4u64 << 30) - 256;
            if limits.max_buffer_size > BIND_CAP {
                limits.max_storage_buffer_binding_size =
                    limits.max_storage_buffer_binding_size.min(BIND_CAP);
                limits.max_uniform_buffer_binding_size =
                    limits.max_uniform_buffer_binding_size.min(BIND_CAP);
            }
        }
        if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
            eprintln!(
                "[rlx-wgpu] adapter limits: max_buffer_size={:.3}GiB max_storage_binding={:.3}GiB",
                limits.max_buffer_size as f64 / ((1u64 << 30) as f64),
                limits.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
            );
        }
        let required_features = required_features_for(&adapter);

        let (device, queue) = match pollster::block_on(
            adapter.request_device(&device_descriptor(required_features, limits)),
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rlx-wgpu request_device failed: {e}");
                return None;
            }
        };
        if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
            let dl = device.limits();
            eprintln!(
                "[rlx-wgpu] device limits: max_buffer_size={:.3}GiB max_storage_binding={:.3}GiB",
                dl.max_buffer_size as f64 / ((1u64 << 30) as f64),
                dl.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
            );
        }

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

    fn new_default() -> Option<Self> {
        Self::new_with_backends(default_backends())
    }
}

fn default_backends() -> wgpu::Backends {
    if let Some(b) = wgpu::Backends::from_env() {
        return b;
    }
    #[cfg(target_os = "windows")]
    {
        // Both, as one unordered set, with no preference of this crate's own:
        // wgpu registers Vulkan before DX12 and sorts adapters only by device
        // type, which both report identically for the same card — so Vulkan
        // wins, and DX12 is used only when no Vulkan adapter survives.
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
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

// SAFETY: wgpu's Device + Queue are documented thread-safe.
unsafe impl Send for WgpuDevice {}
unsafe impl Sync for WgpuDevice {}

fn default_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE.get_or_init(WgpuDevice::new_default).as_ref()
}

fn vulkan_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::new_with_backends(wgpu::Backends::VULKAN))
        .as_ref()
}

/// Point this crate's own singleton at a Vulkan-only instance. Call before the
/// first [`wgpu_device`] use in that process. Nothing in the runtime calls it:
/// `Device::Vulkan` there is served by `rlx-vulkan`, not by this crate.
pub fn select_vulkan_backend() {
    BACKEND_PREF.store(PREF_VULKAN, Ordering::SeqCst);
}

/// True when a Vulkan adapter is reachable (MoltenVK on macOS, native on Linux/Windows).
pub fn is_vulkan_available() -> bool {
    vulkan_device().is_some()
}

/// Get or initialize the global wgpu device singleton. Returns None
/// on systems with no compatible adapter.
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    if BACKEND_PREF.load(Ordering::SeqCst) == PREF_VULKAN {
        vulkan_device()
    } else {
        default_device()
    }
}

/// Adapter name for calibration cache keys.
pub fn adapter_name() -> Option<String> {
    wgpu_device().map(|d| d.name.clone())
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
