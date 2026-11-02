//! Hardware guardian: real process-local measurements, budgets, thresholds
//! with hysteresis, EMA CPU smoothing, diagnostics and the advisory response
//! ladder.
//!
//! Measurement honesty (Windows; raw `windows-sys` syscalls — no measuring
//! library in between; see `SysMetrics`):
//! * Memory = the process **working set** of *this* process via
//!   `GetProcessMemoryInfo` (`PROCESS_MEMORY_COUNTERS::WorkingSetSize`), not
//!   total system RAM. One syscall per poll, zero heap allocations.
//! * CPU = per-process CPU time via `GetProcessTimes` delta over the real
//!   poll window divided by the window's wall time → "percent of one core"
//!   (the process-local member of the same API family as `GetSystemTimes`).
//!   The **first** sample only opens the window and yields `None` — nothing
//!   is fabricated; the governor EMA smooths across the opening.
//! * A failed / not-yet-open measurement is reported as `None` and the
//!   diagnostic flags turn off; the governor *holds* the affected state
//!   channel and never escalates from missing data.
//! * GPU is **not** measured: Windows has no reliable, cheap per-process GPU
//!   timing that a plain crate can read; visual-detail advice therefore
//!   reacts only to mem/CPU pressure (see crate docs).
//!
//! `SysMetrics::system_cpu_percent()` additionally exposes `GetSystemTimes`-
//! derived **system-wide** load. HUD context only — it never feeds budgets
//! (the budget stays app-only by construction).

use serde::{Deserialize, Serialize};

use crate::units::HardwareMemory;

#[cfg(windows)]
use std::sync::Mutex;
#[cfg(windows)]
use std::time::Instant;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{FILETIME, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes, GetSystemTimes};

/// Anything that can produce real measurements. Implemented by the raw
/// Windows monitor ([`SysMetrics`], production) and by synthetic mocks
/// (tests, no hardware).
pub trait MeasureSource {
    /// Process working-set memory of our process, in bytes.
    ///
    /// Legacy non-optional surface: implementations that cannot fail return
    /// the real value; implementations whose OS call failed return `0`
    /// honestly documented as "no number", never a fabricated usable value.
    /// New code (and the `Governor` itself) uses
    /// [`MeasureSource::try_process_memory_bytes`].
    fn process_memory_bytes(&self) -> HardwareMemory;
    /// Instantaneous-ish per-process CPU usage estimate in percent of a
    /// single core. Raw; the governor applies EMA smoothing itself.
    /// On implementations that cannot fail, `0.0` means "no measurement
    /// yet"; see [`MeasureSource::try_process_cpu_percent`].
    fn process_cpu_percent(&self) -> f32;

    /// Honest-availability surface. `None` = the OS refused / the
    /// measurement window is not open yet. The governor holds the affected
    /// state channel (never escalates from missing data) and reports
    /// `GovernorDiagnostic::{mem_ok, cpu_ok} = false`. The default simply
    /// forwards to the legacy methods, so synthetic mocks work unchanged.
    fn try_process_memory_bytes(&self) -> Option<HardwareMemory> {
        Some(self.process_memory_bytes())
    }
    /// Honest-availability CPU twin of [`MeasureSource::try_process_memory_bytes`].
    fn try_process_cpu_percent(&self) -> Option<f32> {
        Some(self.process_cpu_percent())
    }
}

impl dyn MeasureSource {
    /// Production measurement source: raw Windows monitoring of *this*
    /// process only (OS pseudo-handle from `GetCurrentProcess()`), never
    /// system-wide totals. Callable literally as `<dyn MeasureSource>::real()`.
    /// Synthetic mocks (see `tests.rs`) remain the sources used in tests —
    /// this exists for the coordinator wiring.
    pub fn real() -> Box<dyn MeasureSource + Send + 'static> {
        Box::new(SysMetrics::new())
    }
}

/// Rolling CPU-measurement window (per sample: process kernel+user time
/// and the wall instant at which it was taken).
#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
struct CpuWindow {
    primed: bool,
    /// kernel + user time of our process, FILETIME 100 ns units.
    proc_time_100ns: u64,
    wall: Instant,
}

/// Same machine for system-wide load (`GetSystemTimes`), HUD-only.
#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
struct SysWindow {
    primed: bool,
    idle_100ns: u64,
    total_100ns: u64,
    wall: Instant,
}

/// Production monitor using raw Windows calls against *our own* process —
/// two syscalls per poll, no heap allocations, no measuring library.
///
/// Targeting evidence: the handle comes from `GetCurrentProcess()` (the OS
/// pseudo-handle of *this* process). `GetProcessMemoryInfo` / `GetProcessTimes`
/// are therefore app-only by construction, not by convention. System-wide
/// totals are never fed into the budget; see `system_cpu_percent` (HUD).
///
/// Cheapness: `GetProcessMemoryInfo` and `GetProcessTimes` are single
/// syscalls each (< ~5 µs on practice hardware) — no 1 Hz caching layer is
/// needed; every poll carries a *fresh* measurement.
#[cfg(windows)]
pub struct SysMetrics {
    proc: HANDLE,
    cpu_window: Mutex<CpuWindow>,
    sys_window: Mutex<SysWindow>,
    /// 1 Hz cache for `total_physical_ram` (machine-constant between RAM
    /// sticks changing — there is no reason to re-syscall it every poll).
    ram_cache: Mutex<Option<(Instant, u64)>>,
}

// Send safety: `proc` is the OS *pseudo* handle from `GetCurrentProcess()`
// (a process-global constant, `HANDLE(-1)` semantics); the interior state is
// `Mutex`-protected raw counters. No thread crosses any of them by aliasing.
#[cfg(windows)]
unsafe impl Send for SysMetrics {}

#[cfg(not(windows))]
pub struct SysMetrics;

#[cfg(windows)]
impl SysMetrics {
    pub fn new() -> Self {
        Self {
            proc: unsafe { GetCurrentProcess() },
            cpu_window: Mutex::new(CpuWindow {
                primed: false,
                proc_time_100ns: 0,
                wall: Instant::now(),
            }),
            sys_window: Mutex::new(SysWindow {
                primed: false,
                idle_100ns: 0,
                total_100ns: 0,
                wall: Instant::now(),
            }),
            ram_cache: Mutex::new(None),
        }
    }

    /// System-wide CPU load in percent of *all* cores (`GetSystemTimes`
    /// delta over the call window: `1 - idleΔ/(kernelΔ+userΔ)`) → 0..=100.
    /// `None` until a first sample opened the window, window < 1 ms, the OS
    /// refused, or the total-time delta was zero. **HUD context only** —
    /// this never participates in budgets or the advisory ladder (the
    /// budget is app-only; escalation works from the per-process channel).
    pub fn system_cpu_percent(&self) -> Option<f32> {
        unsafe {
            let (mut idle, mut kernel, mut user) = (
                std::mem::zeroed::<FILETIME>(),
                std::mem::zeroed::<FILETIME>(),
                std::mem::zeroed::<FILETIME>(),
            );
            if GetSystemTimes(&mut idle, &mut kernel, &mut user) == 0 {
                return None; // OS refused — honest unavailability.
            }
            let idle_t = filetime_100ns(idle);
            let total_t = filetime_100ns(kernel).saturating_add(filetime_100ns(user));
            let now = Instant::now();
            let mut w = self.sys_window.lock().expect("SysMetrics poisoned");
            if !w.primed {
                *w = SysWindow { primed: true, idle_100ns: idle_t, total_100ns: total_t, wall: now };
                return None; // first call only opens the window.
            }
            let dt = now.duration_since(w.wall).as_nanos() as f32;
            let d_idle = (idle_t.saturating_sub(w.idle_100ns)) as f64 * 100.0;
            let d_total = (total_t.saturating_sub(w.total_100ns)) as f64 * 100.0;
            *w = SysWindow { primed: true, idle_100ns: idle_t, total_100ns: total_t, wall: now };
            if dt < 1_000.0 || d_total <= 0.0 {
                return None; // window too small / degenerate — no fabrication.
            }
            Some((1.0 - d_idle / d_total).clamp(0.0, 1.0) as f32 * 100.0)
        }
    }

    /// Total physical RAM of the machine in bytes via `GlobalMemoryStatusEx`
    /// (`MEMORYSTATUSEX::ullTotalPhys`). `None` when the OS refused — never
    /// a fabricated number. **Cached at 1 Hz**: total RAM cannot usefully
    /// change faster; a cache miss costs one cheap syscall (which itself
    /// only happens on the first call after a cache expiry). Used to derive
    /// the RAM-proportional default budget (`GovernorConfig::default_for_ram`).
    /// HUD/context only: the budget threshold logic itself stays app-only
    /// against the configured budget constant.
    pub fn total_physical_ram(&self) -> Option<u64> {
        {
            let cache = self.ram_cache.lock().expect("SysMetrics poisoned");
            if let Some((at, val)) = *cache {
                if at.elapsed().as_secs_f64() < 1.0 {
                    return Some(val);
                }
            }
        }
        unsafe {
            let mut stat: MEMORYSTATUSEX = std::mem::zeroed();
            stat.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut stat) == 0 {
                return None; // OS refused — honest unavailability, cache untouched.
            }
            let val = stat.ullTotalPhys;
            *self.ram_cache.lock().expect("SysMetrics poisoned") =
                Some((Instant::now(), val));
            Some(val)
        }
    }
}

#[cfg(not(windows))]
impl SysMetrics {
    /// Non-Windows stub: every measurement is honestly unavailable.
    pub fn new() -> Self {
        Self
    }
    /// Non-Windows stub: honestly unavailable.
    pub fn total_physical_ram(&self) -> Option<u64> {
        None
    }
}

impl Default for SysMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// FILETIME (two u32 halves) → count of 100 ns units.
#[cfg(windows)]
fn filetime_100ns(ft: FILETIME) -> u64 {
    (ft.dwHighDateTime as u64) << 32 | ft.dwLowDateTime as u64
}

#[cfg(windows)]
impl MeasureSource for SysMetrics {
    /// Legacy path: 0 bytes == "no measurement" (never a fabricated value).
    fn process_memory_bytes(&self) -> HardwareMemory {
        self.try_process_memory_bytes().unwrap_or_default()
    }
    /// Legacy path: 0.0 == "no measurement yet" (never a fabricated value).
    fn process_cpu_percent(&self) -> f32 {
        self.try_process_cpu_percent().unwrap_or(0.0)
    }

    fn try_process_memory_bytes(&self) -> Option<HardwareMemory> {
        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            let ok = GetProcessMemoryInfo(self.proc, &mut pmc, cb) != 0;
            if !ok {
                return None; // OS refused — honest unavailability, no invented value.
            }
            Some(HardwareMemory::from_bytes(pmc.WorkingSetSize as u64))
        }
    }

    fn try_process_cpu_percent(&self) -> Option<f32> {
        unsafe {
            let (mut ft_create, mut ft_exit, mut ft_kernel, mut ft_user) = (
                std::mem::zeroed::<FILETIME>(),
                std::mem::zeroed::<FILETIME>(),
                std::mem::zeroed::<FILETIME>(),
                std::mem::zeroed::<FILETIME>(),
            );
            if GetProcessTimes(
                self.proc,
                &mut ft_create,
                &mut ft_exit,
                &mut ft_kernel,
                &mut ft_user,
            ) == 0
            {
                return None; // OS refused — honest unavailability.
            }
            let t = filetime_100ns(ft_kernel).saturating_add(filetime_100ns(ft_user));
            let now = Instant::now();
            let mut w = self.cpu_window.lock().expect("SysMetrics poisoned");
            if !w.primed {
                *w = CpuWindow { primed: true, proc_time_100ns: t, wall: now };
                return None; // first call only opens the measurement window.
            }
            let dt = now.duration_since(w.wall).as_nanos() as f32;
            let dp = (t.saturating_sub(w.proc_time_100ns)) as f32 * 100.0; // 100 ns → ns
            *w = CpuWindow { primed: true, proc_time_100ns: t, wall: now };
            if dt < 1_000.0 {
                return None; // sub-millisecond window: too noisy to be honest.
            }
            // Percent of one core: process CPU time over wall time × 100.
            Some((dp / dt * 100.0).max(0.0))
        }
    }
}

#[cfg(not(windows))]
impl MeasureSource for SysMetrics {
    fn process_memory_bytes(&self) -> HardwareMemory {
        HardwareMemory::default()
    }
    fn process_cpu_percent(&self) -> f32 {
        0.0
    }
    fn try_process_memory_bytes(&self) -> Option<HardwareMemory> {
        None
    }
    fn try_process_cpu_percent(&self) -> Option<f32> {
        None
    }
}

/// Serde default for reading *older* payloads: diagnostics serialized
/// before the ok-flags existed always came from live polls (or mocks), so
/// "assumed ok" is the least-surprising backward-compat choice. Live data
/// always sets the flags honestly.
fn serde_true() -> bool {
    true
}

/// Traffic-light health of the hardware budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub enum GovernorState {
    #[default]
    Green,
    Warn,
    Critical,
}

/// Advisory actions. The governor **advises only**: something else (the
/// coordinator wiring) applies or declines them. The governor never kills
/// organisms, never deletes saves, never writes files to free memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AdvisoryAction {
    #[default]
    None,
    /// Warn-state banner (UI-level only).
    SafeWarn,
    /// Decrease 2D/3D visual fidelity (render loop budget).
    ReduceVisualDetail,
    /// Decrease neural-observatory fidelity / sampling rate.
    ReduceNeuralVisual,
    /// Throttle/queue controllers into smaller batches.
    BatchControllers,
    /// Lower target `steps-per-second` for subsystems.
    ReduceUpdateHz,
    /// Wire-compat only: retained for serde roundtrips of older payloads,
    /// but the escalation ladder no longer emits it (a population action
    /// must never appear as an escalation step). Population budgeting is
    /// expressed advisory-only through `TickBudgets::soft_population_cap`.
    /// Never lethal to living organisms either way.
    PopulationApprox,
    /// Last resort: coordinator may pause the sim loop (menus, etc.).
    SafePause,
}

/// Advisory escalation ladder for a state, mild -> severe. The *first* entry
/// is the currently "recommended" action; all entries are listed so the
/// coordinator can show the full set.
///
/// Safety properties (enforced by
/// `tests::ladder_never_contains_lethal_or_population_actions`):
/// * No entry is lethal — an organism can not be killed by any of them.
/// * No entry is a *population* action; birth budgeting lives in
///   `TickBudgets::soft_population_cap` (advisory), kept out of the
///   escalation ladder.
/// * Visualization degrades **before** simulation is touched: both ladders
///   start with [`AdvisoryAction::ReduceVisualDetail`]; the Warn ladder ends
///   there and only the Critical ladder proceeds to scheduler actions.
pub fn advisory_ladder(state: GovernorState) -> &'static [AdvisoryAction] {
    const WARN: [AdvisoryAction; 2] = [AdvisoryAction::SafeWarn, AdvisoryAction::ReduceVisualDetail];
    const CRIT: [AdvisoryAction; 5] = [
        AdvisoryAction::ReduceVisualDetail,
        AdvisoryAction::ReduceNeuralVisual,
        AdvisoryAction::BatchControllers,
        AdvisoryAction::ReduceUpdateHz,
        AdvisoryAction::SafePause,
    ];
    match state {
        GovernorState::Green => &[],
        GovernorState::Warn => &WARN,
        GovernorState::Critical => &CRIT,
    }
}

/// One poll's diagnostic snapshot. Plain numbers (UI-friendly); budgets are
/// mirrored here so the UI never has to keep the config around.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct GovernorDiagnostic {
    /// Process working-set bytes used by the app. When [`self.mem_ok`] is
    /// `false` this is the **last known good** measurement (0 = none yet).
    pub mem_used: u64,
    /// Memory budget in bytes.
    pub mem_budget: u64,
    /// EMA-smoothed CPU usage, percent of one core. When [`self.cpu_ok`] is
    /// `false` this is the last known EMA (0.0 = never measured).
    pub cpu_used: f32,
    /// CPU budget (same units).
    pub cpu_budget: f32,
    /// Combined (worst-of-both) state.
    pub state: GovernorState,
    /// Milliseconds since the last *de-escalation* (0 = never decayed; state
    /// has been at least as bad as now since startup).
    pub decayed_after_ms: u64,
    /// `true` when THIS poll carried a real OS memory measurement; `false`
    /// = measurement unavailable this poll (held state, no escalation).
    #[serde(default = "serde_true")]
    pub mem_ok: bool,
    /// `true` when THIS poll carried a real CPU measurement (a fresh
    /// `GetProcessTimes` delta); `false` = window not open / OS refused.
    #[serde(default = "serde_true")]
    pub cpu_ok: bool,
}

/// Hardware budget + threshold geometry (fractions are of the budget).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub memory_budget_bytes: u64,
    pub cpu_budget_percent: f32,
    /// Warn threshold as fraction of budget (e.g. 0.85).
    pub warn_threshold: f32,
    /// Critical threshold as fraction of budget (e.g. 0.95).
    pub critical_threshold: f32,
    /// Hysteresis band as fraction of budget; de-escalation requires falling
    /// to `threshold - hysteresis` or below.
    pub hysteresis: f32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            // 1.5 GiB app footprint; leaves the vast majority of a 16 GB
            // box to other processes. CONTRACT: do not change the value.
            memory_budget_bytes: 1_610_612_736,
            cpu_budget_percent: 30.0,
            warn_threshold: 0.85,
            critical_threshold: 0.95,
            hysteresis: 0.05,
        }
    }
}

impl Budget {
    pub(crate) fn clamped(mut self) -> Self {
        self.warn_threshold = self.warn_threshold.clamp(0.05, 0.99);
        self.critical_threshold = self.critical_threshold.clamp(self.warn_threshold + 0.01, 1.0);
        self.hysteresis = self.hysteresis.clamp(0.0, self.warn_threshold * 0.5);
        self.cpu_budget_percent = self.cpu_budget_percent.clamp(1.0, 800.0);
        self.memory_budget_bytes = self.memory_budget_bytes.max(1);
        self
    }

    fn warn_bytes(&self) -> f32 {
        self.warn_threshold * self.memory_budget_bytes as f32
    }
    fn crit_bytes(&self) -> f32 {
        self.critical_threshold * self.memory_budget_bytes as f32
    }
    fn hyst_bytes(&self) -> f32 {
        self.hysteresis * self.memory_budget_bytes as f32
    }

    /// Pure state-machine step with hysteresis: escalation is immediate on
    /// threshold crossing; de-escalation requires dropping below
    /// `(threshold - hysteresis)`. Prevents flapping at the boundary.
    pub(crate) fn advance_mem_state(&self, current: GovernorState, used: HardwareMemory) -> GovernorState {
        let r = used.ratio_over(HardwareMemory::from_bytes(self.memory_budget_bytes));
        match current {
            GovernorState::Green => {
                if r >= self.critical_threshold {
                    GovernorState::Critical
                } else if r >= self.warn_threshold {
                    GovernorState::Warn
                } else {
                    GovernorState::Green
                }
            }
            GovernorState::Warn => {
                if r >= self.critical_threshold {
                    GovernorState::Critical
                } else if used.bytes() as f32 <= self.warn_bytes() - self.hyst_bytes() {
                    GovernorState::Green
                } else {
                    GovernorState::Warn
                }
            }
            GovernorState::Critical => {
                let used = used.bytes() as f32;
                if used <= self.warn_bytes() - self.hyst_bytes() {
                    GovernorState::Green
                } else if used <= self.crit_bytes() - self.hyst_bytes() {
                    GovernorState::Warn
                } else {
                    GovernorState::Critical
                }
            }
        }
    }

    /// Same machine for CPU, driven by the EMA value (percent).
    pub(crate) fn advance_cpu_state(&self, current: GovernorState, ema_pct: f32) -> GovernorState {
        let warn = self.warn_threshold * self.cpu_budget_percent;
        let crit = self.critical_threshold * self.cpu_budget_percent;
        match current {
            GovernorState::Green => {
                if ema_pct >= crit {
                    GovernorState::Critical
                } else if ema_pct >= warn {
                    GovernorState::Warn
                } else {
                    GovernorState::Green
                }
            }
            GovernorState::Warn => {
                if ema_pct >= crit {
                    GovernorState::Critical
                } else if ema_pct <= warn - self.hysteresis * self.cpu_budget_percent {
                    GovernorState::Green
                } else {
                    GovernorState::Warn
                }
            }
            GovernorState::Critical => {
                let hyst = self.hysteresis * self.cpu_budget_percent;
                if ema_pct <= warn - hyst {
                    GovernorState::Green
                } else if ema_pct <= crit - hyst {
                    GovernorState::Warn
                } else {
                    GovernorState::Critical
                }
            }
        }
    }
}
