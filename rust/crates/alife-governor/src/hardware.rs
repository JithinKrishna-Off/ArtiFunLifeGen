//! Hardware guardian: real process-local measurements, budgets, thresholds
//! with hysteresis, EMA CPU smoothing, diagnostics and the advisory response
//! ladder.
//!
//! Measurement honesty:
//! * Memory = the process **working set** of *this* process (`sysinfo
//!   Process::memory()` on Windows), not total system RAM.
//! * CPU = per-process usage reported by sysinfo; the first reading after
//!   startup is 0 by OS convention (needs two refreshes) — the governor EMA
//!   smooths that away.
//! * GPU is **not** measured: Windows has no reliable, cheap per-process GPU
//!   timing that a plain crate can read; visual-detail advice therefore
//!   reacts only to mem/CPU pressure (see crate docs).

use serde::{Deserialize, Serialize};
use sysinfo::{get_current_pid, Pid, ProcessRefreshKind, System};
use std::sync::Mutex;

use crate::units::HardwareMemory;

/// Anything that can produce real measurements. Implemented by the sysinfo
/// monitor (production) and by synthetic mocks (tests, no hardware).
pub trait MeasureSource {
    /// Process working-set memory of our process, in bytes.
    fn process_memory_bytes(&self) -> HardwareMemory;
    /// Instantaneous-ish per-process CPU usage estimate in percent of a
    /// single core. Raw; the governor applies EMA smoothing itself.
    fn process_cpu_percent(&self) -> f32;
}

/// Production monitor on top of sysinfo. Refreshes the *self* process only
/// per call (cheap on the hot path); CPU percent is valid once sysinfo has
/// observed two refreshes, which the poll loop provides.
pub struct SysMetrics {
    sys: Mutex<System>,
    pid: Pid,
}

impl SysMetrics {
    pub fn new() -> Self {
        let pid = get_current_pid().unwrap_or_else(|_| Pid::from_u32(0));
        let mut sys = System::new();
        // Prime once so `process(pid)` exists immediately.
        let _ = sys.refresh_process(pid);
        Self { sys: Mutex::new(sys), pid }
    }
}

impl Default for SysMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MeasureSource for SysMetrics {
    fn process_memory_bytes(&self) -> HardwareMemory {
        let mut sys = self.sys.lock().expect("SysMetrics poisoned");
        let _ = sys.refresh_process_specifics(self.pid, ProcessRefreshKind::new().with_memory().with_cpu());
        sys.process(self.pid)
            .map(|p| HardwareMemory::from_bytes(p.memory()))
            .unwrap_or_default()
    }

    fn process_cpu_percent(&self) -> f32 {
        // Refresh (CPU delta vs previous refresh) then read. The EMA in
        // `Governor` smooths the first-read 0.0 startup artifact.
        let mut sys = self.sys.lock().expect("SysMetrics poisoned");
        let _ = sys.refresh_process_specifics(self.pid, ProcessRefreshKind::new().with_memory().with_cpu());
        sys.process(self.pid).map(|p| p.cpu_usage()).unwrap_or(0.0)
    }
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
    /// Stop *new* births (approx population budget); never lethal to the
    /// living.
    PopulationApprox,
    /// Last resort: coordinator may pause the sim loop (menus, etc.).
    SafePause,
}

/// Advisory escalation ladder for a state, mild -> severe. The *first* entry
/// is the currently "recommended" action; all entries are listed so the
/// coordinator can show the full set.
pub fn advisory_ladder(state: GovernorState) -> &'static [AdvisoryAction] {
    const WARN: [AdvisoryAction; 2] = [AdvisoryAction::SafeWarn, AdvisoryAction::ReduceVisualDetail];
    const CRIT: [AdvisoryAction; 6] = [
        AdvisoryAction::ReduceVisualDetail,
        AdvisoryAction::ReduceNeuralVisual,
        AdvisoryAction::BatchControllers,
        AdvisoryAction::ReduceUpdateHz,
        AdvisoryAction::PopulationApprox,
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
    /// Process working-set bytes used by the app.
    pub mem_used: u64,
    /// Memory budget in bytes.
    pub mem_budget: u64,
    /// EMA-smoothed CPU usage, percent of one core.
    pub cpu_used: f32,
    /// CPU budget (same units).
    pub cpu_budget: f32,
    /// Combined (worst-of-both) state.
    pub state: GovernorState,
    /// Milliseconds since the last *de-escalation* (0 = never decayed; state
    /// has been at least as bad as now since startup).
    pub decayed_after_ms: u64,
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
            // box to other processes.
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
