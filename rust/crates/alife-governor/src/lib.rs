//! # alife-governor — God Commander resource governor
//!
//! Two subsystems, one crate:
//!
//! 1. **Hardware guardian** ([`hardware`] + [`Governor::poll`]): real,
//!    process-local measurements (sysinfo working-set memory, EMA-smoothed
//!    process CPU), budget thresholds with hysteresis, diagnostics and an
//!    *advisory* escalation ladder.
//! 2. **Simulation scheduler** ([`policy`] + [`Governor::plan`]):
//!    deterministic per-tick allocation plans over three strictly separated
//!    unit spaces, plus (opt-in) computational-selection CU starvation
//!    review and a bounded event ring for the UI observatory.
//!
//! ## Safety contract
//! The governor **advises only**. It never kills organisms, never deletes
//! saves, never writes files to "free" memory, and never mutates
//! `alife-core` state. The coordinator wiring (alife-ui / app.rs) decides
//! which advisories to apply.
//!
//! ## Unit separation (compile-level)
//! [`units::HardwareMemory`] (bytes, real), [`units::ComputeUnits`]
//! (fictional CU), [`units::BioEnergy`] (core's f32 energy) are distinct
//! newtypes. Cross-space arithmetic is a compile error:
//!
//! ```compile_fail
//! use alife_governor::units::{ComputeUnits, BioEnergy};
//! let cu = ComputeUnits::new(1.0);
//! let be = BioEnergy::new(1.0);
//! let _sum = cu + be; // ERROR: mismatched types — units must never mix
//! ```
//!
//! ## Honest measurement limitations
//! * **GPU is not measured.** Windows offers no reliable cheap per-process
//!   GPU timing to a plain Rust crate; visual-detail advisories therefore
//!   trigger from mem/CPU pressure only.
//! * Memory is our own process's **working set** — other apps are invisible
//!   to the budget, by design (app-only).
//! * The first `poll` after startup reads CPU as ~0 (OS needs two refresh
//!   samples); the EMA smooths this.
//! * CPU percent is "percent of one core" as reported by sysinfo; on a
//!   10-core machine 100.0 means "one core fully busy".

pub mod events;
pub mod governor;
pub mod hardware;
pub mod policy;
pub mod units;

pub use events::{EventKind, GovernorEvent, EVENT_RING_MAX};
pub use governor::{Governor, GovernorConfig};
pub use hardware::{
    advisory_ladder, AdvisoryAction, Budget, GovernorDiagnostic, GovernorState, MeasureSource,
    SysMetrics,
};
pub use policy::{
    compute_starvation_review, AllocationPlan, ComputeReview, ControllerCost, Policy, PolicyInput,
    SubsystemShares, TickBudgets,
};
pub use units::{BioEnergy, ComputeUnits, HardwareMemory};

#[cfg(test)]
mod tests;
