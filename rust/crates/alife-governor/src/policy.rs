//! Simulation scheduler: allocation policies, plans, tick budgets and the
//! compute (CU) starvation review.
//!
//! Unit discipline (see [`crate::units`]): policies speak fractions (plain
//! `f32` in `[0,1]`) and [`ComputeUnits`]; [`BioEnergy`] appears only as
//! reported metadata in [`PolicyInput`]. Real hardware numbers never appear
//! here except through the blunt knob-scaling produced by the hardware
//! guardian (see `Governor::plan`).

use serde::{Deserialize, Serialize};

use crate::hardware::GovernorState;
use crate::units::{BioEnergy, ComputeUnits};

/// Allocation policy. `ComputationalSelection` is the **opt-in experiment**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Policy {
    /// Hand-tuned middle ground; rendering/observatory kept healthy.
    #[default]
    Balanced,
    /// Queen controller gets the lion's share of tick attention + CU.
    QueenPriority,
    /// Colony/worker metabolism prioritized.
    ColonyPriority,
    /// Flat equal shares.
    EqualShare,
    /// Favor birth/food loops; do not count new births against the CU
    /// advisory budget (governor only; core decides).
    MaxPopulation,
    /// Experiment, opt-in: scarce CU pool; per-controller starvation review
    /// ("expensive controllers spend more CU, simpler ones profit").
    /// Never lethal: review only *advises* deferral and records events.
    ComputationalSelection,
}

/// Per-tick attention/framerate fractions per subsystem. Fractions
/// independently sum to <= 1.0 (checked, then normalized defensively).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubsystemShares {
    pub queen: f32,
    pub colony: f32,
    pub ecology: f32,
    pub weather: f32,
    pub neural_observatory: f32,
    pub rendering: f32,
}

impl SubsystemShares {
    pub const fn parts(&self) -> [(&'static str, f32); 6] {
        [
            ("queen", self.queen),
            ("colony", self.colony),
            ("ecology", self.ecology),
            ("weather", self.weather),
            ("neural_observatory", self.neural_observatory),
            ("rendering", self.rendering),
        ]
    }

    pub fn sum(&self) -> f32 {
        self.parts().iter().map(|(_, f)| f).fold(0.0, |a, b| a + b)
    }

    /// Deterministic normalization: if the raw sum drifted above 1.0 (f32
    /// rounding or a hand-tuned overshoot), scale every share by `1/sum`.
    /// Below 1.0 is allowed (slack is intentional, e.g. MaxPopulation).
    pub(crate) fn normalized(mut self) -> Self {
        let s = self.sum();
        if s > 1.0 {
            for f in [&mut self.queen, &mut self.colony, &mut self.ecology, &mut self.weather, &mut self.neural_observatory, &mut self.rendering] {
                *f /= s;
            }
        }
        self
    }
}

/// Input snapshot the coordinator feeds per plan request. All metadata:
/// nothing here mutates core state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyInput {
    pub tick: u64,
    pub population: u32,
    /// Policy for this plan (coordinator-owned; can change per tick).
    pub policy: Policy,
    /// Is the queen organism alive this tick? Additive, serde-default for
    /// backward compatibility of older serialized inputs.
    #[serde(default)]
    pub queen_alive: bool,
    /// How much work (nanoseconds) the queen controller consumed in the
    /// last tick, as reported by the coordinator. Pure telemetry used in
    /// scheduling-event reasons so the observatory can see queen load;
    /// it never alters budgets by itself. Additive, serde-default.
    #[serde(default)]
    pub queen_compute_ns: u128,
    /// Colony-average bio energy (metadata for heuristics; separate unit
    /// space from CU and hardware).
    pub avg_bio_energy: BioEnergy,
    /// Current CU demand measured by the coordinator's accounting.
    pub demanded_compute: ComputeUnits,
}

/// Tunable per-subsystem knobs the governor produces; the coordinator
/// applies them to its loop. All bounded; serialized for save v3 later.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TickBudgets {
    /// Target sim update Hz (bounded 1..=120).
    pub update_hz_simulation: f32,
    /// Target neural-observatory update Hz (bounded 0.5..=60).
    pub update_hz_neural_observatory: f32,
    /// Target rendering Hz (bounded 1..=240; UI fps).
    pub update_hz_rendering: f32,
    /// Controller batch size for queued/advised work (bounded 1..=4096).
    pub controller_batch_size: u32,
    /// Advisory soft population cap (bounded 0..=100_000). Coordinator
    /// decides how to honor it (block new births etc.), never lethal.
    pub soft_population_cap: u32,
}

impl Default for TickBudgets {
    fn default() -> Self {
        Self {
            update_hz_simulation: 30.0,
            update_hz_neural_observatory: 10.0,
            update_hz_rendering: 60.0,
            controller_batch_size: 256,
            soft_population_cap: 4_000,
        }
    }
}

/// Per-frame timing report the coordinator feeds via
/// `Governor::report_timings`. Unit: **milliseconds** (documented once
/// here, used consistently by every getter/EMA in this crate). Cheap to
/// build; the governor keeps a rolling EMA per field. All zero is the
/// "never reported / nothing measured" sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SubsystemTimings {
    pub sim_ms: f32,
    pub render_ms: f32,
    pub neural_ms: f32,
    pub ecology_ms: f32,
}

/// Rebalance thresholds and factors for [`TickBudgets`] — one documented
/// place for the timing-rebalance formulas operated by
/// `Governor::plan`:
///
/// * `f_i = timing_ema_i / Σ(timing_ema)` — a subsystem's share of measured
///   frame work, each in `[0, 1]` (denominator strictly positive; no NaN).
/// * **render heavy** (`f_render >= 0.50`): rendering drops to exactly
///   30 Hz (`update_hz_rendering = 30.0`), clamped to `1..=240`.
/// * **neural heavy** (`f_neural >= 0.45`): neural-observatory Hz scaled by
///   `0.7` (neural heavy → lower neural Hz), clamped to `0.5..=60`.
/// * **sim heavy** (`f_sim >= 0.60`): *no knob change* — sim cadence is a
///   contracts-grade constant; the governor escalates an advisory
///   (`ReduceUpdateHz`) for the coordinator to act on instead.
/// * ecology timing joins only the Σ denominator; no dedicated knob exists
///   for it today (honestly documented gap).
/// * `SubsystemShares` (the policy split) is **never** touched by timings —
///   the rebalance respects the policy split by construction.
pub(crate) mod timing_rebalance {
    pub const RENDER_HEAVY_FRACTION: f32 = 0.50;
    pub const NEURAL_HEAVY_FRACTION: f32 = 0.45;
    pub const SIM_HEAVY_FRACTION: f32 = 0.60;
    pub const RENDER_HEAVY_HZ: f32 = 30.0;
    pub const NEURAL_HEAVY_SCALE: f32 = 0.7;
}

impl TickBudgets {
    /// Timing-derived rebalance: deterministic from the input fractions,
    /// every produced field passes [`TickBudgets::clamped`] (no NaN: all
    /// arithmetic is over non-negative finite values with a strictly
    /// positive divisor).
    pub(crate) fn rebalanced_by_timings(mut self, f: &TimingFractions) -> Self {
        use timing_rebalance::*;
        if f.render >= RENDER_HEAVY_FRACTION {
            self.update_hz_rendering = RENDER_HEAVY_HZ;
        }
        if f.neural >= NEURAL_HEAVY_FRACTION {
            self.update_hz_neural_observatory *= NEURAL_HEAVY_SCALE;
        }
        // f.sim >= SIM_HEAVY_FRACTION: intentionally no change here (see
        // module docs); the Governor escalates an advisory event instead.
        self.clamped()
    }
}

/// Normalized fractions of measured frame work per subsystem, computed
/// from the timing EMAs. All in `[0, 1]` and finite; `None` before the
/// first non-empty timing report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimingFractions {
    pub sim: f32,
    pub render: f32,
    pub neural: f32,
    pub ecology: f32,
}

impl TimingFractions {
    pub(crate) fn from_ema(t: &TimingEma) -> Option<Self> {
        let total = t.sim + t.render + t.neural + t.ecology;
        if total <= 0.0 {
            return None;
        }
        let norm = |v: f64| (v / total) as f32;
        Some(Self {
            sim: norm(t.sim).clamp(0.0, 1.0),
            render: norm(t.render).clamp(0.0, 1.0),
            neural: norm(t.neural).clamp(0.0, 1.0),
            ecology: norm(t.ecology).clamp(0.0, 1.0),
        })
    }
}

/// Rolling EMA of per-subsystem frame timings (milliseconds, f64 internal
/// precision). Warm-up is documented: starting from all zeros, the EMA
/// approaches the true value geometrically — after `k` reports the residual
/// is `(1 - α)^k` of the initial gap (α = `GovernorConfig::timing_ema_alpha`,
/// default 0.25). Inputs are sanitized: non-finite or negative values are
/// treated as 0.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct TimingEma {
    pub sim: f64,
    pub render: f64,
    pub neural: f64,
    pub ecology: f64,
}

impl TimingEma {
    pub(crate) fn update(&mut self, t: &SubsystemTimings, alpha: f32) {
        let a = alpha as f64;
        let sanitize = |v: f32| -> f64 {
            let v = v as f64;
            if v.is_finite() && v >= 0.0 { v } else { 0.0 }
        };
        let blend = |e: f64, raw: f64| a * raw + (1.0 - a) * e;
        self.sim = blend(self.sim, sanitize(t.sim_ms));
        self.render = blend(self.render, sanitize(t.render_ms));
        self.neural = blend(self.neural, sanitize(t.neural_ms));
        self.ecology = blend(self.ecology, sanitize(t.ecology_ms));
    }
}

impl TickBudgets {
    pub(crate) fn clamped(mut self) -> Self {
        self.update_hz_simulation = self.update_hz_simulation.clamp(1.0, 120.0);
        self.update_hz_neural_observatory = self.update_hz_neural_observatory.clamp(0.5, 60.0);
        self.update_hz_rendering = self.update_hz_rendering.clamp(1.0, 240.0);
        self.controller_batch_size = self.controller_batch_size.clamp(1, 4_096);
        self.soft_population_cap = self.soft_population_cap.min(100_000);
        self
    }

    /// Hardware advice forwarded onto knobs. Green = pass-through, Warn =
    /// ~75%, Critical = ~50%; every field clamped by [`TickBudgets::clamped`].
    pub(crate) fn scaled_by_hardware(mut self, state: GovernorState) -> Self {
        let (hz_k, batch_k, cap_k) = match state {
            GovernorState::Green => (1.0, 1.0, 1.0),
            GovernorState::Warn => (0.75, 0.75, 1.0),
            GovernorState::Critical => (0.5, 0.5, 0.5),
        };
        self.update_hz_simulation *= hz_k;
        self.update_hz_neural_observatory *= hz_k;
        self.update_hz_rendering *= hz_k;
        self.controller_batch_size = (self.controller_batch_size as f32 * batch_k) as u32;
        self.soft_population_cap = (self.soft_population_cap as f32 * cap_k) as u32;
        self.clamped()
    }
}

/// Deterministic, queue-placement-only allocation plan. Pure data: the
/// coordinator applies it (or not); the governor never touches core.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocationPlan {
    pub tick: u64,
    pub policy: Policy,
    pub shares: SubsystemShares,
    /// CU budget the coordinator may spend this tick (its accounting
    /// decides; governor reviews overruns via `review_controllers`).
    pub compute_budget: ComputeUnits,
    /// Hardware-advice-derived tick knobs.
    pub tick_budgets: TickBudgets,
    /// Hardware state this plan was scaled from (serde-default for older
    /// payloads). Source for [`AllocationPlan::to_apply_hints`].
    #[serde(default)]
    pub hardware_state: GovernorState,
    /// True when hardware advice degraded the knobs vs the green baseline.
    pub hardware_scaled: bool,
    /// True when the timing EMA triggered a knob rebalance this tick.
    #[serde(default)]
    pub timing_rebalanced: bool,
}

/// Directly applicable advisory hints an app agent can consume without
/// touching any other governor API. Pure data — the agent applies or
/// declines every field.
///
/// # Documented mapping (level 0 = none → 3 = strongest)
///
/// | Plan state | `step_cap_hint` | `visual_degrade` | `pause_advisory` |
/// |---|---|---|---|
/// | Green | `None` (no restriction) | `0` (full fidelity) | `false` |
/// | Warn | `Some(2)` (`min(app_default, 2)` steps/frame) | `1` (reduce 2D/3D visual detail) | `false` |
/// | Critical | `Some(1)` (min single sim step/frame) | `3` (max: + neural observatory, batched controllers, lowered Hz) | `true` (last-resort advisory only; applying it is the coordinator's decision) |
///
/// # Safety contract
/// There is **no** field that can ever designate an organism (or any
/// population) for destruction — hints can only cap per-frame sim steps,
/// degrade visuals, or suggest a pause. Birth/population budgeting is
/// advisory via `TickBudgets::soft_population_cap` (never lethal).
/// Enforced by `tests::apply_hints_never_propose_organism_destroy` and
/// `tests::ladder_never_contains_lethal_or_population_actions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ApplyHints {
    /// Cap on simulation steps per rendered frame (`None` = no
    /// restriction). Apply as a cap (`min`), never a floor — the hint can
    /// only *lower* app-chosen work.
    pub step_cap_hint: Option<u64>,
    /// Visual degradation level 0..=3 (0 = none, 3 = strongest; see the
    /// table above).
    pub visual_degrade: u8,
    /// `true` only in Critical: the coordinator *may* pause the sim loop
    /// (menus etc.) as the last resort. Advisory only — simulation is never
    /// forced to stop by the governor.
    pub pause_advisory: bool,
}

impl AllocationPlan {
    /// Direct-consumption hints for app agents (see [`ApplyHints`] for the
    /// documented mapping). Pure function of the plan (deterministic):
    /// same plan → same hints.
    pub fn to_apply_hints(&self) -> ApplyHints {
        match self.hardware_state {
            GovernorState::Green => ApplyHints {
                step_cap_hint: None,
                visual_degrade: 0,
                pause_advisory: false,
            },
            GovernorState::Warn => ApplyHints {
                step_cap_hint: Some(crate::governor::STEP_CAP_WARN),
                visual_degrade: 1,
                pause_advisory: false,
            },
            GovernorState::Critical => ApplyHints {
                step_cap_hint: Some(crate::governor::STEP_CAP_CRITICAL),
                visual_degrade: 3,
                pause_advisory: true,
            },
        }
    }
}

impl Policy {
    /// Base shares (fractions of one tick's dispatch attention) + base CU
    /// budget in fictional CU (chosen ~200 CU ≈ a "generous" tick).
    pub(crate) fn shares_and_budget(self, cfg_compute_budget: f32) -> (SubsystemShares, ComputeUnits) {
        let shares = |q, c, e, w, n, r| SubsystemShares { queen: q, colony: c, ecology: e, weather: w, neural_observatory: n, rendering: r }.normalized();
        match self {
            Policy::Balanced => (shares(0.12, 0.26, 0.20, 0.10, 0.17, 0.15), ComputeUnits::new(cfg_compute_budget)),
            Policy::QueenPriority => (shares(0.40, 0.18, 0.12, 0.08, 0.12, 0.10), ComputeUnits::new(cfg_compute_budget)),
            Policy::ColonyPriority => (shares(0.10, 0.40, 0.20, 0.10, 0.10, 0.10), ComputeUnits::new(cfg_compute_budget)),
            Policy::EqualShare => (shares(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), ComputeUnits::new(cfg_compute_budget)),
            Policy::MaxPopulation => (shares(0.08, 0.32, 0.28, 0.10, 0.08, 0.09), ComputeUnits::new(cfg_compute_budget)),
            Policy::ComputationalSelection => (shares(0.10, 0.24, 0.18, 0.10, 0.14, 0.10), ComputeUnits::new(cfg_compute_budget * 0.6)),
        }
    }

    /// Advisory soft population cap per policy (before hardware scaling).
    pub(crate) fn population_cap(self, cfg_base: u32) -> u32 {
        match self {
            Policy::MaxPopulation => cfg_base.saturating_mul(2),
            Policy::ComputationalSelection => (cfg_base / 2).max(1),
            _ => cfg_base,
        }
    }

    /// Queen guarantee: only `QueenPriority` carries one — when the queen
    /// is alive, the queen's dispatch-attention share is never below this
    /// fraction (defensively enforced in `Governor::plan` even though the
    /// hand-tuned share already equals it). All other policies guarantee
    /// nothing about the queen (`None`).
    pub(crate) fn guaranteed_queen_share(self) -> Option<f32> {
        match self {
            Policy::QueenPriority => Some(0.40),
            _ => None,
        }
    }
}

/// One controller's CU cost for a tick, tagged by the coordinator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerCost {
    /// Stable controller id (coordinator-owned).
    pub id: u64,
    pub label: String,
    pub cost: ComputeUnits,
    /// True for the queen controller. Queen costs are **exempt** from
    /// computational selection: never advised for deferral and never
    /// counted against the ordinary-controller CU budget (which is the
    /// `0.6×`-shrunken budget under `ComputationalSelection`). Additive,
    /// serde-default so older serialized rows load as ordinary controllers.
    #[serde(default)]
    pub is_queen: bool,
}

/// The result of a compute-starvation review: controllers advised for
/// deferral this tick (never lethal; coordinator decides to actually
/// defer/queue them) plus the recorded events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeReview {
    /// Most expensive controllers whose deferral brings remaining demand
    /// under budget. Empty when the tick fits.
    pub advised_deferral: Vec<ControllerCost>,
}

/// Review CU demand against a plan's budget when the policy opts into
/// computational selection: greedily defer the most expensive controllers
/// (deterministic ties broken by id) until the remaining demand fits.
/// Queen-tagged controllers ([`ControllerCost::is_queen`]) are exempt:
/// they are never advised for deferral and do not consume the ordinary
/// budget. Purely advisory; every advised deferral is recorded by
/// `Governor`.
pub fn compute_starvation_review(
    policy: Policy,
    budget: ComputeUnits,
    costs: &[ControllerCost],
) -> ComputeReview {
    if !matches!(policy, Policy::ComputationalSelection) {
        return ComputeReview { advised_deferral: Vec::new() };
    }
    // Queen exemption: queen controllers are filtered out entirely — their
    // cost is admitted by guarantee and cannot consume/matter for this
    // review.
    let ordinary: Vec<ControllerCost> =
        costs.iter().filter(|c| !c.is_queen).cloned().collect();
    if ordinary.is_empty() {
        return ComputeReview { advised_deferral: Vec::new() };
    }
    let total: ComputeUnits = ordinary.iter().map(|c| c.cost).sum();
    if total.value() <= budget.value() {
        return ComputeReview { advised_deferral: Vec::new() };
    }
    // Deterministic: cost descending, id ascending on ties.
    let mut sorted: Vec<&ControllerCost> = ordinary.iter().collect();
    sorted.sort_by(|a, b| b.cost.value().partial_cmp(&a.cost.value()).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id)));
    let mut remaining = total;
    let mut advised = Vec::new();
    for c in sorted {
        if remaining.value() <= budget.value() {
            break;
        }
        remaining = remaining.saturating_sub(c.cost);
        advised.push(c.clone());
    }
    ComputeReview { advised_deferral: advised }
}
