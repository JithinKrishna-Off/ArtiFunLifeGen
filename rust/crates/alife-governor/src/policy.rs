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
    fn normalized(mut self) -> Self {
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
    pub queen_alive: bool,
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
    /// True when hardware advice degraded the knobs vs the green baseline.
    pub hardware_scaled: bool,
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
}

/// One controller's CU cost for a tick, tagged by the coordinator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerCost {
    /// Stable controller id (coordinator-owned).
    pub id: u64,
    pub label: String,
    pub cost: ComputeUnits,
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
/// Purely advisory; every advised deferral is recorded by `Governor`.
pub fn compute_starvation_review(
    policy: Policy,
    budget: ComputeUnits,
    costs: &[ControllerCost],
) -> ComputeReview {
    if !matches!(policy, Policy::ComputationalSelection) || costs.is_empty() {
        return ComputeReview { advised_deferral: Vec::new() };
    }
    let total: ComputeUnits = costs.iter().map(|c| c.cost).sum();
    if total.value() <= budget.value() {
        return ComputeReview { advised_deferral: Vec::new() };
    }
    // Deterministic: cost descending, id ascending on ties.
    let mut sorted: Vec<&ControllerCost> = costs.iter().collect();
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