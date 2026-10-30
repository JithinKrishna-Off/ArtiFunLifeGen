//! The `Governor` itself: poll + plan + event ring + config.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::events::{EventKind, EventLog, EVENT_RING_MAX};
use crate::hardware::{
    advisory_ladder, Budget, GovernorDiagnostic, GovernorState, MeasureSource, AdvisoryAction,
};
use crate::policy::{
    compute_starvation_review, AllocationPlan, ComputeReview, ControllerCost, Policy, PolicyInput,
    TickBudgets,
};
use crate::units::ComputeUnits;

/// Governor configuration (serde roundtrips for save v3, unwired for now).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernorConfig {
    /// Process working-set budget (app-only; NOT total system RAM).
    pub memory_budget_bytes: u64,
    /// CPU budget, percent of one core (EMA-smoothed measurement).
    pub cpu_budget_percent: f32,
    /// Warn threshold as fraction of budget.
    pub warn_threshold: f32,
    /// Critical threshold as fraction of budget.
    pub critical_threshold: f32,
    /// Hysteresis band as fraction of budget (anti-flapping).
    pub hysteresis: f32,
    /// CPU EMA smoothing factor in (0, 1]; smaller = smoother.
    pub cpu_ema_alpha: f32,
    /// Event ring capacity, clamped to `1..=EVENT_RING_MAX` (128).
    pub event_capacity: usize,
    /// Base CU budget for a tick (fictional units, see units.rs).
    pub compute_budget_units: f32,
    /// Green-state baseline tick knobs.
    pub base_tick_budgets: TickBudgets,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 1_610_612_736, // 1.5 GiB
            cpu_budget_percent: 30.0,
            warn_threshold: 0.85,
            critical_threshold: 0.95,
            hysteresis: 0.05,
            cpu_ema_alpha: 0.25,
            event_capacity: EVENT_RING_MAX,
            compute_budget_units: 200.0,
            base_tick_budgets: TickBudgets::default(),
        }
    }
}

/// EMA accumulator for CPU percent.
#[derive(Debug, Default)]
struct CpuEma {
    value: Option<f32>,
}

impl CpuEma {
    fn update(&mut self, raw_pct: f32, alpha: f32) -> f32 {
        let raw = if raw_pct.is_finite() { raw_pct.max(0.0) } else { 0.0 };
        self.value = Some(match self.value {
            None => raw,
            Some(e) => alpha * raw + (1.0 - alpha) * e,
        });
        self.value.unwrap()
    }
}

/// God Commander resource governor: hardware guardian + simulation
/// scheduler. Advisory by design (see crate docs for the safety contract).
pub struct Governor {
    budget: Budget,
    alpha: f32,
    compute_budget_units: f32,
    base_tick_budgets: TickBudgets,
    cfg_event_capacity: usize,
    mem_state: GovernorState,
    cpu_state: GovernorState,
    combined: GovernorState,
    cpu_ema: CpuEma,
    last_decay: Option<Instant>,
    events: EventLog,
    recommended: AdvisoryAction,
    last_hw_scaled: bool,
}

impl Governor {
    /// Create with a (clamped into valid ranges) configuration.
    pub fn new(config: GovernorConfig) -> Self {
        let budget = Budget {
            memory_budget_bytes: config.memory_budget_bytes,
            cpu_budget_percent: config.cpu_budget_percent,
            warn_threshold: config.warn_threshold,
            critical_threshold: config.critical_threshold,
            hysteresis: config.hysteresis,
        }
        .clamped();
        let alpha = config.cpu_ema_alpha.clamp(0.01, 1.0);
        let cap = config.event_capacity;
        Self {
            budget,
            alpha,
            compute_budget_units: config.compute_budget_units.max(0.0),
            base_tick_budgets: config.base_tick_budgets.clamped(),
            cfg_event_capacity: cap,
            mem_state: GovernorState::Green,
            cpu_state: GovernorState::Green,
            combined: GovernorState::Green,
            cpu_ema: CpuEma::default(),
            last_decay: None,
            events: EventLog::new(cap),
            recommended: AdvisoryAction::None,
            last_hw_scaled: false,
        }
    }

    pub fn config(&self) -> GovernorConfig {
        GovernorConfig {
            memory_budget_bytes: self.budget.memory_budget_bytes,
            cpu_budget_percent: self.budget.cpu_budget_percent,
            warn_threshold: self.budget.warn_threshold,
            critical_threshold: self.budget.critical_threshold,
            hysteresis: self.budget.hysteresis,
            cpu_ema_alpha: self.alpha,
            event_capacity: self.cfg_event_capacity,
            compute_budget_units: self.compute_budget_units,
            base_tick_budgets: self.base_tick_budgets,
        }
    }

    /// One hardware-guardian poll. Measures, EMA-smooths CPU, advances the
    /// hysteresis state machines, records state-transition events, and
    /// returns the diagnostic snapshot.
    pub fn poll(&mut self, source: &dyn MeasureSource) -> GovernorDiagnostic {
        let mem_used = source.process_memory_bytes();
        let cpu_ema = self.cpu_ema.update(source.process_cpu_percent(), self.alpha);

        let new_mem = self.budget.advance_mem_state(self.mem_state, mem_used);
        let new_cpu = self.budget.advance_cpu_state(self.cpu_state, cpu_ema);
        self.mem_state = new_mem;
        self.cpu_state = new_cpu;

        let combined = new_mem.max(new_cpu);
        if combined != self.combined {
            if combined < self.combined {
                self.last_decay = Some(Instant::now());
            }
            let reason = match (combined >= self.combined, combined) {
                (true, GovernorState::Green) => "state reset".to_string(),
                (true, _) => "hardware budget crossed upward".to_string(),
                (false, _) => "hardware budget crossed downward".to_string(),
            };
            let action = advisory_ladder(combined).first().copied().unwrap_or(AdvisoryAction::None);
            self.events.push(crate::GovernorEvent {
                tick: 0, // caller may tag; scheduler events use real ticks
                kind: EventKind::Hardware,
                action_taken: action,
                reason: format!("{reason} (mem {}/{}, cpu {:.1}/{:.1})",
                    mem_used.bytes(), self.budget.memory_budget_bytes, cpu_ema, self.budget.cpu_budget_percent),
            });
            self.combined = combined;
        }
        self.recommended = advisory_ladder(combined).first().copied().unwrap_or(AdvisoryAction::None);

        GovernorDiagnostic {
            mem_used: mem_used.bytes(),
            mem_budget: self.budget.memory_budget_bytes,
            cpu_used: cpu_ema,
            cpu_budget: self.budget.cpu_budget_percent,
            state: combined,
            decayed_after_ms: self
                .last_decay
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Advisory action currently recommended (after the latest poll).
    pub fn recommended_action(&self) -> AdvisoryAction {
        self.recommended
    }

    /// Full escalation ladder for the current state.
    pub fn advisory_ladder(&self) -> &'static [AdvisoryAction] {
        advisory_ladder(self.combined)
    }

    /// Generate the deterministic allocation plan for this tick.
    /// Hardware advice is forwarded onto bounded tick knobs (Warn ~75%,
    /// Critical ~50%); this may emit one Hardware event when scaling
    /// turns on/off, never spams.
    pub fn plan(&mut self, input: &PolicyInput) -> AllocationPlan {
        let (shares, compute_budget) = input_policy_shares(input.policy, self.compute_budget_units);
        let mut tick_budgets = self.base_tick_budgets.scaled_by_hardware(self.combined);
        tick_budgets.soft_population_cap = tick_budgets
            .soft_population_cap
            .min(input.policy.population_cap(self.base_tick_budgets.soft_population_cap));
        let hardware_scaled = self.combined != GovernorState::Green;
        if hardware_scaled != self.last_hw_scaled {
            if hardware_scaled {
                self.events.push(crate::GovernorEvent {
                    tick: input.tick,
                    kind: EventKind::Hardware,
                    action_taken: self.recommended,
                    reason: format!("tick budgets scaled by hardware state {:?}", self.combined),
                });
            }
            self.last_hw_scaled = hardware_scaled;
        }
        AllocationPlan {
            tick: input.tick,
            policy: input.policy,
            shares,
            compute_budget,
            tick_budgets,
            hardware_scaled,
        }
    }

    /// Review per-controller CU costs against a plan (computational
    /// selection only). Advises deferral of the most expensive controllers
    /// until demand fits; records every advised deferral. Never lethal.
    pub fn review_controllers(
        &mut self,
        plan: &AllocationPlan,
        costs: &[ControllerCost],
    ) -> ComputeReview {
        let review = compute_starvation_review(plan.policy, plan.compute_budget, costs);
        for c in &review.advised_deferral {
            self.events.push(crate::GovernorEvent {
                tick: plan.tick,
                kind: EventKind::ComputeSelection,
                action_taken: AdvisoryAction::BatchControllers,
                reason: format!(
                    "controller {} '{}' ({:.1} CU) advised for deferral; tick budget {:.1} CU",
                    c.id, c.label, c.cost.value(), plan.compute_budget.value()
                ),
            });
        }
        review
    }

    /// Newest-first event ring (bounded, see `EVENT_RING_MAX`).
    pub fn events(&self) -> &[crate::GovernorEvent] {
        self.events.as_slice()
    }
}

fn input_policy_shares(policy: Policy, compute_budget: f32) -> (crate::policy::SubsystemShares, ComputeUnits) {
    policy.shares_and_budget(compute_budget)
}
