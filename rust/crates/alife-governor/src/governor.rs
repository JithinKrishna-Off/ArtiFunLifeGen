//! The `Governor` itself: poll + plan + event ring + config.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::events::{EventKind, EventLog, EVENT_RING_MAX};
use crate::hardware::{
    advisory_ladder, Budget, GovernorDiagnostic, GovernorState, MeasureSource, AdvisoryAction,
};
use crate::policy::{
    compute_starvation_review, AllocationPlan, ComputeReview, ControllerCost, Policy,
    PolicyInput, SubsystemTimings, TickBudgets, TimingEma, TimingFractions, timing_rebalance,
};
use crate::units::{ComputeUnits, HardwareMemory};

fn timings_sim_heavy() -> f32 {
    timing_rebalance::SIM_HEAVY_FRACTION
}

/// Visual degradation hint level for [`GovernorApplyHint::visual_degrade`]
/// in Warn state: visual detail reduced, everything else untouched.
pub const VISUAL_DEGRADE_WARN: u8 = 1;
/// Reserved intermediate level (visual + neural-observatory reduced); the
/// app may park here when ramping between [`VISUAL_DEGRADE_WARN`] and
/// [`VISUAL_DEGRADE_CRITICAL`]. Never emitted by `apply_hint` itself.
pub const VISUAL_DEGRADE_INTERMEDIATE: u8 = 2;
/// Max degradation level for [`GovernorApplyHint::visual_degrade`] in
/// Critical state: visual + neural observatory degraded, controllers
/// batched, update Hz lowered. Simulation core still runs — never a pause.
pub const VISUAL_DEGRADE_CRITICAL: u8 = 3;

/// Advisory hint the app can apply directly. Pure data — a *measurement*
/// surface, never control-flow: the app applies or declines every field.
/// Deriving type: [`Governor::apply_hint`].
///
/// # Documented hint → app-action mapping
///
/// | Field | App action |
/// |---|---|
/// | `step_cap_hint = None` | No restriction; run the app's own default sim steps per rendered frame. |
/// | `step_cap_hint = Some(n)` | Cap: run `min(app_default_steps_per_frame, n)` sim steps this tick. **Never a floor** — it can only lower what the app was going to do. Warn → `Some(2)`, Critical → `Some(1)` (`STEP_CAP_*` consts). |
/// | `visual_degrade = None` | No advisory; the app keeps/chooses its fidelity (Green). |
/// | `visual_degrade = Some(0)` | Explicit full fidelity (may restore). Currently only seen if the app feeds back a Green confirm; `apply_hint` yields `None` in Green instead. |
/// | `visual_degrade = Some(1)` | Reduce 2D/3D visual detail (Warn). |
/// | `visual_degrade = Some(2)` | Intermediate (see `VISUAL_DEGRADE_INTERMEDIATE`). |
/// | `visual_degrade = Some(3)` | Max degrade: + neural observatory fidelity, batched controllers, lowered Hz (Critical). |
/// | `mode` | HUD token: `unmeasured` (no poll yet), `green`, `warn`, `critical`. |
/// | `reasons` | Ready-made HUD strings: measurement line first, then recent event reasons (newest first). |
///
/// Pausing is deliberately **not** carried here (`SafePause` remains
/// reachable via [`Governor::recommended_action`] / [`Governor::advisory_ladder`]
/// for the coordinator to surface with its own confirmation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernorApplyHint {
    /// Recommended cap on simulation steps per rendered frame this tick
    /// (`None` = no restriction in Green). Apply as a cap (`min`), never a
    /// floor — the hint can only *lower* app-chosen work.
    pub step_cap_hint: Option<u64>,
    /// Visual degradation level 0..=3 (`None` = no advisory in Green).
    /// See the mapping table above and the `VISUAL_DEGRADE_*` consts.
    pub visual_degrade: Option<u8>,
    /// HUD mode token: `unmeasured`, `green`, `warn` or `critical`.
    pub mode: String,
    /// Ready-made reasons for the HUD: first the measurement line (with
    /// hold/no-measurement flags when data is unavailable), then up to the
    /// two newest event reasons from the ring.
    pub reasons: Vec<String>,
}

/// Cap constant story: Warn recommends at most 2 sim steps per rendered
/// frame; Critical reduces heavy stepping to a single step per frame.
/// Constants (not magic in code) so tests and app docs can quote them.
pub const STEP_CAP_WARN: u64 = 2;
pub const STEP_CAP_CRITICAL: u64 = 1;
pub const SNAPSHOT_EVENT_MAX: usize = 32;

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
    /// Timing-report EMA smoothing factor in (0, 1] (see
    /// `report_timings`); smaller = smoother (default 0.25).
    pub timing_ema_alpha: f32,
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
            timing_ema_alpha: 0.25,
        }
    }
}

impl GovernorConfig {
    /// RAM-proportional default: `3/32 of available RAM = 9.375%`, so a
    /// 16 GiB machine yields exactly 1.5 GiB (integer-exact because
    /// 16 GiB is divisible by 32). Budgets computed below 512 MiB are
    /// clamped **up** to 512 MiB (536_870_912 bytes). No upper clamp: the
    /// governor is app-only and advisory, so a bigger machine may honestly
    /// reserve a bigger app budget.
    pub fn default_for_ram(available_bytes: u64) -> Self {
        const MIN: u64 = 512 * 1_048_576; // 512 MiB
        let mut cfg = Self::default();
        cfg.memory_budget_bytes = ((available_bytes / 32) * 3).max(MIN);
        cfg
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
///
/// # Safety contract (verified)
/// This type is **advisory-only**. Verified during development:
/// * No method on `Governor` kills, modifies or deletes anything — plan,
///   poll, review and snapshot return plain data (cloneable, serde /
///   `PartialEq` types from this crate).
/// * No filesystem / process / network APIs anywhere in this crate
///   (`alife-governor` depends only on `serde` + `windows-sys` raw
///   syscalls; it does not
///   depend on `alife-core`, so core state is structurally unreachable).
/// * Critical hardware state escalates *only* the advisory ladder ending
///   in [`AdvisoryAction::SafePause`] — never a direct action.
/// * Tests re-assert this: `boundary_safety_no_lethal_surface`.
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
    /// Last known good memory measurement (`None` = never measured).
    last_mem: Option<HardwareMemory>,
    last_decay: Option<Instant>,
    events: EventLog,
    recommended: AdvisoryAction,
    last_hw_scaled: bool,
    timing_alpha: f32,
    timing_ema: TimingEma,
    last_timing_triggered: bool,
    last_diag: Option<GovernorDiagnostic>,
    last_plan: Option<AllocationPlan>,
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
            last_mem: None,
            last_decay: None,
            events: EventLog::new(cap),
            recommended: AdvisoryAction::None,
            last_hw_scaled: false,
            timing_alpha: config.timing_ema_alpha.clamp(0.01, 1.0),
            timing_ema: TimingEma::default(),
            last_timing_triggered: false,
            last_diag: None,
            last_plan: None,
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
            timing_ema_alpha: self.timing_alpha,
        }
    }

    /// One hardware-guardian poll. Measures, EMA-smooths CPU, advances the
    /// hysteresis state machines, records state-transition events, and
    /// returns the diagnostic snapshot.
    ///
    /// Honest-availability contract: when a measurement is unavailable
    /// (`None` from the [`MeasureSource`] `try_*` surface — e.g. the CPU
    /// window is not open yet, or the OS refused), the affected channel
    /// **holds**: its state is not advanced this poll (missing data never
    /// escalates the governor), the EMA is not fed stale values, and the
    /// diagnostic carries `mem_ok` / `cpu_ok = false` with last-known-good
    /// values. No fabricated measurements, ever.
    pub fn poll(&mut self, source: &dyn MeasureSource) -> GovernorDiagnostic {
        let mem_opt = source.try_process_memory_bytes();
        let cpu_opt = source.try_process_cpu_percent();

        if let Some(m) = mem_opt {
            self.last_mem = Some(m);
        }
        // Held values: last known good (0 = none yet) — clearly flagged.
        let mem_used = if let Some(m) = mem_opt { m } else { self.last_mem.unwrap_or_default() };
        if let Some(c) = cpu_opt {
            self.cpu_ema.update(c, self.alpha);
        }
        let cpu_ema = self.cpu_ema.value.unwrap_or(0.0);

        // State machines advance ONLY on fresh measurements.
        let mem_ok = mem_opt.is_some();
        let cpu_ok = cpu_opt.is_some();
        if let Some(m) = mem_opt {
            self.mem_state = self.budget.advance_mem_state(self.mem_state, m);
        }
        if cpu_ok {
            self.cpu_state = self.budget.advance_cpu_state(self.cpu_state, cpu_ema);
        }

        let combined = self.mem_state.max(self.cpu_state);
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

        let diag = GovernorDiagnostic {
            mem_used: mem_used.bytes(),
            mem_budget: self.budget.memory_budget_bytes,
            cpu_used: cpu_ema,
            cpu_budget: self.budget.cpu_budget_percent,
            state: combined,
            decayed_after_ms: self
                .last_decay
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(0),
            mem_ok,
            cpu_ok,
        };
        self.last_diag = Some(diag);
        diag
    }

    /// Advisory action currently recommended (after the latest poll).
    pub fn recommended_action(&self) -> AdvisoryAction {
        self.recommended
    }

    /// Full escalation ladder for the current state.
    pub fn advisory_ladder(&self) -> &'static [AdvisoryAction] {
        advisory_ladder(self.combined)
    }

    /// Directly applicable advisory hint derived from the latest poll (see
    /// [`GovernorApplyHint`] for the documented field→action mapping).
    /// Pure data from the governor's own state; the app applies or declines
    /// every field. Never lethal: no field can target organisms for
    /// destruction (`tests::apply_hints_never_propose_organism_destroy`).
    pub fn apply_hint(&self) -> GovernorApplyHint {
        let d = self.last_diag;
        let (mode, measurement) = match d {
            None => ("unmeasured".to_string(), "no poll yet".to_string()),
            Some(d) => (
                match d.state {
                    GovernorState::Green => "green".to_string(),
                    GovernorState::Warn => "warn".to_string(),
                    GovernorState::Critical => "critical".to_string(),
                },
                format!(
                    "mem {}/{}{}, cpu {:.1}/{:.1}{}",
                    d.mem_used,
                    d.mem_budget,
                    if d.mem_ok { "" } else { " (held; last known good)" },
                    d.cpu_used,
                    d.cpu_budget,
                    if d.cpu_ok { "" } else { " (held; window not open)" },
                ),
            ),
        };
        let (visual, step_cap) = match self.combined {
            GovernorState::Green => (None, None),
            GovernorState::Warn => (Some(VISUAL_DEGRADE_WARN), Some(STEP_CAP_WARN)),
            GovernorState::Critical => (Some(VISUAL_DEGRADE_CRITICAL), Some(STEP_CAP_CRITICAL)),
        };
        let mut reasons = vec![measurement];
        for e in self.events.as_slice().iter().take(2) {
            reasons.push(e.reason.clone());
        }
        GovernorApplyHint { step_cap_hint: step_cap, visual_degrade: visual, mode, reasons }
    }

    /// Feed the coordinator's per-frame subsystem timings (milliseconds;
    /// see `SubsystemTimings`). They feed a rolling per-subsystem EMA (factor
    /// `GovernorConfig::timing_ema_alpha`); the next `plan()` uses the
    /// normalized fractions to rebalance tick budgets (see
    /// `policy::timing_rebalance` for the formulas). Purely advisory,
    /// deterministic from the same sequence of inputs.
    pub fn report_timings(&mut self, timings: &SubsystemTimings) {
        self.timing_ema.update(timings, self.timing_alpha);
    }

    /// Generate the deterministic allocation plan for this tick.
    /// Hardware advice is forwarded onto bounded tick knobs (Warn ~75%,
    /// Critical ~50%); this may emit one Hardware event when scaling
    /// turns on/off, never spams. Timing-derived rebalance (render heavy
    /// → 30 Hz; neural heavy → lower neural Hz; sim heavy → no knob
    /// change, one advisory saga escalation) is layered on top; `SubsystemShares`
    /// (policy split) are never touched by either mechanism.
    pub fn plan(&mut self, input: &PolicyInput) -> AllocationPlan {
        let (mut shares, compute_budget) = input_policy_shares(input.policy, self.compute_budget_units);
        // Queen guarantee (documented, defensively enforced): under
        // QueenPriority with a living queen the queen share never drops
        // below the guaranteed fraction. Re-normalize only on the
        // defensive bump so the invariant is compositional.
        if input.queen_alive {
            if let Some(guaranteed) = input.policy.guaranteed_queen_share() {
                if shares.queen < guaranteed {
                    shares.queen = guaranteed;
                    shares = shares.normalized();
                }
            }
        }
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

        // Timing-derived rebalance (fractions None before any non-empty
        // report → keep hardware-scaled knobs unchanged).
        let mut timing_rebalanced = false;
        if let Some(fractions) = TimingFractions::from_ema(&self.timing_ema) {
            let rebalanced = tick_budgets.rebalanced_by_timings(&fractions);
            if rebalanced != tick_budgets {
                tick_budgets = rebalanced;
                timing_rebalanced = true;
            }
            // Sim heavy: no knob change (sim cadence is contractual), but
            // escalate one advisory per entered "sim-heavy" episode.
            let sim_heavy = fractions.sim >= timings_sim_heavy();
            if sim_heavy != self.last_timing_triggered {
                if sim_heavy {
                    self.events.push(crate::GovernorEvent {
                        tick: input.tick,
                        kind: EventKind::Scheduling,
                        action_taken: AdvisoryAction::ReduceUpdateHz,
                        reason: format!(
                            "simulation dominates measured frame work ({:.2}); \
                             no governor knob change — advisory: reduce sim load \
                             (queen_compute_ns={})",
                            fractions.sim, input.queen_compute_ns,
                        ),
                    });
                }
                self.last_timing_triggered = sim_heavy;
            }
        }
        let plan = AllocationPlan {
            tick: input.tick,
            policy: input.policy,
            shares,
            compute_budget,
            tick_budgets,
            hardware_state: self.combined,
            hardware_scaled,
            timing_rebalanced,
        };
        self.last_plan = Some(plan.clone());
        plan
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
