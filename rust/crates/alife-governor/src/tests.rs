//! Tests: synthetic `MeasureSource` mocks only — no real hardware dependency.

use crate::*;

struct MockSource {
    mem: HardwareMemory,
    cpu: f32,
}

impl MockSource {
    fn new(mem_bytes: u64, cpu: f32) -> Self {
        Self { mem: HardwareMemory::from_bytes(mem_bytes), cpu }
    }
}

impl MeasureSource for MockSource {
    fn process_memory_bytes(&self) -> HardwareMemory {
        self.mem
    }
    fn process_cpu_percent(&self) -> f32 {
        self.cpu
    }
}

fn budget_bytes() -> u64 {
    1_000_000_000
}

fn cfg() -> GovernorConfig {
    let mut c = GovernorConfig::default();
    c.memory_budget_bytes = budget_bytes();
    c.cpu_budget_percent = 100.0; // easy math
    c.warn_threshold = 0.85;
    c.critical_threshold = 0.95;
    c.hysteresis = 0.05;
    c.cpu_ema_alpha = 0.25;
    c
}

fn mem(x: f32) -> u64 {
    (budget_bytes() as f32 * x) as u64
}

// --- thresholds + hysteresis (no flapping) --------------------------------

#[test]
fn mem_thresholds_and_hysteresis_no_flapping() {
    let mut g = Governor::new(cfg());
    // Green below warn.
    let d = g.poll(&MockSource::new(mem(0.50), 0.0));
    assert_eq!(d.state, GovernorState::Green);
    // Cross warn.
    let d = g.poll(&MockSource::new(mem(0.87), 0.0));
    assert_eq!(d.state, GovernorState::Warn);
    // Drop inside the hysteresis band (0.85 - 0.05 = 0.80 < 0.83) -> stays Warn (no flap).
    let d = g.poll(&MockSource::new(mem(0.83), 0.0));
    assert_eq!(d.state, GovernorState::Warn);
    // Even right under warn threshold but above band -> stays Warn.
    let d = g.poll(&MockSource::new(mem(0.849), 0.0));
    assert_eq!(d.state, GovernorState::Warn);
    // Drop below band -> Green.
    let d = g.poll(&MockSource::new(mem(0.79), 0.0));
    assert_eq!(d.state, GovernorState::Green);

    // Critical is immediate escalation.
    let d = g.poll(&MockSource::new(mem(0.96), 0.0));
    assert_eq!(d.state, GovernorState::Critical);
    // Band between crit-hyst and crit keeps Critical (0.95-0.05=0.90).
    let d = g.poll(&MockSource::new(mem(0.92), 0.0));
    assert_eq!(d.state, GovernorState::Critical);
    // Below 0.90 -> de-escalate to Warn (still >= warn).
    let d = g.poll(&MockSource::new(mem(0.89), 0.0));
    assert_eq!(d.state, GovernorState::Warn);
    // Below 0.80 -> Green.
    let d = g.poll(&MockSource::new(mem(0.79), 0.0));
    assert_eq!(d.state, GovernorState::Green);
}

#[test]
fn cpu_thresholds_and_hysteresis_no_flapping() {
    let mut g = Governor::new(cfg());
    // EMA: first raw reading sets the value directly; 85 -> Warn.
    let d = g.poll(&MockSource::new(0, 85.0));
    assert_eq!(d.cpu_used, 85.0);
    assert_eq!(d.state, GovernorState::Warn);
    // Next reading 84 stays in band (warn band lower edge = 85 - 5 = 80).
    let d = g.poll(&MockSource::new(0, 84.0));
    // ema = 0.25*84 + 0.75*85 = 84.75 -> still Warn.
    assert_eq!(d.cpu_used, 84.75);
    assert_eq!(d.state, GovernorState::Warn);
    // Push over critical.
    let d = g.poll(&MockSource::new(0, 100.0));
    // ema = 0.25*100 + 0.75*84.75 = 88.5625 -> Warn still (>= crit 95? no).
    assert_eq!(d.cpu_used, 88.5625);
    assert_eq!(d.state, GovernorState::Warn);
    // Sustained 100 crosses critical.
    g.poll(&MockSource::new(0, 100.0)); // 91.42
    g.poll(&MockSource::new(0, 100.0)); // 93.57
    let d = g.poll(&MockSource::new(0, 100.0)); // 95.17 >= 95 -> Critical
    assert_eq!(d.state, GovernorState::Critical);
    // Even 0 reading stays Critical inside band (<= crit-hyst=90 needed).
    let d = g.poll(&MockSource::new(0, 0.0)); // ema = 71.38 -> below warn band 80 -> Green
    assert_eq!(d.state, GovernorState::Green);
}

#[test]
fn cpu_ema_math_matches_conditional_expectation() {
    let mut g = Governor::new(cfg());
    let src = || MockSource::new(0, 1.0);
    let r0 = g.poll(&src()); // first sample = 1.0
    assert!((r0.cpu_used - 1.0).abs() < 1e-6, "first EMA sample should be raw");
    let r1 = g.poll(&MockSource::new(0, 0.0));
    assert!((r1.cpu_used - 0.75).abs() < 1e-6);
    let r2 = g.poll(&MockSource::new(0, 0.0));
    assert!((r2.cpu_used - 0.5625).abs() < 1e-6);
    let r3 = g.poll(&MockSource::new(0, 0.0));
    assert!((r3.cpu_used - 0.421875).abs() < 1e-6);
}

// --- policy determinism ----------------------------------------------------

#[test]
fn policy_share_determinism_and_sum_bound() {
    let input = PolicyInput {
        tick: 7,
        population: 120,
        policy: Policy::Balanced,
        queen_alive: true,
        queen_compute_ns: 0,
        avg_bio_energy: BioEnergy::new(50.0),
        demanded_compute: ComputeUnits::new(150.0),
    };
    for policy in [
        Policy::Balanced,
        Policy::QueenPriority,
        Policy::ColonyPriority,
        Policy::EqualShare,
        Policy::MaxPopulation,
        Policy::ComputationalSelection,
    ] {
        let mut a = Governor::new(cfg());
        let mut b = Governor::new(cfg());
        let mut pa = input.clone();
        pa.policy = policy;
        let plan_a = a.plan(&pa);
        let plan_b = b.plan(&pa);
        assert_eq!(plan_a, plan_b, "{policy:?} must be deterministic");
        assert!(
            plan_a.shares.sum() <= 1.0 + 1e-6,
            "{policy:?} shares must sum <= 1.0, got {}",
            plan_a.shares.sum()
        );
        assert!(plan_a.compute_budget.value() > 0.0);
    }
    // QueenPriority actually favors the queen.
    let mut g = Governor::new(cfg());
    let pi = PolicyInput { policy: Policy::QueenPriority, ..input };
    let plan = g.plan(&pi);
    assert!(plan.shares.queen > plan.shares.colony);
    // ComputationalSelection shrinks the CU budget.
    let mut g = Governor::new(cfg());
    let pi = PolicyInput { policy: Policy::ComputationalSelection, ..input };
    let plan = g.plan(&pi);
    let mut g = Governor::new(cfg());
    let pi = PolicyInput { policy: Policy::Balanced, ..input };
    let plan_bal = g.plan(&pi);
    assert!(plan.compute_budget.value() < plan_bal.compute_budget.value());
}

#[test]
fn equal_share_sums_exactly_to_one() {
    let (shares, _) = Policy::EqualShare.shares_and_budget(200.0);
    let s = shares.sum();
    assert!((s - 1.0).abs() < 1e-6, "EqualShare must total 1.0, got {s}");
}

// --- knobs: hardware advice forwarded to tick budgets ---------------------

#[test]
fn tick_budgets_follow_hardware_advice_and_bounded() {
    let mut g = Governor::new(cfg());
    let input = PolicyInput {
        policy: Policy::Balanced,
        tick: 1,
        population: 100,
        queen_alive: true,
        queen_compute_ns: 0,
        avg_bio_energy: BioEnergy::new(1.0),
        demanded_compute: ComputeUnits::new(0.0),
    };
    let green = g.plan(&input);
    // Force CPU critical via sustained load.
    for _ in 0..8 {
        let d = g.poll(&MockSource::new(0, 100.0));
        if d.state == GovernorState::Critical {
            break;
        }
    }
    let crit = g.plan(&input);
    assert!(crit.hardware_scaled);
    assert!(crit.tick_budgets.update_hz_simulation < green.tick_budgets.update_hz_simulation);
    assert!(crit.tick_budgets.controller_batch_size <= green.tick_budgets.controller_batch_size);
    assert!(crit.tick_budgets.soft_population_cap <= green.tick_budgets.soft_population_cap);
    // Bounded: clamps hold (1 <= hz).
    assert!(crit.tick_budgets.update_hz_simulation >= 1.0);
    assert!(crit.tick_budgets.controller_batch_size >= 1);
}

// --- event ring: bound + rotation ------------------------------------------

#[test]
fn event_ring_bounded_and_rotates() {
    let mut g = Governor::new(cfg());
    let mut input = PolicyInput {
        policy: Policy::ComputationalSelection,
        tick: 0,
        population: 10,
        queen_alive: true,
        queen_compute_ns: 0,
        avg_bio_energy: BioEnergy::new(1.0),
        demanded_compute: ComputeUnits::new(0.0),
    };
    // Push far more ComputeSelection events than the 128 capacity.
    for tick in 1..=200u64 {
        input.tick = tick;
        let plan = g.plan(&input);
        g.review_controllers(
            &plan,
            &[ControllerCost {
                id: tick % 2,
                label: "expensive".into(), is_queen: false,
                cost: ComputeUnits::new(999.0),
            }],
        );
    }
    let ev = g.events();
    assert_eq!(ev.len(), 128, "ring must stay bounded");
    // Newest first: the newest ComputeSelection tick (200) must appear.
    assert!(ev.iter().any(|e| e.tick == 200));
    // Oldest pushed tick (1) must have rotated out.
    assert!(!ev.iter().any(|e| e.tick == 1));
}

// --- compute starvation review ----------------------------------------------

#[test]
fn compute_selection_advises_expensive_first_never_lethal() {
    let mut g = Governor::new(cfg());
    let input = PolicyInput {
        policy: Policy::ComputationalSelection,
        tick: 5,
        population: 10,
        queen_alive: true,
        queen_compute_ns: 0,
        avg_bio_energy: BioEnergy::new(1.0),
        demanded_compute: ComputeUnits::new(0.0),
    };
    let plan = g.plan(&input);
    let budget = plan.compute_budget.value();
    let costs = vec![
        ControllerCost { id: 1, label: "cheap".into(), is_queen: false, cost: ComputeUnits::new(budget * 0.2) },
        ControllerCost { id: 2, label: "mid".into(), is_queen: false, cost: ComputeUnits::new(budget * 0.4) },
        ControllerCost { id: 3, label: "expensive".into(), is_queen: false, cost: ComputeUnits::new(budget * 0.8) },
    ];
    let review = g.review_controllers(&plan, &costs);
    // Total = 1.4*budget. Deferring "expensive" (0.8) leaves 0.6 <= 1.0.
    assert_eq!(review.advised_deferral.len(), 1);
    assert_eq!(review.advised_deferral[0].label, "expensive");
    // Event recorded, advisory only.
    let ev = g.events();
    assert!(ev.iter().any(|e| e.kind == EventKind::ComputeSelection
        && e.action_taken == AdvisoryAction::BatchControllers));
    // Demand that fits -> no advice, no events.
    let mut g2 = Governor::new(cfg());
    let pi2 = input.clone();
    let plan2 = g2.plan(&pi2);
    let review2 = g2.review_controllers(&plan2, &costs[..1]);
    assert!(review2.advised_deferral.is_empty());
    assert!(g2.events().is_empty());
}

// --- serialization roundtrips ------------------------------------------------

#[test]
fn diagnostic_and_config_serde_roundtrip() {
    let d = GovernorDiagnostic {
        mem_used: 123_456_789,
        mem_budget: 1_610_612_736,
        cpu_used: 12.5,
        cpu_budget: 30.0,
        state: GovernorState::Warn,
        decayed_after_ms: 42,
        mem_ok: true,
        cpu_ok: true,
    };
    let s = serde_json::to_string(&d).unwrap();
    let d2: GovernorDiagnostic = serde_json::from_str(&s).unwrap();
    assert_eq!(d, d2);

    let mut c = GovernorConfig::default();
    c.memory_budget_bytes = 999;
    let s2 = serde_json::to_string(&c).unwrap();
    let c3: GovernorConfig = serde_json::from_str(&s2).unwrap();
    assert_eq!(c3, c);
    // Governor::config() must roundtrip too.
    let g = Governor::new(c3.clone());
    assert_eq!(g.config(), c3);
}

#[test]
fn plan_and_events_serde_roundtrip() {
    let plan = AllocationPlan {
        tick: 3,
        policy: Policy::MaxPopulation,
        shares: SubsystemShares { queen: 0.1, colony: 0.3, ecology: 0.2, weather: 0.1, neural_observatory: 0.2, rendering: 0.1 },
        compute_budget: ComputeUnits::new(150.5),
        tick_budgets: TickBudgets::default(),
        hardware_state: GovernorState::Green,
        hardware_scaled: false,
        timing_rebalanced: false,
    };
    let s = serde_json::to_string(&plan).unwrap();
    let p2: AllocationPlan = serde_json::from_str(&s).unwrap();
    assert_eq!(p2, plan);
}

// --- apply hints: direct-consumption advisory surface ------------------------

/// Synthetic input shared by poll→plan→hints tests.
fn input(tick: u64, policy: Policy) -> PolicyInput {
    PolicyInput {
        tick,
        population: 128,
        policy,
        queen_alive: true,
        queen_compute_ns: 0,
        avg_bio_energy: BioEnergy::new(50.0),
        demanded_compute: ComputeUnits::new(100.0),
    }
}

#[test]
fn poll_plan_apply_hints_happy_path() {
    let mut g = Governor::new(cfg());
    let src = || MockSource::new(mem(0.50), 0.0);
    let diag = g.poll(&src());
    assert_eq!(diag.state, GovernorState::Green);
    assert!(diag.mem_ok && diag.cpu_ok);

    let plan = g.plan(&input(1, Policy::Balanced));
    let hints = plan.to_apply_hints();
    assert_eq!(hints.step_cap_hint, None, "Green: no step cap");
    assert_eq!(hints.visual_degrade, 0, "Green: no visual degradation");
    assert!(!hints.pause_advisory, "Green: never a pause advisory");
    assert_eq!(plan.hardware_state, GovernorState::Green);
    assert!(!plan.hardware_scaled);

    // Escalate via sustained over-budget memory.
    for tick in 2..=64 {
        let d = g.poll(&MockSource::new(mem(0.97), 0.0));
        if d.state == GovernorState::Critical {
            break;
        }
        let _ = tick;
    }
    assert_eq!(g.poll(&MockSource::new(mem(0.97), 0.0)).state, GovernorState::Critical);
    let plan = g.plan(&input(70, Policy::Balanced));
    let hints = plan.to_apply_hints();
    assert_eq!(hints.step_cap_hint, Some(crate::STEP_CAP_CRITICAL));
    assert_eq!(hints.visual_degrade, 3);
    assert!(hints.pause_advisory, "Critical carries the pause advisory");
    assert_eq!(plan.hardware_state, GovernorState::Critical);

    // Governor::apply_hint (HUD surface) stays consistent.
    let hint = g.apply_hint();
    assert_eq!(hint.mode, "critical");
    assert_eq!(hint.step_cap_hint, hints.step_cap_hint);
    assert_eq!(hint.visual_degrade, Some(3));
}

#[test]
fn apply_hints_deescalation_no_flapping() {
    let mut g = Governor::new(cfg());
    g.poll(&MockSource::new(mem(0.87), 0.0)); // Warn
    let w1 = g.plan(&input(1, Policy::Balanced)).to_apply_hints();
    assert_eq!(w1.visual_degrade, 1);
    assert_eq!(w1.step_cap_hint, Some(crate::STEP_CAP_WARN));
    // Inside the hysteresis band → still Warn → identical hints (no flap).
    g.poll(&MockSource::new(mem(0.849), 0.0));
    let w2 = g.plan(&input(2, Policy::Balanced)).to_apply_hints();
    assert_eq!(w1, w2, "hints must not flap inside the hysteresis band");
    // Below the band → Green → identical Green hints.
    g.poll(&MockSource::new(mem(0.79), 0.0));
    let g1 = g.plan(&input(3, Policy::Balanced)).to_apply_hints();
    assert_eq!(g1, ApplyHints::default(), "de-escalated hints are the zero hint");
    // Mapping sanity: only levels 0, 1, 3 are ever emitted by plan
    // (level 2 is the reserved intermediate, never generated).
    let emitted = [w1.visual_degrade, w2.visual_degrade, g1.visual_degrade];
    assert!(emitted.into_iter().all(|l| [0u8, 1, 3].contains(&l)));
}

#[test]
fn apply_hints_never_propose_organism_destroy() {
    // Enumerate every reachable plan (Green/Warn/Critical, every policy):
    // hints must contain no organism/population destruction surface at all.
    for policy in [
        Policy::Balanced,
        Policy::QueenPriority,
        Policy::ColonyPriority,
        Policy::EqualShare,
        Policy::MaxPopulation,
        Policy::ComputationalSelection,
    ] {
        for mem_frac in [0.5f32, 0.87, 0.97] {
            let mut g = Governor::new(cfg());
            for tick in 0..24u64 {
                g.poll(&MockSource::new(mem(mem_frac), 0.0));
                let plan = g.plan(&input(tick, policy));
                let h = plan.to_apply_hints();
                assert!(h.visual_degrade <= 3);
                // Only ever a *cap*: a None hint lowers nothing; Some(n) may
                // only lower app-chosen steps (never a floor, never lethal).
                if let Some(n) = h.step_cap_hint {
                    assert!(n >= 1);
                }
                assert!(!h.pause_advisory || plan.hardware_state == GovernorState::Critical);
                let hud = g.apply_hint();
                assert!(
                    !hud.reasons.iter().any(|r| r.contains("kill") || r.contains("destroy")),
                    "reasons must never advertise lethal semantics"
                );
                for e in g.events() {
                    assert!(
                        e.action_taken != AdvisoryAction::PopulationApprox,
                        "escalation ladder must never emit a population action"
                    );
                }
            }
        }
    }
}

#[test]
fn ladder_never_contains_lethal_or_population_actions() {
    use crate::hardware::{
        advisory_ladder as ladder, AdvisoryAction, GovernorState as St,
    };
    for state in [St::Green, St::Warn, St::Critical] {
        for action in ladder(state) {
            match action {
                AdvisoryAction::None
                | AdvisoryAction::SafeWarn
                | AdvisoryAction::ReduceVisualDetail
                | AdvisoryAction::ReduceNeuralVisual
                | AdvisoryAction::BatchControllers
                | AdvisoryAction::ReduceUpdateHz
                | AdvisoryAction::SafePause => {}
                // PopulationApprox is wire-compat only: must never appear as
                // an escalation step (no population/lethal semantics).
                AdvisoryAction::PopulationApprox => {
                    panic!("ladder must never contain a population action")
                }
            }
        }
    }
    assert!(ladder(St::Green).is_empty());
    assert!(ladder(St::Warn).contains(&AdvisoryAction::ReduceVisualDetail));
    assert!(ladder(St::Critical).contains(&AdvisoryAction::SafePause));
}

#[test]
fn boundary_safety_no_lethal_surface() {
    // Structural: with a fully Critical governor, the entire advisory
    // surface (plan, hints, ladder, HUD hint, events) contains no operation
    // that could kill an organism — everything is a capped/advisory scalar.
    let mut g = Governor::new(cfg());
    for _ in 0..64 {
        let d = g.poll(&MockSource::new(mem(0.99), 0.0));
        if d.state == GovernorState::Critical {
            break;
        }
    }
    assert_eq!(g.advisory_ladder().last(), Some(&AdvisoryAction::SafePause));
    let plan = g.plan(&input(99, Policy::QueenPriority));
    let h = plan.to_apply_hints();
    // Queen alive + Critical: the queen is NEVER on a destruction path —
    // the strongest tool here is a step cap of 1 (sim keeps running).
    assert_eq!(h.step_cap_hint, Some(crate::STEP_CAP_CRITICAL));
    // The soft population cap is a budget advisory only (0.5× in Critical):
    // birth-budgeting, never a kill order.
    assert_eq!(plan.tick_budgets.soft_population_cap, 2_000);
}

// --- real Windows measurements (this machine; same ranges, not exact) -------

#[test]
fn sysmetrics_real_returns_plausible_values() {
    let src = crate::hardware::SysMetrics::new();
    // Memory: working set of the test process — sane range 1 MiB..=10 GiB.
    let m = src.try_process_memory_bytes().expect("working-set must be measurable");
    assert!(
        m.bytes() >= 1_000_000 && m.bytes() <= 10_750_000_000,
        "working set implausibly small/large: {} bytes",
        m.bytes()
    );
    // Total physical RAM: sane range 1 GiB..=2 TiB.
    let ram = src.total_physical_ram().expect("GlobalMemoryStatusEx must succeed");
    assert!(
        ram >= 1_073_741_824 && ram <= 2_199_023_255_552,
        "total RAM implausible: {ram}"
    );
    // First CPU sample only opens the window (honest None — no fabrication).
    assert_eq!(src.try_process_cpu_percent(), None);
    std::thread::sleep(std::time::Duration::from_millis(5));
    // Second sample: a fresh delta → defined percent of one core.
    let c = src.try_process_cpu_percent().expect("CPU window must open");
    assert!(c.is_finite() && c >= 0.0, "cpu percent: {c}");
    // System-wide (HUD-only channel) opens its own window the same way.
    assert_eq!(src.system_cpu_percent(), None);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let s = src.system_cpu_percent().expect("system window must open");
    assert!(s.is_finite() && (0.0..=100.0).contains(&s), "sys cpu: {s}");
    // The production constructor via the dyn trait object must work too.
    let real = <dyn MeasureSource>::real();
    let via_dyn = real.try_process_memory_bytes().expect("real() must measure");
    assert!(via_dyn.bytes() >= 1_000_000);
}

#[test]
fn timings_unit_is_milliseconds_everywhere() {
    // report_timings input unit: milliseconds (documented on
    // SubsystemTimings); EMA + rebalance only consume *fractions*, so the
    // unit must not leak into knob changes — here it provably does not.
    let mut g = Governor::new(cfg());
    for _ in 0..40 {
        g.report_timings(&SubsystemTimings {
            sim_ms: 4.0,
            render_ms: 5.0,
            neural_ms: 1.0,
            ecology_ms: 1.0,
        });
    }
    let plan = g.plan(&input(1, Policy::Balanced));
    // render fraction ≠ heavy (5/11 ≈ 0.45 < 0.50) → baseline render Hz untouched.
    assert!(!plan.timing_rebalanced);
    assert_eq!(plan.tick_budgets.update_hz_rendering, 60.0);
    // Now render-heavy in ms units → exactly 30 Hz rebalance, once.
    for _ in 0..60 {
        g.report_timings(&SubsystemTimings {
            sim_ms: 1.0,
            render_ms: 9.0,
            neural_ms: 0.0,
            ecology_ms: 0.0,
        });
    }
    let plan = g.plan(&input(2, Policy::Balanced));
    assert!(plan.timing_rebalanced);
    assert_eq!(plan.tick_budgets.update_hz_rendering, 30.0);
}

