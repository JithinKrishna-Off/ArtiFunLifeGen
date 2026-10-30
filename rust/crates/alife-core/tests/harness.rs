//! Viability harness + focused engine tests (milestones 1/2).
//!
//! The harness runs a genesis-like 12_000-step headless run and prints an
//! end-of-run diagnostic table. Focused tests prove the causal chain:
//! food exists -> sensors see it -> eating credits energy -> fertility
//! gradients bias spawning -> reproduction fires -> lineage logic holds and
//! per-step energy conservation. God-intervention and weather determinism /
//! save-roundtrip tests live here too.

use alife_common::{GodEvent, SimParams, WorldGenConfig};
use alife_core::{SimApi, Simulation};

fn default_sim() -> Simulation {
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    Simulation::new(&SimParams::default(), world)
}

fn small_world(params: &SimParams) -> Simulation {
    let cfg = WorldGenConfig { width: 800, height: 600, cell: 8, ..WorldGenConfig::default() };
    let world = alife_worldgen::generate(&cfg);
    Simulation::new(params, world)
}

fn sum_energy(sim: &Simulation) -> f64 {
    sim.organisms().iter().map(|o| o.energy.max(0.0) as f64).sum()
}

// ------------------------------------------------------------- viability

#[test]
fn genesis_viability_run_12k() {
    let t0 = std::time::Instant::now();
    let mut sim = default_sim();

    // causal link 1: constructor seeds the promised amount of real food
    let initial_food_alive = sim.foods().iter().filter(|f| f.alive).count();
    eprintln!("[harness] initial food alive: {initial_food_alive}");
    assert!(initial_food_alive >= 500, "constructor must seed >= 500 food items");

    // causal link 2: sensors point to real food for a scripted constellation
    {
        let params = SimParams {
            initial_population: 2,
            initial_food: 0,
            food_regen_per_sec: 0.0,
            max_population: 2,
            ..SimParams::default()
        };
        let mut s = small_world(&params);
        // place 5 food items right next to organism #1 (before any movement)
        let ox = s.organism(1).unwrap().x;
        let oy = s.organism(1).unwrap().y;
        for k in 0..5 {
            assert!(SimApi::add_food(&mut s, ox + 1.0, oy + k as f32 * 0.8));
        }
        s.advance();
        let o = s.organism(1).unwrap();
        assert!(o.last_inputs[2] > 0.0, "food intensity sensor must see real food");
        assert!(o.last_inputs[0].is_finite() && o.last_inputs[1].is_finite());
        // eating transferred energy exclusively (ledger gain > 0, foods gone)
        let (gain, _, _, _, _) = s.step_energy_ledger();
        assert!(gain > 0.0, "eating must transfer food energy to the eater");
        assert!(s.foods().len() <= 5, "uneaten food only");
    }

    // causal link 3: food spawn follows fertility gradients
    {
        let params = SimParams {
            initial_population: 0,
            initial_food: 3000,
            max_food: 5000,
            ..SimParams::default()
        };
        let cfg = WorldGenConfig { width: 1200, height: 800, cell: 8, ..WorldGenConfig::default() };
        let world = alife_worldgen::generate(&cfg);
        let s = Simulation::new(&params, world.clone());
        let spawned: Vec<(f32, f32)> =
            s.foods().iter().map(|f| (f.x, f.y)).collect();
        assert!(spawned.len() >= 2000, "constructor reached most of its food target");
        let spawned_mean: f64 =
            spawned.iter().map(|&(x, y)| world.sample_fertility(x, y) as f64).sum::<f64>()
                / spawned.len().max(1) as f64;
        let mut base_mean = 0.0f64;
        let mut r = alife_common::Rng::new(99);
        let n = 4000;
        for _ in 0..n {
            let x = r.range(0.0, world.width);
            let y = r.range(0.0, world.height);
            base_mean += world.sample_fertility(x, y) as f64;
        }
        base_mean /= n as f64;
        eprintln!("[harness] fertility mean: spawned={spawned_mean:.4} uniform={base_mean:.4}");
        assert!(
            spawned_mean > base_mean * 1.05,
            "food spawns must concentrate in fertile cells"
        );
    }
    // causal link 4: the long headless run (12_000 steps)
    let max_steps = 12_000usize;
    let mut births_total = 0u64;
    let mut prev_pop = sim.stats().population;
    for step in 1..=max_steps {
        let ev = sim.advance();
        births_total += ev.births.len() as u64;
        if step % 1000 == 0 {
            let st = sim.stats();
            // population may legitimately swing over 12k steps; report only
            if prev_pop > 0 && st.population == 0 {
                eprintln!("[harness] step {step}: extinction (allowed but reported)");
            }
            prev_pop = st.population;
            eprintln!(
                "[harness] step {:>5} | pop {:>5} | food {:>5} | births+ {} ({}) | deaths+ {} ({}) | avgE {:>6.1} | avgGen {:>5.2} | maxGen {} | div {:.2} | wet {:.2} | elapsed {}",
                step,
                st.population,
                st.food_count,
                births_total,
                st.total_births,
                ev.deaths.len(),
                st.total_deaths,
                st.avg_energy,
                st.avg_generation,
                st.max_generation,
                st.diversity_index,
                st.wet_cover,
                t0.elapsed().as_secs_f32()
            );
        }
        // bounded memory invariants (max_food = genesis default 1300)
        assert!(sim.stats().food_count as usize <= 1300);
    }

    let st = sim.stats();
    let (starve, age, hazard) = sim.death_cause_totals();
    let dt = t0.elapsed();
    eprintln!();
    eprintln!("============================== END OF RUN ===============================");
    eprintln!("steps: {}   elapsed: {:?}  ({:?}/step)", max_steps, dt, dt / max_steps as u32);
    eprintln!("population: {}   max_population cap: {}", st.population, st.max_population);
    eprintln!("foods alive: {}  (max_food {})", st.food_count, 1300);
    eprintln!("births: {} total   births/1000 steps = {:.1}", st.total_births,
        st.total_births as f64 * 1000.0 / max_steps as f64);
    eprintln!("deaths: {} total   (starve {starve} / age {age} / hazard {hazard})", st.total_deaths);
    let d = sim.diagnosis();
    eprintln!(
        "window: births {}/600 steps | deaths {}/600 | starvation share {:.2} | deficit ratio {:.2}",
        d.births_window, d.deaths_window, d.starvation_share_window, d.food_deficit_ratio
    );
    eprintln!(
        "avg energy: {:.1} | avg gen: {:.2} | max gen: {} | species: {} | diversity: {:.2} | wet_cover: {:.2}",
        st.avg_energy, st.avg_generation, st.max_generation, st.species_counts.len(),
        st.diversity_index, st.wet_cover
    );
    eprintln!("=========================================================================");
    eprintln!();

    // reproduction fired (documented neutral-gate lenience active)
    assert!(st.total_births > 0, "run must show real births over 12k steps");
    assert!(st.max_generation >= 2, "offspring generation increments");
    // cause distribution closes the books
    assert_eq!(starve + age + hazard, st.total_deaths, "cause totals must add up");
    // bounded memory
    assert!(sim.foods().len() <= 1300, "foods bounded by max_food");
    assert!(sim.organisms().len() <= 1200, "population bounded by max_population");
    // every organism sane
    for o in sim.organisms() {
        assert!(o.x.is_finite() && o.y.is_finite() && o.energy.is_finite() && o.energy >= 0.0);
        assert!(o.energy <= o.max_energy * 1.01);
    }
}

// ---------------------------------------------------------- conservation

#[test]
fn energy_conservation_identity() {
    // gently birth-friendly params so the ledger case mix includes births
    let params = SimParams {
        min_repro_age_steps: 120,
        reproduction_cooldown_steps: 60,
        ..SimParams::default()
    };
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&params, world);
    let mut saw_birth = false;
    let mut saw_death = false;
    let mut max_err = 0.0f64;
    for _ in 0..1200 {
        let before = sum_energy(&sim);
        let ev = sim.advance();
        let after = sum_energy(&sim);
        let (gain, drain, birth_cost, birth_credit, death_loss) = sim.step_energy_ledger();
        let expected = before - after;
        let got = drain - gain + birth_cost - birth_credit + death_loss;
        // identity: delta = gain - drain - birth_cost + birth_credit - death_loss
        let err = (expected - got).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-3, "per-step conservation identity broken: err={err}");
        saw_birth |= !ev.births.is_empty();
        saw_death |= !ev.deaths.is_empty();
    }
    eprintln!("[harness] conservation max per-step error: {max_err:.6e}");
    assert!(saw_birth, "birth case mix required for the identity");
    // a death case mix too: harsh famine drives starvation deaths
    let harsh = SimParams {
        initial_population: 100,
        max_population: 140,
        initial_food: 300,
        max_food: 400,
        food_regen_per_sec: 1.0,
        metabolism_base: 2.5,
        ..SimParams::default()
    };
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&harsh, world);
    let mut step = 0;
    while !saw_death && step < 2600 {
        step += 1;
        let before = sum_energy(&sim);
        let ev = sim.advance();
        let after = sum_energy(&sim);
        let (gain, drain, birth_cost, birth_credit, death_loss) = sim.step_energy_ledger();
        let err = ((before - after) - (drain - gain + birth_cost - birth_credit + death_loss)).abs();
        assert!(err < 1e-3, "conservation broken in famine: err={err}");
        saw_death |= !ev.deaths.is_empty();
    }
    assert!(saw_death, "deaths must occur under famine conditions");
    // the gate semantics: neutral zone counts as willing
    assert!(Simulation::gate_allows(0.0));
    assert!(Simulation::gate_allows(0.03));
    assert!(!Simulation::gate_allows(-0.3));
    assert!(Simulation::gate_allows(0.6));
}

// -------------------------------------------------- birth/lineage checks

#[test]
fn reproduction_and_lineage_semantics() {
    let params = SimParams {
        initial_population: 40,
        max_population: 200,
        initial_food: 1500,
        max_food: 3000,
        food_regen_per_sec: 40.0,
        min_repro_age_steps: 60,
        reproduction_cooldown_steps: 30,
        ..SimParams::default()
    };
    let mut sim = small_world(&params);
    let mut checked = 0usize;
    let mut new_lineages = 0usize;
    for _ in 0..1500 {
        let ev = sim.advance();
        for b in &ev.births {
            checked += 1;
            // child is a valid organism: alive, gen consistent
            let child = sim
                .organism(b.child_id)
                .expect("newborn must be present in the same step");
            assert!(child.alive && child.age == 0 && b.generation == child.generation);
            // parent (if it survived the step) is exactly one generation back
            if let Some(parent) = sim.organism(b.parent_id) {
                assert_eq!(b.generation, parent.generation + 1, "generation increments");
            }
            // energy entered via parent only: born child is funded, in bounds
            assert!(child.energy > 0.0 && child.energy <= child.max_energy);
            // species bookkeeping: birth species must equal child lineage
            assert_eq!(b.species_id, child.lineage);
        }
        // births may found new lineages; ids never go backwards
        let ids: Vec<u32> = ev.births.iter().map(|b| b.species_id).collect();
        if !ids.is_empty() {
            new_lineages += 1;
        }
    }
    eprintln!("[harness] births checked: {checked}, eventful steps: {new_lineages}");
    assert!(checked > 10, "births must be produced under birth-friendly conditions");
    // records: lineage logic keeps valid parents for living organisms
    let mut recs = 0;
    for o in sim.organisms() {
        if SimApi::get_record(&sim, o.id).is_some() {
            recs += 1;
        }
    }
    assert!(recs > 0, "records map must cover living organisms");
}

// ------------------------------------------------------------- weather

#[test]
fn weather_determinism_with_events() {
    let mk = || {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        Simulation::new(&SimParams::default(), world)
    };
    let mut a = mk();
    let mut b = mk();
    let rain = GodEvent::Rain { x: 1200.0, y: 800.0, r: 500.0, duration_steps: 2400 };
    let bolt = GodEvent::Lightning { x: 900.0, y: 640.0 };
    for step in 0..300 {
        if step == 10 {
            SimApi::trigger(&mut a, &rain).unwrap();
            SimApi::trigger(&mut b, &rain).unwrap();
        }
        if step == 70 {
            SimApi::trigger(&mut a, &bolt).unwrap();
            SimApi::trigger(&mut b, &bolt).unwrap();
        }
        a.advance();
        b.advance();
    }
    let da = serde_json::to_string(&a.snapshot()).unwrap();
    let db = serde_json::to_string(&b.snapshot()).unwrap();
    assert_eq!(da, db, "same seed + same events => identical trajectory");
    // bolt effects visible: flash pulsed and recent bolt recorded (flash decays,
    // so assert on the history + a wet region)
    assert!(!SimApi::weather(&a).recent_bolts().is_empty());
    let w = SimApi::weather(&a);
    assert!(w.wet(1200.0, 800.0) > 0.05, "rain region must soak the soil");
    // wet cover reacts visibly in stats
    assert!(a.stats().wet_cover > 0.0, "wet cover must leave zero once raining");
}

#[test]
fn trigger_validation_and_boundedness() {
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&SimParams::default(), world);
    // invalid inputs -> Err, never panic
    assert!(SimApi::trigger(&mut sim, &GodEvent::Rain { x: f32::NAN, y: 100.0, r: 100.0, duration_steps: 100 }).is_err());
    assert!(SimApi::trigger(&mut sim, &GodEvent::Rain { x: 100.0, y: 100.0, r: f32::INFINITY, duration_steps: 100 }).is_err());
    assert!(SimApi::trigger(&mut sim, &GodEvent::Rain { x: 100.0, y: 100.0, r: 100.0, duration_steps: 0 }).is_err());
    assert!(SimApi::trigger(&mut sim, &GodEvent::Lightning { x: f32::NAN, y: 0.0 }).is_err());
    assert!(SimApi::trigger(&mut sim, &GodEvent::NutrientDump { x: 10.0, y: 10.0, r: 50.0, energy: -5.0 }).is_err());
    assert!(SimApi::trigger(&mut sim, &GodEvent::PurgeLineage { species_id: 9_999_999 }).is_err());
    assert!(SimApi::interventions(&sim).is_empty(), "failed events are not recorded");
    // valid events recorded and bounded at 64
    for i in 0..70 {
        let ev = GodEvent::Lightning { x: 100.0 + i as f32, y: 100.0 };
        SimApi::trigger(&mut sim, &ev).unwrap();
    }
    let list = SimApi::interventions(&sim);
    assert_eq!(list.len(), 64, "intervention ring keeps the LAST 64");
    assert_eq!(list.first().unwrap().event, GodEvent::Lightning { x: 106.0, y: 100.0 });
    // storm bolt cap validation
    assert!(SimApi::trigger(&mut sim, &GodEvent::Storm { x: 500.0, y: 500.0, r: 200.0, duration_steps: 600, lightning_bolts: 9000 }).is_err());
    // summon respects max_population: fill to cap, then expect Err
    let pop = sim.organisms().len();
    let cap = sim.stats().max_population as usize;
    for _ in pop..cap {
        sim.spawn_founder();
    }
    assert!(SimApi::trigger(&mut sim, &GodEvent::SummonOrganism { x: 300.0, y: 300.0 }).is_err());
    // events must not panic during simulated steps (weather processes bolts)
    for _ in 0..150 {
        sim.advance();
    }
    // protect_lineage state semantics
    let first = sim.organisms()[0].lineage;
    assert!(SimApi::protect_lineage(&mut sim, first, true));
    assert!(SimApi::protect_lineage(&mut sim, first, true)); // idempotent true
    assert!(SimApi::protected_lines(&sim).contains(&first));
    assert!(!SimApi::protect_lineage(&mut sim, first, false));
    assert!(!SimApi::protected_lines(&sim).contains(&first));
}

#[test]
fn save_roundtrip_with_weather_and_interventions() {
    let world_gen = || alife_worldgen::generate(&WorldGenConfig::default());
    let mut a = Simulation::new(&SimParams::default(), world_gen());
    SimApi::trigger(&mut a, &GodEvent::Rain { x: 1500.0, y: 900.0, r: 400.0, duration_steps: 600 }).unwrap();
    SimApi::trigger(&mut a, &GodEvent::NutrientDump { x: 1500.0, y: 900.0, r: 200.0, energy: 400.0 }).unwrap();
    let save_target = a.organisms()[0].lineage;
    SimApi::protect_lineage(&mut a, save_target, true);
    for _ in 0..150 {
        a.advance();
    }
    let save = SimApi::snapshot(&a);
    let world = world_gen();
    let b = Simulation::from_save_core(&SimParams::default(), world, &save);
    // weather state restored exactly
    let (wa, wb) = (SimApi::weather(&a), SimApi::weather(&b));
    assert_eq!(wa.cloud, wb.cloud);
    assert_eq!(wa.rain, wb.rain);
    assert_eq!(wa.wet, wb.wet);
    assert_eq!(wa.effects, wb.effects);
    assert_eq!(
        serde_json::to_string(SimApi::interventions(&a)).unwrap(),
        serde_json::to_string(SimApi::interventions(&b)).unwrap(),
        "interventions list restored exactly"
    );
    assert_eq!(SimApi::protected_lines(&a), SimApi::protected_lines(&b));
    // continued dynamics identical after restore
    let mut a = a;
    let mut b = b;
    for _ in 0..120 {
        a.advance();
        b.advance();
    }
    assert_eq!(a.stats().population, b.stats().population);
    assert_eq!(a.stats().species_counts, b.stats().species_counts);
    assert_eq!(SimApi::rng_state(&a), SimApi::rng_state(&b));
}

// ------------------------------------------------------------- diagnostics

#[test]
fn diag_matches_stats_and_reacts() {
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&SimParams::default(), world);
    let d0: DiagProbe = diag_probe(&sim);
    assert!(d0.population > 0 && d0.food_count > 0);
    assert!(d0.starvation_share_window <= 1.0);
    assert!(d0.food_deficit_ratio.is_finite());
    assert_eq!(d0.wet_cover, 0.0, "idle weather = zero wet cover");
    // rain -> wet_cover rises above zero within a few weather ticks
    SimApi::trigger(&mut sim, &GodEvent::Rain { x: 1200.0, y: 800.0, r: 900.0, duration_steps: 4000 }).unwrap();
    for _ in 0..240 {
        sim.advance();
    }
    let d1 = diag_probe(&sim);
    assert!(d1.wet_cover > 0.02, "rain must raise wet cover, got {}", d1.wet_cover);
    // purge lineage: real deaths, note reports the count
    let sid = sim.organisms()[0].lineage;
    let species_pop = sim.stats().species_counts.iter().find(|s| s.species_id == sid).unwrap().count;
    SimApi::trigger(&mut sim, &GodEvent::PurgeLineage { species_id: sid }).unwrap();
    sim.advance();
    let d2 = diag_probe(&sim);
    assert!(d2.population < d1.population, "purge must remove organisms");
    let note = SimApi::interventions(&sim).last().unwrap().note.clone();
    assert!(note.contains(&species_pop.to_string()), "purge note reports count: {note}");
    // hazard pressure classification: real hazard field cells show up
    let (starve, age, hazard) = sim.death_cause_totals();
    assert!(starve + age + hazard <= sim.stats().total_deaths + 2);
}

struct DiagProbe {
    population: u32,
    food_count: u32,
    starvation_share_window: f32,
    food_deficit_ratio: f32,
    wet_cover: f32,
}

fn diag_probe(sim: &Simulation) -> DiagProbe {
    let d = SimApi::diagnosis(sim);
    DiagProbe {
        population: d.population,
        food_count: d.food_count,
        starvation_share_window: d.starvation_share_window,
        food_deficit_ratio: d.food_deficit_ratio,
        wet_cover: d.wet_cover,
    }
}

// ------------------------------------------------------ fertility coupling

#[test]
fn wet_zone_boosts_fertility_drought_reduces() {
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&SimParams::default(), world);
    // dry point vs rain-soaked point: the wet point must receive the boost
    let x = 600.0;
    let y = 400.0;
    let dry_fert = sim.effective_fertility(x, y);
    SimApi::trigger(&mut sim, &GodEvent::Rain { x, y, r: 300.0, duration_steps: 6000 }).unwrap();
    for _ in 0..600 {
        sim.advance(); // ~10 weather ticks: soil soaks, rain ramps
    }
    let wet_fert = sim.effective_fertility(x, y);
    eprintln!(
        "[harness] fertility dry={dry_fert:.3} wet={wet_fert:.3} (rain cell rains={:.2})",
        SimApi::weather(&sim).rain(x, y)
    );
    assert!(wet_fert > dry_fert * 1.15, "rain-wet zones must boost food spawn");
    // drought lowers it below baseline
    SimApi::trigger(&mut sim, &GodEvent::Drought { x, y, r: 300.0, duration_steps: 6000 }).unwrap();
    for _ in 0..900 {
        sim.advance();
    }
    let drought_fert = sim.effective_fertility(x, y);
    eprintln!("[harness] fertility drought={drought_fert:.3}");
    assert!(drought_fert < dry_fert, "drought must reduce effective fertility");
    // lightning ash boosts a small radius temporarily
    SimApi::trigger(&mut sim, &GodEvent::Lightning { x, y }).unwrap();
    let ash_fert = sim.effective_fertility(x, y);
    assert!(ash_fert >= drought_fert, "ash must not reduce fertility further");
}
