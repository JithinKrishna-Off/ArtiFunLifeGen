//! Memory budget acceptance tests ("memory_alpha").
//!
//! They assert structural steady-state bounds (exact sizes — RSS probes are
//! unreliable on Windows where the OS allocator hides heap behavior):
//! - organisms Vec capacity <= 256 after 5000 steps from initial pop 40
//!   (single reserve at construction = min(max_population, 4096); the death
//!   compaction reuses the same backing buffer instead of rebuilding per step)
//! - foods Vec capacity <= 4096 (same single reserve)
//! - lineage records bounded (RECORDS_CAP trim; ancestors of living organisms
//!   are protected, so the bound is cap + live ancestry slack)
//! - HistoryBuffer pair-merge downsampling keeps samples <= 2048 forever
//!
//! Additive benches-as-tests: no shared state, own seed => cannot affect the
//! determinism of any other test.

use alife_common::{SimParams, StepEvents, WorldGenConfig};
use alife_core::{SimApi, Simulation};

fn params_for_bounds() -> SimParams {
    SimParams {
        seed: 97,
        initial_population: 40,
        // small cap: the one-time reserve min(max_population, 4096) then
        // backs a Vec whose capacity stays under 256 slots end-to-end
        max_population: 200,
        initial_food: 600,
        max_food: 3000,
        food_regen_per_sec: 30.0,
        min_repro_age_steps: 60,
        reproduction_cooldown_steps: 30,
        ..SimParams::default()
    }
}

#[test]
fn vec_capacities_and_records_bounded_after_5000_steps() {
    let t0 = std::time::Instant::now();
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let mut sim = Simulation::new(&params_for_bounds(), world);

    let probe0 = sim.memory_probe();
    eprintln!(
        "[memory_alpha] initial: organisms_cap={} foods_cap={} records={} ({} ms construction)",
        probe0.organisms_capacity,
        probe0.foods_capacity,
        probe0.records_len,
        t0.elapsed().as_millis()
    );

    let mut events_seen = StepEvents::default();
    let mut peak_pop = 0usize;
    for _step in 1..=5_000 {
        let ev = sim.advance();
        peak_pop = peak_pop.max(sim.stats().population as usize);
        if !ev.births.is_empty() || !ev.deaths.is_empty() {
            events_seen = ev;
        }
    }
    let probe = sim.memory_probe();
    eprintln!(
        "[memory_alpha] after 5000 steps ({} ms): peak_pop={} final_pop={} \
         organisms_cap={} foods_cap={} records_len={} pending_deaths_cap={}",
        t0.elapsed().as_millis(),
        peak_pop,
        sim.stats().population,
        probe.organisms_capacity,
        probe.foods_capacity,
        probe.records_len,
        probe.pending_deaths_capacity,
    );

    // --- asserted budget bounds ---
    assert!(
        probe.organisms_capacity <= 256,
        "organisms Vec capacity grew to {} (must stay reserved-once <= 256)",
        probe.organisms_capacity
    );
    assert!(
        probe.foods_capacity <= 4096,
        "foods Vec capacity is {} (must be <= 4096)",
        probe.foods_capacity
    );
    // records: the trim keeps dead records bounded; ancestors of living
    // organisms are protected, so observed len = kept-dead + live ancestry
    assert!(
        probe.records_len <= 512 + probe.organisms_capacity,
        "records_len={} exceeds the trim bound",
        probe.records_len
    );
    // deferred-death scratch never accumulates
    assert!(probe.pending_deaths_capacity <= 4096);
    assert!(peak_pop > 0, "run must be a real dynamic run");
    let _ = events_seen;
}

#[test]
fn history_buffer_pair_merge_keeps_2048_samples_bounded() {
    use alife_common::persist::HistoryBuffer;
    use alife_data::step_history;

    let t0 = std::time::Instant::now();
    let mut h = HistoryBuffer::new(20, 2048);
    let ev = StepEvents::default();

    // synthetic long run (150k sim steps): must cross several doubling
    // boundaries (interval 20 -> 40 -> 80 -> ...) and stay bounded forever.
    let steps = 150_000u64;
    for step in 1..=steps {
        let mut stats = alife_common::SimStats {
            step,
            ..alife_common::SimStats::default()
        };
        stats.population = 40 + ((step / 700) % 100) as u32;
        stats.food_count = 800;
        stats.avg_energy = 40.0;
        stats.avg_age = (step / 100) as f32;
        stats.avg_generation = (step / 5000) as f32;
        stats.diversity_index = 0.5;
        stats.species_counts = match step % 3 {
            0 => vec![alife_common::SpeciesCount { species_id: 1, count: stats.population }],
            1 => vec![
                alife_common::SpeciesCount { species_id: 1, count: 30 },
                alife_common::SpeciesCount { species_id: 2, count: 10 },
            ],
            _ => vec![
                alife_common::SpeciesCount { species_id: 1, count: 30 },
                alife_common::SpeciesCount { species_id: 2, count: 20 },
                alife_common::SpeciesCount { species_id: 3, count: 10 },
            ],
        };
        step_history(&mut h, &stats, &ev);
        assert!(
            h.samples.len() <= 2048,
            "history buffer over-bounded: {} samples at step {}",
            h.samples.len(),
            step
        );
        for s in &h.samples {
            assert!(s.species_top.len() <= 5, "species_top must stay capped at 5");
        }
    }
    eprintln!(
        "[memory_alpha] history: {} steps -> {} samples, interval={}, pending b/d = {}/{} ({} ms)",
        steps,
        h.samples.len(),
        h.interval,
        h.pending_births,
        h.pending_deaths,
        t0.elapsed().as_millis()
    );
    assert!(h.samples.len() <= 2048);
    assert!(h.interval >= 20, "interval never shrinks below default");
    assert_eq!(
        h.pending_births, 0,
        "pending counters reset on every emission step"
    );
    assert_eq!(h.pending_deaths, 0);
}
