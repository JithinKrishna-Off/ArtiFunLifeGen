use alife_core::SimApi;
use alife_core::Simulation;
use alife_common::{SimParams, WorldGenConfig};

#[test]
fn heavy_stress() {
    let world = alife_worldgen::generate(&WorldGenConfig::default());
    let params = SimParams::default();
    let mut sim = Simulation::new(&params, world);
    sim.spawn_organisms(1200);
    // warmup
    for _ in 0..100 {
        sim.advance();
    }
    let t0 = std::time::Instant::now();
    let n = 600;
    for _ in 0..n {
        sim.advance();
    }
    let total = t0.elapsed();
    eprintln!(
        "pop={} foods={} per_step={:?}",
        sim.stats().population,
        sim.foods().len(),
        total / n
    );
}
