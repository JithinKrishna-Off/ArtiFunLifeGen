use std::path::{Path, PathBuf};
use alife_core::queen_bridge::MaleCnsQueenBrainFactory;
use alife_core::queen::QueenBrainFactory as _;
use alife_core::SimApi as _;
use alife_core::Simulation;

fn find_artifact() -> Option<PathBuf> {
    let rel = Path::new("appdata").join("malecns166k").join("queen-subnetwork-166k.alifecns");
    std::env::current_dir().ok()
        .map(|d| d.canonicalize().unwrap_or(d))
        .and_then(|p| {
            let mut w = Some(p);
            for _ in 0..6 {
                let Some(x) = w else { break };
                let c = x.join(&rel);
                if c.is_file() { return Some(c) }
                w = x.parent().map(|q| q.to_path_buf());
            }
            None
        })
}

#[test]
fn queen_longevity_probe() {
    let Some(path) = find_artifact() else { return; };
    let sub = alife_neural::connectome_binary::load(&path).unwrap();
    let cfg = alife_common::WorldGenConfig { width: 800, height: 600, cell: 8, ..Default::default() };
    let world = alife_worldgen::generate(&cfg);
    let mut sim = Simulation::new(&alife_common::SimParams::default(), world);
    for _ in 0..40 { sim.step(); }
    let qid = sim.organisms().iter().find(|o| o.alive).map(|o| o.id).unwrap();
    sim.crown_queen(qid);
    let factory = MaleCnsQueenBrainFactory::from_connectome(sub);
    sim.set_queen_brain(Some(factory.build()));
    let mut out = String::new();
    for s in 1..=7200u32 {
        sim.step();
        if s % 600 == 0 {
            let alive = sim.queen_id().is_some();
            let st = sim.stats();
            out.push_str(&format!(
                "step {s} t={:.0}s queen_alive={alive} pop={} queen_brain_ticks={} exec_ns={}\n",
                s as f32 / 60.0,
                st.population,
                sim.queen_brain_ticks_so_far(),
                sim.queen_brain_stats().map(|b| b.exec_ns).unwrap_or(0),
            ));
            if !alive {
                out.push_str(&format!("QUEEN DIED around step {s} (t={:.1}s)\n", s as f32 / 60.0));
                break;
            }
        }
    }
    eprintln!("{out}");
}
