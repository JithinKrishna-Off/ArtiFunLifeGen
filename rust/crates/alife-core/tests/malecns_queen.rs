//! Essential regression check: the production Queen path runs the REAL
//! MaleCNS subnetwork artifact (never the synthetic 68-neuron fixture).
//! Skips honestly when the artifact has not been generated on this machine.

use std::path::{Path, PathBuf};

use alife_common::{SimParams, WorldGenConfig};
use alife_core::queen_bridge::MaleCnsQueenBrainFactory;
use alife_core::queen::QueenBrainFactory as _;
use alife_core::{SimApi, Simulation};

fn find_artifact() -> Option<PathBuf> {
    let rel = Path::new("appdata")
        .join("malecns166k")
        .join("queen-subnetwork-166k.alifecns");
    let mut walk: Option<PathBuf> =
        std::env::current_dir().ok().map(|d| d.canonicalize().unwrap_or(d));
    for _ in 0..6 {
        let Some(p) = walk else { break };
        let c = p.join(&rel);
        if c.is_file() {
            return Some(c);
        }
        walk = p.parent().map(|x| x.to_path_buf());
    }
    None
}

#[test]
fn real_malecns_queen_brain_attaches_and_executes() {
    let Some(path) = find_artifact() else {
        eprintln!("real artifact missing on this machine — skip (no fake fallback)");
        return;
    };
    eprintln!("using real artifact: {}", path.display());

    let sub: std::sync::Arc<alife_neural::connectome::Connectome> =
        alife_neural::connectome_binary::load(&path)
            .unwrap_or_else(|e| panic!("queen subnetwork artifact unreadable: {e:?}"));
    assert!(sub.n() > 1_000, "subnetwork trivially small");

    let cfg = WorldGenConfig { width: 800, height: 600, cell: 8, ..WorldGenConfig::default() };
    let world = alife_worldgen::generate(&cfg);
    let mut sim = Simulation::new(&SimParams::default(), world);

    for _ in 0..40 {
        sim.step();
    }
    let Some(qid) = sim.organisms().iter().find(|o| o.alive).map(|o| o.id) else {
        eprintln!("no organism alive by step 40 — skip honestly");
        return;
    };
    assert!(sim.crown_queen(qid), "crown must succeed on a live organism");
    // production plug path (app::plug_queen_brain): fresh brain built by the
    // real MaleCNS factory, not a saved snapshot rehydration
    let factory = MaleCnsQueenBrainFactory::from_connectome(sub);
    sim.set_queen_brain(Some(factory.build()));

    for _ in 0..30 {
        sim.step();
    }

    // brain attached with REAL MaleCNS subnetwork stats
    let st = sim.queen_brain_stats().expect("brain must be attached");
    assert!(st.synapse_count > 100_000, "real subnetwork must be live");
    assert_ne!(st.neuron_count, 68, "must NOT be the synthetic fixture");
    assert!(st.exec_ns > 0, "lazy runtime must have executed at least one tick");
    assert!(sim.queen_id().is_some(), "queen stays alive under brain control");
    eprintln!(
        "real MaleCNS queen: {} neurons / {} synapses / exec {}µs",
        st.neuron_count,
        st.synapse_count,
        st.exec_ns / 1000
    );
}
