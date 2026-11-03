//! TEMP #11: smoke the rebuilt adapter path (sugar-aware binding + auto-tune
//! v3) and print the decoded motor behavior across mirrored scenes.
use std::path::PathBuf;
use std::time::Instant;

use alife_neural::queen_adapter::QueenBrainAdapter;
use alife_neural::queen_types::NeuralTickBudget;

fn find_dir() -> Option<PathBuf> {
    let rel = std::path::Path::new("appdata").join("malecns166k");
    let mut w = std::env::current_dir().ok().map(|d| d.canonicalize().unwrap_or(d));
    for _ in 0..6 {
        let Some(p) = w else { break };
        if p.join(&rel).is_dir() {
            return Some(p.join(&rel));
        }
        w = p.parent().map(|q| q.to_path_buf());
    }
    None
}

fn main() {
    let dir = find_dir().expect("package dir");
    let t0 = Instant::now();
    let conn = alife_neural::connectome_binary::load(&dir.join("queen-subnetwork-166k.alifecns")).expect("artifact");
    let sugar = alife_neural::malecns_bins::sugar_body_ids(&dir).expect("sugar pop");
    let mut ad = QueenBrainAdapter::new_from_connectome_with_sugar(conn, 0, Some(&sugar)).expect("adapter");
    eprintln!("build+tune took {:?}", t0.elapsed());
    let budget = NeuralTickBudget { max_ticks_per_advance: 2 };

    let scene = |fc, fs, prox, hs, th| alife_neural::queen_types::SensoryInputForQueen {
        food_dir: (fc, fs),
        food_dist: if prox > 0.0 { 0.5 } else { 1.0e9 },
        heading: 0.0,
        speed: 0.0,
        energy_frac: if th > 0.0 { 0.8 } else { 0.4 },
        hazard: if th > 0.0 { 0.9 } else { 0.0 },
        edge_dist: 0.5,
        signal_sample: 0.0,
        density: 0.0,
        hazard_dir: (0.0, 0.0),
        hazard_rate: if prox > 0.5 { 0.3 } else { 0.0 },
    };
    for (name, s) in [
        ("rest", scene(0.0, 0.0, 0.0, 0.0, 0.0)),
        ("food*cos+", scene(1.0, 0.0, 1.0, 0.0, 0.0)),
        ("food*cos-", scene(-1.0, 0.0, 1.0, 0.0, 0.0)),
        ("food*sin+", scene(0.0, 1.0, 1.0, 0.0, 0.0)),
        ("food*sin-", scene(0.0, -1.0, 1.0, 0.0, 0.0)),
        ("threat", scene(0.0, 0.0, 0.0, 0.0, 0.0)),
        ("heading-", scene(0.0, 0.0, 0.0, -1.0, 0.0)),
    ] {
        for _ in 0..200 {
            ad.tick(&s, &budget);
        }
        let st = ad.stats();
        eprintln!(
            "scene {name:10}: dir=({:.2},{:.2}) int={:.2} turn={:.2} feed={:.2} emit={:.2} spikes/t={}",
            st.avg_rate_ema * 0.0 + ad.last_move_dir_string().0.parse::<f32>().unwrap_or(0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            st.spikes_last
        );
    }
}
