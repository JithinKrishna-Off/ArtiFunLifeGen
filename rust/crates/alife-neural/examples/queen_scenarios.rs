//! TEMP validation: the Queen's MaleCNS brain under 7 CONTROLLED scenes.
//! For each scenario: settle with the scene constant, then record the decoded
//! motor outputs and the kinematics of a queen puck driven ONLY by the brain
//! (no scripted behaviour). Cross-scenario comparison = stimulus dependence.
use std::path::{Path, PathBuf};

use alife_neural::queen_types::{NeuralTickBudget, SensoryInputForQueen};

fn find_artifact() -> Option<PathBuf> {
    let rel = Path::new("appdata").join("malecns166k").join("queen-subnetwork-166k.alifecns");
    let mut w = std::env::current_dir().ok().map(|d| d.canonicalize().unwrap_or(d));
    for _ in 0..6 {
        let Some(p) = w else { break };
        let c = p.join(&rel);
        if c.is_file() { return Some(c) }
        w = p.parent().map(|q| q.to_path_buf());
    }
    None
}

const TURN_RATE: f32 = 1.5; // rad/s at |turn| = 1 (body-level, documented)
const CRUISE: f32 = 12.0; // units/s at intensity 1 (body-level, documented)
const ADVANCE_S: f32 = 0.0333;

fn main() {
    let Some(path) = find_artifact() else {
        eprintln!("artifact missing — build it first (malecns_bins)");
        std::process::exit(1);
    };
    let conn = std::sync::Arc::new(alife_neural::connectome_binary::load(&path).expect("load"));

    // A: food near, no threat; B: food far, low energy; C: food near + threat;
    // D: threat from left; E: threat from right; F: obstacle ahead; G: empty.
    let scenarios: Vec<(&str, SensoryInputForQueen)> = vec![
        ("A food-near, no threat", SensoryInputForQueen {
            food_dir: (1.0, 0.0), food_dist: 2.0, energy_frac: 0.5, ..Default::default() }),
        ("B food-far, low energy", SensoryInputForQueen {
            food_dir: (1.0, 0.0), food_dist: 1e8, energy_frac: 0.08, ..Default::default() }),
        ("C food-near + threat fwd", SensoryInputForQueen {
            food_dir: (1.0, 0.0), food_dist: 2.0, energy_frac: 0.5,
            hazard: 0.8, hazard_dir: (1.0, 0.0), hazard_rate: 0.6, ..Default::default() }),
        ("D threat-from-left", SensoryInputForQueen {
            hazard: 0.8, hazard_dir: (0.0, -1.0), hazard_rate: 0.6,
            energy_frac: 0.5, ..Default::default() }),
        ("E threat-from-right", SensoryInputForQueen {
            hazard: 0.8, hazard_dir: (0.0, 1.0), hazard_rate: 0.6,
            energy_frac: 0.5, ..Default::default() }),
        ("F obstacle-ahead", SensoryInputForQueen {
            hazard: 1.0, hazard_dir: (-1.0, 0.0), hazard_rate: 0.8,
            energy_frac: 0.5, ..Default::default() }),
        ("G no-stimuli", SensoryInputForQueen {
            energy_frac: 0.5, ..Default::default() }),
    ];

    let budget = NeuralTickBudget { max_ticks_per_advance: 2 };
    println!(
        "{:<25} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} | {:>7} {:>7} {:>5} {:>5}",
        "scenario", "md.x", "md.y", "inten", "turn", "feed", "emit", "pos.x", "pos.y", "hdg", "enrg"
    );

    let mut rows: Vec<(String, f32, f32, f32, f32, f32, f32, String)> = Vec::new();
    for (name, senses) in &scenarios {
        // FRESH adapter per scenario: fixed tuned binding, clean EMA state
        // (~0.4 s probe cost per scenario, deterministic)
        let mut adapter = alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(
            std::sync::Arc::clone(&conn), 0,
        )
        .expect("adapter over real connectome");
        let s = adapter.stats();

        // settle 200 advances with the scene held constant
        for _ in 0..200 {
            let _ = adapter.tick(senses, &budget);
        }
        // measurement phase: brain-driven kinematics, 100 advances
        let (mut x, mut y, mut heading, mut energy) = (0.0f32, 0.0f32, 0.0f32, 0.8f32);
        let mut sums = [0.0f32; 6]; // mdx mdy intensity turn feed emit
        const N: u32 = 100;
        for _ in 0..N {
            let out = adapter.tick(senses, &budget);
            sums[0] += out.move_dir.0;
            sums[1] += out.move_dir.1;
            sums[2] += out.move_intensity;
            sums[3] += out.turn;
            sums[4] += out.feed_seek;
            sums[5] += out.emit_signal;
            heading += out.turn * TURN_RATE * ADVANCE_S;
            let spd = CRUISE * out.move_intensity;
            x += out.move_dir.0 * spd * ADVANCE_S;
            y += out.move_dir.1 * spd * ADVANCE_S;
            // feeding feedback: only reachable food + active feed_seek gains
            if senses.food_dist < 6.0 && out.feed_seek > 0.1 {
                energy = (energy + 0.02 * out.feed_seek).min(1.0);
            }
            energy = (energy - 0.01).max(0.0);
        }
        let st = adapter.stats();
        rows.push((
            name.to_string(),
            sums[0] / N as f32, sums[1] / N as f32,
            sums[2] / N as f32, sums[3] / N as f32,
            sums[4] / N as f32, sums[5] / N as f32,
            format!("({:5.1},{:5.1}) hdg{:5.2} spk/t{:4.0}", x, y, heading, st.spikes_last as f32),
        ));
        let _ = s;
    }
    for (name, mdx, mdy, inten, turn, feed, emit, kin) in &rows {
        println!(
            "{:<25} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} | {}",
            name, mdx, mdy, inten, turn, feed, emit, kin
        );
    }
}
