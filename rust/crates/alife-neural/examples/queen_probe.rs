//! TEMP diagnostic: decoded queen motor under the region-grounded binding.

use std::path::{Path, PathBuf};

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

fn main() {
    let Some(path) = find_artifact() else { std::process::exit(1) };
    let conn = alife_neural::connectome_binary::load(&path).expect("load");
    let mut adapter = alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(
        std::sync::Arc::clone(&conn), 0,
    ).expect("adapter");
    let rt = adapter.runtime();
    eprintln!("output pools: {:?}", rt.output_pools().map(|p| p.iter().map(|v| v.len()).collect::<Vec<_>>()));
    let ib = rt.input_binding().map(<[usize]>::to_vec);
    eprintln!("input binding len: {:?}", ib.as_ref().map(|b| b.len()));

    use alife_neural::sensory::*;
    // alternating scenes: (food ahead, low energy) <-> (hazard ahead, full)
    let budget = alife_neural::queen_types::NeuralTickBudget { max_ticks_per_advance: 2 };
    let mut prev = [-9.0f32; 5]; // mdx mdy intensity turn feed
    let mut scene_counts = [[0usize; 8]; 2]; // coarse dir histogram per scene
    let mut out_spikes: usize = 0;
    for t in 0..800 {
        let scene_a = (t / 200) % 2 == 0;
        // explicit channel scene, alternating every 200 ticks
        let mut ch = [0.0f32; CHANNEL_COUNT];
        if scene_a {
            ch[CH_FOOD_COS] = 1.0; ch[CH_FOOD_SIN] = 0.0; ch[CH_FOOD_PROX] = 0.9;
            ch[CH_GROUND] = 0.1; ch[CH_QUEEN] = 0.2;
        } else {
            ch[CH_OBSTACLE] = 0.8; ch[CH_LIGHT_COS] = 0.7; ch[CH_GROUND] = 0.8;
        }
        let senses = alife_neural::queen_types::SensoryInputForQueen {
            food_dir: if scene_a { (1.0, 0.0) } else { (0.0, 0.0) },
            food_dist: if scene_a { 2.0 } else { 1e9 },
            heading: t as f32 * 0.001,
            speed: 0.0,
            energy_frac: if scene_a { 0.2 } else { 0.9 },
            hazard: if scene_a { 0.05 } else { 0.8 },
            edge_dist: 1.0,
            signal_sample: if scene_a { 0.2 } else { 0.0 },
            density: 0.1,
            hazard_dir: (0.0, 0.0),
            hazard_rate: 0.0,
        };
        let out = adapter.tick(&senses, &budget);
        let st = adapter.stats();
        out_spikes = st.spikes_last as usize;
        if t % 100 == 0 {
            eprintln!(
                "t={} scene{} md=({:.2},{:.2}) i={:.3} turn={:.3} feed={:.3} emit={:.3} spikes={}",
                t, scene_a as u8, out.move_dir.0, out.move_dir.1, out.move_intensity,
                out.turn, out.feed_seek, out.emit_signal, out_spikes
            );
        }
        // coarse movement-direction histogram (8 bins) per scene
        if out.move_dir.0 != 0.0 || out.move_dir.1 != 0.0 {
            let a = out.move_dir.1.atan2(out.move_dir.0);
            let bin = (((a.rem_euclid(std::f32::consts::TAU)) / (std::f32::consts::TAU / 8.0)) as usize).min(7);
            scene_counts[scene_a as usize][bin] += 1;
        }
        let _ = &mut prev; 
    }
    let total_scene0: usize = scene_counts[0].iter().sum();
    let total_scene1: usize = scene_counts[1].iter().sum();
    eprintln!("\nscene A (food) active-out ticks: {total_scene0}, hist: {:?}", scene_counts[0]);
    eprintln!("scene B (hazard) active-out ticks: {total_scene1}, hist: {:?}", scene_counts[1]);
}
