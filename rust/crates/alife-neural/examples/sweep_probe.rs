//! TEMP audit A4 #4: weight_scale x pulse_current sweep on the real artifact.
//! Metrics: mean spikes/tick (budget-independent), afferent response to a
//! driven channel vs silent, postsyn transmission, runaway check.
use std::path::{Path, PathBuf};

use alife_neural::runtime::{FlyBrainRuntime, TickBudget};

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

fn build(path: &Path, ws: f32, pulse: f32) -> FlyBrainRuntime {
    let conn = alife_neural::connectome_binary::load_memmap(path).expect("mm");
    let mut rt = FlyBrainRuntime::new(conn, 0).expect("rt");
    rt.weight_scale = ws;
    rt.sensory.baseline_hz = 3.0;
    rt.sensory.gain_hz = 60.0;
    rt.sensory.pulse_current = pulse;
    rt.encoders = vec![alife_neural::sensory::Encoder::Rate; 22];
    rt
}

fn main() {
    let Some(path) = find_artifact() else { std::process::exit(1) };
    let b_un = TickBudget::unbounded();
    let b_default = TickBudget::default();

    for ws in [0.15f32, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0] {
        for pulse in [15.0f32, 30.0] {
            // silent run
            let mut rt = build(&path, ws, pulse);
            {
                let n = rt.connectome().n();
                let bnd = vec![0usize; 22];
                let mut m = rt.connectome().clone();
                let _ = m;
                let _ = n;
                // contiguous default binding: first 22 cells get the pulses
            }
            let mut spikes_s = 0u64;
            for _ in 0..400 {
                let _ = rt.tick_channels(&[0.0f32; 11], &b_un);
                spikes_s += rt.last_stats().spikes;
            }
            let mean_s = spikes_s as f64 / 400.0;
            let v_sig = {
                let st = rt.state();
                st.v.iter().map(|&x| (x.abs() as f64)).fold(0.0f64, f64::max)
            };

            // driven run: channel 8 (food cos) fully on
            let mut rt = build(&path, ws, pulse);
            let mut chans = [0.0f32; 11];
            chans[8] = 1.0;
            let mut spikes_d = 0u64;
            for _ in 0..400 {
                let _ = rt.tick_channels(&chans, &b_un);
                spikes_d += rt.last_stats().spikes;
            }
            let mean_d = spikes_d as f64 / 400.0;

            // budgeted identical-experience check: does default budget still clip?
            let mut rt = build(&path, ws, pulse);
            let mut drops = 0u64;
            for _ in 0..100 {
                let _ = rt.tick_channels(&chans, &b_default);
                drops += rt.last_stats().dropped_synops + rt.last_stats().dropped_spikes;
            }

            eprintln!(
                "ws={ws:5.3} pulse={pulse:4.1}: silent_sp/t={mean_s:8.1} driven_sp/t={mean_d:8.1} max|v|={v_sig:6.0} drops100={drops}"
            );
        }
    }
}
