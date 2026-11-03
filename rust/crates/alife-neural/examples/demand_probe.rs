//! TEMP audit A4 #10: synops/spike demand at fp operating point (ws .45,
//! clamp 20, binding v2), with different budgets; warm steady state.
use std::path::{Path, PathBuf};

use alife_neural::runtime::{FlyBrainRuntime, TickBudget};
use alife_neural::sensory::Encoder;

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

fn main() {
    let Some(path) = find_artifact() else { std::process::exit(1) };
    for cap in [65_536usize, 131_072, 262_144, 400_000] {
        let conn = alife_neural::connectome_binary::load_memmap(&path).expect("mm");
        let mut rt = FlyBrainRuntime::new(conn, 3).expect("rt");
        rt.weight_scale = 0.45;
        rt.event_current_clamp = 20.0;
        rt.sensory.baseline_hz = 3.0;
        rt.sensory.gain_hz = 60.0;
        rt.sensory.pulse_current = 30.0;
        rt.encoders = vec![Encoder::Rate; 22];
        let budget = TickBudget { max_spikes: 4096, max_synops: cap };
        let chans = [0.0f32; 11];
        let mut sp = 0u64;
        let mut dsyn = 0u64;
        let mut dsp = 0u64;
        for _ in 0..500 {
            let _ = rt.tick_channels(&chans, &budget);
            let st = rt.last_stats();
            sp += st.spikes;
            dsyn += st.dropped_synops;
            dsp += st.dropped_spikes;
        }
        eprintln!(
            "cap={cap:7}: spikes/500={sp} dropped_synops/500={dsyn} dropped_spikes/500={dsp}"
        );
        dsyn = 0; dsp = 0; sp = 0;
    }
}
