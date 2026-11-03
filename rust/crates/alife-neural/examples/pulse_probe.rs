//! TEMP audit A4 #3: why doesn't the pulse make the afferent cell spike?
//! Membrane trace + pulse_current sweep on one strongly-driven cb_sensory cell.
use std::path::{Path, PathBuf};

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
    let conn = alife_neural::connectome_binary::load(&path).expect("load");
    let n = conn.n();
    // pick drive cell: strongest cb_sensory
    let cls = |i: usize| conn.region_name(conn.node_meta[i].region_idx).map(|r| r.to_string());
    let mut best = 0usize;
    let mut best_out = 0u32;
    for i in 0..n {
        if cls(i).as_deref() != Some("cb_sensory") { continue; }
        let out = conn.row_offsets[i + 1] - conn.row_offsets[i];
        if out > best_out { best_out = out; best = i; }
    }
    eprintln!("drive cell {best} out_deg={best_out}");
    for pulse in [15.0f32, 30.0, 60.0, 120.0] {
        let mut rt = alife_neural::runtime::FlyBrainRuntime::new(
            alife_neural::connectome_binary::load_memmap(&path).expect("mm"),
            0,
        )
        .expect("rt");
        rt.sensory.baseline_hz = 3.0;
        rt.sensory.gain_hz = 60.0;
        rt.sensory.pulse_current = pulse;
        rt.encoders = vec![alife_neural::sensory::Encoder::Rate; 22];
        let b = alife_neural::runtime::TickBudget { max_spikes: 4096, max_synops: 131_072 };
        // binding: channel 8 slot 0 -> `best`; every other slot -> filler cell
        // n-1 (real, valid; its pulses land on one stray indexed cell — the
        // question here is only about `best`'s membrane).
        let mut bnd = vec![n - 1; 22];
        bnd[16] = best;
        assert!(rt.set_input_binding(&bnd));
        let chans = [0.0f32; 11];
        let mut spikes = 0u64;
        let mut vsum = 0f64;
        let mut vmin = 1e9f32;
        let mut vmax = -1e9f32;
        for t in 0..600 {
            let _ = rt.tick_channels(&chans, &b);
            let st = rt.state();
            if st.spiked.contains(&(best as u32)) { spikes += 1; }
            let v = st.v[best];
            vsum += v as f64;
            vmin = vmin.min(v);
            vmax = vmax.max(v);
            if t < 60 && pulse == 15.0 {
                eprintln!("t={t:3} v={v:7.2} refr={} nspked={}", st.refractory[best], st.spiked.len());
            }
        }
        eprintln!(
            "pulse={pulse:5.1}: spikes={spikes} v_avg={:.2} v_min={vmin:.2} v_max={vmax:.2}",
            vsum / 600.0
        );
    }
}
