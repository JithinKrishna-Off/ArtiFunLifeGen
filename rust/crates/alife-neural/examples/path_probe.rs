//! TEMP audit: afferent->motor path strength over the real MaleCNS subnetwork.
//! For each of the 22 bound afferent slots: out-degree and total outgoing
//! effective weight; then a driven-vs-idle race: how many spikes occur in the
//! 6 motor cells when the food channels fire vs silence.
use std::path::{Path, PathBuf};

use alife_neural::sensory::{CH_FOOD_COS, CH_FOOD_PROX};

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
    let mut adapter =
        alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(conn, 0).expect("adapter");
    let rt = adapter.runtime_mut();
    let binding: Vec<usize> = rt.input_binding().expect("region binding installed").to_vec();
    let motor_binding = rt.output_cells().copied();
    eprintln!("motor binding (fallback) {motor_binding:?}");
    let pools: Option<Vec<Vec<u32>>> = rt.output_pools().map(|p| p.to_vec());
    eprintln!("pools sizes {:?}", pools.as_ref().map(|p| p.iter().map(|v| v.len()).collect::<Vec<_>>()));
    eprintln!("\n-- afferent slots (out-degree, sum effective weight) --");
    for (ai, &idx) in binding.iter().enumerate() {
        let (ch, k) = (ai / 2, ai % 2);
        let signed = alife_neural::sensory::SIGNED_CHANNELS[ch];
        // effective out-weight sum for THIS neuron's row
        let (rs, re) = (
            rt.connectome().row_offsets[idx] as usize,
            rt.connectome().row_offsets[idx + 1] as usize,
        );
        let mut wsum = 0.0f32;
        for p in rs..re {
            wsum += rt.effective_weight(p, idx as u32);
        }
        eprintln!(
            "slot ch={ch} k={k} signed={signed} neuron={idx:6} out_deg={:4} eff_w={:9.2}",
            re - rs, wsum
        );
    }

    // RACE: drive ONLY the food channels hard for 300 ticks vs all-silence;
    // count motor-cell spikes and (for contrast) whole-net spikes.
    let budget = alife_neural::runtime::TickBudget::default();
    let mut motor_spikes = [0usize; 6];
    let mut food_afferent_spikes = 0usize;
    let mut net_spikes = 0usize;
    let mut chans = [0.0f32; 11];
    chans[CH_FOOD_COS] = 1.0;
    chans[CH_FOOD_PROX] = 1.0;
    for _ in 0..300 {
        let _ = rt.tick_channels(&chans, &budget);
        for &s in rt.spike_ids_last() {
            let u = s as usize;
            if let Some(pools_ok) = pools.as_ref() {
                if let Some(slot) = pools_ok.iter().position(|m| m.contains(&(u as u32))) {
                    motor_spikes[slot] += 1;
                }
            }
            if binding.contains(&u) {
                food_afferent_spikes += 1;
            }
            net_spikes += 1;
        }
    }
    eprintln!("\n-- 300 ticks with FOOD channels fully driven --");
    eprintln!("afferent spikes: {food_afferent_spikes}; net spikes: {net_spikes}");
    eprintln!("motor-pool spikes: {motor_spikes:?}");

    // zeta contrast: same run with the food channels silenced instead
    let budget2 = alife_neural::runtime::TickBudget::default();
    let mut motor_spikes2 = [0usize; 6];
    let mut net_spikes2 = 0usize;
    for _ in 0..300 {
        let _ = rt.tick_channels(&[0.0f32; 11], &budget2);
        for &s in rt.spike_ids_last() {
            let u = s as usize;
            if let Some(pools_ok) = pools.as_ref() {
                if let Some(slot) = pools_ok.iter().position(|m| m.contains(&(u as u32))) {
                    motor_spikes2[slot] += 1;
                }
            }
            net_spikes2 += 1;
        }
    }
    eprintln!("\n-- 300 ticks all-silent (motor-drift baseline) --");
    eprintln!("net spikes: {net_spikes2}");
    eprintln!("motor-pool spikes: {motor_spikes2:?}");
}
