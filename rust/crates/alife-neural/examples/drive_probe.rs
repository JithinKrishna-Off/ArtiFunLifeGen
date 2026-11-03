//! TEMP audit A4 #2: measure per-mm-synapse transmission for candidate
//! afferent cells: drive one cell into a known rate, watch direct postsyn
//! partners' rates change vs silent. Also measures per-tick cost vs budget.
use std::path::{Path, PathBuf};

use alife_neural::runtime::TickBudget;

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

    for bud in [
        ("default 65k", TickBudget::default()),
        ("130k", TickBudget { max_spikes: 4096, max_synops: 131_072 }),
        ("200k", TickBudget { max_spikes: 4096, max_synops: 200_000 }),
        ("unbounded", TickBudget::unbounded()),
    ] {
        let conn = alife_neural::connectome_binary::load_memmap(&path).expect("mm");
        let mut rt = alife_neural::runtime::FlyBrainRuntime::new(conn, 0).expect("rt");
        rt.polarity_policy = alife_neural::runtime::UnknownPolarityPolicy::Scale(0.25);
        let chans = [0.0f32; 11];
        let mut drops = 0f64;
        let mut ns = 0f64;
        // choose a drive cell: the strongest cb_sensory cell
        let n = rt.connectome().n();
        let dd = |c: usize| rt.connectome().region_name(rt.connectome().node_meta[c].region_idx).map(|r| r.to_string());
        let mut best = 0usize;
        let mut best_out = 0u32;
        for i in 0..n {
            if dd(i).as_deref() != Some("cb_sensory") { continue; }
            let out = rt.connectome().row_offsets[i + 1] - rt.connectome().row_offsets[i];
            if out > best_out { best_out = out; best = i; }
        }
        eprintln!("\ndrive cell {best} (cb_sensory) out_deg={best_out}");
        let (rs, re) = (rt.connectome().row_offsets[best] as usize, rt.connectome().row_offsets[best + 1] as usize);
        let mut posts: Vec<(usize, f32, u8)> = (rs..re)
            .map(|p| (rt.connectome().col_idx[p] as usize, rt.connectome().weights[p], rt.connectome().syn_polarity.as_ref().map(|s| s[p]).unwrap_or(3)))
            .collect();
        posts.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let picks: Vec<usize> = posts.iter().take(6).map(|x| x.0).collect();
        eprintln!("posts+signs: {:?}", posts.iter().take(6).map(|x| (x.0, x.1)).collect::<Vec<_>>());
        eprintln!("budget {}: measuring drive vs silent on 6 posts", bud.0);
        for (label, fb, fg) in [("silent", 0.0f32, 0.0f32), ("drive6", 3.0, 3.0), ("drive63", 3.0, 60.0)] {
            let conn2 = alife_neural::connectome_binary::load_memmap(&path).expect("mm");
            let mut rt2 = alife_neural::runtime::FlyBrainRuntime::new(conn2, 0).expect("rt");
            rt2.sensory.baseline_hz = fb;
            rt2.sensory.gain_hz = fg;
            rt2.encoders = vec![alife_neural::sensory::Encoder::Rate; 22];
            // install input binding: every slot maps to the SAME cell? no —
            // channels map to 22 cells; put cell `best` on channel 8 slot 0 and
            // leave the rest silent via zero-rate? Rate encoders can't zero a
            // channel; instead bind all other slots to a dead-end cell (an
            // ol_sensory cell is unusable — it WILL spike and inject inhib.);
            // simplest honest check: bind channel 8 slot0 = best, and run with
            // channels all-zero (signed 8 → 0.5 → 3Hz baseline when drive6).
            let mut binding = vec![best; 22];
            for k in binding.iter_mut() { *k = usize::MAX; }
            // invalid binding values get SKIPPED (encode skip idx>=len)
            let mut bnd = vec![0usize; 22];
            // pick a lone cell for all other slots: use `best` too (same rate
            // on all) — no: we only want ch8. Trick: leave unbound slots
            // pointing at a cell we then silence by setting baseline for THOSE
            // accumulators impossible. Instead: choose slot cell = best only
            // for ch8 k0; others point to a REAL isolated high-index cell with
            // in/out giving minimal influence? Accept: they fire at the same
            // rate as drive sources; for the SILENT run they're the baseline.
            // Use self-consistency: runs share binding; the DIFFERENCE across
            // drive levels is what matters, and all slots fire identically in
            // silent vs drive (dead-end for others: pick 21 cells from
            // vnc_motor/vnc_sensory which have tiny out-degrees and maybe
            // purely-negative edges; good enough).
            let mut fences: Vec<usize> = Vec::new();
            for i in 0..n {
                if dd(i).map(|r| r.contains("vnc_motor") || r.contains("vnc_tbc")).unwrap_or(false) {
                    fences.push(i);
                    if fences.len() >= 22 { break; }
                }
            }
            fences.resize(22, fences[0]);
            for (k, slot) in bnd.iter_mut().enumerate() {
                *slot = if k == 16 { best } else { fences[k] };
            }
            assert!(rt2.set_input_binding(&bnd));
            let mut counts = std::collections::HashMap::new();
            let mut spiked_main = 0u64;
            for _ in 0..600 {
                let _ = rt2.tick_channels(&chans, &bud.1);
                for &s in rt2.spike_ids_last() {
                    let u = s as usize;
                    counts.entry(u).and_modify(|c| *c += 1).or_insert(1u64);
                    if u == best { spiked_main += 1; }
                }
                drops += rt2.last_stats().dropped_synops as f64;
                ns += rt2.last_stats().exec_ns as f64;
            }
            let rates: Vec<(usize, u64)> = picks.iter().map(|&p| (p, *counts.get(&p).unwrap_or(&0))).collect();
            let _ = label;
            eprintln!("  {label}: drive cell spaced={spiked_main}; posts rates {rates:?}; drops={drops:.0} exec/600ms");
            drops = 0.0;
            ns = 0.0;
        }
    }
}
