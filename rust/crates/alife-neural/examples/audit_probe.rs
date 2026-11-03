//! TEMP audit A4: structural capability of the queen subnetwork artifact.
//! 1. per-class out/in-degree stats inside the artifact
//! 2. synops demand vs budget (default vs unbounded), output-cell response
use std::path::{Path, PathBuf};

use alife_neural::sensory::CHANNEL_COUNT;
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
    let conn = alife_neural::connectome_binary::load(&path).expect("load");
    let n = conn.n();
    eprintln!("n={} edges={}", n, conn.edge_count());

    // classify
    let mut in_deg = vec![0u32; n];
    for i in 0..n {
        let (rs, re) = (conn.row_offsets[i] as usize, conn.row_offsets[i + 1] as usize);
        for p in rs..re {
            in_deg[conn.col_idx[p] as usize] += 1;
        }
    }
    let mut by: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for i in 0..n {
        if let Some(r) = conn.region_name(conn.node_meta[i].region_idx) {
            by.entry(r.to_string()).or_default().push(i);
        }
    }
    eprintln!(
        "\n{:<22} {:>7} {:>7} {:>7} {:>8} {:>8} {:>10}",
        "class", "cells", "out>0", "in>0", "outP>0", "inP>0", "avgInDeg"
    );
    for (label, cells) in by {
        let member: std::collections::HashSet<usize> = cells.iter().copied().collect();
        let (mut out_gt0, mut in_gt0, mut out_pos) = (0usize, 0usize, 0usize);
        let mut has_inpos = vec![false; n];
        for i in 0..n {
            if !member.contains(&i) { continue; }
            let (rs, re) = (conn.row_offsets[i] as usize, conn.row_offsets[i + 1] as usize);
            for p in rs..re {
                if conn.weights[p] > 0.0 {
                    has_inpos[i] = true;
                    break;
                }
            }
        }
        for &c in &cells {
            let (rs, re) = (conn.row_offsets[c] as usize, conn.row_offsets[c + 1] as usize);
            if re > rs { out_gt0 += 1; }
            for p in rs..re {
                if conn.weights[p] > 0.0 { out_pos += 1; break; }
            }
            if in_deg[c] > 0 { in_gt0 += 1; }
        }
        let in_pos = has_inpos.iter().filter(|&&b| b).count();
        let avg_in = cells.iter().map(|&c| in_deg[c] as f64).sum::<f64>() / cells.len().max(1) as f64;
        eprintln!(
            "{:<22} {:>7} {:>7} {:>7} {:>8} {:>8} {:>10.1}",
            label,
            cells.len(),
            out_gt0,
            in_gt0,
            out_pos,
            in_pos,
            avg_in
        );
    }

    // ---- runtime: budget comparison under drive
    let scenes: Vec<(&str, [f32; CHANNEL_COUNT])> = vec![
        ("silent   ", [0.0; CHANNEL_COUNT]),
        ("foodC+pro", scene(|c| c == 8 || c == 7)),
        ("foodS1   ", scene(|c| c == 9)),
        ("allmax   ", scene(|_| true)),
    ];
    for budget_kind in [false, true] {
        eprintln!("\n== budget {} ==", if budget_kind { "unbounded" } else { "default" });
        for (name, chans) in &scenes {
            let conn = alife_neural::connectome_binary::load_memmap(&path).expect("mm");
            let mut ad = alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(
                conn, 0,
            )
            .expect("adapter");
            {
                let rt = ad.runtime_mut();
                rt.sensory.baseline_hz = 3.0;
                rt.sensory.gain_hz = 60.0;
            }
            let b = if budget_kind { TickBudget::unbounded() } else { TickBudget::default() };
            let pools: Option<Vec<Vec<u32>>> = ad.runtime().output_pools().map(|p| p.to_vec());
            let cells: Option<[usize; 6]> = ad.runtime().output_cells().copied();
            let (mut spikes, mut drops) = (0f64, 0f64);
            let mut spikee = [0u64; 6];
            for _ in 0..300 {
                let _ = ad.runtime_mut().tick_channels(&chans, &b);
                let st = ad.runtime().last_stats();
                spikes += st.spikes as f64;
                drops += (st.dropped_synops + st.dropped_spikes) as f64;
                for &s in ad.runtime().spike_ids_last() {
                    let u = s as usize;
                    if let Some(pp) = &pools {
                        if let Some(slot) = pp.iter().position(|m| m.contains(&(u as u32))) {
                            spikee[slot] += 1;
                        }
                    } else if let Some(cc) = &cells {
                        if let Some(slot) = cc.iter().position(|&c| c == u) {
                            spikee[slot] += 1;
                        }
                    }
                }
            }
            eprintln!("{name} spikes/300={spikes} drops={drops:.0} out_spikes={spikee:?}");
        }
    }
}

fn scene(f: impl Fn(usize) -> bool) -> [f32; 11] {
    let mut c = [0.0f32; 11];
    for i in 0..11 {
        if f(i) { c[i] = 1.0; }
    }
    c
}
