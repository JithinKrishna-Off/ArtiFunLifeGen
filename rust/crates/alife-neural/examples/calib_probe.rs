//! TEMP audit: neural operating point of the MaleCNS queen subnetwork.
//! Reports per-population firing rates with no sensory input (baseline G) and
//! under the adapter's real sensory scenes, so the decoder calibration and
//! any operating-point decisions are made on measure, not hope.
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
    let conn = std::sync::Arc::new(alife_neural::connectome_binary::load(&path).expect("load"));

    // population view over region labels
    let n = conn.n();
    let mut region_of = vec![None; n];
    let mut pop_index: std::collections::BTreeMap<String, usize> = Default::default();
    let mut pops: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let name = conn.region_name(conn.node_meta[i].region_idx).unwrap_or("").to_string();
        let k = *pop_index.entry(name.clone()).or_insert_with(|| {
            pops.push(Vec::new()); pops.len() - 1
        });
        pops[k].push(i);
        region_of[i] = Some(k);
    }
    eprintln!("populations: {}", pops.len());

    // scenes: G = nothing; A = food near; B = threat; F = obstacle
    let scenes: [(&str, [f32; 11]); 4] = [
        ("G", [0.0; 11]),
        ("A-food", {
            let mut c = [0.0; 11];
            c[7] = 0.9; c[8] = 1.0; c[6] = 0.2; c[10] = 0.2; c
        }),
        ("B-threat", {
            let mut c = [0.0; 11];
            c[5] = 0.8; c[3] = 0.7; c[6] = 0.8; c
        }),
        ("F-obs", {
            let mut c = [0.0; 11];
            c[5] = 1.0; c[1] = 1.0; c
        }),
    ];

    for (label, chans) in scenes {
        // fresh runtime per scene: KNOWN intrinsic state (no carry-over)
        let mut rt = alife_neural::runtime::FlyBrainRuntime::new(std::sync::Arc::clone(&conn), 0).unwrap();
        rt.select_region(|_| false); // start ungated anyway
        rt.clear_selectivity();

        // settle 300 ticks with the scene constant
        for _ in 0..300 {
            let _ = rt.tick_channels(&chans, &Default::default());
        }
        // measure 200 ticks: per-population spike totals
        let mut pop_sp = vec![0.0f32; pops.len()];
        for _ in 0..200 {
            let _ = rt.tick_channels(&chans, &Default::default());
            for &s in rt.spike_ids_last() {
                if let Some(k) = region_of[s as usize] {
                    pop_sp[k] += 1.0;
                }
            }
        }
        // per-population report (top by rate, plus descending/motor detail)
        let meas_s = 200.0 * 0.001;
        let mut rows: Vec<(String, usize, f32)> = pops.iter().enumerate()
            .map(|(k, cells)| (format!("pop{k}"), cells.len(),
                if cells.is_empty() { 0.0 } else { pop_sp[k] / cells.len() as f32 / meas_s }))
            .collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        eprintln!("\n=== scene {label} — {scenes:?} [Hz mean per population, rate + size] ===", scenes = label);
        for (name, size, hz) in rows.iter().take(12) {
            eprintln!("{name:26} size={size:6}  mean rate = {hz:7.2} Hz");
        }
        // label candidates in descending/motor populations
        for (k, cells) in pops.iter().enumerate() {
            let nm = pop_index.iter().find(|(_, &v)| v == k).unwrap().0.clone();
            if nm.contains("descending") || nm.contains("motor") || nm.contains("ascending") {
                eprintln!("  pop{nm:24} size={:6} mean={:.2} Hz", cells.len(), pop_sp[k] / cells.len() as f32 / meas_s);
            }
        }
        // overall
        let total: f32 = pop_sp.iter().sum();
        eprintln!("total spikes/200 ticks = {:.0} (mean/all-cells = {:.2} Hz)",
            total, total / n as f32 / meas_s);
        let _ = &region_of;
    }
}
