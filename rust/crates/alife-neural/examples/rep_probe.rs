//! TEMP audit A4 #9: REPEATABILITY of scene responses. Two independent runs
//! per scene; SNR = between-scene spread vs run-to-run rep sigma per cell.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use alife_neural::malecns_bins::{import_full, load_package, subnetwork_for_queen, QueenSubnetOptions};
use alife_neural::runtime::{FlyBrainRuntime, TickBudget};
use alife_neural::sensory::Encoder;

fn find_pkg() -> Option<PathBuf> {
    let rel = ["appdata", "malecns166k"].iter().collect::<PathBuf>();
    let mut w = std::env::current_dir().ok().map(|d| d.canonicalize().unwrap_or(d));
    for _ in 0..6 {
        let Some(p) = w else { break };
        let c = p.join(&rel);
        if c.join("manifest.json").is_file() { return Some(c) }
        w = p.parent().map(|q| q.to_path_buf());
    }
    None
}

fn main() {
    let Some(pkg) = find_pkg() else { std::process::exit(1) };
    let loaded = load_package(&pkg).expect("package");
    let full = import_full(loaded).expect("import");
    let (sub, _) = subnetwork_for_queen(&full.connectome, &full.populations, QueenSubnetOptions::default()).expect("sub");
    let pkg_sub_path = pkg.join("queen-subnetwork-166k.alifecns");

    let cls_of = |c: &alife_neural::connectome::Connectome, i: usize| c.region_name(c.node_meta[i].region_idx).map(|s| s.to_string()).unwrap_or_default();
    let n_sub = sub.n();
    let pos_out_score = |c: &alife_neural::connectome::Connectome, i: usize| -> f32 {
        let (rs, re) = (c.row_offsets[i] as usize, c.row_offsets[i + 1] as usize);
        (rs..re).filter(|&p| c.weights[p] > 0.0).map(|p| c.weights[p]).sum::<f32>()
    };
    let top_cells = |c: &alife_neural::connectome::Connectome, want: &dyn Fn(&str) -> bool, k: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..c.n()).filter(|&i| want(&cls_of(c, i))).collect();
        v.sort_by(|&a, &b| pos_out_score(c, b).partial_cmp(&pos_out_score(c, a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        v.truncate(k);
        v
    };
    let body_of = |c: &alife_neural::connectome::Connectome, i: usize| c.node_meta[i].body_id;
    let sugar_bodies: Vec<u64> = full.populations.sugar.iter().map(|&i| body_of(&full.connectome, i as usize)).collect();
    let by_body: std::collections::HashMap<u64, usize> = (0..n_sub).map(|i| (body_of(&sub, i), i)).collect();
    let mut sugar_sorted: Vec<usize> = sugar_bodies.iter().map(|&b| by_body[&b]).collect();
    sugar_sorted.sort_by(|&a, &b| pos_out_score(&sub, b).partial_cmp(&pos_out_score(&sub, a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));

    let vis = top_cells(&sub, &|r: &str| r == "visual_projection", 8);
    let vis2 = top_cells(&sub, &|r: &str| r == "visual_centrifugal" || r == "ol_intrinsic", 8);
    let vnc_s = top_cells(&sub, &|r: &str| r == "vnc_sensory", 4);
    let asc = top_cells(&sub, &|r: &str| r == "sensory_ascending" || r == "ascending_neuron", 4);
    let take2 = |v: &[usize], off: usize, into: &mut Vec<usize>| { into.push(v[off]); into.push(v[off + 1]); };
    let mut seq: Vec<usize> = Vec::new();
    take2(&vis, 0, &mut seq);
    take2(&vis, 2, &mut seq);
    take2(&vis, 4, &mut seq);
    take2(&vis2, 0, &mut seq);
    take2(&vis2, 2, &mut seq);
    take2(&asc, 0, &mut seq);
    take2(&vnc_s, 0, &mut seq);
    take2(&sugar_sorted, 0, &mut seq);
    take2(&sugar_sorted, 2, &mut seq);
    take2(&sugar_sorted, 4, &mut seq);
    take2(&asc, 2, &mut seq);
    let bnd = seq;
    eprintln!("binding: {bnd:?}");

    let cand_desc: Vec<usize> = top_cells(&sub, &|r: &str| r == "descending_neuron", 96);
    let cand_cbm: Vec<usize> = top_cells(&sub, &|r: &str| r == "cb_motor", 6);
    let cand_vis: Vec<usize> = top_cells(&sub, &|r: &str| r == "visual_centrifugal" || r == "visual_projection", 96);

    struct Sc { name: &'static str, chans: [f32; 11] }
    let mk = |name: &'static str, f: Vec<(usize, f32)>| {
        let mut chans = [0.0f32; 11];
        for (c, v) in f { chans[c] = v; }
        Sc { name, chans }
    };
    let scenes = vec![
        mk("rest", vec![]),
        mk("fC+", vec![(8, 1.0), (7, 0.9)]),
        mk("fC-", vec![(8, -1.0), (7, 0.9)]),
        mk("fS+", vec![(9, 1.0), (7, 0.9)]),
        mk("fS-", vec![(9, -1.0), (7, 0.9)]),
        mk("hz", vec![(5, 1.0)]),
        mk("hg+", vec![(4, 1.0)]),
        mk("hg-", vec![(4, -1.0)]),
        mk("allmax", (0..11).map(|c| (c, 1.0)).collect()),
    ];
    let ws = 0.45f32;
    let reps = 2;
    // rates[scene][rep][cell]
    let mut rates: Vec<Vec<Vec<f32>>> = Vec::new();
    for sc in &scenes {
        let mut per_rep = Vec::new();
        for _ in 0..reps {
            let conn = alife_neural::connectome_binary::load_memmap(&pkg_sub_path).expect("mm");
            let mut rt = FlyBrainRuntime::new(conn, 3).expect("rt");
            rt.weight_scale = ws;
            rt.event_current_clamp = 20.0;
            rt.sensory.baseline_hz = 3.0;
            rt.sensory.gain_hz = 60.0;
            rt.sensory.pulse_current = 30.0;
            rt.encoders = vec![Encoder::Rate; 22];
            assert!(rt.set_input_binding(&bnd));
            let budget = TickBudget { max_spikes: 4096, max_synops: 400_000 };
            let mut counts: Vec<u32> = vec![0; rt.connectome().n()];
            let warm = 300;
            for t in 0..warm + 900 {
                let _ = rt.tick_channels(&sc.chans, &budget);
                if t >= warm {
                    for &s in rt.spike_ids_last() { counts[s as usize] += 1; }
                }
            }
            per_rep.push(counts.iter().map(|&c| c as f32 / 900.0 * 1000.0).collect::<Vec<f32>>());
        }
        rates.push(per_rep);
    }
    let _ = repr;

    // per-cell: scene mean (over reps), run-to-run sigma (sqrt of scene-avg rep variance)
    let cells_of = |name: &str| -> &Vec<usize> {
        match name {
            "desc" => &cand_desc,
            "cbm" => &cand_cbm,
            "visual" => &cand_vis,
            _ => unreachable!(),
        }
    };
    for pool_name in ["desc", "cbm", "visual"] {
        let cells = cells_of(pool_name);
        let mut rows: Vec<(f32, f32, f32, usize)> = Vec::new(); // (snr, spread, sig, cell)
        for &c in cells {
            // scene means and rep sigma
            let mut s_means = vec![0.0f32; scenes.len()];
            let mut s_sig = vec![0.0f32; scenes.len()];
            for (si, reps_of_scene) in rates.iter().enumerate() {
                let m: f32 = reps_of_scene.iter().map(|r| r[c]).sum::<f32>() / reps as f32;
                let var = reps_of_scene.iter().map(|r| (r[c] - m).powi(2)).sum::<f32>() / reps as f32;
                s_means[si] = m;
                s_sig[si] = var.max(1.0).sqrt().max(1.0);
            }
            let sig = s_sig.iter().sum::<f32>() / s_sig.len() as f32;
            let r0 = s_means[0];
            let spread = s_means.iter().map(|&m| (m - r0).abs()).fold(0.0f32, f32::max);
            // signed spread with max |scene-rest|
            let mut best = (r0, 0.0f32, 0i32);
            for (si, &m) in s_means.iter().enumerate().skip(1) {
                let d = m - r0;
                if d.abs() > best.1 { best = (d, d.abs(), d.signum() as i32); }
            }
            let snr = best.1 / sig;
            rows.push((snr, best.0, sig, c));
        }
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = rows.iter().take(8).map(|(snr, d, sig, c)| format!("#{c} d={d:.1} sig={sig:.1} snr={snr:.1}")).collect();
        let n_good = rows.iter().filter(|r| r.0 > 2.0).count();
        eprintln!("{pool_name}: {}/{} cells with SNR>2; top: {}", n_good, cells.len(), top.join("  "));
    }
}

#[allow(dead_code)]
fn repr() {}

// placeholder fn reference to keep imports
#[allow(dead_code)]
fn keep(_x: BTreeMap<usize, usize>) {}
