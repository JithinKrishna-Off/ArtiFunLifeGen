//! TEMP audit A4 #8: per-cell scene-response SNR at calibrated operating
//! points (binding v2, event clamp, pulse 30). Decides the auto-tune redesign.
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

    let cls_of = |c: &alife_neural::connectome::Connectome, i: usize| {
        c.region_name(c.node_meta[i].region_idx).map(|s| s.to_string()).unwrap_or_default()
    };
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

    let vis = top_cells(&sub, &|r| r == "visual_projection", 6);
    let vis2 = top_cells(&sub, &|r| r == "visual_centrifugal" || r == "ol_intrinsic", 6);
    let vnc_s = top_cells(&sub, &|r| r == "vnc_sensory", 4);
    let asc = top_cells(&sub, &|r: &str| {
        r == "sensory_ascending" || r == "ascending_neuron"
    }, 4);
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
    let mut bnd = seq;
    // cells_per_channel 2 → raise to 4 for population coding? keep 2 first.
    let want_per_channel = 4usize;
    if want_per_channel > 2 {
        // duplicate: builder rebuild for 4 (write bnd manually below)
    }
    let want_per_channel_used = 2usize;
    let _ = want_per_channel;
    let _ = want_per_channel_used;
    eprintln!("binding: {bnd:?}");

    // all candidate pools for the decode stage
    let pool_names = vec![
        "descending_neuron", "cb_motor", "vnc_motor", "sensory_ascending", "efferent_ascending",
        "ascending_neuron", "visual_centrifugal", "visual_projection", "cb_endocrine",
    ];
    let mut candidates: BTreeMap<String, Vec<usize>> = Default::default();
    for name in &pool_names {
        candidates.insert(name.to_string(), top_cells(&sub, &|r: &str| r == *name, 96));
    }

    struct Sc { name: &'static str, chans: [f32; 11] }
    let mk = |name: &'static str, f: Vec<(usize, f32)>| {
        let mut chans = [0.0f32; 11];
        for (c, v) in f { chans[c] = v; }
        Sc { name, chans }
    };
    let scenes = vec![
        mk("rest ", vec![]),
        mk("f+   ", vec![(8, 1.0), (7, 0.9)]),
        mk("f-   ", vec![(8, -1.0), (7, 0.9)]),
        mk("fS+  ", vec![(9, 1.0), (7, 0.9)]),
        mk("fS-  ", vec![(9, -1.0), (7, 0.9)]),
        mk("hzc+ ", vec![(5, 1.0)]),
        mk("hg+  ", vec![(4, 1.0)]),
        mk("hg-  ", vec![(4, -1.0)]),
    ];

    for ws in [0.45f32, 0.6] {
        eprintln!("\n===== ws={ws} clamp=20 pulse=30 window 3..63 =====");
        let rates_per_scene: Vec<(String, Vec<f32>, f64, f32)> = scenes.iter().map(|sc| {
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
            let mut total = 0u64;
            let warm = 300;
            for t in 0..warm + 600 {
                let _ = rt.tick_channels(&sc.chans, &budget);
                if t >= warm {
                    for &s in rt.spike_ids_last() { counts[s as usize] += 1; total += 1; }
                }
            }
            let rates: Vec<f32> = counts.iter().map(|&c| c as f32 / 600.0 * 1000.0).collect();
            (sc.name.to_string(), rates, total as f64 / 600.0, rt.state().v.iter().map(|x| x.abs()).fold(0.0f32, f32::max))
        }).collect();
        for (name, _, tot, vabs) in &rates_per_scene {
            eprintln!("scene {name}: sea={tot:6.1}/t |v|max={vabs:.0}");
        }
        // per-pool stats across scenes
        let rest_rates = &rates_per_scene[0].1;
        for (name, cells) in &candidates {
            // pooled + SNR per cell: spread max scene diff, noise sigma within-rest
            let mut cell_rows: Vec<(usize, f32, f32)> = Vec::new();
            for &c in cells {
                let r0 = rest_rates[c];
                let mut spread = 0.0f32;
                for (_, r_scene, _, _) in &rates_per_scene {
                    spread = spread.max((r_scene[c] - r0).abs());
                }
                // noise sigma: crude via scene variance between mirrored scenes? poor proxy
                // use f- and f+ mean as "other" only
                cell_rows.push((c, spread, r0));
            }
            // top by spread
            cell_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = cell_rows.iter().take(5).map(|(c, spr, r0)| format!("#{c} s={spr:.1} b={r0:.1}")).collect();
            let mut pspread = 0.0f32;
            let mut pbase = 0.0f32;
            for &c in cells {
                let r0 = rest_rates[c];
                let mut spread = 0.0f32;
                for (_, r_scene, _, _) in &rates_per_scene { spread = spread.max((r_scene[c] - r0).abs()); }
                pspread += spread; pbase += r0;
            }
            let k = cells.len().max(1);
            eprintln!("{:<20} pooled_spread/t={:.2} pooled_base/t={:.2} top[{}]", name, pspread / k as f32, pbase / k as f32, top.join(" "));
        }
    }
}
