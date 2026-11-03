//! TEMP audit A4 #7: event-clamp + weight_scale + binding v2 sweep: sea level,
//! scene discrimination on decode-candidate pools, budget fit.
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
    let (sub, _info) = subnetwork_for_queen(&full.connectome, &full.populations, QueenSubnetOptions::default()).expect("sub");

    let cls_of = |c: &alife_neural::connectome::Connectome, i: usize| {
        c.region_name(c.node_meta[i].region_idx).map(|s| s.to_string()).unwrap_or_default()
    };
    let n_sub = sub.n();
    let pos_out_score = |c: &alife_neural::connectome::Connectome, i: usize| -> f32 {
        let (rs, re) = (c.row_offsets[i] as usize, c.row_offsets[i + 1] as usize);
        (rs..re).filter(|&p| c.weights[p] > 0.0).map(|p| c.weights[p]).sum::<f32>()
    };
    let top_cells = |c: &alife_neural::connectome::Connectome, want: &dyn Fn(&str) -> bool, k: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..c.n())
            .filter(|&i| want(&cls_of(c, i)))
            .collect();
        v.sort_by(|&a, &b| pos_out_score(c, b).partial_cmp(&pos_out_score(c, a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        v.truncate(k);
        v
    };
    let body_of = |c: &alife_neural::connectome::Connectome, i: usize| c.node_meta[i].body_id;
    let sugar_bodies: Vec<u64> = full.populations.sugar.iter().map(|&i| body_of(&full.connectome, i as usize)).collect();
    let by_body: std::collections::HashMap<u64, usize> = (0..n_sub).map(|i| (body_of(&sub, i), i)).collect();
    let mut sugar_sorted: Vec<usize> = sugar_bodies.iter().map(|&b| by_body[&b]).collect();
    sugar_sorted.sort_by(|&a, &b| pos_out_score(&sub, b).partial_cmp(&pos_out_score(&sub, a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));

    let vis = top_cells(&sub, &|r| r == "visual_projection", 8);
    let vis2 = top_cells(&sub, &|r| r == "visual_centrifugal" || r == "ol_intrinsic", 8);
    let vnc_s = top_cells(&sub, &|r| r == "vnc_sensory", 4);
    let asc = top_cells(&sub, &|r| r == "sensory_ascending", 4);
    let take2 = |v: &[usize], off: usize, into: &mut Vec<usize>| {
        into.push(v[off]); into.push(v[off + 1]);
    };
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
    let sub_path = pkg.join("queen-subnetwork-166k.alifecns");
    eprintln!("binding v2: {bnd:?}");

    let dec_desc: Vec<usize> = top_cells(&sub, &|r| r == "descending_neuron", 96);

    struct Sc { name: &'static str, chans: [f32; 11] }
    let mk = |name: &'static str, f: Vec<(usize, f32)>| {
        let mut chans = [0.0f32; 11];
        for (c, v) in f { chans[c] = v; }
        Sc { name, chans }
    };
    let scenes = vec![
        mk("rest    ", vec![]),
        mk("foodcos+", vec![(8, 1.0), (7, 0.9)]),
        mk("foodcos-", vec![(8, -1.0), (7, 0.9)]),
        mk("hdgS+   ", vec![(4, 1.0)]),
        mk("hdgS-   ", vec![(4, -1.0)]),
        mk("allmax  ", (0..11).map(|c| (c, 1.0)).collect()),
    ];

    for ws in [0.3f32, 0.4, 0.5] {
        for clamp in [40.0f32, 15.0] {
            eprintln!("\n== ws={ws} clamp={clamp} (pulses 30) ==");
            for sc in &scenes {
                let conn = alife_neural::connectome_binary::load_memmap(&sub_path).expect("mm");
                let mut rt = FlyBrainRuntime::new(conn, 3).expect("rt");
                rt.weight_scale = ws;
                rt.event_current_clamp = clamp;
                rt.sensory.baseline_hz = 3.0;
                rt.sensory.gain_hz = 60.0;
                rt.sensory.pulse_current = 30.0;
                rt.encoders = vec![Encoder::Rate; 22];
                assert!(rt.set_input_binding(&bnd));
                let budget = TickBudget { max_spikes: 4096, max_synops: 200_000 };
                let mut counts: BTreeMap<usize, u64> = Default::default();
                let mut total = 0u64;
                let mut drops = 0u64;
                for t in 0..200 + 400 {
                    let _ = rt.tick_channels(&sc.chans, &budget);
                    if t >= 200 {
                        for &s in rt.spike_ids_last() {
                            counts.entry(s as usize).and_modify(|c| *c += 1).or_insert(1);
                            total += 1;
                        }
                    }
                    drops += rt.last_stats().dropped_synops + rt.last_stats().dropped_spikes;
                }
                let vabs: f32 = rt.state().v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let line = format!(
                    "{} tot/t={:6.1} drops/t={:7.1} |v|max={:5.0} desc_mean={:.2}",
                    sc.name,
                    total as f64 / 400.0,
                    drops as f64 / 600.0,
                    vabs,
                    dec_desc.iter().map(|&c| counts.get(&c).copied().unwrap_or(0)).sum::<u64>() as f64 / dec_desc.len() as f64 / 400.0
                );
                // top 3 responders inside desc
                let mut rows: Vec<(usize, u64)> = dec_desc.iter().map(|&c| (c, counts.get(&c).copied().unwrap_or(0))).collect();
                rows.sort_by(|a, b| b.1.cmp(&a.1));
                let top3: Vec<String> = rows.iter().take(3).map(|(c, n)| format!("#{}={}", c, n)).collect();
                eprintln!("{line} top [{top3:?}]");
            }
        }
    }
}
