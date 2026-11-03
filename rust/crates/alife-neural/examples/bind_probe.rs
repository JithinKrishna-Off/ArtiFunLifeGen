//! TEMP audit A4 #6: binding v2 (sugar food cells, structural pools) +
//! weight_scale sweep; scenes per channel; measure pool discrimination.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use alife_neural::malecns_bins::{import_full, load_package, subnetwork_for_queen, QueenSubnetOptions};
use alife_neural::runtime::{FlyBrainRuntime, TickBudget};
use alife_neural::sensory::Encoder;
use alife_neural::state::LifParams;

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

    // structural cell scores inside the subnetwork: positive out weight
    let pos_out_score = |c: &alife_neural::connectome::Connectome, i: usize| -> f32 {
        let (rs, re) = (c.row_offsets[i] as usize, c.row_offsets[i + 1] as usize);
        (rs..re).filter(|&p| c.weights[p] > 0.0).map(|p| c.weights[p]).sum::<f32>()
    };

    let top_cells = |c: &alife_neural::connectome::Connectome, want: &dyn Fn(&str) -> bool, k: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..c.n())
            .filter(|&i| want(&cls_of(c, i)))
            .collect();
        v.sort_by(|&a, &b| {
            let fa = pos_out_score(c, a);
            let fb = pos_out_score(c, b);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        v.truncate(k);
        v
    };

    // sugar body ids -> subnetwork indices
    let body_of = |c: &alife_neural::connectome::Connectome, i: usize| c.node_meta[i].body_id;
    let sugar_bodies: Vec<u64> = full.populations.sugar.iter().map(|&i| body_of(&full.connectome, i as usize)).collect();
    let by_body: std::collections::HashMap<u64, usize> = (0..n_sub).map(|i| (body_of(&sub, i), i)).collect();
    let sugar_sub: Vec<usize> = sugar_bodies.iter().map(|&b| by_body[&b]).collect();
    // sort sugar_sub by pos_out_score desc (deterministic)
    let mut sugar_sorted = sugar_sub.clone();
    sugar_sorted.sort_by(|&a, &b| {
        let fa = pos_out_score(&sub, a);
        let fb = pos_out_score(&sub, b);
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    eprintln!("sugar in sub: {} cells; top out_deg: {:?}", sugar_sorted.len(), sugar_sorted.iter().map(|&i| sub.row_offsets[i + 1] - sub.row_offsets[i]).collect::<Vec<_>>());

    // ---- binding v2 (cells_per_channel = 2)
    let vis = top_cells(&sub, &|r| r == "visual_projection", 8);
    let vis2 = top_cells(&sub, &|r| r == "visual_centrifugal" || r == "ol_intrinsic", 8);
    let sugar2 = sugar_sorted.clone();
    let vnc_s = top_cells(&sub, &|r| r == "vnc_sensory", 4);
    let asc = top_cells(&sub, &|r| r == "sensory_ascending", 4);
    // layout: ai = ch*2+k for k 0..2 (11 channels x 2)
    let mut bnd = vec![0usize; 22];
    let take2 = |v: &[usize], off: usize, into: &mut Vec<usize>| {
        into.push(v[off]);
        into.push(v[off + 1]);
    };
    let mut seq: Vec<usize> = Vec::new();
    take2(&vis, 0, &mut seq); // ch0 optic speed
    take2(&vis, 2, &mut seq); // ch1 optic cos
    take2(&vis, 4, &mut seq); // ch2 optic sin
    take2(&vis2, 0, &mut seq); // ch3 heading cos
    take2(&vis2, 2, &mut seq); // ch4 heading sin
    take2(&asc, 0, &mut seq); // ch5 threat
    take2(&vnc_s, 0, &mut seq); // ch6 ground
    take2(&sugar2, 0, &mut seq); // ch7 food prox
    take2(&sugar2, 2, &mut seq); // ch8 food cos
    take2(&sugar2, 4, &mut seq); // ch9 food sin
    take2(&asc, 2, &mut seq); // ch10 signal
    bnd.copy_from_slice(&seq);
    eprintln!("binding v2: {bnd:?}");

    // decode candidate pools by class
    let pool_lists: Vec<(&str, Vec<usize>)> = vec![
        ("descending", top_cells(&sub, &|r| r == "descending_neuron", 96)),
        ("cb_motor", top_cells(&sub, &|r| r == "cb_motor", 6)),
        ("vnc_motor1", (0..n_sub).filter(|&i| cls_of(&sub, i) == "vnc_motor").collect()),
    ];

    for ws in [0.2f32, 0.25, 0.3] {
        for pulse in [15.0f32, 30.0] {
            // scenes: rest / foodcos+ / foodcos- / allmax
            struct Sc { name: &'static str, chans: [f32; 11] }
            let mk = |name, mut f: Vec<(usize, f32)>| {
                let mut chans = [0.0f32; 11];
                for (c, v) in f.split_off(0) { chans[c] = v; }
                Sc { name, chans }
            };
            let scenes = vec![
                mk("rest    ", vec![]),
                mk("foodcos+", vec![(8, 1.0), (7, 0.9)]),
                mk("foodcos-", vec![(8, -1.0), (7, 0.9)]),
                mk("allmax  ", (0..11).map(|c| (c, 1.0)).collect()),
            ];
            eprintln!("\n== ws={ws} pulse={pulse} ==");
            let params = LifParams::default();
            let ring = params.max_delay(1);
            let budget = TickBudget { max_spikes: 4096, max_synops: 200_000 };
            for sc in &scenes {
                let conn = alife_neural::connectome_binary::load_memmap(&pkg.join("queen-subnetwork-166k.alifecns")).expect("mm");
                let mut rt = FlyBrainRuntime::new(conn, 3).expect("rt");
                rt.weight_scale = ws;
                rt.sensory.baseline_hz = 3.0;
                rt.sensory.gain_hz = 60.0;
                rt.sensory.pulse_current = pulse;
                rt.encoders = vec![Encoder::Rate; 22];
                assert!(rt.set_input_binding(&bnd));
                let mut counts: BTreeMap<usize, u64> = Default::default();
                let mut total = 0u64;
                let mut drops = 0u64;
                let warm = 200;
                for t in 0..warm + 400 {
                    let _ = rt.tick_channels(&sc.chans, &budget);
                    if t >= warm {
                        for &s in rt.spike_ids_last() {
                            counts.entry(s as usize).and_modify(|c| *c += 1).or_insert(1);
                            total += 1;
                        }
                    }
                    drops += rt.last_stats().dropped_synops + rt.last_stats().dropped_spikes;
                }
                // pool mean rates per scene
                let mut line = format!("{} tot/t={:6.1} drops/t={:6.1} pools:", sc.name, total as f64 / 400.0, drops as f64 / 400.0);
                for (label, cells) in &pool_lists {
                    let sum: u64 = cells.iter().map(|&c| counts.get(&c).copied().unwrap_or(0)).sum();
                    let mean = sum as f64 / cells.len() as f64 / 400.0;
                    line.push_str(&format!(" {label}={mean:.2}"));
                }
                eprintln!("{line}");
            }
        }
    }
}
