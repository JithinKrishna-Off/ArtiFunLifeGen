//! TEMP diagnostic: WHERE does the food-scene difference disappear?
//! Scenes A (food near), B (food far/low energy), G (no stimuli) differ ONLY
//! in the food/energy senses. We inject the mapped channel vectors, then
//! measure, per scene:
//!   1. the exact channel values fed to the brain (adapter.map_channels);
//!   2. per-channel afferent cell rates (the bound input cells, with labels);
//!   3. rates of the direct downstream partners of the food-afferent cells
//!      (CSR out-edges, top |weight| partners);
//!   4. rates of the annotated descending/motor candidate populations;
//!   5. baseline-subtracted (vs G) comparison of every measured stage.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use alife_neural::queen_types::{NeuralTickBudget, SensoryInputForQueen};

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

const CH_NAMES: [&str; 11] = [
    "optic_speed", "optic_cos", "optic_sin", "light_cos", "light_sin",
    "obstacle", "ground", "food_prox", "food_cos", "food_sin", "queen",
];

fn main() {
    let Some(path) = find_artifact() else {
        eprintln!("artifact missing — build it first (malecns_bins)");
        std::process::exit(1);
    };
    let conn = Arc::new(alife_neural::connectome_binary::load(&path).expect("load"));

    // ---- global weight/polarity audit ------------------------------------
    {
        let neg = conn.weights.iter().filter(|&&w| w < 0.0).count();
        let pos = conn.weights.iter().filter(|&&w| w > 0.0).count();
        let zero = conn.weights.iter().filter(|&&w| w == 0.0).count();
        let pol = match &conn.syn_polarity {
            Some(sp) => {
                let unk = sp.iter().filter(|&&p| p == 2).count();
                let exc = sp.iter().filter(|&&p| p == 0).count();
                let inh = sp.iter().filter(|&&p| p == 1).count();
                format!("some: exc={exc} inh={inh} unknown={unk}")
            }
            None => "None".to_string(),
        };
        let node_pol: [u64; 3] = {
            let mut a = [0u64; 3];
            for m in &conn.node_meta {
                a[m.polarity as usize] += 1;
            }
            a
        };
        eprintln!(
            "weights: neg={neg} pos={pos} zero={zero}; syn_polarity {pol}; node polarity exc={} inh={} unknown={}",
            node_pol[0], node_pol[1], node_pol[2]
        );
    }

    // candidate motor classes (same proposal the adapter installs)
    let (_, classes) = alife_neural::queen_adapter::propose_region_binding(&conn, 2)
        .expect("region proposal");
    let budget_deg = 2usize; // decode driver: advances per measurement tick

    // scenes: A vs B differ in food distance + energy; G silent
    let scenes: Vec<(&str, SensoryInputForQueen)> = vec![
        ("A", SensoryInputForQueen {
            food_dir: (1.0, 0.0), food_dist: 2.0, energy_frac: 0.5, ..Default::default() }),
        ("B", SensoryInputForQueen {
            food_dir: (1.0, 0.0), food_dist: 1e8, energy_frac: 0.08, ..Default::default() }),
        ("G", SensoryInputForQueen {
            energy_frac: 0.5, ..Default::default() }),
    ];

    struct SceneData {
        chans: Vec<f32>,
        aff_rate: [f32; 11],           // Hz per channel (mean over its bound cells)
        aff_cells: Vec<(String, u32)>, // channel 9 (food_prox) cell ids + labels
        down_ids: Vec<u32>,            // downstream partner ids of food cells
        down_ids_desc: Vec<u32>,
        down_rate: f32,
        down_rate_desc: f32,
        class_rate: Vec<(String, f32)>, // pooled Hz per candidate class
        cell_counts: Vec<u32>,          // spikes per neuron (measured window)
    }
    let mut data: Vec<SceneData> = Vec::new();

    // budget experiment: does the per-tick propagation budget (bounded by
    // default) starve the food afferent rows? Compare default vs unbounded.
    let budget_mode = std::env::var("TRACE_BUDGET").unwrap_or_else(|_| "bounded".into());
    let budget = match budget_mode.as_str() {
        "unbounded" => alife_neural::runtime::TickBudget::unbounded(),
        _ => alife_neural::runtime::TickBudget::default(),
    };
    eprintln!("queen trace: budget_mode={budget_mode}");

    for (tag, senses) in &scenes {
        // FRESH adapter per scene (deterministic, identical to the runner).
        // TRACE_CPC overrides the afferent cells-per-channel (default 2).
        let cpc: usize = std::env::var("TRACE_CPC").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let mut adapter = alife_neural::queen_adapter::QueenBrainAdapter::new_for_subnetwork_with_cells(
            Arc::clone(&conn), 0, cpc,
        )
        .expect("adapter");
        let chans = adapter.map_channels(senses);
        let rt = adapter.runtime_mut();
        let binding: Vec<usize> = rt.input_binding().expect("binding").to_vec();        let cc = rt.sensory.channel_cells;

        // food afferent cells = channels 7..=9 (cos/sin/prox)
        let food_cells: Vec<usize> = binding[7 * cc..10 * cc].to_vec();
        // direct downstream partners of the food cells (CSR), top by |weight|
        let mut partners: Vec<(u32, f32)> = Vec::new();
        for &c in &food_cells {
            let (s, e) = (conn.row_offsets[c] as usize, conn.row_offsets[c + 1] as usize);
            for p in s..e {
                partners.push((conn.col_idx[p], conn.weights[p].abs()));
            }
        }
        partners.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        partners.truncate(64);
        let down_ids: Vec<u32> = partners.iter().map(|p| p.0).collect();
        // same for descending candidate cells (their upstream is not needed —
        // we measure the class itself below)

        // candidate class members (as flat id list, capped for the trace)
        let mut class_cells: Vec<(&str, Vec<usize>)> = Vec::new();
        for cls in classes.iter().take(4) {
            class_cells.push((cls.label.as_str(), cls.cells.clone()));
        }

        // run: 200 settle advances + 100 measured advances (2 ticks each)
        let mut counts_aff = vec![0u64; binding.len()];
        let mut counts_down = vec![0u64; down_ids.len()];
        let mut counts_class: Vec<Vec<u64>> = class_cells.iter().map(|(_, cs)| vec![0u64; cs.len()]).collect();
        let mut cell_counts: Vec<u32> = vec![0; conn.n()];
        let mut ticks = 0u64;
        let mut measure = false;
        for _ in 0..300 {
            if ticks == 400 {
                measure = true;
            }
            // reproduce adapter.tick(): map_channels is already computed; run
            // `budget_deg` raw 1 ms ticks per advance
            for _ in 0..budget_deg {
                let _ = rt.tick_channels(&chans, &budget);
            }
            if measure {
                for &s in rt.spike_ids_last() {
                    let u = s as usize;
                    cell_counts[u] += 1;
                    if let Some(slot) = binding.iter().position(|&b| b == u) {
                        counts_aff[slot] += 1;
                    }
                    if let Some(slot) = down_ids.iter().position(|d| *d as usize == u) {
                        counts_down[slot] += 1;
                    }
                    for (mi, (_, cs)) in class_cells.iter().enumerate() {
                        if let Some(slot) = cs.iter().position(|&c| c == u) {
                            counts_class[mi][slot] += 1;
                        }
                    }
                }
            }
            ticks += budget_deg as u64;
        }
        let dt_ms = 1000.0;
        let t_s = (ticks - 400) as f32 * 0.001;
        // per-channel afferent rate
        let mut aff_rate = [0.0f32; 11];
        for ch in 0..11 {
            let sum: u64 = counts_aff[ch * cc..(ch + 1) * cc].iter().sum();
            aff_rate[ch] = sum as f32 / (cc as f32 * t_s);
        }
        let aff_cells: Vec<(String, u32)> = binding[9 * cc..10 * cc]
            .iter()
            .map(|&c| {
                let m = &conn.node_meta[c];
                (format!("{}:{}", conn.region_name(m.region_idx).unwrap_or("?"), c as u32), 0)
            })
            .map(|(s, _)| (s, 0))
            .collect();
        let measured_down: f32 = counts_down.iter().sum::<u64>() as f32
            / (down_ids.len().max(1) as f32 * t_s);
        let class_rate: Vec<(String, f32)> = class_cells
            .iter()
            .zip(&counts_class)
            .map(|((label, cs), cnt)| {
                let sum: u64 = cnt.iter().sum();
                (label.to_string(), sum as f32 / (cs.len().max(1) as f32 * t_s))
            })
            .collect();
        data.push(SceneData {
            chans: chans.to_vec(),
            aff_rate,
            aff_cells,
            down_ids,
            down_ids_desc: Vec::new(),
            down_rate: measured_down,
            down_rate_desc: 0.0,
            class_rate,
            cell_counts,
        });
        eprintln!("scene {tag}: ticks_measured={ticks}");
        // out-structure of the food afferents (stationary fact, printed once)
        if *tag == "A" {
            eprintln!("\nfood afferent out-structure:");
            for &c in &food_cells {
                let (s, e) = (conn.row_offsets[c] as usize, conn.row_offsets[c + 1] as usize);
                eprintln!("  ol_sensory:{c} outdeg={}", e - s);
                for p in s..e {
                    let post = conn.col_idx[p] as usize;
                    let in_s = conn.row_offsets[post] as usize;
                    let in_e = conn.row_offsets[post + 1] as usize;
                    eprintln!(
                        "    -> {}:{} w={:.4} post_indeg={}",
                        conn.region_name(conn.node_meta[post].region_idx).unwrap_or("?"),
                        post,
                        conn.weights[p],
                        in_e - in_s
                    );
                }
            }
        }
        let _ = (&counts_down, &dt_ms);
    }

    // ---- report ----------------------------------------------------------
    let g = &data[2];
    // full binding dump once (channel -> cell ids + labels)
    {
        let mut adapter = alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(
            Arc::clone(&conn), 0,
        ).expect("adapter");
        let rt = adapter.runtime_mut();
        let binding: Vec<usize> = rt.input_binding().expect("binding").to_vec();
        let cc = rt.sensory.channel_cells;
        eprintln!("mapping = {:#?}", adapter.mapping());
        eprintln!("\n== input binding ==");
        let mut seen: std::collections::HashMap<usize, Vec<&str>> = Default::default();
        for ch in 0..11 {
            eprint!("  {:<12}:", CH_NAMES[ch]);
            for k in 0..cc {
                let c = binding[ch * cc + k];
                let label = conn.region_name(conn.node_meta[c].region_idx).unwrap_or("?");
                eprint!(" {}:{c}", label);
                seen.entry(c).or_default().push(CH_NAMES[ch]);
            }
            eprintln!();
        }
        let dups: Vec<(&usize, &Vec<&str>)> =
            seen.iter().filter(|(_, tags)| tags.len() > 1).collect();
        if !dups.is_empty() {
            eprintln!("!! DUPLICATE-BOUND CELLS: {:?}", dups);
        }
    }
    for (i, (tag, _)) in scenes.iter().enumerate() {
        let d = &data[i];
        eprintln!("\n== scene {tag} ==");
        let chs: Vec<String> = d
            .chans
            .iter()
            .enumerate()
            .map(|(ci, &v)| format!("{}={v:.3}", CH_NAMES[ci]))
            .collect();
        eprintln!("channels: [{}]", chs.join(", "));
        eprintln!("aff rates Hz (Δ vs G):");
        for ci in 0..11 {
            let delta = d.aff_rate[ci] - g.aff_rate[ci];
            eprintln!(
                "  {:<12} {:>7.2} ({:>6.2})",
                CH_NAMES[ci], d.aff_rate[ci], delta
            );
        }
        eprintln!("food_prox cells: {:?}", d.aff_cells.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>());
        eprintln!(
            "food-downstream partners (top-art): n={} pooled {:>6.2} Hz (Δ {:>6.2})",
            d.down_ids.len(),
            d.down_rate,
            d.down_rate - g.down_rate
        );
        eprintln!("candidate class pooled rates Hz (Δ vs G):");
        for (label, rate) in &d.class_rate {
            let g_rate = g.class_rate.iter().find(|(l, _)| l == label).map(|(_, r)| *r).unwrap_or(0.0);
            eprintln!("  {:<24} {:>7.2} ({:>6.2})", label, rate, rate - g_rate);
        }
        // where in the whole 40k network is the scene difference VISIBLE?
        // top-15 by |Δrate| vs G, with labels.
        let label_of = |i: usize| conn.region_name(conn.node_meta[i].region_idx).unwrap_or("?");
        let n_ticks = d.cell_counts.iter().max().copied().unwrap_or(0); // not used; keep simple
        let _ = n_ticks;
        let mut deltas: Vec<(i64, usize)> = (0..conn.n())
            .map(|i| {
                let dr = (d.cell_counts[i] as i64 - g.cell_counts[i] as i64) * 1000i64
                    / 200i64; // measured window = 200 ticks
                (dr, i)
            })
            .collect();
        deltas.sort_unstable_by_key(|(dr, _)| dr.abs());
        deltas.reverse();
        deltas.truncate(15);
        eprintln!("top |Δrate| neurons vs G (milli-Hz):");
        for (dr, i) in &deltas {
            if dr.abs() < 1 {
                break;
            }
            eprintln!(
                "  {}:{:>7} Δ{} mHz (scene {} vs G {})",
                label_of(*i), i, dr, d.cell_counts[*i], g.cell_counts[*i]
            );
        }
    }
}
