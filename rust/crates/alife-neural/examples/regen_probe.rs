//! TEMP audit A4 #5: regenerate the queen closure from the full graph and
//! compare identity vs the shipped artifact; inspect sugar cells, lamina,
//! descending in/out structure, and per-class positive-edge stats.
use std::path::{Path, PathBuf};

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
    eprintln!("pkg = {}", pkg.display());
    let loaded = alife_neural::malecns_bins::load_package(&pkg).expect("package");
    let full = alife_neural::malecns_bins::import_full(loaded).expect("import");
    eprintln!(
        "full: n={} e={} regions={}",
        full.connectome.n(),
        full.connectome.edge_count(),
        full.connectome.regions.len()
    );

    let (sub, info) = alife_neural::malecns_bins::subnetwork_for_queen(
        &full.connectome,
        &full.populations,
        alife_neural::malecns_bins::QueenSubnetOptions::default(),
    )
    .expect("sub");
    eprintln!(
        "regenerated sub: n={} e={} hops={} truncated={}",
        info.neurons, info.edges, info.hops, info.truncated
    );
    eprintln!("seed_total={}", info.seed_total);
    println!(
        "[id] regenerated sub identity = {:?}",
        alife_neural::connectome_binary::connectome_identity(&sub)
    );
    // compare to shipped artifact
    let art_path = pkg.join("queen-subnetwork-166k.alifecns");
    let art = alife_neural::connectome_binary::load_memmap(&art_path).expect("artifact");
    println!("[id] shipped    art identity  = {:?}", alife_neural::connectome_binary::connectome_identity(&art));
    println!(
        "[id] equal = {}",
        alife_neural::connectome_binary::connectome_identity(&sub).4
            == alife_neural::connectome_binary::connectome_identity(&art).4
    );

    // ---- sugar cells
    let cls_of = |c: &alife_neural::connectome::Connectome, i: usize| c.region_name(c.node_meta[i].region_idx).map(|s| s.to_string());
    eprintln!("\n-- sugar population cells in the FULL graph --");
    for &s in &full.populations.sugar {
        let c = cls_of(&full.connectome, s as usize).unwrap_or_default();
        let (rs, re) = (full.connectome.row_offsets[s as usize] as usize, full.connectome.row_offsets[s as usize + 1] as usize);
        let mut posw = 0.0f32; let mut negw = 0.0f32;
        for p in rs..re {
            let w = full.connectome.weights[p];
            if w > 0.0 { posw += w } else { negw += w }
        }
        eprintln!("idx={s:7} class={c:<18} out_deg={:4} posw={posw:8.1} negw={negw:8.1}", re - rs);
    }

    // ---- closure-level stats on the REGENERATED subnetwork
    let sub_ref = &sub;
    let mut by: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for i in 0..sub_ref.n() {
        by.entry(cls_of(sub_ref, i).unwrap_or_default()).or_default().push(i);
    }
    eprintln!("\n-- regenerated sub per-class --");
    for (label, cells) in by {
        let (mut pos_out_cells, mut out_cells, mut in_cells) = (0usize, 0usize, 0usize);
        for &c in &cells {
            let (rs, re) = (sub_ref.row_offsets[c] as usize, sub_ref.row_offsets[c + 1] as usize);
            if re > rs { out_cells += 1; }
            let mut any_pos = false;
            for p in rs..re { if sub_ref.weights[p] > 0.0 { any_pos = true; break; } }
            if any_pos { pos_out_cells += 1; }
        }
        eprintln!("{:<22} n={:<6} out={:<6} out_pos={:<6}", label, cells.len(), out_cells, pos_out_cells);
        let _ = in_cells;
    }
}
