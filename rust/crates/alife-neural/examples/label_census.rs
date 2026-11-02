//! TEMP diagnostic: anatomical census of the imported connectome. Counts
//! neurons per region label (and per superclass keyword) so the sensory
//! bindings use REAL populations, not filler.
fn main() {
    let rel = std::path::Path::new("appdata").join("malecns166k").join("queen-subnetwork-166k.alifecns");
    let mut w = std::env::current_dir().ok().map(|d| d.canonicalize().unwrap_or(d));
    let mut path = None;
    for _ in 0..6 {
        let Some(p) = w else { break };
        let c = p.join(&rel);
        if c.is_file() { path = Some(c); break }
        w = p.parent().map(|q| q.to_path_buf());
    }
    let Some(path) = path else { eprintln!("artifact missing"); std::process::exit(1) };
    let conn = alife_neural::connectome_binary::load(&path).expect("load");
    eprintln!("n={} regions={}", conn.n(), conn.regions.len());

    // per-region counts
    let mut counts: Vec<(usize, u64)> = vec![(0, 0); conn.regions.len()];
    for i in 0..conn.n() {
        let r = conn.node_meta[i].region_idx as usize;
        counts[r].1 += 1;
        counts[r].0 = r;
    }
    let mut rows: Vec<(String, u64)> = counts
        .into_iter()
        .map(|(r, c)| {
            let name = conn.region_name(r as u32).unwrap_or("?").to_string();
            (name, c)
        })
        .filter(|(_, c)| *c > 0)
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    eprintln!("regions (count desc):");
    for (name, c) in &rows {
        eprintln!("  {:>7}  {name}", c);
    }

    // keyword census: what modality words exist at all?
    eprintln!("\nkeyword census (brain-wide node counts):");
    for kw in [
        "sugar", "gustatory", "taste", "chemo", "orco", "odor", "olfac", "labellum",
        "pharyn", "fruity", "bitter", "salt", "water", "Gr", "antenn", "orbit",
        "visual", "optic", "mechano", "proprio", "nocicep", "fruit",
    ] {
        let mut n = 0u64;
        for i in 0..conn.n() {
            if let Some(r) = conn.region_name(conn.node_meta[i].region_idx) {
                if r.to_lowercase().contains(&kw.to_lowercase()) {
                    n += 1;
                }
            }
        }
        if n > 0 {
            eprintln!("  {kw:<12} {n}");
        }
    }
}
