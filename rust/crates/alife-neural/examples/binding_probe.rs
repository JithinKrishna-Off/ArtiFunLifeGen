//! TEMP diagnostic: why the region binding proposal returns None.
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
    match alife_neural::queen_adapter::propose_region_binding(&conn, 2, None) {
        Some((binding, classes)) => {
            eprintln!("binding len {}", binding.len());
            for c in &classes {
                eprintln!("  class {:<22} cells={}", c.label, c.cells.len());
            }
        }
        None => eprintln!("proposal: None"),
    }
}

