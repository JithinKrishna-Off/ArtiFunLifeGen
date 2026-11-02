//! Runner: import the MaleCNS v1.0 flat-connectome binary package from the
//! local (external, untracked) directory, compile BOTH artifacts
//! (`malecns-166k-full.alifecns` + `queen-subnetwork-166k.alifecns`) and
//! report ACTUAL neuron/edge counts, timing, sizes and identity hashes.
//!
//! Usage:
//!   cargo run -p alife-neural --example import_malecns [-- <package_dir> <out_dir>]
//!
//! Defaults: package `../../../appdata/malecns166k` (relative to the crate),
//! artifacts written next to the package under `artifacts/`.
//! If the package has not landed yet this exits gracefully with the ready
//! message (the coordinator drops the data during the session).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use alife_neural::connectome_binary;
use alife_neural::malecns_bins::{
    subnetwork_for_queen, import_full, load_package, file_sha256, QueenSubnetOptions,
};

fn main() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The coordinator may stage the package either at the repo root or under
    // the rust workspace root; both resolve to external untracked dirs.
    let candidates = [
        crate_root.join("../../../appdata/malecns166k"),
        crate_root.join("../../appdata/malecns166k"),
    ];
    let mut args = std::env::args().skip(1);
    let pkg = args.next().map(PathBuf::from).unwrap_or_else(|| {
        candidates
            .iter()
            .find(|p| p.join("manifest.json").is_file())
            .cloned()
            .unwrap_or(candidates[0].clone())
    });
    let out = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| pkg.clone());
    std::fs::create_dir_all(&out).expect("create out dir");

    if !pkg.join("manifest.json").exists() {
        println!(
            "[malecns] package not present yet at {} — runner is ready; \
             (waiting for the coordinator to drop the malecns166k flat package)",
            pkg.display()
        );
        return;
    }

    let t_all = Instant::now();
    println!("[malecns] importing package {}", pkg.display());

    let loaded = load_package(&pkg).expect("package failed validation");
    println!(
        "[malecns] validated: neurons={} edges={} contacts={:?} orientation={} \
         manifest_file_hashes={} (wait: load) ",
        loaded.neurons(),
        loaded.edges(),
        loaded.manifest.synaptic_contacts,
        loaded.manifest.orientation,
        loaded.manifest.file_hashes,
    );

    let t_import = Instant::now();
    let full = import_full(loaded).expect("full import failed");
    println!(
        "[malecns] full connectome: name={} neurons={} edges={} (+{}/-{}) regions={} cell_types={} transposed={} import_ms={}",
        full.connectome.name,
        full.stats.neurons,
        full.stats.edges,
        full.stats.positive_edges,
        full.stats.negative_edges,
        full.stats.regions,
        full.stats.cell_types,
        full.stats.transposed,
        t_import.elapsed().as_millis(),
    );
    let (fname, fver, fn_, fe, fhash) = connectome_binary::connectome_identity(&full.connectome);
    println!("[malecns] full identity: {fname}/{fver} n={fn_} e={fe} fnv1a-64={fhash:#x}");

    // ---- artifact 1: full connectome ----
    let t0 = Instant::now();
    let full_path = out.join("malecns-full-166k.alifecns");
    connectome_binary::save(&full.connectome, &full_path).expect("write full artifact");
    let full_bytes = std::fs::metadata(&full_path).expect("stat").len();
    let full_sha = file_sha256(&full_path).expect("sha");
    println!(
        "[malecns] full artifact: {} bytes={} sha256={} ms={}",
        full_path.display(),
        full_bytes,
        full_sha,
        t0.elapsed().as_millis(),
    );

    // ---- Queen subnetwork ----
    let t1 = Instant::now();
    let opts = QueenSubnetOptions::default();
    let (sub, info) = subnetwork_for_queen(&full.connectome, &full.populations, opts)
        .expect("subnetwork extraction");
    println!(
        "[malecns] queen subnetwork: name={} neurons={} edges={} hops={} cap={} truncated={} \
         seeds(retina={},lamina={},sugar={},descending={}) ms={}",
        sub.name,
        info.neurons,
        info.edges,
        info.hops,
        info.max_neurons,
        info.truncated,
        info.seeds_found_retina,
        info.seeds_found_lamina,
        info.seeds_found_sugar,
        info.seeds_found_descending,
        t1.elapsed().as_millis(),
    );
    let (_, sver, sn, se, shash) = connectome_binary::connectome_identity(&sub);
    println!("[malecns] sub identity: /{sver} n={sn} e={se} fnv1a-64={shash:#x}");

    let t2 = Instant::now();
    let sub_path = out.join("queen-subnetwork-166k.alifecns");
    connectome_binary::save(&sub, &sub_path).expect("write subnetwork artifact");
    let sub_bytes = std::fs::metadata(&sub_path).expect("stat").len();
    let sub_sha = file_sha256(&sub_path).expect("sha");
    println!(
        "[malecns] subnetwork artifact: {} bytes={} sha256={} ms={}",
        sub_path.display(),
        sub_bytes,
        sub_sha,
        t2.elapsed().as_millis(),
    );

    // ---- QueenBrainAdapter on the subnetwork + per-tick cost measurement ----
    match alife_neural::queen_adapter::QueenBrainAdapter::new_from_connectome(
        Arc::new(sub),
        42,
    ) {
        Ok(mut adapter) => {
            let senses = alife_neural::queen_types::SensoryInputForQueen {
                food_dir: (1.0, 0.0),
                food_dist: 0.5,
                heading: 0.0,
                speed: 0.5,
                energy_frac: 0.8,
                hazard: 0.0,
                edge_dist: 0.5,
                density: 0.2,
                signal_sample: 0.0,
                hazard_dir: (0.0, 0.0),
                hazard_rate: 0.0,
            };
            let budget = alife_neural::queen_types::NeuralTickBudget {
                max_ticks_per_advance: 1,
            };
            let ticks = 2000u64;
            let t3 = Instant::now();
            let mut spikes_total = 0u64;
            for _ in 0..ticks {
                let out = adapter.tick(&senses, &budget);
                let _ = out;
                spikes_total += adapter.stats().spikes_last;
            }
            let total = t3.elapsed();
            let per_ns = total.as_nanos() / u128::from(ticks);
            let st = adapter.stats();
            println!(
                "[malecns] runtime per-tick cost on the REAL subnetwork: \
                 neurons={} synapses={} avg={} ns/{}us per 1 ms tick ({} ticks; spikes_total={})",
                adapter.runtime().connectome().n(),
                adapter.runtime().connectome().edge_count(),
                per_ns,
                per_ns / 1000,
                ticks,
                spikes_total,
            );
            println!(
                "[malecns] mapping_ok(default first22/last6 binding)={} mesh_mem_est={} bytes",
                adapter.runtime().default_mapping_ok(),
                adapter.runtime().connectome().memory_bytes(),
            );
            let _ = st;
        }
        Err(e) => println!("[malecns] runtime build FAILED: {e}"),
    }

    println!("[malecns] total wall time: {} ms", t_all.elapsed().as_millis());
}
