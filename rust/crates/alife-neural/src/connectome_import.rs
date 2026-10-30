//! Streaming normalized-TSV -> compact-index -> CSR import.
//!
//! Scaling design for the full MaleCNS brain (~166,700 neurons / millions of
//! edges, verified target scale in `FLYBRAIN_IMPLEMENTATION.md`):
//! - PASS 1 (edges.tsv streamed): collect the set of body IDs appearing in
//!   edges + nodes; map IDs to compact indices via SORTED unique order, so the
//!   mapping is a stable, deterministic function of the accumulated ID set
//!   (same ID set => same indices, independent of row order).
//! - PASS 2 (edges.tsv streamed again): counting pass (out-degree per pre).
//! - PASS 3: place edges into the pre-sized CSR arrays in file order
//!   (deterministic in-row layout), computing prefix offsets first.
//! No dense matrices, no second full copy of the edge list.
//!
//! Missing/partial annotations are RETAINED as "unavailable" fields with
//! `ProvenanceTag::Unavailable`; the neuron itself is always kept with its
//! exact body ID. Unknown endpoint policy is configurable (`ImportOptions`).

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::connectome::{
    Connectome, ConnectomeError, NeuronMeta, Polarity, NO_IDX,
};
use crate::connectome_format::{EdgeTsvReader, FormatError, NodeTsvReader, Sidecar};
use crate::provenance::{ProvenanceRecord, ProvenanceTag, EDGES_FILE, NODES_FILE, SIDECAR_FILE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownNodePolicy {
    /// Keep the edge; create an implicit node with unavailable annotations.
    /// This is the data-faithful default (the edge exists in the source).
    CreateImplicit,
    /// Reject the edge (strict mode; counts reported).
    RejectEdge,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ImportOptions {
    pub unknown_nodes: UnknownNodePolicy,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            unknown_nodes: UnknownNodePolicy::CreateImplicit,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImportStats {
    pub neurons: u64,
    pub edges: u64,
    pub edges_annotated_nodes_between: u64,
    pub nodes_annotated: u64,
    pub nodes_unannotated: u64,
    pub edges_rejected_unknown_node: u64,
    pub regions: u64,
    pub cell_types: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("format error: {0}")]
    Format(#[from] FormatError),
    #[error("connectome validation: {0}")]
    Connectome(#[from] ConnectomeError),
    #[error("missing file: {0}")]
    MissingFile(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Import result: connectome + sidecar + stats.
#[derive(Debug)]
pub struct ImportResult {
    pub connectome: Connectome,
    pub sidecar: Sidecar,
    pub stats: ImportStats,
}

/// Import a normalized dataset directory (one dataset per directory; datasets
/// are NEVER merged by this crate).
pub fn import_dir(path: &Path, opts: &ImportOptions) -> Result<ImportResult, ImportError> {
    let edges_path = path.join(EDGES_FILE);
    let nodes_path = path.join(NODES_FILE);
    let sidecar_path = path.join(SIDECAR_FILE);
    for p in [&edges_path, &nodes_path, &sidecar_path] {
        if !p.exists() {
            return Err(ImportError::MissingFile(p.display().to_string()));
        }
    }

    // ---- pass 1a: nodes.tsv -> body ids + annotations -----------------------
    let nodes_file = std::fs::File::open(&nodes_path)?;
    let mut nreader = NodeTsvReader::new(BufReader::new(nodes_file))?;
    let mut nodes: Vec<(u64, String, String, String, String)> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    while let Some(nr) = nreader.next_node()? {
        all_ids.push(nr.body_id);
        nodes.push((
            nr.body_id,
            nr.cell_type,
            nr.region,
            nr.status,
            nr.soma_side,
        ));
    }

    // ---- pass 1b: edges.tsv -> unknown-candidate ids + reject-policy count --
    let edges_file = std::fs::File::open(&edges_path)?;
    let mut ereader = EdgeTsvReader::new(BufReader::new(edges_file))?;
    let mut edge_ids_extra: Vec<u64> = Vec::new();
    let mut rejected = 0u64;
    match opts.unknown_nodes {
        UnknownNodePolicy::CreateImplicit => {
            while let Some(e) = ereader.next_edge()? {
                all_ids.push(e.body_pre);
                all_ids.push(e.body_post);
            }
        }
        UnknownNodePolicy::RejectEdge => {
            // Need node membership: first pass collects edges and counts rejects
            // WITHOUT storing them (streaming, worst case one extra pass later).
            let node_ids: std::collections::HashSet<u64> = all_ids.iter().copied().collect();
            while let Some(e) = ereader.next_edge()? {
                let known = node_ids.contains(&e.body_pre) && node_ids.contains(&e.body_post);
                if known {
                    edge_ids_extra.push(e.body_pre);
                    edge_ids_extra.push(e.body_post);
                } else {
                    rejected += 1;
                }
            }
            all_ids.extend(edge_ids_extra.drain(..));
        }
    }

    // ---- stable compact mapping: sorted unique IDs -> index ----------------
    all_ids.sort_unstable();
    all_ids.dedup();
    let n = all_ids.len();
    let index_of: HashMap<u64, u32> =
        all_ids.iter().enumerate().map(|(i, &b)| (b, i as u32)).collect();

    // ---- node metadata with real side tables ("" -> Unavailable) -----------
    // Materialize EVERY compact node's exact body ID first (nodes.tsv bodies
    // plus implicit bodies that only appear in edges). Missing annotations
    // stay Unavailable; presence of an implicit node in the source edges is
    // itself a source fact.
    let mut node_meta: Vec<NeuronMeta> = (0..n)
        .map(|_| NeuronMeta {
            body_id: 0,
            region_idx: NO_IDX,
            cell_type_idx: NO_IDX,
            polarity: Polarity::Unknown,
            origin: ProvenanceTag::Unavailable,
        })
        .collect();
    for (i, &b) in all_ids.iter().enumerate() {
        node_meta[i].body_id = b;
    }

    // side tables built from nodes.tsv annotations ONLY (empty field -> never
    // enters a table; the neuron keeps Unavailable indices)
    let mut regions: Vec<String> = Vec::new();
    let mut region_idx: HashMap<String, u32> = HashMap::new();
    let mut ctypes: Vec<String> = Vec::new();
    let mut ctype_idx: HashMap<String, u32> = HashMap::new();
    for (_, ct, rg, _, _) in nodes.iter() {
        if !rg.is_empty() && !region_idx.contains_key(rg.as_str()) {
            region_idx.insert(rg.clone(), regions.len() as u32);
            regions.push(rg.clone());
        }
        if !ct.is_empty() && !ctype_idx.contains_key(ct.as_str()) {
            ctype_idx.insert(ct.clone(), ctypes.len() as u32);
            ctypes.push(ct.clone());
        }
    }

    let mut nodes_annotated = 0u64;
    let mut nodes_unannotated = 0u64;
    for (body_id, ct, rg, _status, _soma) in nodes.drain(..) {
        let i = index_of[&body_id] as usize;
        let m = &mut node_meta[i];
        m.body_id = body_id;
        if m.origin == ProvenanceTag::Unavailable {
            m.origin = ProvenanceTag::SuppliedBySource;
        }
        if !rg.is_empty() {
            m.region_idx = region_idx[rg.as_str()];
        }
        if !ct.is_empty() {
            m.cell_type_idx = ctype_idx[ct.as_str()];
        }
        if rg.is_empty() && ct.is_empty() {
            nodes_unannotated += 1;
        } else {
            nodes_annotated += 1;
        }
    }

    // ---- pass 2/3: wire edges into CSR (file order per row) ----------------
    let edges_file = std::fs::File::open(&edges_path)?;
    let mut ereader = EdgeTsvReader::new(BufReader::new(edges_file))?;
    let mut outdeg = vec![0u32; n + 1];
    let mut kept = 0u64;
    match opts.unknown_nodes {
        UnknownNodePolicy::RejectEdge => {
            while let Some(e) = ereader.next_edge()? {
                let a = index_of.get(&e.body_pre);
                let b = index_of.get(&e.body_post);
                if let (Some(&a), Some(_b)) = (a, b) {
                    outdeg[a as usize + 1] += 1;
                    kept += 1;
                }
            }
        }
        UnknownNodePolicy::CreateImplicit => {
            while let Some(e) = ereader.next_edge()? {
                let a = index_of[&e.body_pre] as usize;
                outdeg[a + 1] += 1;
                kept += 1;
            }
        }
    }
    for i in 1..=n {
        outdeg[i] += outdeg[i - 1];
    }
    let row_offsets = outdeg;
    let m = kept as usize;
    let mut col_idx = vec![0u32; m];
    let mut weights = vec![0.0f32; m];
    let mut fill = row_offsets.clone();
    let mut annotated_pairs = 0u64;
    let edges_file = std::fs::File::open(&edges_path)?;
    let mut ereader = EdgeTsvReader::new(BufReader::new(edges_file))?;
    while let Some(e) = ereader.next_edge()? {
        let (a, b) = match opts.unknown_nodes {
            UnknownNodePolicy::CreateImplicit => (
                index_of[&e.body_pre],
                // post may be a node only present in edges (implicit);
                // index_of covers it (pass 1b added both endpoints).
                index_of[&e.body_post],
            ),
            UnknownNodePolicy::RejectEdge => {
                let (Some(a), Some(b)) = (
                    index_of.get(&e.body_pre).copied(),
                    index_of.get(&e.body_post).copied(),
                ) else {
                    continue; // already counted in `rejected`
                };
                (a, b)
            }
        };
        if node_meta[a as usize].cell_type_idx != NO_IDX {
            annotated_pairs += 1;
        }
        let p = fill[a as usize] as usize;
        fill[a as usize] += 1;
        col_idx[p] = b;
        weights[p] = e.weight; // VERBATIM
    }

    // implicit nodes: annotate their correct origin (existence from source
    // edges; annotations Unavailable)
    for met in node_meta.iter_mut() {
        if met.body_id == 0 {
            unreachable!("index must be materialized");
        }
    }

    let sidecar = read_sidecar(&sidecar_path)?;
    let connectome = Connectome::from_csr(
        sidecar.dataset.clone(),
        guessed_template(&sidecar),
        node_meta,
        regions,
        ctypes,
        row_offsets,
        col_idx,
        weights,
        None,
        None,
    )?;

    let stats = ImportStats {
        neurons: n as u64,
        edges: kept,
        edges_annotated_nodes_between: annotated_pairs,
        nodes_annotated,
        nodes_unannotated,
        edges_rejected_unknown_node: rejected,
        regions: region_idx.len() as u64,
        cell_types: ctype_idx.len() as u64,
    };
    Ok(ImportResult {
        connectome,
        sidecar,
        stats,
    })
}

/// Build a provenance record from the imported result. Human-supplied fields
/// (source sha256) stay blank unless filled in the sidecar.
pub fn provenance_from(result: &ImportResult) -> ProvenanceRecord {
    let mut rec = ProvenanceRecord::malecns_default();
    rec.original_filename = result
        .sidecar
        .source_files
        .first()
        .cloned()
        .unwrap_or_default();
    rec.source_sha256 = result.sidecar.source_sha256.clone();
    rec.converter_version = result.sidecar.converter_version.clone();
    rec.converted_at_utc = result.sidecar.converted_at_utc.clone();
    rec.imported_neurons = result.stats.neurons;
    rec.imported_edges = result.stats.edges;
    rec
}

fn guessed_template(sidecar: &Sidecar) -> String {
    sidecar.dataset.clone()
}

/// Read + validate the sidecar JSON.
pub fn read_sidecar(path: &Path) -> Result<Sidecar, ImportError> {
    let s = std::fs::read_to_string(path)?;
    let sc: Sidecar = serde_json::from_str(&s)?;
    sc.validate()?;
    Ok(sc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{Polarity, NO_IDX};
    use crate::provenance::ProvenanceTag;

    /// Suite-local SYNTHETIC sample (see testdata/malecns-sample/README.md).
    fn sample_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join("malecns-sample")
    }

    #[test]
    fn imports_synthetic_sample_tsv() {
        let r = import_dir(&sample_dir(), &ImportOptions::default()).expect("import");
        assert_eq!(r.stats.neurons, 4, "3 nodes.tsv bodies + 1 implicit from edges");
        assert_eq!(r.stats.edges, 5);
        assert_eq!(r.stats.nodes_annotated, 3);
        assert_eq!(r.stats.nodes_unannotated, 0);
        assert_eq!(r.stats.regions, 3);
        assert_eq!(r.stats.cell_types, 3);
        // stable mapping: sorted unique body ids -> ascending compact indices
        let ids: Vec<u64> = r.connectome.node_meta.iter().map(|m| m.body_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        // CSR correctness: body 3 -> [1,2] wait, edges: 1->2, 1->3, 2->3, 3->1, 4->2
        assert_eq!(r.connectome.row_offsets, vec![0, 2, 3, 4, 5]);
        assert_eq!(
            r.connectome.col_idx,
            vec![1, 2, // body 1 rows
                 2,   // body 2
                 0,   // body 3
                 1]   // body 4
        );
        // weights preserved VERBATIM (all exactly representable in f32)
        assert_eq!(r.connectome.weights, vec![0.5, 3.25, 1.5, -2.0, 0.875]);
        assert!(r.connectome.validate().is_ok());
    }

    #[test]
    fn annotation_retention_incl_missing_fields() {
        let r = import_dir(&sample_dir(), &ImportOptions::default()).expect("import");
        // body 1: motor
        let m1 = &r.connectome.node_meta[0];
        assert_eq!(m1.region_idx, 0); // "adult"
        assert_eq!(m1.cell_type_idx, 0);
        assert_eq!(m1.origin, ProvenanceTag::SuppliedBySource);
        // body 2: missing somaSide is retained as empty (no invented value):
        // cell type "sensory", region "sensory"
        let m2 = &r.connectome.node_meta[1];
        assert_eq!(r.connectome.region_name(m2.region_idx), Some("sensory"));
        assert_eq!(r.connectome.cell_type_name(m2.cell_type_idx), Some("sensory"));
        // body 4: exists only in edges -> implicit, all annotations Unavailable
        let m4 = &r.connectome.node_meta[3];
        assert_eq!(m4.body_id, 4);
        assert_eq!(m4.region_idx, NO_IDX);
        assert_eq!(m4.cell_type_idx, NO_IDX);
        assert_eq!(m4.polarity, Polarity::Unknown);
        assert_eq!(m4.origin, ProvenanceTag::Unavailable);
    }

    #[test]
    fn reject_edge_policy_counts_unknown_endpoints() {
        let opts = ImportOptions {
            unknown_nodes: UnknownNodePolicy::RejectEdge,
        };
        let r = import_dir(&sample_dir(), &opts).expect("import");
        // body 4 edges (row 4->2) are rejected; rows never touching 4 kept.
        assert_eq!(r.stats.edges, 4);
        assert_eq!(r.stats.edges_rejected_unknown_node, 1);
        assert_eq!(r.stats.neurons, 3);
        // weights of kept edges still verbatim
        assert!(r.connectome.weights.contains(&0.5) && r.connectome.weights.contains(&-2.0));
        assert!(!r.connectome.weights.contains(&0.875));
    }

    #[test]
    fn sidecar_missing_is_error() {
        let tmp = std::env::temp_dir().join("alife-neural-test-missing");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join(EDGES_FILE), "body_pre\tbody_post\tweight\n1\t2\t1.0\n").unwrap();
        std::fs::write(tmp.join(NODES_FILE), "bodyId\ttype\tsuperclass\tstatus\tsomaSide\n1\tt\ts\tS\t\n2\tt\ts\tS\t\n").unwrap();
        let err = import_dir(&tmp, &ImportOptions::default()).unwrap_err();
        assert!(matches!(err, ImportError::MissingFile(_)));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn provenance_from_roundtrips_through_json() {
        let r = import_dir(&sample_dir(), &ImportOptions::default()).expect("import");
        let rec = provenance_from(&r);
        assert_eq!(rec.dataset, crate::provenance::DATASET_NAME);
        // roundtrip: to_json -> from_json -> equal
        let json = rec.to_json();
        let back: ProvenanceRecord = serde_json::from_str(&json).expect("json roundtrip");
        assert_eq!(back, rec);
        assert_eq!(back.imported_neurons, 4);
        assert_eq!(back.imported_edges, 5);
        // download-dependent fields stay blank (never fabricated)
        assert_eq!(back.source_sha256, "");
    }

    #[test]
    fn subnetwork_extract_of_import_keeps_source_body_ids() {
        let r = import_dir(&sample_dir(), &ImportOptions::default()).expect("import");
        let mut regions = std::collections::HashSet::new();
        regions.insert("adult".to_string());
        regions.insert("sensory".to_string());
        let ex = crate::subnetwork::extract(&r.connectome, &crate::subnetwork::SubnetworkFilter {
            regions,
            cell_types: Default::default(),
        }, "slice");
        // bodies 1 and 2 extracted with their ORIGINAL body ids; the induced
        // edge 1->2 survives with its verbatim source weight
        assert_eq!(ex.body_ids, vec![1, 2]);
        assert_eq!(ex.connectome.weights, vec![0.5]);
    }
}
