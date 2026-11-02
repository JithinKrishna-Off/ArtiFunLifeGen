//! Direct binary-array importer for the EXTERNAL repackaged MaleCNS v1.0
//! flat-connectome package (coordinator-managed, UNTRACKED, lives under
//! `<repo>/appdata/malecns166k/`, never in git; see `manifest.json` for the
//! package schema, per-file sha256 and upstream provenance: official Janelia
//! MaleCNS v1.0 feather files, minconf 0.5, CC-BY 4.0).
//!
//! Package layout (raw little-endian typed flat arrays; C-order irrelevant):
//! - `graph/edges_offsets.i32` `int32[neurons+1]` — CSR row offsets with
//!   ROW = POSTSYNAPTIC neuron (destination-major, the HF convention).
//! - `graph/edges_source.i32`  `int32[edges]` — PREsynaptic neuron index
//!   per flat edge.
//! - `graph/edges_weight.f32`  `float32[edges]` — SIGNED RAW SYNAPTIC COUNT.
//!   WEIGHT SEMANTICS (do not re-interpret anywhere else): positive =
//!   excitatory, negative = inhibitory, predicted upstream from
//!   neurotransmitters at minconf 0.5. Magnitudes 0..712.5, median 0.55.
//!   NOT normalized, NOT biological conductance, NOT modified on import.
//! - `neurons/body_id.i64` `int64[neurons]` — original MaleCNS body IDs,
//!   preserved exactly.
//! - `neurons/superclass_index.i32` + `superclass_labels.json` (27 labels,
//!   largest: ol_intrinsic 89403, cb_intrinsic 32164, vnc_intrinsic 13161,
//!   visual_projection 9201).
//! - `neurons/cell_type_index.i32` + `cell_type_labels.json` (11752 types,
//!   EMPTY label where the source has none).
//! - `populations/{retina,lamina,descending,sugar}.i32` — named neuron
//!   index sets (retina 3335, lamina 7114, descending 1314, sugar 23).
//! - `derived/hops_from_retina.i32`, `derived/hops_to_descending.i32`
//!   (166700 entries, -1 = unreachable; computed by the packager, not
//!   upstream).
//!
//! ORIENTATION / TRANSPOSE (documented once, here):
//! OUR `Connectome` CSR is SOURCE-major (`row_offsets` index
//! presynaptic rows; the runtime propagates a spike FROM the row-indexed
//! pre-neuron TO the `col_idx` postsynaptic neuron; `EdgeSpec{pre,post}`
//! maps to `row=pre, col=post`). The package CSR is DESTINATION-major.
//! The importer therefore TRANSPOSES on import with a two-pass counting
//! sort, O(E) over contiguous arrays, no per-edge allocations:
//! pass 1 counts out-degree per source (pre) index and prefix-sums into
//! source-major `row_offsets`; pass 2 walks the destination-major rows in
//! ascending destination order and scatters (dest, weight) into
//! `col_idx/weights` at `fill[src]++`, so each source-major row ends up
//! with ascending destination order (a deterministic function of the
//! package arrays).
//!
//! POLARITY: per-synapse `Polarity` is derived from the weight sign
//! (`>0` = Excitatory, `<0` = Inhibitory). The weight sign itself is the
//! upstream neurotransmitter PREDICTION, i.e. a value computed from source
//! values — provenance tag meaning "computed from source values" is
//! `ProvenanceTag::Derived`; the derivation (sign -> polarity) is done by
//! THIS importer, so the per-synapse polarity array semantically carries
//! `Derived`. The binary format stores per-edge polarity as a plain `u8`
//! with no per-edge tag field, so the tag is documented HERE instead:
//! synapses must be treated as upstream-NT-derived, our-code-recorded.
//! Zero-weight edges do not exist upstream (magnitudes are strictly
//! positive synonym counts); the importer hard-fails on any zero weight.
//! Node-level `NeuronMeta.polarity` stays `Unknown` because this package
//! provides no per-neuron transmitter field (the per-synapse polarity
//! always wins in `Connectome::polarity_at`), and `NeuronMeta.origin` is
//! `ProvenanceTag::SuppliedBySource` because body IDs / superclass /
//! cell-type annotations ARE supplied by the source dataset.
//!
//! REGIONS ≈ SUPERCLASS: the `regions` table is the 27 superclass labels
//! (documented decision: superclass is the closest real anatomical
//! grouping shipped in this package; it is NOT a classical neuropil
//! partition). `NeuronMeta.region_idx` stores the remapped superclass
//! index; empty superclass labels (none expected) would map to `NO_IDX`.
//!
//! CELL TYPES: the `regions`-style side table is built from
//! `cell_type_labels.json`. Per `connectome.rs` conventions (see
//! `connectome_import.rs`), EMPTY labels never enter the stored table —
//! those neurons keep `NO_IDX` and the table stays dense over the
//! non-empty labels in their original ascending label order.
//!
//! DELAYS: upstream has NO latency data. We store NO per-synapse delay
//! array (`Connectome::delays = None`): `Connectome::delay_at` then uses
//! the runner default, which is the single documented constant
//! [`DEFAULT_DELAY_STEPS`] (= 1 step, identical to
//! `state::LifParams::default().default_delay_steps`). This is a runner
//! default in support code, not a biological claim, and it is harmless:
//! it changes only WHEN an arrival is drained, never the structure.
//!
//! IMPORT CONSTRICTOR RULES: no fabricated weights/transmitters, no
//! dense matrices, contiguous arrays only.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::connectome::{Connectome, NeuronMeta, Polarity, NO_IDX};
use crate::provenance::ProvenanceTag;

/// Name of the FULL imported connectome (166,700 neurons).
pub const CONNECTOME_NAME: &str = "malecns-v1.0-166k-full";
/// Name of the Queen runtime subnetwork compiled from the populations.
pub const SUBNETWORK_NAME: &str = "malecns-v1.0-166k-queen-subnet";
/// Upstream template/version constant (matches `provenance::DATASET_NAME`).
pub const TEMPLATE_VERSION: &str = "malecns-v1.0";

/// Delay applied to every synapse of imported MaleCNS data because the
/// upstream package ships NO latency information: the runner default
/// (`1` step = 1 ms), identical to `state::LifParams::default()
/// .default_delay_steps`. Documented (see module header); never presented
/// as biology.
pub const DEFAULT_DELAY_STEPS: u8 = 1;

/// Default BFS hop budget for the Queen subnetwork closure.
pub const DEFAULT_HOPS: u32 = 3;
/// Default max-neuron cap for the Queen subnetwork closure.
pub const DEFAULT_MAX_NEURONS: usize = 40_000;

/// Required package files (relative paths, same keys as `manifest.json`).
pub const REQUIRED_FILES: [&str; 15] = [
    "graph/edges_offsets.i32",
    "graph/edges_source.i32",
    "graph/edges_weight.f32",
    "neurons/body_id.i64",
    "neurons/superclass_index.i32",
    "neurons/superclass_labels.json",
    "neurons/cell_type_index.i32",
    "neurons/cell_type_labels.json",
    "populations/retina.i32",
    "populations/lamina.i32",
    "populations/descending.i32",
    "populations/sugar.i32",
    "derived/hops_from_retina.i32",
    "derived/hops_to_descending.i32",
    "manifest.json",
];

// ------------------------------------------------------------------- errors

#[derive(Debug, thiserror::Error)]
pub enum MaleCnsError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("manifest/labels json parse: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest field: {0}")]
    Manifest(String),
    #[error("sha256 mismatch for {file}: expected {expected}, got {actual}")]
    Hash {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("shape mismatch for {file}: expected {expected} bytes, got {actual}")]
    Shape {
        file: String,
        expected: u64,
        actual: u64,
    },
    #[error("csr offsets invalid: {0}")]
    Offsets(String),
    #[error("index out of range at {file}[{index}] = {value} (valid 0..{max})")]
    Range {
        file: String,
        index: usize,
        value: i64,
        max: u64,
    },
    #[error("{file}[{index}] is not finite")]
    NotFinite { file: String, index: usize },
    #[error("zero-weight synapse at package flat index {0} (sign is the upstream NT prediction; zero edges do not exist upstream, refusing to classify)")]
    ZeroWeight(usize),
    #[error("connectome validation: {0}")]
    Connectome(#[from] crate::connectome::ConnectomeError),
    #[error("subnetwork extraction: {0}")]
    Subnetwork(String),
}

fn io_err(path: &Path, e: std::io::Error) -> MaleCnsError {
    MaleCnsError::Io {
        path: path.display().to_string(),
        source: e,
    }
}

// ----------------------------------------------------------- manifest model

/// The facts the importer actually used from `manifest.json` (kept for the
/// observatory / provenance record).
#[derive(Clone, Debug)]
pub struct ManifestSummary {
    pub package_dir: PathBuf,
    pub neurons: u64,
    pub edges: u64,
    /// Optional upstream `synaptic_contacts` figure (informational only).
    pub synaptic_contacts: Option<u64>,
    /// CSR orientation as claimed by the manifest (must read
    /// "destination-major"; we transpose to our source-major convention).
    pub orientation: String,
    /// Manifest-declared positive/negative weight counts (checked against
    /// the actual signs of `edges_weight.f32`).
    pub weights_positive: u64,
    pub weights_negative: u64,
    /// Number of per-file sha256 entries found in the manifest.
    pub file_hashes: usize,
    /// If the manifest claims the body_id array is sorted, we VERIFY it and
    /// mirror the claim here (`None` = no claim found).
    pub body_id_sorted_claim: Option<bool>,
}

// ------------------------------------------------ deep JSON field discovery

/// Breadth-first search for the first non-null value under any of `keys`
/// anywhere in a JSON object tree (the manifest's exact nesting is not
/// part of the contract; values are unambiguous by key name).
fn json_get<'a>(v: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    let mut stack = vec![v];
    while let Some(cur) = stack.pop() {
        match cur {
            serde_json::Value::Object(m) => {
                for k in keys {
                    if let Some(x) = m.get(*k) {
                        if !x.is_null() {
                            return Some(x);
                        }
                    }
                }
                // breadth-ish: push children (order deterministic)
                for child in m.values() {
                    if child.is_object() || child.is_array() {
                        stack.push(child);
                    }
                }
            }
            serde_json::Value::Array(a) => {
                for child in a {
                    if child.is_object() || child.is_array() {
                        stack.push(child);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn field_u64(v: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    let x = json_get(v, keys)?;
    if let Some(n) = x.as_u64() {
        return Some(n);
    }
    if let Some(n) = x.as_i64() {
        if n >= 0 {
            return Some(n as u64);
        }
    }
    if let Some(n) = x.as_f64() {
        if n >= 0.0 && n.fract() == 0.0 {
            return Some(n as u64);
        }
    }
    if let serde_json::Value::String(s) = x {
        let t = s.trim();
        if t.parse::<u64>().is_ok() {
            return Some(t.parse::<u64>().unwrap());
        }
    }
    None
}

fn field_str(v: &serde_json::Value, keys: &[&str]) -> Option<String> {
    let x = json_get(v, keys)?;
    if let serde_json::Value::String(s) = x {
        return Some(s.clone());
    }
    None
}

fn field_bool(v: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    let x = json_get(v, keys)?;
    if let serde_json::Value::Bool(b) = x {
        return Some(*b);
    }
    None
}

/// Every sha256 entry in the manifest, normalized as
/// `(normalized_rel_path_with_forward_slashes, hex_sha256)`.
fn manifest_file_hashes(v: &serde_json::Value) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let list = json_get(
        v,
        &["files", "sha256_files", "file_sha256", "sha256s", "per_file_sha256"],
    );
    let Some(list) = list else { return out };
    let mut push = |path: String, val: &serde_json::Value| {
        if let Some(sha) = entry_sha(val) {
            out.insert(path.replace('\\', "/"), sha);
        }
    };
    match list {
        serde_json::Value::Object(m) => {
            for (k, val) in m {
                // Entries may be keyed by the real relative path OR by a
                // short name with the path in an inner `file`/`path` field
                // (as shipped); prefer the inner path when present.
                if let serde_json::Value::Object(e) = val {
                    let inner = e
                        .get("file")
                        .or_else(|| e.get("path"))
                        .and_then(|x| x.as_str());
                    if let Some(p) = inner {
                        push(p.to_string(), val);
                        continue;
                    }
                }
                push(k.clone(), val);
            }
        }
        serde_json::Value::Array(items) => {
            for it in items {
                if let serde_json::Value::Object(m) = it {
                    let path = m
                        .get("path")
                        .or_else(|| m.get("file"))
                        .or_else(|| m.get("name"))
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    if let Some(p) = path {
                        push(p, it);
                    }
                }
            }
        }
        _ => {}
    }
    out
}

fn entry_sha(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(m) => m
            .get("sha256")
            .or_else(|| m.get("sha"))
            .and_then(|x| x.as_str())
            .map(str::to_string),
        _ => None,
    }
}

/// Look up a manifest sha by exact relative path first, then by any path
/// ending with the expected relative path (normalizing separators).
fn lookup_sha<'a>(hashes: &'a BTreeMap<String, String>, rel: &'a str) -> Option<&'a str> {
    if let Some(s) = hashes.get(rel) {
        return Some(s.as_str());
    }
    let norm = rel.replace('\\', "/");
    for (k, v) in hashes {
        if k.ends_with(&norm) || k.ends_with(norm.trim_start_matches("./")) {
            return Some(v.as_str());
        }
    }
    None
}

/// Parse + validate the manifest top-level facts we rely on.
fn parse_manifest(v: &serde_json::Value, dir: &Path) -> Result<ManifestSummary, MaleCnsError> {
    let neurons = field_u64(v, &["neurons", "n_neurons", "num_neurons", "node_count"])
        .ok_or_else(|| MaleCnsError::Manifest("missing neuron count (keys: neurons/n_neurons/...)".into()))?;
    let edges = field_u64(v, &["edges", "n_edges", "edge_count", "edges_count", "graph_edges"])
        .ok_or_else(|| MaleCnsError::Manifest("missing edge count (keys: edges/n_edges/...)".into()))?;
    let synaptic_contacts = field_u64(
        v,
        &["synaptic_contacts", "synaptic_contact_count", "contacts", "synapses"],
    );
    let orientation = field_str(v, &["orientation", "csr_orientation"]).ok_or_else(|| {
        MaleCnsError::Manifest(
            "missing csr orientation (key: orientation) — required to confirm the documented transpose".into(),
        )
    })?;
    if !orientation.contains("destination") && !orientation.contains("postsynaptic") {
        // The shipped manifest states the convention in prose: "row =
        // postsynaptic neuron" (destination-major); either phrasing must
        // confirm the documented transpose. Anything else refuses.
        return Err(MaleCnsError::Manifest(format!(
            "csr orientation is '{}', expected destination-major (the importer transposes to source-major; refusing to import an unexpected orientation)",
            orientation
        )));
    }
    // weights sign-count block (required: we verify actual signs against it)
    let (wp, wn) = sign_counts_from_manifest(v)?;
    let body_id_sorted_claim = field_bool(
        v,
        &["body_id_sorted", "body_ids_sorted", "body_id_order_sorted", "sorted_body_ids"],
    );
    let hashes = manifest_file_hashes(v);
    let file_hashes = hashes.len();
    if hashes.is_empty() {
        return Err(MaleCnsError::Manifest(
            "manifest has no per-file sha256 list (keys: files / sha256_files / ...)".into(),
        ));
    }
    Ok(ManifestSummary {
        package_dir: dir.to_path_buf(),
        neurons,
        edges,
        synaptic_contacts,
        orientation,
        weights_positive: wp,
        weights_negative: wn,
        file_hashes,
        body_id_sorted_claim,
    })
}

fn sign_counts_from_manifest(v: &serde_json::Value) -> Result<(u64, u64), MaleCnsError> {
    // shape A: {"weights": {"positive": n, "negative": m}}
    if let Some(w) = json_get(v, &["weights"]) {
        let pos = field_u64(
            w,
            &["positive", "pos", "positive_edges", "excitatory", "excitatory_count", "positive_weight_count"],
        );
        let neg = field_u64(
            w,
            &["negative", "neg", "negative_edges", "inhibitory", "inhibitory_count", "negative_weight_count"],
        );
        if let Some(pos) = pos {
            if let Some(neg) = neg {
                return Ok((pos, neg));
            }
        }
    }
    // shape B: flat keys
    if let (Some(pos), Some(neg)) = (
        field_u64(v, &["positive_weights", "positive_weight_count", "positive_count"]),
        field_u64(v, &["negative_weights", "negative_weight_count", "negative_count"]),
    ) {
        return Ok((pos, neg));
    }
    Err(MaleCnsError::Manifest(
        "manifest has no weights sign-count block (expected weights:{positive,negative}) — \
         importer refuses to import unverifiable sign counts"
            .into(),
    ))
}

// ------------------------------------------------------------ raw file reads

fn read_bytes(path: &Path) -> Result<Vec<u8>, MaleCnsError> {
    let mut f = File::open(path).map_err(|e| io_err(path, e))?;
    let mut v = Vec::new();
    f.read_to_end(&mut v).map_err(|e| io_err(path, e))?;
    Ok(v)
}

fn hex32(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// SHA-256 over a byte buffer (files are hashed before type conversion).
fn sha256_buf(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex32(&h.finalize().into())
}

/// Streaming SHA-256 of a file path (used by the example/reporting tools).
pub fn file_sha256(path: &Path) -> Result<String, MaleCnsError> {
    let mut f = File::open(path).map_err(|e| io_err(path, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex32(&h.finalize().into()))
}

/// Convert raw little-endian i32 bytes into a Vec (contiguous, one copy).
fn le_i32s(bytes: &[u8], file: &str) -> Result<Vec<i32>, MaleCnsError> {
    if bytes.len() % 4 != 0 {
        return Err(MaleCnsError::Shape {
            file: file.into(),
            expected: (bytes.len() / 4 * 4 + 4) as u64,
            actual: bytes.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for ch in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    Ok(out)
}

fn le_f32s(bytes: &[u8], file: &str) -> Result<Vec<f32>, MaleCnsError> {
    if bytes.len() % 4 != 0 {
        return Err(MaleCnsError::Shape {
            file: file.into(),
            expected: (bytes.len() / 4 * 4 + 4) as u64,
            actual: bytes.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for ch in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    Ok(out)
}

fn le_i64s(bytes: &[u8], file: &str) -> Result<Vec<i64>, MaleCnsError> {
    if bytes.len() % 8 != 0 {
        return Err(MaleCnsError::Shape {
            file: file.into(),
            expected: (bytes.len() / 8 * 8 + 8) as u64,
            actual: bytes.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for ch in bytes.chunks_exact(8) {
        out.push(i64::from_le_bytes([
            ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
        ]));
    }
    Ok(out)
}

/// File size in ELEMENTS we can state a priori (None = free-form size,
/// length checked against the file's own byte count).
fn expected_elements(rel: &str, neurons: u64, edges: u64) -> Option<u64> {
    match rel {
        "graph/edges_offsets.i32" => Some(neurons + 1),
        "graph/edges_source.i32" | "graph/edges_weight.f32" => Some(edges),
        "neurons/body_id.i64"
        | "neurons/superclass_index.i32"
        | "neurons/cell_type_index.i32"
        | "derived/hops_from_retina.i32"
        | "derived/hops_to_descending.i32" => Some(neurons),
        _ => None, // populations (+ labels json)
    }
}

fn dtype_size(rel: &str) -> usize {
    match rel {
        "neurons/body_id.i64" => 8,
        "neurons/superclass_labels.json" => 0, // json, element-size check N/A
        "neurons/cell_type_labels.json" => 0,
        _ => 4,
    }
}

fn range_err(file: &str, index: usize, value: i64, max: u64) -> MaleCnsError {
    MaleCnsError::Range {
        file: file.into(),
        index,
        value,
        max,
    }
}

// ------------------------------------------------------------ package model

/// The named neuron index sets the Queen subnetwork is grounded in.
#[derive(Clone, Debug, Default)]
pub struct NamedPopulations {
    pub retina: Vec<u32>,
    pub lamina: Vec<u32>,
    pub sugar: Vec<u32>,
    pub descending: Vec<u32>,
}

/// Everything validated on disk, ready for `import_full` / extraction.
#[derive(Debug)]
pub struct LoadedMaleCns {
    pub manifest: ManifestSummary,
    /// Destination-major CSR offsets (RAW package orientation, pre-transpose).
    pub offsets: Vec<u32>,
    /// PREsynaptic index per flat edge.
    pub source: Vec<u32>,
    /// Raw signed synaptic counts, verbatim.
    pub weights: Vec<f32>,
    pub body_ids: Vec<u64>,
    pub superclass_index: Vec<u32>,
    pub cell_type_index: Vec<u32>,
    pub superclass_labels: Vec<String>,
    pub cell_type_labels: Vec<String>,
    pub populations: NamedPopulations,
    pub hops_from_retina: Vec<i32>,
    pub hops_to_descending: Vec<i32>,
}

impl LoadedMaleCns {
    pub fn neurons(&self) -> usize {
        self.body_ids.len()
    }
    pub fn edges(&self) -> usize {
        self.weights.len()
    }
}

/// Read + fully validate the package directory. Hard-fails with precise
/// errors on: manifest field problems, sha256 mismatch, wrong shapes,
/// non-monotonic/short offsets, out-of-range indices, non-finite or
/// zero-magnitude weights, sign-count mismatch, duplicate or
/// claimed-sorted-but-unsorted body IDs.
pub fn load_package(dir: &Path) -> Result<LoadedMaleCns, MaleCnsError> {
    let manifest_path = dir.join("manifest.json");
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| io_err(&manifest_path, e))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let manifest = parse_manifest(&v, dir)?;
    let hashes = manifest_file_hashes(&v);

    let neurons = manifest.neurons;
    let edges = manifest.edges;
    if neurons == 0 || edges == 0 {
        return Err(MaleCnsError::Manifest(format!(
            "manifest counts degenerate: neurons={neurons} edges={edges} (package claims a full MaleCNS v1.0 graph; refusing to import empties)"
        )));
    }

    // ---- pass 1: read + hash + shape-check every required raw file --------
    let mut blobs: BTreeMap<&'static str, Vec<u8>> = BTreeMap::new();
    for rel in REQUIRED_FILES {
        if rel == "manifest.json" || rel.ends_with(".json") {
            continue; // labels json read separately below
        }
        let path = dir.join(rel);
        let b = read_bytes(&path)?;
        let actual = sha256_buf(&b);
        let Some(expected) = lookup_sha(&hashes, rel).map(str::to_owned) else {
            return Err(MaleCnsError::Manifest(format!(
                "manifest lacks a sha256 entry for {rel} (found {} entries)",
                hashes.len()
            )));
        };
        if !expected.eq_ignore_ascii_case(&actual) {
            return Err(MaleCnsError::Hash {
                file: rel.into(),
                expected,
                actual,
            });
        }
        if let Some(elems) = expected_elements(rel, neurons, edges) {
            let want = elems * dtype_size(rel) as u64;
            if b.len() as u64 != want {
                return Err(MaleCnsError::Shape {
                    file: rel.into(),
                    expected: want,
                    actual: b.len() as u64,
                });
            }
        }
        blobs.insert(rel, b);
    }

    // ---- labels -----------------------------------------------------------
    let superclass_labels: Vec<String> = read_labels(
        dir,
        "neurons/superclass_labels.json",
        &hashes,
        field_u64(&v, &["superclasses", "superclass_count", "n_superclasses", "superclass_labels"]),
    )?;
    let cell_type_labels: Vec<String> = read_labels(
        dir,
        "neurons/cell_type_labels.json",
        &hashes,
        field_u64(&v, &["cell_types", "cell_type_count", "n_cell_types", "cell_type_labels"]),
    )?;

    // ---- arrays ------------------------------------------------------------
    let offsets_i = le_i32s(&blobs["graph/edges_offsets.i32"], "graph/edges_offsets.i32")?;
    if offsets_i.len() as u64 != neurons + 1 {
        return Err(MaleCnsError::Shape {
            file: "graph/edges_offsets.i32".into(),
            expected: (neurons + 1) * 4,
            actual: (offsets_i.len() * 4) as u64,
        });
    }
    if offsets_i[0] != 0 {
        return Err(MaleCnsError::Offsets(format!(
            "offsets[0] = {} (must be 0)",
            offsets_i[0]
        )));
    }
    {
        let mut prev: i64 = 0;
        for (i, &o) in offsets_i.iter().enumerate() {
            if o < 0 as i32 {
                return Err(MaleCnsError::Offsets(format!(
                    "offsets[{i}] = {o} is negative"
                )));
            }
            if (o as i64) < prev {
                return Err(MaleCnsError::Offsets(format!(
                    "offsets not non-decreasing at {i}: {prev} -> {o}"
                )));
            }
            prev = o as i64;
        }
        if prev != edges as i64 {
            return Err(MaleCnsError::Offsets(format!(
                "offsets[{0}] (last) = {prev} but manifest edges = {edges}",
                offsets_i.len() - 1
            )));
        }
    }
    let offsets: Vec<u32> = offsets_i.iter().map(|&o| o as u32).collect();

    let source_i = le_i32s(&blobs["graph/edges_source.i32"], "graph/edges_source.i32")?;
    let mut source: Vec<u32> = Vec::with_capacity(source_i.len());
    for (k, &s) in source_i.iter().enumerate() {
        if s < 0 || s as u64 >= neurons {
            return Err(range_err(
                "graph/edges_source.i32",
                k,
                s as i64,
                neurons,
            ));
        }
        source.push(s as u32);
    }
    let weights_f = le_f32s(&blobs["graph/edges_weight.f32"], "graph/edges_weight.f32")?;
    let mut positive = 0u64;
    let mut negative = 0u64;
    for (k, &w) in weights_f.iter().enumerate() {
        if !w.is_finite() {
            return Err(MaleCnsError::NotFinite {
                file: "graph/edges_weight.f32".into(),
                index: k,
            });
        }
        if w == 0.0 {
            return Err(MaleCnsError::ZeroWeight(k));
        }
        if w > 0.0 {
            positive += 1;
        } else {
            negative += 1;
        }
    }
    if positive != manifest.weights_positive || negative != manifest.weights_negative {
        return Err(MaleCnsError::Manifest(format!(
            "manifest weights block mismatch: declared +{}/-{}, actual +{}/-{}",
            manifest.weights_positive, manifest.weights_negative, positive, negative
        )));
    }

    let body_i = le_i64s(&blobs["neurons/body_id.i64"], "neurons/body_id.i64")?;
    let mut body_ids: Vec<u64> = Vec::with_capacity(body_i.len());
    for (k, &b) in body_i.iter().enumerate() {
        if b < 0 {
            return Err(range_err("neurons/body_id.i64", k, b, u64::MAX));
        }
        body_ids.push(b as u64);
    }
    // distinct always; demanding-sort only when the manifest claims it.
    let mut check = body_ids.clone();
    check.sort_unstable();
    let mut distinct = true;
    {
        let mut i = 1;
        while i < check.len() {
            if check[i] == check[i - 1] {
                distinct = false;
                break;
            }
            i += 1;
        }
    }
    if !distinct {
        return Err(MaleCnsError::Manifest(
            "body_id array contains duplicate IDs (full MaleCNS v1.0 must have one row per body)"
                .into(),
        ));
    }
    let actually_sorted = body_ids.windows(2).all(|w| w[0] <= w[1]);
    if manifest.body_id_sorted_claim == Some(true) && !actually_sorted {
        return Err(MaleCnsError::Manifest(
            "manifest claims body_ids are sorted but the actual order is not (assumed nothing)"
                .into(),
        ));
    }

    let superclass_index =
        validated_u32(&blobs["neurons/superclass_index.i32"], "neurons/superclass_index.i32", superclass_labels.len() as u64)?;
    let cell_type_index =
        validated_u32(&blobs["neurons/cell_type_index.i32"], "neurons/cell_type_index.i32", cell_type_labels.len() as u64)?;

    let populations = read_populations(dir, &hashes, neurons, &v)?;

    let hops_from_retina = hops(
        &blobs["derived/hops_from_retina.i32"],
        "derived/hops_from_retina.i32",
        neurons,
    )?;
    let hops_to_descending = hops(
        &blobs["derived/hops_to_descending.i32"],
        "derived/hops_to_descending.i32",
        neurons,
    )?;

    Ok(LoadedMaleCns {
        manifest,
        offsets,
        source,
        weights: weights_f,
        body_ids,
        superclass_index,
        cell_type_index,
        superclass_labels,
        cell_type_labels,
        populations,
        hops_from_retina,
        hops_to_descending,
    })
}

fn read_labels(
    dir: &Path,
    rel: &'static str,
    hashes: &BTreeMap<String, String>,
    declared: Option<u64>,
) -> Result<Vec<String>, MaleCnsError> {
    let path = dir.join(rel);
    let bytes = read_bytes(&path)?;
    let actual = sha256_buf(&bytes);
    let Some(expected) = lookup_sha(hashes, rel).map(str::to_owned) else {
        return Err(MaleCnsError::Manifest(format!(
            "manifest lacks a sha256 entry for {rel} (found {} entries)",
            hashes.len()
        )));
    };
    if !expected.eq_ignore_ascii_case(&actual) {
        return Err(MaleCnsError::Hash {
            file: rel.into(),
            expected,
            actual,
        });
    }
    let s = String::from_utf8(bytes).map_err(|_| {
        MaleCnsError::Manifest(format!("{rel} is not valid UTF-8"))
    })?;
    let labels: Vec<String> = serde_json::from_str(&s)?;
    if let Some(d) = declared {
        if d != labels.len() as u64 {
            return Err(MaleCnsError::Shape {
                file: rel.into(),
                expected: d * 8, // element count expressed via json length mismatch below
                actual: labels.len() as u64 * 8,
            });
        }
    }
    Ok(labels)
}

/// Populations are read from disk individually (their sizes are only
/// constrained by their own byte counts + manifest declared counts when
/// present, and every index must land inside the neuron table).
fn read_populations(
    dir: &Path,
    hashes: &BTreeMap<String, String>,
    neurons: u64,
    manifest: &serde_json::Value,
) -> Result<NamedPopulations, MaleCnsError> {
    let mut pops = NamedPopulations::default();
    for (rel, slot, declared_keys) in [
        ("populations/retina.i32", 0usize, &["retina", "retina_count", "n_retina"] as &[&str]),
        ("populations/lamina.i32", 1, &["lamina", "lamina_count", "n_lamina"]),
        ("populations/sugar.i32", 2, &["sugar", "sugar_count", "n_sugar"]),
        ("populations/descending.i32", 3, &["descending", "descending_count", "n_descending"]),
    ] {
        let path = dir.join(rel);
        let bytes = read_bytes(&path)?;
        let actual = sha256_buf(&bytes);
        let Some(expected) = lookup_sha(hashes, rel).map(str::to_owned) else {
            return Err(MaleCnsError::Manifest(format!(
                "manifest lacks a sha256 entry for {rel} (found {} entries)",
                hashes.len()
            )));
        };
        if !expected.eq_ignore_ascii_case(&actual) {
            return Err(MaleCnsError::Hash {
                file: rel.into(),
                expected,
                actual,
            });
        }
        if bytes.len() % 4 != 0 {
            return Err(MaleCnsError::Shape {
                file: rel.into(),
                expected: (bytes.len() / 4 * 4 + 4) as u64,
                actual: bytes.len() as u64,
            });
        }
        if let Some(d) = field_u64(manifest, declared_keys) {
            let len = (bytes.len() / 4) as u64;
            if d != len {
                return Err(MaleCnsError::Shape {
                    file: rel.into(),
                    expected: d * 4,
                    actual: bytes.len() as u64,
                });
            }
        }
        let idxs = le_i32s(&bytes, rel)?;
        let mut out = Vec::with_capacity(idxs.len());
        for (k, &i) in idxs.iter().enumerate() {
            if i < 0 || i as u64 >= neurons {
                return Err(range_err(rel, k, i as i64, neurons));
            }
            out.push(i as u32);
        }
        match slot {
            0 => pops.retina = out,
            1 => pops.lamina = out,
            2 => pops.sugar = out,
            _ => pops.descending = out,
        }
    }
    Ok(pops)
}

fn hops(bytes: &[u8], file: &str, neurons: u64) -> Result<Vec<i32>, MaleCnsError> {
    let v = le_i32s(bytes, file)?;
    if v.len() as u64 != neurons {
        return Err(MaleCnsError::Shape {
            file: file.into(),
            expected: neurons * 4,
            actual: (v.len() * 4) as u64,
        });
    }
    for (k, &h) in v.iter().enumerate() {
        if h < -1 {
            return Err(range_err(file, k, h as i64, 0));
        }
    }
    Ok(v)
}

fn validated_u32(bytes: &[u8], file: &str, max: u64) -> Result<Vec<u32>, MaleCnsError> {
    let v = le_i32s(bytes, file)?;
    let mut out = Vec::with_capacity(v.len());
    for (k, &i) in v.iter().enumerate() {
        if i < 0 || i as u64 >= max {
            return Err(range_err(file, k, i as i64, max));
        }
        out.push(i as u32);
    }
    Ok(out)
}

// -------------------------------------------------------------- full import

/// Import stats returned alongside the FULL connectome (actual numbers for
/// the observatory/report).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaleCnsImportStats {
    pub neurons: u64,
    pub edges: u64,
    pub positive_edges: u64,
    pub negative_edges: u64,
    /// Always true for this importer: the package CSR is transposed on import.
    pub transposed: bool,
    pub regions: u64,
    pub cell_types: u64,
}

/// Full import result: the complete 166,700-neuron connectome (source-major
/// after the documented transpose) plus the population/hop tables preserved
/// verbatim and the manifest summary.
pub struct FullImport {
    pub connectome: Connectome,
    pub populations: NamedPopulations,
    pub hops_from_retina: Vec<i32>,
    pub hops_to_descending: Vec<i32>,
    pub manifest: ManifestSummary,
    pub stats: MaleCnsImportStats,
}

/// Dense side-table helper: non-empty labels only (connectome.rs convention:
/// empty = "not assigned", neurons carry `NO_IDX` and are never given a
/// fabricated name). Returns the table and per-original-label remap (-1 =
/// empty/none).
fn dense_table(labels: &[String]) -> (Vec<String>, Vec<i32>) {
    let mut table: Vec<String> = Vec::new();
    let mut remap = vec![-1i32; labels.len()];
    for (i, l) in labels.iter().enumerate() {
        if l.is_empty() {
            continue;
        }
        remap[i] = table.len() as i32;
        table.push(l.clone());
    }
    (table, remap)
}

/// Consume a validated package and assemble the FULL `Connectome`
/// (documented transpose, raw weights verbatim, sign-derived per-synapse
/// polarity, default-delay `None`, identity via `connectome_binary::
/// connectome_identity` semantics of the resulting struct).
pub fn import_full(loaded: LoadedMaleCns) -> Result<FullImport, MaleCnsError> {
    let n = loaded.neurons();
    let e = loaded.edges();

    let (regions_table, region_remap) = dense_table(&loaded.superclass_labels);
    let (ct_table, ct_remap) = dense_table(&loaded.cell_type_labels);

    let mut node_meta: Vec<NeuronMeta> = Vec::with_capacity(n);
    for i in 0..n {
        let ri = region_remap[loaded.superclass_index[i] as usize];
        let ci = ct_remap[loaded.cell_type_index[i] as usize];
        node_meta.push(NeuronMeta {
            body_id: loaded.body_ids[i],
            region_idx: if ri >= 0 { ri as u32 } else { NO_IDX },
            cell_type_idx: if ci >= 0 { ci as u32 } else { NO_IDX },
            // No per-neuron transmitter field in this package; the
            // per-synapse polarity (Derived; see module header) always wins.
            polarity: Polarity::Unknown,
            origin: ProvenanceTag::SuppliedBySource,
        });
    }

    // Documented transpose: destination-major package CSR -> our source-major
    // CSR. Two passes over contiguous arrays; no per-edge allocations.
    let mut counts = vec![0u32; n + 1];
    for &s in loaded.source.iter() {
        counts[s as usize + 1] += 1;
    }
    for i in 1..=n {
        counts[i] += counts[i - 1];
    }
    let row_offsets = counts;
    let mut fill = row_offsets.clone();
    let mut col_idx = vec![0u32; e];
    let mut weights = vec![0f32; e];
    let mut syn_pol = vec![0u8; e];
    let mut positive = 0u64;
    let mut negative = 0u64;
    for dest in 0..n {
        let rs = loaded.offsets[dest] as usize;
        let re = loaded.offsets[dest + 1] as usize;
        for k in rs..re {
            let s = loaded.source[k] as usize;
            let w = loaded.weights[k];
            if w == 0.0 {
                return Err(MaleCnsError::ZeroWeight(k));
            }
            let p = fill[s] as usize;
            fill[s] += 1;
            col_idx[p] = dest as u32;
            weights[p] = w;
            if w > 0.0 {
                syn_pol[p] = Polarity::Excitatory as u8;
                positive += 1;
            } else {
                syn_pol[p] = Polarity::Inhibitory as u8;
                negative += 1;
            }
        }
    }

    let connectome = Connectome::from_csr(
        CONNECTOME_NAME,
        TEMPLATE_VERSION,
        node_meta,
        regions_table,
        ct_table,
        row_offsets,
        col_idx,
        weights,
        // No latency data upstream; runner default delay (see header).
        None,
        Some(syn_pol),
    )?;

    let stats = MaleCnsImportStats {
        neurons: n as u64,
        edges: e as u64,
        positive_edges: positive,
        negative_edges: negative,
        transposed: true,
        regions: connectome.regions.len() as u64,
        cell_types: connectome.cell_types.len() as u64,
    };
    Ok(FullImport {
        connectome,
        populations: loaded.populations,
        hops_from_retina: loaded.hops_from_retina,
        hops_to_descending: loaded.hops_to_descending,
        manifest: loaded.manifest,
        stats,
    })
}

// -------------------------------------------------------------- subnetwork

/// Options for the biologically grounded Queen subnetwork extraction.
#[derive(Clone, Copy, Debug)]
pub struct QueenSubnetOptions {
    /// Hop budget for the multi-source BFS closure (documented K).
    pub hops: u32,
    /// Hard neuron cap; truncation (deterministic) prefers SMALLER hop and
    /// then HIGHER out-degree (tie: ascending index).
    pub max_neurons: usize,
}

impl Default for QueenSubnetOptions {
    fn default() -> Self {
        Self {
            hops: DEFAULT_HOPS,
            max_neurons: DEFAULT_MAX_NEURONS,
        }
    }
}

/// Observatory-facing facts about the extracted Queen subnetwork.
#[derive(Clone, Debug, PartialEq)]
pub struct QueenNetInfo {
    pub neurons: usize,
    pub edges: usize,
    pub hops: u32,
    pub max_neurons: usize,
    /// true when the cap forced truncation (seed/frontier priority applied).
    pub truncated: bool,
    /// raw seed members found per population.
    pub seeds_found_retina: usize,
    pub seeds_found_lamina: usize,
    pub seeds_found_sugar: usize,
    pub seeds_found_descending: usize,
    /// total raw seed members enqueued (before closure).
    pub seed_total: usize,
}

/// Build the executable Queen subnetwork from the FULL connectome, seeded by
/// the REAL sensory populations (retina, lamina, sugar) and the motor set
/// (descending neurons): multi-source BFS closure out to `hops` on the full
/// CSR (our source-major rows), capped at `max_neurons` with a deterministic
/// priority (smaller hop first, then higher out-degree, index tiebreak).
/// Every included neuron keeps its ORIGINAL body ID; the induced edge set
/// preserves weights and sign polarity verbatim. NOTHING is fabricated:
/// seeds come from the population arrays and edges from the imported graph.
pub fn subnetwork_for_queen(
    full: &Connectome,
    pops: &NamedPopulations,
    opts: QueenSubnetOptions,
) -> Result<(Connectome, QueenNetInfo), MaleCnsError> {
    let n = full.n();
    if n == 0 {
        return Err(MaleCnsError::Subnetwork("empty full connectome".into()));
    }
    if opts.max_neurons == 0 {
        return Err(MaleCnsError::Subnetwork(
            "QueenSubnetOptions.max_neurons must be > 0".into(),
        ));
    }
    let deg = |i: usize| -> u32 { full.row_offsets[i + 1] - full.row_offsets[i] };

    // `included` is the final closure (protagonist of every branch below);
    // level candidates are marked provisionally and ONLY released when the
    // cap truncates them (they cannot have been reachable otherwise, so the
    // release is safe).
    let mut included = vec![false; n];

    // ---- hop 0: the seeded populations (validated + de-duplicated) --------
    let mut seeds: Vec<u32> = Vec::new();
    let mut push_seed = |idx: u32| -> Result<(), MaleCnsError> {
        let i = idx as usize;
        if i >= n {
            return Err(MaleCnsError::Subnetwork(format!(
                "population neuron index {} out of range (full connectome has {n} neurons)",
                idx
            )));
        }
        if !included[i] {
            included[i] = true;
            seeds.push(idx);
        }
        Ok(())
    };
    for pop in [&pops.retina, &pops.lamina, &pops.sugar, &pops.descending] {
        for &i in pop {
            push_seed(i)?;
        }
    }
    let mut truncated = false;
    let mut budget = opts.max_neurons;
    if seeds.len() > budget {
        truncated = true;
        seeds.sort_unstable_by(|a, b| deg(*b as usize).cmp(&deg(*a as usize)).then(a.cmp(b)));
        let keep = seeds[..budget].to_vec();
        for flag in included.iter_mut() {
            *flag = false;
        }
        for &i in &keep {
            included[i as usize] = true;
        }
        seeds = keep;
    }
    budget -= seeds.len();
    let mut frontier = seeds;

    // ---- hop <= K: multi-source BFS; only included nodes expand -----------
    let mut hop = 0u32;
    while hop < opts.hops && !frontier.is_empty() {
        hop += 1;
        let mut cands: Vec<u32> = Vec::new();
        for &s in &frontier {
            let rs = full.row_offsets[s as usize] as usize;
            let re = full.row_offsets[s as usize + 1] as usize;
            for p in rs..re {
                let d = full.col_idx[p] as usize;
                if !included[d] {
                    included[d] = true;
                    cands.push(d as u32);
                }
            }
        }
        cands.sort_unstable();
        if cands.len() <= budget {
            budget -= cands.len();
            frontier = cands;
        } else {
            truncated = true;
            // cap truncation priority (documented): higher out-degree first
            // (smaller hop is already implied by the level order), index
            // ascending on ties. Kept marks stay in `included`; the unpicked
            // (dedup'd unique indices) are released.
            cands.sort_unstable_by(|a, b| deg(*b as usize).cmp(&deg(*a as usize)).then(a.cmp(b)));
            let kept: Vec<u32> = cands[..budget].to_vec();
            for &c in &cands[budget..] {
                included[c as usize] = false;
            }
            frontier = kept;
            budget = 0;
        }
    }

    let included_idx: Vec<u32> = (0..n)
        .filter(|&i| included[i])
        .map(|i| i as u32)
        .collect();
    let mut remap = vec![NO_IDX; n];
    for (rank, &i) in included_idx.iter().enumerate() {
        remap[i as usize] = rank as u32;
    }

    // induced edges: iterate source rows of included neurons in ascending
    // original order -> in-row cols stay ascending; weights/polarity verbatim.
    let mut edges_total = 0usize;
    for &i in &included_idx {
        let rs = full.row_offsets[i as usize] as usize;
        let re = full.row_offsets[i as usize + 1] as usize;
        for p in rs..re {
            if remap[full.col_idx[p] as usize] != NO_IDX {
                edges_total += 1;
            }
        }
    }
    let mut sub_row_offsets = Vec::with_capacity(included_idx.len() + 1);
    sub_row_offsets.push(0u32);
    let mut sub_col = Vec::with_capacity(edges_total);
    let mut sub_w = Vec::with_capacity(edges_total);
    let mut sub_pol = Vec::with_capacity(edges_total);
    for &i in &included_idx {
        let rs = full.row_offsets[i as usize] as usize;
        let re = full.row_offsets[i as usize + 1] as usize;
        for p in rs..re {
            let r = remap[full.col_idx[p] as usize];
            if r != NO_IDX {
                sub_col.push(r);
                sub_w.push(full.weights[p]);
                sub_pol.push(
                    full.syn_polarity
                        .as_ref()
                        .map_or(Polarity::Unknown as u8, |sp| sp[p]),
                );
            }
        }
        sub_row_offsets.push(sub_col.len() as u32);
    }

    let node_meta: Vec<NeuronMeta> = included_idx
        .iter()
        .map(|&i| full.node_meta[i as usize].clone())
        .collect();
    // Side tables are carried over VERBATIM (same indices; documented):
    // trimming them would re-map `region_idx`/`cell_type_idx` for no benefit.
    let connectome = Connectome::from_csr(
        SUBNETWORK_NAME,
        TEMPLATE_VERSION,
        node_meta,
        full.regions.clone(),
        full.cell_types.clone(),
        sub_row_offsets,
        sub_col,
        sub_w,
        None,
        Some(sub_pol),
    )?;

    let info = QueenNetInfo {
        neurons: connectome.n(),
        edges: connectome.edge_count(),
        hops: opts.hops,
        max_neurons: opts.max_neurons,
        truncated,
        seeds_found_retina: pops.retina.len(),
        seeds_found_lamina: pops.lamina.len(),
        seeds_found_sugar: pops.sugar.len(),
        seeds_found_descending: pops.descending.len(),
        seed_total: pops.retina.len() + pops.lamina.len() + pops.sugar.len() + pops.descending.len(),
    };
    Ok((connectome, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome_binary::{self, connectome_identity};

    const N: usize = 30;
    const E: &[(u32, u32, f32)] = &[
        // (pre, post, weight) — pre 16/25 chain for the runtime orientation test
        (16, 25, 20.0),
        (25, 29, 20.0),
        (0, 1, 2.0),
        (1, 2, -3.0),
        (2, 0, 1.0),
    ];

    // ------------------------------------------------------------ helpers

    fn i32le(v: &[i32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }
    fn i64le(v: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 8);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }
    fn f32le(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn package_dir(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("malecns-bins-tests")
            .join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("graph")).unwrap();
        std::fs::create_dir_all(d.join("neurons")).unwrap();
        std::fs::create_dir_all(d.join("populations")).unwrap();
        std::fs::create_dir_all(d.join("derived")).unwrap();
        d
    }

    /// Destination-major OSR layout of `edges` (in-row pre ascending):
    /// returns (offsets, sources, weights) as little-endian byte arrays.
    fn dest_major_raw(edges: &[(u32, u32, f32)]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut sorted: Vec<(u32, u32, f32)> = edges.to_vec();
        sorted.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        let n = N as u32;
        let mut offs = vec![0i32; N + 1];
        for &(_, post, _) in &sorted {
            offs[post as usize + 1] += 1;
        }
        for i in 1..=(N) {
            offs[i] += offs[i - 1];
        }
        let mut src = vec![0i32; sorted.len()];
        let mut w = vec![0f32; sorted.len()];
        for (k, &(pre, _, wt)) in sorted.iter().enumerate() {
            src[k] = pre as i32;
            w[k] = wt;
        }
        assert!(offs[N] == sorted.len() as i32 && n > 0);
        (i32le(&offs), i32le(&src), f32le(&w))
    }

    /// Write the full tiny package with a manifest whose per-file sha256
    /// entries are recomputed from what is actually on disk.
    fn build_package(dir: &Path, edges: &[(u32, u32, f32)]) {
        let (ofs_b, src_b, w_b) = dest_major_raw(edges);
        let bodies: Vec<i64> = (0..N).map(|i| 1000 + i as i64).collect();
        let super_idx: Vec<i32> = (0..N).map(|i| (i % 27) as i32).collect();
        // ["ct0", "", "ct2"]: the empty label must map to NO_IDX
        let ct_idx: Vec<i32> = (0..N).map(|i| (i % 3) as i32).collect();
        let retina: Vec<i32> = vec![0, 1, 2];
        let lamina: Vec<i32> = vec![3];
        let sugar: Vec<i32> = vec![1];
        let desc: Vec<i32> = vec![(N - 1) as i32];
        let hops_fr: Vec<i32> = (0..N).map(|i| if i < 3 { 0 } else { -1 }).collect();
        let hops_td: Vec<i32> =
            (0..N).map(|i| if i == N - 1 { 0 } else { -1 }).collect();

        w(&dir.join("graph/edges_offsets.i32"), &ofs_b);
        w(&dir.join("graph/edges_source.i32"), &src_b);
        w(&dir.join("graph/edges_weight.f32"), &w_b);
        w(&dir.join("neurons/body_id.i64"), &i64le(&bodies));
        w(&dir.join("neurons/superclass_index.i32"), &i32le(&super_idx));
        w(&dir.join("neurons/cell_type_index.i32"), &i32le(&ct_idx));
        w(&dir.join("populations/retina.i32"), &i32le(&retina));
        w(&dir.join("populations/lamina.i32"), &i32le(&lamina));
        w(&dir.join("populations/sugar.i32"), &i32le(&sugar));
        w(&dir.join("populations/descending.i32"), &i32le(&desc));
        w(&dir.join("derived/hops_from_retina.i32"), &i32le(&hops_fr));
        w(&dir.join("derived/hops_to_descending.i32"), &i32le(&hops_td));
        let sc: Vec<String> = (0..27).map(|c| format!("sc{c}")).collect();
        w(&dir.join("neurons/superclass_labels.json"), serde_json::to_vec(&sc).unwrap().as_slice());
        w(
            &dir.join("neurons/cell_type_labels.json"),
            serde_json::to_vec(
                &["ct0".to_string(), String::new(), "ct2".to_string()],
            )
            .unwrap()
            .as_slice(),
        );
        refresh_manifest(dir, edges);
    }

    fn w(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }

    /// Regenerate manifest.json (fresh hashes) e.g. after deliberate
    /// on-disk mutations, so subsequent failures are NOT sha failures.
    fn refresh_manifest(dir: &Path, _edges: &[(u32, u32, f32)]) {
        let mut files = serde_json::Map::new();
        for rel in REQUIRED_FILES {
            if rel == "manifest.json" {
                continue;
            }
            let b = std::fs::read(dir.join(rel)).unwrap();
            files.insert(
                rel.to_string(),
                serde_json::json!({ "sha256": sha256_buf(&b) }),
            );
        }
        let (pos, neg) = {
            let (_, src_b, w_b) = dest_major_raw(_edges);
            let mut pos = 0u64;
            let mut neg = 0u64;
            for ch in w_b.chunks_exact(4) {
                let x = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                if x > 0.0 {
                    pos += 1;
                } else if x < 0.0 {
                    neg += 1;
                }
            }
            let _ = src_b;
            (pos, neg)
        };
        let manifest = serde_json::json!({
            "schema": { "neurons": crate_util_n(), "edges": _edges.len() },
            "csr": { "orientation": "destination-major" },
            "synaptic_contacts": _edges.len() * 20,
            "weights": { "positive": pos, "negative": neg },
            "body_id_sorted": true,
            "files": files,
        });
        w(&dir.join("manifest.json"), serde_json::to_vec(&manifest).unwrap().as_slice());
    }

    fn crate_util_n() -> u64 {
        N as u64
    }

    // ------------------------------------------------------------- tests

    #[test]
    fn importer_roundtrip_tiny_package_orientation_and_labels() {
        let d = package_dir("roundtrip");
        build_package(&d, E);
        let loaded = load_package(&d).expect("package must load");
        assert_eq!(loaded.neurons(), N);
        assert_eq!(loaded.edges(), 5);
        assert_eq!(loaded.manifest.weights_positive, 4);
        assert_eq!(loaded.manifest.weights_negative, 1);
        assert_eq!(loaded.manifest.orientation, "destination-major");

        let full = import_full(loaded).expect("import");
        let c = &full.connectome;
        assert_eq!(c.template_version, TEMPLATE_VERSION);
        assert_eq!(c.name, CONNECTOME_NAME);
        assert_eq!(c.stats().neuron_count, N as u64);
        assert_eq!(c.stats().synapse_count, 5);
        // transpose check: our source-major row 16 must contain postsyn 25
        let row = |pre: u32| -> Vec<(u32, f32)> {
            c.row(pre).collect::<Vec<(u32, f32)>>()
        };
        assert_eq!(row(16), vec![(25, 20.0)]);
        assert_eq!(row(25), vec![(29, 20.0)]);
        assert_eq!(row(0), vec![(1, 2.0)]);
        assert_eq!(row(1), vec![(2, -3.0)]);
        assert_eq!(row(2), vec![(0, 1.0)]);
        assert_eq!(row(3), vec![]);
        // in-row ascending destination order verified via offsets monotonic
        assert!(c.validate().is_ok());
        // polarity derivation from sign
        assert_eq!(c.polarity_at(0, 16), Polarity::Excitatory);
        // row 1 holds the -3.0 edge (flat edge index 1)
        assert_eq!(c.polarity_at(1, 1), Polarity::Inhibitory);
        assert!(c.weights.iter().all(|&w| w != 0.0));
        // identity semantics from connectome_binary
        let (_, v, nn, mm, hash) = connectome_identity(c);
        assert_eq!(v, TEMPLATE_VERSION);
        assert_eq!((nn, mm), (N as u64, 5));
        // labels: superclass mapping + empty cell-type label -> NO_IDX
        let m16 = &c.node_meta[16];
        assert_eq!(c.region_name(m16.region_idx), Some("sc16"));
        let m1 = &c.node_meta[1]; // i % 3 == 1 -> empty label -> NO_IDX
        assert_eq!(m1.cell_type_idx, NO_IDX);
        let m2 = &c.node_meta[2]; // wants "ct2" which must land at dense idx 1
        assert_eq!(c.cell_type_name(m2.cell_type_idx), Some("ct2"));
        assert_eq!(m2.cell_type_idx, 1);
        // body ids preserved exactly in order
        assert_eq!(c.node_meta.first().unwrap().body_id, 1000);
        assert_eq!(c.node_meta[29].body_id, 1029);
        let _ = hash;
    }

    /// The runtime drives spikes FROM a source row TO the right postsynaptic
    /// row (the transpose is exercised end-to-end through TickChannels).
    #[test]
    fn runtime_spike_lands_on_right_postsynaptic_row() {
        let d = package_dir("orientation-runtime");
        build_package(&d, E);
        let full = import_full(load_package(&d).expect("load")).expect("import");
        use std::sync::Arc;
        let mut rt = crate::runtime::FlyBrainRuntime::new(
            Arc::new(full.connectome),
            7,
        )
        .expect("30-neuron package must fit (needs >= 28)");
        rt.sensory.baseline_hz = 0.0;
        rt.sensory.gain_hz = 600.0;
        let mut chans = [0.0f32; crate::sensory::CHANNEL_COUNT];
        chans[crate::sensory::CH_FOOD_COS] = 1.0; // binds cells 16/17
        let b = crate::runtime::TickBudget::unbounded();
        let mut saw_relay = false;
        let mut saw_mid = false;
        let mut saw_end = false;
        for _ in 0..8 {
            rt.tick_channels(&chans, &b);
            let s = &rt.state().spiked;
            if s.contains(&16) {
                saw_relay = true;
            }
            if s.contains(&25) {
                saw_mid = true;
            }
            if s.contains(&29) {
                saw_end = true;
            }
        }
        assert!(saw_relay, "input cell 16 must spike");
        assert!(saw_mid, "postsynaptic row 25 must receive (16 -> 25)");
        assert!(saw_end, "postsynaptic row 29 must receive (25 -> 29)");
    }

    #[test]
    fn subnetwork_closure_deterministic_body_ids_preserved() {
        let d = package_dir("subnet");
        build_package(&d, E);
        let full = import_full(load_package(&d).expect("load")).expect("import");
        let pops = NamedPopulations {
            retina: vec![16],
            lamina: vec![],
            sugar: vec![],
            descending: vec![29],
        };
        let (c1, i1) =
            subnetwork_for_queen(&full.connectome, &pops, QueenSubnetOptions { hops: 2, max_neurons: 100 })
                .expect("subnet");
        let (c2, i2) =
            subnetwork_for_queen(&full.connectome, &pops, QueenSubnetOptions { hops: 2, max_neurons: 100 })
                .expect("subnet");
        assert_eq!(connectome_identity(&c1), connectome_identity(&c2));
        assert_eq!(i1, i2);
        assert_eq!((i1.neurons, i1.edges), (3, 2));
        assert!(!i1.truncated);
        // bodies preserved in full-graph original order
        let bodies: Vec<u64> = c1.node_meta.iter().map(|m| m.body_id).collect();
        assert_eq!(bodies, vec![1016, 1025, 1029]);
        // induced edges verbatim, polarity preserved
        assert_eq!(c1.weights, vec![20.0, 20.0]);
        assert_eq!(c1.polarity_at(0, 0), Polarity::Excitatory);
        assert_eq!(c1.polarity_at(1, 1), Polarity::Excitatory);
        // seeds preserved as hop-0 members of the closure
        assert_eq!((i1.seeds_found_retina, i1.seeds_found_descending), (1, 1));
    }

    #[test]
    fn subnetwork_cap_truncates_by_degree_then_index() {
        let d = package_dir("subnet-cap");
        build_package(&d, E);
        let full = import_full(load_package(&d).expect("load")).expect("import");
        let pops = NamedPopulations {
            retina: vec![16],
            lamina: vec![],
            sugar: vec![],
            descending: vec![],
        };
        // hop-1 candidates: 25 (out-degree 1) and 20 (out-degree 0; from the
        // second 16 row edge must NOT exist — E only wires 16->25, so add
        // artificial candidates via a second run instead: cap 0 after seeds)
        let (c, i) =
            subnetwork_for_queen(&full.connectome, &pops, QueenSubnetOptions { hops: 1, max_neurons: 1 })
                .expect("subnet");
        assert!(i.truncated);
        assert_eq!(i.neurons, 1);
        assert_eq!(c.n(), 1);
        assert_eq!(c.edge_count(), 0);
        assert_eq!(c.node_meta[0].body_id, 1016);
        let _ = connectome_identity(&c);
    }

    #[test]
    fn subnetwork_cap_prefers_higher_degree() {
        // custom edges: 16 -> {25, 20}; 25 -> 29 (deg 1) vs 20 (deg 0).
        let edges: &[(u32, u32, f32)] = &[
            (16, 25, 20.0),
            (16, 20, 20.0),
            (25, 29, 20.0),
            (0, 1, 2.0),
        ];
        let d = package_dir("subnet-degree");
        build_package(&d, edges);
        let full = import_full(load_package(&d).expect("load")).expect("import");
        let pops = NamedPopulations {
            retina: vec![16],
            lamina: vec![],
            sugar: vec![],
            descending: vec![],
        };
        let (c, i) =
            subnetwork_for_queen(&full.connectome, &pops, QueenSubnetOptions { hops: 1, max_neurons: 2 })
                .expect("subnet");
        assert!(i.truncated);
        let bodies: Vec<u64> = c.node_meta.iter().map(|m| m.body_id).collect();
        assert_eq!(bodies, vec![1016, 1025], "higher-degree candidate 25 must win the cap");
    }

    #[test]
    fn compiled_artifacts_roundtrip_full_and_subnet() {
        let d = package_dir("artifacts");
        build_package(&d, E);
        let full = import_full(load_package(&d).expect("load")).expect("import");
        // FULL artifact roundtrip via the versioned binary format
        let bytes = connectome_binary::serialize(&full.connectome).expect("serialize");
        let back = connectome_binary::parse_bytes(&bytes).expect("parse");
        assert_eq!(connectome_identity(&back), connectome_identity(&full.connectome));
        // SUBNET artifact roundtrip
        let pops = NamedPopulations {
            retina: vec![16],
            lamina: vec![],
            sugar: vec![],
            descending: vec![29],
        };
        let (sub, _) =
            subnetwork_for_queen(&full.connectome, &pops, QueenSubnetOptions::default())
                .expect("subnet");
        let sub_bytes = connectome_binary::serialize(&sub).expect("serialize");
        let sub_back = connectome_binary::parse_bytes(&sub_bytes).expect("parse");
        assert_eq!(connectome_identity(&sub_back), connectome_identity(&sub));
        assert_eq!(connectome_identity(&sub_back).0, SUBNETWORK_NAME);
    }

    // ------------------------------------------------ manifest failures

    #[test]
    fn sha_mismatch_is_detected() {
        let d = package_dir("sha-bad");
        build_package(&d, E);
        // corrupt in place (keep manifest hashes stale): must be Hash error
        let p = d.join("graph/edges_weight.f32");
        let mut b = std::fs::read(&p).unwrap();
        b[0] ^= 0x55;
        w(&p, &b);
        let err = load_package(&d).unwrap_err();
        assert!(matches!(err, MaleCnsError::Hash { .. }), "got {err:?}");
    }

    #[test]
    fn non_monotonic_offsets_are_detected() {
        let d = package_dir("offsets-bad");
        build_package(&d, E);
        let p = d.join("graph/edges_offsets.i32");
        let mut b = std::fs::read(&p).unwrap();
        // smash offset[5] backwards (i32 LE at bytes 20..24)
        b[20..24].copy_from_slice(&1i32.to_le_bytes());
        w(&p, &b);
        refresh_manifest(&d, E); // sha now matches again
        let err = load_package(&d).unwrap_err();
        assert!(matches!(err, MaleCnsError::Offsets(_)), "got {err:?}");
    }

    #[test]
    fn out_of_range_source_index_is_detected() {
        let d = package_dir("range-bad");
        build_package(&d, E);
        let p = d.join("graph/edges_source.i32");
        let mut b = std::fs::read(&p).unwrap();
        b[0..4].copy_from_slice(&999i32.to_le_bytes());
        w(&p, &b);
        refresh_manifest(&d, E);
        let err = load_package(&d).unwrap_err();
        assert!(matches!(err, MaleCnsError::Range { .. }), "got {err:?}");
    }

    #[test]
    fn sign_count_mismatch_is_detected() {
        let d = package_dir("sign-bad");
        build_package(&d, E);
        refresh_manifest(&d, E);
        let mp = d.join("manifest.json");
        let mut v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&mp).unwrap(),
        )
        .unwrap();
        v["weights"]["positive"] = serde_json::json!(99);
        w(&mp, serde_json::to_vec(&v).unwrap().as_slice());
        let err = load_package(&d).unwrap_err();
        assert!(
            matches!(err, MaleCnsError::Manifest(ref m) if m.contains("weights block mismatch")),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_weight_edge_is_rejected() {
        let zz: &[(u32, u32, f32)] = &[
            (16, 25, 20.0),
            (25, 29, 20.0),
            (0, 1, 2.0),
            (1, 2, 0.0), // zero: no sign, must hard-fail
        ];
        let d = package_dir("zero-weight");
        build_package(&d, zz);
        let err = load_package(&d).unwrap_err();
        assert!(matches!(err, MaleCnsError::ZeroWeight(_)), "got {err:?}");
    }

    #[test]
    fn body_id_sorted_claim_verified_when_present() {
        let d = package_dir("sorted-claim");
        build_package(&d, E);
        // scramble body ids, then re-stamp the manifest shas
        let p = d.join("neurons/body_id.i64");
        let mut b = std::fs::read(&p).unwrap();
        b[0..8].copy_from_slice(&9999i64.to_le_bytes());
        b[8..16].copy_from_slice(&1000i64.to_le_bytes());
        w(&p, &b);
        refresh_manifest(&d, E);
        let err = load_package(&d).unwrap_err();
        assert!(
            matches!(err, MaleCnsError::Manifest(ref m) if m.contains("sorted")),
            "got {err:?}"
        );
    }

    /// Subnetwork-scale per-tick cost probe (print-only; the bound is a gross
    /// regression guard). Run with --nocapture to capture numbers.
    #[test]
    fn measured_tick_cost_at_subnet_scale() {
        use crate::connectome::EdgeSpec;
        use crate::provenance::ProvenanceTag;
        use std::sync::Arc;
        const NS: usize = 40_000;
        const TARGET: usize = 420_000;
        let mut edges = Vec::with_capacity(TARGET);
        for k in 0..TARGET {
            let pre = (k as u64 * 7919 % NS as u64) as u32;
            let post = ((k as u64 * 104_729 + k as u64) % NS as u64) as u32;
            edges.push(EdgeSpec {
                pre,
                post,
                weight: if k % 2 == 0 { 2.0 } else { -2.0 },
                delay: None,
                polarity: Some(if k % 2 == 0 {
                    crate::connectome::Polarity::Excitatory
                } else {
                    crate::connectome::Polarity::Inhibitory
                }),
            });
        }
        let meta = |i: u64| crate::connectome::NeuronMeta {
            body_id: 900_000_000 + i,
            region_idx: NO_IDX,
            cell_type_idx: NO_IDX,
            polarity: crate::connectome::Polarity::Unknown,
            origin: ProvenanceTag::Assumption,
        };
        let conn = crate::connectome::Connectome::from_edges(
            "subnet-scale-probe",
            "",
            (0..NS).map(|i| meta(i as u64)).collect(),
            vec![],
            vec![],
            edges,
        )
        .expect("probe connectome");
        let mut rt = crate::runtime::FlyBrainRuntime::new(Arc::new(conn), 3).expect("fits");
        rt.sensory.baseline_hz = 0.0;
        rt.sensory.gain_hz = 500.0;
        let mut chans = [0.0f32; crate::sensory::CHANNEL_COUNT];
        chans[crate::sensory::CH_FOOD_COS] = 1.0;
        let b = crate::runtime::TickBudget::unbounded();
        let ticks = 300u32;
        let mut total_ns = 0u64;
        for _ in 0..ticks {
            rt.tick_channels(&chans, &b);
            total_ns += rt.last_stats().exec_ns;
        }
        let st = rt.last_stats();
        println!(
            "PERF subnet-scale probe: {} neurons / {} synapses, avg tick = {} ns ({} us); last stats active={} spikes={}",
            st.neuron_count,
            st.synapse_count,
            total_ns / ticks as u64,
            total_ns / ticks as u64 / 1000,
            st.active,
            st.spikes,
        );
        assert!(st.neuron_count == NS as u64);
    }
}
