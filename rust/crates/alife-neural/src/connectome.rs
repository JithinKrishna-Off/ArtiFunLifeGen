//! CSR sparse synapse storage, compact neuron arrays and side tables.
//!
//! Storage model (NO dense matrices anywhere in this crate):
//! - `row_offsets: Vec<u32>` (length `n + 1`), `col_idx: Vec<u32>`, `weights: Vec<f32>`
//!   form a standard CSR row-major fan-out adjacency of the directed synapse graph.
//! - Optional per-synapse `delays: Vec<u8>` (propagation delay in steps, >= 1) and
//!   optional per-synapse `syn_polarity` (`Polarity as u8`). When `syn_polarity` is
//!   `None`, polarity falls back to the pre-synaptic neuron's polarity, and when
//!   that is `Unknown` the runtime's `UnknownPolarityPolicy` applies.
//! - Neurons are stored compactly: `node_meta: Vec<NeuronMeta>` with region and
//!   cell-type side tables (`regions`, `cell_types` string vectors; `u32::MAX`
//!   index = not assigned / unavailable, never a fabricated name).
//!
//! Provenance: every neuron carries a [`ProvenanceTag`] (see `provenance.rs`).
//! Data NOT from the ground-truth source must carry `Assumption` (structure we
//! invented, e.g. the synthetic fixture) or `Unavailable`.

use serde::{Deserialize, Serialize};

use crate::provenance::ProvenanceTag;

/// Synaptic/functional polarity of a neuron or synapse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Polarity {
    Excitatory = 0,
    Inhibitory = 1,
    /// Source does not state it (e.g. neurotransmitter table not imported).
    /// Runtime behavior is governed by `UnknownPolarityPolicy`.
    Unknown = 2,
}

impl Polarity {
    pub fn from_u8(v: u8) -> Polarity {
        match v {
            0 => Polarity::Excitatory,
            1 => Polarity::Inhibitory,
            _ => Polarity::Unknown,
        }
    }
}

pub const NO_IDX: u32 = u32::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronMeta {
    /// Original source body ID, preserved exactly for source data. For synthetic
    /// fixtures we allocate IDs below SYNTHETIC_BASE and label provenance.
    pub body_id: u64,
    /// Index into `regions`, `NO_IDX` when unavailable.
    pub region_idx: u32,
    /// Index into `cell_types`, `NO_IDX` when unavailable.
    pub cell_type_idx: u32,
    pub polarity: Polarity,
    pub origin: ProvenanceTag,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphStats {
    pub neuron_count: u64,
    pub synapse_count: u64,
    pub neurons_without_region: u64,
    pub neurons_polarity_unknown: u64,
    pub synapses_polarity_unknown: u64,
    pub max_out_degree: u64,
}

/// Errors surfaced by validation / construction.
#[derive(Debug, thiserror::Error)]
pub enum ConnectomeError {
    #[error("row_offsets length {0} does not match node count {1}")]
    RowOffsetsLen(usize, usize),
    #[error("row_offsets not non-decreasing at index {0}")]
    RowOffsetsOrder(usize),
    #[error("synapse column index {0} out of range (neurons={1})")]
    ColOutOfRange(u64, u64),
    #[error("synapse weight not finite at flat index {0}")]
    WeightNotFinite(u64),
    #[error("synapse delay {0} at flat index {1} is zero (delays must be >= 1)")]
    DelayZero(u8, u64),
    #[error("node body_id duplicate: {0}")]
    DuplicateBodyId(u64),
    #[error("synapse delay array length mismatch (rows={0}, got {1})")]
    DelayLen(usize, usize),
    #[error("synapse polarity array length mismatch (rows={0}, got {1})")]
    SynPolarityLen(usize, usize),
    #[error("edge endpoint {0} out of range (neurons={1})")]
    EdgeIndexOutOfRange(u32, u64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connectome {
    pub name: String,
    /// Dataset template/version this connectome was compiled from (empty for
    /// synthetic fixtures).
    pub template_version: String,
    pub node_meta: Vec<NeuronMeta>,
    pub regions: Vec<String>,
    pub cell_types: Vec<String>,
    pub row_offsets: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub weights: Vec<f32>,
    pub delays: Option<Vec<u8>>,
    pub syn_polarity: Option<Vec<u8>>,
}

/// One synapse in builder form (pre-CSR).
#[derive(Clone, Copy, Debug)]
pub struct EdgeSpec {
    pub pre: u32,
    pub post: u32,
    pub weight: f32,
    pub delay: Option<u8>,
    pub polarity: Option<Polarity>,
}

impl Connectome {
    /// Construct from final CSR arrays. Validates constraints; never mutates
    /// meaning of stored values.
    #[allow(clippy::too_many_arguments)]
    pub fn from_csr(
        name: impl Into<String>,
        template_version: impl Into<String>,
        node_meta: Vec<NeuronMeta>,
        regions: Vec<String>,
        cell_types: Vec<String>,
        row_offsets: Vec<u32>,
        col_idx: Vec<u32>,
        weights: Vec<f32>,
        delays: Option<Vec<u8>>,
        syn_polarity: Option<Vec<u8>>,
    ) -> Result<Self, ConnectomeError> {
        let conn = Self {
            name: name.into(),
            template_version: template_version.into(),
            node_meta,
            regions,
            cell_types,
            row_offsets,
            col_idx,
            weights,
            delays,
            syn_polarity,
        };
        conn.validate()?;
        Ok(conn)
    }

    /// Construct from an unordered edge list. Sorting is stable: edges are
    /// laid out by ascending `pre`, then `post`, then original order, so the
    /// CSR layout is a deterministic function of the input list.
    pub fn from_edges(
        name: impl Into<String>,
        template_version: impl Into<String>,
        node_meta: Vec<NeuronMeta>,
        regions: Vec<String>,
        cell_types: Vec<String>,
        mut edges: Vec<EdgeSpec>,
    ) -> Result<Self, ConnectomeError> {
        let n = node_meta.len();
        for e in &edges {
            if e.pre as u64 >= n as u64 || e.post as u64 >= n as u64 {
                return Err(ConnectomeError::EdgeIndexOutOfRange(e.pre, n as u64));
            }
        }
        edges.sort_by(|a, b| {
            a.pre.cmp(&b.pre)
                .then(a.post.cmp(&b.post))
        });
        let mut row_offsets = Vec::with_capacity(n + 1);
        row_offsets.push(0u32);
        let mut col_idx = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());
        let mut delays = Vec::with_capacity(edges.len());
        let mut syn_pol = if edges.iter().any(|e| e.polarity.is_some()) {
            Some(Vec::<u8>::with_capacity(edges.len()))
        } else {
            None
        };
        let mut cur_pre = 0u32;
        for e in &edges {
            while cur_pre < e.pre {
                row_offsets.push(col_idx.len() as u32);
                cur_pre += 1;
            }
            col_idx.push(e.post);
            weights.push(e.weight);
            delays.push(e.delay.unwrap_or(0)); // 0 = "use runtime default"
            if let Some(sp) = syn_pol.as_mut() {
                sp.push(e.polarity.map_or(2, |p| p as u8));
            }
        }
        while (row_offsets.len() as usize) < n + 1 {
            row_offsets.push(col_idx.len() as u32);
        }
        let delays = Some(delays).filter(|_| edges.iter().any(|e| e.delay.is_some()));
        Self::from_csr(
            name,
            template_version,
            node_meta,
            regions,
            cell_types,
            row_offsets,
            col_idx,
            weights,
            delays,
            syn_pol,
        )
    }

    pub fn n(&self) -> usize {
        self.node_meta.len()
    }

    pub fn edge_count(&self) -> usize {
        self.weights.len()
    }

    pub fn validate(&self) -> Result<(), ConnectomeError> {
        let n = self.node_meta.len();
        if self.row_offsets.len() != n + 1 {
            return Err(ConnectomeError::RowOffsetsLen(self.row_offsets.len(), n));
        }
        let mut prev = 0u32;
        for (i, &v) in self.row_offsets.iter().enumerate() {
            if v < prev {
                return Err(ConnectomeError::RowOffsetsOrder(i));
            }
            prev = v;
        }
        for (i, &c) in self.col_idx.iter().enumerate() {
            if c as usize >= n {
                return Err(ConnectomeError::ColOutOfRange(c as u64, n as u64));
            }
            if !self.weights[i].is_finite() {
                return Err(ConnectomeError::WeightNotFinite(i as u64));
            }
        }
        if let Some(d) = &self.delays {
            if d.len() != self.weights.len() {
                return Err(ConnectomeError::DelayLen(self.weights.len(), d.len()));
            }
            // 0 encodes "use runtime default delay"; any other stored value is
            // used verbatim as an explicit delay (>= 1 by construction).
        }
        if let Some(sp) = &self.syn_polarity {
            if sp.len() != self.weights.len() {
                return Err(ConnectomeError::SynPolarityLen(
                    self.weights.len(),
                    sp.len(),
                ));
            }
        }
        Ok(())
    }

    /// Lightweight CSR row iteration helper.
    pub fn row(&self, pre: u32) -> impl Iterator<Item = (u32, f32)> + '_ {
        let start = self.row_offsets[pre as usize] as usize;
        let end = self.row_offsets[pre as usize + 1] as usize;
        (start..end).map(move |i| (self.col_idx[i], self.weights[i]))    }

    /// Effective synaptic delay for synapse i (explicit delay or runtime default).
    pub fn delay_at(&self, i: usize, default_delay: u8) -> u8 {
        match &self.delays {
            Some(d) => {
                let dv = d[i];
                if dv == 0 {
                    default_delay.max(1)
                } else {
                    dv
                }
            }
            None => default_delay.max(1),
        }
    }

    /// Effective polarity for synapse i (synapse flag > pre-cell flag, or Unknown).
    pub fn polarity_at(&self, i: usize, pre: u32) -> Polarity {
        if let Some(sp) = &self.syn_polarity {
            let p = Polarity::from_u8(sp[i]);
            if p != Polarity::Unknown {
                return p;
            }
        }
        self.node_meta[pre as usize].polarity
    }

    pub fn stats(&self) -> GraphStats {
        let mut st = GraphStats::default();
        st.neuron_count = self.n() as u64;
        st.synapse_count = self.weights.len() as u64;
        let mut max_deg = 0u64;
        for i in 0..self.n() {
            let deg = (self.row_offsets[i + 1] - self.row_offsets[i]) as u64;
            max_deg = max_deg.max(deg);
            if self.node_meta[i].region_idx == NO_IDX {
                st.neurons_without_region += 1;
            }
            if self.node_meta[i].polarity == Polarity::Unknown {
                st.neurons_polarity_unknown += 1;
            }
        }
        if let Some(sp) = &self.syn_polarity {
            st.synapses_polarity_unknown =
                sp.iter().filter(|&&p| p == Polarity::Unknown as u8) .count() as u64;
        }
        st.max_out_degree = max_deg;
        st
    }

    /// Memory footprint estimate in bytes (all stored arrays + metadata).
    pub fn memory_bytes(&self) -> u64 {
        let sz_strings: u64 = self
            .regions
            .iter()
            .chain(self.cell_types.iter())
            .map(|s| (s.len() + std::mem::size_of::<usize>()) as u64)
            .sum();
        let meta = (self.node_meta.len()
            * std::mem::size_of::<NeuronMeta>()) as u64;
        let name = (self.name.len() + self.template_version.len()) as u64;
        (self.row_offsets.len() * 4
            + self.col_idx.len() * 4
            + self.weights.len() * 4) as u64
            + self.delays.as_ref().map_or(0, |d| d.len() as u64)
            + self.syn_polarity.as_ref().map_or(0, |s| s.len() as u64)
            + meta
            + sz_strings
            + name
            + (self.n() * 4) as u64
    }

    /// Structural identity: (name, template version, n, edges, content hash).
    pub fn identity(&self) -> ConnectomeIdentity {
        let mut h = fnv64_offset();
        hash_bytes(&mut h, self.name.as_bytes());
        hash_bytes(&mut h, self.template_version.as_bytes());
        hash_u64(&mut h, self.n() as u64);
        hash_u64(&mut h, self.weights.len() as u64);
        for &v in &self.row_offsets {
            hash_u64(&mut h, v as u64);
        }
        for &v in &self.col_idx {
            hash_u64(&mut h, v as u64);
        }
        for &v in &self.weights {
            hash_u64(&mut h, v.to_bits() as u64);
        }
        if let Some(d) = &self.delays {
            hash_bytes(&mut h, d);
        }
        if let Some(sp) = &self.syn_polarity {
            hash_bytes(&mut h, sp);
        }
        for m in &self.node_meta {
            hash_u64(&mut h, m.body_id);
        }
        ConnectomeIdentity {
            name: self.name.clone(),
            template_version: self.template_version.clone(),
            neuron_count: self.n() as u64,
            edge_count: self.weights.len() as u64,
            content_hash: h,
        }
    }

    pub fn region_name(&self, idx: u32) -> Option<&str> {
        self.regions.get(idx as usize).map(|s| s.as_str())
    }

    pub fn cell_type_name(&self, idx: u32) -> Option<&str> {
        self.cell_types.get(idx as usize).map(|s| s.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectomeIdentity {
    pub name: String,
    pub template_version: String,
    pub neuron_count: u64,
    pub edge_count: u64,
    /// 64-bit FNV-1a over the full structural arrays (identity/checksum;
    /// NOT a cryptographic guarantee).
    pub content_hash: u64,
}

// ------------------------------------------------------------- small hashing
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

pub fn fnv64_offset() -> u64 {
    FNV_OFFSET
}

pub fn hash_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

pub fn hash_u64(h: &mut u64, v: u64) {
    hash_bytes(h, &v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: u64) -> NeuronMeta {
        NeuronMeta {
            body_id: id,
            region_idx: NO_IDX,
            cell_type_idx: NO_IDX,
            polarity: Polarity::Unknown,
            origin: ProvenanceTag::SuppliedBySource,
        }
    }

    #[test]
    fn validate_catches_bad_edge() {
        let ok = Connectome::from_edges(
            "t",
            "",
            vec![meta(1), meta(2)],
            vec![],
            vec![],
            vec![EdgeSpec { pre: 0, post: 1, weight: 1.0, delay: None, polarity: None }],
        )
        .unwrap();
        assert!(ok.validate().is_ok());

        let bad = Connectome {
            node_meta: ok.node_meta.clone(),
            regions: vec![],
            cell_types: vec![],
            name: ok.name.clone(),
            template_version: String::new(),
            row_offsets: ok.row_offsets.clone(),
            col_idx: vec![7],
            weights: vec![1.0],
            delays: ok.delays.clone(),
            syn_polarity: ok.syn_polarity.clone(),
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn stats_and_memory_scale_linearly() {
        let build = |n: usize| {
            let meta: Vec<NeuronMeta> = (0..n).map(|i| meta(i as u64)).collect();
            let edges: Vec<EdgeSpec> = (0..n)
                .map(|i| EdgeSpec {
                    pre: i as u32,
                    post: ((i + 1) % n) as u32,
                    weight: 1.0,
                    delay: None,
                    polarity: None,
                })
                .collect();
            Connectome::from_edges("s", "", meta, vec![], vec![], edges).unwrap()
        };
        let a = build(100);
        let b = build(200);
        let ma = a.memory_bytes();
        let mb = b.memory_bytes();
        assert_eq!(a.stats().neuron_count, 100);
        assert_eq!(a.stats().synapse_count, 100);
        let ratio = mb as f64 / ma as f64;
        assert!(
            (1.7..2.3).contains(&ratio),
            "memory estimate must scale ~linearly, ratio={ratio}"
        );
    }
}
