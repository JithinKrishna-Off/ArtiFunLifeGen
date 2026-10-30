//! Region (superclass/type) and cell-type filters; subnetwork extraction that
//! preserves ORIGINAL body IDs and per-neuron provenance.
//!
//! Filters are name-based against the connectome's side tables only. A name
//! that is not in the tables selects nothing (missing annotations are never
//! invented or expanded into guesses).
//!
//! `default_mapping` is an explicitly DOCUMENTED HEURISTIC for imported
//! connectomes whose sensory/motor regions we cannot identify biologically
//! yet: inputs occupy the first neurons, outputs the last (same default
//! binding as `sensory.rs`; tagged `ProvenanceTag::Assumption` where the caller
//! records it). It returns `None` when the connectome is too small for
//! inputs + outputs. No biological claim is attached to any binding.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::connectome::{Connectome, EdgeSpec, NeuronMeta, NO_IDX};
use crate::sensory::{inputs_needed, SensoryConfig, CHANNEL_COUNT};
use crate::motor::OUTPUT_CELL_COUNT;

/// Selection criterion against node side tables.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SubnetworkFilter {
    /// Region names (side-table `regions` entries) to keep, case-sensitive.
    pub regions: HashSet<String>,
    /// Cell-type names (side-table `cell_types` entries) to keep,
    /// case-sensitive.
    pub cell_types: HashSet<String>,
}

impl SubnetworkFilter {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty() && self.cell_types.is_empty()
    }

    /// Does node i satisfy the filter (either criterion matches)?
    /// An empty filter selects every node.
    pub fn selects(&self, conn: &Connectome, i: usize) -> bool {
        if self.is_empty() {
            return true;
        }
        let m = &conn.node_meta[i];
        if let Some(r) = conn.region_name(m.region_idx) {
            if self.regions.contains(r) {
                return true;
            }
        }
        if let Some(t) = conn.cell_type_name(m.cell_type_idx) {
            if self.cell_types.contains(t) {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractResult {
    pub connectome: Connectome,
    /// old compact index -> new compact index (for callers tracking mapping).
    pub index_map: Vec<u32>,
    /// Bodies included (original source IDs, exact).
    pub body_ids: Vec<u64>,
}

/// Extract the induced subgraph over nodes matching `filter`. Original body
/// IDs and per-node `origin` provenance tags are preserved VERBATIM (this is
/// a structural slice, not a re-derivation). Edges are deduplicated to the
/// first occurrence per (pre, post) pair in CSR scan order to keep the extract
/// deterministic.
pub fn extract(conn: &Connectome, filter: &SubnetworkFilter, name: &str) -> ExtractResult {
    let n = conn.n();
    let mut keep = vec![false; n];
    for i in 0..n {
        keep[i] = filter.selects(conn, i);
    }
    let mut remap: Vec<u32> = vec![NO_IDX; n];
    let mut nodes: Vec<NeuronMeta> = Vec::new();
    let mut body_ids: Vec<u64> = Vec::new();
    for i in 0..n {
        if keep[i] {
            remap[i] = nodes.len() as u32;
            nodes.push(NeuronMeta {
                body_id: conn.node_meta[i].body_id,
                region_idx: conn.node_meta[i].region_idx,
                cell_type_idx: conn.node_meta[i].cell_type_idx,
                polarity: conn.node_meta[i].polarity,
                origin: conn.node_meta[i].origin,
            });
            body_ids.push(conn.node_meta[i].body_id);
        }
    }
    // Region/cell-type tables are carried over unchanged (indices remain valid
    // because node metadata indices are copied verbatim).
    let mut edges: Vec<EdgeSpec> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for pre in 0..n {
        if !keep[pre] {
            continue;
        }
        for (post, w) in conn.row(pre as u32) {
            if !keep[post as usize] {
                continue;
            }
            let key = (remap[pre], remap[post as usize]);
            if !seen.insert(key) {
                continue;
            }
            let rs = conn.row_offsets[pre] as usize;
            let syn_i = rs + position_in_row(conn, pre as u32, post);
            edges.push(EdgeSpec {
                pre: remap[pre],
                post: remap[post as usize],
                weight: w,
                delay: conn
                    .delays
                    .as_ref()
                    .map(|d| d[syn_i])
                    .filter(|&d| d != 0),
                polarity: conn
                    .syn_polarity
                    .as_ref()
                    .map(|sp| crate::connectome::Polarity::from_u8(sp[syn_i]))
                    .filter(|p| *p != crate::connectome::Polarity::Unknown),
            });
        }
    }
    let connectome = Connectome::from_edges(
        name.to_string(),
        conn.template_version.clone(),
        nodes,
        conn.regions.clone(),
        conn.cell_types.clone(),
        edges,
    )
    .expect("extraction of a validated connectome must build");
    ExtractResult {
        connectome,
        index_map: remap,
        body_ids,
    }
}

/// Position of `post` within pre-row `pre`'s CSR slice (first match).
fn position_in_row(conn: &Connectome, pre: u32, post: u32) -> usize {
    let rs = conn.row_offsets[pre as usize] as usize;
    let re = conn.row_offsets[pre as usize + 1] as usize;
    let mut i = rs;
    while i < re && conn.col_idx[i] != post {
        i += 1;
    }
    i - rs
}

/// Size-checked default mapping: usable when the connectome holds the default
/// sensory block plus the output cells.
pub fn default_mapping(conn: &Connectome, outputs: usize) -> Option<SensoryConfig> {
    let cfg = SensoryConfig::default();
    if conn.n() >= inputs_needed(&cfg) + outputs && cfg.channel_cells >= 1 {
        Some(cfg)
    } else {
        None
    }
}

/// Number of channels the runtime expects (re-exported constant for callers).
pub const CHANNELS: usize = CHANNEL_COUNT;

pub const SUGGESTED_OUTPUT_CELLS: usize = OUTPUT_CELL_COUNT;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{EdgeSpec, NeuronMeta, Polarity, NO_IDX};
    use crate::provenance::ProvenanceTag;

    fn meta(id: u64, region: Option<u32>, ct: Option<u32>) -> NeuronMeta {
        NeuronMeta {
            body_id: id,
            region_idx: region.unwrap_or(NO_IDX),
            cell_type_idx: ct.unwrap_or(NO_IDX),
            polarity: Polarity::Unknown,
            origin: ProvenanceTag::SuppliedBySource,
        }
    }

    fn small_conn() -> Connectome {
        let nodes = vec![
            meta(101, Some(0), Some(0)),
            meta(202, Some(1), None),
            meta(303, Some(1), Some(0)),
            meta(404, None, None),
        ];
        let edges = vec![
            EdgeSpec { pre: 0, post: 1, weight: 1.0, delay: None, polarity: Some(Polarity::Excitatory) },
            EdgeSpec { pre: 1, post: 2, weight: 2.0, delay: Some(1), polarity: Some(Polarity::Inhibitory) },
            EdgeSpec { pre: 2, post: 3, weight: 3.0, delay: None, polarity: None },
        ];
        Connectome::from_edges(
            "t",
            "",
            nodes,
            vec!["optic".to_string(), "central".to_string()],
            vec!["motor".to_string()],
            edges,
        )
        .unwrap()
    }

    #[test]
    fn extract_keeps_source_ids_and_provenance() {
        let conn = small_conn();
        let mut regions = HashSet::new();
        regions.insert("central".to_string());
        let f = SubnetworkFilter {
            regions,
            cell_types: HashSet::new(),
        };
        let r = extract(&conn, &f, "central-slice");
        assert_eq!(r.body_ids, vec![202, 303]); // ORIGINAL body ids
        let st = r.connectome.stats();
        assert_eq!(st.neuron_count, 2);
        assert_eq!(st.synapse_count, 1); // edge 1->2 survives; 2->3 exits slice
        assert_eq!(r.connectome.node_meta[0].origin, ProvenanceTag::SuppliedBySource);
        assert_eq!(r.connectome.node_meta[1].region_idx, 1);
    }

    #[test]
    fn empty_filter_selects_all() {
        let conn = small_conn();
        let r = extract(&conn, &SubnetworkFilter::default(), "all");
        assert_eq!(r.body_ids, vec![101, 202, 303, 404]);
        assert_eq!(r.connectome.stats().synapse_count, 3);
    }

    #[test]
    fn cell_type_filter_and_default_mapping() {
        let conn = small_conn();
        let mut cts = HashSet::new();
        cts.insert("motor".to_string());
        let f = SubnetworkFilter { regions: HashSet::new(), cell_types: cts };
        let r = extract(&conn, &f, "motor-cells");
        assert_eq!(r.body_ids, vec![101, 303]);
        // Neither the 2-neuron slice nor the 4-neuron whole connectome can
        // host the default 22-input + outputs mapping.
        assert!(default_mapping(&r.connectome, 1).is_none());
        assert!(default_mapping(&conn, 1).is_none());
        // ... while a runtime-sized connectome can.
        assert!(default_mapping(&crate::fixtures::synthetic_fixture().connectome, 6).is_some());
    }
}
