//! Provenance records and field provenance tags for imported connectomes.
//!
//! RULES:
//! - Facts about a dataset are only written here after being verified against
//!   official pages. Unverified fields stay "unverified" / blank.
//! - We never invent biology. Every structural field in a connectome carries a
//!   [`ProvenanceTag`] stating where it came from.

use serde::{Deserialize, Serialize};

/// Field-level provenance tag (attached to connectome fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProvenanceTag {
    /// Value came directly from the source dataset.
    SuppliedBySource = 0,
    /// Value computed by our code from source values (documented derivation).
    Derived = 1,
    /// Heuristic/plausible value used in place of missing data (documented).
    Estimated = 2,
    /// Structural assumption of our wiring scheme, not present in the source.
    Assumption = 3,
    /// No data available; field is a placeholder and must not be read as fact.
    Unavailable = 4,
}

/// Dataset identity constants. VERIFIED facts only (see
/// `FLYBRAIN_IMPLEMENTATION.md`, "Dataset facts" section, for the checked URLs
/// and dates).
///
/// - v1.0 released 2026-06-08 per official release notes page (verified
///   2026-10-30): https://male-cns.janelia.org/release/
/// - The official download page (https://male-cns.janelia.org/download)
///   states "The Male CNS is licensed under CC-BY" and links
///   https://creativecommons.org/licenses/by/4.0/ , i.e. CC-BY 4.0
///   (verified 2026-10-30).
/// - Connectivity source file (download page, verified 2026-10-30):
///   `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, 1.1 GB.
/// - Annotations source file (download page, verified 2026-10-30):
///   `body-annotations-male-cns-v1.0-minconf-0.5.feather`, 13 MB.
pub const DATASET_NAME: &str = "MaleCNS v1.0";

/// Official download page (verified 2026-10-30).
pub const SOURCE_URL: &str = "https://male-cns.janelia.org/download";

/// Fallback text required by spec if a future record's URL cannot be verified.
pub const SOURCE_URL_UNVERIFIED: &str = "verify at flyem.janelia.org";

/// License as stated on the official download page, linking the CC-BY 4.0
/// license deed. Status: verified (2026-10-30).
pub const LICENSE: &str = "CC-BY 4.0";
pub const LICENSE_STATUS_VERIFIED: &str = "verified";
pub const LICENSE_STATUS_UNVERIFIED: &str = "unverified";

/// Canvas-side normalized TSV file names.
pub const EDGES_FILE: &str = "edges.tsv";
pub const NODES_FILE: &str = "nodes.tsv";
pub const SIDECAR_FILE: &str = "sidecar.json";

/// Full provenance record for one imported connectome.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceRecord {
    /// Dataset identity, e.g. `"MaleCNS v1.0"`.
    pub dataset: String,
    /// Official page documenting the files. Filled only from verified pages;
    /// otherwise `"verify at flyem.janelia.org"` (never a fabricated URL).
    pub source_url: String,
    /// Dataset license. Only `"CC-BY 4.0"` after verifying it is stated for the
    /// DATASET (a paper's license does not establish the dataset license).
    /// Otherwise `"unverified"`.
    pub license: String,
    /// `"verified"` or `"unverified"` labeling for `license`.
    pub license_status: String,
    /// Original upstream filename (as downloaded), e.g. the 1.1 GB feather.
    pub original_filename: String,
    /// SHA-256 of the original download as supplied by the human performing
    /// the manual download. Blank until provided. We never fabricate it.
    pub source_sha256: String,
    /// Version of `tools/convert_malecns.py` that produced the normalized TSVs.
    pub converter_version: String,
    /// Imported neuron and edge counts (post-import).
    pub imported_neurons: u64,
    pub imported_edges: u64,
    /// Import timestamp (RFC 3339, UTC) and conversion timestamp, if supplied
    /// by the sidecar. Blank when unavailable.
    pub imported_at_utc: String,
    pub converted_at_utc: String,
}

impl ProvenanceRecord {
    /// Record for the verified MaleCNS v1.0 dataset, with download-dependent
    /// fields left blank per policy.
    pub fn malecns_default() -> Self {
        Self {
            dataset: DATASET_NAME.to_string(),
            source_url: SOURCE_URL.to_string(),
            license: LICENSE.to_string(),
            license_status: LICENSE_STATUS_VERIFIED.to_string(),
            original_filename: String::new(),
            source_sha256: String::new(),
            converter_version: String::new(),
            imported_neurons: 0,
            imported_edges: 0,
            imported_at_utc: String::new(),
            converted_at_utc: String::new(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("provenance json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_is_lossless() {
        let mut rec = ProvenanceRecord::malecns_default();
        rec.original_filename = "connectome-weights-male-cns-v1.0-minconf-0.5.feather".into();
        rec.source_sha256 = "abcd".into();
        rec.converter_version = "1.0.0".into();
        rec.imported_neurons = 166_700;
        rec.imported_edges = 4_000_000;
        rec.imported_at_utc = "2026-10-30T12:00:00Z".into();
        let json = rec.to_json();
        let back: ProvenanceRecord = serde_json::from_str(&json).expect("roundtrip");
        assert_eq!(back, rec);
        assert_eq!(back.imported_edges, 4_000_000);
    }

    #[test]
    fn default_record_keeps_download_facts_blank() {
        let rec = ProvenanceRecord::malecns_default();
        // verified constants carry through ...
        assert_eq!(rec.dataset, DATASET_NAME);
        assert_eq!(rec.license_status, LICENSE_STATUS_VERIFIED);
        // ... while download-dependent facts stay blank (never fabricated)
        assert_eq!(rec.source_sha256, "");
        assert_eq!(rec.original_filename, "");
        assert_eq!(rec.imported_neurons, 0);
        assert_eq!(rec.imported_edges, 0);
    }

    #[test]
    fn field_tags_roundtrip_by_repr() {
        for (tag, byte) in [
            (ProvenanceTag::SuppliedBySource, 0u8),
            (ProvenanceTag::Derived, 1),
            (ProvenanceTag::Estimated, 2),
            (ProvenanceTag::Assumption, 3),
            (ProvenanceTag::Unavailable, 4),
        ] {
            let json = serde_json::to_string(&tag).expect("tag");
            let back: ProvenanceTag = serde_json::from_str(&json).expect("tag");
            assert_eq!(back, tag);
            assert_eq!(tag as u8, byte);
        }
    }
}
