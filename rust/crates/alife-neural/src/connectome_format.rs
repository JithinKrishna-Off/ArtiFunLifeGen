//! Normalized TSV schema types, sidecar schema, and line parsers.
//!
//! SOURCE-INDEPENDENT FORMAT (v1):
//! The converter (`tools/convert_malecns.py`) produces three files per dataset
//! run:
//! - `edges.tsv`  : `body_pre \t body_post \t weight` (one row per synapse;
//!   weights preserved verbatim from the source, meaning never altered)
//! - `nodes.tsv`  : `bodyId \t type \t superclass \t status \t somaSide`
//!   (one row per body; vertical alignment with annotation availability)
//! - `sidecar.json`: metadata sidecar (see [Sidecar])
//!
//! Rules enforced here:
//! - Body IDs are preserved EXACTLY as decimal integers (no rounding, no
//!   re-mapping at file level).
//! - A missing annotation field is EMPTY STRING. Nothing is ever invented.
//! - Column presence is validated before parsing rows; a header mismatch is a
//!   hard error.
//! - The format is source-independent so a FAFB v783 converter can later emit
//!   the same three files WITHOUT touching the LIF runtime, and datasets are
//!   never merged by this crate (one directory = one dataset/import).

use std::io::BufRead;

use serde::{Deserialize, Serialize};

use crate::provenance::{EDGES_FILE, NODES_FILE, SIDECAR_FILE};

pub const FORMAT_NAME: &str = "flybrain-normalized-connectome";
pub const FORMAT_VERSION: u32 = 1;

pub const EDGE_COLUMNS: [&str; 3] = ["body_pre", "body_post", "weight"];
pub const NODE_COLUMNS: [&str; 5] = ["bodyId", "type", "superclass", "status", "somaSide"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeRecord {
    pub body_pre: u64,
    pub body_post: u64,
    /// Stored verbatim; this crate never rescales or reweights.
    pub weight: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeRecord {
    pub body_id: u64,
    /// cell type ("" = not stated by source)
    pub cell_type: String,
    /// region/superclass ("" = not stated by source)
    pub region: String,
    /// reconstruction status ("" = not stated)
    pub status: String,
    /// soma side ("" = not stated)
    pub soma_side: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Sidecar {
    pub format: String,
    pub format_version: u32,
    pub dataset: String,
    /// Original upstream filenames (weights + annotations feather).
    pub source_files: Vec<String>,
    pub converter: String,
    pub converter_version: String,
    /// SHA-256 of the ORIGINAL downloads ("" until the human who performed
    /// the manual download supplies them; converter may fill from disk).
    pub source_sha256: String,
    pub rows_in: u64,
    pub rows_out_edges: u64,
    pub rows_out_nodes: u64,
    pub converted_at_utc: String,
    pub notes: String,
}

impl Sidecar {
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.format != FORMAT_NAME {
            return Err(FormatError::BadSidecar(format!(
                "sidecar format is '{}', expected '{FORMAT_NAME}'",
                self.format
            )));
        }
        if self.format_version > FORMAT_VERSION {
            return Err(FormatError::BadSidecar(format!(
                "sidecar format_version {} is newer than supported {FORMAT_VERSION}",
                self.format_version
            )));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("TSV parse error at line {line}: {msg}")]
    Tsv { line: u64, msg: String },
    #[error("bad sidecar: {0}")]
    BadSidecar(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Validate a TSV header line contains exactly the expected columns
/// (whitespace is ignored, byte order and BOM handled).
pub fn validate_header(header: &str, expected: &[&str]) -> Result<(), FormatError> {
    let header = header.strip_prefix('\u{feff}').unwrap_or(header);
    if header.starts_with('#') {
        return Err(FormatError::Tsv {
            line: 0,
            msg: "header starts with '#' comment; comments are allowed only before the header row".into(),
        });
    }
    let got: Vec<&str> = header.split('\t').map(|s| s.trim()).collect();
    if got.iter().map(|&c| c).cmp(expected.iter().map(|&c| c)) != std::cmp::Ordering::Equal {
        return Err(FormatError::Tsv {
            line: 0,
            msg: format!("header mismatch: got {got:?}, expected {expected:?}"),
        });
    }
    Ok(())
}

/// Parse one `edges.tsv` line (comments '#' and blank lines skipped upstream).
pub fn parse_edge_line(line: &str, line_no: u64) -> Result<Option<EdgeRecord>, FormatError> {
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != EDGE_COLUMNS.len() {
        return Err(FormatError::Tsv {
            line: line_no,
            msg: format!("expected {} tab-separated fields, got {}", EDGE_COLUMNS.len(), parts.len()),
        });
    }
    let body_pre = parse_u64(parts[0]).ok_or_else(|| FormatError::Tsv {
        line: line_no,
        msg: format!("body_pre not an exact decimal integer: {:?}", parts[0]),
    })?;
    let body_post = parse_u64(parts[1]).ok_or_else(|| FormatError::Tsv {
        line: line_no,
        msg: format!("body_post not an exact decimal integer: {:?}", parts[1]),
    })?;
    let weight = parse_f32(parts[2]).ok_or_else(|| FormatError::Tsv {
        line: line_no,
        msg: format!("weight not a finite f32: {:?}", parts[2]),
    })?;
    Ok(Some(EdgeRecord {
        body_pre,
        body_post,
        weight,
    }))
}

/// Parse one `nodes.tsv` line ("" fields retained as unannotated).
pub fn parse_node_line(line: &str, line_no: u64) -> Result<Option<NodeRecord>, FormatError> {
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != NODE_COLUMNS.len() {
        return Err(FormatError::Tsv {
            line: line_no,
            msg: format!("expected {} tab-separated fields, got {}", NODE_COLUMNS.len(), parts.len()),
        });
    }
    let body_id = parse_u64(parts[0]).ok_or_else(|| FormatError::Tsv {
        line: line_no,
        msg: format!("bodyId not an exact decimal integer: {:?}", parts[0]),
    })?;
    Ok(Some(NodeRecord {
        body_id,
        cell_type: parts[1].to_string(),
        region: parts[2].to_string(),
        status: parts[3].to_string(),
        soma_side: parts[4].to_string(),
    }))
}

fn parse_u64(s: &str) -> Option<u64> {
    if s.is_empty() || s.contains('+') || s.contains('-') {
        return None;
    }
    s.parse::<u64>().ok()
}

fn parse_f32(s: &str) -> Option<f32> {
    let v: f32 = s.parse().ok()?;
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

/// Streaming reader yielding validated edge records in file order.
pub struct EdgeTsvReader<R: BufRead> {
    inner: R,
    line_no: u64,
}

impl<R: BufRead> EdgeTsvReader<R> {
    pub fn new(mut inner: R) -> Result<Self, FormatError> {
        let mut first = String::new();
        // skip comment lines, then read header
        let n = read_header_line(&mut inner, &mut first)?;
        if n == 0 {
            return Err(FormatError::Tsv { line: 0, msg: "empty edges file".into() });
        }
        validate_header(&first, &EDGE_COLUMNS)?;
        Ok(Self { inner, line_no: 0 })
    }

    /// `Ok(None)` at EOF.
    pub fn next_edge(&mut self) -> Result<Option<EdgeRecord>, FormatError> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.inner.read_line(&mut line)?;
            if n == 0 {
                return Ok(None);
            }
            self.line_no += 1;
            let t = line.trim_end_matches(['\r', '\n']);
            let rec = parse_edge_line(t, self.line_no)?;
            if rec.is_some() {
                return Ok(rec);
            }
        }
    }
}

pub struct NodeTsvReader<R: BufRead> {
    inner: R,
    line_no: u64,
}

impl<R: BufRead> NodeTsvReader<R> {
    pub fn new(mut inner: R) -> Result<Self, FormatError> {
        let mut first = String::new();
        let n = read_header_line(&mut inner, &mut first)?;
        if n == 0 {
            return Err(FormatError::Tsv { line: 0, msg: "empty nodes file".into() });
        }
        validate_header(&first, &NODE_COLUMNS)?;
        Ok(Self { inner, line_no: 0 })
    }

    pub fn next_node(&mut self) -> Result<Option<NodeRecord>, FormatError> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.inner.read_line(&mut line)?;
            if n == 0 {
                return Ok(None);
            }
            self.line_no += 1;
            let t = line.trim_end_matches(['\r', '\n']);
            let rec = parse_node_line(t, self.line_no)?;
            if rec.is_some() {
                return Ok(rec);
            }
        }
    }
}

fn read_header_line<R: BufRead>(inner: &mut R, buf: &mut String) -> Result<usize, FormatError> {
    loop {
        buf.clear();
        let n = inner.read_line(buf)?;
        if n == 0 {
            return Ok(0);
        }
        let t = buf.trim_end_matches(['\r', '\n']);
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        *buf = t.to_string();
        // header is the FIRST non-comment row; comments may only precede it
        return Ok(n);
    }
}

pub fn is_edges_file(name: &str) -> bool {
    name == EDGES_FILE
}
pub fn is_nodes_file(name: &str) -> bool {
    name == NODES_FILE
}
pub fn is_sidecar_file(name: &str) -> bool {
    name == SIDECAR_FILE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SYNTHETIC sample data (see testdata/malecns-sample/README.md); embedded
    /// at compile time verbatim.
    const SAMPLE_EDGES: &str = include_str!("../testdata/malecns-sample/edges.tsv");

    #[test]
    fn parses_sample_edges_stream() {
        let mut r = EdgeTsvReader::new(std::io::Cursor::new(SAMPLE_EDGES.as_bytes()))
            .expect("header valid");
        let mut out = Vec::new();
        while let Some(e) = r.next_edge().expect("parse") {
            out.push(e);
        }
        assert_eq!(
            out,
            vec![
                EdgeRecord { body_pre: 1, body_post: 2, weight: 0.5 },
                EdgeRecord { body_pre: 2, body_post: 3, weight: 1.5 },
                EdgeRecord { body_pre: 3, body_post: 1, weight: -2.0 },
                EdgeRecord { body_pre: 1, body_post: 3, weight: 3.25 },
                EdgeRecord { body_pre: 4, body_post: 2, weight: 0.875 },
            ]
        );
    }

    #[test]
    fn header_must_match_exactly() {
        assert!(validate_header("body_pre\tbody_post\tweight", &EDGE_COLUMNS).is_ok());
        assert!(validate_header("body_post\tbody_pre\tweight", &EDGE_COLUMNS).is_err());
        assert!(validate_header("body_pre\tbody_post", &EDGE_COLUMNS).is_err());
        // BOM tolerated
        assert!(validate_header("\u{feff}body_pre\tbody_post\tweight", &EDGE_COLUMNS).is_ok());
    }

    #[test]
    fn parse_errors_are_positioned() {
        let err = parse_edge_line("x\t2\t1.0", 12).unwrap_err();
        assert!(match err {
            FormatError::Tsv { line: 12, .. } => true,
            _ => false,
        });
        assert!(parse_edge_line("1\t2\tnan", 1).is_err());
        assert!(parse_edge_line("1\t2\t-0", 1).is_ok(), "-0 is a finite f32");
        assert!(parse_edge_line("1\t2\tinf", 1).is_err());
        assert!(parse_u64("018") == Some(18), "leading zeros are still exact decimal");
        assert!(parse_u64("").is_none() && parse_u64("+1").is_none() && parse_u64("1.0").is_none());
    }

    #[test]
    fn nodes_missing_fields_stay_empty() {
        // 5 columns: status and somaSide both EMPTY strings here
        let rec = parse_node_line("7\tT5\tCNS\t\t", 3).unwrap();
        let rec = rec.expect("row");
        assert_eq!(rec.body_id, 7);
        assert_eq!(rec.cell_type, "T5");
        assert_eq!(rec.region, "CNS");
        assert_eq!(rec.status, "");
        assert_eq!(rec.soma_side, ""); // missing field, never invented
        let err = parse_node_line("7\tT5", 3).unwrap_err(); // too few fields
        assert!(matches!(err, FormatError::Tsv { line: 3, .. }));
    }

    #[test]
    fn sidecar_validates_format_and_version() {
        let ok = Sidecar {
            format: FORMAT_NAME.into(),
            format_version: FORMAT_VERSION,
            dataset: "S".into(),
            source_files: vec!["a.feather".into()],
            converter: "c".into(),
            converter_version: "1".into(),
            source_sha256: "".into(),
            rows_in: 0,
            rows_out_edges: 0,
            rows_out_nodes: 0,
            converted_at_utc: "".into(),
            notes: "".into(),
        };
        assert!(ok.validate().is_ok());
        let wrong = Sidecar {
            format: "other".into(),
            ..ok.clone()
        };
        assert!(wrong.validate().is_err());
        let future = Sidecar {
            format_version: FORMAT_VERSION + 9,
            ..ok
        };
        assert!(future.validate().is_err());
    }
}
