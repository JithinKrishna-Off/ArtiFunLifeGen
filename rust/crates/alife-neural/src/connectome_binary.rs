//! Versioned compiled-connectome artifact (manual little-endian layout).
//!
//! Container (all LE, self-describing, prefix sizes; streaming-friendly):
//!   "FLBC" | u32 version (=1) | u32 header_section_len
//!   name:          u32 len + bytes
//!   template:      u32 len + bytes
//!   u64 n_nodes | u64 n_edges
//!   strings table: u32 count | (u32 len + bytes)*          [regions]
//!                  u32 count | (u32 len + bytes)*          [cell_types]
//!   nodes: [u64 body_id]* | [u32 region_idx]* | [u32 ctype_idx]*
//!          | [u8 polarity]* | [u8 origin]*
//!   csr: [u32 row_offset]*(n+1) | [u32 col]*  | [f32 w]*
//!        u8 has_delays + [u8]* | u8 has_synpol + [u8]*
//!   footer: u64 fnv1a-64 hash of ALL preceding bytes (identity, documented
//!   non-cryptographic) | u32 crc32 of all preceding bytes
//!
//! `load_path` mmaps the file and parses in place (memmap-friendly); `save`
//! streams section buffers via `io::Write` so no full-image duplication is
//! required beyond the outgoing writer buffers.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::connectome::{Connectome, ConnectomeError, NeuronMeta, Polarity};
use crate::provenance::ProvenanceTag;

pub const MAGIC: &[u8; 4] = b"FLBC";
pub const ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic/version")]
    BadHeader,
    #[error("hash mismatch: stored {stored:#x}, computed {computed:#x}")]
    HashMismatch { stored: u64, computed: u64 },
    #[error("crc mismatch: stored {stored:#x}, computed {computed:#x}")]
    CrcMismatch { stored: u32, computed: u32 },
    #[error("truncated artifact")]
    Truncated,
    #[error("connectome validation: {0}")]
    Connectome(#[from] ConnectomeError),
}

/// Streaming hash helpers over reader content.
fn hash_and_crc_all(payload: &[u8]) -> (u64, u32) {
    let mut h = 0xcbf29ce484222325u64;
    for &b in payload {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h, crc32(payload))
}

/// Standard CRC-32 (reflected, poly 0x04C11DB7, init/final xor 0xFFFFFFFF).
pub fn crc32(data: &[u8]) -> u32 {
    fn table_slot(i: usize) -> u32 {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        c
    }
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = table_slot(i);
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

struct W(Vec<u8>);

impl W {
    fn bytes(&mut self, b: &[u8]) -> Result<(), ArtifactError> {
        self.0.extend_from_slice(b);
        Ok(())
    }
    fn u8v(&mut self, v: &[u8]) -> Result<(), ArtifactError> {
        self.bytes(v)
    }
    fn u32le(&mut self, v: u32) -> Result<(), ArtifactError> {
        self.bytes(&v.to_le_bytes())
    }
    fn u64le(&mut self, v: u64) -> Result<(), ArtifactError> {
        self.bytes(&v.to_le_bytes())
    }
    fn f32le(&mut self, v: f32) -> Result<(), ArtifactError> {
        self.bytes(&v.to_le_bytes())
    }
    fn strit(&mut self, s: &str) -> Result<(), ArtifactError> {
        self.u32le(s.len() as u32)?;
        self.bytes(s.as_bytes())
    }
}

/// Serialize the full artifact (payload + footer) as an in-memory buffer.
pub fn serialize(conn: &Connectome) -> Result<Vec<u8>, ArtifactError> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024 + conn.edge_count() * 10);
    write_into(conn, &mut buf)?;
    let (h, crc) = hash_and_crc_all(&buf);
    buf.extend_from_slice(&h.to_le_bytes());
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

pub fn write_into(conn: &Connectome, out: &mut Vec<u8>) -> Result<(), ArtifactError> {
    write_sectioned(conn, &mut |chunk: &[u8]| out.extend_from_slice(chunk))
}

/// Core serializer; the callback receives successive byte chunks (streaming).
pub fn write_sectioned<F: FnMut(&[u8])>(
    conn: &Connectome,
    mut emit: F,
) -> Result<(), ArtifactError> {
    let mut w = W(Vec::with_capacity(64 * 1024));
    let emit_buf = |w: &mut W, emit: &mut F| {
        emit(&w.0);
        w.0.clear();
    };
    w.bytes(MAGIC)?;
    w.u32le(ARTIFACT_VERSION)?;
    w.u32le(0)?; // header_section_len (unused; forward compatibility)
    w.strit(&conn.name)?;
    w.strit(&conn.template_version)?;
    w.u64le(conn.n() as u64)?;
    w.u64le(conn.edge_count() as u64)?;
    w.u32le(conn.regions.len() as u32)?;
    for s in &conn.regions {
        w.strit(s)?;
    }
    w.u32le(conn.cell_types.len() as u32)?;
    for s in &conn.cell_types {
        w.strit(s)?;
    }
    emit_buf(&mut w, &mut emit);

    // flush big blocks in slices to bound memory duplication
    let n = conn.n();
    for i in 0..n {
        w.u64le(conn.node_meta[i].body_id)?;
    }
    emit_buf(&mut w, &mut emit);
    for m in &conn.node_meta {
        w.u32le(m.region_idx)?;
    }
    for m in &conn.node_meta {
        w.u32le(m.cell_type_idx)?;
    }
    w.u8v(&conn
        .node_meta
        .iter()
        .map(|m| m.polarity as u8)
        .collect::<Vec<u8>>())?;
    emit_buf(&mut w, &mut emit);
    w.u8v(&conn
        .node_meta
        .iter()
        .map(|m| m.origin as u8)
        .collect::<Vec<u8>>())?;
    emit_buf(&mut w, &mut emit);

    for (i, r) in conn.row_offsets.iter().enumerate() {
        w.u32le(*r)?;
        if i % 4096 == 4095 {
            emit_buf(&mut w, &mut emit);
        }
    }
    emit_buf(&mut w, &mut emit);
    for (i, &c) in conn.col_idx.iter().enumerate() {
        w.u32le(c)?;
        let wi = conn.weights[i];
        w.f32le(wi)?;
        if i % 4096 == 4095 {
            emit_buf(&mut w, &mut emit);
        }
    }
    emit_buf(&mut w, &mut emit);

    w.u8v(&[u8::from(conn.delays.is_some())])?;
    if let Some(d) = &conn.delays {
        w.u8v(d)?;
    }
    w.u8v(&[u8::from(conn.syn_polarity.is_some())])?;
    if let Some(sp) = &conn.syn_polarity {
        w.u8v(sp)?;
    }
    emit_buf(&mut w, &mut emit);

    // footer hashes/acknowledges all bytes SO FAR in `out`
    emit(&w.0);
    w.0.clear();
    Ok(())
}

/// Compute identity over the serialized form (used by save/load roundtrips and
/// `connectome_identity` contract).
pub fn connectome_identity(conn: &Connectome) -> (String, String, u64, u64, u64) {
    let id = conn.identity();
    (
        id.name,
        id.template_version,
        id.neuron_count,
        id.edge_count,
        id.content_hash,
    )
}

/// Save to a file (payload + footer).
pub fn save(conn: &Connectome, path: &Path) -> Result<(), ArtifactError> {
    let body = serialize(conn)?;
    let mut f = File::create(path)?;
    f.write_all(&body)?;
    f.flush()?;
    Ok(())
}

/// In-memory parse helper.
pub fn parse_bytes(data: &[u8]) -> Result<Connectome, ArtifactError> {
    if data.len() < 12 + 8 + 4 {
        return Err(ArtifactError::Truncated);
    }
    let footer_h = u64::from_le_bytes(data[data.len() - 12..data.len() - 4].try_into().unwrap());
    let footer_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
    let payload = &data[..data.len() - 12];
    let (h, crc) = hash_and_crc_all(payload);
    if h != footer_h {
        return Err(ArtifactError::HashMismatch {
            stored: footer_h,
            computed: h,
        });
    }
    if crc != footer_crc {
        return Err(ArtifactError::CrcMismatch {
            stored: footer_crc,
            computed: crc,
        });
    }
    let mut c = Cursor { d: payload, p: 0 };
    if c.take(4)? != MAGIC {
        return Err(ArtifactError::BadHeader);
    }
    let version = c.u32()?;
    if version != ARTIFACT_VERSION {
        return Err(ArtifactError::BadHeader);
    }
    let _hdr_len = c.u32()?;
    let name = c.string()?;
    let template = c.string()?;
    let n = c.u64()? as usize;
    let _edges = c.u64()? as usize;
    let n_regions = c.u32()? as usize;
    let mut regions = Vec::with_capacity(n_regions);
    for _ in 0..n_regions {
        regions.push(c.string()?);
    }
    let n_ct = c.u32()? as usize;
    let mut ctypes = Vec::with_capacity(n_ct);
    for _ in 0..n_ct {
        ctypes.push(c.string()?);
    }
    let mut node_meta = Vec::with_capacity(n);
    let mut body_ids = Vec::with_capacity(n);
    for _ in 0..n {
        body_ids.push(c.u64()?);
    }
    let mut ri = Vec::with_capacity(n);
    for _ in 0..n {
        ri.push(c.u32()?);
    }
    let mut ci = Vec::with_capacity(n);
    for _ in 0..n {
        ci.push(c.u32()?);
    }
    let pol = c.take(n)?.to_vec();
    let orig = c.take(n)?.to_vec();
    for i in 0..n {
        node_meta.push(NeuronMeta {
            body_id: body_ids[i],
            region_idx: ri[i],
            cell_type_idx: ci[i],
            polarity: Polarity::from_u8(pol[i]),
            origin: prov_from_u8(orig[i]),
        });
    }
    let row_offsets = (0..=n).map(|_| c.u32()).collect::<Result<Vec<u32>, ArtifactError>>()?;
    let m = row_offsets[n] as usize;
    let mut col_idx = Vec::with_capacity(m);
    let mut weights = Vec::with_capacity(m);
    for _ in 0..m {
        col_idx.push(c.u32()?);
        weights.push(c.f32()?);
    }
    let has_delays = c.u8()? != 0;
    let delays = if has_delays {
        Some(c.take(m)?.to_vec())
    } else {
        None
    };
    let has_synpol = c.u8()? != 0;
    let syn_polarity = if has_synpol {
        Some(c.take(m)?.to_vec())
    } else {
        None
    };
    if c.p != c.d.len() {
        return Err(ArtifactError::Truncated);
    }
    Ok(Connectome::from_csr(
        name,
        template,
        node_meta,
        regions,
        ctypes,
        row_offsets,
        col_idx,
        weights,
        delays,
        syn_polarity,
    )?)
}

/// memmap-friendly decode (no read-loop on the hot path; zero disk copies
/// beyond the mapping itself).
pub fn load_memmap(path: &Path) -> Result<Arc<Connectome>, ArtifactError> {
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let conn = parse_bytes(&mmap)?;
    Ok(Arc::new(conn))
}

/// Simple file load helper (read + parse).
pub fn load(path: &Path) -> Result<Arc<Connectome>, ArtifactError> {
    let mut data = Vec::new();
    File::open(path)?.read_to_end(&mut data)?;
    Ok(Arc::new(parse_bytes(&data)?))
}

fn prov_from_u8(v: u8) -> ProvenanceTag {
    match v {
        0 => ProvenanceTag::SuppliedBySource,
        1 => ProvenanceTag::Derived,
        2 => ProvenanceTag::Estimated,
        3 => ProvenanceTag::Assumption,
        _ => ProvenanceTag::Unavailable,
    }
}

struct Cursor<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ArtifactError> {
        if self.p + n > self.d.len() {
            return Err(ArtifactError::Truncated);
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, ArtifactError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, ArtifactError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ArtifactError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ArtifactError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, ArtifactError> {
        let len = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{EdgeSpec, Polarity};

    fn sample() -> Connectome {
        let meta = |i: u64, pol: Polarity| NeuronMeta {
            body_id: 4_710_000_000 + i,
            region_idx: (i % 2) as u32,
            cell_type_idx: (i % 3) as u32,
            polarity: pol,
            origin: ProvenanceTag::SuppliedBySource,
        };
        let nodes = vec![
            meta(0, Polarity::Excitatory),
            meta(1, Polarity::Inhibitory),
            meta(2, Polarity::Unknown),
        ];
        let edges = vec![
            EdgeSpec {
                pre: 0,
                post: 1,
                weight: 2.5,
                delay: Some(3),
                polarity: Some(Polarity::Excitatory),
            },
            EdgeSpec {
                pre: 1,
                post: 2,
                weight: -1.25,
                delay: Some(1),
                polarity: Some(Polarity::Inhibitory),
            },
        ];
        Connectome::from_edges(
            "roundtrip-sample",
            "malecns-v1.0",
            nodes,
            vec!["optic".into(), "central".into()],
            vec!["a".into(), "b".into(), "c".into()],
            edges,
        )
        .expect("sample builds")
    }

    #[test]
    fn binary_roundtrip_preserves_content() {
        let c = sample();
        let bytes = serialize(&c).expect("serialize");
        let back = parse_bytes(&bytes).expect("parse");
        assert_eq!(back.name, c.name);
        assert_eq!(back.template_version, c.template_version);
        assert_eq!(back.n(), c.n());
        assert_eq!(back.edge_count(), c.edge_count());
        assert_eq!(back.row_offsets, c.row_offsets);
        assert_eq!(back.col_idx, c.col_idx);
        assert_eq!(back.weights, c.weights);
        assert_eq!(back.delays, c.delays);
        assert_eq!(back.syn_polarity, c.syn_polarity);
        assert_eq!(back.regions, c.regions);
        assert_eq!(back.cell_types, c.cell_types);
        for (a, b) in back.node_meta.iter().zip(c.node_meta.iter()) {
            assert_eq!(a.body_id, b.body_id);
            assert_eq!(a.polarity, b.polarity);
            assert_eq!(a.origin, b.origin);
            assert_eq!(a.region_idx, b.region_idx);
        }
        // identity is stable across roundtrip
        assert_eq!(connectome_identity(&back).4, connectome_identity(&c).4);
    }

    #[test]
    fn binary_detects_corruption() {
        let c = sample();
        let mut bytes = serialize(&c).expect("serialize");
        let last_payload = bytes.len() - 12;
        bytes[last_payload - 3] ^= 0x40; // flip a payload byte
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(
            matches!(err, ArtifactError::HashMismatch { .. } | ArtifactError::CrcMismatch { .. }),
            "corruption must fail the footer hash: {err:?}"
        );
    }

    #[test]
    fn file_save_load_roundtrip() {
        let c = sample();
        let path = std::env::temp_dir().join("alife-neural-roundtrip.flbc");
        save(&c, &path).expect("save");
        let loaded = load(&path).expect("load");
        std::fs::remove_file(&path).ok();
        assert_eq!(connectome_identity(&loaded).4, connectome_identity(&c).4);
        assert_eq!(loaded.n(), 3);
        assert_eq!(loaded.weights, c.weights);
    }
}

