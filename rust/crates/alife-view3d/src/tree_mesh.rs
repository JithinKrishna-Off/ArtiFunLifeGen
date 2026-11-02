//! Shared low-poly tree + bush meshes (pure CPU, deterministic).
//!
//! ONE static vertex buffer serves three sub-draws, laid out so the LOD
//! sub-draw trick works exactly like the grass blade lattice:
//!
//! ```text
//!   [ trunk  (144 verts / 48 tris) ][ canopy L0 (48 v / 16 tris) ]  -> LOD-B span: 0..192
//!   [ trunk ][ canopy L0 ][ canopy L1 ][ canopy L2 ]             -> LOD-A span: 0..288
//!   [ bush (36 verts / 12 tris) ]                                -> bush span: 288..324
//! ```
//!
//! LOD-B = shared trunk + the bottom canopy layer (<= 80 tris budget:
//! actual 64); LOD-A = trunk + 3 stacked canopy layers (<= 190 budget:
//! actual 96). The bush blob is appended after the tree spans and drawn
//! with a non-zero draw range start.
//!
//! No runtime RNG: every vertex jitter comes from an in-file integer hash
//! over explicit seeds, so the mesh is bit-identical across runs/tests.

use bytemuck::{Pod, Zeroable};

/// nominal tree mesh height (trunk 7.0 + top-canopy apex ~7.8); instance
/// `scale` = `world_height / NOM_TREE_H`
pub const NOM_TREE_H: f32 = 7.8;
/// nominal bush mesh height; instance `scale` = `world_height / NOM_BUSH_H`
pub const NOM_BUSH_H: f32 = 1.0;

/// triangle counts of the three draw spans (for the budget report)
pub const TRIS_LOD_A: u32 = 96;
pub const TRIS_LOD_B: u32 = 64;
pub const TRIS_BUSH: u32 = 12;

/// mesh vertex: 32 B = pos 3 + normal 3 + part (0 trunk / 1 canopy) +
/// yfrac (vertical fraction 0..1 within the nominal mesh height)
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct MeshVert {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub part: f32,
    pub yfrac: f32,
}

pub struct TreeMesh {
    pub verts: Vec<MeshVert>,
    /// LOD-B draw span: [0, lod_b_len) (trunk + canopy L0)
    pub lod_b_len: u32,
    /// LOD-A draw span: [0, full_len)
    pub full_len: u32,
    /// bush draw span: [bush_first, bush_first + bush_len)
    pub bush_first: u32,
    pub bush_len: u32,
}

const TAU: f32 = std::f32::consts::TAU;

// ---------------------------------------------------------------- hashes

/// xorshift-style integer hash (32 bit) — deterministic, no RNG state
fn ih32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// uniform-ish float in [0,1) from two integer seeds
fn hf(a: u32, b: u32) -> f32 {
    let v = ih32(a ^ b.wrapping_mul(0x9E37_79B9));
    ((v >> 8) as f32) / 0x0100_0000 as f32
}

// ------------------------------------------------------------- geometry

fn push_tri(out: &mut Vec<MeshVert>, a: [f32; 3], b: [f32; 3], c: [f32; 3], part: f32, nom_h: f32) {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let mut n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 1e-9 {
        n = [n[0] / l, n[1] / l, n[2] / l];
    } else {
        n = [0.0, 1.0, 0.0];
    }
    for p in [a, b, c] {
        out.push(MeshVert { pos: p, nrm: n, part, yfrac: (p[1] / nom_h).clamp(0.0, 1.0) });
    }
}

/// one ring vertex around (cx, cz); the per-k angle jitter is FIXED per k
/// (shared by every ring) so the tube stays untwisted
fn ring_pt(cx: f32, cz: f32, r: f32, y: f32, k: u32, segs: u32, seed: u32) -> [f32; 3] {
    let ang = k as f32 / segs as f32 * TAU + (hf(seed, k.wrapping_mul(7).wrapping_add(1)) - 0.5) * 0.14;
    [cx + r * ang.cos(), y, cz + r * ang.sin()]
}

/// tapered trunk: 8 verts/ring, 4 rings ( ys/radii below), 3 bands of
/// quads = 144 verts / 48 tris
fn push_trunk(out: &mut Vec<MeshVert>, seed: u32) {
    const SEGS: u32 = 8;
    const RINGS: [(f32, f32); 4] = [(0.0, 0.50), (2.2, 0.42), (4.6, 0.33), (7.0, 0.24)];
    for band in 0..RINGS.len() - 1 {
        let (y0, r0) = RINGS[band];
        let (y1, r1) = RINGS[band + 1];
        for k in 0..SEGS {
            let p00 = ring_pt(0.0, 0.0, r0, y0, k, SEGS, seed);
            let p01 = ring_pt(0.0, 0.0, r0, y0, k + 1, SEGS, seed);
            let p10 = ring_pt(0.0, 0.0, r1, y1, k, SEGS, seed);
            let p11 = ring_pt(0.0, 0.0, r1, y1, k + 1, SEGS, seed);
            push_tri(out, p00, p01, p11, 0.0, NOM_TREE_H);
            push_tri(out, p00, p11, p10, 0.0, NOM_TREE_H);
        }
    }
}

/// one flattened-canopy layer (bipyramid): ring of `SEGS` points between a
/// jittered top apex and a jittered bottom apex = 48 verts / 16 tris
fn push_layer(out: &mut Vec<MeshVert>, y_bot: f32, y_top: f32, r: f32, seed: u32) {
    const SEGS: u32 = 8;
    let y_mid = (y_bot + y_top) * 0.5;
    let top = [
        (hf(seed, 101) - 0.5) * 0.6,
        y_top,
        (hf(seed, 202) - 0.5) * 0.6,
    ];
    let bot = [
        (hf(seed, 303) - 0.5) * 0.4,
        y_bot,
        (hf(seed, 404) - 0.5) * 0.4,
    ];
    for k in 0..SEGS {
        let a = ring_pt(0.0, 0.0, r, y_mid, k, SEGS, seed);
        let b = ring_pt(0.0, 0.0, r, y_mid, k + 1, SEGS, seed);
        push_tri(out, a, top, b, 1.0, NOM_TREE_H);
        push_tri(out, b, bot, a, 1.0, NOM_TREE_H);
    }
}

/// bush blob: 6-sided flattened globe (top fan + bottom fan over a wide
/// mid ring) = 36 verts / 12 tris; nominal radius 1.2, height 1.0
fn push_bush(out: &mut Vec<MeshVert>, seed: u32) {
    const SEGS: u32 = 6;
    let top = [(hf(seed, 11) - 0.5) * 0.3, 1.0, (hf(seed, 22) - 0.5) * 0.3];
    let bot = [(hf(seed, 33) - 0.5) * 0.3, 0.0, (hf(seed, 44) - 0.5) * 0.3];
    for k in 0..SEGS {
        let a = ring_pt(0.0, 0.0, 1.2, 0.55, k, SEGS, seed);
        let b = ring_pt(0.0, 0.0, 1.2, 0.55, k + 1, SEGS, seed);
        push_tri(out, a, top, b, 1.0, NOM_BUSH_H);
        push_tri(out, b, bot, a, 1.0, NOM_BUSH_H);
    }
}

/// Canonical spans (asserted so a geometry drift fails loudly in tests):
/// trunk 144 verts, canopy layer 48 verts, LOD-B = 192 verts (64 tris),
/// LOD-A = 288 verts (96 tris), bush 36 verts (12 tris).
pub fn build() -> TreeMesh {
    let mut verts = Vec::with_capacity(324);
    push_trunk(&mut verts, 0xA1B2_C3D4);
    // LOD-B carries only the bottom canopy layer (shared trunk is reused
    // by both LODs, so its span always starts at vertex 0)
    push_layer(&mut verts, 4.3, 6.0, 2.5, 0x51F0_0D1);
    let lod_b_len = verts.len() as u32;
    debug_assert_eq!(lod_b_len, 192);
    push_layer(&mut verts, 5.4, 6.9, 1.9, 0x51F0_0D2);
    push_layer(&mut verts, 6.4, 7.8, 1.35, 0x51F0_0D3);
    let full_len = verts.len() as u32;
    debug_assert_eq!(full_len, 288);
    let bush_first = full_len;
    push_bush(&mut verts, 0xB055_00D5);
    let bush_len = verts.len() as u32 - bush_first;
    debug_assert_eq!(bush_len, 36);
    TreeMesh { verts, lod_b_len, full_len, bush_first, bush_len }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_and_tris_match_budget() {
        let m = build();
        assert_eq!(m.lod_b_len, 192);
        assert_eq!(m.full_len, 288);
        assert_eq!(m.bush_first, 288);
        assert_eq!(m.bush_len, 36);
        // non-indexed: verts = 3 * tris
        assert_eq!(m.lod_b_len / 3, TRIS_LOD_B);
        assert_eq!(m.full_len / 3, TRIS_LOD_A);
        assert_eq!(m.bush_len / 3, TRIS_BUSH);
        // budgets from the mission
        assert!(TRIS_LOD_A <= 190, "{}", TRIS_LOD_A);
        assert!(TRIS_LOD_B <= 80, "{}", TRIS_LOD_B);
    }

    #[test]
    fn mesh_is_deterministic() {
        let a = build();
        let b = build();
        assert_eq!(a.verts.len(), b.verts.len());
        for (x, y) in a.verts.iter().zip(b.verts.iter()) {
            assert_eq!(x.pos, y.pos);
            assert_eq!(x.nrm, y.nrm);
        }
    }

    #[test]
    fn tree_heights_within_spec() {
        let m = build();
        let ymax = m.verts[..m.full_len as usize].iter().map(|v| v.pos[1]).fold(0.0f32, f32::max);
        // world 5.5..9.5 => mesh 7.8 tall times scale 0.705..1.218
        assert!((ymax - 7.8).abs() < 0.1, "ymax {}", ymax);
        let all_finite = m.verts.iter().all(|v| {
            v.pos.iter().all(|x| x.is_finite()) && v.nrm.iter().all(|x| x.is_finite() && x.abs() <= 1.001)
        });
        assert!(all_finite);
    }
}
