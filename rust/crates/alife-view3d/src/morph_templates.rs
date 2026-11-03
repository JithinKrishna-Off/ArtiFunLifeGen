//! Procedural organism template meshes (pure CPU, deterministic, built once
//! at `Morph::new`). Every organism brand reduces to a mix of three template
//! geometries, scaled/translucent per instance:
//!
//! - `SHELL`  unit-sphere shell stack: displaced membrane (breathing, ws id 0)
//!            + cytoplasm core (id 1) + nucleus-like inner sphere (id 2,
//!            alpha-gated per instance so it only exists when the trait
//!            table supports it, size > 1.1).
//! - `TAIL`   two tapered bezier flagella tubes extending backward
//!            (local -X), ids 3 (tail A) and 4 (tail B); the second tail is
//!            collapsed per instance when the species variant has one tail.
//! - `QUAD`   camera-facing billboard corner template (id 5) for organisms
//!            beyond the full-morph ring.
//!
//! Geometry convention: template space is body-radius units (radius 1.0),
//! local +X is the organism's heading direction; the vertex shader rotates
//! template verts by `rot_y(heading)` (template +X -> world forward).

use bytemuck::{Pod, Zeroable};

/// WC id codes (vertex aux.y): membrane / cytoplasm core / nucleus / tails / billboard.
pub const CODE_MEMBRANE: f32 = 0.0;
pub const CODE_CORE: f32 = 1.0;
pub const CODE_NUCLEUS: f32 = 2.0;
pub const CODE_TAIL_A: f32 = 3.0;
pub const CODE_TAIL_B: f32 = 4.0;
pub const CODE_BILLBOARD: f32 = 5.0;

// vertex ranges (all soup triangles, non-indexed)
pub const SHELL_VBASE: u32 = 0;
pub const SHELL_VCOUNT: u32 = 1440; // 320 faces discharged membrane + 2x 80-face inner spheres
pub const TAIL_VBASE: u32 = SHELL_VBASE + SHELL_VCOUNT;
pub const TAIL_VCOUNT: u32 = 1152; // 2 tails x 16 rings x 6 sides
pub const QUAD_VBASE: u32 = TAIL_VBASE + TAIL_VCOUNT;
pub const QUAD_VCOUNT: u32 = 6;

pub const QUEEN_VBASE: u32 = QUAD_VBASE + QUAD_VCOUNT;
/// Queen template vertex count (morph_queen: giant-insect queen — head,
/// thorax, banded abdomen, 6 jointed legs, antennae, eyes, mandibles, 2
/// wing quads — appended after the billboard quad in the uploaded soup).
pub const QUEEN_VCOUNT: u32 = super::queen::QUEEN_VCOUNT;

/// One template vertex: loc0 = vec4 (local pos, 0), loc1 = (param, id).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct TVert {
    pub pos: [f32; 4],
    pub aux: [f32; 2],
}

const TAU: f32 = std::f32::consts::TAU;

pub struct Templates {
    pub shell: Vec<TVert>,
    pub tail: Vec<TVert>,
    pub quad: Vec<TVert>,
}

impl Templates {
    pub fn vcount(&self) -> u32 {
        (self.shell.len() + self.tail.len() + self.quad.len()) as u32
    }
}

#[inline]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-9 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ------------------------------------------------------------- icosphere

/// Base icosahedron (12 verts / 20 faces), radius 1.
fn ico_base() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let v: Vec<[f32; 3]> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|p| norm3(*p))
    .collect();
    let f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (v, f)
}

/// midpoint vertex with subdivision cache (borrowed explicitly)
fn mid(cache: &mut std::collections::HashMap<u64, u32>, pos: &mut Vec<[f32; 3]>, a: u32, b: u32) -> u32 {
    let key = a.min(b) as u64 * (1 << 32) + a.max(b) as u64;
    if let Some(m) = cache.get(&key) {
        return *m;
    }
    let pa = pos[a as usize];
    let pb = pos[b as usize];
    let m = norm3([
        (pa[0] + pb[0]) * 0.5,
        (pa[1] + pb[1]) * 0.5,
        (pa[2] + pb[2]) * 0.5,
    ]);
    pos.push(m);
    cache.insert(key, pos.len() as u32 - 1);
    pos.len() as u32 - 1
}

/// Subdivided unit icosphere as soup triangle positions (3 verts / face).
/// Public so the queen template (morph_queen) reuses the exact same
/// base geometry family — no new geometry pipeline is introduced.
pub fn sphere_soup(depth: u32) -> Vec<[f32; 3]> {
    let (mut pos, mut idx) = ico_base();
    let mut cache: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for _ in 0..depth {
        let mut next = Vec::with_capacity(idx.len() * 4);
        for f in &idx {
            let (a, b, c) = (f[0], f[1], f[2]);
            let ab = mid(&mut cache, &mut pos, a, b);
            let bc = mid(&mut cache, &mut pos, b, c);
            let ca = mid(&mut cache, &mut pos, c, a);
            for tri in [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]] {
                next.push(tri);
            }
        }
        idx = next;
        cache.clear();
    }
    let mut out = Vec::with_capacity(idx.len() * 3);
    for f in &idx {
        for i in 0..3 {
            out.push(pos[f[i] as usize]);
        }
    }
    out
}

/// Two-lobed membrane displacement (static, baked): soft bumps on opposite
/// sides of the cell axis plus a low-amplitude organic sine ripple.
fn membrane_radius(n: [f32; 3]) -> f32 {
    let a1 = norm3([1.0, 0.25, 0.1]);
    let a2 = norm3([-1.0, 0.15, -0.3]);
    let d1 = (n[0] * a1[0] + n[1] * a1[1] + n[2] * a1[2]).max(0.0);
    let d2 = (n[0] * a2[0] + n[1] * a2[1] + n[2] * a2[2]).max(0.0);
    1.0 + 0.11 * d1 * d1 + 0.09 * d2 * d2 + 0.02 * (9.0 * n[0] + 4.1 * n[1] + 2.3 * n[2]).sin()
}

fn push_sphere(
    out: &mut Vec<TVert>,
    depth: u32,
    radius: f32,
    id: f32,
    param: f32,
    displace: bool,
) {
    for p in sphere_soup(depth) {
        let r = if displace { radius * membrane_radius(p) } else { radius };
        out.push(TVert {
            pos: [p[0] * r, p[1] * r, p[2] * r, 0.0],
            aux: [param, id],
        });
    }
}

// ------------------------------------------------------------- flagella

/// Cubic bezier point / tangent.
fn bez(cp: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
    let u = 1.0 - t;
    let a = u * u * u;
    let b = 3.0 * u * u * t;
    let c = 3.0 * u * t * t;
    let d = t * t * t;
    let mut o = [0.0; 3];
    for k in 0..3 {
        o[k] = a * cp[0][k] + b * cp[1][k] + c * cp[2][k] + d * cp[3][k];
    }
    o
}

fn bez_tan(cp: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
    let u = 1.0 - t;
    let mut o = [0.0; 3];
    for k in 0..3 {
        o[k] = 3.0 * u * u * (cp[1][k] - cp[0][k])
            + 6.0 * u * t * (cp[2][k] - cp[1][k])
            + 3.0 * t * t * (cp[3][k] - cp[2][k]);
    }
    norm3(o)
}

/// Ring basis perpendicular to the tangent (with a sane fallback).
fn tube_basis(tan: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up: [f32; 3] = if tan[1].abs() > 0.94 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let n0 = norm3([
        tan[1] * up[2] - tan[2] * up[1],
        tan[2] * up[0] - tan[0] * up[2],
        tan[0] * up[1] - tan[1] * up[0],
    ]);
    let b0 = norm3([
        tan[1] * n0[2] - tan[2] * n0[1],
        tan[2] * n0[0] - tan[0] * n0[2],
        tan[0] * n0[1] - tan[1] * n0[0],
    ]);
    (n0, b0)
}

/// Tapered bezier tube: `steps` rings x `sides` sides as soup quads.
/// Vertex aux = (ring param t, tail id) so the vertex shader can run the
/// flagellar wave along t.
fn push_tail(out: &mut Vec<TVert>, cp: &[[f32; 3]; 4], id: f32) {
    const STEPS: u32 = 16;
    const SIDES: u32 = 6;
    let radius = |t: f32| 0.17 * (1.0 - t).powf(0.8) + 0.015;
    for i in 0..STEPS {
        let t0 = i as f32 / STEPS as f32;
        let t1 = (i + 1) as f32 / STEPS as f32;
        let (c0, c1) = (bez(cp, t0), bez(cp, t1));
        let (tan0, tan1) = (bez_tan(cp, t0), bez_tan(cp, t1));
        let (n0, b0) = tube_basis(tan0);
        let (n1, b1) = tube_basis(tan1);
        let (r0, r1) = (radius(t0), radius(t1));
        for j in 0..SIDES {
            let a0 = TAU * j as f32 / SIDES as f32;
            let a1 = TAU * ((j + 1) % SIDES) as f32 / SIDES as f32;
            let pt = |c: [f32; 3], n: [f32; 3], b: [f32; 3], a: f32, r: f32| {
                [c[0] + (n[0] * a.cos() + b[0] * a.sin()) * r,
                 c[1] + (n[1] * a.cos() + b[1] * a.sin()) * r,
                 c[2] + (n[2] * a.cos() + b[2] * a.sin()) * r]
            };
            let v00 = pt(c0, n0, b0, a0, r0);
            let v01 = pt(c0, n0, b0, a1, r0);
            let v10 = pt(c1, n1, b1, a0, r1);
            let v11 = pt(c1, n1, b1, a1, r1);
            for (p, t) in [(v00, t0), (v10, t1), (v11, t1), (v00, t0), (v11, t1), (v01, t0)] {
                out.push(TVert { pos: [p[0], p[1], p[2], 0.0], aux: [t, id] });
            }
        }
    }
}

// flagella bezier chords (body-radius units); extend backward (-X = behind).
// Re-scaled for the ORGANISM_UNIT_RADIUS tiny rebase: reach ~2.5 body radii
// (was ~9.8) so the whole flagellate silhouette stays ~1.4 world units at
// size 1, and strands stay thin capsules hugging the tapered body.
const TAIL_A: [[f32; 3]; 4] = [
    [-0.60, 0.0, 0.0],
    [-1.35, 0.20, 0.16],
    [-2.05, -0.12, 0.28],
    [-2.48, 0.22, 0.06],
];
const TAIL_B: [[f32; 3]; 4] = [
    [-0.60, 0.08, -0.06],
    [-1.30, -0.18, 0.24],
    [-1.95, 0.16, -0.26],
    [-2.60, -0.15, 0.10],
];

// ------------------------------------------------------------- assembly

/// Builds all templates deterministically (same inputs => bit-identical soup).
pub fn build() -> Templates {
    let mut shell = Vec::with_capacity(SHELL_VCOUNT as usize);
    // membrane: 2 subdivisions, displaced (param 1 => breathing weight)
    push_sphere(&mut shell, 2, 1.0, CODE_MEMBRANE, 1.0, true);
    // cytoplasm core tint (fixed shell inside the membrane)
    push_sphere(&mut shell, 1, 0.80, CODE_CORE, 0.0, false);
    // nucleus-like inner sphere (alpha-gated per instance)
    push_sphere(&mut shell, 1, 0.42, CODE_NUCLEUS, 0.0, false);

    let mut tail = Vec::with_capacity(TAIL_VCOUNT as usize);
    push_tail(&mut tail, &TAIL_A, CODE_TAIL_A);
    push_tail(&mut tail, &TAIL_B, CODE_TAIL_B);

    let mut quad = Vec::with_capacity(QUAD_VCOUNT as usize);
    // corner template (x, y in [-1,1]); geometry expanded camera-facing in VS
    for c in [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ] {
        quad.push(TVert { pos: [c[0], c[1], 0.0, 0.0], aux: [0.0, CODE_BILLBOARD] });
    }

    debug_assert_eq!(shell.len() as u32, SHELL_VCOUNT);
    debug_assert_eq!(tail.len() as u32, TAIL_VCOUNT);
    debug_assert_eq!(quad.len() as u32, QUAD_VCOUNT);
    Templates { shell, tail, quad }
}

// --------------------------------------------------------------- A1 region
// Organism (non-queen) archetype-derived templates. Queen template ranges
// (QUEEN_VBASE/QUEEN_VCOUNT -> morph_queen) are agent-owned and untouched.

use crate::creature_mesh as cmet;

/// Organism mesh for a real sim-derived descriptor (archetype-cached,
/// deterministic, built by `creature_mesh`). The renderer agent integrates
/// these with `creature_mesh::hue_channel` + `RENDER_SCALE * body_length`
/// as the instance hue/radius (see creature_mesh module docs for the recipe).
pub fn archetype_mesh(m: &alife_common::morphology::CreatureMorph) -> &'static [TVert] {
    cmet::archetype_mesh(m)
}

/// Brand fallback (no sim-derived CreatureMorph available): a generic small
/// organism per brand rendered at the same silhouette sizes as the legacy
/// shell/flagella templates (~1.39 world units nominal, in the [0.8, 2.0]
/// band checked by morph_tests). brand: 0 cell / 1 flagellate / 2 colony /
/// 3 filament; queen templates stay in `morph_queen` (never here).
pub fn organism_brand_fallback(brand: u32) -> &'static [TVert] {
    cmet::brand_fallback(brand)
}

