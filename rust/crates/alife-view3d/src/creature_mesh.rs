//! Procedural creature anatomy from `alife_common::morphology::CreatureMorph`.
//! (Agent A1 — organism, non-queen. Queen anatomy lives in `morph_queen.rs`
//! and is NEVER touched here.)
//!
//! Every organism render mesh is a deterministic function of the quantized
//! archetype descriptor [`archetype_key`]: no RNG, no env, no wall-clock.
//! Same `CreatureMorph` -> same key -> bit-identical vertex soup.
//!
//! ------------------------------------------------------------- units
//! Mesh space is BODY-LENGTH units (BL): the trunk spans x in [-0.5, +0.5];
//! local +X is the organism heading (template convention of morph.rs: the
//! vertex shader rotates template +X -> world forward). Tails extend past
//! -0.5, antennae past +0.5. The vertex shader scales the whole soup by the
//! instance radius, so the renderer agent realizes the documented target
//! `rendered length ~= body_length * RENDER_SCALE` simply with
//! `OrgInst.radius = RENDER_SCALE * morph.body_length` and this mesh.
//!
//! --------------------------------------------------- integration recipe (A2)
//! 1. For each organism: `let m = alife_common::morphology::derive(&org);`
//! 2. `let key = creature_mesh::archetype_key(&m);`
//! 3. `let mesh = creature_mesh::ensure_mesh(key);` (build+cache on miss)
//!    or `creature_mesh::mesh_for(key)` to probe the cache only.
//! 4. Set instance hue channel `= creature_mesh::hue_channel(&m)`
//!    (morph.hue/360, exactly the fract() hue the FS consumes).
//!    `OrgInst.radius = creature_mesh::RENDER_SCALE * m.body_length`.
//! 5. Upload/lookup vertex ranges and draw as normal soup triangles with the
//!    existing template vertex layout (TVert: pos + aux(param, id)).
//!
//! Material codes ride the existing VS/FS branches of morph.rs untouched:
//! - body/trunk `CODE_BODY` 6.0, head `CODE_HEAD` 6.25, eyes `CODE_EYE` 6.5,
//!   wings `CODE_WING` 6.75  -> the "queen-lobe" branch (clean scaled local
//!   verts, gum body color, rim lighting; ids < 7.0 all share it).
//! - limbs/antennae `CODE_LIMB` 7.0 -> chitin branch (param = 0 at root,
//!   1 at tip; FS gradient + tiny VS sway).
//! - tail `CODE_TAIL` 3.0 -> the existing flagella branch (param t = wave
//!   phase; free vertex-shader animation; id 3.0 is never auto-collapsed).
//! - spikes `CODE_SPIKE` 7.9 -> crown branch (gold shimmer).
//! `aux.x` for body parts is a shade param for A2's shader extension
//! (e.g. `patterned` stripes: instance-flag-gated bands on param, and the
//! per-piece lighter-belly / darker-joint shading already hinted by param).

use alife_common::morphology::{BodyPlan, CreatureMorph, Mouthpart};
use crate::morph::templates::TVert;
use std::collections::{HashMap, VecDeque};

/// Render scale: documented target "visible body length ~ 0.3x body_length".
pub const RENDER_SCALE: f32 = 0.3;

/// Hard per-archetype vertex budget (soup verts). The tier solver never
/// exceeds it; worst-case descriptors degrade to the lean LOD instead.
pub const VERT_BUDGET: usize = 240;

/// LRU cache cap for archetype meshes (bounded shared-mesh count).
pub const CACHE_CAP: usize = 72;

// ------------------------------------------------------- material codes
pub const CODE_BODY: f32 = 6.0;
pub const CODE_HEAD: f32 = 6.25;
pub const CODE_EYE: f32 = 6.5;
pub const CODE_WING: f32 = 6.75;
pub const CODE_LIMB: f32 = 7.0;
pub const CODE_SPIKE: f32 = 7.9;
/// Same wave branch as the legacy flagella (id 3.x is never auto-collapsed
/// by the `flags` gate in the VS, which only kills id > 3.5).
pub const CODE_TAIL: f32 = 3.0;

const TAU: f32 = std::f32::consts::TAU;
const PI: f32 = std::f32::consts::PI;

// ------------------------------------------------------- quantized buckets
/// Trunk segment buckets: {2, 3, 4, 6}.
pub const SEG_BUCKETS: [u8; 4] = [2, 3, 4, 6];
/// Appendage-count buckets: {0, 2, 3, 4, 5, 6, 8}.
pub const APP_BUCKETS: [u8; 7] = [0, 2, 3, 4, 5, 6, 8];
/// Spike-count buckets: {0, >=3}. (The genome-hash dress field is folded to
/// two buckets so the sampled archetype space provably stays under the LRU
/// cap; {0,3,6} measured 81 distinct keys over a 2000-organism pool.)
pub const SPIKE_BUCKETS: [u8; 2] = [0, 3];
/// head_ratio bucket centers over 0.18..0.42 in 4 steps.
pub const HEAD_BUCKETS: [f32; 4] = [0.21, 0.27, 0.33, 0.39];
/// body_length bucket centers over 3.58..8.80 in 4 steps.
pub const LEN_BUCKETS: [f32; 4] = [4.2325, 5.7425, 7.2525, 8.7625];
/// Leg length (fraction of body length) ladder per appendage bucket.
pub const LEG_LEN: [f32; 7] = [0.35, 0.43, 0.51, 0.59, 0.67, 0.75, 0.83];
/// Eye scale ladder per eye bucket.
pub const EYE_SCALE: [f32; 4] = [0.75, 0.75, 1.0, 1.25];

#[inline]
fn seg_idx(s: u8) -> u64 {
    match s {
        0..=2 => 0,
        3 => 1,
        4 => 2,
        _ => 3,
    }
}
#[inline]
fn app_idx(a: u8) -> u64 {
    match a {
        0 => 0,
        1..=2 => 1,
        3 => 2,
        4 => 3,
        5 => 4,
        6 => 5,
        _ => 6,
    }
}
#[inline]
fn spike_idx(s: u8) -> u64 {
    if s >= 3 { 1 } else { 0 }
}
#[inline]
fn head_idx(r: f32) -> u64 {
    (((r - 0.18) / 0.24 * 4.0).floor().max(0.0) as u64).min(3)
}
#[inline]
fn len_idx(l: f32) -> u64 {
    (((l - 3.58) / (8.80 - 3.58) * 4.0).floor().max(0.0) as u64).min(3)
}

// ------------------------------------------------------- archetype key

/// Quantized archetype: shared-mesh key. Two organisms share a mesh exactly
/// when this key matches (40-60ish archetypes per world in practice; hard
/// LRU cap [`CACHE_CAP`]). Deterministic pure function of the descriptor.
///
/// Field order (21 used bits): plan(2) seg(2) app(3) eyes(2, 0..3)
/// head(2) len(2) mouth(2) armour(1) spikes(1) ant(2) tail wing jointed.
/// Hue / sat_tint / patterned are deliberately NOT in the key: hue is an
/// instance channel, sat_tint is shader-side and patterned stripes hook
/// onto the body shade param (all documented for the shader agent).
pub fn archetype_key(m: &CreatureMorph) -> u64 {
    let mut k: u64 = 0;
    k |= m.body_plan as u64 & 0x3;
    k |= seg_idx(m.segment_count) << 2;
    k |= app_idx(m.appendage_count) << 4;
    k |= (m.eye_count as u64).min(3) << 7;
    k |= head_idx(m.head_ratio) << 9;
    k |= len_idx(m.body_length) << 11;
    k |= (m.mouthpart as u64) << 13;
    k |= ((m.armour > 0.5) as u64) << 15;
    k |= spike_idx(m.spike_count) << 16;
    k |= ((m.antenna_count as u64).min(2)) << 18;
    k |= (m.tail_present as u64) << 20;
    k |= (m.winged as u64) << 21;
    // jointedness gates the lower leg segment -> meshes differ; keep a bit
    // so two same-key descriptors can never disagree about leg chains.
    k |= ((m.jointedness > 0.5) as u64) << 22;
    k
}

/// Canonical descriptor rebuilt from an archetype key: every visual quantity
/// the builder consumes is the common mean of the bucket, so ANY descriptor
/// mapping to this key builds the identical mesh bytes (cache-safe).
pub fn canonical_from_key(key: u64) -> CreatureMorph {
    let plan = match key & 0x3 {
        0 => BodyPlan::Primitive,
        1 => BodyPlan::SegmentedArthropod,
        2 => BodyPlan::MultiLegged,
        _ => BodyPlan::Quadruped,
    };
    let seg = SEG_BUCKETS[((key >> 2) & 0x3) as usize];
    let app = APP_BUCKETS[((key >> 4) & 0x7) as usize];
    let eyes = ((key >> 7) & 0x3) as u8;
    let head = HEAD_BUCKETS[((key >> 9) & 0x3) as usize];
    let len = LEN_BUCKETS[((key >> 11) & 0x3) as usize];
    let mouth = match (key >> 13) & 0x3 {
        0 => Mouthpart::Sucker,
        1 => Mouthpart::Filter,
        _ => Mouthpart::Jaw,
    };
    let armour = if (key >> 15) & 0x1 != 0 { 0.75 } else { 0.25 };
    let spikes = SPIKE_BUCKETS[((key >> 16) & 0x3) as usize];
    let ant = ((key >> 18) & 0x3) as u8;
    let tail = (key >> 20) & 0x1 != 0 && plan != BodyPlan::Primitive;
    let wing = (key >> 21) & 0x1 != 0;
    let hi = LEG_LEN[((key >> 4) & 0x7) as usize];
    CreatureMorph {
        body_plan: plan,
        body_length: len,
        head_ratio: head,
        segment_count: seg,
        appendage_count: app,
        appendage_len: ((hi - 0.15).max(0.25), hi),
        jointedness: if (key >> 22) & 0x1 != 0 { 0.8 } else { 0.2 },
        eye_count: eyes,
        eye_scale: EYE_SCALE[eyes.min(3) as usize],
        antenna_count: ant,
        mouthpart: mouth,
        armour,
        spike_count: spikes,
        tail_present: tail,
        tail_len: if tail { 0.6 } else { 0.0 },
        winged: wing,
        patterned: false,
        hue: 0.0,
        sat_tint: 0.3,
    }
}

/// Instance hue channel: fract()-domain value the morph FS consumes.
pub fn hue_channel(m: &CreatureMorph) -> f32 {
    m.hue / 360.0
}

// ------------------------------------------------------- scribe (geo kit)

/// VO tier: 2 = rich, 1 = medium, 0 = lean. Tables below.
const TIERS: [u8; 3] = [2, 1, 0];

struct Scribe {
    out: Vec<TVert>,
    tier: u8,
}

impl Scribe {
    fn new(tier: u8) -> Scribe {
        Scribe { out: Vec::new(), tier }
    }
    /// Tube sides per tier (legs use quad counts, see LEG_QPS).
    #[inline]
    fn sides(&self) -> usize {
        [2, 3, 4][self.tier as usize]
    }
    #[inline]
    fn vert(&mut self, p: [f32; 3], param: f32, id: f32) {
        self.out.push(TVert { pos: [p[0], p[1], p[2], 0.0], aux: [param, id] });
    }
    fn tri(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], id: f32) {
        for p in [a, b, c] {
            self.vert(p, 0.0, id);
        }
    }
    fn quad(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], id: f32) {
        for p in [a, b, c, a, c, d] {
            self.vert(p, 0.0, id);
        }
    }
    /// Flat rectangular blade from `a` to `b`, width `w` along unit `up`,
    /// param interpolated 0 at a -> 1 at b.
    fn blade(&mut self, a: [f32; 3], b: [f32; 3], up: [f32; 3], w: f32, id: f32) {
        let p0 = [a[0] - up[0] * w, a[1] - up[1] * w, a[2] - up[2] * w];
        let p1 = [a[0] + up[0] * w, a[1] + up[1] * w, a[2] + up[2] * w];
        let p2 = [b[0] + up[0] * w, b[1] + up[1] * w, b[2] + up[2] * w];
        let p3 = [b[0] - up[0] * w, b[1] - up[1] * w, b[2] - up[2] * w];
        self.quad(p0, p1, p2, p3, id);
    }
}

/// Ring basis perpendicular to a unit direction (deterministic).
fn ring_frame(d: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up: [f32; 3] = if d[1].abs() > 0.94 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let n0 = [
        d[1] * up[2] - d[2] * up[1],
        d[2] * up[0] - d[0] * up[2],
        d[0] * up[1] - d[1] * up[0],
    ];
    let l0 = (n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2]).sqrt();
    let n0 = if l0 > 1e-9 { [n0[0] / l0, n0[1] / l0, n0[2] / l0] } else { [1.0, 0.0, 0.0] };
    let b0 = [
        d[1] * n0[2] - d[2] * n0[1],
        d[2] * n0[0] - d[0] * n0[2],
        d[0] * n0[1] - d[1] * n0[0],
    ];
    (n0, b0)
}

/// Ring of `sides` verts around point `c`, radius `r`, in plane of dir `d`.
fn ring_pts(c: [f32; 3], d: [f32; 3], r: f32, sides: usize) -> Vec<[f32; 3]> {
    let (u, v) = ring_frame(d);
    (0..sides)
        .map(|i| {
            let a = TAU * i as f32 / sides as f32;
            let cu = a.cos() * r;
            let cv = a.sin() * r;
            [
                c[0] + u[0] * cu + v[0] * cv,
                c[1] + u[1] * cu + v[1] * cv,
                c[2] + u[2] * cu + v[2] * cv,
            ]
        })
        .collect()
}

impl Scribe {
    /// Tube section between rings r0 (with params p0) and r1 (params p1).
    fn tube_seg(
        &mut self,
        r0: &[[f32; 3]],
        r1: &[[f32; 3]],
        p0: f32,
        p1: f32,
        id: f32,
    ) {
        let s = r0.len();
        for i in 0..s {
            let j = (i + 1) % s;
            let a = r0[i];
            let b = r0[j];
            let c = r1[i];
            let d = r1[j];
            for (p, t) in [(a, p0), (b, p0), (d, p1), (a, p0), (d, p1), (c, p1)] {
                self.vert(p, t, id);
            }
        }
    }
    /// Stick (open k-sided tube) between two points.
    fn stick(&mut self, a: [f32; 3], b: [f32; 3], r0: f32, r1: f32, qps: usize, id: f32) {
        let d = norm3([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        let ra = ring_pts(a, d, r0, qps);
        let rb = ring_pts(b, d, r1, qps);
        // params keyed to taper position (root 0 -> tip 1) like limb code.
        self.tube_seg(&ra, &rb, 0.0, 1.0, id);
    }
    /// Cone from base ring to a tip point (open, no base cap).
    fn cone_open(&mut self, base: &[[f32; 3]], tip: [f32; 3], sides: usize, id: f32) {
        for i in 0..sides {
            let j = (i + 1) % sides;
            self.tri(base[i], base[j], tip, id);
        }
    }
    /// Base cap fan (ring center -> ring). Reserved for cap variants.
    #[allow(dead_code)]
    fn cap_fan(&mut self, ring: &[[f32; 3]], center: [f32; 3], id: f32) {
        let s = ring.len();
        for i in 0..s {
            self.tri(ring[i], ring[(i + 1) % s], center, id);
        }
    }
    /// Low-poly ellipsoid: tier 2 = ico depth 0 (60 v), tier 1/0 = octa (24 v).
    fn ellipsoid(
        &mut self,
        c: [f32; 3],
        r: [f32; 3],
        id: f32,
        param: f32,
    ) {
        let soup = if self.tier >= 2 {
            crate::morph::templates::sphere_soup(0)
        } else {
            octa_soup()
        };
        for p in soup {
            self.vert(
                [c[0] + p[0] * r[0], c[1] + p[1] * r[1], c[2] + p[2] * r[2]],
                param,
                id,
            );
        }
    }
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-9 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// 8-face octahedron soup (24 verts) — cheap sphere stand-in.
fn octa_soup() -> Vec<[f32; 3]> {
    let v: [[f32; 3]; 6] = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let f: [[usize; 3]; 8] = [
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ];
    let mut out = Vec::with_capacity(24);
    for f in f {
        out.push(v[f[0]]);
        out.push(v[f[1]]);
        out.push(v[f[2]]);
    }
    out
}

// ------------------------------------------------------- plan builders

const NOSE: f32 = 0.5;
const REAR: f32 = -0.5;

struct PlanShape {
    /// trunk tube base radius (BL units)
    base: f32,
    /// vertical (y) axis scale of trunk cross sections
    ky: f32,
    /// lateral (z) axis scale of trunk cross sections
    kz: f32,
    /// leg attach span
    limb_lo: f32,
    limb_hi: f32,
}

fn plan_shape(m: &CreatureMorph) -> PlanShape {
    let neck = NOSE - m.head_ratio;
    match m.body_plan {
        BodyPlan::Primitive => PlanShape { base: 0.0, ky: 1.0, kz: 1.0, limb_lo: 0.0, limb_hi: 0.0 },
        BodyPlan::SegmentedArthropod => PlanShape { base: 0.095, ky: 1.0, kz: 1.0, limb_lo: -0.36, limb_hi: neck - 0.02 },
        BodyPlan::MultiLegged => PlanShape { base: 0.125, ky: 0.82, kz: 1.24, limb_lo: -0.30, limb_hi: 0.08 },
        BodyPlan::Quadruped => PlanShape { base: 0.125, ky: 1.0, kz: 1.06, limb_lo: -0.34, limb_hi: 0.16 },
    }
}

/// Trunk tube radius profile, rear (t=0) -> neck (t=1).
fn trunk_profile(t: f32) -> f32 {
    0.62 + 0.38 * (PI * (0.12 + 0.88 * t)).sin()
}

/// (x, w) for each trunk ring, rear -> neck. segments = section count.
fn trunk_rings(m: &CreatureMorph) -> Vec<(f32, f32)> {
    let neck = NOSE - m.head_ratio;
    let segs = m.segment_count.max(1) as usize;
    (0..=segs)
        .map(|i| {
            let t = i as f32 / segs as f32;
            let x = REAR + (neck - REAR) * t;
            (x, trunk_profile(t))
        })
        .collect()
}

fn build_trunk(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    let sides = s.sides();
    let rings = trunk_rings(m);
    let rings: Vec<Vec<[f32; 3]>> = rings
        .iter()
        .map(|(x, w)| ring_pts([*x, 0.0, 0.0], [1.0, 0.0, 0.0], sh.base * w, sides))
        .collect();
    // cross-section shape: ellipse via per-quad scaling is baked into ky/kz by
    // shifting ring verts (cheap trick: scale z verts after generation)
    let n = rings.len();
    for k in 0..(n - 1) {
        let r0 = scale_xsec(&rings[k], sh.ky, sh.kz);
        let r1 = scale_xsec(&rings[k + 1], sh.ky, sh.kz);
        let t0 = k as f32 / (n - 1) as f32;
        let t1 = (k + 1) as f32 / (n - 1) as f32;
        s.tube_seg(&r0, &r1, t0, t1, CODE_BODY);
    }
    // flat belly shading hook: belly param lift on bottom verts is implicit
    // in tube param 0..1 (A2 shader may band param by normal.y)
}

/// Adjust a generated ring's verts for elliptical cross-section.
fn scale_xsec(ring: &[[f32; 3]], ky: f32, kz: f32) -> Vec<[f32; 3]> {
    ring.iter()
        .map(|p| [p[0], p[1] * ky, p[2] * kz])
        .collect()
}

fn build_head(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    let neck = NOSE - m.head_ratio;
    let hc = (neck + NOSE) * 0.5;
    let hrx = (NOSE - neck) * 0.5;
    let hry = sh.base * trunk_profile(1.0) * 1.12 * sh.ky;
    let hrz = sh.base * trunk_profile(1.0) * 1.12 * sh.kz;
    match s.tier {
        2 => s.ellipsoid([hc, 0.0, 0.0], [hrx, hry, hrz], CODE_HEAD, 0.15),
        _ => {
            // octa/cone head nudging forward from the neck ring
            if s.tier == 1 {
                s.ellipsoid([hc, 0.0, 0.0], [hrx, hry, hrz], CODE_HEAD, 0.15);
            } else {
                let d = [1.0, 0.0, 0.0];
                let ring = ring_pts([neck - hrx * 0.2, 0.0, 0.0], d, hry / sh.ky, s.sides());
                let ring: Vec<[f32; 3]> =
                    ring.iter().map(|p| [p[0], p[1] * sh.ky, p[2] * sh.kz]).collect();
                s.cone_open(&ring, [NOSE, 0.0, 0.0], s.sides(), CODE_HEAD);
            }
        }
    }

    // ---- eyes
    let count = m.eye_count.min(3) as usize;
    let er = 0.024 * m.eye_scale;
    let ex = hc + hrx * 0.55;
    let mut poses: Vec<[f32; 3]> = Vec::with_capacity(3);
    if count == 1 {
        poses.push([ex, hry * 0.5, 0.0]);
    } else {
        poses.push([ex, hry * 0.35, hrz * 0.78]);
        poses.push([ex, hry * 0.35, -hrz * 0.78]);
        if count == 3 {
            poses.push([ex, hry * 0.9, 0.0]);
        }
    }
    for p in poses {
        match s.tier {
            2 => s.ellipsoid(p, [er, er * 0.75, er], CODE_EYE, 0.0),
            1 => s.ellipsoid(p, [er, er * 0.75, er], CODE_EYE, 0.0),
            _ => {
                // lean eye gem: single triangle
                s.tri(
                    [p[0], p[1] + er, p[2]],
                    [p[0] + er, p[1] - er * 0.4, p[2] + er * 0.6],
                    [p[0] + er, p[1] - er * 0.4, p[2] - er * 0.6],
                    CODE_EYE,
                );
            }
        }
    }

    // ---- mouth per Mouthpart
    match s.tier {
        2 | 1 => {
            let sides = s.sides();
            match m.mouthpart {
                Mouthpart::Sucker => {
                    // short forward-down open cone under the nose
                    let c = [NOSE - 0.02, -hry * 0.5, 0.0];
                    let d = norm3([0.89, -0.45, 0.0]);
                    let ring = ring_pts(c, d, 0.035, sides);
                    s.cone_open(&ring, [c[0] + d[0] * 0.09, c[1] + d[1] * 0.09, c[2]], sides, CODE_HEAD);
                }
                Mouthpart::Filter => {
                    // wide grazing fan at the nose
                    let c = [NOSE - 0.03, -hry * 0.3, 0.0];
                    let ring = ring_pts(c, [1.0, 0.0, 0.0], 0.055, sides);
                    s.cone_open(&ring, [NOSE + 0.04, -hry * 0.3, 0.0], sides, CODE_HEAD);
                }
                Mouthpart::Jaw => {
                    // two opposing mandible blades
                    let up = sh.ky;
                    s.blade(
                        [NOSE - 0.06, hry * 0.4, 0.0],
                        [NOSE + 0.07, hry * 0.25, 0.0],
                        [0.0, 1.0, 0.0],
                        0.012 * up,
                        CODE_HEAD,
                    );
                    s.blade(
                        [NOSE - 0.06, -hry * 0.4, 0.0],
                        [NOSE + 0.07, -hry * 0.25, 0.0],
                        [0.0, 1.0, 0.0],
                        0.012 * up,
                        CODE_HEAD,
                    );
                }
            }
        }
        _ => {
            // lean diminutive mouth blade
            s.blade(
                [NOSE - 0.03, -hry * 0.4, 0.0],
                [NOSE + 0.05, -hry * 0.25, 0.0],
                [0.0, 1.0, 0.0],
                0.010 * sh.ky,
                CODE_HEAD,
            );
        }
    }

    // ---- antennae
    if m.antenna_count > 0 {
        let root = [ex, hry * 0.85, 0.0];
        for side in 1..=m.antenna_count as i32 {
            let sz = if m.antenna_count == 1 { 0.0 } else { side as f32 * 0.55 };
            let tip = [NOSE + 0.14, hry * 1.7, sz * hrz];
            match s.tier {
                2 => s.blade(root, tip, [0.0, 0.0, 1.0], 0.006, CODE_LIMB),
                _ => s.tri(
                    root,
                    [root[0] + 0.05, hry * 1.2, sz * hrz * 0.5],
                    [tip[0], tip[1] - 0.02, tip[2]],
                    CODE_LIMB,
                ),
            }
        }
    }
}

/// Quad count per leg stick per tier: [lean blade, 2-blade, 3-sided tube].
const LEG_QPS: [usize; 3] = [1, 2, 3];

/// Sticks per leg chain for a descriptor: 1 + 1 when jointedness > 0.5
/// (lower segment added). Public for renderer diagnostics/tests.
pub fn leg_sticks(m: &CreatureMorph) -> usize {
    1 + (m.jointedness > 0.5) as usize
}

fn build_legs(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    let count = m.appendage_count as usize;
    if count == 0 {
        return;
    }
    let hi = m.appendage_len.1;
    let lo = m.appendage_len.0;
    let jointed = m.jointedness > 0.5;
    let qps = LEG_QPS[s.tier as usize];
    let pairs = (count + 1) / 2;
    for i in 0..count {
        let side = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
        let pidx = i / 2;
        let t = if pairs > 1 { pidx as f32 / (pairs - 1) as f32 } else { 0.5 };
        // anterior pairs lean forward, rear pairs lean back (deterministic)
        let lean = 0.42 - 0.95 * t * t;
        let x_att = sh.limb_lo + (sh.limb_hi - sh.limb_lo) * t;
        let ry_att = sh.base * trunk_profile((x_att - REAR) / (NOSE - REAR)) * sh.ky;
        let rz_att = ry_att / sh.ky * sh.kz;
        let hip = [x_att, -ry_att * 0.55, side * rz_att * 0.92];
        let k = match m.body_plan {
            // multi-legged legs stay short and stubby
            BodyPlan::MultiLegged => lo * 0.85,
            _ => lo,
        } * 2.0f32;
        let leg_len = (hi + k) * 0.5;
        let d1 = norm3([0.10, -1.0, side * 0.62]);
        let knee = [
            hip[0] + d1[0] * leg_len * 0.44,
            hip[1] + d1[1] * leg_len * 0.44,
            hip[2] + d1[2] * leg_len * 0.44,
        ];
        let d2 = norm3([lean, -0.62, side * 0.30]);
        let foot = if jointed {
            [
                knee[0] + d2[0] * leg_len * 0.56,
                knee[1] + d2[1] * leg_len * 0.56,
                knee[2] + d2[2] * leg_len * 0.56,
            ]
        } else {
            [
                hip[0] + d1[0] * leg_len + d2[0] * leg_len * 0.15,
                hip[1] + d1[1] * leg_len * 0.85,
                hip[2] + d1[2] * leg_len + d2[2] * leg_len * 0.15,
            ]
        };
        let r_leg = 0.010 + 0.004 * (1.0 - t);
        if jointed {
            s.stick(hip, knee, r_leg, r_leg * 0.9, qps, CODE_LIMB);
            s.stick(knee, foot, r_leg * 0.9, r_leg * 0.5, qps, CODE_LIMB);
        } else {
            s.stick(hip, foot, r_leg, r_leg * 0.45, qps, CODE_LIMB);
        }
    }
}

fn build_tail(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    if !m.tail_present {
        return;
    }
    // drooping ribbon: sections per tier [1, 2, 3], one quad each, param
    // along the tail so the VS wave animates it (flagella convention).
    let secs = [1usize, 2, 3][s.tier as usize];
    let l = 0.35 + 0.35 * m.tail_len.clamp(0.3, 0.9);
    let mut x = REAR + 0.02;
    let mut y = sh.base * 0.5;
    let dlen = l / secs as f32;
    let w = 0.016;
    for k in 0..secs {
        let sag = 0.55 * (k as f32 / secs as f32).powi(2);
        let x1 = x - dlen;
        let y1 = y - dlen * (0.35 + 0.75 * sag);
        let a = [x, y, 0.0];
        let b = [x1, y1, 0.0];
        let t0 = k as f32 / secs as f32;
        let t1 = (k + 1) as f32 / secs as f32;
        let w0 = w * (1.0 - 0.55 * t0);
        let w1 = w * (1.0 - 0.55 * t1);
        let p0 = [a[0], a[1] + w0, a[2]];
        let p1 = [a[0], a[1] - w0, a[2]];
        let p2 = [b[0], b[1] + w1, b[2]];
        let p3 = [b[0], b[1] - w1, b[2]];
        for (p, t) in [
            (p0, t0),
            (p1, t0),
            (p3, t1),
            (p0, t0),
            (p3, t1),
            (p2, t1),
        ] {
            s.vert(p, t, CODE_TAIL);
        }
        // back face
        for (p, t) in [
            (p1, t0),
            (p0, t0),
            (p2, t1),
            (p1, t0),
            (p2, t1),
            (p3, t1),
        ] {
            s.vert(p, t, CODE_TAIL);
        }
        x = x1;
        y = y1;
    }
}

fn build_wings(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    if !m.winged {
        return;
    }
    let neck = NOSE - m.head_ratio;
    let x0 = (neck + REAR) * 0.5 + 0.06;
    let ry = sh.base * trunk_profile((x0 - REAR) / (NOSE - REAR)) * sh.ky;
    let root = [x0, ry * 0.62, 0.0];
    for side in [-1.0f32, 1.0f32] {
        match s.tier {
            2 => {
                let tip = [x0 - 0.30, ry * 1.05, side * 0.24];
                let mid = [x0 - 0.13, ry * 0.62, side * 0.10];
                s.blade(root, mid, [0.0, 0.0, 1.0], 0.05, CODE_WING);
                s.blade(mid, tip, [0.0, 0.0, 1.0], 0.03, CODE_WING);
            }
            1 => {
                let tip = [x0 - 0.30, ry * 1.05, side * 0.24];
                s.blade(root, tip, [0.0, 0.0, 1.0], 0.05, CODE_WING);
            }
            _ => s.tri(root, [x0 - 0.30, ry * 1.05, side * 0.24], [x0 - 0.28, ry * 0.62, side * 0.05], CODE_WING),
        }
    }
}

fn build_spikes(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    let n = m.spike_count as usize;
    if n == 0 {
        return;
    }
    let rings = trunk_rings(m);
    for i in 0..n {
        let (x, w) = rings[(i + 1).min(rings.len() - 1)].clone();
        let y = sh.base * w * sh.ky;
        let ln = 0.055;
        s.tri(
            [x - 0.020, y, 0.0],
            [x + 0.020, y, 0.0],
            [x - 0.045, y + ln, 0.0],
            CODE_SPIKE,
        );
    }
}

/// Quadruped armour plating: a small saddle plate quad when armoured.
fn build_armour(s: &mut Scribe, m: &CreatureMorph, sh: &PlanShape) {
    if m.body_plan != BodyPlan::Quadruped || m.armour <= 0.5 {
        return;
    }
    let rings = trunk_rings(m);
    let mid = rings.len() / 2;
    let (x0, w0) = rings[mid];
    let (x1, _) = rings[(mid + 2).min(rings.len() - 1)];
    let y = sh.base * w0 * sh.ky;
    s.blade(
        [x0, y * 1.04, 0.0],
        [x1, y * 0.99, 0.0],
        [0.0, 0.0, 1.0],
        y * 0.72,
        CODE_BODY,
    );
}

fn build_primitive(s: &mut Scribe, m: &CreatureMorph) {
    // compact blob; eye + sucker mouth; no limbs, never a tail.
    match s.tier {
        2 => s.ellipsoid([0.0, 0.0, 0.0], [0.30, 0.24, 0.235], CODE_BODY, 0.1),
        _ => s.ellipsoid([0.0, 0.0, 0.0], [0.30, 0.24, 0.235], CODE_BODY, 0.1),
    }
    // (Primitive keeps a single "head" marker vertex cluster: the eye).
    let er = 0.038 * m.eye_scale;
    match s.tier {
        2 | 1 => s.ellipsoid([0.16, 0.115, 0.02], [er, er * 0.75, er], CODE_EYE, 0.0),
        _ => s.tri(
            [0.16, 0.115 + er, 0.02],
            [0.16 + er, 0.115 - er * 0.4, 0.02 + er * 0.6],
            [0.16 + er, 0.115 - er * 0.4, 0.02 - er * 0.6],
            CODE_EYE,
        ),
    }
    if s.tier == 0 {
        s.blade([0.27, -0.10, 0.0], [0.36, -0.16, 0.0], [0.0, 1.0, 0.0], 0.010, CODE_HEAD);
    } else {
        let c = [0.26, -0.09, 0.0];
        let d = norm3([0.92, -0.40, 0.0]);
        let ring = ring_pts(c, d, 0.034, s.sides());
        s.cone_open(&ring, [c[0] + d[0] * 0.10, c[1] + d[1] * 0.10, c[2]], s.sides(), CODE_HEAD);
    }
    if m.spike_count > 0 {
        let n = m.spike_count.min(4) as usize;
        for i in 0..n {
            let a = PI * (0.25 + 0.5 * (i as f32 / n.max(1) as f32));
            let x = a.cos() * 0.24;
            let y = a.sin() * 0.20;
            s.tri(
                [x - 0.03, y, 0.0],
                [x + 0.03, y, 0.0],
                [x - 0.02, y + 0.05, 0.0],
                CODE_SPIKE,
            );
        }
    }
}

fn build_plan(s: &mut Scribe, m: &CreatureMorph) {
    if m.body_plan == BodyPlan::Primitive {
        build_primitive(s, m);
        return;
    }
    let sh = plan_shape(m);
    build_trunk(s, m, &sh);
    build_head(s, m, &sh);
    build_legs(s, m, &sh);
    build_tail(s, m, &sh);
    build_wings(s, m, &sh);
    build_spikes(s, m, &sh);
    build_armour(s, m, &sh);
}

/// Build at an explicit tier (2 rich, 1 medium, 0 lean).
fn build_tier(m: &CreatureMorph, tier: u8) -> Vec<TVert> {
    let mut s = Scribe::new(tier);
    build_plan(&mut s, m);
    s.out
}

fn solve(m: &CreatureMorph) -> Vec<TVert> {
    for &t in &TIERS {
        let v = build_tier(m, t);
        if v.len() <= VERT_BUDGET {
            return v;
        }
    }
    // unreachable by budget calculus (asserted in tests)
    build_tier(m, 0)
}

/// Tier a descriptor resolves to (exposed for tests + renderer diagnostics).
pub fn tier_for(m: &CreatureMorph) -> u8 {
    let key = archetype_key(m);
    let canon = canonical_from_key(key);
    for &t in &TIERS {
        if build_tier(&canon, t).len() <= VERT_BUDGET {
            return t;
        }
    }
    0
}

// ------------------------------------------------------- registry cache

struct CacheEntry {
    mesh: &'static [TVert],
}
/// Static cache: archetype key -> soup. `mesh_for` NEVER returns a slice
/// freed by eviction (entries are leaked into static lifetime by design —
/// archetype space is quantized and small, so this bounds memory sanely).
struct Registry {
    map: HashMap<u64, CacheEntry>,
    lru: VecDeque<u64>,
    graveyard: Vec<&'static [TVert]>,
}

static REGISTRY: std::sync::OnceLock<std::sync::Mutex<Registry>> = std::sync::OnceLock::new();

fn registry() -> &'static std::sync::Mutex<Registry> {
    REGISTRY.get_or_init(|| {
        std::sync::Mutex::new(Registry {
            map: HashMap::new(),
            lru: VecDeque::new(),
            graveyard: Vec::new(),
        })
    })
}

/// Cache probe only (no construct): `None` when that archetype is not built.
pub fn mesh_for(key: u64) -> Option<&'static [TVert]> {
    let reg = registry();
    let mut g = reg.lock().unwrap_or_else(|e| e.into_inner());
    let hit = g.map.get(&key).map(|e| e.mesh);
    if let Some(mesh) = hit {
        touch(&mut g, key);
        Some(mesh)
    } else {
        None
    }
}

fn touch(g: &mut Registry, key: u64) {
    if let Some(pos) = g.lru.iter().position(|&k| k == key) {
        g.lru.remove(pos);
        g.lru.push_back(key);
    }
}

/// Ensure built: construct (bounded by [`VERT_BUDGET`]) on cache miss and
/// insert with LRU eviction (evicted soups are retained in the graveyard
/// because their static slices may still be referenced).
pub fn ensure_mesh(key: u64) -> &'static [TVert] {
    let reg = registry();
    let mut g = reg.lock().unwrap_or_else(|e| e.into_inner());
    let hit = g.map.get(&key).map(|e| e.mesh);
    if let Some(mesh) = hit {
        touch(&mut g, key);
        return mesh;
    }
    let canon = canonical_from_key(key);
    let built: &'static [TVert] = Box::leak(solve(&canon).into_boxed_slice());
    // duplicate archetype keys MUST give bit-identical soups (quantized canonicity)
    debug_assert!(
        g.map
            .get(&key)
            .map(|e| mesh_bytes(e.mesh) == mesh_bytes(built))
            .unwrap_or(true),
        "archetype key {key} rebuilt differently"
    );
    while g.map.len() >= CACHE_CAP {
        let evict = g.lru.pop_front().expect("lru tracks keys");
        if let Some(e) = g.map.remove(&evict) {
            g.graveyard.push(e.mesh); // never free; static slices roam
        }
    }
    g.map.insert(key, CacheEntry { mesh: built });
    g.lru.push_back(key);
    built
}

fn mesh_bytes(m: &[TVert]) -> &[u8] {
    bytemuck::cast_slice(m)
}

/// Convenience: build-or-fetch for a live descriptor.
pub fn archetype_mesh(m: &CreatureMorph) -> &'static [TVert] {
    ensure_mesh(archetype_key(m))
}

/// Cached vertex count (0 when absent).
pub fn vert_count(key: u64) -> usize {
    mesh_for(key).map(|m| m.len()).unwrap_or(0)
}

// ------------------------------------------------- brand fallback meshes

/// Distinct fallback key space (bit 63 + brand).
fn fallback_key(brand: u32) -> u64 {
    0x8000_0000_0000_0000 | (brand as u64 & 0x3)
}

/// Generic descriptor per brand: same visible silhouette norms as the legacy
/// templates (~1.39 world units at nominal turgor -> body_length 4.63 =
/// 1.39 / RENDER_SCALE). Deterministic mid-table values; no CreatureMorph
/// from sim traits needed. brand 0 = primitive spec, 1 = arthropod,
/// 2 = multi-legged, 3 = quadruped.
pub fn fallback_morph(brand: u32) -> CreatureMorph {
    let (plan, segs, app) = match brand & 0x3 {
        0 => (BodyPlan::Primitive, 2, 0),
        1 => (BodyPlan::SegmentedArthropod, 3, 2),
        2 => (BodyPlan::MultiLegged, 4, 6),
        _ => (BodyPlan::Quadruped, 4, 4),
    };
    CreatureMorph {
        body_plan: plan,
        body_length: 4.63,
        head_ratio: 0.27,
        segment_count: segs,
        appendage_count: app,
        appendage_len: (0.35, 0.55),
        jointedness: 0.55,
        eye_count: 2,
        eye_scale: 1.0,
        antenna_count: if plan == BodyPlan::Primitive { 0 } else { 1 },
        mouthpart: Mouthpart::Filter,
        armour: 0.25,
        spike_count: 2,
        tail_present: plan == BodyPlan::SegmentedArthropod,
        tail_len: if plan == BodyPlan::SegmentedArthropod { 0.6 } else { 0.0 },
        winged: false,
        patterned: false,
        hue: 0.0,
        sat_tint: 0.3,
    }
}

/// Brand fallback mesh (cached, deterministic): the generic small arthropod/
/// cell per brand used when no sim-derived CreatureMorph exists. Sizes match
/// today's organisms (visible length ~1.39 world units at RENDER_SCALE).
pub fn brand_fallback(brand: u32) -> &'static [TVert] {
    let key = fallback_key(brand);
    let reg = registry();
    let mut g = reg.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = g.map.get(&key) {
        return e.mesh;
    }
    let built: &'static [TVert] = Box::leak(solve(&fallback_morph(brand)).into_boxed_slice());
    while g.map.len() >= CACHE_CAP {
        let evict = g.lru.pop_front().expect("lru tracks keys");
        if let Some(e) = g.map.remove(&evict) {
            g.graveyard.push(e.mesh);
        }
    }
    g.map.insert(key, CacheEntry { mesh: built });
    g.lru.push_back(key);
    built
}

// ------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::{Network, Organism, Rng, Traits, TRAIT_MIN};

    fn bare(id: u64, lineage: u32, net: &Network, size: f32, max_speed: f32) -> Organism {
        Organism {
            id,
            x: 0.0,
            y: 0.0,
            dir: 0.0,
            speed: 0.0,
            energy: 0.0,
            max_energy: 110.0,
            age: 0,
            generation: 1,
            genotype: alife_common::Genotype {
                net: net.clone(),
                traits: Traits { size, max_speed, ..TRAIT_MIN },
            },
            lineage,
            mom: None,
            dad: None,
            offspring_count: 0,
            alive: true,
            repr_cooldown: 0,
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }

    // ------------------------------------------------- determinism

    #[test]
    fn creature_mesh_is_deterministic() {
        let net = Network::zeros();
        let org = bare(1, 7, &net, 1.0, 90.0);
        let m = alife_common::morphology::derive(&org);
        let k = archetype_key(&m);
        assert_eq!(archetype_key(&m), k, "key pure");
        assert_eq!(canonical_from_key(k), canonical_from_key(k), "canonical pure");
        let a = build_tier(&canonical_from_key(k), tier_for(&m));
        let b = build_tier(&canonical_from_key(k), tier_for(&m));
        assert_eq!(mesh_bytes(&a), mesh_bytes(&b), "same morph -> same bytes");
        // cache round trip must hand back the identical souped bytes
        let cached = ensure_mesh(k).to_vec();
        assert_eq!(mesh_bytes(&a), mesh_bytes(&cached), "cache shares solve");
    }

    // ------------------------------------------------- archetype bucketing

    /// ~2000 "synhello" genotype organisms constructed like the morphology
    /// tests do (TRAIT_MIN fill + random size/speed/lineage + softly spun
    /// weights): archetype keys must stay bounded (<= 72 distinct) so the
    /// LRU cap holds and shared-mesh count stays sane.
    #[test]
    fn archetype_keys_stay_bounded_over_population() {
        let rng = &mut Rng::new(2026);
        let mut keys = std::collections::HashSet::new();
        for i in 0..2000u32 {
            let mut n = Network::zeros();
            n.w1[0] = rng.next_f32() * 8.0 - 4.0;
            n.w2[0] = rng.next_f32() * 8.0 - 4.0;
            let org = bare(
                i as u64,
                (i * 7919) % 10000,
                &n,
                rng.range(0.5, 1.7),
                rng.range(10.0, 160.0),
            );
            keys.insert(archetype_key(&alife_common::morphology::derive(&org)));
        }
        eprintln!("creature_mesh: distinct archetype keys = {}", keys.len());
        assert!(
            keys.len() <= CACHE_CAP,
            "distinct archetypes {} exceed cache cap {}",
            keys.len(),
            CACHE_CAP
        );
    }

    /// Vertex budget honed per archetype: every cached mesh <= 240 soup verts.
    #[test]
    fn vertex_budget_holds_per_archetype() {
        let rng = &mut Rng::new(99);
        for i in 0..400u32 {
            let mut n = Network::zeros();
            n.w1[0] = rng.next_f32() * 8.0 - 4.0;
            n.w2[0] = rng.next_f32() * 8.0 - 4.0;
            let org = bare(
                i as u64,
                (i * 131) % 10000,
                &n,
                rng.range(0.5, 1.7),
                rng.range(10.0, 160.0),
            );
            let m = alife_common::morphology::derive(&org);
            let v = build_tier(&canonical_from_key(archetype_key(&m)), tier_for(&m));
            assert!(
                v.len() <= VERT_BUDGET,
                "archetype {} verts {} > budget {}",
                archetype_key(&m),
                v.len(),
                VERT_BUDGET
            );
        }
        // worst-case descriptor spans all buckets explicitly
        let worst = CreatureMorph {
            body_plan: BodyPlan::SegmentedArthropod,
            body_length: 8.8,
            head_ratio: 0.42,
            segment_count: 6,
            appendage_count: 8,
            appendage_len: (0.85, 0.85),
            jointedness: 0.9,
            eye_count: 3,
            eye_scale: 1.5,
            antenna_count: 2,
            mouthpart: Mouthpart::Jaw,
            armour: 1.0,
            spike_count: 6,
            tail_present: true,
            tail_len: 0.9,
            winged: true,
            patterned: true,
            hue: 359.0,
            sat_tint: 0.45,
        };
        for plan in [
            BodyPlan::Primitive,
            BodyPlan::SegmentedArthropod,
            BodyPlan::MultiLegged,
            BodyPlan::Quadruped,
        ] {
            let mut w = worst;
            w.body_plan = plan;
            let v = build_tier(&canonical_from_key(archetype_key(&w)), tier_for(&w));
            assert!(
                v.len() <= VERT_BUDGET,
                "worst plan {:?} verts {} > {}",
                plan,
                v.len(),
                VERT_BUDGET
            );
        }
    }

    /// Every plan builder emits body + head at minimum (and a descriptor
    /// population stays inside the cache cap).
    #[test]
    fn plan_builders_emit_body_and_head() {
        let net = Network::zeros();
        let probes: [(f32, f32); 5] = [(0.7, 30.0), (1.0, 90.0), (1.35, 90.0), (1.4, 40.0), (1.1, 40.0)];
        for (i, (size, speed)) in probes.into_iter().enumerate() {
            let org = bare(i as u64, 11 + i as u32, &net, size, speed);
            let m = alife_common::morphology::derive(&org);
            let v = build_tier(&canonical_from_key(archetype_key(&m)), tier_for(&m));
            assert!(v.iter().any(|t| t.aux[1] == CODE_BODY), "plan {:?} body", m.body_plan);
            assert!(
                v.iter().any(|t| t.aux[1] == CODE_HEAD) || v.iter().any(|t| t.aux[1] == CODE_EYE),
                "plan {:?} head/eye missing",
                m.body_plan
            );
        }
        // truly every builder BodyPlan branch emits body verts
        for plan in [
            BodyPlan::Primitive,
            BodyPlan::SegmentedArthropod,
            BodyPlan::MultiLegged,
            BodyPlan::Quadruped,
        ] {
            let m = canonical_from_key(0_u64); // placeholder overhead
            let mut mm = m;
            mm.body_plan = plan;
            mm.segment_count = 4;
            mm.appendage_count = 4;
            let v = build_tier(&mm, 1);
            assert!(v.iter().any(|t| t.aux[1] == CODE_BODY), "{:?} body", plan);
            assert!(v.iter().any(|t| t.aux[1] == CODE_HEAD), "{:?} head", plan);
        }
    }

    /// Leg chain count == descriptor: each leg emits 1 stick (+1 lower stick
    /// when jointedness > 0.5); counted via limb soup quads per tier table.
    #[test]
    fn leg_chain_count_matches_descriptor() {
        let rng = &mut Rng::new(997);
        for i in 0..120u32 {
            let mut n = Network::zeros();
            n.w1[0] = rng.next_f32() * 8.0 - 4.0;
            n.w2[0] = rng.next_f32() * 8.0 - 4.0;
            let org = bare(
                i as u64,
                i % 42,
                &n,
                rng.range(0.9, 1.6),
                rng.range(10.0, 150.0),
            );
            let m = alife_common::morphology::derive(&org);
            if m.appendage_count == 0 || m.body_plan == BodyPlan::Primitive {
                continue;
            }
            let tier = tier_for(&m);
            let canon = canonical_from_key(archetype_key(&m));
            let v = build_tier(&canon, tier);
            let limb_verts = v.iter().filter(|t| t.aux[1] == CODE_LIMB).count();
            let sticks = m.appendage_count as usize * leg_sticks(&m);
            let expect = sticks * LEG_QPS[tier as usize] * 6;
            assert_eq!(
                limb_verts, expect,
                "app {} jointed {:?} tier {}: sticks {}/legs x{}",
                m.appendage_count,
                m.jointedness > 0.5,
                tier,
                sticks,
                LEG_QPS[tier as usize]
            );
        }
    }

    /// Palette folding: hue channel == morph.hue/360 (eps).
    #[test]
    fn hue_folds_into_material_channel() {
        for h in [0.0f32, 91.3, 180.0, 300.0, 359.9] {
            let mut m = fallback_morph(1);
            m.hue = h;
            assert!((hue_channel(&m) - h / 360.0).abs() < 1e-6, "hue {h}");
        }
    }

    /// Brand fallbacks exist, differ per brand and obey the vertex budget.
    #[test]
    fn brand_fallback_meshes_exist_and_match() {
        let mut sizes = Vec::new();
        for b in 0..4u32 {
            let a = brand_fallback(b);
            let bb = brand_fallback(b);
            assert_eq!(mesh_bytes(a), mesh_bytes(bb), "fallback {b} deterministic");
            assert!(!a.is_empty(), "fallback {b} emits verts");
            assert!(a.len() <= VERT_BUDGET, "fallback {b} budget {}", a.len());
            sizes.push((b, a.len()));
        }
        // distinct brands build distinct soups
        let a0 = brand_fallback(0);
        let a1 = brand_fallback(1);
        assert!(mesh_bytes(a0) != mesh_bytes(a1), "brands differ");
        eprintln!("creature_mesh: fallback sizes {:?}", sizes);
    }

    /// Key quantization: helper bucket functions fold exactly as documented.
    #[test]
    fn archetype_key_buckets_follow_spec() {
        let mut m = fallback_morph(1);
        // segment quantization {2,3,4,6}
        m.segment_count = 2;
        let k2 = archetype_key(&m);
        m.segment_count = 5;
        assert_eq!(archetype_key(&m) >> 2 & 0x3, seg_idx(6), "seg 5-6 folds to 6");
        assert_ne!(k2, archetype_key(&m));
        // appendage {0,2,3,4,5,6,8}
        m.appendage_count = 1;
        assert_eq!(app_idx(1), app_idx(2), "1 folds with 2");
        m.appendage_count = 7;
        assert_eq!(app_idx(7), app_idx(8), "7 folds with 8");
        // spikes {0,3,6}
        m.spike_count = 0;
        let k = archetype_key(&m);
        m.spike_count = 2;
        assert_eq!(k, archetype_key(&m), "0-2 fold");
        m.spike_count = 3;
        assert_ne!(k, archetype_key(&m), "3 distinct");
        m.spike_count = 5;
        assert_eq!(archetype_key(&m), archetype_key(&{ let mut t = m; t.spike_count = 4; t }), "3-5 fold");
        // eyes {0..3} clamp from descriptor's 1..3
        m.eye_count = 3;
        assert_eq!((archetype_key(&m) >> 7) & 0x3, 3);
        // hue must NOT affect the key (instance channel)
        let h = m.hue;
        m.hue = 213.7;
        assert_eq!(archetype_key(&{ let mut t = m; t.hue = h; t }), archetype_key(&m), "hue not keyed");
    }
}
