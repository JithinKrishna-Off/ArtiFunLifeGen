//! Queen visual identity: a prebaked, deterministic GIANT-INSECT template
//! (head + thorax + segmented abdomen + 6 jointed legs + 2 wing quads).
//! Built ONCE at `Morph::new` and appended to the morph vertex soup as a
//! fourth range (`QUEEN`), so one instanced draw renders any number of
//! queens (bounded by the planner's `QUEEN_CAP` reserved slot inside the
//! 60-morph budget).
//!
//! # Anatomy (fixed giant-insect archetype, deterministic bake)
//!
//! | Feature                  | Geometry                                         | verts |
//! |--------------------------|--------------------------------------------------|-------|
//! | head                     | ovoid icosphere (snout bias toward +X)           | 60    |
//! | compound eyes (2)        | dark half-dome bumps on the head sides            | 60    |
//! | antennae (2)             | thin tapered strands rising from the head top     | 108   |
//! | mandibles (2)            | small inward-curving jaw prongs under the snout   | 24    |
//! | thorax                   | 4-ring profile tube; widest segment, slight hump  | 108   |
//! | abdomen                  | 3 tapering ovoid bumps rearward, banded by steps  | 180   |
//! | ovipositor               | 6-sided spine at the tail tip                     | 18    |
//! | wings (2)                | thin pale swept-back quads on the thorax top      | 12    |
//! | legs (6)                 | 3 pairs x 3 tapered capsules (coxa/femur/tibia    | 324   |
//! |                          | + pointed tarsus tip), high insect knees          |       |
//! | **total**                |                                                  | **894** |
//!
//! Budget: whole insect <= ~900 template verts (894 baked). No RNG: every
//! position is a constant or a closed-form walk, bit-identical every run.
//!
//! # Canonical scale contract
//!
//! Template x spans exactly `[-1.67, +1.61]` = `QUEEN_TEMPLATE_LEN` 3.28
//! `radius` units (== `decomp::QUEEN_BODY_LEN`); the ovipositor tip drives
//! the rear bound and the mandible tips the forward bound. `decomp` derives
//! `QUEEN_SCALE` from that length, so this constant pair is a cross-file
//! contract (asserted by the in-file tests).
//!
//! The template ground line sits at y = -0.875 (`QUEEN_FOOT_Y`): `decomp`
//! lifts the origin by `QUEEN_GROUND_LIFT` (0.88) body radii, so the six
//! foot tips touch the terrain while the abdomen belly (-0.42 at crop)
//! clears it — the insect stands on its legs, not its belly.
//!
//! # WGSL branch mapping (morph.rs shader is read-only here)
//!
//! - ids in 5.5..6.6 route to the "queen lobes" body branch: breathing
//!   pulse **and** instance-radius scaling. ALL scaled dark-chitin parts
//!   (head/eyes/jaws/antennae/legs/ovipositor/thorax) ride it; distinct ids
//!   exist so tests can walk per-part vertex ranges, and future shader work
//!   can split materials without renumbering.
//! - the abdomen uses 6.75 (> 6.6) to also get the shader's abdominal band
//!   pulse (`sin(local.x * 7.0)`), reinforcing the segmented-bands read.
//! - wings use 8.45 (>= 7.5 branch): radius-scaled + slow shimmer, fixed
//!   pale-amber value — the only "pale" material the locked fragment chain
//!   offers.
//! - the 7.0..7.5 wave branch (old limb-stub wobble) is intentionally
//!   UNUSED: it never multiplies by `i.iradius`, so anything drawn there
//!   renders at fixed ~1 world unit regardless of queen size (legs would
//!   detach from a breathing/drained body and stop reaching the ground).
//!   Leg/antenna motion therefore rides the whole-body breathing pulse plus
//!   the gait-driven `wave` channel once a shader pass adds r-scaling there.
//!
//! Palette rides the existing HSV fragment chain (species hue + queen hue
//! shift): one mid-tone chitin body family, dark toward tips via the tube
//! param, pale amber wings. Anatomy is fixed-crown: nothing here feeds back
//! into simulation state.

use super::templates::TVert;

/// WC id codes (vertex aux.y) — queen family, all inside the
/// radius-scaled "queen lobes" branch (5.5..6.6) except the abdomen
/// (6.75, extra band pulse) and the wings (8.45, pale-gold shimmer branch).
pub const CODE_QHEAD: f32 = 6.0;
pub const CODE_QTHORAX: f32 = 6.25;
pub const CODE_QEYE: f32 = 6.2;
pub const CODE_QJAW: f32 = 6.3;
pub const CODE_QANTENNA_A: f32 = 6.45;
pub const CODE_QANTENNA_B: f32 = 6.5;
pub const CODE_QLEG: f32 = 6.55;
pub const CODE_QOVI: f32 = 6.58;
pub const CODE_QABDOMEN: f32 = 6.75;
pub const CODE_QWING: f32 = 8.45;

// ------------------------------------------------------------- vertex counts

/// depth-0 icosphere ovoid, snout-biased displacement
pub const HEAD_VCOUNT: u32 = 60;
/// 2 compound eyes as half-dome bumps (10 best-facing faces each)
pub const EYES_VCOUNT: u32 = 60;
/// 2 strands x 3 rings x 3 sides
pub const ANTENNAE_VCOUNT: u32 = 108;
/// 2 jaw prong cones x 4 sides
pub const MANDIBLES_VCOUNT: u32 = 24;
/// profile tube: 4 rings x 6 sides (3 spans)
pub const THORAX_VCOUNT: u32 = 108;
/// 3 tapering ovoid bumps (depth-0 icospheres)
pub const ABDOMEN_VCOUNT: u32 = 180;
/// tail spine cone x 6 sides
pub const OVIPOSITOR_VCOUNT: u32 = 18;
/// 2 thin swept-back wing quads (2 tris each)
pub const WINGS_VCOUNT: u32 = 12;
/// 6 legs x 3 capsules x (1 ring x 3 sides)
pub const LEGS_VCOUNT: u32 = 324;

pub const QUEEN_VCOUNT: u32 = HEAD_VCOUNT
    + EYES_VCOUNT
    + ANTENNAE_VCOUNT
    + MANDIBLES_VCOUNT
    + THORAX_VCOUNT
    + ABDOMEN_VCOUNT
    + OVIPOSITOR_VCOUNT
    + WINGS_VCOUNT
    + LEGS_VCOUNT;

// ------------------------------------------------------------- scale contract

/// Template x extents (radius units): mandible tips .. ovipositor tip.
pub const QUEEN_BODY_MIN_X: f32 = -1.67;
pub const QUEEN_BODY_MAX_X: f32 = 1.61;
/// == decomp::QUEEN_BODY_LEN: 3.28 radius units.
pub const QUEEN_TEMPLATE_LEN: f32 = QUEEN_BODY_MAX_X - QUEEN_BODY_MIN_X;

/// Template height of the six foot tips: just above the ground line the
/// planner seats at `-QUEEN_GROUND_LIFT` (0.88) body radii.
pub const QUEEN_FOOT_Y: f32 = -0.875;

/// 3 leg pairs (front / mid / hind) x 2 sides.
pub const LEG_PAIRS: u32 = 3;
/// capsules per leg: coxa, femur, tibia (tibia tapers into the foot tip).
pub const LEG_SEGMENTS: u32 = 3;

const TAU: f32 = std::f32::consts::TAU;

// ------------------------------------------------------------- small helpers

#[inline]
fn tv(p: [f32; 3], param: f32, id: f32) -> TVert {
    TVert { pos: [p[0], p[1], p[2], 0.0], aux: [param, id] }
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

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[cfg(test)]
#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2]))
        .sqrt()
}

/// Ring-base point around a center with orthonormal basis (n, b).
#[inline]
fn ring_pt(c: [f32; 3], n: [f32; 3], b: [f32; 3], a: f32, r: f32) -> [f32; 3] {
    let (ca, sa) = (a.cos(), a.sin());
    [
        c[0] + (n[0] * ca + b[0] * sa) * r,
        c[1] + (n[1] * ca + b[1] * sa) * r,
        c[2] + (n[2] * ca + b[2] * sa) * r,
    ]
}

/// Stable perpendicular ring basis for a tangent (same rule as templates).
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

/// Gentle organic ripple shared by the queen ovoids (baked, subtle so the
/// insect silhouette stays clean).
fn queen_ripple(n: [f32; 3]) -> f32 {
    1.0 + 0.03 * (7.1 * n[0] + 3.3 * n[1] + 2.7 * n[2]).sin()
        + 0.015 * (4.7 * n[1] + 6.1 * n[2]).sin()
}

/// Straight tapered tube between two points (`rings` spans, `sides` sides,
/// soup quads; vertex aux.x = tube param 0..1 so the shader can taper
/// pigment to the tip and wave along the strand).
fn push_taper_tube(
    out: &mut Vec<TVert>,
    from: [f32; 3],
    to: [f32; 3],
    r0: f32,
    r1: f32,
    rings: u32,
    sides: u32,
    id: f32,
) {
    let dir = norm3([to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
    let (n, b) = tube_basis(dir);
    for i in 0..rings {
        let t0 = i as f32 / rings as f32;
        let t1 = (i + 1) as f32 / rings as f32;
        let c0 = mix3(from, to, t0);
        let c1 = mix3(from, to, t1);
        let rr0 = r0 + (r1 - r0) * t0;
        let rr1 = r0 + (r1 - r0) * t1;
        for j in 0..sides {
            let a0 = TAU * j as f32 / sides as f32;
            let a1 = TAU * ((j + 1) % sides) as f32 / sides as f32;
            let v00 = ring_pt(c0, n, b, a0, rr0);
            let v01 = ring_pt(c0, n, b, a1, rr0);
            let v10 = ring_pt(c1, n, b, a0, rr1);
            let v11 = ring_pt(c1, n, b, a1, rr1);
            for (p, t) in [(v00, t0), (v10, t1), (v11, t1), (v00, t0), (v11, t1), (v01, t0)] {
                out.push(tv(p, t, id));
            }
        }
    }
}

/// Straight cone from `base` toward `tip` (open base — it sits inside the
/// body). `sides` triangles; vertex aux.x param 0 (base) .. 1 (tip).
fn push_cone(out: &mut Vec<TVert>, base: [f32; 3], tip: [f32; 3], r0: f32, sides: u32, id: f32) {
    let dir = norm3([tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]]);
    let (n, b) = tube_basis(dir);
    for j in 0..sides {
        let a0 = TAU * j as f32 / sides as f32;
        let a1 = TAU * ((j + 1) % sides) as f32 / sides as f32;
        let p0 = ring_pt(base, n, b, a0, r0);
        let p1 = ring_pt(base, n, b, a1, r0);
        for (p, t) in [(p0, 0.0), (tip, 1.0), (p1, 0.0)] {
            out.push(tv(p, t, id));
        }
    }
}

/// Thin flat quad (2 soup triangles) from 4 corners in order.
fn push_quad(out: &mut Vec<TVert>, p: [[f32; 3]; 4], param: f32, id: f32) {
    for tri in [[0usize, 1, 2], [0, 2, 3]] {
        for i in tri {
            out.push(tv(p[i], param, id));
        }
    }
}

/// Displaced depth-0 icosphere ovoid: `at` center, `r` radius, `scale`
/// per-axis, `disp` unit-sphere radius modifier. 60 soup verts.
fn push_oval(
    out: &mut Vec<TVert>,
    at: [f32; 3],
    r: f32,
    scale: [f32; 3],
    disp: impl Fn([f32; 3]) -> f32,
    param: f32,
    id: f32,
) {
    for p in super::templates::sphere_soup(0) {
        let rr = r * disp(p);
        out.push(tv(
            [
                at[0] + p[0] * rr * scale[0],
                at[1] + p[1] * rr * scale[1],
                at[2] + p[2] * rr * scale[2],
            ],
            param,
            id,
        ));
    }
}

/// Half dome "bump": exactly the 10 depth-0 icosphere faces whose centroids
/// face `dir` most (top-10 selection -> deterministic, direction-safe).
/// 30 soup verts each — two eyes cost 60.
fn push_half_dome(out: &mut Vec<TVert>, at: [f32; 3], r: f32, dir: [f32; 3], param: f32, id: f32) {
    let soup = super::templates::sphere_soup(0);
    debug_assert_eq!(soup.len(), 60);
    let faces: Vec<[[f32; 3]; 3]> =
        soup.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    debug_assert_eq!(faces.len(), 20);
    let mut order: Vec<(f32, usize)> = faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let c = norm3(mix3(mix3(f[0], f[1], 0.5), f[2], 2.0 / 3.0));
            (c[0] * dir[0] + c[1] * dir[1] + c[2] * dir[2], i)
        })
        .collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, fi) in order.iter().take(10) {
        for v in faces[fi] {
            out.push(tv(
                [at[0] + v[0] * r, at[1] + v[1] * r, at[2] + v[2] * r],
                param,
                id,
            ));
        }
    }
}

/// Profile tube through explicit ring centers + radii with ONE shared ring
/// basis (axes stay near-colinear here, so no per-segment basis twist).
/// (centers.len() - 1) spans x sides x 6 verts; aux.x param walks 0..1.
fn push_profile_tube(
    out: &mut Vec<TVert>,
    centers: &[[f32; 3]],
    radii: &[f32],
    sides: u32,
    id: f32,
) {
    let rings = centers.len();
    debug_assert_eq!(rings, radii.len());
    debug_assert!(rings >= 2);
    let axis = norm3([
        centers[rings - 1][0] - centers[0][0],
        centers[rings - 1][1] - centers[0][1],
        centers[rings - 1][2] - centers[0][2],
    ]);
    let (n, b) = tube_basis(axis);
    for i in 0..rings - 1 {
        let (c0, c1) = (centers[i], centers[i + 1]);
        let (r0, r1) = (radii[i], radii[i + 1]);
        let t0 = i as f32 / (rings - 1) as f32;
        let t1 = (i + 1) as f32 / (rings - 1) as f32;
        for j in 0..sides {
            let a0 = TAU * j as f32 / sides as f32;
            let a1 = TAU * ((j + 1) % sides) as f32 / sides as f32;
            let v00 = ring_pt(c0, n, b, a0, r0);
            let v01 = ring_pt(c0, n, b, a1, r0);
            let v10 = ring_pt(c1, n, b, a0, r1);
            let v11 = ring_pt(c1, n, b, a1, r1);
            for (p, t) in [(v00, t0), (v10, t1), (v11, t1), (v00, t0), (v11, t1), (v01, t0)] {
                out.push(tv(p, t, id));
            }
        }
    }
}

// ------------------------------------------------------------- anatomy data

const HEAD_AT: [f32; 3] = [1.02, 0.08, 0.0];
const HEAD_R: f32 = 0.44;
const HEAD_SCALE: [f32; 3] = [1.16, 0.96, 1.0];

const EYE_AT: [f32; 3] = [1.13, 0.22, 0.40];
const EYE_R: f32 = 0.16;

const ANT_BASE: [f32; 3] = [1.16, 0.30, 0.12];
const ANT_TIP: [f32; 3] = [1.55, 0.62, 0.30];
const ANT_R0: f32 = 0.035;
const ANT_R1: f32 = 0.012;

const JAW_BASE: [f32; 3] = [1.32, -0.10, 0.15];
const JAW_TIP: [f32; 3] = [1.61, -0.22, 0.02];
const JAW_R0: f32 = 0.055;

/// Widest body segment with a slight middle hump (ring 1: y +0.10, r 0.52).
const THORAX_CTRS: [[f32; 3]; 4] = [
    [0.48, 0.00, 0.0],
    [0.22, 0.10, 0.0],
    [-0.04, 0.06, 0.0],
    [-0.24, -0.02, 0.0],
];
const THORAX_RADII: [f32; 4] = [0.40, 0.52, 0.50, 0.38];

/// Segmented gaster: 3 ovoid bumps tapering rearward (silhouette steps read
/// as bands; the shader band pulse on id > 6.6 reinforces it). Belly clears
/// the ground line: lowest ovoid bottom ~ -0.42.
const ABD_CTRS: [[f32; 3]; 3] = [
    [-0.52, 0.02, 0.0],
    [-0.98, 0.00, 0.0],
    [-1.30, -0.02, 0.0],
];
const ABD_RADII: [f32; 3] = [0.50, 0.42, 0.32];
const ABD_SQUASH: [f32; 3] = [0.88, 0.86, 0.84];

const OVI_BASE: [f32; 3] = [-1.50, -0.05, 0.0];
const OVI_TIP: [f32; 3] = [-1.67, -0.05, 0.0];
const OVI_R0: f32 = 0.055;

/// Thin swept-back wing quad (z scaled by side sign); roots sink slightly
/// into the thorax top so the wings grow out of the back.
const WING_ROOT_FRONT: [f32; 3] = [0.34, 0.50, 0.10];
const WING_ROOT_REAR: [f32; 3] = [-0.30, 0.46, 0.13];
const WING_TIP_REAR: [f32; 3] = [-1.02, 0.40, 0.44];
const WING_TIP_FRONT: [f32; 3] = [-0.02, 0.44, 0.42];

/// One leg = [anchor, elbow (coxa end), knee, foot tip] in template radius
/// units. Anchors sit inside the thorax profile; knees ride high (insect
/// gait); foot tips land on the ground line `QUEEN_FOOT_Y`.
/// Deterministic constant walk — no RNG.
pub fn leg_chains() -> [[[f32; 3]; 4]; 6] {
    // (anchor, knee, foot (x, z)) per pair; y of the foot is QUEEN_FOOT_Y.
    const PAIRS: [([f32; 3], [f32; 3], [f32; 2]); 3] = [
        ([0.34, -0.22, 0.26], [0.62, 0.14, 0.46], [0.72, 0.34]), // front
        ([0.06, -0.26, 0.30], [0.34, 0.16, 0.55], [0.30, 0.48]), // mid
        ([-0.22, -0.22, 0.26], [-0.30, 0.10, 0.50], [-0.62, 0.24]), // hind
    ];
    let mut out: [[[f32; 3]; 4]; 6] = [[[0.0; 3]; 4]; 6];
    for p in 0..3 {
        for s in 0..2 {
            let sg = if s == 0 { 1.0 } else { -1.0 };
            let (a, k, f) = PAIRS[p];
            let anchor = [a[0], a[1], a[2] * sg];
            let knee = [k[0], k[1], k[2] * sg];
            let foot = [f[0], QUEEN_FOOT_Y, f[1] * sg];
            let elbow = mix3(anchor, knee, 0.35); // coxa is the short first span
            out[p * 2 + s] = [anchor, elbow, knee, foot];
        }
    }
    out
}

// ------------------------------------------------------------- part builders

fn push_head(out: &mut Vec<TVert>) {
    // snout bias toward +X on top of the shared ripple
    let disp = |n: [f32; 3]| -> f32 {
        let snout = n[0].max(0.0);
        queen_ripple(n) + 0.07 * snout * snout
    };
    push_oval(out, HEAD_AT, HEAD_R, HEAD_SCALE, disp, 0.0, CODE_QHEAD);
}

fn push_eyes(out: &mut Vec<TVert>) {
    for z in [EYE_AT[2], -EYE_AT[2]] {
        // pure dome (no ripple): hard ellipsoid bump reading
        push_half_dome(out, [EYE_AT[0], EYE_AT[1], z], EYE_R, [0.0, 0.0, z.signum()], 0.0, CODE_QEYE);
    }
}

fn push_mandibles(out: &mut Vec<TVert>) {
    for z in [JAW_BASE[2], -JAW_BASE[2]] {
        let base = [JAW_BASE[0], JAW_BASE[1], z];
        // tips converge inward/downward under the snout (jaw prong curve read)
        let tip = [JAW_TIP[0], JAW_TIP[1], JAW_TIP[2] * z.signum() * -1.0];
        push_cone(out, base, tip, JAW_R0, 4, CODE_QJAW);
    }
}

fn push_antennae(out: &mut Vec<TVert>) {
    for s in [1.0, -1.0] {
        let id = if s > 0.0 { CODE_QANTENNA_A } else { CODE_QANTENNA_B };
        let base = [ANT_BASE[0], ANT_BASE[1], ANT_BASE[2] * s];
        let tip = [ANT_TIP[0], ANT_TIP[1], ANT_TIP[2] * s];
        push_taper_tube(out, base, tip, ANT_R0, ANT_R1, 3, 3, id);
    }
}

fn push_thorax(out: &mut Vec<TVert>) {
    push_profile_tube(out, &THORAX_CTRS, &THORAX_RADII, 6, CODE_QTHORAX);
}

fn push_abdomen(out: &mut Vec<TVert>) {
    for i in 0..3 {
        push_oval(
            out,
            ABD_CTRS[i],
            ABD_RADII[i],
            [1.0, ABD_SQUASH[i], ABD_SQUASH[i]],
            queen_ripple,
            0.0,
            CODE_QABDOMEN,
        );
    }
}

fn push_ovipositor(out: &mut Vec<TVert>) {
    push_cone(out, OVI_BASE, OVI_TIP, OVI_R0, 6, CODE_QOVI);
}

fn push_wings(out: &mut Vec<TVert>) {
    for s in [1.0, -1.0] {
        let z = |v: [f32; 3]| [v[0], v[1], v[2] * s];
        push_quad(
            out,
            [
                z(WING_ROOT_FRONT),
                z(WING_ROOT_REAR),
                z(WING_TIP_REAR),
                z(WING_TIP_FRONT),
            ],
            0.0,
            CODE_QWING,
        );
    }
}

fn push_legs(out: &mut Vec<TVert>) {
    for chain in leg_chains().iter() {
        let (a, e, k, f) = (chain[0], chain[1], chain[2], chain[3]);
        // coxa: short thick span out of the thorax underside
        push_taper_tube(out, a, e, 0.075, 0.062, 1, 3, CODE_QLEG);
        // femur: up to the high knee
        push_taper_tube(out, e, k, 0.062, 0.045, 1, 3, CODE_QLEG);
        // tibia: down the far side, tapering into the pointed tarsus tip
        push_taper_tube(out, k, f, 0.045, 0.012, 1, 3, CODE_QLEG);
    }
}

/// Builds the full queen template (pure, deterministic; 894 verts).
pub fn build() -> Vec<TVert> {
    let mut out: Vec<TVert> = Vec::with_capacity(QUEEN_VCOUNT as usize);
    push_head(&mut out);
    push_eyes(&mut out);
    push_mandibles(&mut out);
    push_antennae(&mut out);
    push_thorax(&mut out);
    push_abdomen(&mut out);
    push_ovipositor(&mut out);
    push_wings(&mut out);
    push_legs(&mut out);
    debug_assert_eq!(out.len() as u32, QUEEN_VCOUNT);
    out
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decomp;
    use bytemuck::cast_slice;

    fn pos3(t: &TVert) -> [f32; 3] {
        [t.pos[0], t.pos[1], t.pos[2]]
    }

    fn count_id(v: &[TVert], id: f32) -> u32 {
        v.iter().filter(|t| t.aux[1] == id).count() as u32
    }

    #[test]
    fn template_segment_counts_and_budget() {
        let v = build();
        assert_eq!(v.len() as u32, QUEEN_VCOUNT, "build matches the declared soup");
        assert!(QUEEN_VCOUNT <= 900, "giant-insect budget ~900, got {}", QUEEN_VCOUNT);
        // exact per-part counts (must sum to the whole)
        assert_eq!(HEAD_VCOUNT + EYES_VCOUNT + ANTENNAE_VCOUNT + MANDIBLES_VCOUNT
            + THORAX_VCOUNT + ABDOMEN_VCOUNT + OVIPOSITOR_VCOUNT + WINGS_VCOUNT + LEGS_VCOUNT,
            QUEEN_VCOUNT);
        assert_eq!(count_id(&v, CODE_QHEAD), HEAD_VCOUNT);
        assert_eq!(count_id(&v, CODE_QEYE), EYES_VCOUNT);
        assert_eq!(
            count_id(&v, CODE_QANTENNA_A) + count_id(&v, CODE_QANTENNA_B),
            ANTENNAE_VCOUNT
        );
        assert_eq!(count_id(&v, CODE_QJAW), MANDIBLES_VCOUNT);
        assert_eq!(count_id(&v, CODE_QTHORAX), THORAX_VCOUNT);
        assert_eq!(count_id(&v, CODE_QABDOMEN), ABDOMEN_VCOUNT);
        assert_eq!(count_id(&v, CODE_QOVI), OVIPOSITOR_VCOUNT);
        assert_eq!(count_id(&v, CODE_QWING), WINGS_VCOUNT);
        assert_eq!(count_id(&v, CODE_QLEG), LEGS_VCOUNT);
        // legs: 6 legs x 3 capsules, exactly as declared
        assert_eq!(LEG_PAIRS * 2 * LEG_SEGMENTS * 18, LEGS_VCOUNT);
        // 3 tapering abdominal bumps (segmented gaster)
        assert_eq!(ABDOMEN_VCOUNT, 3 * 60);
    }

    #[test]
    fn canonical_body_length_is_3_28() {
        // morph_decomp::QUEEN_BODY_LEN contract: 3.28 radius units
        assert!((decomp::QUEEN_BODY_LEN - QUEEN_TEMPLATE_LEN).abs() < 1e-4);
        let v = build();
        let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
        for t in &v {
            min_x = min_x.min(t.pos[0]);
            max_x = max_x.max(t.pos[0]);
        }
        assert!(
            (min_x - QUEEN_BODY_MIN_X).abs() < 5e-3,
            "rear bound {} != the ovipositor tip",
            min_x
        );
        assert!(
            (max_x - QUEEN_BODY_MAX_X).abs() < 5e-3,
            "front bound {} != the mandible tips",
            max_x
        );
        assert!((max_x - min_x - 3.28).abs() < 1e-2, "canonical length 3.28");
        // no degenerate / non-finite verts
        for t in &v {
            for c in t.pos.iter() {
                assert!(c.is_finite());
            }
        }
    }

    #[test]
    fn six_leg_attachment_points() {
        let chains = leg_chains();
        assert_eq!(chains.len(), 6);
        for p in 0..3 {
            let a = &chains[p * 2];
            let b = &chains[p * 2 + 1];
            // mirrored pairs share (x, y) and flip z
            assert_eq!(a[0][0], b[0][0]);
            assert_eq!(a[0][1], b[0][1]);
            assert!((a[0][2] + b[0][2]).abs() < 1e-6);
            // anchors sit below the thorax axis and clear of the midline
            assert!(a[0][1] < -0.1, "anchor under the thorax belly");
            assert!(a[0][2] > 0.2, "anchor clear of the midline");
        }
        // front / mid / hind anchors step rearward along the thorax, and
        // every foot tip rides the planner ground line
        assert!(chains[0][0][0] > chains[2][0][0], "front attach forward");
        assert!(chains[2][0][0] > chains[4][0][0], "mid attach forward of hind");
        for chain in &chains {
            assert!(chain[3][1] == QUEEN_FOOT_Y);
        }
        // each attachment point has coxa geometry within a coxa radius
        let v = build();
        for (li, chain) in chains.iter().enumerate() {
            let near = v
                .iter()
                .filter(|t| t.aux[1] == CODE_QLEG)
                .map(|t| dist3(pos3(t), chain[0]))
                .fold(f32::MAX, f32::min);
            assert!(near < 0.15, "leg {} has no coxa verts at its anchor", li);
        }
        // feet actually at the planner ground line (-QUEEN_GROUND_LIFT)
        for chain in &chains {
            assert!(
                (chain[3][1] - -decomp::QUEEN_GROUND_LIFT).abs() <= 0.02,
                "foot tip must ride the morph_decomp ground line"
            );
        }
    }

    #[test]
    fn ids_route_into_valid_shader_branches() {
        let v = build();
        for t in &v {
            let id = t.aux[1];
            let in_body = id > 5.5 && id < 7.0;
            let in_wing = id == CODE_QWING;
            assert!(in_body || in_wing, "id {} outside shader-safe ranges", id);
        }
        // abdomen exercises the > 6.6 band-pulse sub-branch
        assert!(count_id(&v, CODE_QABDOMEN) > 0 && CODE_QABDOMEN > 6.6);
        // no stray old codes (limb wave branch is unused by design)
        for t in &v {
            let id = t.aux[1];
            assert!(!(id > 7.0 && id < 7.5), "7.0..7.5 branch never scales by radius");
        }
    }

    #[test]
    fn build_is_bit_identical_across_calls() {
        let a = build();
        let b = build();
        assert_eq!(
            cast_slice::<TVert, u8>(&a),
            cast_slice::<TVert, u8>(&b),
            "deterministic template bake"
        );
    }
}
