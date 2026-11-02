//! Deterministic tree + bush placement (pure CPU, testable).
//! Analogous to `grass_place.rs` but hosts WIDER coverage: the tree window
//! mirrors the grass window x2.0 (mission: place trees at medium/far
//! distance, beyond the grass near ring), and TREE CHUNKS are 64x64 units —
//! the SAME size as the terrain `CHUNK` — so per-chunk instance batches
//! align with terrain chunk draws for free.
//!
//! Everything derives from the real `WorldModel` fields: fertility,
//! moisture, hazard, elevation, biome id. Heights are WORLD units
//! (trunk+canopy = 5.5..9.5, bushes 0.9..1.5); `scale` folds that into the
//! shared mesh nominal heights. Water cutoff: nothing below the sea
//! shallow line (`e < WATER_E`), matching the terrain sea convention.

use crate::grass::place as gp;
use crate::scene::CameraMode;
use crate::terrain::{HEIGHT_SCALE, SEA_LEVEL};
use crate::tree_mesh::{NOM_BUSH_H, NOM_TREE_H};
use alife_worldgen::WorldModel;
use bytemuck::{Pod, Zeroable};

/// tree-chunk side == terrain chunk side (64 units) for free alignment
pub const CHUNK: f32 = 64.0;

/// tree window = grass window * 2.0 (see module docs; alignment with the
/// terrain plan chunk set is approximate — documented coordination risk)
const TREE_WINDOW_MUL: f32 = 2.0;

/// no trees/bushes when the elevation field is below this (SEA_LEVEL +
/// shallow margin so trunks never stand in water)
pub const WATER_E: f32 = SEA_LEVEL + 0.02;

/// LOD rings (mission): near < 160 -> LOD-A mesh, mid < 420 -> LOD-B,
/// far (beyond 420, to the window edge) -> LOD-B + dim 0.55 + density *0.5
pub const RING_NEAR: f32 = 160.0;
pub const RING_MID: f32 = 420.0;

/// anti-overlap radii (chunk-local, like grass_place): trees need crown
/// room, bushes tuck tighter but stay off tree footprints
pub const SPACING_TREE: f32 = 8.0;
pub const SPACING_BUSH: f32 = 2.6;
pub const SPACING_BUSH_TREE: f32 = 2.2;

/// candidate hash attempts per 64-chunk (before density/ring gates)
const TREES_CAND: u32 = 10;
const BUSH_CAND: u32 = 14;

/// keep-probability normalizer: on a lush chunk (~f .75 x m .5) this
/// yields ~1 tree and ~1.4 bushes per chunk near the focus
const TREE_RATE_MUL: f32 = 0.28;
const BUSH_RATE_MUL: f32 = 0.28;

/// deterministic planet caps (2400x1600 world): 400 trees x 96 tris +
/// 800 bushes x 12 tris = 48.0k tris, inside the ~150k budget
pub const CAP_TREE_PLANET: usize = 400;
pub const CAP_BUSH_PLANET: usize = 800;
pub const CAP_TREE_HABITAT: usize = 300;
pub const CAP_BUSH_HABITAT: usize = 600;
pub const CAP_TREE_FOLLOW: usize = 150;
pub const CAP_BUSH_FOLLOW: usize = 300;

pub const MODE_ORBIT: u8 = gp::MODE_ORBIT;
pub const MODE_FOLLOW: u8 = gp::MODE_FOLLOW;
pub const MODE_PLANET: u8 = gp::MODE_PLANET;

/// same mode classification the grass windowing uses
pub fn mode_of(cm: &CameraMode) -> u8 {
    gp::mode_of(cm)
}

/// tree placement window: grass window * 2.0 (planet covers the world)
pub fn tree_window_for(mode: u8, world: &WorldModel) -> f32 {
    (gp::window_for(mode, world) * TREE_WINDOW_MUL).min(world.width.max(world.height) * 1.6)
}

/// hard instance ceiling for a mode
fn caps(mode: u8) -> (usize, usize) {
    if mode == MODE_PLANET {
        (CAP_TREE_PLANET, CAP_BUSH_PLANET)
    } else if mode == MODE_FOLLOW {
        (CAP_TREE_FOLLOW, CAP_BUSH_FOLLOW)
    } else {
        (CAP_TREE_HABITAT, CAP_BUSH_HABITAT)
    }
}

/// Per-instance record, 48 bytes = 12 x f32 (documented stride; the
/// attribute layout in `tree.rs` mirrors this order exactly).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct TreeInst {
    /// anchor in world coords (trunk / bush base centre)
    pub x: f32,
    pub z: f32,
    /// ground contact Y (elevation * HEIGHT_SCALE, terrain.rs convention)
    pub base_y: f32,
    /// deterministic yaw in radians
    pub yaw: f32,
    /// world height / nominal mesh height (trees 5.5..9.5 -> ~0.70..1.22)
    pub scale: f32,
    /// canopy hue; quantized into 3 variants (green 0/1/2, toxic override)
    pub hue: f32,
    /// LOD ring dim: 1.0 near/mid, 0.55 far
    pub dim: f32,
    /// 0 = tree (trunk+canopy mesh), 1 = bush (blob mesh)
    pub kind: f32,
    /// radial scale multiplier (trunk radius 0.4..0.7 -> 0.8..1.4)
    pub rad: f32,
    /// deterministic shader-variation seed
    pub seed: f32,
    pub pad0: f32,
    pub pad1: f32,
}

pub struct TreeBuilder {
    pub gen: u32,
    /// cache cell this build is attached to (identity only)
    pub cell: (i32, i32),
    pub mode: u8,
    /// camera focus in world coords (drives LOD rings + caps)
    pub focus: (f32, f32),
    /// adaptive density multiplier (0.4..=1.0), folded into drop rate
    pub lod_scale: f32,
}

pub struct TreeChunkPlan {
    pub cx: i32,
    pub cy: i32,
    pub trees: Vec<TreeInst>,
    pub bushes: Vec<TreeInst>,
}

pub struct TreeBuildResult {
    pub chunks: Vec<TreeChunkPlan>,
    pub trees: usize,
    pub bushes: usize,
}

// ------------------------------------------------------------------ hash

/// per-(world, chunk, attempt) placement stream seed: prime a splitmix
/// stream state (the mixer lives in grass_place and is crate-shared)
pub fn tree_hash(gen: u32, cx: i32, cy: i32, idx: u32) -> u64 {
    let mut s: u64 = 0x5445_4552_0A11_5C1F; // "TREES" salt
    s ^= (gen as u64).rotate_left(40);
    s ^= (cx as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= (cy as i64 as u64).rotate_left(29);
    s ^= (idx as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    s ^= 0x2545_F491_4F6C_DD1D;
    gp::hash01(&mut s); // prime: one splitmix step so the state is well mixed
    s
}

// ------------------------------------------------------------- density

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// biome density multipliers: fertile biomes (0/1/4) get canopies, rocky
/// (2) fewer, toxic (3) rare sickly trunks, niche (5) sparse
const BIOME_MUL: [f32; 6] = [1.00, 0.80, 0.15, 0.06, 0.85, 0.30];

/// per-candidate tree keep probability (before ring/lod_scale/patch)
pub fn tree_density(f: f32, m: f32, hz: f32, biome: u32) -> f32 {
    let f = clamp01(f);
    let m = clamp01(m);
    let hz = clamp01(hz);
    let b = (biome % 6) as usize;
    let mut d = clamp01(f * (0.30 + 0.70 * m)) * BIOME_MUL[b];
    // toxic flats: needs a sliver of fertility even to keep a sick trunk
    if b == 3 {
        d *= clamp01(f * 2.5);
    }
    // hazard keeps ground bare (same sharp curve as grass)
    if hz > 0.15 {
        d *= clamp01((0.35 - hz) / 0.20);
    }
    d
}

/// per-candidate bush keep probability
pub fn bush_density(f: f32, m: f32, hz: f32, biome: u32) -> f32 {
    let f = clamp01(f);
    let b = (biome % 6) as usize;
    let mut d = clamp01(f * (0.35 + 0.65 * clamp01(m))) * BIOME_MUL[b] * 1.25;
    if b == 3 {
        d *= clamp01(f * 2.5) * 0.5;
    }
    if clamp01(hz) > 0.15 {
        d *= clamp01((0.35 - hz) / 0.20);
    }
    d
}

/// chunk-ring multiplier: near/mid full density, far half (mission item 5)
pub fn ring_rate(cd: f32) -> (f32, f32) {
    if cd < RING_NEAR {
        (1.0, 1.0)
    } else if cd < RING_MID {
        (1.0, 1.0)
    } else {
        (0.5, 0.55)
    }
}

// ---------------------------------------------------------------- build

pub fn build(world: &WorldModel, b: &TreeBuilder) -> TreeBuildResult {
    let window = tree_window_for(b.mode, world);
    let half = window * 0.5;

    let x0 = (b.focus.0 - half).clamp(0.0, world.width.max(CHUNK));
    let y0 = (b.focus.1 - half).clamp(0.0, world.height.max(CHUNK));
    let c0x = ((x0 / CHUNK).floor() as i32).max(0);
    let c0y = ((y0 / CHUNK).floor() as i32).max(0);
    let c1x = (((x0 + window) / CHUNK).ceil() as i32)
        .max(((world.width / CHUNK).ceil() as i32).min(c0x + 96));
    let c1y = (((y0 + window) / CHUNK).ceil() as i32)
        .max(((world.height / CHUNK).ceil() as i32).min(c0y + 96));

    let mut chunks: Vec<TreeChunkPlan> = Vec::new();
    for cy in c0y..=c1y {
        for cx in c0x..=c1x {
            let ox = cx as f32 * CHUNK;
            let oy = cy as f32 * CHUNK;
            if ox > x0 + window || oy > y0 + window {
                continue;
            }
            // chunk-ring LOD from the CHUNK CENTRE distance to the focus
            let ccx = ox + CHUNK * 0.5;
            let ccz = oy + CHUNK * 0.5;
            let cd = ((ccx - b.focus.0).powi(2) + (ccz - b.focus.1).powi(2)).sqrt();
            let (rate_mul, dim) = ring_rate(cd);
            if rate_mul <= f32::EPSILON {
                continue;
            }
            let mut trees: Vec<TreeInst> = Vec::new();
            let mut bushes: Vec<TreeInst> = Vec::new();
            let mut planted_t: Vec<(f32, f32)> = Vec::new();
            let mut planted_b: Vec<(f32, f32)> = Vec::new();

            // ---- trees
            for i in 0..TREES_CAND {
                let mut s = tree_hash(b.gen, cx, cy, i);
                let jx = gp::hash01(&mut s);
                let jy = gp::hash01(&mut s);
                let slot = gp::hash01(&mut s);
                let rh = gp::hash01(&mut s);
                let rw = gp::hash01(&mut s);
                let ryaw = gp::hash01(&mut s);
                let rvar = gp::hash01(&mut s);
                let px = (ox + jx * CHUNK).clamp(0.0, world.width - 0.001);
                let pz = (oy + jy * CHUNK).clamp(0.0, world.height - 0.001);
                let e = world.sample_elevation(px, pz);
                // WATER CUTOFF: nothing below the sea shallow line
                if e < WATER_E {
                    continue;
                }
                let f = world.sample_fertility(px, pz);
                let m = world.sample_moisture(px, pz);
                let hz = world.sample_hazard(px, pz);
                let bio = world.biome_id(px, pz) as u32;
                let dense = tree_density(f, m, hz, bio);
                let rate = dense * rate_mul * b.lod_scale * TREE_RATE_MUL
                    * (0.25 + 0.75 * gp::patch_mask(b.gen, px, pz));
                if slot >= rate {
                    continue;
                }
                // terrain fit: no trees on bluffs or in dips that drown roots
                if gp::clump_slope_norm(world, px, pz).is_none() {
                    continue;
                }
                // chunk-local anti-overlap (mission: like grass_place)
                let mut ok = true;
                for (qx, qz) in &planted_t {
                    let ddx = px - *qx;
                    let ddz = pz - *qz;
                    if ddx * ddx + ddz * ddz < SPACING_TREE * SPACING_TREE {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                planted_t.push((px, pz));
                // heights world 5.5..9.5 -> scale vs the 7.8 nominal mesh
                let h = lerp(5.5, 9.5, rh);
                let mut scale = h / NOM_TREE_H;
                // 3 canopy hue variants (toxic biomes go sickly yellow)
                let b3 = bio % 6 == 3;
                let hue = if b3 {
                    lerp(0.50, 0.58, rh)
                } else {
                    let v = (rvar * 3.0).min(2.999).floor();
                    0.30 + v * 0.04 + (rw - 0.5) * 0.012
                };
                if b3 {
                    scale *= 0.85;
                }
                trees.push(TreeInst {
                    x: px,
                    z: pz,
                    base_y: e * HEIGHT_SCALE - 0.25,
                    yaw: ryaw * std::f32::consts::TAU,
                    scale,
                    hue,
                    dim,
                    kind: 0.0,
                    rad: lerp(0.4, 0.7, rw) / 0.5,
                    seed: jx,
                    pad0: 0.0,
                    pad1: 0.0,
                });
            }

            // ---- bushes
            for i in 0..BUSH_CAND {
                let mut s = tree_hash(b.gen, cx, cy, i + 100);
                let jx = gp::hash01(&mut s);
                let jy = gp::hash01(&mut s);
                let slot = gp::hash01(&mut s);
                let rh = gp::hash01(&mut s);
                let ryaw = gp::hash01(&mut s);
                let rc = gp::hash01(&mut s);
                let px = (ox + jx * CHUNK).clamp(0.0, world.width - 0.001);
                let pz = (oy + jy * CHUNK).clamp(0.0, world.height - 0.001);
                let e = world.sample_elevation(px, pz);
                if e < WATER_E {
                    continue;
                }
                let f = world.sample_fertility(px, pz);
                let m = world.sample_moisture(px, pz);
                let hz = world.sample_hazard(px, pz);
                let bio = world.biome_id(px, pz) as u32;
                let dense = bush_density(f, m, hz, bio);
                let rate = dense * rate_mul * b.lod_scale * BUSH_RATE_MUL
                    * (0.35 + 0.65 * gp::patch_mask(b.gen, px, pz));
                if slot >= rate {
                    continue;
                }
                // bushes keep off tree footprints + each other (chunk-local)
                let mut ok = true;
                for (qx, qz) in &planted_t {
                    let ddx = px - *qx;
                    let ddz = pz - *qz;
                    if ddx * ddx + ddz * ddz < SPACING_BUSH_TREE * SPACING_BUSH_TREE {
                        ok = false;
                        break;
                    }
                }
                for (qx, qz) in &planted_b {
                    let ddx = px - *qx;
                    let ddz = pz - *qz;
                    if ddx * ddx + ddz * ddz < SPACING_BUSH * SPACING_BUSH {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                planted_b.push((px, pz));
                let b3 = bio % 6 == 3;
                let hue = if b3 {
                    lerp(0.44, 0.52, rh)
                } else {
                    lerp(0.28, 0.42, rc)
                };
                let h = lerp(0.9, 1.5, rh);
                bushes.push(TreeInst {
                    x: px,
                    z: pz,
                    base_y: e * HEIGHT_SCALE - 0.12,
                    yaw: ryaw * std::f32::consts::TAU,
                    scale: h / NOM_BUSH_H,
                    hue,
                    dim,
                    kind: 1.0,
                    rad: lerp(0.85, 1.35, ryaw),
                    seed: jx,
                    pad0: 0.0,
                    pad1: 0.0,
                });
            }

            if !trees.is_empty() || !bushes.is_empty() {
                chunks.push(TreeChunkPlan { cx, cy, trees, bushes });
            }
        }
    }

    // deterministic budget trim: nearest chunks to the focus win
    let (fcx, fcz) = b.focus;
    chunks.sort_by(|a, b2| {
        let da = (a.cx as f32 * CHUNK - fcx).hypot(a.cy as f32 * CHUNK - fcz);
        let db = (b2.cx as f32 * CHUNK - fcx).hypot(b2.cy as f32 * CHUNK - fcz);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let (ct, cb) = caps(b.mode);
    let mut total_t = 0usize;
    let mut total_b = 0usize;
    let mut keep = 0usize;
    for keep_idx in 0..chunks.len() {
        let ch = &chunks[keep_idx];
        if total_t + ch.trees.len() > ct || total_b + ch.bushes.len() > cb {
            break;
        }
        total_t += ch.trees.len();
        total_b += ch.bushes.len();
        keep += 1;
    }
    chunks.truncate(keep);
    TreeBuildResult { chunks, trees: total_t, bushes: total_b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_2400x1600() -> WorldModel {
        let cfg = alife_common::WorldGenConfig {
            seed: 1337,
            width: 2400,
            height: 1600,
            cell: 8,
            terrain_scale: 1.0,
            detail: 4,
            warp: 0.35,
            moisture_scale: 0.8,
            resource_density: 1.0,
            hazard_amount: 0.4,
            fertility_contrast: 1.0,
            hazard_pulse_period: 0.0,
        };
        alife_worldgen::generate(&cfg)
    }

    fn build_at(mode: u8, fx: f32, fz: f32, world: &WorldModel, scale: f32) -> TreeBuildResult {
        build(
            world,
            &TreeBuilder {
                gen: world.generation_tag,
                cell: ((fx / CHUNK) as i32, (fz / CHUNK) as i32),
                mode,
                focus: (fx, fz),
                lod_scale: scale,
            },
        )
    }

    #[test]
    fn placement_is_deterministic() {
        let world = world_2400x1600();
        let a = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        let c = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        assert_eq!(a.trees, c.trees);
        assert_eq!(a.bushes, c.bushes);
        assert_eq!(a.chunks.len(), c.chunks.len());
        for (ca, cb) in a.chunks.iter().zip(c.chunks.iter()) {
            assert_eq!((ca.cx, ca.cy), (cb.cx, cb.cy));
            assert_eq!(ca.trees.len(), cb.trees.len());
            for (ta, tb) in ca.trees.iter().zip(cb.trees.iter()) {
                assert_eq!(ta.x, tb.x);
                assert_eq!(ta.z, tb.z);
                assert_eq!(ta.yaw, tb.yaw);
                assert_eq!(ta.hue, tb.hue);
                assert_eq!(ta.scale, tb.scale);
            }
            for (ta, tb) in ca.bushes.iter().zip(cb.bushes.iter()) {
                assert_eq!(ta.x, tb.x);
                assert_eq!(ta.hue, tb.hue);
            }
        }
    }

    #[test]
    fn nothing_underwater() {
        let world = world_2400x1600();
        let res = build_at(MODE_PLANET, world.width * 0.5, world.height * 0.5, &world, 1.0);
        let mut seen = 0usize;
        for ch in res.chunks.iter() {
            for t in &ch.trees {
                assert!(
                    world.sample_elevation(t.x, t.z) >= WATER_E - 1e-4,
                    "tree at elevation {} below cutoff {}",
                    world.sample_elevation(t.x, t.z),
                    WATER_E
                );
                seen += 1;
            }
            for t in &ch.bushes {
                assert!(world.sample_elevation(t.x, t.z) >= WATER_E - 1e-4);
                seen += 1;
            }
        }
        assert!(seen > 0, "sane world should host some vegetation");
    }

    #[test]
    fn no_overlaps_within_min_dist() {
        let world = world_2400x1600();
        let res = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        for ch in &res.chunks {
            for (i, a) in ch.trees.iter().enumerate() {
                for other in ch.trees.iter().skip(i + 1) {
                    let d2 = (a.x - other.x).powi(2) + (a.z - other.z).powi(2);
                    assert!(
                        d2 >= SPACING_TREE * SPACING_TREE - 1e-3,
                        "overlapping trees d {:.3}",
                        d2.sqrt()
                    );
                }
            }
            for (i, a) in ch.bushes.iter().enumerate() {
                for other in ch.bushes.iter().skip(i + 1) {
                    let d2 = (a.x - other.x).powi(2) + (a.z - other.z).powi(2);
                    assert!(
                        d2 >= SPACING_BUSH * SPACING_BUSH - 1e-3,
                        "overlapping bushes d {:.3}",
                        d2.sqrt()
                    );
                }
                for t in &ch.trees {
                    let d2 = (a.x - t.x).powi(2) + (a.z - t.z).powi(2);
                    assert!(d2 >= SPACING_BUSH_TREE * SPACING_BUSH_TREE - 1e-3);
                }
            }
        }
    }

    #[test]
    fn planet_budget_2400x1600() {
        let world = world_2400x1600();
        let res = build_at(MODE_PLANET, world.width * 0.5, world.height * 0.5, &world, 1.0);
        assert!(res.trees <= CAP_TREE_PLANET, "trees {}", res.trees);
        assert!(res.bushes <= CAP_BUSH_PLANET, "bushes {}", res.bushes);
        // tri math: LOD-A tree worst-case 96 tris, bush 12 tris (mission
        // budget: <= ~150k extra tris)
        let tris = res.trees * crate::tree_mesh::TRIS_LOD_A as usize
            + res.bushes * crate::tree_mesh::TRIS_BUSH as usize;
        assert!(tris <= 150_000, "tree tri budget exceeded: {}", tris);
    }

    #[test]
    fn instance_height_ranges() {
        let world = world_2400x1600();
        let res = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        let mut trees_seen = 0;
        let mut bush_seen = 0;
        for ch in &res.chunks {
            for a in &ch.trees {
                let h = a.scale * NOM_TREE_H;
                assert!((5.5..=9.5).contains(&h), "tree height {}", h);
                assert_eq!(a.kind, 0.0);
                assert!((0.29..=0.62).contains(&a.hue), "hue {}", a.hue);
                assert!((0.0..std::f32::consts::TAU + 0.01).contains(&a.yaw));
                assert_eq!(a.dim, 1.0, "orbit focus is inside the near/mid ring");
                trees_seen += 1;
            }
            for a in &ch.bushes {
                let h = a.scale * NOM_BUSH_H;
                assert!((0.9..=1.5).contains(&h), "bush height {}", h);
                assert_eq!(a.kind, 1.0);
                bush_seen += 1;
            }
        }
        assert!(trees_seen > 0 || bush_seen > 0);
    }

    #[test]
    fn tree_heights_mesh_nominals_agree() {
        assert!((NOM_TREE_H - 7.8).abs() < 1e-6);
        assert!((NOM_BUSH_H - 1.0).abs() < 1e-6);
    }
}
