//! Deterministic vegetation placement + species rules (pure CPU, testable).
//! Everything is derived from the real `WorldModel` fields only: fertility,
//! moisture, hazard, elevation, biome id. No invented biome inputs.
//!
//! PLACEMENT TARGET: crossed-plane CLUMPS (GPU Gems style). Each instance
//! anchors a 7-blade clump (~56 triangles), so budgets count CLUMPS, not
//! single blades. `x`/`z` are the clump anchor, `height` is the nominal
//! blade base height (0.7..3.2 units; rare meadow outlyer patches go
//! 3.5..5.0), `width` the blade width, `lean_phase` the clump yaw phase
//! and `pad` a clump shape variant. Slope and height-variance gates keep
//! clumps off cliffs and local dips that would bury the blades.

use crate::scene::CameraMode;
use crate::terrain::HEIGHT_SCALE;
#[cfg(test)]
use alife_worldgen::WorldModel;
use alife_worldgen::WorldSampling;
use bytemuck::{Pod, Zeroable};

pub const CHUNK: f32 = 32.0;
/// hard clump budget (<= 9k clumps resident; also sized the GPU buffer)
pub const MAX_BLADES: usize = 9_000;
/// triangles per CLUMP now (7 blades x ~8 tris); keep the name for callers
pub const TRIS_PER_BLADE: usize = 56;
/// total triangle budget the deterministic chunk trim must honor
const TRIS_BUDGET: usize = 300_000;
/// per-mode clump caps enforced by the deterministic nearest-first trim:
/// planet holds a slim margin under the hard triangle ceiling
/// (300_000 / 56 = 5,357 clumps), habitat windows land 2.5k..4k clumps.
const CLUMP_CAP_PLANET: usize = 5_300;
const CLUMP_CAP_HABITAT: usize = 4_000;
/// effective instance ceiling for a mode
fn inst_cap(mode: u8) -> usize {
    let by_tri = TRIS_BUDGET / TRIS_PER_BLADE;
    if mode == MODE_PLANET {
        CLUMP_CAP_PLANET.min(by_tri)
    } else {
        CLUMP_CAP_HABITAT.min(by_tri)
    }
}

/// jittered clump-anchor candidate attempts per 32x32 chunk (before density drop)
const CAND_HABITAT: u32 = 340;
const CAND_PLANET: u32 = 140;
/// directional keep factors: the habitat window is small and dense, the
/// planet window sweeps the whole world so its directional keep drops hard
/// to hold the worst-case tri budget. Per fertile chunk this keeps the
/// ~110-130 clump target in the lush near ring.
const HABITAT_KEEP_MUL: f32 = 2.2;
const PLANET_KEEP_MUL: f32 = 0.99;

/// window + ring radii (near full density, mid 0.4x, beyond = none)
const WINDOW_ORBIT: f32 = 260.0;
const NEAR_ORBIT: f32 = 105.0;
const WINDOW_FOLLOW: f32 = 90.0;
const WINDOW_PLANET_MARGIN: f32 = 1.05;
const NEAR_PLANET: f32 = 220.0;
const MID_PLANET: f32 = 650.0;

/// anti-overlap radius between clump anchors inside one chunk
const CLUMP_SPACING: f32 = 2.2;

// ------------------------------------------------------------- streaming
/// streaming: how far clump placement may reach BEYOND the simulated bounds
/// so vegetation continues across the horizon. Mirrors the terrain plan's
/// streamed margin ring hard cap (EXTRA_CHUNKS_CAP = 8 terrain chunks of
/// 64 units), so both horizons align; every out-of-reach field read goes
/// through the deterministic `VirtualWorld` view (see `WorldSampling`).
const GRASS_MARGIN: f32 = 512.0;

pub const MODE_ORBIT: u8 = 0;
pub const MODE_FOLLOW: u8 = 1;
pub const MODE_PLANET: u8 = 2;

pub fn mode_of(cm: &CameraMode) -> u8 {
    match cm {
        CameraMode::Orbit { .. } => MODE_ORBIT,
        CameraMode::Follow { .. } => MODE_FOLLOW,
        CameraMode::Planet => MODE_PLANET,
    }
}

pub fn window_for(mode: u8, world: &impl WorldSampling) -> f32 {
    match mode {
        MODE_PLANET => (world.width().max(world.height()) * WINDOW_PLANET_MARGIN).max(32.0),
        MODE_FOLLOW => WINDOW_FOLLOW,
        _ => WINDOW_ORBIT,
    }
}

/// Density rings: near full density, mid 0.4x, beyond = none. The orbit
/// mid ring reaches 0.72x the window so habitat mode keeps a healthy
/// clump count (2500..4000) even over lean biomes.
pub fn rings_for(mode: u8, window: f32) -> (f32, f32) {
    match mode {
        MODE_FOLLOW => (WINDOW_FOLLOW * 0.5, WINDOW_FOLLOW * 0.5),
        MODE_PLANET => (NEAR_PLANET, MID_PLANET),
        _ => (NEAR_ORBIT, window * 0.72),
    }
}

/// Per-clump vegetation record (48 bytes, all-f32, bytemuck Pod).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GrassInst {
    /// CLUMP ANCHOR in world coords
    pub x: f32,
    pub z: f32,
    /// nominal blade base height in world units (species/fertility driven)
    pub height: f32,
    /// blade width in world units
    pub width: f32,
    /// hue in [0,1)
    pub tint_hue: f32,
    /// clump yaw phase in [0,1)
    pub lean_phase: f32,
    /// biome id of the root cell
    pub biome: f32,
    /// ring LOD dim: 1.0 near ring, 0.60 mid ring
    pub dim: f32,
    /// extra deterministic per-clump variation seed
    pub seed: f32,
    /// underwater wetness factor (shore algae ring)
    pub wet: f32,
    /// species wind-stiffness multiplier (dry tufts stiff, grass 1.0)
    pub sway: f32,
    /// clump shape variant in [-0.5, 0.5]
    pub pad: f32,
}

pub struct Builder {
    pub gen: u32,
    /// cache cell this build is attached to (for identity only)
    pub cell: (i32, i32),
    pub mode: u8,
    /// camera focus in world coords (drives LOD rings)
    pub focus: (f32, f32),
    /// adaptive density multiplier (0.25..=1.0), folded into drop rate
    pub lod_scale: f32,
}

pub struct ChunkPlan {
    pub cx: i32,
    pub cy: i32,
    pub insts: Vec<GrassInst>,
}

pub struct BuildResult {
    pub chunks: Vec<ChunkPlan>,
    pub total: usize,
}

// ---------------------------------------------------------------- species

pub struct Rule {
    /// keep probability at 1.0x ring density
    pub density: f32,
    pub h0: f32,
    pub h1: f32,
    pub w0: f32,
    pub w1: f32,
    pub hue0: f32,
    pub hue1: f32,
    pub wet: f32,
    pub sway: f32,
}

const BIOME_DENS_MUL: [f32; 6] = [1.25, 0.80, 0.20, 0.30, 1.00, 0.55];

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Species rule from the real fields (elevation `e` is the raw 0..1 field).
/// Heights are now nominal BLADE heights 0.7..3.2 units (clump outlyers go
/// taller via [`meadow_outlyer`]).
pub fn species_rule(f_in: f32, m_in: f32, hz_in: f32, e_in: f32, biome: u32) -> Rule {
    let f = clamp01(f_in);
    let m = clamp01(m_in);
    let hz = clamp01(hz_in);
    let e = clamp01(e_in);
    let b = (biome % 6) as usize;

    let mul = BIOME_DENS_MUL[b];
    let mut rule = Rule {
        density: clamp01(f * (0.30 + 0.70 * m) * mul),
        // most grass reads low against the models: short sward,
        // fertile-damp ground tops out around 3.2 units
        h0: lerp(0.7, 1.7, f * 0.6 + m * 0.4),
        h1: lerp(1.1, 3.2, f * 0.6 + m * 0.4),
        w0: 0.05,
        w1: 0.10,
        hue0: 0.24,
        hue1: 0.34,
        wet: if m > 0.7 { (m - 0.7) * 0.6 } else { 0.0 },
        sway: 1.0,
    };
    match b {
        0 => {
            // Fertile Bloom: tall dense grass
            rule.hue0 = 0.24;
            rule.hue1 = 0.33;
        }
        1 => {
            // Sparse Plains: medium, drier look
            rule.h0 *= 0.78;
            rule.h1 *= 0.78;
            rule.hue0 = 0.27;
            rule.hue1 = 0.36;
        }
        2 => {
            // Rocky Highlands: sparse sage tufts
            rule.h0 = 0.6;
            rule.h1 = 1.4;
            rule.hue0 = 0.17;
            rule.hue1 = 0.26;
            rule.w0 = 0.04;
            rule.w1 = 0.07;
            rule.sway = 0.7;
        }
        3 => {
            // Toxic Flats: sparse sickly yellow-green, needs some fertility
            rule.h0 = 0.8;
            rule.h1 = 1.9;
            rule.hue0 = 0.52;
            rule.hue1 = 0.62;
            rule.sway = 0.8;
            rule.density *= clamp01(f * 2.5);
        }
        4 => {
            // Lowland Basin: damp darker marsh grass
            rule.h0 = 1.1;
            rule.h1 = 3.0;
            rule.hue0 = 0.30;
            rule.hue1 = 0.38;
            rule.wet = m.max(0.3) * 0.35;
        }
        _ => {
            // Niche Pocket: short wide mossy tufts
            rule.h0 = 0.6;
            rule.h1 = 1.2;
            rule.hue0 = 0.31;
            rule.hue1 = 0.40;
            rule.w0 = 0.09;
            rule.w1 = 0.14;
            rule.sway = 0.6;
        }
    }
    // moss patches: modest fertility + high moisture + low hazard
    if f >= 0.2 && f <= 0.4 && m >= 0.6 && hz < 0.2 && (b == 2 || b == 4 || b == 5) {
        rule.density = clamp01(rule.density + 0.35);
        rule.h0 = rule.h0.min(0.7);
        rule.h1 = rule.h1.min(1.2);
        rule.w0 = 0.11;
        rule.w1 = 0.17;
        rule.hue0 = 0.33;
        rule.hue1 = 0.41;
        rule.sway = rule.sway.min(0.5);
    }
    // dry conditions shift hue toward yellow-brown
    if m < 0.35 {
        let t = ((0.35 - m) * 0.9).min(0.8);
        rule.hue0 = lerp(rule.hue0, 0.12, t);
        rule.hue1 = lerp(rule.hue1, 0.16, t);
    }
    // hazard keeps cells bare: sharp curve to zero
    rule.density *= if hz > 0.15 {
        clamp01((0.35 - hz) / 0.20)
    } else {
        1.0
    };
    // shoreline algal ring: sparse wet tufts at the sea edge
    if e < 0.42 {
        if e < 0.36 {
            rule.density = 0.0;
        } else {
            let depth = (0.42 - e) / 0.06;
            rule.density = clamp01((0.22 + 0.55 * m) * 0.5);
            rule.h0 = 0.5;
            rule.h1 = 1.2;
            rule.w0 = 0.07;
            rule.w1 = 0.11;
            rule.hue0 = 0.40;
            rule.hue1 = 0.52;
            rule.sway = 0.55;
            rule.wet = clamp01(depth) * 0.9;
        }
    }
    rule
}

/// Occasional second-pass species: shrubs on fertile-but-dry ground.
pub fn shrub_override(f: f32, m: f32, hz: f32) -> Option<Rule> {
    if f > 0.55 && m < 0.35 && hz < 0.25 {
        Some(Rule {
            density: 0.005,
            h0: 3.4,
            h1: 5.8,
            w0: 0.20,
            w1: 0.30,
            hue0: 0.12,
            hue1: 0.20,
            wet: 0.0,
            sway: 0.75,
        })
    } else {
        None
    }
}

/// Sparse stiff dry tufts (cactus-like) on hot dry ground.
pub fn dry_tuft_override(m: f32, e: f32, hz: f32) -> Option<Rule> {
    if m < 0.20 && e > 0.50 && hz < 0.25 {
        Some(Rule {
            density: 0.0035,
            h0: 2.2,
            h1: 3.6,
            w0: 0.04,
            w1: 0.07,
            hue0: 0.11,
            hue1: 0.15,
            wet: 0.0,
            sway: 0.25,
        })
    } else {
        None
    }
}

/// MEADOW OUTLYER: rare tall broad patch on high-fertility mid-moisture
/// land in the lush biomes only (0/1/4). Reads as a lone taller sward
/// island standing out of the surrounding lawn.
pub fn meadow_outlyer(f: f32, m: f32, hz: f32, e: f32, biome: u32) -> Option<Rule> {
    let b = biome % 6;
    if (b == 0 || b == 1 || b == 4) && f > 0.60 && m > 0.40 && m < 0.72 && hz < 0.18 && e >= 0.42 {
        Some(Rule {
            density: 0.003,
            h0: 3.5,
            h1: 5.0,
            w0: 0.12,
            w1: 0.16,
            hue0: 0.27,
            hue1: 0.31,
            wet: 0.0,
            sway: 0.9,
        })
    } else {
        None
    }
}

/// Candidate species pick: stiff tufts and shrubs claim rare hash slots,
/// the meadow outlyer claims its own rare slot, everything else uses the
/// field-driven rule.
pub fn pick_rule(f: f32, m: f32, hz: f32, e: f32, biome: u32, slot: f32) -> Rule {
    let t = clamp01(f);
    if let Some(r) = dry_tuft_override(m, e, hz) {
        if slot < r.density {
            return r;
        }
    }
    if let Some(r) = meadow_outlyer(t, m, hz, e, biome) {
        if slot < r.density {
            return r;
        }
    }
    if let Some(r) = shrub_override(t, m, hz) {
        if slot < r.density {
            return r;
        }
    }
    species_rule(t, m, hz, e, biome)
}

// ------------------------------------------------------------------- hash

/// splitmix64 — deterministic placement hash
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// uniform float in [0,1) from the running stream
pub(crate) fn hash01(seed: &mut u64) -> f32 {
    (splitmix(seed) >> 40) as f32 / 0x0100_0000_u64 as f32
}

#[cfg(test)]
pub fn hash01_test(seed: &mut u64) -> f32 {
    hash01(seed)
}

/// corner hash for value-noise grids (deterministic, world-tagged)
fn corner_hash(gen: u32, cx: i32, cz: i32, salt: u64) -> f32 {
    let mut s: u64 = 0x4742_5F56; // "GBV" salt
    s = (s << 8) | (gen as u64 & 0xFF);
    s ^= (cx as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= (cz as i64 as u64).rotate_left(21);
    s ^= salt;
    splitmix(&mut s);
    hash01(&mut s)
}

fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// spatially coherent value noise in [0,1] on a square lattice of `cell` units.
/// Drives clumpy vegetation patches so the sward breaks into irregular
/// clusters with gaps, instead of a uniform full-density carpet.
fn vnoise(gen: u32, px: f32, pz: f32, cell: f32, salt: u64) -> f32 {
    let x = px / cell;
    let z = pz / cell;
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = smootherstep(x - x0 as f32);
    let tz = smootherstep(z - z0 as f32);
    let h00 = corner_hash(gen, x0, z0, salt);
    let h10 = corner_hash(gen, x0 + 1, z0, salt);
    let h01 = corner_hash(gen, x0, z0 + 1, salt);
    let h11 = corner_hash(gen, x0 + 1, z0 + 1, salt);
    lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), tz)
}

/// clumpiness multiplier in [0,1]: three noise octaves. A THIRD coarse octave
/// (~110 unit cells) adds biome-scale structure so whole MEADOWS rise above
/// the reference while wide CLEARINGS read almost bare inside otherwise
/// fertile ground; the mid (38) and fine (9) octaves carve the familiar
/// clumps and gaps. Low values carve gaps in dense areas; high values keep
/// fertile lawns lush.
pub fn patch_mask(gen: u32, px: f32, pz: f32) -> f32 {
    let meadow = vnoise(gen, px, pz, 110.0, 0x3333);
    let coarse = vnoise(gen, px, pz, 38.0, 0x1111);
    let fine = vnoise(gen, px, pz, 9.0, 0x2222);
    let mixed = 0.30 * meadow + 0.40 * coarse + 0.30 * fine;
    // map so most lawn stays but true bald patches appear
    (0.15 + 0.85 * mixed).clamp(0.0, 1.0)
}

/// hash of (world generation tag, chunk coords, clump index) -> state stream
pub fn blade_hash(gen: u32, cx: i32, cy: i32, idx: u32) -> u64 {
    let mut s: u64 = 0x4742_5F53; // "GBS" salt
    s = (s << 8) | (gen as u64 & 0xFF);
    s ^= (cx as u32 as u64).wrapping_mul(0x9E37_79B9_3F7A_33A1);
    s ^= (cy as u32 as u64).rotate_left(17);
    s ^= (idx as u64).wrapping_mul(0xC2B2_AE3D_C6D4_BF0B);
    splitmix(&mut s)
}

// ------------------------------------------------------------- grounding

/// Ground fit test for a clump anchor: samples the terrain elevation a
/// horizontal stride apart and rejects (a) visually steep slopes
/// (normalized rise over the stride > 0.55) and (b) local dips whose
/// height variance (> 3.0 world units over 4.0 units) would bury the
/// blades. Returns the normalized slope in [0,1] when the anchor is fine.
pub fn clump_slope_norm(world: &impl WorldSampling, px: f32, pz: f32) -> Option<f32> {
    const STRIDE: f32 = 2.0;
    let dhx = (world.sample_elevation(px + STRIDE, pz) - world.sample_elevation(px - STRIDE, pz))
        * HEIGHT_SCALE;
    let dhz = (world.sample_elevation(px, pz + STRIDE) - world.sample_elevation(px, pz - STRIDE))
        * HEIGHT_SCALE;
    let drop = dhx.abs().max(dhz.abs()).min(dhx.abs() + dhz.abs());
    // grade over the full horizontal span (2*STRIDE per axis)
    let slope_norm = clamp01(drop / (STRIDE * 2.0));
    if slope_norm > 0.55 || drop > 3.0 {
        None
    } else {
        Some(slope_norm)
    }
}

// ------------------------------------------------------------------ build

/// ring density multiplier for a candidate (density LOD folded into drop rate)
pub fn ring_rate(d: f32, near: f32, mid: f32) -> f32 {
    if d < near {
        1.0
    } else if d < mid {
        0.4
    } else {
        0.0
    }
}

pub fn build(world: &impl WorldSampling, b: &Builder) -> BuildResult {
    let window = window_for(b.mode, world);
    let (near, mid) = rings_for(b.mode, window);
    let half = window * 0.5;
    let cands = if b.mode == MODE_PLANET { CAND_PLANET } else { CAND_HABITAT };
    // planet mode sweeps the widest window, so its directional keep drops
    let mode_mul = if b.mode == MODE_PLANET { PLANET_KEEP_MUL } else { HABITAT_KEEP_MUL };

    // streaming: the placement window may hang over the simulated bounds by
    // at most GRASS_MARGIN on each side (was clamped into the world rect);
    // candidate coords clamp to (-margin, width+margin) instead of
    // (0, width) so vegetation continues to the streamed terrain edge while
    // staying bounded (never unbounded growth)
    let x0 = (b.focus.0 - half).clamp(-GRASS_MARGIN, world.width() + GRASS_MARGIN);
    let y0 = (b.focus.1 - half).clamp(-GRASS_MARGIN, world.height() + GRASS_MARGIN);
    let c0x = ((x0 / CHUNK).floor() as i32).max(-(GRASS_MARGIN / CHUNK) as i32);
    let c0y = ((y0 / CHUNK).floor() as i32).max(-(GRASS_MARGIN / CHUNK) as i32);
    let c1x = (((x0 + window) / CHUNK).ceil() as i32)
        .max((((world.width() + GRASS_MARGIN) / CHUNK).ceil() as i32).min(c0x + 96));
    let c1y = (((y0 + window) / CHUNK).ceil() as i32)
        .max((((world.height() + GRASS_MARGIN) / CHUNK).ceil() as i32).min(c0y + 96));

    let mut chunks: Vec<ChunkPlan> = Vec::new();
    for cy in c0y..=c1y {
        for cx in c0x..=c1x {
            let ox = cx as f32 * CHUNK;
            let oy = cy as f32 * CHUNK;
            if ox > x0 + window || oy > y0 + window {
                continue;
            }
            let mut insts = Vec::new();
            // anti-overlap: clump anchors placed in this chunk so far
            let mut placed: Vec<(f32, f32)> = Vec::new();
            for i in 0..cands {
                let mut s = blade_hash(b.gen, cx, cy, i);
                let jx = hash01(&mut s);
                let jy = hash01(&mut s);
                let slot = hash01(&mut s);
                let rh = hash01(&mut s);
                let rw = hash01(&mut s);
                let rc = hash01(&mut s);
                let rl = hash01(&mut s);
                // streaming: clamped to the streamed margin instead of the
                // simulated bounds; VirtualWorld sampling keeps these
                // deterministic and continuous with the engine fields
                let px = (ox + jx * CHUNK).clamp(-GRASS_MARGIN, world.width() + GRASS_MARGIN);
                let pz = (oy + jy * CHUNK).clamp(-GRASS_MARGIN, world.height() + GRASS_MARGIN);

                let f = world.sample_fertility(px, pz);
                let m = world.sample_moisture(px, pz);
                let hz = world.sample_hazard(px, pz);
                let e = world.sample_elevation(px, pz);
                let bio = world.biome_id(px, pz) as u32;

                let rule = pick_rule(f, m, hz, e, bio, slot);
                let d = ((px - b.focus.0).powi(2) + (pz - b.focus.1).powi(2)).sqrt();
                let rate = ring_rate(d, near, mid)
                    * rule.density
                    * b.lod_scale
                    * mode_mul
                    // global density trim + patch clustering (gaps/ tufts)
                    * 0.62
                    * (0.12 + 0.88 * patch_mask(b.gen, px, pz));
                if rate <= f32::EPSILON || rh > rate {
                    continue;
                }
                // terrain fit: no clumps on bluffs or in dips that bury blades
                if clump_slope_norm(world, px, pz).is_none() {
                    continue;
                }
                // deterministic anti-overlap against anchors placed in
                // this chunk (linear scan over a small vec is fine at
                // ~340 candidates / ~120 keeps per chunk)
                let (ax, az) = (px, pz);
                let mut ok = true;
                for (qx, qz) in &placed {
                    let ddx = ax - *qx;
                    let ddz = az - *qz;
                    if ddx * ddx + ddz * ddz < CLUMP_SPACING * CLUMP_SPACING {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                placed.push((ax, az));
                // consume one more hash for the per-clump shape variant so
                // tint (rc) and shape stay independent
                let rshape = hash01(&mut s);
                insts.push(GrassInst {
                    x: ax,
                    z: az,
                    height: lerp(rule.h0, rule.h1, rh),
                    width: lerp(rule.w0, rule.w1, rw),
                    tint_hue: lerp(rule.hue0, rule.hue1, rc),
                    lean_phase: rl,
                    biome: bio as f32,
                    dim: if d < near { 1.0 } else { 0.60 },
                    seed: jx,
                    wet: rule.wet,
                    sway: rule.sway,
                    pad: rshape - 0.5,
                });
            }
            if !insts.is_empty() {
                chunks.push(ChunkPlan { cx, cy, insts });
            }
        }
    }
    // deterministic budget trim: nearest chunks to the focus win, order
    // preserved so the near ring is never sacrificed to the cap
    let (fcx, fcz) = b.focus;
    chunks.sort_by(|a, b| {
        let da = (a.cx as f32 * CHUNK - fcx).hypot(a.cy as f32 * CHUNK - fcz);
        let db = (b.cx as f32 * CHUNK - fcx).hypot(b.cy as f32 * CHUNK - fcz);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = inst_cap(b.mode);
    let mut total = 0usize;
    let mut keep = 0usize;
    for c in &chunks {
        if total + c.insts.len() > cap {
            break;
        }
        total += c.insts.len();
        keep += 1;
    }
    chunks.truncate(keep);
    BuildResult { chunks, total }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_world() -> WorldModel {
        alife_worldgen::generate(&alife_worldgen::WorldGenConfig::default())
    }

    fn build_at(mode: u8, fx: f32, fz: f32, world: &WorldModel, scale: f32) -> BuildResult {
        build(
            world,
            &Builder {
                gen: world.generation_tag,
                cell: ((fx / CHUNK) as i32, (fz / CHUNK) as i32),
                mode,
                focus: (fx, fz),
                lod_scale: scale,
            },
        )
    }

    #[test]
    fn placement_is_deterministic_local() {
        let world = default_world();
        let a = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        let c = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        assert_eq!(a.total, c.total);
        assert_eq!(a.chunks.len(), c.chunks.len());
        for (ca, cb) in a.chunks.iter().zip(c.chunks.iter()) {
            assert_eq!(ca.cx, cb.cx);
            assert_eq!(ca.cy, cb.cy);
            assert_eq!(ca.insts.len(), cb.insts.len());
            for (ia, ib) in ca.insts.iter().zip(cb.insts.iter()) {
                assert_eq!(ia.x, ib.x);
                assert_eq!(ia.z, ib.z);
                assert_eq!(ia.pad, ib.pad);
            }
        }
    }

    #[test]
    fn clump_budget_planet_and_tri_cap() {
        let world = default_world();
        let res = build_at(MODE_PLANET, world.width * 0.5, world.height * 0.5, &world, 1.0);
        assert!(res.total <= inst_cap(MODE_PLANET), "planet clump cap exceeded: {}", res.total);
        assert!(
            res.total * TRIS_PER_BLADE <= TRIS_BUDGET,
            "planet tri budget exceeded: {}",
            res.total * TRIS_PER_BLADE
        );
    }

    #[test]
    fn habitat_window_clumps_in_range() {
        let world = default_world();
        for (fx, fz) in [(900.0, 700.0), (300.0, 300.0), (1600.0, 1200.0)] {
            let res = build_at(MODE_ORBIT, fx, fz, &world, 1.0);
            assert!(
                (2500..=4000).contains(&res.total),
                "habitat window clumps out of range at ({fx},{fz}): {}",
                res.total
            );
        }
    }

    /// tuning probe (run explicitly): cargo test -p alife-view3d probe_clump_counts -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_clump_counts() {
        let world = default_world();
        let planet = build_at(MODE_PLANET, world.width * 0.5, world.height * 0.5, &world, 1.0);
        println!(
            "planet total {} tris {} chunks {}",
            planet.total,
            planet.total * TRIS_PER_BLADE,
            planet.chunks.len()
        );
        for (fx, fz) in [(900.0, 700.0), (300.0, 300.0), (1600.0, 1200.0)] {
            let r = build_at(MODE_ORBIT, fx, fz, &world, 1.0);
            let mut max = 0usize;
            for c in &r.chunks {
                max = max.max(c.insts.len());
            }
            let mut fsum = 0.0;
            let mut msum = 0.0;
            let mut esum = 0.0;
            let n = 25usize;
            for yy in 0..n {
                for xx in 0..n {
                    let x = fx - WINDOW_ORBIT * 0.5 + (xx as f32 / (n - 1) as f32) * WINDOW_ORBIT;
                    let z = fz - WINDOW_ORBIT * 0.5 + (yy as f32 / (n - 1) as f32) * WINDOW_ORBIT;
                    let (x, z) = (x.clamp(0.0, world.width - 1.0), z.clamp(0.0, world.height - 1.0));
                    fsum += world.sample_fertility(x, z);
                    msum += world.sample_moisture(x, z);
                    esum += world.sample_elevation(x, z);
                }
            }
            let nn = (n * n) as f32;
            println!(
                "orbit ({fx},{fz}) total {} tris {} chunks {} maxchunk {} f {:.2} m {:.2} e {:.2}",
                r.total,
                r.total * TRIS_PER_BLADE,
                r.chunks.len(),
                max,
                fsum / nn,
                msum / nn,
                esum / nn
            );
        }
        let f = build_at(MODE_FOLLOW, 900.0, 700.0, &world, 1.0);
        println!("follow total {} tris {}", f.total, f.total * TRIS_PER_BLADE);
    }

    #[test]
    fn slope_gate_and_overlap() {
        let world = default_world();
        // every kept clump must respect the anti-overlap radius inside a chunk
        let res = build_at(MODE_ORBIT, 900.0, 700.0, &world, 1.0);
        for c in &res.chunks {
            let list = &c.insts;
            for (i, a) in list.iter().enumerate() {
                for other in list.iter().skip(i + 1) {
                    let d2 = (a.x - other.x).powi(2) + (a.z - other.z).powi(2);
                    assert!(
                        d2 >= CLUMP_SPACING * CLUMP_SPACING - 1e-3,
                        "overlapping clumps at d2 {:.3}",
                        d2
                    );
                }
            }
        }
        // grounding fn: flat fill world always passes, monotone ramp rejects
        let g = |v: f32| alife_worldgen::FieldGrid::new(4, 4, v);
        let flat = WorldModel {
            width: 96.0,
            height: 96.0,
            cell: 1.0,
            elevation: g(0.8),
            moisture: g(0.5),
            temperature: g(0.5),
            fertility: g(0.9),
            hazard: g(0.0),
            biome: g(0.0),
            generation_tag: 7,
            hazard_pulse_period: 0.0,
        };
        assert!(clump_slope_norm(&flat, 48.0, 48.0).is_some());
    }

    #[test]
    fn meadow_outlyer_matches_spec() {
        // lush biomes 0/1/4 on high-fert mid-moist land -> rare tall patch
        let r = meadow_outlyer(0.8, 0.55, 0.0, 0.8, 0).unwrap();
        assert_eq!(r.density, 0.003);
        assert_eq!(r.h0, 3.5);
        assert_eq!(r.h1, 5.0);
        assert_eq!(r.w0, 0.12);
        assert_eq!(r.w1, 0.16);
        assert_eq!(r.hue0, 0.27);
        assert_eq!(r.hue1, 0.31);
        // wrong biomes / extremes reject
        assert!(meadow_outlyer(0.8, 0.55, 0.0, 0.8, 2).is_none());
        assert!(meadow_outlyer(0.8, 0.55, 0.0, 0.8, 3).is_none());
        assert!(meadow_outlyer(0.8, 0.55, 0.0, 0.8, 5).is_none());
        assert!(meadow_outlyer(0.8, 0.9, 0.0, 0.8, 0).is_none());
        assert!(meadow_outlyer(0.8, 0.1, 0.0, 0.8, 0).is_none());
        assert!(pick_rule(0.8, 0.55, 0.0, 0.8, 0, 0.001).h0 >= 3.5);
        assert!(pick_rule(0.8, 0.55, 0.0, 0.8, 0, 0.9).h1 < 3.3);
    }

    #[test]
    fn patch_mask_has_third_coarse_octave() {
        // the 110-unit meadow octave must actually move the mask
        let mut same = 0;
        for i in 0..64 {
            let x = i as f32 * 17.0;
            let a = patch_mask(3, x, x * 0.7);
            let b = vnoise(3, x, x * 0.7, 38.0, 0x1111);
            if (a - b).abs() < 1e-4 {
                same += 1;
            }
        }
        assert!(same < 64, "patch_mask ignores the meadow octave");
    }
}
