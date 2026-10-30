//! Deterministic vegetation placement + species rules (pure CPU, testable).
//! Everything is derived from the real `WorldModel` fields only: fertility,
//! moisture, hazard, elevation, biome id. No invented biome inputs.

use crate::scene::CameraMode;
use alife_worldgen::WorldModel;
use bytemuck::{Pod, Zeroable};

pub const CHUNK: f32 = 32.0;
/// hard instance budget (<= 45k blades, <= 300k triangles at 6 tris/blade)
pub const MAX_BLADES: usize = 45_000;
pub const TRIS_PER_BLADE: usize = 6;

/// jittered candidate attempts per 32x32 chunk (before density drop)
const CAND_HABITAT: u32 = 1200;
const CAND_PLANET: u32 = 380;

/// window + ring radii (near full density, mid 0.4x, beyond = none)
const WINDOW_ORBIT: f32 = 260.0;
const NEAR_ORBIT: f32 = 105.0;
const WINDOW_FOLLOW: f32 = 90.0;
const WINDOW_PLANET_MARGIN: f32 = 1.05;
const NEAR_PLANET: f32 = 220.0;
const MID_PLANET: f32 = 650.0;

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

pub fn window_for(mode: u8, world: &WorldModel) -> f32 {
    match mode {
        MODE_PLANET => (world.width.max(world.height) * WINDOW_PLANET_MARGIN).max(32.0),
        MODE_FOLLOW => WINDOW_FOLLOW,
        _ => WINDOW_ORBIT,
    }
}

/// Density rings: near full density, mid 0.4x, beyond = none.
pub fn rings_for(mode: u8, window: f32) -> (f32, f32) {
    match mode {
        MODE_FOLLOW => (WINDOW_FOLLOW * 0.5, WINDOW_FOLLOW * 0.5),
        MODE_PLANET => (NEAR_PLANET, MID_PLANET),
        _ => (NEAR_ORBIT, window * 0.5),
    }
}

/// Per-instance vegetation record (48 bytes, all-f32, bytemuck Pod).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GrassInst {
    pub x: f32,
    pub z: f32,
    /// blade height in world units (species + fertility driven)
    pub height: f32,
    /// blade width in world units
    pub width: f32,
    /// hue in [0,1)
    pub tint_hue: f32,
    /// lean/wind phase in [0,1)
    pub lean_phase: f32,
    /// biome id of the root cell
    pub biome: f32,
    /// ring LOD dim: 1.0 near ring, 0.55 mid ring
    pub dim: f32,
    /// extra deterministic per-blade variation seed
    pub seed: f32,
    /// underwater wetness factor (shore algae ring)
    pub wet: f32,
    /// species wind-stiffness multiplier (dry tufts stiff, grass 1.0)
    pub sway: f32,
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
pub fn species_rule(f_in: f32, m_in: f32, hz_in: f32, e_in: f32, biome: u32) -> Rule {
    let f = clamp01(f_in);
    let m = clamp01(m_in);
    let hz = clamp01(hz_in);
    let e = clamp01(e_in);
    let b = (biome % 6) as usize;

    let mul = BIOME_DENS_MUL[b];
    let mut rule = Rule {
        density: clamp01(f * (0.30 + 0.70 * m) * mul),
        h0: lerp(1.4, 4.4, f * 0.6 + m * 0.4),
        h1: lerp(2.0, 5.6, f * 0.6 + m * 0.4),
        w0: 0.07,
        w1: 0.12,
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
            rule.h0 = 1.1;
            rule.h1 = 2.2;
            rule.hue0 = 0.17;
            rule.hue1 = 0.26;
            rule.w0 = 0.05;
            rule.w1 = 0.09;
            rule.sway = 0.7;
        }
        3 => {
            // Toxic Flats: sparse sickly yellow-green, needs some fertility
            rule.h0 = 1.2;
            rule.h1 = 2.6;
            rule.hue0 = 0.52;
            rule.hue1 = 0.62;
            rule.sway = 0.8;
            rule.density *= clamp01(f * 2.5);
        }
        4 => {
            // Lowland Basin: damp darker marsh grass
            rule.h0 = 2.0;
            rule.h1 = 4.2;
            rule.hue0 = 0.30;
            rule.hue1 = 0.38;
            rule.wet = m.max(0.3) * 0.35;
        }
        _ => {
            // Niche Pocket: short wide mossy tufts
            rule.h0 = 0.9;
            rule.h1 = 1.7;
            rule.hue0 = 0.31;
            rule.hue1 = 0.40;
            rule.w0 = 0.13;
            rule.w1 = 0.20;
            rule.sway = 0.6;
        }
    }
    // moss patches: modest fertility + high moisture + low hazard
    if f >= 0.2 && f <= 0.4 && m >= 0.6 && hz < 0.2 && (b == 2 || b == 4 || b == 5) {
        rule.density = clamp01(rule.density + 0.35);
        rule.h0 = rule.h0.min(1.0);
        rule.h1 = rule.h1.min(1.6);
        rule.w0 = 0.15;
        rule.w1 = 0.22;
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
            rule.h0 = 0.7;
            rule.h1 = 1.7;
            rule.w0 = 0.09;
            rule.w1 = 0.14;
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
            density: 0.007,
            h0: 4.2,
            h1: 6.5,
            w0: 0.28,
            w1: 0.44,
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
            density: 0.006,
            h0: 2.8,
            h1: 4.4,
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

/// Candidate species pick: stiff tufts and shrubs claim rare hash slots,
/// everything else uses the field-driven rule.
pub fn pick_rule(f: f32, m: f32, hz: f32, e: f32, biome: u32, slot: f32) -> Rule {
    let t = clamp01(f);
    if let Some(r) = dry_tuft_override(m, e, hz) {
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

/// hash of (world generation tag, chunk coords, blade index) -> state stream
pub fn blade_hash(gen: u32, cx: i32, cy: i32, idx: u32) -> u64 {
    let mut s: u64 = 0x4742_5F53; // "GBS" salt
    s = (s << 8) | (gen as u64 & 0xFF);
    s ^= (cx as u32 as u64).wrapping_mul(0x9E37_79B9_3F7A_33A1);
    s ^= (cy as u32 as u64).rotate_left(17);
    s ^= (idx as u64).wrapping_mul(0xC2B2_AE3D_C6D4_BF0B);
    splitmix(&mut s)
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

pub fn build(world: &WorldModel, b: &Builder) -> BuildResult {
    let window = window_for(b.mode, world);
    let (near, mid) = rings_for(b.mode, window);
    let half = window * 0.5;
    let cands = if b.mode == MODE_PLANET { CAND_PLANET } else { CAND_HABITAT };

    let x0 = (b.focus.0 - half).clamp(0.0, world.width.max(CHUNK));
    let y0 = (b.focus.1 - half).clamp(0.0, world.height.max(CHUNK));
    let c0x = ((x0 / CHUNK).floor() as i32).max(0);
    let c0y = ((y0 / CHUNK).floor() as i32).max(0);
    let c1x = (((x0 + window) / CHUNK).ceil() as i32).max(((world.width / CHUNK).ceil() as i32).min(c0x + 96));
    let c1y = (((y0 + window) / CHUNK).ceil() as i32).max(((world.height / CHUNK).ceil() as i32).min(c0y + 96));

    let mut chunks: Vec<ChunkPlan> = Vec::new();
    for cy in c0y..=c1y {
        for cx in c0x..=c1x {
            let ox = cx as f32 * CHUNK;
            let oy = cy as f32 * CHUNK;
            if ox > x0 + window || oy > y0 + window {
                continue;
            }
            let mut insts = Vec::new();
            for i in 0..cands {
                let mut s = blade_hash(b.gen, cx, cy, i);
                let jx = hash01(&mut s);
                let jy = hash01(&mut s);
                let slot = hash01(&mut s);
                let rh = hash01(&mut s);
                let rw = hash01(&mut s);
                let rc = hash01(&mut s);
                let rl = hash01(&mut s);
                let px = (ox + jx * CHUNK).clamp(0.0, world.width - 0.001);
                let pz = (oy + jy * CHUNK).clamp(0.0, world.height - 0.001);

                let f = world.sample_fertility(px, pz);
                let m = world.sample_moisture(px, pz);
                let hz = world.sample_hazard(px, pz);
                let e = world.sample_elevation(px, pz);
                let bio = world.biome_id(px, pz) as u32;

                let rule = pick_rule(f, m, hz, e, bio, slot);
                let d = ((px - b.focus.0).powi(2) + (pz - b.focus.1).powi(2)).sqrt();
                let rate = ring_rate(d, near, mid) * rule.density * b.lod_scale;
                if rate <= f32::EPSILON || rh > rate {
                    continue;
                }
                insts.push(GrassInst {
                    x: px,
                    z: pz,
                    height: lerp(rule.h0, rule.h1, rh),
                    width: lerp(rule.w0, rule.w1, rw),
                    tint_hue: lerp(rule.hue0, rule.hue1, rc),
                    lean_phase: rl,
                    biome: bio as f32,
                    dim: if d < near { 1.0 } else { 0.55 },
                    seed: jx,
                    wet: rule.wet,
                    sway: rule.sway,
                    pad: 0.0,
                });
            }
            if !insts.is_empty() {
                chunks.push(ChunkPlan { cx, cy, insts });
            }
        }
    }
    // deterministic budget trim: nearest chunks to the focus win
    let (fcx, fcz) = b.focus;
    chunks.sort_by(|a, b| {
        let da = (a.cx as f32 * CHUNK - fcx).hypot(a.cy as f32 * CHUNK - fcz);
        let db = (b.cx as f32 * CHUNK - fcx).hypot(b.cy as f32 * CHUNK - fcz);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut total = 0usize;
    let mut keep = 0usize;
    for c in &chunks {
        if total + c.insts.len() > MAX_BLADES {
            break;
        }
        total += c.insts.len();
        keep += 1;
    }
    chunks.truncate(keep);
    BuildResult { chunks, total }
}
