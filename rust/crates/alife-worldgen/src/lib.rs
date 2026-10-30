//! Seeded procedural world generation.
//!
//! Contract: `generate(&WorldGenConfig) -> WorldModel`. Correlated fields:
//! elevation, moisture, temperature, fertility, hazard, biome index. All
//! randomness comes from seeded SuperSimplex noise (deterministic per cfg.seed).

use noise::{NoiseFn, SuperSimplex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use alife_common::WorldGenConfig;

pub const BIOME_COUNT: usize = 6;
pub const BIOME_NAMES: [&str; BIOME_COUNT] = [
    "Fertile Bloom",
    "Sparse Plains",
    "Rocky Highlands",
    "Toxic Flats",
    "Lowland Basin",
    "Niche Pocket",
];

/// Suggested base rgb color per biome id (renderer may refine).
pub const BIOME_COLORS: [[f32; 3]; BIOME_COUNT] = [
    [0.11, 0.40, 0.24],
    [0.30, 0.30, 0.35],
    [0.36, 0.32, 0.30],
    [0.44, 0.24, 0.19],
    [0.12, 0.24, 0.35],
    [0.23, 0.37, 0.40],
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldGrid {
    pub w: u32,
    pub h: u32,
    pub data: Vec<f32>,
}

impl FieldGrid {
    pub fn new(w: u32, h: u32, fill: f32) -> Self {
        Self { w, h, data: vec![fill; (w * h) as usize] }
    }
    /// Bilinear sample in WORLD coordinates over a world of (world_w, world_h).
    pub fn sample(&self, world_x: f32, world_y: f32, world_w: f32, world_h: f32) -> f32 {
        let gx = (world_x / world_w * self.w as f32 - 0.5).clamp(0.0, self.w as f32 - 0.001);
        let gy = (world_y / world_h * self.h as f32 - 0.5).clamp(0.0, self.h as f32 - 0.001);
        let x0 = gx as u32;
        let y0 = gy as u32;
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let at = |x: u32, y: u32| self.data[(y * self.w + x) as usize];
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }
}

/// The complete procedural world: correlated environmental fields on a grid
/// whose cell size is `cell` world units. Static structure; the simulation
/// layers dynamic state (foods, edits) on top.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldModel {
    pub width: f32,
    pub height: f32,
    pub cell: f32,
    pub elevation: FieldGrid,
    pub moisture: FieldGrid,
    pub temperature: FieldGrid,
    pub fertility: FieldGrid,
    pub hazard: FieldGrid,
    /// biome class id per cell (index into BIOME_NAMES)
    pub biome: FieldGrid,
    pub generation_tag: u32,
    /// > 0: world hazard intensity pulses with this period in seconds
    pub hazard_pulse_period: f32,
}

impl WorldModel {
    pub fn sample_elevation(&self, x: f32, y: f32) -> f32 {
        self.elevation.sample(x, y, self.width, self.height)
    }
    pub fn sample_moisture(&self, x: f32, y: f32) -> f32 {
        self.moisture.sample(x, y, self.width, self.height)
    }
    pub fn sample_temperature(&self, x: f32, y: f32) -> f32 {
        self.temperature.sample(x, y, self.width, self.height)
    }
    pub fn sample_fertility(&self, x: f32, y: f32) -> f32 {
        self.fertility.sample(x, y, self.width, self.height)
    }
    pub fn sample_hazard(&self, x: f32, y: f32) -> f32 {
        self.hazard.sample(x, y, self.width, self.height)
    }
    pub fn biome_id(&self, x: f32, y: f32) -> usize {
        let v = self.biome.sample(x, y, self.width, self.height);
        (v as usize).min(BIOME_COUNT - 1)
    }
    pub fn grid_w(&self) -> u32 {
        self.elevation.w
    }
    pub fn grid_h(&self) -> u32 {
        self.elevation.h
    }
}

// ============================================================ noise plumbing

/// fbm in roughly [-1, 1], lacunarity ~2.0, gain ~0.5.
fn fbm(n: &SuperSimplex, x: f64, y: f64, octaves: u32) -> f64 {
    let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
    for o in 0..octaves {
        let off = o as f64 * 57.31;
        sum += amp * n.get([x * freq + off, y * freq - off]);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Ridged multifractal in [0, 1]: sharp inverted |noise| ridges (branching
/// landforms, toxin veins).
fn ridged(n: &SuperSimplex, x: f64, y: f64, octaves: u32) -> f64 {
    let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
    for o in 0..octaves {
        let off = o as f64 * 91.7;
        let t = 1.0 - n.get([x * freq + off, y * freq - off]).abs();
        sum += amp * t * t;
        norm += amp;
        amp *= 0.5;
        freq *= 2.05;
    }
    sum / norm
}

fn smoothstep(a: f32, b: f32, v: f32) -> f32 {
    let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per-session noise instances + clamped config. Values in [0..N_PAIR^7*u32]:
/// seeds are cfg.seed * PRIME + k, distinct per field.
struct Gen {
    el: SuperSimplex,
    rd: SuperSimplex,
    wx: SuperSimplex,
    wy: SuperSimplex,
    mo: SuperSimplex,
    tp: SuperSimplex,
    pk: SuperSimplex,
    hz: SuperSimplex,
    bi: SuperSimplex,
    span: f32,       // terrain coordinate span across min(gw,gh)
    mspan: f32,      // moisture span
    oct: u32,        // 1..=6
    warp: f32,       // domain-warp strength (coordinate units)
    haz: f32,        // 0..1 hazard_amount
    haz_thr: f32,    // ridged threshold for hazard foci
    res: f32,        // resource_density
    contrast: f32,   // fertility_contrast 0..2
}

fn field_seed(seed: u32, k: u32) -> u32 {
    seed.wrapping_mul(0x9E3779B9).wrapping_add(0x5851F42Du32.wrapping_mul(k) ^ 0x6B2D)
}

impl Gen {
    fn build(cfg: &WorldGenConfig) -> Self {
        let sane = |v: f32, d: f32| -> f32 { if v.is_finite() { v } else { d } };
        let seed = cfg.seed;
        let haz = sane(cfg.hazard_amount, 0.35).clamp(0.0, 1.0);
        let contrast = sane(cfg.fertility_contrast, 1.0).clamp(0.0, 2.0);
        let res = sane(cfg.resource_density, 1.0).clamp(0.05, 5.0);
        let warp = sane(cfg.warp, 0.0).clamp(0.0, 1.0);
        let oct = sane(cfg.detail as f32, 4.0).clamp(1.0, 6.0) as u32;
        let span = 3.2 * sane(cfg.terrain_scale, 1.0).clamp(0.2, 3.0);
        let mspan = 3.0 * sane(cfg.moisture_scale, 1.0).clamp(0.2, 3.0);
        Self {
            el: SuperSimplex::new(field_seed(seed, 1)),
            rd: SuperSimplex::new(field_seed(seed, 2)),
            wx: SuperSimplex::new(field_seed(seed, 3)),
            wy: SuperSimplex::new(field_seed(seed, 4)),
            mo: SuperSimplex::new(field_seed(seed, 5)),
            tp: SuperSimplex::new(field_seed(seed, 6)),
            pk: SuperSimplex::new(field_seed(seed, 7)),
            hz: SuperSimplex::new(field_seed(seed, 8)),
            bi: SuperSimplex::new(field_seed(seed, 9)),
            span,
            mspan,
            oct,
            warp: warp * 0.9,
            haz,
            haz_thr: 0.90 - 0.28 * haz,
            res,
            contrast,
        }
    }

    /// All six fields for one cell.
    fn cell(&self, cx: u32, cy: u32, gw: u32, gh: u32) -> [f32; 6] {
        let (x, y) = (cx as f32, cy as f32);
        let s = self.span / (gw.min(gh)) as f32;
        // aspect-correct noise coordinates
        let nx = x * s;
        let ny = y * s;
        let q = self.oct;

        // domain warp: two low-freq fbm fields displace the sample point
        let dx = fbm(&self.wx, (nx * 0.5) as f64, (ny * 0.5) as f64, 2) as f32 * self.warp;
        let dy = fbm(&self.wy, (nx * 0.5 + 43.7) as f64, (ny * 0.5 + 17.3) as f64, 2) as f32 * self.warp;

        // ELEVATION = fbm base + ridged orography, both warped
        let e0 = (fbm(&self.el, (nx + dx) as f64, (ny + dy) as f64, q) * 0.5 + 0.5) as f32;
        let r = ridged(&self.rd, (nx * 0.85 + dx) as f64, (ny * 0.85 + dy) as f64, q.min(4)) as f32;
        let elev = (0.72 * e0 + 0.28 * r).clamp(0.0, 1.0);

        // TEMPERATURE = latitude banding (hot equatorial mid-band) + moderate noise
        let lat = (y / gh as f32 * 2.0 - 1.0).abs();
        let tgrad = (1.1 - 1.2 * lat).clamp(0.0, 1.0);
        let tn = (fbm(&self.tp, (nx * 1.35) as f64, (ny * 1.35) as f64, 2) * 0.5 + 0.5) as f32;
        let temp = (0.78 * tgrad + 0.22 * tn).clamp(0.0, 1.0);

        // MOISTURE = fbm(moisture_scale) biased wet in elevation basins
        let sm = self.mspan / (gw.min(gh)) as f32;
        let m0 = (fbm(&self.mo, ((x * sm + dx * 0.6)) as f64, ((y * sm + dy * 0.6)) as f64, q.min(4))
            * 0.5
            + 0.5) as f32;
        let moisture = (0.74 * m0 + (1.0 - elev) * 0.17 + 0.05).clamp(0.0, 1.0);

        // FERTILITY = moisture habitability * thermal habitability, lowland
        // bonus, plus pocket-noise archipelago of rich niche patches; shaped by
        // contrast and multiplied by resource_density.
        let temp_ok = (1.0 - (2.0 * (temp - 0.52).abs()).powf(1.6)).clamp(0.0, 1.0);
        let moist_ok = smoothstep(0.24, 0.60, moisture) * (1.0 - 0.45 * smoothstep(0.85, 1.0, moisture));
        let lowland = (1.0 - smoothstep(0.45, 0.85, elev)) * 0.28;
        let base_f = (moist_ok * temp_ok * (0.72 + lowland) + 0.07).clamp(0.0, 1.0);
        let pn = fbm(&self.pk, (nx * 2.6) as f64, (ny * 2.6) as f64, 2) as f32;
        let pocket = smoothstep(0.12, 0.42, pn);
        let shaped = 0.38 + (base_f * (1.0 + 0.55 * pocket) - 0.38) * (0.45 + 0.55 * self.contrast);
        let fert = (shaped * self.res).clamp(0.0, 1.0);

        // HAZARD = ridged toxin foci above a coverage threshold; else ~ambient 0.
        let hr = ridged(&self.hz, (nx * 1.9) as f64, (ny * 1.9) as f64, 3) as f32;
        let foci = smoothstep(self.haz_thr, self.haz_thr + 0.15, hr);
        let hazard = (foci * (0.25 + 0.85 * self.haz) + 0.03 * self.haz).clamp(0.0, 1.0);

        // BIAS noise wiggles biome thresholds => organic boundaries.
        let bias = fbm(&self.bi, (nx * 0.9) as f64, (ny * 0.9) as f64, 2) as f32;

        let biome = if hazard > 0.16 + 0.09 * bias && hazard >= fert * 0.9 {
            3.0 // Toxic Flats
        } else if elev > 0.63 + 0.05 * bias {
            2.0 // Rocky Highlands
        } else if fert > 0.60 - 0.06 * bias && hazard < 0.14 {
            0.0 // Fertile Bloom
        } else if elev < 0.34 + 0.05 * bias && moisture > 0.55 {
            4.0 // Lowland Basin
        } else if pocket > 0.55 || (fert > 0.5 && hazard > 0.08) {
            5.0 // Niche Pocket (rich/hazardous islands)
        } else {
            1.0 // Sparse Plains
        };

        [elev, moisture, temp, fert, hazard, biome]
    }
}

fn c01(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Generate a deterministic world: identical cfg => identical WorldModel.
pub fn generate(cfg: &WorldGenConfig) -> WorldModel {
    let cell = cfg.cell.max(1);
    let gw = (cfg.width / cell).max(8).min(2048);
    let gh = (cfg.height / cell).max(8).min(2048);
    let g = Gen::build(cfg);
    let n = (gw * gh) as usize;

    // Interleaved layout: [elev, moist, temp, fert, hz, biome] per cell.
    let mut raw = vec![0.0f32; n * 6];
    raw.par_chunks_mut(6).enumerate().for_each(|(i, out)| {
        let i = i as u32;
        let (x, y) = (i % gw, i / gw);
        let v = g.cell(x, y, gw, gh);
        for (o, &val) in v.iter().enumerate() {
            let val = if o == 5 { val } else { c01(val) };
            out[o] = if val.is_finite() { val } else { 0.0 };
        }
    });

    let grid = |k: usize| FieldGrid {
        w: gw,
        h: gh,
        data: (0..n).map(|i| raw[i * 6 + k]).collect(),
    };

    WorldModel {
        width: cfg.width.max(1) as f32,
        height: cfg.height.max(1) as f32,
        cell: cell as f32,
        elevation: grid(0),
        moisture: grid(1),
        temperature: grid(2),
        fertility: grid(3),
        hazard: grid(4),
        biome: grid(5),
        generation_tag: cfg.seed,
        hazard_pulse_period: cfg.hazard_pulse_period.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical() -> WorldGenConfig {
        WorldGenConfig {
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
        }
    }

    #[test]
    fn deterministic_normalized_and_diverse() {
        let cfg = canonical();
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.elevation.data, b.elevation.data);
        assert_eq!(a.hazard.data, b.hazard.data);
        assert_eq!(a.biome.data, b.biome.data);
        for g in [&a.elevation, &a.moisture, &a.temperature, &a.fertility, &a.hazard] {
            assert!(g.data.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
        }
        let mut kinds = [0u32; 6];
        for v in &a.biome.data {
            kinds[*v as usize] += 1;
        }
        assert!(kinds.iter().filter(|k| **k > 0).count() >= 4, "biomes present: {:?}", kinds);
        // hazard is mostly-zero with hotspots
        let hot = a.hazard.data.iter().filter(|v| **v > 0.2).count();
        assert!(hot > a.hazard.data.len() / 100 && hot < a.hazard.data.len() / 3);
        // different seeds differ visibly
        let mut c2 = cfg;
        c2.seed = 0xBEEF;
        assert_ne!(generate(&c2).elevation.data, a.elevation.data);
        assert_ne!(generate(&c2).biome.data, a.biome.data);
    }

    #[test]
    fn hostile_config_is_clamped_not_panicking() {
        let c = WorldGenConfig {
            seed: 7,
            width: 100,
            height: 80,
            cell: 0,
            terrain_scale: f32::NAN,
            detail: 99,
            warp: -3.0,
            moisture_scale: 0.0,
            resource_density: 50.0,
            hazard_amount: 2.0,
            fertility_contrast: -1.0,
            hazard_pulse_period: 0.0,
        };
        let w = generate(&c);
        for g in [&w.elevation, &w.moisture, &w.temperature, &w.fertility, &w.hazard] {
            assert!(g.data.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
        }
    }

    #[test]
    fn all_zero_default_still_valid() {
        // alife-common derives Default => all-zero config; must remain valid.
        let z = generate(&WorldGenConfig::default());
        assert!(z.elevation.data.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn perf_target_2400x1600_cell8() {
        let cfg = canonical();
        let t = std::time::Instant::now();
        let w = generate(&cfg);
        let dt = t.elapsed();
        eprintln!("gen time: {dt:?}, grid {}x{}", w.grid_w(), w.grid_h());
        if cfg!(debug_assertions) {
            assert!(dt.as_millis() < 4000);
        } else {
            assert!(dt.as_millis() < 400);
        }
    }
}
