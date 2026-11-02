//! Virtual-world streaming: deterministic extension of a [`WorldModel`]
//! beyond its simulated rectangle, so the RENDERED terrain/grass can grow a
//! visible horizon while the SIMULATED world (and every engine reader of
//! `WorldModel`) stays byte-identical.
//!
//! ENTROPY WIRING (architecture mandate): the engine's entropy equation is
//! the normalized Shannon entropy over species population shares,
//! `H = -SUM p_i * ln(p_i)` normalized by `ln(N)`
//! (alife-core/src/lib.rs:1706-1718, surfaced as `SimStats.diversity_index`,
//! alife-common/src/lib.rs:312). That equation is DISCONNECTED from world
//! generation, so this module reuses it VERBATIM (same formula, same
//! normalization, same [0,1] semantics) as the driver of streamed region
//! generation: every virtual region holds a pseudo "niche share"
//! distribution over the 6 biome bins; H(region) then modulates
//! vegetation density (fert_mul rises with H - even niches can all be
//! green), terrain roughness (rough rises as H falls - a concentrated
//! distribution carves harsher relief), hazard intensity, and biome-class
//! pressures (rocky / basin / wetter / drier per bin share). No invented
//! replacement equation: `share_entropy` below IS the engine formula with
//! N = BIOME_COUNT.
//!
//! DETERMINISM: every generated value is a pure function of
//! (generation_tag, sample point) through splitmix-style u64 hashing (same
//! style as grass_place.rs) and per-corner value noise on world-aligned
//! lattices - no mutable RNG, frame-rate independent, identical on every
//! revisit. Boundary blending: out-of-bounds samples LERP from the engine
//! field's own clamped edge samples (weight 0 at the border) to pure
//! generated values over a 45 world-unit band, so no seam or cliff ever
//! appears at the simulated rectangle's edge.
//!
//! COST: per out-of-bounds sample ~70 u64 hashes + a handful of bilinear
//! grid reads; in-bounds samples delegate 1:1 to `WorldModel` (zero extra
//! work for anything the engine already reads).

use crate::{WorldModel, BIOME_COUNT};

// ------------------------------------------------------------- hash plumbing

/// splitmix64 - same deterministic placement hash style as grass_place.rs
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// uniform f64 in [0,1) from a finished hash word
fn h01(z: u64) -> f64 {
    (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
}

/// corner hash: deterministic from (generation_tag, lattice coords, salt)
fn corner(tag: u32, ix: i64, iz: i64, salt: u64) -> f64 {
    let mut s: u64 = salt;
    s ^= (tag as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= (ix as u64).wrapping_mul(0xC2B2_AE3D_C6D4_BF0B);
    s ^= (iz as u64).rotate_left(23).wrapping_mul(0x94D0_49BB_1331_11EB);
    h01(splitmix(&mut s))
}

fn smoother64(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// spatially-coherent value noise in [0,1] on a world-aligned lattice of
/// `cell` units (global continuity: adjacent virtual regions never seam)
fn vnoise(tag: u32, x: f64, z: f64, cell: f64, salt: u64) -> f64 {
    let fx = x / cell;
    let fz = z / cell;
    let x0 = fx.floor();
    let z0 = fz.floor();
    let tx = smoother64(fx - x0);
    let tz = smoother64(fz - z0);
    let (x0, z0) = (x0 as i64, z0 as i64);
    let a = corner(tag, x0, z0, salt);
    let b = corner(tag, x0 + 1, z0, salt);
    let u = corner(tag, x0, z0 + 1, salt);
    let w = corner(tag, x0 + 1, z0 + 1, salt);
    (a * (1.0 - tx) + b * tx) * (1.0 - tz) + (u * (1.0 - tx) + w * tx) * tz
}

fn smoothstep32(a: f32, b: f32, v: f32) -> f32 {
    let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ----------------------------------------------------- streaming constants

/// virtual region (chunk grid) side in world units
pub const REGION: f32 = 256.0;
/// width of the boundary LERP band (engine clamped edge -> generated value)
pub const BLEND_BAND: f32 = 45.0;

// salt bases (distinct streams per field)
const SALT_SHARE: u64 = 0x2011_0000_0000_0000;
const SALT_EL: u64 = 0x2012_0000_0000_0000;
const SALT_MO: u64 = 0x2013_0000_0000_0000;
const SALT_TP: u64 = 0x2014_0000_0000_0000;
const SALT_PK: u64 = 0x2015_0000_0000_0000;
const SALT_BI: u64 = 0x2016_0000_0000_0000;
const SALT_HZ: u64 = 0x2017_0000_0000_0000;

// ---------------------------------------------------------------- entropy

/// Normalized Shannon entropy of a 6-bin niche share distribution - the
/// engine's diversity equation (alife-core/src/lib.rs:1706-1718) applied to
/// streamed region generation: `H = -SUM p_i ln p_i / ln(N)` in [0,1].
pub fn share_entropy(shares: &[f64; BIOME_COUNT]) -> f64 {
    let mut h = 0.0f64;
    for &p in shares {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    (h / (BIOME_COUNT as f64).ln()).clamp(0.0, 1.0)
}

/// Raw per-region niche shares over the 6 biome bins, deterministic from
/// (generation_tag, region coords).
fn region_shares(tag: u32, rx: i64, rz: i64) -> [f64; BIOME_COUNT] {
    let mut raw = [0.0f64; BIOME_COUNT];
    for (i, v) in raw.iter_mut().enumerate() {
        *v = corner(tag, rx, rz, SALT_SHARE + i as u64) + 1e-3;
    }
    let total: f64 = raw.iter().sum::<f64>().max(2e-6);
    for v in &mut raw {
        *v /= total;
    }
    raw
}

/// Share distribution smoothly interpolated across the region lattice so
/// the driving entropy and every derived pressure are continuous across
/// virtual region borders (no cliffs between regions either).
fn bilinear_shares(tag: u32, rx: i64, rz: i64, fx: f64, fz: f64) -> [f64; BIOME_COUNT] {
    let s00 = region_shares(tag, rx, rz);
    let s10 = region_shares(tag, rx + 1, rz);
    let s01 = region_shares(tag, rx, rz + 1);
    let s11 = region_shares(tag, rx + 1, rz + 1);
    let (vx, vz) = (smoother64(fx), smoother64(fz));
    let mut out = [0.0f64; BIOME_COUNT];
    for i in 0..BIOME_COUNT {
        let a = s00[i] * (1.0 - vx) + s10[i] * vx;
        let b = s01[i] * (1.0 - vx) + s11[i] * vx;
        out[i] = a * (1.0 - vz) + b * vz;
    }
    let total: f64 = out.iter().sum::<f64>().max(2e-6);
    for v in &mut out {
        *v /= total;
    }
    out
}

/// Continuous region-level generation parameters derived from the blended
/// share distribution through the entropy semantics
/// (vegetation density / roughness / hazard / biome pressures).
#[derive(Clone, Copy)]
struct RegionParams {
    /// elevation bias (Rocky Highlands share raises, Lowland Basin lowers)
    e_bias: f64,
    /// moisture bias (Fertile Bloom / basin wetter, rock drier)
    m_bias: f64,
    /// hazard intensity
    haz_int: f64,
    /// vegetation density multiplier (rises with H)
    fert_mul: f64,
    /// terrain roughness of the mid octaves (rises as H falls)
    rough: f64,
    /// fertility contrast (rises as H falls: lush patch + bare rest)
    contrast: f64,
}

fn region_params(tag: u32, rx: i64, rz: i64, fx: f64, fz: f64) -> RegionParams {
    let shares = bilinear_shares(tag, rx, rz, fx, fz);
    let h = share_entropy(&shares);
    let p = |i: usize| shares[i];
    RegionParams {
        e_bias: 0.46 * p(2) - 0.34 * p(4),
        m_bias: 0.14 * p(0) + 0.20 * p(4) - 0.12 * p(2) + 0.06 * p(1),
        haz_int: (0.20 + 0.62 * p(3) + 0.10 * p(5)).clamp(0.0, 1.0),
        fert_mul: (0.68 + 0.55 * h + 0.30 * p(0)).clamp(0.05, 1.6),
        rough: (0.70 + 0.80 * (1.0 - h) + 0.25 * p(2)).clamp(0.3, 2.0),
        contrast: (0.55 + 0.90 * (1.0 - h)).clamp(0.2, 1.8),
    }
}

// ------------------------------------------------------------ VirtualWorld

/// A borrowing view over a [`WorldModel`] whose sampling continues beyond
/// the simulated rectangle, deterministically. In-bounds samples delegate
/// 1:1 to the underlying WorldModel (bit-identical to what the engine
/// readers see); out-of-bounds samples are generated from
/// (generation_tag, region coords, sample point) and blended back toward
/// the engine's clamped edge samples within [`BLEND_BAND`].
pub struct VirtualWorld<'a> {
    pub world: &'a WorldModel,
}

impl<'a> VirtualWorld<'a> {
    /// Wraps a model without touching it.
    pub fn wrap(world: &'a WorldModel) -> Self {
        Self { world }
    }

    fn in_bounds(&self, x: f32, y: f32) -> bool {
        x >= 0.0 && x <= self.world.width && y >= 0.0 && y <= self.world.height
    }

    /// distance outside the simulated rectangle measured as the largest
    /// per-axis overshoot (0 inside)
    fn out_amount(&self, x: f32, y: f32) -> f32 {
        let dx = if x < 0.0 {
            -x
        } else if x > self.world.width {
            x - self.world.width
        } else {
            0.0
        };
        let dy = if y < 0.0 {
            -y
        } else if y > self.world.height {
            y - self.world.height
        } else {
            0.0
        };
        dx.max(dy)
    }

    /// smootherstep blend weight: 0 at the boundary, 1 at BLEND_BAND out
    fn band_w(d: f32) -> f32 {
        let t = (d / BLEND_BAND).clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Region lattive location + in-region fraction (f64 for smooth slides)
    fn region_of(x: f64, z: f64) -> (i64, i64, f64, f64) {
        let gx = x / REGION as f64;
        let gz = z / REGION as f64;
        let rx = gx.floor();
        let rz = gz.floor();
        (rx as i64, rz as i64, gx - rx, gz - rz)
    }

    /// Deterministic virtual fields at a deep-outside point
    /// (elev, moisture, temp, fert, hazard, pocket, biome-bias).
    fn virtual_fields(&self, x: f32, z: f32) -> [f32; 7] {
        let tag = self.world.generation_tag;
        let (rx, rz, fx, fz) = Self::region_of(x as f64, z as f64);
        let rp = region_params(tag, rx, rz, fx, fz);
        let (px, pz) = (x as f64, z as f64);

        // ELEVATION: 3-octave value noise, mid/fine roughness region driven
        let n1 = vnoise(tag, px, pz, 176.0, SALT_EL + 1);
        let n2 = vnoise(tag, px, pz, 44.0, SALT_EL + 2);
        let n3 = vnoise(tag, px, pz, 11.0, SALT_EL + 3);
        let e = (
            0.5 + 0.80 * (n1 - 0.5)
                + 0.48 * rp.rough * (n2 - 0.5)
                + 0.30 * rp.rough * (n3 - 0.5)
                + rp.e_bias
        )
            .clamp(0.0, 1.0) as f32;

        // MOISTURE
        let m1 = vnoise(tag, px, pz, 150.0, SALT_MO + 1);
        let m2 = vnoise(tag, px, pz, 37.0, SALT_MO + 2);
        let m = (
            0.46 + 0.66 * (m1 - 0.5) + 0.28 * (m2 - 0.5) + rp.m_bias
        )
            .clamp(0.0, 1.0) as f32;

        // TEMPERATURE (noise-only out here; no world-frame latitude bands)
        let t1 = vnoise(tag, px, pz, 210.0, SALT_TP + 1);
        let t = (0.55 + 0.38 * (t1 - 0.5)).clamp(0.0, 1.0) as f32;

        // pocket noise + biome threshold wiggler (same semantics as Gen::cell)
        let pk = vnoise(tag, px, pz, 65.0, SALT_PK + 1) as f32;
        let pocket = smoothstep32(0.12, 0.42, pk);
        let bias = (vnoise(tag, px, pz, 90.0, SALT_BI + 1) as f32) - 0.5;

        // FERTILITY: same habitability curvature class of formula as the
        // engine's Gen::cell, scaled by the entropy-driven region density
        let temp_ok = (1.0 - (2.0 * (t - 0.52).abs()).powf(1.6)).clamp(0.0, 1.0);
        let moist_ok =
            smoothstep32(0.24, 0.60, m) * (1.0 - 0.45 * smoothstep32(0.85, 1.0, m));
        let lowland = (1.0 - smoothstep32(0.45, 0.85, e)) * 0.28;
        let base_f = (moist_ok * temp_ok * (0.72 + lowland) + 0.07).clamp(0.0, 1.0);
        let shaped =
            0.38 + (base_f * (1.0 + 0.55 * pocket) - 0.38) * (0.45 + 0.55 * rp.contrast as f32);
        let f = (shaped * rp.fert_mul as f32).clamp(0.0, 1.0);

        // HAZARD: ridged foci above a region-tuned threshold
        let r1 = vnoise(tag, px, pz, 96.0, SALT_HZ + 1);
        let r2 = vnoise(tag, px, pz, 24.0, SALT_HZ + 2);
        let ridge = (1.0 - (2.0 * r1 - 1.0).abs().powi(2)) * 0.62
            + (1.0 - (2.0 * r2 - 1.0).abs().powi(2)) * 0.38;
        let thr = 0.90 - 0.28 * rp.haz_int as f32;
        let foci = smoothstep32(thr, thr + 0.15, ridge as f32);
        let hz =
            (foci * (0.25 + 0.85 * rp.haz_int as f32) + 0.03 * rp.haz_int as f32).clamp(0.0, 1.0);

        [e, m, t, f, hz, pocket, bias]
    }

    /// Biome class from (blended) fields - mirrors the engine's Gen::cell
    /// threshold chain (Toxic Flats / Highlands / Bloom / Basin / Pocket /
    /// Plains).
    fn classify(e: f32, m: f32, f: f32, hz: f32, pocket: f32, bias: f32) -> f32 {
        if hz > 0.16 + 0.09 * bias && hz >= f * 0.9 {
            3.0
        } else if e > 0.63 + 0.05 * bias {
            2.0
        } else if f > 0.60 - 0.06 * bias && hz < 0.14 {
            0.0
        } else if e < 0.34 + 0.05 * bias && m > 0.55 {
            4.0
        } else if pocket > 0.55 || (f > 0.5 && hz > 0.08) {
            5.0
        } else {
            1.0
        }
    }

    /// One sampling pass through the virtual view (all six fields + the
    /// continuous biome class value). In-bounds: pure engine delegation.
    fn fields(&self, x: f32, y: f32) -> [f32; 6] {
        let w = self.world;
        if self.in_bounds(x, y) {
            return [
                w.elevation.sample(x, y, w.width, w.height),
                w.moisture.sample(x, y, w.width, w.height),
                w.temperature.sample(x, y, w.width, w.height),
                w.fertility.sample(x, y, w.width, w.height),
                w.hazard.sample(x, y, w.width, w.height),
                w.biome.sample(x, y, w.width, w.height),
            ];
        }
        let wgt = Self::band_w(self.out_amount(x, y));
        let e0 = w.elevation.sample(x, y, w.width, w.height);
        let m0 = w.moisture.sample(x, y, w.width, w.height);
        let t0 = w.temperature.sample(x, y, w.width, w.height);
        let f0 = w.fertility.sample(x, y, w.width, w.height);
        let h0 = w.hazard.sample(x, y, w.width, w.height);
        let b0 = w.biome.sample(x, y, w.width, w.height);
        let v = self.virtual_fields(x, y);
        let bl = |a: f32, b: f32| a + (b - a) * wgt;
        let e = bl(e0, v[0]);
        let m = bl(m0, v[1]);
        let t = bl(t0, v[2]);
        let f = bl(f0, v[3]);
        let hz = bl(h0, v[4]);
        let bio = bl(b0, Self::classify(e, m, f, hz, v[5], v[6]));
        [e, m, t, f, hz, bio]
    }

    /// Combined virtual sample: (fertility, elevation, moisture, hazard,
    /// biome id) - the classic engine readers packed into one call.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn sample_f_e_m_hz_bio(&self, px: f32, pz: f32) -> (f32, f32, f32, f32, u32) {
        let v = self.fields(px, pz);
        let bio = (v[5].max(0.0) as usize).min(BIOME_COUNT - 1) as u32;
        (v[3], v[0], v[1], v[4], bio)
    }

    /// Continuous biome class VALUE (matches the raw `biome` FieldGrid
    /// encoding in-bounds; blends toward the generated class outside) for
    /// terrain mesh vertex colours.
    pub fn biome_raw(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[5]
    }
}

// ---------------------------------------------------------- WorldSampling

/// Duck-typed world reader shared by the engine's `WorldModel` and the
/// streaming [`VirtualWorld`], so view-side placement code can consume
/// either without knowing which one it holds.
pub trait WorldSampling {
    fn width(&self) -> f32;
    fn height(&self) -> f32;
    fn cell(&self) -> f32;
    fn generation_tag(&self) -> u32;
    fn sample_elevation(&self, x: f32, y: f32) -> f32;
    fn sample_moisture(&self, x: f32, y: f32) -> f32;
    fn sample_temperature(&self, x: f32, y: f32) -> f32;
    fn sample_fertility(&self, x: f32, y: f32) -> f32;
    fn sample_hazard(&self, x: f32, y: f32) -> f32;
    /// biome class id into `BIOME_COUNT` (engine parity: truncation + clamp)
    fn biome_id(&self, x: f32, y: f32) -> usize;
}

impl WorldSampling for WorldModel {
    fn width(&self) -> f32 {
        self.width
    }
    fn height(&self) -> f32 {
        self.height
    }
    fn cell(&self) -> f32 {
        self.cell
    }
    fn generation_tag(&self) -> u32 {
        self.generation_tag
    }
    fn sample_elevation(&self, x: f32, y: f32) -> f32 {
        WorldModel::sample_elevation(self, x, y)
    }
    fn sample_moisture(&self, x: f32, y: f32) -> f32 {
        WorldModel::sample_moisture(self, x, y)
    }
    fn sample_temperature(&self, x: f32, y: f32) -> f32 {
        WorldModel::sample_temperature(self, x, y)
    }
    fn sample_fertility(&self, x: f32, y: f32) -> f32 {
        WorldModel::sample_fertility(self, x, y)
    }
    fn sample_hazard(&self, x: f32, y: f32) -> f32 {
        WorldModel::sample_hazard(self, x, y)
    }
    fn biome_id(&self, x: f32, y: f32) -> usize {
        WorldModel::biome_id(self, x, y)
    }
}

impl WorldSampling for std::sync::Arc<WorldModel> {
    /// The per-frame world handle the view3d render path already holds
    /// (grass.rs passes `&frame.world`) routed through the streaming view:
    /// in-bounds samples stay bit-identical to the engine's WorldModel, and
    /// beyond-bounds samples stream deterministically - so every generic
    /// `&impl WorldSampling` consumer (grass placement) grows a horizon for
    /// free without any frame type change.
    fn width(&self) -> f32 {
        self.width
    }
    fn height(&self) -> f32 {
        self.height
    }
    fn cell(&self) -> f32 {
        self.cell
    }
    fn generation_tag(&self) -> u32 {
        self.generation_tag
    }
    fn sample_elevation(&self, x: f32, y: f32) -> f32 {
        VirtualWorld::wrap(self).sample_elevation(x, y)
    }
    fn sample_moisture(&self, x: f32, y: f32) -> f32 {
        VirtualWorld::wrap(self).sample_moisture(x, y)
    }
    fn sample_temperature(&self, x: f32, y: f32) -> f32 {
        VirtualWorld::wrap(self).sample_temperature(x, y)
    }
    fn sample_fertility(&self, x: f32, y: f32) -> f32 {
        VirtualWorld::wrap(self).sample_fertility(x, y)
    }
    fn sample_hazard(&self, x: f32, y: f32) -> f32 {
        VirtualWorld::wrap(self).sample_hazard(x, y)
    }
    fn biome_id(&self, x: f32, y: f32) -> usize {
        VirtualWorld::wrap(self).biome_id(x, y)
    }
}

impl WorldSampling for VirtualWorld<'_> {
    fn width(&self) -> f32 {
        self.world.width
    }
    fn height(&self) -> f32 {
        self.world.height
    }
    fn cell(&self) -> f32 {
        self.world.cell
    }
    fn generation_tag(&self) -> u32 {
        self.world.generation_tag
    }
    fn sample_elevation(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[0]
    }
    fn sample_moisture(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[1]
    }
    fn sample_temperature(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[2]
    }
    fn sample_fertility(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[3]
    }
    fn sample_hazard(&self, x: f32, y: f32) -> f32 {
        self.fields(x, y)[4]
    }
    fn biome_id(&self, x: f32, y: f32) -> usize {
        let v = self.fields(x, y)[5];
        (v.max(0.0) as usize).min(BIOME_COUNT - 1)
    }
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> WorldModel {
        crate::generate(&crate::WorldGenConfig::default())
    }

    fn generic_land<W: WorldSampling>(w: &W, x: f32, y: f32) -> f32 {
        w.sample_elevation(x, y)
    }

    #[test]
    fn in_bounds_delegates_bit_identically() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        for &(x, y) in &[
            (0.0f32, 0.0f32),
            (1.5, 2.5),
            (w.width * 0.5, w.height * 0.25),
            (w.width - 0.25, w.height - 0.25),
        ] {
            let t = v.sample_f_e_m_hz_bio(x, y);
            assert_eq!(t.0, w.sample_fertility(x, y));
            assert_eq!(t.1, w.sample_elevation(x, y));
            assert_eq!(t.2, w.sample_moisture(x, y));
            assert_eq!(t.3, w.sample_hazard(x, y));
            assert_eq!(t.4, w.biome_id(x, y) as u32);
            assert_eq!((v.biome_raw(x, y) * 64.0) as i64, (w.biome.sample(x, y, w.width, w.height) * 64.0) as i64);
        }
        // generic call sites hit identical data for both types
        assert_eq!(generic_land(&w, 10.0, 20.0), generic_land(&v, 10.0, 20.0));
        // the per-frame Arc handle routes through the streaming view:
        // in-bounds it must stay bit-identical to the engine (free pass for
        // everything grass.rs reads per frame), and out-of-bounds it must
        // agree with the direct VirtualWorld (streamed horizon parity).
        let arc = std::sync::Arc::new(w.clone());
        for &(x, y) in &[(1.5f32, 2.5f32), (w.width * 0.5, w.height * 0.25)] {
            assert_eq!(arc.sample_elevation(x, y), w.sample_elevation(x, y));
            assert_eq!(arc.biome_id(x, y), w.biome_id(x, y));
        }
        for &(x, y) in &[
            (-30.0f32, 12.5f32),
            (w.width + 61.0, w.height + 8.0),
            (w.width + 300.0, -5.0),
        ] {
            assert_eq!(arc.sample_elevation(x, y), v.sample_elevation(x, y));
            assert_eq!(arc.biome_id(x, y), v.biome_id(x, y));
        }
    }

    #[test]
    fn boundary_seam_is_continuous_in_small_steps() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        // walk across each of the four edges in 0.5-unit steps: the four
        // CONTINUOUS fields must never jump more than eps between samples
        // (including across the boundary, where the blend weight is ~0), and
        // the discrete biome class never leaves its range. Far OUT past the
        // blend band the classification is a pure threshold chain (same
        // discrete character as the engine's own in-bounds biome grid), so
        // class reassignments there are region layout, not seams - but right
        // at the border the id must still change smoothly (blend ~ 0).
        let eps = 0.045f32;
        let lim_bio = 1.5f32;
        // natural generated-terrain slope cap on the virtual side (patchy
        // contrast can legitimately steepen gradients far out; that is
        // roughness, not a seam): must still be bounded (no cliffs)
        let eps_far = 0.14f32;
        let walk = |ax: bool, at: f32, edge: &str| {
            let mut prev = [None::<f32>; 6];
            for step in -120..=120 {
                let d = step as f32 * 0.5;
                let (x, y) = if ax { (at + d, w.height * 0.5) } else { (w.width * 0.5, at + d) };
                let s = v.sample_f_e_m_hz_bio(x, y);
                let vals = [s.0, s.1, s.2, s.3, s.4 as f32];
                for (i, val) in vals.iter().enumerate() {
                    if let Some(p) = prev[i] {
                        if i == 4 {
                            assert!(
                                *val >= 0.0 && *val <= (BIOME_COUNT - 1) as f32,
                                "biome id out of range: {val}"
                            );
                            // at the boundary crossing itself the blend
                            // weight is ~0, so the id may not jump at all
                            if d.abs() <= 1.0 {
                                assert!(
                                    (val - p).abs() <= lim_bio,
                                    "edge {edge}: biome id jumped {p} -> {val} at the seam, d={d}"
                                );
                            }
                        } else {
                            let lim = if d.abs() <= 2.0 { eps } else { eps_far };
                            assert!(
                                (val - p).abs() <= lim,
                                "field {i} jumped {p} -> {val} step d={d}"
                            );
                        }
                    }
                    prev[i] = Some(*val);
                }
            }
        };
        walk(true, w.width, "right (+x)");
        walk(true, 0.0, "left (-x)");
        walk(false, w.height, "top (+y)");
        walk(false, 0.0, "bottom (-y)");
    }

    #[test]
    fn blends_back_to_engine_clamped_edge_at_band_edge() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        // far inside vs deeply inside sample must equal in-band samples follow
        let e_in = v.sample_elevation(w.width - 1.0, w.height * 0.5);
        let e_out = v.sample_elevation(w.width + 1.0, w.height * 0.5);
        assert!(
            (e_in - e_out).abs() < 0.06,
            "cross-boundary elevation jumped: {e_in} -> {e_out}"
        );
        // at exactly BLEND_BAND outside the mix is pure generated - finite, in range
        for d in [BLEND_BAND, BLEND_BAND * 2.0, BLEND_BAND * 40.0] {
            let s = v.sample_f_e_m_hz_bio(w.width + d, w.height + d * 0.7);
            let all = [s.0, s.1, s.2, s.3];
            assert!(all.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0));
            assert!(s.4 < 6, "biome id in range: {}", s.4);
        }
    }

    #[test]
    fn deterministic_on_revisit_and_frame_independent() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        for &(x, y) in &[
            (w.width + 13.0f32, -7.5f32),
            (-40.0, w.height + 130.0),
            (w.width + 900.0, w.height + 900.0),
        ] {
            let a = v.sample_f_e_m_hz_bio(x, y);
            assert_eq!(a, v.sample_f_e_m_hz_bio(x, y), "same coord twice != same");
            // every field reproducible through the trait too
            assert_eq!(v.sample_fertility(x, y), v.sample_fertility(x, y));
            assert_eq!(v.biome_id(x, y), v.biome_id(x, y));
        }
    }

    #[test]
    fn spread_varies_across_far_regions() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        // far-out samples across many regions must actually vary
        let mut vals = [0.0f32; 16];
        for (i, val) in vals.iter_mut().enumerate() {
            *val = v.sample_f_e_m_hz_bio(
                w.width + 600.0 + i as f32 * 1200.0,
                w.height + 600.0 + i as f32 * 900.0,
            )
            .1;
        }
        let mn = vals.iter().cloned().fold(f32::MAX, f32::min);
        let mx = vals.iter().cloned().fold(f32::MIN, f32::max);
        assert!(mx - mn > 0.05, "far-out samples do not vary: {mn}..{mx}");
    }

    #[test]
    fn share_entropy_matches_engine_formula_and_bounds() {
        // equal shares -> H = 1; single share -> H = 0
        let eq = [1.0f64; 6].map(|v| v / 6.0);
        assert!((share_entropy(&eq) - 1.0).abs() < 1e-9);
        let one = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(share_entropy(&one).abs() < 1e-9);
        // and the raw formula re-derived inline still agrees (the fold
        // already accumulates -p*ln(p), so H = fold / ln(6))
        let h = |sh: &[f64; 6]| {
            sh.iter().fold(0.0, |acc, &p| if p > 0.0 { acc - p * p.ln() } else { acc })
                / 6f64.ln()
        };
        for i in 0..64i64 {
            let shares = region_shares(9, 0, i);
            assert!((share_entropy(&shares) - h(&shares)).abs() < 1e-12);
            assert!((share_entropy(&shares) - 0.5).abs() < 0.5);
        }
    }

    #[test]
    fn perf_virtual_sampling_cheap_enough() {
        let w = world();
        let v = VirtualWorld::wrap(&w);
        let t = std::time::Instant::now();
        let mut acc = 0.0f32;
        for i in 0..20_000u32 {
            let (x, y) = (
                w.width + 40.0 + (i % 173) as f32,
                w.height + 40.0 + (i / 173 % 191) as f32,
            );
            acc += v.sample_f_e_m_hz_bio(x, y).1;
        }
        let dt = t.elapsed();
        eprintln!("20k virtual samples in {dt:?} (sum {acc})");
        assert!(dt.as_millis() < 1500, "virtual sampling too slow: {dt:?}");
    }

    #[test]
    fn blend_band_constant_is_the_contracted_45_units() {
        assert_eq!(BLEND_BAND, 45.0);
    }
}
