//! Weather and climate dynamic layer.
//!
//! A coarse grid (world-grid downsampled ~8x) carries cloud cover, rain
//! intensity and a soil-moisture modifier (`wet` in [-1, 1]). Clouds advect
//! slowly with a fixed per-world wind; rain regions set by god events raise
//! the cloud layer; soil moisture follows rain up and evaporates down. The
//! layer updates at most once per `WEATHER_EVERY` steps (1 sim-second) and
//! never per frame.
//!
//! Documented coupling numbers (all deliberately small, no hidden boosts):
//! - rain regions lift cloud cover at 0.9/s toward 1.0 inside their radius
//! - drought effects dry clouds at 0.35/s and pull `wet` toward -0.85
//! - soil moisture relaxes toward its target at 0.12/s (soak/evaporation)
//! - cloud over 0.7 rains: rain => (cloud - 0.7) / 0.3, ramped at 0.6/s
//! - "wet cover" = fraction of coarse cells with rain > 0.1 or wet > 0.3
//!
//! Limitations (bounded by design): field edges clamp (no wrap), lightning
//! bolt placement and storm scheduling consume the engine RNG (see god
//! events), and expiry of region effects can lag one weather tick (<= 1
//! sim-second) behind its `until` step.

use alife_common::sanitize;
use alife_worldgen::WorldModel;
use serde::{Deserialize, Serialize};

/// Ambient cloud level everywhere else (slow relaxation target).
pub const CLOUD_BASE: f32 = 0.12;
/// Cloud cover above which rain starts.
pub const RAIN_CLOUD: f32 = 0.7;
/// /s relaxation of cloud cover toward ambient.
const AMBIENT_RELAX: f32 = 0.05;
/// /s lift of cloud cover inside rain/storm regions (scaled by soft edge).
const LIFT_RATE: f32 = 0.9;
/// /s cloud dissipation inside drought regions (scaled by soft edge).
const DRY_RATE: f32 = 0.35;
/// /s approach of rain intensity toward its cloud-derived target.
const RAIN_RATE: f32 = 0.6;
/// /s approach of soil moisture toward its target (soak/evaporation).
const WET_RATE: f32 = 0.12;
/// /s multiplicative decay of the visual lightning flash.
const FLASH_DECAY: f32 = 2.5;
/// Drought target soil-moisture modifier.
const DROUGHT_TARGET: f32 = -0.85;
/// Max catch-up window when a weather tick was skipped (bounded).
pub const MAX_DT: f32 = 30.0;
/// Bounded number of concurrent region effects (oldest dropped).
pub const EFFECTS_CAP: usize = 16;
/// Bounded number of fertility boost/burn scars.
pub const BOOSTS_CAP: usize = 32;
/// Bounded recent-bolt history.
pub const BOLTS_CAP: usize = 64;
/// Meteor/fire hazard energy drain per second inside the fire circle.
pub const FIRE_DPS: f32 = 30.0;

/// Kinds of timed region effects. `Fire` drains energy (weather-side hazard)
/// inside its full radius; `Rain`/`Storm` lift the cloud layer; `Drought`
/// dries clouds and soil.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectKind {
    Rain,
    Drought,
    Storm,
    Fire,
}

/// A timed circular weather effect (god intervention or catastrophe).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct RegionEffect {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub kind: EffectKind,
    /// absolute step at which the effect ends
    pub until: u64,
    /// storm only: lightning bolts still scheduled
    pub pending_bolts: u32,
    /// storm only: absolute step of the next scheduled bolt
    pub next_bolt_at: u64,
}

/// Local fertility modifier (ash = positive, burn = negative) with expiry.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FertBoost {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub amount: f32,
    pub until: u64,
}

/// The full weather state; serialized inside SaveCore for exact restores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherState {
    /// world span the coarse grid maps onto
    pub ww: f32,
    pub wh: f32,
    pub gw: u32,
    pub gh: u32,
    /// cloud cover per coarse cell, 0..1
    pub cloud: Vec<f32>,
    /// rain intensity per coarse cell, 0..1
    pub rain: Vec<f32>,
    /// soil moisture modifier per coarse cell, -1..1
    pub wet: Vec<f32>,
    /// wind drift in coarse cells per second
    pub wind_x: f32,
    pub wind_y: f32,
    /// decaying global flash pulse (lightning)
    pub flash: f32,
    /// last bolts: (x, y, step), bounded, oldest dropped
    pub recent_bolts: Vec<(f32, f32, u64)>,
    /// active region effects, bounded
    pub effects: Vec<RegionEffect>,
    /// fertility boost/burn scars, bounded
    pub fert_boosts: Vec<FertBoost>,
    /// last computed wet cover (fraction of coarse cells)
    pub cover: f32,
    /// scratch buffer for cloud advection (no per-tick allocation)
    buf_c: Vec<f32>,
}

impl WeatherState {
    /// Coarse grid = world grid downsampled ~8x (bounded 4..96 per axis).
    pub fn new(world: &WorldModel) -> Self {
        let gw = ((world.grid_w() as f32) / 8.0).round().max(4.0).min(96.0) as u32;
        let gh = ((world.grid_h() as f32) / 8.0).round().max(4.0).min(96.0) as u32;
        // gentle deterministic per-world wind from the generation tag
        let ang = world.generation_tag as f32 * 0.137;
        let n = (gw * gh) as usize;
        Self {
            ww: world.width.max(1.0),
            wh: world.height.max(1.0),
            gw,
            gh,
            cloud: vec![CLOUD_BASE; n],
            rain: vec![0.0; n],
            wet: vec![0.0; n],
            wind_x: 0.45 * ang.cos(),
            wind_y: 0.30 * ang.sin(),
            flash: 0.0,
            recent_bolts: Vec::new(),
            effects: Vec::new(),
            fert_boosts: Vec::new(),
            cover: 0.0,
            buf_c: vec![0.0; n],
        }
    }

    // ---------------------------------------------------------- sampling

    fn coord(&self, x: f32, y: f32) -> (f32, f32) {
        let u = (sanitize(x) / self.ww).clamp(0.0, 1.0) * (self.gw.max(2) - 1) as f32;
        let v = (sanitize(y) / self.wh).clamp(0.0, 1.0) * (self.gh.max(2) - 1) as f32;
        (u, v)
    }

    /// Bilinear sample of a coarse field at grid coordinates (clamped edges).
    fn sgrid(data: &[f32], gw: u32, gh: u32, u: f32, v: f32) -> f32 {
        let gm = gw.max(2);
        let gh2 = gh.max(2);
        let gx = u.clamp(0.0, (gm - 1) as f32);
        let gy = v.clamp(0.0, (gh2 - 1) as f32);
        let x0 = gx as u32;
        let y0 = gy as u32;
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let x1 = (x0 + 1).min(gm - 1);
        let y1 = (y0 + 1).min(gh2 - 1);
        let at = |x: u32, y: u32| data[(y * gm + x) as usize];
        let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Cloud cover at world coords in [0, 1].
    pub fn cloud(&self, x: f32, y: f32) -> f32 {
        let (u, v) = self.coord(x, y);
        Self::sgrid(&self.cloud, self.gw, self.gh, u, v).clamp(0.0, 1.0)
    }

    /// Rain intensity at world coords in [0, 1].
    pub fn rain(&self, x: f32, y: f32) -> f32 {
        let (u, v) = self.coord(x, y);
        Self::sgrid(&self.rain, self.gw, self.gh, u, v).clamp(0.0, 1.0)
    }

    /// Soil moisture modifier at world coords in [-1, 1] (rain = wet > 0).
    pub fn wet(&self, x: f32, y: f32) -> f32 {
        let (u, v) = self.coord(x, y);
        Self::sgrid(&self.wet, self.gw, self.gh, u, v).clamp(-1.0, 1.0)
    }

    /// Current global flash pulse in [0, 1].
    pub fn flash(&self) -> f32 {
        self.flash.clamp(0.0, 1.0)
    }

    /// Last lightning impacts: (x, y, step), bounded to [`BOLTS_CAP`].
    pub fn recent_bolts(&self) -> &[(f32, f32, u64)] {
        &self.recent_bolts
    }

    /// Wet cover fraction (last weather tick).
    pub fn wet_cover(&self) -> f32 {
        self.cover.clamp(0.0, 1.0)
    }

    // ------------------------------------------------------ region logic

    /// 1.0 inside 80% of the radius, fading to 0 at the rim (fire: binary).
    fn edge(&self, e: &RegionEffect, x: f32, y: f32) -> f32 {
        let dx = x - e.x;
        let dy = y - e.y;
        let d = (dx * dx + dy * dy).sqrt();
        let r = e.r.max(1.0);
        if d >= r {
            return 0.0;
        }
        if e.kind == EffectKind::Fire {
            return 1.0;
        }
        let t = ((d / r) - 0.8).clamp(0.0, 0.2) / 0.2;
        1.0 - t
    }

    /// Max lift influence from active Rain/Storm effects (0..1).
    pub fn cloud_lift(&self, x: f32, y: f32) -> f32 {
        let mut best = 0.0f32;
        for e in &self.effects {
            if e.kind == EffectKind::Rain || e.kind == EffectKind::Storm {
                let v = self.edge(e, x, y);
                if v > best {
                    best = v;
                }
            }
        }
        best
    }

    /// Max drought influence at a point (0..1).
    pub fn drought_edge(&self, x: f32, y: f32) -> f32 {
        let mut best = 0.0f32;
        for e in &self.effects {
            if e.kind == EffectKind::Drought {
                let v = self.edge(e, x, y);
                if v > best {
                    best = v;
                }
            }
        }
        best
    }

    /// Fire hazard energy drain per second at a point (0 or [`FIRE_DPS`]).
    pub fn fire_dps(&self, x: f32, y: f32) -> f32 {
        for e in &self.effects {
            if e.kind == EffectKind::Fire {
                let dx = x - e.x;
                let dy = y - e.y;
                if dx * dx + dy * dy <= e.r * e.r {
                    return FIRE_DPS;
                }
            }
        }
        0.0
    }

    /// Sum of local fertility boost/burnModifiers (ash > 0, burn < 0).
    pub fn fert_boost(&self, x: f32, y: f32) -> f32 {
        let mut s = 0.0f32;
        for b in &self.fert_boosts {
            let dx = x - b.x;
            let dy = y - b.y;
            if dx * dx + dy * dy <= b.r * b.r {
                s += b.amount;
            }
        }
        s.clamp(-3.0, 3.0)
    }

    // ----------------------------------------------------------- dynamics

    /// Advance the layer by `dt` sim-seconds. Expired effects are drained
    /// into `expired` so callers can clean anything linked to them (bounded).
    pub fn update(&mut self, dt: f32, step: u64, expired: &mut Vec<RegionEffect>) {
        let dtf = dt.clamp(0.001, MAX_DT);

        // expire region effects
        let mut i = 0;
        while i < self.effects.len() {
            if self.effects[i].until <= step {
                expired.push(self.effects.remove(i));
            } else {
                i += 1;
            }
        }
        // expire fertility scars
        let mut i = 0;
        while i < self.fert_boosts.len() {
            if self.fert_boosts[i].until <= step {
                self.fert_boosts.remove(i);
            } else {
                i += 1;
            }
        }

        let gw = self.gw.max(2);
        let gh = self.gh.max(2);
        let cell_w = self.ww / gw as f32;
        let cell_h = self.wh / gh as f32;

        // pass 1: advect clouds + relaxation + region lift/dry
        for gy in 0..gh {
            for gx in 0..gw {
                let idx = (gy * gw + gx) as usize;
                let u = gx as f32 + 0.5 - self.wind_x * dtf;
                let v = gy as f32 + 0.5 - self.wind_y * dtf;
                let mut c = Self::sgrid(&self.cloud, gw, gh, u, v);
                c += (CLOUD_BASE - c) * (AMBIENT_RELAX * dtf).min(1.0);
                let wx = (gx as f32 + 0.5) * cell_w;
                let wy = (gy as f32 + 0.5) * cell_h;
                let lift = self.cloud_lift(wx, wy);
                if lift > 0.0 {
                    c += (1.0 - c) * (LIFT_RATE * lift * dtf).min(1.0);
                }
                let dry = self.drought_edge(wx, wy);
                if dry > 0.0 {
                    c -= c * (DRY_RATE * dry * dtf).min(1.0);
                }
                self.buf_c[idx] = c.clamp(0.0, 1.0);
            }
        }
        std::mem::swap(&mut self.cloud, &mut self.buf_c);

        // pass 2: rain ramp + soil moisture soak/evaporation + cover count
        let mut covered = 0u64;
        for gy in 0..gh {
            for gx in 0..gw {
                let idx = (gy * gw + gx) as usize;
                let c = self.cloud[idx];
                let target_rain = if c > RAIN_CLOUD {
                    ((c - RAIN_CLOUD) / (1.0 - RAIN_CLOUD)).min(1.0)
                } else {
                    0.0
                };
                self.rain[idx] += (target_rain - self.rain[idx]) * (RAIN_RATE * dtf).min(1.0);
                let wx = (gx as f32 + 0.5) * cell_w;
                let wy = (gy as f32 + 0.5) * cell_h;
                let mut target_wet = self.rain[idx];
                let dry = self.drought_edge(wx, wy);
                if dry > 0.0 {
                    target_wet = (target_wet + DROUGHT_TARGET * dry).clamp(-1.0, 1.0);
                }
                self.wet[idx] += (target_wet - self.wet[idx]) * (WET_RATE * dtf).min(1.0);
                self.wet[idx] = self.wet[idx].clamp(-1.0, 1.0);
                if c > 0.3 || self.rain[idx] > 0.1 {
                    covered += 1;
                }
            }
        }

        // pass 3: flash decay + cover
        self.flash -= self.flash * (FLASH_DECAY * dtf).min(1.0);
        let total = (gw * gh) as f64;
        self.cover = (covered as f64 / total) as f32;
    }

    /// Push a region effect, bounded: the oldest is drained into `expired`.
    pub fn add_effect(&mut self, e: RegionEffect, expired: &mut Vec<RegionEffect>) {
        if self.effects.len() >= EFFECTS_CAP {
            expired.push(self.effects.remove(0));
        }
        self.effects.push(e);
    }

    /// Sanity guard used on restore: rebuilds the coarse fields if dimensions
    /// and lengths disagree, so corrupt/foreign saves cannot poison the sim.
    pub fn ensure_valid(&mut self) {
        self.gw = self.gw.max(2);
        self.gh = self.gh.max(2);
        self.ww = if self.ww.is_finite() && self.ww > 0.0 { self.ww } else { 1.0 };
        self.wh = if self.wh.is_finite() && self.wh > 0.0 { self.wh } else { 1.0 };
        let n = (self.gw * self.gh) as usize;
        if self.cloud.len() != n || self.rain.len() != n || self.wet.len() != n {
            self.cloud = vec![CLOUD_BASE; n];
            self.rain = vec![0.0; n];
            self.wet = vec![0.0; n];
        }
        if self.buf_c.len() != n {
            self.buf_c = vec![0.0; n];
        }
        self.recent_bolts.truncate(BOLTS_CAP);
        self.effects.truncate(EFFECTS_CAP);
        for e in self.effects.iter_mut() {
            e.x = if e.x.is_finite() { e.x } else { 0.0 };
            e.y = if e.y.is_finite() { e.y } else { 0.0 };
            e.r = if e.r.is_finite() { e.r.max(1.0) } else { 1.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::WorldGenConfig;

    #[test]
    fn sampling_stays_in_bounds() {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let w = WeatherState::new(&world);
        for v in [
            w.cloud(-1000.0, -1000.0),
            w.cloud(1e6, 1e6),
            w.rain(0.0, 0.0),
            w.wet(123.0, 321.0),
        ] {
            assert!(v.is_finite());
        }
        assert!((w.cloud(123.0, 456.0) - CLOUD_BASE).abs() < 1e-5); // ambient initially
    }

    #[test]
    fn rain_effect_raises_cloud_and_cover() {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let mut w = WeatherState::new(&world);
        w.add_effect(
            RegionEffect {
                x: world.width * 0.5,
                y: world.height * 0.5,
                r: 400.0,
                kind: EffectKind::Rain,
                until: 1 << 40,
                pending_bolts: 0,
                next_bolt_at: 0,
            },
            &mut Vec::new(),
        );
        let cx = world.width * 0.5;
        let cy = world.height * 0.5;
        for _ in 0..10 {
            let mut exp = Vec::new();
            w.update(1.0, 60, &mut exp);
        }
        assert!(w.cloud(cx, cy) > CLOUD_BASE + 0.5);
        assert!(w.rain(cx, cy) > 0.0);
        assert!(w.wet_cover() > 0.05);
        assert!(exp_empty_probe(&mut w));
    }

    fn exp_empty_probe(w: &mut WeatherState) -> bool {
        let mut exp = Vec::new();
        w.update(1.0, 1 << 41, &mut exp);
        w.effects.is_empty()
    }
}
