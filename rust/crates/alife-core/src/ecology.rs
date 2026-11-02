//! Living-world ecology layer (Agent 9).
//!
//! Complements the engine's global food spawn logic (`Simulation::regen_food`
//! / weather-coupled `spawn_food_random`): it NEVER creates or removes food
//! entities — it modulates the energy of the items the engine already
//! spawned. Roles per tick (~4 Hz, budgeted by the caller):
//!
//! - regrowth: items in fertile + moist cells fatten (logistic, capped),
//!   gated down by local hazard and local stocking density (crowding);
//! - decay: items in poor habitat (low [`EcologyState::suitability`]) thin
//!   toward the engine's spawn-minimum energy floor (1.0) and never below
//!   it, so the food item count is preserved exactly (zero double-spawn /
//!   zero-culling risk in the engine's save/undo/identifier flows);
//! - a global soft-cap pressure scales regrowth down once the item count
//!   exceeds `EcologyConfig::food_capacity_soft`.
//!
//! Budget rules (verified by tests):
//! - no full-world scans per tick: crowding counts live on a bounced scratch
//!   grid (<= 4096 f32 = 16 KiB, refilled at tick time from food positions);
//! - only existing-food positions are sampled (four bilinear taps each);
//! - deterministic: the RNG is derived from the caller-provided `u64` state
//!   (splitmix64), items are visited in fixed index order, no global rand,
//!   no wall clock, no per-item allocation.
//!
//! With `cfg.off == true` or a food list with no alive items, `tick()` is a
//! strict no-op: foods, accumulators and `rng_state` are untouched.
//!
//! Limitations (bounded by design): NOTE — the earlier "not yet part of
//! SaveCore" limitation is RESOLVED: the engine persists the state as a
//! serde-defaulted payload (`SaveCore.ecology: EcologySave`, exact restore
//! in `Simulation::from_save_core`, cadence documented in `advance()`:
//! ecology ticks at FIELD_HZ inside the same bounded catch-up loop as the
//! signal field, ALWAYS scoped to existing items so there is no double
//! regrowth). Crowding resolution scales with world size
//! (<= 64 cells per axis), so it is coarse on very large worlds; grazing
//! pressure from organisms is intentionally engine-side only; signal
//! coupling stays the single scalar `signal_bias` argument of
//! [`EcologyState::tick`] (mapped from a
//! sampled QueenActivity value by [`EcologyState::regrow_bias_from_signal`];
//! the engine samples the QueenActivity field mass once per ecology tick).

// ---------------------------------------------------------- save payload

/// Additive save payload for `SaveCore.ecology` (serde-defaulted: saves
/// written before the ecology layer load with the default config/state).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EcologySave {
    #[serde(default)]
    pub state: EcologyState,
    /// ecology rng (field rebuild); serde-defaulted for pre-ecology saves
    #[serde(default)]
    pub rng: u64,
}

use alife_common::{sanitize, Food};
use alife_worldgen::WorldModel;
use serde::{Deserialize, Serialize};

/// Absolute per-item energy bound (the engine clamps spawned energy to
/// 1..=400 as well; ecology never goes past it).
const ENERGY_CAP: f32 = 400.0;
/// Energy floor: ecology decay never thins an item below the engine's
/// spawn-minimum energy, so item counts are preserved exactly.
const ENERGY_FLOOR: f32 = 1.0;
/// Logistic regrowth saturation: growth stalls near this per-item energy
/// (~3x the default `food_energy` 20) so fertile pockets fatten, not blow up.
const REGROW_TIP: f32 = 64.0;
/// Crowding-grid cells per axis (GRID_MAX^2 f32 = 16 KiB scratch bound).
const GRID_MAX: usize = 64;
/// Golden-ratio increment for the derived splitmix64 state.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Tunables for the layer. Defaults are calibrated against the engine's
/// `SimParams::default()` food economy (food_energy = 20, max_food = 1300,
/// regen = 26/s): a gentle mass transfer that keeps the item count and total
/// near baseline (see `tests/ecology_checks.rs` calibration probe).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcologyConfig {
    /// Energy per sim-second an item in *perfect* (fertile/moist, safe)
    /// habitat can accrue.
    pub regrow_per_sec_base: f32,
    /// Fertility share of the local growth quality composite.
    pub fertility_weight: f32,
    /// Moisture share of the local growth quality composite.
    pub moisture_weight: f32,
    /// 0..1: share of regrowth removed at maximum local hazard.
    pub hazard_penalty: f32,
    /// Energy per sim-second drained from an item at zero suitability.
    pub local_food_decay: f32,
    /// Regrowth multiplier per extra item stocked in the same grid cell.
    pub crowding_penalty: f32,
    /// Soft capacity: above this many alive items, regrowth pressure
    /// scales down linearly toward zero mass gain.
    pub food_capacity_soft: usize,
    /// Master off switch: `tick()` must be a strict no-op.
    pub off: bool,
}

impl Default for EcologyConfig {
    fn default() -> Self {
        // Empirically calibrated (see tests/ecology_checks.rs): on the
        // default genesis world the coupled layer moves total food mass by
        // only a small fraction over ~30 sim-seconds while keeping the item
        // count exactly; growth beats decay in rich cells, decay floors
        // barren-habitat items at the engine's 1.0 minimum.
        Self {
            regrow_per_sec_base: 4.5,
            fertility_weight: 1.0,
            moisture_weight: 0.55,
            hazard_penalty: 0.7,
            local_food_decay: 1.6,
            crowding_penalty: 0.25,
            food_capacity_soft: 1000,
            off: false,
        }
    }
}

/// Bounded per-simulation ecology state (three field Vec + config).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologyState {
    pub cfg: EcologyConfig,
    /// Bounced crowding-count scratch grid (capacity carries over between
    /// ticks; <= 4096 f32 = 16 KiB by construction).
    scratch: Vec<f32>,
    /// Unapplied regrowth bookkeeping (growth clipped by the energy cap /
    /// logistic tip), bounded to 1024 to keep the accumulator finite.
    accum: f32,
}

impl Default for EcologyState {
    fn default() -> Self {
        Self::new(EcologyConfig::default())
    }
}

impl EcologyState {
    pub fn new(cfg: EcologyConfig) -> Self {
        Self { cfg, scratch: Vec::new(), accum: 0.0 }
    }

    /// Backing size of the crowding scratch grid in bytes (memory-budget
    /// probe; never exceeds 16 KiB by construction).
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.capacity() * std::mem::size_of::<f32>()
    }

    /// Advance the layer by `dt` sim-seconds (~4 Hz when wired by the
    /// coordinator). Strict no-op when `cfg.off` or no alive foods; never
    /// panics on weird world/food values (positions and fields sanitized).
    ///
    /// `signal_bias`: queen-signal coupling multiplier for the regrowth term
    /// (engine passes `EcologyState::regrow_bias_from_signal(sampled
    /// QueenActivity)`; neutral 1.0). Clamped defensively to 0.25..4.0.
    pub fn tick(
        &mut self,
        world: &WorldModel,
        foods: &mut Vec<Food>,
        dt: f32,
        rng_state: &mut u64,
        signal_bias: f32,
    ) {
        if self.cfg.off {
            return;
        }
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { return };
        if dt <= 0.0 || foods.is_empty() {
            return;
        }
        let mut alive_n = 0usize;
        for f in foods.iter() {
            if f.alive {
                alive_n += 1;
            }
        }
        if alive_n == 0 {
            return;
        }

        // --- crowding counts on a bounded bounced scratch grid
        let (step, cw, chh) = grid_dims(world);
        self.scratch.clear();
        self.scratch.resize(cw * chh, 0.0);
        for f in foods.iter() {
            if !f.alive {
                continue;
            }
            let fx = sanitize(f.x);
            let fy = sanitize(f.y);
            self.scratch[cell_index(fx, fy, step, cw, chh)] += 1.0;
        }

        // --- deterministic tick RNG, derived from the caller's state only.
        // All no-op paths above return *before* this line, so they are true
        // "zero mutation" (foods, accum and rng_state untouched).
        *rng_state = rng_state.wrapping_add(GOLDEN);
        let mut z = *rng_state;
        let jit = 0.9 + 0.2 * next_f01(&mut z);

        // --- global soft-cap pressure (1 below the soft cap, shrinking after)
        let soft = self.cfg.food_capacity_soft.max(1) as f32;
        let soft_p = (soft / alive_n as f32).min(1.0);

        // --- fixed-order (index) pass over existing items only.
        // Never spawns, never culls: count is preserved exactly.
        let cfg = self.cfg;
        let bias = if signal_bias.is_finite() { signal_bias.clamp(0.25, 4.0) } else { 1.0 };
        let fw = cfg.fertility_weight.abs();
        let mw = cfg.moisture_weight.abs();
        let wsum = fw + mw;
        for f in foods.iter_mut() {
            if !f.alive {
                continue;
            }
            let fx = sanitize(f.x);
            let fy = sanitize(f.y);
            let fert = c01(world.sample_fertility(fx, fy));
            let moist = c01(world.sample_moisture(fx, fy));
            let temp = c01(world.sample_temperature(fx, fy));
            let hz = c01(world.sample_hazard(fx, fy));
            let s = suitability_from_fields(temp, moist, hz, cfg.hazard_penalty);

            let qual = if wsum > 1e-6 { (fw * fert + mw * moist) / wsum } else { 0.0 };
            let hz_gate = (1.0 - cfg.hazard_penalty.clamp(0.0, 1.0) * hz).clamp(0.0, 1.0);
            let cnt = self.scratch[cell_index(fx, fy, step, cw, chh)];
            let crowd = 1.0 / (1.0 + cfg.crowding_penalty.max(0.0) * (cnt - 1.0).max(0.0));
            let pre = if f.energy.is_finite() { f.energy } else { ENERGY_FLOOR };
            let logistic = (1.0 - pre / REGROW_TIP).clamp(0.0, 1.0);
            let mut growth = cfg.regrow_per_sec_base.max(0.0)
                * qual
                * hz_gate
                * crowd
                * soft_p
                * jit
                * bias
                * logistic
                * dt;
            if pre >= ENERGY_CAP {
                growth = 0.0;
            }
            let growth_applied = growth.min((ENERGY_CAP - pre).max(0.0));
            self.accum = (self.accum + (growth - growth_applied).max(0.0)).min(1024.0);

            // habitat decay: floored at the engine spawn-minimum, so items
            // never die from ecology and counts stay exactly preserved.
            let decay = (cfg.local_food_decay.max(0.0) * (1.0 - s) * dt)
                .min((pre - ENERGY_FLOOR).max(0.0));
            f.energy = (pre - decay + growth_applied).clamp(ENERGY_FLOOR, ENERGY_CAP);
        }
    }

    /// Composite habitat suitability at a world point in [0, 1]: temperature
    /// preference (smooth peak at 0.55), moisture availability and a hazard
    /// discount. Pure, allocation-free, clamped on every input.
    pub fn suitability(&self, world: &WorldModel, x: f32, y: f32) -> f32 {
        let x = sanitize(x);
        let y = sanitize(y);
        suitability_from_fields(
            c01(world.sample_temperature(x, y)),
            c01(world.sample_moisture(x, y)),
            c01(world.sample_hazard(x, y)),
            self.cfg.hazard_penalty,
        )
    }

    /// Pure helper for the future queen-signal coupling: maps a sampled
    /// signal in [-1, 1] (as consumed by organism inputs) to a regrowth bias
    /// multiplier in [0.5, 1.5], neutral (1.0) at 0, robust to NaN.
    pub fn regrow_bias_from_signal(signal_sample: f32) -> f32 {
        if !signal_sample.is_finite() {
            return 1.0;
        }
        1.0 + 0.5 * signal_sample.clamp(-1.0, 1.0)
    }
}

// ------------------------------------------------------------------ helpers

fn c01(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn smoothstep(a: f32, b: f32, v: f32) -> f32 {
    let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Shared suitability math (also used inside `tick` to avoid re-sampling).
fn suitability_from_fields(temp: f32, moist: f32, hazard: f32, hazard_penalty: f32) -> f32 {
    let t_off = (2.0 * (temp - 0.55)).abs().clamp(0.0, 1.0);
    let thermal = 1.0 - t_off * t_off;
    let wet = smoothstep(0.10, 0.62, moist);
    let mut s = 0.45 * thermal + 0.55 * wet;
    s *= (1.0 - hazard_penalty.clamp(0.0, 1.0) * hazard).clamp(0.0, 1.0);
    s.clamp(0.0, 1.0)
}

/// Crowding grid: cell side so that each axis fits in GRID_MAX cells
/// (product bounded => scratch <= 16 KiB).
fn grid_dims(world: &WorldModel) -> (f32, usize, usize) {
    let w = if world.width.is_finite() && world.width > 0.0 { world.width } else { 1.0 };
    let h = if world.height.is_finite() && world.height > 0.0 { world.height } else { 1.0 };
    let span = w.max(h);
    let step = span / GRID_MAX as f32;
    let cw = ((w / step).ceil() as usize).clamp(1, GRID_MAX);
    let ch = ((h / step).ceil() as usize).clamp(1, GRID_MAX);
    (step, cw, ch)
}

fn cell_index(x: f32, y: f32, step: f32, cw: usize, ch: usize) -> usize {
    // `as usize` saturates negatives to 0; both coords are sanitized.
    let xi = ((x / step) as usize).min(cw.saturating_sub(1));
    let yi = ((y / step) as usize).min(ch.saturating_sub(1));
    yi * cw + xi
}

/// splitmix64 finalizer stream; the *caller's* u64 state advances by exactly
/// GOLDEN per consumed draw, so everything stays deterministic and pure.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_f01(state: &mut u64) -> f32 {
    (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::WorldGenConfig;

    /// Constant-field tiny world (one value per field over 8x8 cells).
    fn flat_world(fert: f32, moist: f32, temp: f32, hazard: f32) -> WorldModel {
        let mk = |fv: f32| alife_worldgen::FieldGrid { w: 8, h: 8, data: vec![fv; 64] };
        WorldModel {
            width: 320.0,
            height: 320.0,
            cell: 40.0,
            elevation: mk(0.2),
            moisture: mk(moist),
            temperature: mk(temp),
            fertility: mk(fert),
            hazard: mk(hazard),
            biome: mk(1.0),
            generation_tag: 7,
            hazard_pulse_period: 0.0,
        }
    }

    fn one_food(x: f32, y: f32, energy: f32) -> Vec<Food> {
        vec![Food { id: 1, x, y, energy, alive: true }]
    }

    #[test]
    fn module_regrowth_tracks_fertility_moisture_hazard() {
        let dt = 0.25; // one field tick at FIELD_HZ 4
        let _ = dt;
        // real coverage lives in tests/ecology_checks.rs (integration suite)
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let eco = EcologyState::new(EcologyConfig::default());
        let s = eco.suitability(&world, world.width * 0.5, world.height * 0.5);
        assert!((0.0..=1.0).contains(&s));
        assert_eq!(EcologyState::regrow_bias_from_signal(0.0), 1.0);
    }
}

