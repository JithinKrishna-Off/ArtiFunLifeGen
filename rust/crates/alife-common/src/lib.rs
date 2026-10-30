//! Artificial Life — shared data model and deterministic utilities.
//!
//! This crate is the single source of truth for cross-crate contracts.
//! Ownership: integration coordinator. Other crates must conform to the
//! types and semantics documented here and must not fork their own copies.
//!
//! UNITS: world units; energy in abstract units; time in fixed simulation
//! STEPS (60 steps/second at speed 1); angles in radians.

use serde::{Deserialize, Serialize};

// ------------------------------------------------------------- neural net
pub const NET_INPUTS: usize = 11;
pub const NET_HIDDEN: usize = 10;
pub const NET_OUTPUTS: usize = 3;

/** Sensor vector layout, all values normalized to [-1, 1]. */
pub const INPUT_LABELS: [&str; NET_INPUTS] = [
    "cos food dir",
    "sin food dir",
    "food intensity",
    "cos peer dir",
    "sin peer dir",
    "peer density",
    "hazard intensity",
    "cos hazard dir",
    "sin hazard dir",
    "energy level",
    "age phase",
];

pub const OUTPUT_LABELS: [&str; NET_OUTPUTS] = ["turn", "thrust", "reproduce gate"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Network {
    /// [NET_HIDDEN * NET_INPUTS] row-major
    pub w1: Vec<f32>,
    /// [NET_HIDDEN]
    pub b1: Vec<f32>,
    /// [NET_OUTPUTS * NET_HIDDEN] row-major
    pub w2: Vec<f32>,
    /// [NET_OUTPUTS]
    pub b2: Vec<f32>,
}

impl Network {
    pub fn zeros() -> Self {
        Self {
            w1: vec![0.0; NET_HIDDEN * NET_INPUTS],
            b1: vec![0.0; NET_HIDDEN],
            w2: vec![0.0; NET_OUTPUTS * NET_HIDDEN],
            b2: vec![0.0; NET_OUTPUTS],
        }
    }
    pub fn gene_count() -> usize {
        NET_HIDDEN * NET_INPUTS + NET_HIDDEN + NET_OUTPUTS * NET_HIDDEN + NET_OUTPUTS
    }
}

// ------------------------------------------------------------- genetics
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Traits {
    /// max movement speed, units/second
    pub max_speed: f32,
    /// movement drain multiplier (lower is cheaper)
    pub efficiency: f32,
    /// sensor range in world units
    pub sensor_range: f32,
    /// basal metabolism multiplier
    pub metabolism: f32,
    /// maximum lifespan in steps
    pub lifespan: u32,
    /// body radius factor (visual/collision radius ~ 5.5 * size)
    pub size: f32,
    /// fraction of max_energy required to reproduce
    pub repr_threshold: f32,
    /// fraction of max_energy transferred out of parent per reproduction
    pub repr_cost: f32,
}

pub const TRAIT_MIN: Traits = Traits {
    max_speed: 18.0,
    efficiency: 0.6,
    sensor_range: 40.0,
    metabolism: 0.6,
    lifespan: 2400,
    size: 0.65,
    repr_threshold: 0.5,
    repr_cost: 0.2,
};
pub const TRAIT_MAX: Traits = Traits {
    max_speed: 150.0,
    efficiency: 2.4,
    sensor_range: 300.0,
    metabolism: 1.9,
    lifespan: 14000,
    size: 1.6,
    repr_threshold: 0.95,
    repr_cost: 0.6,
};

pub fn clamp_traits(t: &mut Traits) {
    let (min, max) = (&TRAIT_MIN, &TRAIT_MAX);
    macro_rules! c {
        ($f:ident) => {
            t.$f = t.$f.clamp(min.$f, max.$f);
        };
    }
    c!(max_speed);
    c!(efficiency);
    c!(sensor_range);
    c!(metabolism);
    t.lifespan = t.lifespan.clamp(min.lifespan, max.lifespan);
    c!(size);
    c!(repr_threshold);
    c!(repr_cost);
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Genotype {
    pub net: Network,
    pub traits: Traits,
}

/// Maximum storable energy; larger bodies store more.
pub fn max_energy_for(t: &Traits) -> f32 {
    110.0 + 65.0 * t.size
}

pub fn organism_radius(t: &Traits) -> f32 {
    5.5 * t.size
}

// ------------------------------------------------------------- rng
/// Deterministic mulberry32 PRNG. Serializable for exact save/restore.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rng {
    s: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self { s: seed | 1 }
    }
    pub fn next_u32(&mut self) -> u32 {
        self.s = self.s.wrapping_add(0x6D2B79F5);
        let mut t = self.s;
        t = (t ^ (t >> 15)).wrapping_mul(1 | t);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t));
        t ^ (t >> 14)
    }
    /// uniform [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }
    /// uniform [a, b)
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next_f32()
    }
    pub fn int(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_f32() * n as f32) as usize % n }
    }
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
    /// Box-Muller gaussian (sigma = 1)
    pub fn gauss(&mut self) -> f32 {
        let u = (1.0 - self.next_f32()).max(1e-20);
        let v = self.next_f32();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
    pub fn gauss_sigma(&mut self, sigma: f32) -> f32 {
        self.gauss() * sigma
    }
}

// ------------------------------------------------------------- environment
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Obstacle {
    pub x: f32,
    pub y: f32,
    pub r: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct HazardZone {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    /// energy drained per second while inside (before pulse)
    pub dps: f32,
    /// 0 = steady; else oscillates between 0.15x and 1.85x with this period (seconds)
    pub pulse_period: f32,
    pub phase: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FoodPatch {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    /// relative share of food_regen this patch receives
    pub richness: f32,
}

/// Dynamic environment layer owned by the simulation (on top of the static
/// procedural world fields from alife-worldgen).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EnvironmentEdits {
    pub obstacles: Vec<Obstacle>,
    pub hazards: Vec<HazardZone>,
    pub patches: Vec<FoodPatch>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Food {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    pub energy: f32,
    pub alive: bool,
}

// ------------------------------------------------------------- organism
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Organism {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    /// heading in radians
    pub dir: f32,
    /// current forward speed, units/second
    pub speed: f32,
    pub energy: f32,
    pub max_energy: f32,
    /// steps lived
    pub age: u32,
    pub generation: u32,
    pub genotype: Genotype,
    /// lineage id === species id
    pub lineage: u32,
    pub mom: Option<u64>,
    pub dad: Option<u64>,
    pub offspring_count: u32,
    pub alive: bool,
    pub repr_cooldown: u32,
    /// last sensor vector (engine-owned, overwritten each step)
    pub last_inputs: Vec<f32>,
    /// last action vector (engine-owned, overwritten each step)
    pub last_outputs: Vec<f32>,
}

// ------------------------------------------------------------- stats & events
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeathCause {
    Starvation,
    OldAge,
    Hazard,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BirthEvent {
    pub step: u64,
    pub child_id: u64,
    pub parent_id: u64,
    pub species_id: u32,
    pub x: f32,
    pub y: f32,
    pub generation: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeathEvent {
    pub step: u64,
    pub id: u64,
    pub species_id: u32,
    pub cause: DeathCause,
    pub x: f32,
    pub y: f32,
    pub generation: u32,
    pub age: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StepEvents {
    pub births: Vec<BirthEvent>,
    pub deaths: Vec<DeathEvent>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeciesCount {
    pub species_id: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SimStats {
    pub step: u64,
    pub population: u32,
    pub max_population: u32,
    pub total_births: u64,
    pub total_deaths: u64,
    /// births/deaths over the last 600 steps (10 sim-seconds)
    pub births_window: u32,
    pub deaths_window: u32,
    pub avg_energy: f32,
    pub avg_age: f32,
    pub avg_generation: f32,
    pub max_generation: u32,
    pub food_count: u32,
    pub species_counts: Vec<SpeciesCount>,
    /// Shannon-like diversity: normalized entropy over species population shares, in [0,1]
    pub diversity_index: f32,
    /// wet/rain coverage in [0,1] (weather system), 0 when weather is idle
    #[serde(default)]
    pub wet_cover: f32,
}

// ------------------------------------------------------------- simulation params
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimParams {
    pub seed: u32,
    pub initial_population: u32,
    pub max_population: u32,
    pub initial_food: u32,
    pub max_food: u32,
    /// new food items per second across patches/fertility map
    pub food_regen_per_sec: f32,
    /// mean energy per food item (relative variance)
    pub food_energy: f32,
    pub food_energy_variance: f32,
    /// basal drain per second at metabolism=1
    pub metabolism_base: f32,
    /// drain per second per speed unit at efficiency=1
    pub movement_cost: f32,
    /// drain per second at sensor_range=100: drain += sensing_cost * sqrt(range/100)
    pub sensing_cost: f32,
    /// steps between reproductions (per organism)
    pub reproduction_cooldown_steps: u32,
    /// steps before an organism may first reproduce
    pub min_repro_age_steps: u32,
    /// probability per gene of mutating
    pub mutation_rate: f32,
    /// gaussian sigma multiplier for mutation steps
    pub mutation_magnitude: f32,
    /// probability a mutating gene jumps 10x farther
    pub big_jump_chance: f32,
    pub wrapping: bool,
}

impl Default for SimParams {
    /// "Genesis" default: balanced, immediately interesting.
    fn default() -> Self {
        Self {
            seed: 1337,
            initial_population: 240,
            max_population: 1200,
            initial_food: 800,
            max_food: 1300,
            food_regen_per_sec: 26.0,
            food_energy: 20.0,
            food_energy_variance: 0.35,
            metabolism_base: 0.55,
            movement_cost: 0.011,
            sensing_cost: 0.004,
            reproduction_cooldown_steps: 240,
            min_repro_age_steps: 600,
            mutation_rate: 0.07,
            mutation_magnitude: 1.0,
            big_jump_chance: 0.02,
            wrapping: false,
        }
    }
}

/// Parameter keys changeable on a running world.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct LiveParams {
    pub max_population: Option<u32>,
    pub max_food: Option<u32>,
    pub food_regen_per_sec: Option<f32>,
    pub food_energy: Option<f32>,
    pub metabolism_base: Option<f32>,
    pub movement_cost: Option<f32>,
    pub sensing_cost: Option<f32>,
    pub mutation_rate: Option<f32>,
    pub mutation_magnitude: Option<f32>,
    pub big_jump_chance: Option<f32>,
}

// ------------------------------------------------------------- interaction
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InteractionMode {
    Inspect,
    Food,
    Obstacle,
    Patch,
    Hazard,
    Erase,
}

// ------------------------------------------------------------- lineage record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineageRecord {
    pub id: u64,
    pub mom: Option<u64>,
    pub dad: Option<u64>,
    pub generation: u32,
    pub species_id: u32,
    pub born_step: u64,
    pub death_step: Option<u64>,
    pub cause: Option<DeathCause>,
}

// ------------------------------------------------------------- util
pub fn clamp(v: f32, min: f32, max: f32) -> f32 {
    v.clamp(min, max)
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn wrap_pi(a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let tau = pi * 2.0;
    let mut a = a % tau;
    if a > pi {
        a -= tau;
    }
    if a < -pi {
        a += tau;
    }
    a
}

pub fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Species/lineage hue: golden-angle walk, deterministic, distinct from UI colors.
pub fn species_hue(species_id: u32) -> f32 {
    (species_id as f64 * 137.508) as f32 % 360.0
}

// ------------------------------------------------------------- world config
/// Configuration for procedural world generation. Defined here (not in
/// alife-worldgen) so persistence and presets can reference it without a
/// circular dependency. alife-worldgen consumes it via `generate(&Self)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldGenConfig {
    pub seed: u32,
    pub width: u32,
    pub height: u32,
    pub cell: u32,
    pub terrain_scale: f32,
    pub detail: u32,
    pub warp: f32,
    pub moisture_scale: f32,
    pub resource_density: f32,
    pub hazard_amount: f32,
    pub fertility_contrast: f32,
    /// > 0: world hazard intensity pulses with this period in seconds
    pub hazard_pulse_period: f32,
}

impl Default for WorldGenConfig {
    /// "Genesis" defaults.
    fn default() -> Self {
        Self {
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
}

// ------------------------------------------------------------- persistence contract
/// Versioned save format contract. alife-data implements file IO against
/// this; the renderer/UI consumes it for save/load flows.
pub mod persist {
    use super::*;

    pub const FORMAT_ID: &str = "artificial-life/save";
    /// Save format version: 1 = legacy (header only), 2 = header + embedded
    /// engine `core` blob (weather / lineage history / interventions).
    pub const FORMAT_VERSION: u32 = 2;

    /// One aggregated statistics sample, taken every `interval` steps.
    #[derive(Clone, Debug, Serialize, Deserialize, Default)]
    pub struct HistorySample {
        pub step: u64,
        pub population: u32,
        pub births: u32,
        pub deaths: u32,
        pub avg_energy: f32,
        pub avg_age: f32,
        pub avg_generation: f32,
        pub food_count: u32,
        /// top species (id, count) at sample time
        pub species_top: Vec<(u32, u32)>,
        pub diversity: f32,
        /// wet/rain coverage in [0,1] at sample time (weather view)
        #[serde(default)]
        pub wet_cover: f32,
        /// true when sample came from a downsampled (merged) pair
        #[serde(default)]
        pub merged: bool,
    }

    /// Bounded, downsampled history buffer. When `max` is exceeded, the buffer
    /// merges adjacent sample pairs (interval doubles) — charts stay bounded.
    #[derive(Clone, Debug, Serialize, Deserialize, Default)]
    pub struct HistoryBuffer {
        pub interval: u32,
        pub max: usize,
        pub samples: Vec<HistorySample>,
        /// births/deaths accumulated since the last emitted sample
        pub pending_births: u32,
        pub pending_deaths: u32,
    }

    impl HistoryBuffer {
        pub fn new(interval: u32, max: usize) -> Self {
            Self { interval, max, ..Default::default() }
        }
    }

    /// Complete world snapshot for exact restoration.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct WorldSave {
        pub format_id: String,
        pub version: u32,
        pub name: String,
        pub preset: Option<String>,
        pub params: SimParams,
        pub world_config: WorldGenConfig,
        pub step_count: u64,
        pub organisms: Vec<Organism>,
        pub foods: Vec<Food>,
        pub edits: EnvironmentEdits,
        pub rng_state: u32,
        pub records: Vec<LineageRecord>,
        pub history: HistoryBuffer,
        /// v2: engine SaveCore blob (serde_json) capturing weather state,
        /// interventions and everything core needs to restore exactly.
        /// None => v1 save (best-effort restore).
        #[serde(default)]
        pub core: Option<serde_json::Value>,
    }
}

// ------------------------------------------------------------- god mode contract
/// Player interventions. Every variant must cause a real state change in the
/// engine; the engine implements `trigger(&mut self, &GodEvent) -> Result<(), String>`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum GodEvent {
    /// local rainfall event: wets the region for `duration_steps`, feeds fertility
    Rain { x: f32, y: f32, r: f32, duration_steps: u32 },
    /// dries a region (moisture/fertility penalty)
    Drought { x: f32, y: f32, r: f32, duration_steps: u32 },
    /// storm cloud + `lightning_bolts` strikes during the event window
    Storm { x: f32, y: f32, r: f32, duration_steps: u32, lightning_bolts: u32 },
    /// single lightning strike (small damage hazard scar + fertility boost)
    Lightning { x: f32, y: f32 },
    /// scatter energy/nutrients (food energy total `energy` inside r)
    NutrientDump { x: f32, y: f32, r: f32, energy: f32 },
    /// spawn one fresh organism at (x, y) with a random founder genome
    SummonOrganism { x: f32, y: f32 },
    /// remove all organisms of one species/lineage
    PurgeLineage { species_id: u32 },
    /// planet-scale interventions
    Catastrophe { kind: CatastropheKind },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CatastropheKind {
    /// meteor: burn an impact zone, add dust cloud + shockwave hazards
    Meteor,
    /// global storm: world-wide rain + lightning barrage
    GlobalStorm,
    /// global drought shock
    GlobalDrought,
}

impl GodEvent {
    pub fn label(&self) -> &'static str {
        match self {
            GodEvent::Rain { .. } => "Rain",
            GodEvent::Drought { .. } => "Drought",
            GodEvent::Storm { .. } => "Storm",
            GodEvent::Lightning { .. } => "Lightning",
            GodEvent::NutrientDump { .. } => "Nutrients",
            GodEvent::SummonOrganism { .. } => "Summon",
            GodEvent::PurgeLineage { .. } => "Purge lineage",
            GodEvent::Catastrophe { .. } => "Catastrophe",
        }
    }
    /// Center + radius of affected area if spatial (for preview/confirmation).
    pub fn region(&self) -> Option<(f32, f32, f32)> {
        match self {
            GodEvent::Rain { x, y, r, .. }
            | GodEvent::Drought { x, y, r, .. }
            | GodEvent::Storm { x, y, r, .. }
            | GodEvent::NutrientDump { x, y, r, .. } => {
                Some((*x, *y, if r.is_finite() { *r } else { 40.0 }))
            }
            GodEvent::Lightning { x, y, .. } | GodEvent::SummonOrganism { x, y, .. } => {
                Some((*x, *y, 40.0))
            }
            _ => None,
        }
    }
}

/// Recorded intervention (in order applied).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub step: u64,
    pub event: GodEvent,
    /// short result note ("+26 food", "purged species 14: 92 died", ...)
    pub note: String,
}

// ------------------------------------------------------------- presets contract
/// Named experiment presets. Values are real parameter sets, not labels.
pub mod presets {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Preset {
        pub id: &'static str,
        pub name: &'static str,
        pub description: &'static str,
        pub world: WorldGenConfig,
        pub sim: SimParams,
    }

    impl Preset {
        pub fn from(id: &'static str, name: &'static str, description: &'static str) -> Self {
            Self {
                id,
                name,
                description,
                world: WorldGenConfig::default(),
                sim: SimParams::default(),
            }
        }
        pub fn with_world(mut self, f: impl FnOnce(&mut WorldGenConfig)) -> Self {
            f(&mut self.world);
            self
        }
        pub fn with_sim(mut self, f: impl FnOnce(&mut SimParams)) -> Self {
            f(&mut self.sim);
            self
        }
    }
}
