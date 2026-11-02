//! Simulation engine: fixed-step organism lifecycle, energy economy, spatial
//! hashing, deterministic RNG, environment edits and lineage records.
//!
//! Contract: `Simulation::new(&SimParams, WorldModel)` takes ownership of the
//! world and keeps it under the public field `world`. The engine is
//! single-threaded and deterministic: same params + seed + call sequence =>
//! identical trajectory. `step()` advances exactly one 1/60 fixes step and
//! never depends on frame rate.

pub mod edit;
pub mod ecology;
pub mod queen;
pub mod queen_bridge;
pub mod signals;
pub mod spatial;
pub mod weather;

pub use queen::{
    NeuralTickBudget, QueenBrain, QueenBrainFactory, QueenBrainSnapshot, QueenBrainStats,
    QueenConfig, QueenDeath, QueenEvent, QueenEventKind, QueenSnapshot, QueenState,
    SensoryInputForQueen, QUEEN_ACTIVITY_KIND,
};

/// The real signal field (Agent 7's module; API superset of the old
/// fallback: new/emit/tick/sample/active_mass used by the engine).
pub use signals::SignalField as QueenSignalField;

pub use weather::{
    EffectKind, FertBoost, RegionEffect, WeatherState, BOLTS_CAP, BOOSTS_CAP, EFFECTS_CAP,
    FIRE_DPS,
};

use alife_common::{
    max_energy_for, organism_radius, sanitize, wrap_pi, BirthEvent, CatastropheKind, DeathCause,
    DeathEvent, EnvironmentEdits, Food, FoodPatch, GodEvent, HazardZone, InterventionRecord,
    InteractionMode, LineageRecord, LiveParams, Obstacle, Organism, Rng, SimParams, SimStats,
    SpeciesCount, StepEvents, NET_INPUTS, NET_OUTPUTS,
};
use alife_evo::{genotype_distance, mutate_genotype, random_genotype, think};
use alife_worldgen::WorldModel;
use edit::EditEntry;
use ecology::{EcologyConfig, EcologyState};
use serde::{Deserialize, Serialize};
use spatial::SpatialGrid;
use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;

/// Genotype distance beyond which a newborn starts a new species/lineage.
pub const NEW_SPECIES_DISTANCE: f32 = 0.62;
/// Fixed simulation timestep (60 steps per sim second).
pub const DT: f32 = 1.0 / 60.0;
const STEP_HZ: f32 = 60.0;
/// Extra reach around an organism for eating a food item.
const PICKUP_RADIUS: f32 = 3.0;
/// Spatial hash cell side.
const GRID_CELL: f32 = 48.0;
/// Rolling stats window (600 steps = 10 sim seconds).
const WINDOW_LEN: usize = 600;
/// Lineage record map bound: at most this many records are retained; the
/// oldest already-dead records that are NOT ancestors of a living organism
/// are evicted first, so living lineages keep a full family tree.
const RECORDS_CAP: usize = 512;
/// Eviction slack over [`RECORDS_CAP`]: the trim fires once past this mark
/// and evicts back down to it, amortizing the scan cost over batches.
const RECORDS_SLACK: usize = 64;
/// Editor undo depth (multi-level, bounded).
const EDIT_STACK_CAP: usize = 256;
/// redo stack bound (never persisted; session-local)
const REDO_STACK_CAP: usize = 64;
/// Share of food_regen_per_sec that patches receive, scaled by richness.
const PATCH_REGEN_SHARE: f32 = 0.5;
/// Signal/food field tick rate (Hz) in sim time and the per-tick dt.
const FIELD_HZ: f32 = 4.0;
const FIELD_TICK_DT: f32 = 1.0 / FIELD_HZ;
/// Field accumulator never banks more than a couple of ticks of catch-up.
const FIELD_ACC_CAP: f32 = 2.0 * FIELD_TICK_DT;
/// Fractional food budget never carries more than a few items across steps.
const FOOD_ACC_CAP: f32 = 4.0;
/// Weather tick interval in steps (default 60 = 1 sim-second).
const WEATHER_EVERY: u32 = 60;
/// Minimum weather tick interval (bounded so accelerate runs still advance).
const WEATHER_EVERY_MIN: u32 = 6;
/// God intervention history bound (keep LAST this many).
const INTERV_CAP: usize = 64;
/// Protected-lineage list bound.
const PROTECTED_CAP: usize = 64;
/// Lightning strike radius (units) and damage (energy).
const LIGHTNING_RADIUS: f32 = 14.0;
const LIGHTNING_DMG: f32 = 25.0;
/// Steps between scheduled storm bolts.
const BOLT_INTERVAL_STEPS: u32 = 30;
/// Fertility ash radius/boost/duration of a lightning strike (see weather).
const BOLT_ASH_RADIUS: f32 = 30.0;
const BOLT_ASH_AMOUNT: f32 = 0.9;
const BOLT_ASH_STEPS: u64 = 1800;
/// Meteor constants: impact radius, fire lifetime, burn scar, fert damage.
const METEOR_RADIUS: f32 = 180.0;
const METEOR_FIRE_STEPS: u64 = 5400;
const METEOR_BURN_AMOUNT: f32 = -0.9;
const METEOR_BURN_STEPS: u64 = 7200;
/// Global catastrophe durations (steps): storm 30 s, drought 180 s.
const GLOBAL_STORM_STEPS: u64 = 1800;
const GLOBAL_STORM_BOLTS: u32 = 30;
const GLOBAL_DROUGHT_STEPS: u64 = 10800;
/// Death-cause classification: a death in a cell whose hazard pressure is at
/// least this energy/s counts as Hazard rather than Starvation.
const HAZARD_CLASSIFY_MIN: f32 = 3.0;
/// Initial food seeding budget = INITIAL_SEED_MULTIPLE x initial_food attempts.
const INITIAL_SEED_MULTIPLE: u32 = 3;
/// Organism signal sensing: max contribution of the QueenActivity sample to
/// the food-intensity input slot (blend; slot count unchanged).
const SIGNAL_SENSE_BLEND: f32 = 0.25;

/// Complete engine snapshot for exact save/restore. All fields are required
/// on restore; the persistence layer serializes this whole struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveCore {
    pub step_count: u64,
    pub organisms: Vec<Organism>,
    pub foods: Vec<Food>,
    pub edits: EnvironmentEdits,
    pub rng_state: u32,
    pub records: Vec<LineageRecord>,
    pub edit_stack: Vec<EditEntry>,
    pub next_id: u64,
    pub next_lineage: u32,
    pub total_births: u64,
    pub total_deaths: u64,
    pub births_window: u32,
    pub deaths_window: u32,
    pub births_ring: Vec<u32>,
    pub deaths_ring: Vec<u32>,
    pub food_acc_base: f32,
    pub food_acc_patch: f32,
    /// v2 weather layer state (exact restore)
    pub weather: WeatherState,
    /// last god interventions (bounded [INTERV_CAP])
    pub interventions: Vec<InterventionRecord>,
    /// protected lineage ids (hazard-immune)
    pub protected: Vec<u32>,
    /// death-cause totals: (starvation, old age, hazard)
    pub total_starve: u64,
    pub total_age: u64,
    pub total_hazard: u64,
    /// death-cause rolling window rings (same layout as deaths_ring)
    pub starve_ring: Vec<u32>,
    pub age_ring: Vec<u32>,
    pub hazard_ring: Vec<u32>,
    /// death-cause rolling window counters
    pub starve_window: u32,
    pub age_window: u32,
    pub hazard_window: u32,
    /// weather cadence configured for this run
    pub weather_every: u32,
    /// v2 queen state (additive; older v2 saves without this / null load as None)
    #[serde(default)]
    pub queen: Option<queen::QueenSnapshot>,
    /// v2 ecology layer state (additive; pre-ecology saves load the default
    /// config/state — `EcologySave` is fully serde-defaulted)
    #[serde(default)]
    pub ecology: ecology::EcologySave,
}

/// Stable read/step API consumed by UI, renderer and app.
pub trait SimApi {
    fn step(&mut self) -> StepEvents;
    fn stats(&self) -> &SimStats;
    fn step_count(&self) -> u64;
    fn organisms(&self) -> &[Organism];
    fn foods(&self) -> &[Food];
    fn edits(&self) -> &EnvironmentEdits;
    fn pick(&self, x: f32, y: f32, r: f32) -> Option<u64>;
    fn spawn_organisms(&mut self, n: usize);
    fn add_food(&mut self, x: f32, y: f32) -> bool;
    fn add_obstacle(&mut self, x: f32, y: f32, r: f32) -> bool;
    fn add_patch(&mut self, x: f32, y: f32, r: f32) -> bool;
    fn add_hazard(&mut self, x: f32, y: f32, r: f32) -> bool;
    fn remove_at(&mut self, x: f32, y: f32, r: f32) -> bool;
    fn undo_edit(&mut self) -> bool;
    fn update_live_params(&mut self, patch: &LiveParams);
    fn organism(&self, id: u64) -> Option<&Organism>;
    fn get_record(&self, id: u64) -> Option<LineageRecord>;
    fn rng_state(&self) -> u32;
    fn snapshot(&self) -> SaveCore;
    fn restore(&mut self, params: &SimParams, world: WorldModel, save: &SaveCore);
    // ---------------- god mode / weather / diagnostics (engine-implemented)
    /// Apply a god intervention. Errors on invalid input (never panics).
    fn trigger(&mut self, ev: &GodEvent) -> Result<(), String>;
    /// Last applied interventions (bounded [INTERV_CAP], oldest dropped).
    fn interventions(&self) -> &[InterventionRecord];
    /// Toggle hazard immunity for a lineage. Returns the resulting state.
    fn protect_lineage(&mut self, species_id: u32, on: bool) -> bool;
    /// Currently protected lineage ids.
    fn protected_lines(&self) -> Vec<u32>;
    /// Read access to the weather layer (render sampling + UI).
    fn weather(&self) -> &WeatherState;
    /// Cheap aggregate diagnostics for UI cards.
    fn diagnosis(&self) -> Diag;
}

pub struct Simulation {
    pub params: SimParams,
    pub world: WorldModel,
    edits: EnvironmentEdits,
    organisms: Vec<Organism>,
    foods: Vec<Food>,
    records: HashMap<u64, LineageRecord>,
    stats: SimStats,
    step_count: u64,
    rng: Rng,
    next_id: u64,
    next_lineage: u32,
    org_grid: SpatialGrid,
    food_grid: SpatialGrid,
    scratch_food: Vec<u32>,
    scratch_peer: Vec<u32>,
    // reusable per-step / per-event scratch buffers (never reset, only
    // cleared; capacity is backed by the peak, not grown per step)
    scratch_newborns: Vec<Organism>,
    scratch_dead: Vec<(usize, DeathCause)>,
    scratch_species: HashMap<u32, u32>,
    scratch_counts: Vec<SpeciesCount>,
    scratch_expired: Vec<RegionEffect>,
    scratch_bolts: Vec<(usize, f32, f32)>,
    scratch_set: HashSet<u64>,
    scratch_pairs: Vec<(u64, u64)>,
    edit_stack: Vec<EditEntry>,
    /// undone edits that can be redone (`Ctrl+Y`); cleared by any new edit;
    /// NOT persisted (a loaded save starts with an empty redo stack)
    redo_stack: Vec<EditEntry>,
    // weather / god mode
    pub weather: WeatherState,
    last_weather_step: u64,
    next_weather_step: u64,
    weather_every: u32,
    interventions: Vec<InterventionRecord>,
    protected: Vec<u32>,
    // deaths deferred from interventions (purges) to the next step's events
    pending_deaths: Vec<DeathEvent>,
    // per-step energy ledger (reset at the top of every advance)
    step_gain: f64,
    /// threat-closure memory for the queen's brain senses (not saved state:
    /// a restored brain re-measures from rest, so the rate just starts at 0)
    queen_prev_hazard: f32,
    queen_prev_hazard_step: u64,
    step_drain: f64,
    step_birth_cost: f64,
    step_birth_credit: f64,
    step_death_loss: f64,
    // fractional food budget accumulators (base fertility map / patches)
    food_acc_base: f32,
    food_acc_patch: f32,
    // counters and rolling windows
    total_births: u64,
    total_deaths: u64,
    births_window: u32,
    deaths_window: u32,
    births_ring: Vec<u32>,
    deaths_ring: Vec<u32>,
    // death causes: totals + rolling windows (starve / age / hazard)
    total_starve: u64,
    total_age: u64,
    total_hazard: u64,
    starve_window: u32,
    age_window: u32,
    hazard_window: u32,
    starve_ring: Vec<u32>,
    age_ring: Vec<u32>,
    hazard_ring: Vec<u32>,
    // ---------------------------------------------------------- queen layer
    /// signal field (fallback module until Agent 7's SignalField lands — see
    /// `queen::` todo note). Ticks at FIELD_HZ inside `advance()`.
    pub signals: QueenSignalField,
    /// 4 Hz accumulator for `signals.tick`
    field_acc: f32,
    /// single queen slot (see `queen::QueenState`)
    queen: QueenState,
    queen_events: Vec<queen::QueenEvent>,
    queen_config: queen::QueenConfig,
    /// optional factory rehydrating a saved queen brain on load
    queen_brain_factory: Option<Box<dyn queen::QueenBrainFactory>>,
    /// transient: the LAST brain motor output from `queen_step_brain`,
    /// consumed by `update_one` when steering the queen (never persisted;
    /// zeroed = no brain / no motor).
    last_queen_motor: queen::QueenMotorOut,
    /// ecology layer (energy modulation of existing food items, ~4 Hz)
    pub ecology: EcologyState,
    /// deterministic splitmix64 seed-state for the ecology tick
    ecology_rng: u64,
    // per-advance compute report accumulators (reset by compute_report())
    rep_sim_ns: u64,
    rep_queen_brain_ns: u64,
    rep_field_ns: u64,
}

impl Simulation {
    pub fn new(params: &SimParams, world: WorldModel) -> Simulation {
        let mut sim = Self::blank(params, world);
        let founders = params.initial_population.min(params.max_population) as usize;
        for _ in 0..founders {
            sim.spawn_founder();
        }
        // Initial seeding = `initial_food` spawned items, with an attempts
        // budget of INITIAL_SEED_MULTIPLE x target (fertility gating rejects
        // some placements, and a fresh world must start with real food; the
        // budget keeps spawning attempts bounded and seed-deterministic).
        let target = params.initial_food.min(100_000) as usize;
        let mut spawned = 0usize;
        let mut attempts = 0usize;
        let budget = target.saturating_mul(INITIAL_SEED_MULTIPLE as usize).max(8);
        while spawned < target && attempts < budget {
            attempts += 1;
            if sim.spawn_food_random() {
                spawned += 1;
            }
        }
        sim.refresh_stats();
        sim
    }

    fn blank(params: &SimParams, world: WorldModel) -> Simulation {
        let ww = world.width;
        let wh = world.height;
        Self {
            params: params.clone(),
            edits: EnvironmentEdits::default(),
            organisms: Vec::with_capacity(params.max_population.min(4096) as usize),
            foods: Vec::with_capacity(params.max_food.min(4096) as usize),
            records: HashMap::new(),
            stats: SimStats::default(),
            step_count: 0,
            rng: Rng::new(params.seed | 1),
            next_id: 1,
            next_lineage: 1,
            org_grid: SpatialGrid::new(world.width, world.height, GRID_CELL),
            food_grid: SpatialGrid::new(world.width, world.height, GRID_CELL),
            scratch_food: Vec::new(),
            scratch_peer: Vec::new(),
            scratch_newborns: Vec::new(),
            scratch_dead: Vec::new(),
            scratch_species: HashMap::new(),
            scratch_counts: Vec::new(),
            scratch_expired: Vec::new(),
            scratch_bolts: Vec::new(),
            scratch_set: HashSet::new(),
            scratch_pairs: Vec::new(),
            edit_stack: Vec::new(),
            redo_stack: Vec::new(),
            weather: WeatherState::new(&world),
            last_weather_step: 0,
            next_weather_step: WEATHER_EVERY as u64,
            weather_every: WEATHER_EVERY,
            interventions: Vec::new(),
            protected: Vec::new(),
            pending_deaths: Vec::new(),
            step_gain: 0.0,
            queen_prev_hazard: 0.0,
            queen_prev_hazard_step: 0,
            step_drain: 0.0,
            step_birth_cost: 0.0,
            step_birth_credit: 0.0,
            step_death_loss: 0.0,
            food_acc_base: 0.0,
            food_acc_patch: 0.0,
            total_births: 0,
            total_deaths: 0,
            births_window: 0,
            deaths_window: 0,
            births_ring: vec![0; WINDOW_LEN],
            deaths_ring: vec![0; WINDOW_LEN],
            total_starve: 0,
            total_age: 0,
            total_hazard: 0,
            starve_window: 0,
            age_window: 0,
            hazard_window: 0,
            starve_ring: vec![0; WINDOW_LEN],
            age_ring: vec![0; WINDOW_LEN],
            hazard_ring: vec![0; WINDOW_LEN],
            signals: QueenSignalField::new(ww, wh, 64),
            field_acc: 0.0,
            queen: QueenState::default(),
            queen_events: Vec::new(),
            queen_config: QueenConfig::default(),
            queen_brain_factory: None,
            last_queen_motor: queen::QueenMotorOut::default(),
            ecology: EcologyState::new(EcologyConfig::default()),
            ecology_rng: (params.seed as u64) | 1,
            rep_sim_ns: 0,
            rep_queen_brain_ns: 0,
            rep_field_ns: 0,
            world,
        }
    }

    /// Rolling window update for a per-step counter ring.
    fn roll_window(ring: &mut [u32], window: &mut u32, value: u32, widx: usize) {
        let widx = widx % WINDOW_LEN;
        *window = window.saturating_sub(ring[widx]).saturating_add(value);
        ring[widx] = value;
    }

    /// Advance exactly one fixed step (1/60 sim second). Stable at any call
    /// rate: the engine has no internal time-scale notion; accelerated runs
    /// simply call this more often per wall-second (weather ticks stay in
    /// sim-time and catch up at most `MAX_DT` seconds after a skip).
    pub fn advance(&mut self) -> StepEvents {
        let t_sim = std::time::Instant::now();
        self.step_count += 1;
        let dt = DT;
        let mut events = StepEvents::default();

        // per-step energy ledger starts from zero
        self.step_gain = 0.0;
        self.step_drain = 0.0;
        self.step_birth_cost = 0.0;
        self.step_birth_credit = 0.0;
        self.step_death_loss = 0.0;

        self.rebuild_grids();

        // weather ticks in sim-time (see weather_tick)
        self.weather_tick(dt);

        // signal field ticks at FIELD_HZ in sim time (bounded catch-up).
        // The ecology layer runs INSIDE the same bounded loop (documented
        // cadence: ~4 Hz, always accompanies a field tick). It NEVER creates
        // or removes food — the engine's per-step `regen_food(dt)` spawn
        // budget is untouched and there is NO double regrowth: ecology only
        // modulates the ENERGY of items the engine already spawned.
        self.field_acc = (self.field_acc + dt).min(FIELD_ACC_CAP);
        let t_field = std::time::Instant::now();
        while self.field_acc >= FIELD_TICK_DT {
            self.field_acc -= FIELD_TICK_DT;
            self.signals.tick(FIELD_TICK_DT);
            // queen-activity coupling: bias regrowth by the kind-4 sample at
            // the queen's broadcast point (neutral world center otherwise)
            let (sx, sy) = if self.queen.alive {
                self.organisms
                    .iter()
                    .find(|o| o.alive && o.id == self.queen.org_id as u64)
                    .map(|o| (o.x, o.y))
                    .unwrap_or((self.world.width * 0.5, self.world.height * 0.5))
            } else {
                (self.world.width * 0.5, self.world.height * 0.5)
            };
            let s4 = self.signals.sample(queen::QUEEN_ACTIVITY_KIND, sx, sy);
            let bias = EcologyState::regrow_bias_from_signal(s4);
            self.ecology
                .tick(&self.world, &mut self.foods, FIELD_TICK_DT, &mut self.ecology_rng, bias);
        }
        self.rep_field_ns += t_field.elapsed().as_nanos() as u64;

        // deaths deferred from interventions (purges) are surfaced here
        if !self.pending_deaths.is_empty() {
            events.deaths.extend(self.pending_deaths.drain(..));
        }

        // Queen brain: senses from the SAME pre-move sensory context the
        // ordinary net uses (documented ordering: brain ticks BEFORE body
        // updates; the motor output is applied to her movement slots so the
        // ordinary `update_one` actuation/bounds/obstacle/eating path fires).
        self.queen_step_brain();

        for i in 0..self.organisms.len() {
            self.update_one(i, dt);
        }

        // reproduction: consumes post-drain energy of this step
        let pop = self.organisms.len();
        for i in 0..pop {
            self.try_reproduce(i, &mut events);
        }
        self.organisms.extend(self.scratch_newborns.drain(..));

        // deaths; starvation with real hazard pressure at the death cell is
        // classified as Hazard (documented rule: >= HAZARD_CLASSIFY_MIN).
        // Two in-place passes over the SAME Vec (capacity preserved, no
        // per-step reallocation): classify first, then compact survivors.
        let step = self.step_count;
        let queen_id64 = if self.queen.alive { Some(self.queen.org_id as u64) } else { None };
        self.scratch_dead.clear();
        for (i, o) in self.organisms.iter().enumerate() {
            // QUEEN LIFECYCLE RULE: she is the one immortal organism. She
            // never dies biologically — starvation, ageing and hazard still
            // hit her ENERGY (she can be drained to zero and recover via the
            // survival floor), but no death cause is ever produced for her
            // and she is never removed by the ordinary lifecycle. God
            // interventions respect the same rule (see PurgeLineage/meteor).
            if Some(o.id) == queen_id64 {
                continue;
            }
            let life = o.genotype.traits.lifespan;
            let cause = if !o.energy.is_finite() || o.energy <= 0.0 {
                if self.hazard_pressure(o.x, o.y) >= HAZARD_CLASSIFY_MIN {
                    Some(DeathCause::Hazard)
                } else {
                    Some(DeathCause::Starvation)
                }
            } else if o.age >= life {
                Some(DeathCause::OldAge)
            } else {
                None
            };
            if let Some(c) = cause {
                self.scratch_dead.push((i, c));
            }
        }
        let mut starve_n = 0u32;
        let mut age_n = 0u32;
        let mut hazard_n = 0u32;
        let mut death_loss = 0.0f64;
        let mut w = 0usize;
        let mut dead_i = 0usize;
        let n_orgs = self.organisms.len();
        for r in 0..n_orgs {
            if dead_i < self.scratch_dead.len() && self.scratch_dead[dead_i].0 == r {
                let (_, c) = self.scratch_dead[dead_i];
                dead_i += 1;
                let o = &self.organisms[r];
                // deaths carrying leftover energy go with the corpse
                death_loss += o.energy.max(0.0) as f64;
                events.deaths.push(DeathEvent {
                    step,
                    id: o.id,
                    species_id: o.lineage,
                    cause: c,
                    x: o.x,
                    y: o.y,
                    generation: o.generation,
                    age: o.age,
                });
                if let Some(rec) = self.records.get_mut(&o.id) {
                    rec.death_step = Some(step);
                    rec.cause = Some(c);
                }
                // corpse is dropped here (recycling is intentionally fine for
                // exact determinism: the compaction keeps survivor order)
            } else if w != r {
                self.organisms.swap(w, r);
                w += 1;
            } else {
                w = r + 1;
            }
        }
        self.organisms.truncate(w);
        self.step_death_loss += death_loss;
        for d in &events.deaths {
            match d.cause {
                DeathCause::Starvation => starve_n += 1,
                DeathCause::OldAge => age_n += 1,
                DeathCause::Hazard => hazard_n += 1,
            }
        }
        self.total_starve += starve_n as u64;
        self.total_age += age_n as u64;
        self.total_hazard += hazard_n as u64;

        // queen bookkeeping: death detection (covers purge/meteor removals
        // which surface here as deferred deaths) + energy-history ring
        self.queen_post_step(&events.deaths);

        // food upkeep
        self.foods.retain(|f| f.alive);
        self.regen_food(dt);

        // counters / rolling window (windows reset after load attempts)
        let births_this = events.births.len() as u32;
        let deaths_this = events.deaths.len() as u32;
        self.total_births += births_this as u64;
        self.total_deaths += deaths_this as u64;
        let widx = (self.step_count % WINDOW_LEN as u64) as usize;
        Self::roll_window(&mut self.births_ring, &mut self.births_window, births_this, widx);
        Self::roll_window(&mut self.deaths_ring, &mut self.deaths_window, deaths_this, widx);
        Self::roll_window(&mut self.starve_ring, &mut self.starve_window, starve_n, widx);
        Self::roll_window(&mut self.age_ring, &mut self.age_window, age_n, widx);
        Self::roll_window(&mut self.hazard_ring, &mut self.hazard_window, hazard_n, widx);

        self.refresh_stats();
        self.rep_sim_ns += t_sim.elapsed().as_nanos() as u64;
        events
    }

    // -------------------------------------------------------------- queen api

    /// Crowns `org_id` as queen (vectors 0.2). Only ALIVE ids accepted; there
    /// is no respawn on death. Returns false when the id is unknown/dead or
    /// she is already crowned with that id.
    pub fn crown_queen(&mut self, org_id: u64) -> bool {
        let alive = self
            .organisms
            .iter()
            .any(|o| o.id == org_id && o.alive);
        if !alive {
            return false;
        }
        let id32 = org_id as u32;
        if self.queen.alive && self.queen.org_id == id32 {
            return false;
        }
        self.queen = QueenState {
            alive: true,
            org_id: id32,
            crowned_tick: self.step_count,
            death_cause: None,
            brain: None,
            energy_history: Vec::new(),
            brain_ticks: 0,
            brain_restore_failed: false,
            reclaim_brain: None,
        };
        self.queen_events.push(queen::QueenEvent {
            tick: self.step_count,
            kind: queen::QueenEventKind::Crowned,
            org_id: id32,
        });
        queen::trim_events(&mut self.queen_events);
        true
    }

    /// Current queen organism id, if she is alive.
    pub fn queen_id(&self) -> Option<u32> {
        if self.queen.alive {
            Some(self.queen.org_id)
        } else {
            None
        }
    }

    /// Renderer/UI flag helper: is this organism the living queen.
    pub fn is_queen(&self, id: u64) -> bool {
        self.queen.alive && self.queen.org_id as u64 == id
    }

    /// Bounded queen event ring (<= 64, oldest dropped).
    pub fn queen_events(&self) -> &[queen::QueenEvent] {
        &self.queen_events
    }

    pub fn queen_config(&self) -> queen::QueenConfig {
        self.queen_config
    }

    pub fn set_queen_config(&mut self, c: queen::QueenConfig) {
        self.queen_config = c;
    }

    /// Plugs a queen brain (set to None to run the default net only).
    pub fn set_queen_brain(&mut self, brain: Option<Box<dyn queen::QueenBrain>>) {
        self.queen.brain = brain;
    }

    /// Registers the factory used to rehydrate a saved queen brain on load.
    /// If the sim holds a restored queen whose brain bytes could not be
    /// rehydrated yet (factory was registered too late, e.g. the UI loads
    /// its artifact AFTER the save), the snapshot bytes are attempted once
    /// here — honest success (brain plugs, flag clears) or honest failure
    /// (flag stays true).
    pub fn set_queen_brain_factory(&mut self, f: Option<Box<dyn queen::QueenBrainFactory>>) {
        self.queen_brain_factory = f;
        if self.queen.alive && self.queen.brain.is_none() {
            if let Some(snap) = self.queen.reclaim_brain.take() {
                let built = self.build_brain_from(&snap);
                if let Some(b) = built {
                    self.queen.brain = Some(b);
                    self.queen.brain_restore_failed = false;
                } else {
                    self.queen.brain_restore_failed = true;
                }
            }
        }
    }

    /// Honest factory-driven rehydration of a serde brain snapshot: builds
    /// through the registered factory and restores the bytes. `None` when no
    /// factory is registered or the bytes do not round-trip.
    fn build_brain_from(
        &mut self,
        snap: &queen::QueenBrainSnapshot,
    ) -> Option<Box<dyn queen::QueenBrain>> {
        let mut b = self.queen_brain_factory.as_ref()?.build();
        if b.restore(&snap.bytes) {
            Some(b)
        } else {
            None
        }
    }


    /// Additive accessor: the last `from_save_core` could not rebuild the
    /// saved queen brain (factory missing or snapshot bytes unusable). Such
    /// a queen loads alive, honestly WITHOUT a brain (driven by her evolved
    /// net) — never a panic, never a synthetic fixture pretending to be the
    /// real subnetwork.
pub fn queen_brain_restore_failed(&self) -> bool {
        self.queen.brain_restore_failed
    }

    /// Cheap brain counters for compute report / UI (None = no brain plugged).
    pub fn queen_brain_stats(&self) -> Option<queen::QueenBrainStats> {
        self.queen.brain.as_ref().map(|b| b.stats())
    }

    pub fn queen_reign_step(&self) -> u64 {
        if self.queen.alive {
            self.step_count.saturating_sub(self.queen.crowned_tick)
        } else {
            0
        }
    }

    /// Per-move queen bookkeeping after body updates + death compaction:
    /// detects her death (covers purge/meteor removals surfaced as deferred
    /// deaths), records `Died(..)` events, and keeps the energy-fraction ring.
    fn queen_post_step(&mut self, deaths: &[DeathEvent]) {
        if self.queen.alive {
            for d in deaths {
                if d.id as u32 == self.queen.org_id {
                    self.queen.alive = false;
                    self.queen.death_cause = Some(QueenDeath::from(d.cause));
                    self.queen.brain = None;
                    self.queen_events.push(queen::QueenEvent {
                        tick: self.step_count,
                        kind: queen::QueenEventKind::Died(QueenDeath::from(d.cause)),
                        org_id: self.queen.org_id,
                    });
                    queen::trim_events(&mut self.queen_events);
                    break;
                }
            }
        }
        // energy-history ring while she is alive
        if self.queen.alive {
            if let Some(o) = self
                .organisms
                .iter()
                .find(|o| o.id == self.queen.org_id as u64 && o.alive)
            {
                self.queen.push_energy_frac(o.energy / o.max_energy.max(1.0));
            }
        }
    }

    /// Additive save payload for `SaveCore.queen`.
    /// One queen-brain tick burst per advance, before body updates:
    /// samples the kind-4 field at her position, builds the queen senses
    /// from the ordinary sense_* context, ticks the brain under the
    /// configured budget, applies the motor output to her movement path,
    /// and surfaces `emit_signal` as a kind-4 emission + documented energy
    /// cost. Paused (budget 0) is legal: the bridge persists the last
    /// output, which is then applied unchanged.
    fn queen_step_brain(&mut self) {
        if !self.queen.alive || self.queen.brain.is_none() {
            return;
        }
        let idx = match self
            .organisms
            .iter()
            .position(|o| o.alive && o.id == self.queen.org_id as u64)
        {
            Some(i) => i,
            None => return,
        };
        // kind-4 = QueenActivity; the queen also senses her own emissions —
        // documented (the field is her broadcast medium, sample first so the
        // tick-later emission of THIS advance is not echoed into her senses)
        let (qx, qy) = { let o = &self.organisms[idx]; (o.x, o.y) };
        let sig_sample = self.signals.sample(queen::QUEEN_ACTIVITY_KIND, qx, qy);
        let senses = self.brain_senses(idx, sig_sample);
        let budget = queen::NeuralTickBudget {
            max_ticks_per_advance: self.queen_config.brain_ticks_per_advance,
        };
        let t0 = std::time::Instant::now();
        let motor = {
            let Some(b) = self.queen.brain.as_mut() else { return };
            b.tick(&senses, &budget)
        };
        // honest measured wall cost of the tick burst (compute report)
        self.rep_queen_brain_ns += t0.elapsed().as_nanos() as u64;
        self.queen.brain_ticks += budget.max_ticks_per_advance as u64;
        self.last_queen_motor = motor;
        self.apply_queen_motor(idx, &motor, &senses);
        // emission -> kind 4 at her position + energy cost, ledger-visible
        let emit = motor.emit_signal.clamp(0.0, 1.0);
        if emit > 0.0 {
            self.signals.emit(queen::QUEEN_ACTIVITY_KIND, qx, qy, emit);
            let cost = self.queen_config.signal_emit_cost.max(0.0)
                * emit
                * self.params.food_energy.max(1.0);
            if cost > 0.0 {
                let o = &mut self.organisms[idx];
                o.energy = (o.energy - cost).max(0.0);
                self.step_drain += cost as f64;
            }
        }
    }

    /// Queen sensory snapshot — documented reuse: the SAME sense_food /
    /// sense_peers / sense_hazard context the ordinary 11-input net gets,
    /// plus the caller-provided kind-4 signal sample, heading and normalized
    /// edge distance. `speed` is normalized by her body max speed (bridge
    /// contract keeps everything in [0, 1]).
    fn brain_senses(&mut self, i: usize, signal_sample: f32) -> queen::SensoryInputForQueen {
        let (px, py, dir, spd, energy, max_e, traits, alive) = {
            let o = &self.organisms[i];
            (o.x, o.y, o.dir, o.speed, o.energy, o.max_energy, o.genotype.traits, o.alive)
        };
        if !alive {
            return queen::SensoryInputForQueen::default();
        }
        let mut inp = [0.0f32; NET_INPUTS];
        let food = self.sense_food(px, py, dir, &traits, &mut inp);
        self.sense_peers(i, px, py, dir, &traits, &mut inp);
        let hazard_grad = self.sense_hazard(px, py, dir, &mut inp);
        // threat approach rate: closure of the normalized hazard pressure
        // between consecutive brain samples (transient, remembered between
        // advances; a restored brain starts measuring from rest = 0)
        let h_now = inp[6];
        let hazard_rate = {
            let delta = h_now - self.queen_prev_hazard;
            let dstep = self.step_count.saturating_sub(self.queen_prev_hazard_step).max(1);
            (delta / (dstep as f32 / 60.0)).clamp(0.0, 1.0)
        };
        self.queen_prev_hazard = h_now;
        self.queen_prev_hazard_step = self.step_count;
        let food_dir = match food {
            Some((dx, dy, _)) => {
                let m = (dx * dx + dy * dy).sqrt();
                if m > 1e-6 {
                    (dx / m, dy / m)
                } else {
                    (0.0, 0.0)
                }
            }
            None => (0.0, 0.0),
        };
        let edge = if self.params.wrapping {
            1.0 // no reachable edge in a wrap-connected world
        } else {
            let w = self.world.width.max(1.0);
            let h = self.world.height.max(1.0);
            let d = px.min(w - px).min(py.min(h - py)).max(0.0);
            (d / (w.max(h) * 0.5)).clamp(0.0, 1.0)
        };
        queen::SensoryInputForQueen {
            food_dir,
            food_dist: food.map(|(_, _, d)| d).unwrap_or(f32::INFINITY),
            heading: wrap_pi(if dir.is_finite() { dir } else { 0.0 }),
            speed: (spd / traits.max_speed.max(1e-6)).clamp(0.0, 1.0),
            energy_frac: ((energy / max_e.max(1.0)) as f32)
                .clamp(0.0, 1.0),
            hazard: inp[6],
            edge_dist: edge,
            signal_sample: if signal_sample.is_finite() { signal_sample.clamp(0.0, 1.0) } else { 0.0 },
            density: inp[5],
            hazard_dir: hazard_grad,
            hazard_rate,
        }
    }

        /// Applies the brain motor output to the queen's body slots so the
    /// ordinary `update_one` actuation integrates it (documented ordering:
    /// this PRE-steer happens BEFORE update_one, which then applies the
    /// brain-recorded turn at the standard 0.12 rad/step and smooths thrust
    /// toward `move_intensity * max_speed`):
    /// - heading pulled toward the brain's desired world direction
    ///   (`move_dir`) proportionally to `move_intensity`;
    /// - `feed_seek` pulls her toward the nearest food exactly analog to the
    ///   ordinary food-driven steering (same 0.12 rad/step constant);
    /// - `turn` / `move_intensity` / `feed_seek` are recorded into
    ///   `last_outputs[0..3]` (slots of the ordinary net contract: turn,
    ///   thrust, repro-gate — the brain-steered queen's slot 2 route to
    ///   reproduction is declined separately in `try_reproduce`).
    fn apply_queen_motor(
        &mut self,
        idx: usize,
        motor: &queen::QueenMotorOut,
        senses: &queen::SensoryInputForQueen,
    ) {
        const QUEEN_STEER_RATE: f32 = 0.35; // world-dir pull per step
        const QUEEN_FEED_TURN: f32 = 0.12; // same constant as the net mapping
        let dir = { let o = &self.organisms[idx]; o.dir };
        let di = motor.move_intensity.clamp(0.0, 1.0);
        let fs = motor.feed_seek.clamp(0.0, 1.0);
        let mut steer = 0.0f32;
        let md_m = motor.move_dir.0.hypot(motor.move_dir.1);
        if md_m > 1e-6 && motor.move_dir.0.is_finite() && motor.move_dir.1.is_finite() {
            let desired = motor.move_dir.1.atan2(motor.move_dir.0);
            steer += wrap_pi(desired - dir) * di * QUEEN_STEER_RATE;
        }
        // nearest-food pull (world-frame unit dir from the shared senses)
        let (fdx, fdy) = senses.food_dir;
        if (fdx != 0.0 || fdy != 0.0) && fdx.is_finite() && fdy.is_finite() {
            let rel = wrap_pi(fdy.atan2(fdx) - dir);
            steer += rel * fs * QUEEN_FEED_TURN;
        }
        let o = &mut self.organisms[idx];
        o.dir = wrap_pi(dir + steer);
        if o.last_outputs.len() == NET_OUTPUTS {
            o.last_outputs[0] = motor.turn.clamp(-1.0, 1.0);
            o.last_outputs[1] = di;
            o.last_outputs[2] = fs;
        }
    }

    /// Additive save payload for `SaveCore.queen`.
    fn queen_snapshot(&self) -> Option<queen::QueenSnapshot> {
        if !self.queen.alive && self.queen.org_id == 0 {
            return None;
        }
        let brain = self.queen.brain.as_ref().map(|b| queen::QueenBrainSnapshot {
            identity: b.identity(),
            stats: b.stats(),
            bytes: b.snapshot(),
        });
        Some(queen::QueenSnapshot {
            alive: self.queen.alive,
            org_id: self.queen.org_id,
            crowned_tick: self.queen.crowned_tick,
            death_cause: self.queen.death_cause,
            energy_history: self.queen.energy_history.clone(),
            brain_ticks: self.queen.brain_ticks,
            events: self.queen_events.clone(),
            config: self.queen_config,
            brain,
        })
    }

    fn rebuild_grids(&mut self) {
        self.org_grid.clear();
        self.food_grid.clear();
        for (idx, o) in self.organisms.iter().enumerate() {
            if o.alive {
                self.org_grid.insert(o.x, o.y, idx as u32);
            }
        }
        for (idx, f) in self.foods.iter().enumerate() {
            if f.alive {
                self.food_grid.insert(f.x, f.y, idx as u32);
            }
        }
    }

    // ------------------------------------------------------ per-organism core

    fn update_one(&mut self, i: usize, dt: f32) {
        let (px, py, dir, speed, energy, max_e, age, traits, alive) = {
            let o = &self.organisms[i];
            (
                o.x, o.y, o.dir, o.speed, o.energy, o.max_energy, o.age,
                o.genotype.traits, o.alive,
            )
        };
        if !alive {
            return;
        }
        // sanitize snapshot state (corruption may never propagate)
        let px = if px.is_finite() { px } else { self.world.width * 0.5 };
        let py = if py.is_finite() { py } else { self.world.height * 0.5 };
        let dir = if dir.is_finite() { dir } else { 0.0 };
        let speed = if speed.is_finite() { speed.max(0.0) } else { 0.0 };
        let energy = if energy.is_finite() { energy } else { 0.0 };

        // sensors, filled in INPUT_LABELS order, normalized to [-1, 1]
        let mut inp = [0.0f32; NET_INPUTS];
        let _ = self.sense_food(px, py, dir, &traits, &mut inp);
        self.sense_peers(i, px, py, dir, &traits, &mut inp);
        let _ = self.sense_hazard(px, py, dir, &mut inp);
        // ambient signal blend (documented slot: food-intensity slot 2; the
        // 11-input count stays UNBROKEN — no slot is added, and existing
        // genotypes/saves just get a mildly modulated food intensity).
        // Blend strength: up to SIGNAL_SENSE_BLEND of the QueenActivity
        // sample at this organism's position.
        {
            let s = self.signals.sample(queen::QUEEN_ACTIVITY_KIND, px, py);
            if s.is_finite() {
                let bl = inp[2] + SIGNAL_SENSE_BLEND * s.clamp(0.0, 1.0);
                inp[2] = bl.clamp(0.0, 1.0);
            }
        }
        inp[9] = ((energy / max_e.max(1.0)) * 2.0 - 1.0).clamp(-1.0, 1.0);
        let lifespan = traits.lifespan.max(1) as f32;
        inp[10] = ((age as f32 / lifespan) * 2.0 - 1.0).clamp(-1.0, 1.0);
        for v in inp.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // controller: fill last_inputs / last_outputs, run the network.
        // The queen WITH a plugged brain skips the evolved net: her
        // motor comes from `queen_step_brain` (documented ordering: brain
        // ticks BEFORE update_one; the recorded turn / thrust / feed_seek
        // slots are applied through the ordinary actuation below).
        let queen_brain_steer = self.queen.alive
            && self.queen.brain.is_some()
            && self.organisms[i].id == self.queen.org_id as u64;
        let mut out = [0.0f32; NET_OUTPUTS];
        {
            let o = &mut self.organisms[i];
            o.last_inputs.copy_from_slice(&inp);
            if queen_brain_steer {
                let m = self.last_queen_motor;
                out[0] = m.turn.clamp(-1.0, 1.0);
                out[1] = m.move_intensity.clamp(0.0, 1.0);
                out[2] = m.feed_seek.clamp(0.0, 1.0);
            } else {
                think(&o.genotype.net, &inp, &mut out);
            }
        }
        for v in out.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // act: turn proportional to out[0], smoothed thrust toward out[1]
        {
            let o = &mut self.organisms[i];
            o.last_outputs.copy_from_slice(&out);
            o.dir = wrap_pi(dir + sanitize(out[0]) * 0.12);
            let target = sanitize(out[1]).max(0.0) * traits.max_speed;
            o.speed = sanitize(speed + (target - speed) * 0.25);
            o.x = sanitize(px + o.dir.cos() * o.speed * dt);
            o.y = sanitize(py + o.dir.sin() * o.speed * dt);
            o.age += 1;
            if o.repr_cooldown > 0 {
                o.repr_cooldown -= 1;
            }
        }

        // bounds & obstacle collision
        let (bx, by, hit) = self.apply_bounds(self.organisms[i].x, self.organisms[i].y);
        {
            let o = &mut self.organisms[i];
            o.x = bx;
            o.y = by;
            if hit {
                o.speed = 0.0;
            }
        }
        self.resolve_obstacles(i);

        // consumption then energy accounting (ledger: only applied deltas)
        let gained = self.eat_food(i);
        {
            let o = &mut self.organisms[i];
            let pre = o.energy;
            o.energy = (pre + gained).min(o.max_energy).max(0.0);
            self.step_gain += (o.energy - pre).max(0.0) as f64;
        }
        let pre_drain = self.organisms[i].energy;
        self.apply_drain(i, dt);
        self.step_drain += (pre_drain - self.organisms[i].energy).max(0.0) as f64;
    }

    fn sense_food(
        &mut self,
        px: f32,
        py: f32,
        dir: f32,
        t: &alife_common::Traits,
        inp: &mut [f32; NET_INPUTS],
    ) -> Option<(f32, f32, f32)> {
        let search = (t.sensor_range * 1.5).max(1.0);
        self.food_grid.query_into(px, py, search, &mut self.scratch_food);
        let mut best: Option<(f32, f32, f32)> = None; // (dx, dy, energy)
        let mut best_d2 = search * search;
        for k in 0..self.scratch_food.len() {
            let fj = self.scratch_food[k] as usize;
            if fj >= self.foods.len() {
                continue;
            }
            let f = &self.foods[fj];
            if !f.alive {
                continue;
            }
            let (dx, dy) = self.wdelta(f.x - px, f.y - py);
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = Some((dx, dy, f.energy));
            }
        }
        if let Some((dx, dy, energy)) = best {
            let d = best_d2.sqrt();
            let rel = wrap_pi(dy.atan2(dx) - dir);
            inp[0] = rel.cos();
            inp[1] = rel.sin();
            // intensity spun up/down by that item's energy richness
            let factor =
                ((energy / self.params.food_energy.max(1.0)).clamp(0.0, 2.0) * 0.5).clamp(0.0, 1.0);
            inp[2] = (1.0 / (1.0 + d / 120.0)) * factor;
            return Some((dx, dy, d));
        }
        None
    }

    fn sense_peers(
        &mut self,
        i: usize,
        px: f32,
        py: f32,
        dir: f32,
        t: &alife_common::Traits,
        inp: &mut [f32; NET_INPUTS],
    ) {
        let r = t.sensor_range.max(1.0);
        self.org_grid.query_into(px, py, r, &mut self.scratch_peer);
        let mut best: Option<(f32, f32)> = None;
        let mut best_d2 = r * r;
        let mut count = 0u32;
        for k in 0..self.scratch_peer.len() {
            let j = self.scratch_peer[k] as usize;
            if j == i || j >= self.organisms.len() {
                continue;
            }
            let o2 = &self.organisms[j];
            if !o2.alive {
                continue;
            }
            let (dx, dy) = self.wdelta(o2.x - px, o2.y - py);
            let d2 = dx * dx + dy * dy;
            if d2 <= r * r {
                count += 1;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = Some((dx, dy));
                }
            }
        }
        if let Some((dx, dy)) = best {
            let rel = wrap_pi(dy.atan2(dx) - dir);
            inp[3] = rel.cos();
            inp[4] = rel.sin();
        }
        inp[5] = (count as f32 / 8.0).min(1.0);
    }

    fn hsample(&self, x: f32, y: f32) -> f32 {
        let v = self.world.sample_hazard(x, y);
        if v.is_finite() { v.clamp(0.0, 2.0) } else { 0.0 }
    }

    fn sense_hazard(&self, px: f32, py: f32, dir: f32, inp: &mut [f32; NET_INPUTS]) -> (f32, f32) {
        let h = self.hsample(px, py);
        inp[6] = (h * 2.0).clamp(0.0, 1.0);
        // hazard "up-gradient": samples at +/- 8 units on both axes
        let gx = self.hsample(px + 8.0, py) - self.hsample(px - 8.0, py);
        let gy = self.hsample(px, py + 8.0) - self.hsample(px, py - 8.0);
        let gmag = (gx * gx + gy * gy).sqrt();
        if gmag > 1e-4 {
            // relative (net-input frame) AND world-frame direction returned:
            // the world vector is what the queen brain's threat-direction
            // channels consume (documented: hazard gradient = threat direction)
            let rel = wrap_pi(gy.atan2(gx) - dir);
            inp[7] = rel.cos();
            inp[8] = rel.sin();
            (gx / gmag, gy / gmag)
        } else {
            (0.0, 0.0)
        }
    }

    fn apply_bounds(&self, x: f32, y: f32) -> (f32, f32, bool) {
        let w = self.world.width.max(1.0);
        let h = self.world.height.max(1.0);
        let x = if x.is_finite() { x } else { w * 0.5 };
        let y = if y.is_finite() { y } else { h * 0.5 };
        if self.params.wrapping {
            (x.rem_euclid(w), y.rem_euclid(h), false)
        } else {
            let mut hit = false;
            let nx = if x < 0.0 {
                hit = true;
                0.0
            } else if x > w {
                hit = true;
                w
            } else {
                x
            };
            let ny = if y < 0.0 {
                hit = true;
                0.0
            } else if y > h {
                hit = true;
                h
            } else {
                y
            };
            (nx, ny, hit)
        }
    }

    fn wrap_pos(&self, x: f32, y: f32) -> (f32, f32) {
        let (nx, ny, _) = self.apply_bounds(x, y);
        (nx, ny)
    }

    /// Wrapping-aware delta between two world points.
    fn wdelta(&self, dx: f32, dy: f32) -> (f32, f32) {
        let dx = if dx.is_finite() { dx } else { 0.0 };
        let dy = if dy.is_finite() { dy } else { 0.0 };
        if self.params.wrapping {
            let w = self.world.width.max(1.0);
            let h = self.world.height.max(1.0);
            let wx = if dx > w * 0.5 { dx - w } else if dx < -w * 0.5 { dx + w } else { dx };
            let wy = if dy > h * 0.5 { dy - h } else if dy < -h * 0.5 { dy + h } else { dy };
            (wx, wy)
        } else {
            (dx, dy)
        }
    }

    fn resolve_obstacles(&mut self, i: usize) {
        let orad = organism_radius(&self.organisms[i].genotype.traits);
        for k in 0..self.edits.obstacles.len() {
            let obs = self.edits.obstacles[k];
            let (o_x, o_y) = {
                let o = &self.organisms[i];
                (o.x, o.y)
            };
            let dx = o_x - obs.x;
            let dy = o_y - obs.y;
            let rsum = obs.r + orad;
            if dx * dx + dy * dy < rsum * rsum {
                let d = (dx * dx + dy * dy).sqrt().max(1e-3);
                // push out along the normal and cancel speed
                let (px, py) =
                    self.wrap_pos(obs.x + (dx / d) * (rsum + 0.5), obs.y + (dy / d) * (rsum + 0.5));
                let o = &mut self.organisms[i];
                o.x = sanitize(px);
                o.y = sanitize(py);
                o.speed = 0.0;
            }
        }
    }

    fn eat_food(&mut self, i: usize) -> f32 {
        let pick_r = organism_radius(&self.organisms[i].genotype.traits) + PICKUP_RADIUS;
        let (ox, oy) = {
            let o = &self.organisms[i];
            (o.x, o.y)
        };
        self.food_grid.query_into(ox, oy, pick_r, &mut self.scratch_food);
        let reach2 = pick_r * pick_r;
        let mut gained = 0.0f32;
        for k in 0..self.scratch_food.len() {
            let fj = self.scratch_food[k] as usize;
            if fj >= self.foods.len() {
                continue;
            }
            let f = self.foods[fj]; // Copy
            if !f.alive {
                continue;
            }
            let (dx, dy) = self.wdelta(f.x - ox, f.y - oy);
            if dx * dx + dy * dy <= reach2 {
                // exclusive consumption: first eater claims the item
                gained += f.energy.max(0.0);
                self.foods[fj].alive = false;
            }
        }
        gained
    }

    /// World-hazard + zone + fire pressure at a point (energy/s, pulse
    /// applied). Used for drain and for post-hoc death classification.
    fn hazard_pressure(&self, x: f32, y: f32) -> f32 {
        let h_raw = self.hsample(x, y);
        let h = if self.world.hazard_pulse_period > 0.05 {
            let pulse = 0.15
                + 1.7
                    * (0.5
                        + 0.5
                            * (self.step_count as f32 / STEP_HZ * TAU
                                / self.world.hazard_pulse_period)
                                .sin());
            h_raw * pulse
        } else {
            h_raw
        };
        let mut zone_dps = 0.0f32;
        for zone in &self.edits.hazards {
            let dx = x - zone.x;
            let dy = y - zone.y;
            if dx * dx + dy * dy <= zone.r * zone.r {
                zone_dps += zone.dps.max(0.0) * self.zone_pulse_scale(zone);
            }
        }
        h * 6.0 + zone_dps + self.weather.fire_dps(x, y)
    }

    fn apply_drain(&mut self, i: usize, dt: f32) {
        let (ox, oy) = {
            let o = &self.organisms[i];
            (o.x, o.y)
        };
        let t = self.organisms[i].genotype.traits;
        let spd = self.organisms[i].speed;

        // weather modifiers (documented: deliberately small)
        let rain_i = self.weather.rain(ox, oy);
        let move_mult = if rain_i > 0.6 { 1.05 } else { 1.0 }; // rain waste
        let washed = if rain_i > 0.6 { 0.02 } else { 0.0 }; // "washed out"
        let drought = if self.weather.drought_edge(ox, oy) > 0.5 { 0.03 } else { 0.0 };

        let mut drain = self.params.metabolism_base * t.metabolism
            + self.params.movement_cost * spd * t.efficiency * move_mult
            + self.params.sensing_cost * (t.sensor_range / 100.0).sqrt()
            + washed
            + drought;
        if !drain.is_finite() {
            drain = 0.0;
        }

        // hazard drain is skipped entirely for protected lineages (god mode);
        // weather fire counts as hazard pressure too, so it is skipped as well
        if !self.protected.contains(&self.organisms[i].lineage) {
            drain += self.hazard_pressure(ox, oy);
        }
        // queen metabolism multiplier (documented: drain scales by the
        // config; 1.5 = she loses 1.5x what an identical body would)
        let is_queen = self.queen.alive && self.organisms[i].id == self.queen.org_id as u64;
        let qcfg = self.queen_config;
        if is_queen {
            drain *= qcfg.metabolism_mult.max(0.0);
        }
        // QUEEN SURVIVAL FLOOR: below `survival_floor` of her max energy she
        // enters the crisis state — drain is scaled by `crisis_drain_scale`
        // (metabolic saving) and a slow recovery trickle adds
        // `survival_regen_per_sec` of her maximum energy back each second, so
        // she always survives and can re-feed. Not a god shield: she still
        // experiences the low-energy state her brain senses.
        let crisis = is_queen
            && self.organisms[i].energy
                < self.organisms[i].max_energy.max(1.0) * qcfg.survival_floor.max(0.0);
        if crisis {
            drain *= qcfg.crisis_drain_scale.max(0.0).min(1.0);
        }
        drain = drain.max(0.0);
        let o = &mut self.organisms[i];
        o.energy -= drain * dt;
        if crisis {
            let max_e = o.max_energy.max(1.0);
            let regen = max_e * qcfg.survival_regen_per_sec.max(0.0) * dt;
            o.energy = (o.energy + regen).min(o.max_energy.max(1.0));
            self.step_gain += regen as f64;
        }
        if !o.energy.is_finite() || o.energy <= 0.0 {
            o.energy = 0.0;
        }
    }

    /// Pulsing hazard zones oscillate 0.15x..1.85x of dps with their period.
    fn zone_pulse_scale(&self, zone: &HazardZone) -> f32 {
        if zone.pulse_period > 0.05 {
            let phase = zone.phase + (self.step_count as f32 / STEP_HZ) * (TAU / zone.pulse_period);
            0.15 + 1.7 * (0.5 + 0.5 * phase.sin())
        } else {
            1.0
        }
    }

    // ---------------------------------------------------------- reproduction

    /// Reproduction gate on the neural output. Documented neutral lenience:
    /// outputs within +/-0.05 of zero count as willing (ties are float noise,
    /// not refusal); below -0.05 the network declines.
    pub fn gate_allows(out2: f32) -> bool {
        out2 > 0.0 || out2.abs() <= 0.05
    }

    fn try_reproduce(&mut self, i: usize, events: &mut StepEvents) {
        // the queen with a plugged brain never breeds through the evolved
        // gate (her slot 2 records feed_seek; deliberate steering authority
        // replaces the evolved repro decision)
        let queen_brainrepro = self.queen.alive
            && self.queen.brain.is_some()
            && self.organisms[i].id == self.queen.org_id as u64;
        let gate_ok = {
            let o = &self.organisms[i];
            o.alive
                && !queen_brainrepro
                && o.repr_cooldown == 0
                && o.age >= self.params.min_repro_age_steps
                && o.energy >= o.genotype.traits.repr_threshold * o.max_energy
                && Self::gate_allows(o.last_outputs.get(2).copied().unwrap_or(0.0))
        };
        if !gate_ok {
            return;
        }
        if self.organisms.len() + self.scratch_newborns.len() >= self.params.max_population as usize {
            return;
        }
        let (parent_id, px, py, parent_dir, parent_gen, parent_lineage, parent_radius) = {
            let o = &self.organisms[i];
            (
                o.id,
                o.x,
                o.y,
                o.dir,
                o.generation,
                o.lineage,
                organism_radius(&o.genotype.traits),
            )
        };
        let child_gen = mutate_genotype(&self.organisms[i].genotype, &mut self.rng, &self.params);
        let new_species =
            genotype_distance(&child_gen, &self.organisms[i].genotype) > NEW_SPECIES_DISTANCE;
        let lineage = if new_species {
            let l = self.next_lineage.max(1);
            self.next_lineage = self.next_lineage.wrapping_add(1).max(1);
            l
        } else {
            parent_lineage
        };
        let max_e = max_energy_for(&child_gen.traits);
        let cost = (self.organisms[i].genotype.traits.repr_cost * self.organisms[i].max_energy).max(0.0);
        // placement adjacent to the parent, heading parent-ish
        let spawn_ang = wrap_pi(parent_dir + self.rng.gauss_sigma(0.6));
        let spawn_r = parent_radius * 3.0 + self.rng.range(0.0, 8.0);
        let (cx, cy) = self.wrap_pos(px + spawn_ang.cos() * spawn_r, py + spawn_ang.sin() * spawn_r);
        let child = Organism {
            id: self.next_id,
            x: cx,
            y: cy,
            dir: spawn_ang,
            speed: 0.0,
            energy: (cost * 0.9).min(max_e * 0.4).clamp(1.0, max_e),
            max_energy: max_e,
            age: 0,
            generation: parent_gen + 1,
            genotype: child_gen,
            lineage,
            mom: Some(parent_id),
            dad: None,
            offspring_count: 0,
            alive: true,
            repr_cooldown: self.params.reproduction_cooldown_steps,
            last_inputs: vec![0.0; NET_INPUTS],
            last_outputs: vec![0.0; NET_OUTPUTS],
        };
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.record_born(child.id, child.lineage, child.generation, Some(parent_id));
        events.births.push(BirthEvent {
            step: self.step_count,
            child_id: child.id,
            parent_id,
            species_id: child.lineage,
            x: child.x,
            y: child.y,
            generation: child.generation,
        });
        // ledger: credit the child's born energy; parent cost set below
        self.step_birth_credit += child.energy as f64;
        self.scratch_newborns.push(child);
        let o = &mut self.organisms[i];
        let pre_cost = o.energy;
        o.energy -= cost;
        if o.energy < 0.0 {
            o.energy = 0.0;
        }
        o.repr_cooldown = self.params.reproduction_cooldown_steps;
        o.offspring_count += 1;
        self.step_birth_cost += (pre_cost - o.energy).max(0.0) as f64;
    }

    // -------------------------------------------------------------- spawning

    pub fn spawn_founder(&mut self) -> u64 {
        let x = self.rng.range(0.0, self.world.width.max(1.0));
        let y = self.rng.range(0.0, self.world.height.max(1.0));
        self.spawn_founder_at(x, y)
    }

    /// Spawn a fresh founder organism with a random genome at a given cell.
    /// Spawns an entirely new lineage id (god summon founders). RNG order is
    /// identical to `spawn_founder` (genotype, dir) so both are interchangeable.
    pub fn spawn_founder_at(&mut self, x: f32, y: f32) -> u64 {
        let genotype = random_genotype(&mut self.rng);
        let max_e = max_energy_for(&genotype.traits);
        let (x, y) = self.wrap_pos(x, y);
        let lineage = self.next_lineage.max(1);
        self.next_lineage = self.next_lineage.wrapping_add(1).max(1);
        let o = Organism {
            id: self.next_id,
            x,
            y,
            dir: self.rng.range(0.0, TAU),
            speed: 0.0,
            energy: max_e * 0.55,
            max_energy: max_e,
            age: 0,
            generation: 0,
            genotype,
            lineage,
            mom: None,
            dad: None,
            offspring_count: 0,
            alive: true,
            repr_cooldown: 0,
            last_inputs: vec![0.0; NET_INPUTS],
            last_outputs: vec![0.0; NET_OUTPUTS],
        };
        let id = o.id;
        self.record_born(id, lineage, 0, None);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.organisms.push(o);
        id
    }

    fn record_born(&mut self, id: u64, species_id: u32, generation: u32, mom: Option<u64>) {
        self.records.insert(
            id,
            LineageRecord {
                id,
                mom,
                dad: None,
                generation,
                species_id,
                born_step: self.step_count,
                death_step: None,
                cause: None,
            },
        );
        self.evict_records();
    }

    /// Keeps the records map bounded at [`RECORDS_CAP`]. Only already-dead
    /// records are evicted (oldest death first), and the full ancestry of
    /// every living organism is protected (mom-chains walked before
    /// eviction), so living lineages never lose their family tree. Fully
    /// deterministic: the eviction decision depends only on the record set.
    fn evict_records(&mut self) {
        let cap = RECORDS_CAP + RECORDS_SLACK;
        if self.records.len() <= cap {
            return;
        }
        // 1. protected set: every living organism and its ancestor chain
        self.scratch_set.clear();
        for o in &self.organisms {
            let mut cur = Some(o.id);
            while let Some(id) = cur {
                if self.scratch_set.contains(&id) {
                    break;
                }
                self.scratch_set.insert(id);
                cur = self.records.get(&id).and_then(|r| r.mom);
            }
        }
        // 2. evictable candidates: dead and unprotected, oldest death first
        self.scratch_pairs.clear();
        for (id, r) in &self.records {
            if r.death_step.is_some() && !self.scratch_set.contains(id) {
                self.scratch_pairs.push((r.death_step.unwrap(), *id));
            }
        }
        if self.scratch_pairs.is_empty() {
            return;
        }
        self.scratch_pairs.sort_unstable();
        let remove = self.records.len().saturating_sub(cap);
        let mut removed = 0usize;
        for (_, id) in self.scratch_pairs.iter() {
            if removed >= remove {
                break;
            }
            if self.records.remove(id).is_some() {
                removed += 1;
            }
        }
    }

    // ------------------------------------------------------------------ food

    fn push_food(&mut self, x: f32, y: f32) -> bool {
        if self.foods.len() >= self.params.max_food as usize {
            return false;
        }
        let (x, y) = self.wrap_pos(x, y);
        let var = self.params.food_energy_variance.clamp(0.0, 4.0);
        let base = self.params.food_energy.max(1.0);
        let energy = (base * (1.0 + self.rng.range(-var, var))).clamp(1.0, 400.0);
        self.foods.push(Food { id: self.next_id, x, y, energy, alive: true });
        self.next_id = self.next_id.wrapping_add(1).max(1);
        true
    }

    /// God-mode food: exact energy per item, no rng variance (placement only).
    fn spawn_food_exact(&mut self, x: f32, y: f32, energy: f32) -> bool {
        if self.foods.len() >= self.params.max_food as usize {
            return false;
        }
        let (x, y) = self.wrap_pos(x, y);
        self.foods.push(Food { id: self.next_id, x, y, energy: energy.clamp(1.0, 400.0), alive: true });
        self.next_id = self.next_id.wrapping_add(1).max(1);
        true
    }

    fn spawn_food_random(&mut self) -> bool {
        if self.foods.len() >= self.params.max_food as usize {
            return false;
        }
        let x = self.rng.range(0.0, self.world.width.max(1.0));
        let y = self.rng.range(0.0, self.world.height.max(1.0));
        // weather-coupled fertility: wet zones spawn more, drought zones less
        let fert = self.effective_fertility(x, y);
        if !self.rng.chance((fert * 2.0).clamp(0.0, 1.0)) {
            return false;
        }
        self.push_food(x, y)
    }

    /// Weather-coupled fertility at a point:
    /// `fertility * clamp(1.0 + 0.35 * wet, 0.5..1.7) + boosts`, clamped to
    /// 0..3. Neutral (wet=0) multiplies by exactly 1.0; full rain (wet→1)
    /// gives +35% food spawn ("+~30%" spec intent); drought (wet→-1) down to
    /// 0.5..0.65. Hazard is intentionally NOT weather-coupled. (Note: the
    /// literal task formula `0.5 + 0.9*(1+wet)` yields 1.4 at neutral and
    /// only +21% at full rain — contradicts the stated +30% coupling, so the
    /// consistent neutral-preserving form above is used; documented.)
    pub fn effective_fertility(&self, x: f32, y: f32) -> f32 {
        let base = self.world.sample_fertility(x, y);
        let base = if base.is_finite() { base.clamp(0.0, 3.0) } else { 0.0 };
        let w = self.weather.wet(x, y);
        let mult = (1.0 + 0.35 * w).clamp(0.5, 1.7);
        base * mult + self.weather.fert_boost(x, y)
    }

    /// Patch spawn: pick a patch weighted by richness, drop a point inside it.
    fn spawn_food_patch(&mut self) -> bool {
        if self.foods.len() >= self.params.max_food as usize || self.edits.patches.is_empty() {
            return false;
        }
        let total: f32 = self.edits.patches.iter().map(|p| p.richness.max(0.001)).sum();
        let total = total.max(1e-6);
        let mut pick = self.rng.next_f32() * total;
        let mut chosen = (0.0f32, 0.0f32, 10.0f32);
        for p in &self.edits.patches {
            let w = p.richness.max(0.001);
            if pick <= w {
                chosen = (p.x, p.y, p.r);
                break;
            }
            pick -= w;
        }
        let a = self.rng.range(0.0, TAU);
        let rr = self.rng.next_f32().sqrt() * chosen.2.max(1.0);
        self.push_food(chosen.0 + a.cos() * rr, chosen.1 + a.sin() * rr)
    }

    fn regen_food(&mut self, dt: f32) {
        let base = self.params.food_regen_per_sec.max(0.0) * dt;
        self.food_acc_base = (self.food_acc_base + base).min(FOOD_ACC_CAP);
        while self.food_acc_base >= 1.0 {
            self.food_acc_base -= 1.0;
            if !self.spawn_food_random() {
                break;
            }
        }
        let richness: f32 = self
            .edits
            .patches
            .iter()
            .map(|p| p.richness.max(0.0).min(1_000.0))
            .sum();
        let patch_rate = richness * self.params.food_regen_per_sec.max(0.0) * PATCH_REGEN_SHARE * dt;
        self.food_acc_patch = (self.food_acc_patch + patch_rate).min(FOOD_ACC_CAP);
        while self.food_acc_patch >= 1.0 {
            self.food_acc_patch -= 1.0;
            if !self.spawn_food_patch() {
                break;
            }
        }
    }

    // ----------------------------------------------------------------- stats

    fn refresh_stats(&mut self) {
        let n = self.organisms.len();
        let mut energy = 0.0f64;
        let mut age = 0.0f64;
        let mut gen = 0.0f64;
        let mut max_gen = 0u32;
        self.scratch_species.clear();
        for o in &self.organisms {
            if o.energy.is_finite() {
                energy += o.energy as f64;
            }
            age += o.age as f64;
            gen += o.generation as f64;
            max_gen = max_gen.max(o.generation);
            *self.scratch_species.entry(o.lineage).or_insert(0) += 1;
        }
        // recycled counts buffer: capacity carries over between steps
        let mut species_counts = std::mem::take(&mut self.scratch_counts);
        species_counts.clear();
        species_counts.extend(self.scratch_species.iter().map(|(k, v)| SpeciesCount {
            species_id: *k,
            count: *v,
        }));
        // deterministic order: count desc, species id asc
        species_counts.sort_by(|a, b| b.count.cmp(&a.count).then(a.species_id.cmp(&b.species_id)));
        let mut entropy = 0.0f64;
        let nf = n.max(1) as f64;
        for c in &species_counts {
            let p = c.count as f64 / nf;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        let div = if species_counts.len() > 1 {
            (entropy / (species_counts.len() as f64).ln()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let prev = std::mem::replace(
            &mut self.stats,
            SimStats {
                step: self.step_count,
                population: n as u32,
                max_population: self.params.max_population,
                total_births: self.total_births,
                total_deaths: self.total_deaths,
                births_window: self.births_window,
                deaths_window: self.deaths_window,
                avg_energy: (energy / nf) as f32,
                avg_age: (age / nf) as f32,
                avg_generation: (gen / nf) as f32,
                max_generation: max_gen,
                food_count: self.foods.iter().filter(|f| f.alive).count() as u32,
                species_counts,
                diversity_index: div as f32,
                wet_cover: self.weather.wet_cover(),
            },
        );
        // keep the previous stats' counts buffer for the next refresh
        self.scratch_counts = prev.species_counts;
    }

    // ---------------------------------------------------------- save / load

    pub fn snapshot(&self) -> SaveCore {
        // preallocated exactly: no 2x growth overshoot on a potentially
        // multi-thousand-record buffer
        let mut records: Vec<LineageRecord> = Vec::with_capacity(self.records.len());
        records.extend(self.records.values().cloned());
        records.sort_by_key(|r| (r.born_step, r.id));
        SaveCore {
            step_count: self.step_count,
            organisms: self.organisms.clone(),
            foods: self.foods.clone(),
            edits: self.edits.clone(),
            rng_state: self.rng_state(),
            records,
            edit_stack: self.edit_stack.clone(),
            next_id: self.next_id,
            next_lineage: self.next_lineage,
            total_births: self.total_births,
            total_deaths: self.total_deaths,
            births_window: self.births_window,
            deaths_window: self.deaths_window,
            births_ring: self.births_ring.clone(),
            deaths_ring: self.deaths_ring.clone(),
            food_acc_base: self.food_acc_base,
            food_acc_patch: self.food_acc_patch,
            weather: self.weather.clone(),
            interventions: self.interventions.clone(),
            protected: self.protected.clone(),
            total_starve: self.total_starve,
            total_age: self.total_age,
            total_hazard: self.total_hazard,
            starve_ring: self.starve_ring.clone(),
            queen: self.queen_snapshot(),
            ecology: ecology::EcologySave {
                state: self.ecology.clone(),
                rng: self.ecology_rng,
            },
            age_ring: self.age_ring.clone(),
            hazard_ring: self.hazard_ring.clone(),
            starve_window: self.starve_window,
            age_window: self.age_window,
            hazard_window: self.hazard_window,
            weather_every: self.weather_every,
        }
    }

    pub fn from_save_core(params: &SimParams, world: WorldModel, save: &SaveCore) -> Simulation {
        let mut sim = Self::restore_common(params, world, save.step_count);
        // restore borrows the save instead of cloning whole collections:
        // alive organisms/foods are cloned one at a time, records are
        // shallow-POD and inserted directly.
        for r in &save.records {
            sim.records.insert(r.id, r.clone());
        }
        sim.organisms
            .reserve(save.organisms.iter().filter(|o| o.alive).count());
        for o in &save.organisms {
            if !o.alive {
                continue;
            }
            let mut o = o.clone();
            normalize_org(&mut o, sim.world.width, sim.world.height);
            sim.organisms.push(o);
        }
        sim.foods.reserve(save.foods.iter().filter(|f| f.alive).count());
        for f in &save.foods {
            if f.alive {
                // Food is Copy: no per-item allocation at all
                sim.foods.push(*f);
            }
        }
        sim.edits = save.edits.clone();
        sim.next_id = save.next_id.max(1);
        sim.next_lineage = save.next_lineage.max(1);
        sim.total_births = save.total_births;
        sim.total_deaths = save.total_deaths;
        sim.births_window = save.births_window;
        sim.deaths_window = save.deaths_window;
        sim.births_ring = fix_ring(&save.births_ring);
        sim.deaths_ring = fix_ring(&save.deaths_ring);
        sim.food_acc_base = sane_acc(save.food_acc_base);
        sim.food_acc_patch = sane_acc(save.food_acc_patch);
        for (k, e) in save.edit_stack.iter().enumerate() {
            if k < EDIT_STACK_CAP {
                sim.edit_stack.push(e.clone());
            }
        }
        // v2 weather + god-mode state (exact restore)
        sim.weather = save.weather.clone();
        sim.weather.ensure_valid();
        sim.weather_every = save.weather_every.max(WEATHER_EVERY_MIN);
        let s0 = save.step_count;
        sim.last_weather_step = s0;
        sim.next_weather_step = s0 + sim.weather_every as u64;
        sim.interventions = save.interventions.clone();
        if sim.interventions.len() > INTERV_CAP {
            sim.interventions.drain(0..sim.interventions.len() - INTERV_CAP);
        }
        sim.protected = save.protected.clone();
        sim.total_starve = save.total_starve;
        sim.total_age = save.total_age;
        sim.total_hazard = save.total_hazard;
        sim.starve_ring = fix_ring(&save.starve_ring);
        sim.age_ring = fix_ring(&save.age_ring);
        sim.hazard_ring = fix_ring(&save.hazard_ring);
        sim.starve_window = save.starve_window;
        sim.age_window = save.age_window;
        sim.hazard_window = save.hazard_window;
        set_rng_state(&mut sim.rng, save.rng_state, params.seed);
        // v2 ecology layer (additive; pre-ecology saves load the serde
        // default `EcologySave` and keep the freshly initialized state)
        sim.ecology = save.ecology.state.clone();
        sim.ecology_rng = save.ecology.rng;
        // v2 queen layer (additive; pre-queen saves keep the null default
        // slot). A brain is rebuilt ONLY through the registered factory —
        // honest `brain_restore_failed` telemetry otherwise (with the bytes
        // stashed for a later `set_queen_brain_factory` reclaim).
        sim.restore_queen_snapshot(save.queen.as_ref());
        sim.refresh_stats();
        sim
    }

    /// Applies `SaveCore.queen` (see [`Simulation::queen_snapshot`]). Dead
    /// queens restore dead (brain dropped, no respawn); an alive queen whose
    /// organism is missing from the restored population is demoted honestly
    /// (alive=false, `Other` cause). The brain is rebuilt only via the
    /// registered factory.
    fn restore_queen_snapshot(&mut self, qs: Option<&queen::QueenSnapshot>) {
        let Some(qs) = qs else {
            return; // never-crowned default slot is already exact
        };
        if !qs.alive && qs.org_id == 0 {
            return;
        }
        self.queen.org_id = qs.org_id;
        self.queen.crowned_tick = qs.crowned_tick;
        self.queen.death_cause = qs.death_cause;
        self.queen.energy_history.clone_from(&qs.energy_history);
        self.queen.cap_energy_history();
        self.queen.brain_ticks = qs.brain_ticks;
        self.queen_config = qs.config;
        self.queen_events.clone_from(&qs.events);
        queen::trim_events(&mut self.queen_events);
        self.queen.alive = qs.alive;
        if self.queen.alive {
            let org_ok = self
                .organisms
                .iter()
                .any(|o| o.alive && o.id == self.queen.org_id as u64);
            if !org_ok {
                self.queen.alive = false;
                self.queen.death_cause = Some(QueenDeath::Other);
            }
        }
        if self.queen.alive {
            if let Some(bs) = &qs.brain {
                if let Some(b) = self.build_brain_from(bs) {
                    self.queen.brain = Some(b);
                } else {
                    // honest telemetry: alive queen, unusable/missing brain
                    self.queen.brain_restore_failed = true;
                    self.queen.reclaim_brain = Some(bs.clone());
                }
            }
        }
    }

    /// Contracted individual-accessor restore path (v1 saves). Fields not
    /// covered by the argument list (undo stack, window rings, totals,
    /// weather, interventions) are derived or reset; counters are
    /// reconstructed from the largest ids seen.
    pub fn from_save(
        params: &SimParams,
        world: WorldModel,
        step_count: u64,
        organisms: Vec<Organism>,
        foods: Vec<Food>,
        edits: EnvironmentEdits,
        rng_state: u32,
        records: Vec<LineageRecord>,
    ) -> Simulation {
        let mut next_id = 1u64;
        for o in &organisms {
            next_id = next_id.max(o.id.wrapping_add(1));
        }
        for f in &foods {
            next_id = next_id.max(f.id.wrapping_add(1));
        }
        let mut next_lineage = 1u32;
        let mut dead_count = 0u64;
        for r in &records {
            next_lineage = next_lineage.max(r.species_id.wrapping_add(1));
            if r.death_step.is_some() {
                dead_count += 1;
            }
        }
        for o in &organisms {
            next_lineage = next_lineage.max(o.lineage.wrapping_add(1));
        }
        let births_total = records.len() as u64; // lower-bound approximation

        let mut sim = Self::restore_common(params, world, step_count);
        for r in records {
            sim.records.insert(r.id, r);
        }
        for mut o in organisms {
            if !o.alive {
                continue;
            }
            normalize_org(&mut o, sim.world.width, sim.world.height);
            sim.organisms.push(o);
        }
        for f in foods {
            if f.alive {
                sim.foods.push(f);
            }
        }
        sim.edits = edits;
        sim.next_id = next_id.max(1);
        sim.next_lineage = next_lineage.max(1);
        sim.total_births = births_total;
        sim.total_deaths = dead_count;
        set_rng_state(&mut sim.rng, rng_state, params.seed);
        sim.refresh_stats();
        sim
    }

    /// Blank sim already positioned at `step_count` (rings zeroed, weather
    /// freshly initialized and scheduled from the restored step).
    fn restore_common(params: &SimParams, world: WorldModel, step_count: u64) -> Simulation {
        let mut sim = Self::blank(params, world);
        sim.step_count = step_count;
        sim.last_weather_step = step_count;
        sim.next_weather_step = step_count + WEATHER_EVERY as u64;
        sim
    }

    // -------------------------------------------------------- editor actions

    pub fn spawn_organisms_n(&mut self, n: usize) {
        let room = (self.params.max_population as usize).saturating_sub(self.organisms.len());
        for _ in 0..n.min(room) {
            self.spawn_founder();
        }
        self.refresh_stats();
    }

    pub fn add_food_at(&mut self, x: f32, y: f32) -> bool {
        if self.foods.len() >= self.params.max_food as usize {
            return false;
        }
        let (x, y) = self.wrap_pos(x, y);
        let energy = self.params.food_energy.max(1.0);
        let food = Food { id: self.next_id, x, y, energy, alive: true };
        self.foods.push(food);
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.push_edit(EditEntry {
            mode: InteractionMode::Food,
            x,
            y,
            r: 0.0,
            added_food_ids: vec![id],
            added_foods: vec![Food { id, x, y, energy, alive: true }],
            ..EditEntry::default()
        });
        self.refresh_stats();
        true
    }

    pub fn add_obstacle_at(&mut self, x: f32, y: f32, r: f32) -> bool {
        let obs = Obstacle { x: sanitize(x), y: sanitize(y), r: r.clamp(2.0, 600.0) };
        self.edits.obstacles.push(obs);
        self.push_edit(EditEntry {
            mode: InteractionMode::Obstacle,
            x,
            y,
            r: obs.r,
            added_obstacles: vec![obs],
            ..EditEntry::default()
        });
        true
    }

    pub fn add_patch_at(&mut self, x: f32, y: f32, r: f32) -> bool {
        let patch = FoodPatch { x, y, r: r.clamp(8.0, 600.0), richness: 1.0 };
        self.edits.patches.push(patch);
        self.push_edit(EditEntry {
            mode: InteractionMode::Patch,
            x,
            y,
            r: patch.r,
            added_patches: vec![patch],
            ..EditEntry::default()
        });
        true
    }

    pub fn add_hazard_at(&mut self, x: f32, y: f32, r: f32) -> bool {
        let zone = HazardZone { x, y, r: r.clamp(8.0, 600.0), dps: 5.0, pulse_period: 0.0, phase: 0.0 };
        self.edits.hazards.push(zone);
        self.push_edit(EditEntry {
            mode: InteractionMode::Hazard,
            x,
            y,
            r: zone.r,
            added_hazards: vec![zone],
            ..EditEntry::default()
        });
        true
    }

    pub fn remove_items_at(&mut self, x: f32, y: f32, r: f32) -> bool {
        let within = |cx: f32, cy: f32, radius: f32| -> bool {
            let dx = cx - x;
            let dy = cy - y;
            dx * dx + dy * dy <= radius * radius
        };
        let mut entry = EditEntry {
            mode: InteractionMode::Erase,
            x,
            y,
            r,
            ..EditEntry::default()
        };
        let mut any = false;
        let obs = std::mem::take(&mut self.edits.obstacles);
        let (keep, rm): (Vec<Obstacle>, Vec<Obstacle>) = obs
            .into_iter()
            .partition(|o| !within(o.x, o.y, r));
        if !rm.is_empty() {
            any = true;
            entry.removed_obstacles = rm;
        }
        self.edits.obstacles = keep;

        let zones = std::mem::take(&mut self.edits.hazards);
        let (keep, rm): (Vec<HazardZone>, Vec<HazardZone>) = zones
            .into_iter()
            .partition(|h| !within(h.x, h.y, r));
        if !rm.is_empty() {
            any = true;
            entry.removed_hazards = rm;
        }
        self.edits.hazards = keep;

        let patches = std::mem::take(&mut self.edits.patches);
        let (keep, rm): (Vec<FoodPatch>, Vec<FoodPatch>) = patches
            .into_iter()
            .partition(|p| !within(p.x, p.y, r));
        if !rm.is_empty() {
            any = true;
            entry.removed_patches = rm;
        }
        self.edits.patches = keep;

        // foods in half the radius are picked up too
        let foods = std::mem::take(&mut self.foods);
        let (keep, rm): (Vec<Food>, Vec<Food>) = foods
            .into_iter()
            .filter(|f| f.alive)
            .partition(|f| !within(f.x, f.y, r * 0.5));
        if !rm.is_empty() {
            any = true;
            entry.removed_foods = rm;
        }
        self.foods = keep;

        if !any {
            return false;
        }
        self.push_edit(entry);
        self.refresh_stats();
        true
    }

    pub fn pop_undo(&mut self) -> bool {
        let Some(entry) = self.edit_stack.pop() else {
            return false;
        };
        // revert additions
        if !entry.added_food_ids.is_empty() {
            let ids = &entry.added_food_ids;
            self.foods.retain(|f| !ids.contains(&f.id));
        }
        for obs in &entry.added_obstacles {
            if let Some(pos) = self.edits.obstacles.iter().position(|v| v == obs) {
                self.edits.obstacles.remove(pos);
            }
        }
        for hz in &entry.added_hazards {
            if let Some(pos) = self.edits.hazards.iter().position(|v| v == hz) {
                self.edits.hazards.remove(pos);
            }
        }
        for pt in &entry.added_patches {
            if let Some(pos) = self.edits.patches.iter().position(|v| v == pt) {
                self.edits.patches.remove(pos);
            }
        }
        // restore removals
        self.edits.obstacles.extend(entry.removed_obstacles.iter().copied());
        self.edits.hazards.extend(entry.removed_hazards.iter().copied());
        self.edits.patches.extend(entry.removed_patches.iter().copied());
        self.foods.extend(entry.removed_foods.iter().copied());
        self.refresh_stats();
        if self.redo_stack.len() >= REDO_STACK_CAP {
            self.redo_stack.remove(0);
        }
        self.redo_stack.push(entry);
        true
    }

    /// Re-applies the most recently undone editor edit (the inverse of
    /// [`Simulation::pop_undo`]). Any new edit clears the redo stack. Redo
    /// state is session-local: it is not part of saves.
    pub fn redo_edit(&mut self) -> bool {
        let Some(entry) = self.redo_stack.pop() else {
            return false;
        };
        // re-remove what the op had removed (revert restored them)
        let rm = &entry.removed_obstacles;
        if !rm.is_empty() {
            self.edits.obstacles.retain(|o| !rm.contains(o));
        }
        let rm = &entry.removed_hazards;
        if !rm.is_empty() {
            self.edits.hazards.retain(|h| !rm.contains(h));
        }
        let rm = &entry.removed_patches;
        if !rm.is_empty() {
            self.edits.patches.retain(|p| !rm.contains(p));
        }
        let rm = &entry.removed_foods;
        if !rm.is_empty() {
            self.foods.retain(|f| !rm.contains(f));
        }
        // re-add what the op had added
        self.foods.extend(entry.added_foods.iter().copied());
        self.edits.obstacles.extend(entry.added_obstacles.iter().copied());
        self.edits.hazards.extend(entry.added_hazards.iter().copied());
        self.edits.patches.extend(entry.added_patches.iter().copied());
        self.refresh_stats();
        if self.edit_stack.len() >= EDIT_STACK_CAP {
            self.edit_stack.remove(0);
        }
        self.edit_stack.push(entry);
        true
    }

    fn push_edit(&mut self, entry: EditEntry) {
        if self.edit_stack.len() >= EDIT_STACK_CAP {
            self.edit_stack.remove(0);
        }
        self.redo_stack.clear();
        self.edit_stack.push(entry);
    }

    // ----------------------------------------------------------- live params

    pub fn apply_live_params(&mut self, patch: &LiveParams) {
        if let Some(v) = patch.max_population {
            self.params.max_population = v.max(10);
        }
        if let Some(v) = patch.max_food {
            self.params.max_food = v.max(10);
        }
        if let Some(v) = patch.food_regen_per_sec {
            self.params.food_regen_per_sec = v.clamp(0.0, 200.0);
        }
        if let Some(v) = patch.food_energy {
            self.params.food_energy = v.clamp(1.0, 120.0);
        }
        if let Some(v) = patch.metabolism_base {
            self.params.metabolism_base = v.clamp(0.01, 4.0);
        }
        if let Some(v) = patch.movement_cost {
            self.params.movement_cost = v.clamp(0.0, 0.2);
        }
        if let Some(v) = patch.sensing_cost {
            self.params.sensing_cost = v.clamp(0.0, 0.08);
        }
        if let Some(v) = patch.mutation_rate {
            self.params.mutation_rate = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.mutation_magnitude {
            self.params.mutation_magnitude = v.clamp(0.05, 8.0);
        }
        if let Some(v) = patch.big_jump_chance {
            self.params.big_jump_chance = v.clamp(0.0, 1.0);
        }
    }

    // ------------------------------------------- weather layer / god mode

    /// Weather tick, called at most once per `weather_every` steps. The delta
    /// is measured in sim time, so a skipped tick (lag / accelerated runs)
    /// catches up bounded (`MAX_DT` seconds max). Storm bolts are random
    /// impacts that consume the single sim RNG (documented determinism
    /// limitation: god events steer biology deterministically from then on).
    fn weather_tick(&mut self, _dt: f32) {
        if self.step_count < self.next_weather_step {
            return;
        }
        let interval = self.weather_every.max(WEATHER_EVERY_MIN) as u64;
        let elapsed = self.step_count.saturating_sub(self.last_weather_step) as f32 * DT;
        self.last_weather_step = self.step_count;
        self.next_weather_step = self.step_count + interval;
        // region effects are self-contained (rain/drought/storm/fire); the
        // expired list has no external objects to clean up
        self.scratch_expired.clear();
        self.weather.update(elapsed, self.step_count, &mut self.scratch_expired);
        self.scratch_expired.clear();

        // scheduled storm bolts
        let now = self.step_count;
        self.scratch_bolts.clear();
        for (i, e) in self.weather.effects.iter().enumerate() {
            if e.kind == EffectKind::Storm && e.pending_bolts > 0 && now >= e.next_bolt_at {
                let a = self.rng.range(0.0, TAU);
                let rr = self.rng.next_f32().sqrt() * e.r.max(1.0);
                let (bx, by) = self.wrap_pos(e.x + a.cos() * rr, e.y + a.sin() * rr);
                self.scratch_bolts.push((i, bx, by));
            }
        }
        let strikes_n = self.scratch_bolts.len();
        for k in 0..strikes_n {
            let (i, bx, by) = self.scratch_bolts[k];
            {
                let e = &mut self.weather.effects[i];
                e.pending_bolts = e.pending_bolts.saturating_sub(1);
                e.next_bolt_at = now + BOLT_INTERVAL_STEPS as u64;
            }
            self.do_lightning(bx, by);
        }
        self.scratch_bolts.clear();
    }

    /// Lightning semantics at a cell: organisms within LIGHTNING_RADIUS lose
    /// LIGHTNING_DMG energy (0-energied die at the next reap), nutrient ash
    /// boosts fertility in a small radius for BOLT_ASH_STEPS, flash pulses
    /// and the bolt enters the bounded recent-bolt history. Never panics.
    pub fn do_lightning(&mut self, x: f32, y: f32) -> u32 {
        let (x, y) = self.wrap_pos(x, y);
        let mut hit = 0u32;
        let r2 = LIGHTNING_RADIUS * LIGHTNING_RADIUS;
        for o in &mut self.organisms {
            if !o.alive {
                continue;
            }
            let dx = o.x - x;
            let dy = o.y - y;
            if dx * dx + dy * dy <= r2 {
                o.energy -= LIGHTNING_DMG;
                if !o.energy.is_finite() || o.energy <= 0.0 {
                    o.energy = 0.0;
                }
                hit += 1;
            }
        }
        if self.weather.fert_boosts.len() >= BOOSTS_CAP {
            self.weather.fert_boosts.remove(0);
        }
        self.weather.fert_boosts.push(FertBoost {
            x,
            y,
            r: BOLT_ASH_RADIUS,
            amount: BOLT_ASH_AMOUNT,
            until: self.step_count + BOLT_ASH_STEPS,
        });
        self.weather.flash = 1.0;
        if self.weather.recent_bolts.len() >= BOLTS_CAP {
            self.weather.recent_bolts.remove(0);
        }
        self.weather.recent_bolts.push((x, y, self.step_count));
        hit
    }

    fn push_effect(&mut self, e: RegionEffect) {
        self.scratch_expired.clear();
        self.weather.add_effect(e, &mut self.scratch_expired);
        self.scratch_expired.clear();
    }

    fn sane_pos(x: f32, y: f32) -> Result<(), String> {
        if x.is_finite() && y.is_finite() {
            Ok(())
        } else {
            Err(format!("invalid coordinates x={x} y={y}"))
        }
    }

    fn sane_radius(r: f32, min: f32, max: f32, what: &str) -> Result<f32, String> {
        if !r.is_finite() {
            return Err(format!("invalid {what} radius {r}"));
        }
        Ok(r.clamp(min, max))
    }

    /// Apply a god intervention. Invalid species ids / coords return Err;
    /// events never panic. The note summarizes the concrete state change.
    pub fn trigger(&mut self, ev: &GodEvent) -> Result<(), String> {
        let note = self.apply_god(ev)?;
        if self.interventions.len() >= INTERV_CAP {
            self.interventions.remove(0);
        }
        self.interventions.push(InterventionRecord {
            step: self.step_count,
            event: *ev,
            note,
        });
        self.refresh_stats();
        Ok(())
    }

    fn apply_god(&mut self, ev: &GodEvent) -> Result<String, String> {
        match *ev {
            GodEvent::Rain { x, y, r, duration_steps } => {
                Self::sane_pos(x, y)?;
                if duration_steps == 0 {
                    return Err("rain duration must be > 0".into());
                }
                let r = Self::sane_radius(r, 8.0, 2500.0, "rain")?;
                self.push_effect(RegionEffect {
                    x,
                    y,
                    r,
                    kind: EffectKind::Rain,
                    until: self.step_count + duration_steps as u64,
                    pending_bolts: 0,
                    next_bolt_at: 0,
                });
                Ok(format!("rain r={r} for {duration_steps} steps"))
            }
            GodEvent::Drought { x, y, r, duration_steps } => {
                Self::sane_pos(x, y)?;
                if duration_steps == 0 {
                    return Err("drought duration must be > 0".into());
                }
                let r = Self::sane_radius(r, 8.0, 2500.0, "drought")?;
                self.push_effect(RegionEffect {
                    x,
                    y,
                    r,
                    kind: EffectKind::Drought,
                    until: self.step_count + duration_steps as u64,
                    pending_bolts: 0,
                    next_bolt_at: 0,
                });
                Ok(format!("drought r={r} for {duration_steps} steps (food spawn -, drain +0.03)"))
            }
            GodEvent::Storm { x, y, r, duration_steps, lightning_bolts } => {
                Self::sane_pos(x, y)?;
                if duration_steps == 0 {
                    return Err("storm duration must be > 0".into());
                }
                if lightning_bolts > 1000 {
                    return Err(format!("too many lightning bolts ({lightning_bolts}, max 1000)"));
                }
                let r = Self::sane_radius(r, 8.0, 2500.0, "storm")?;
                self.push_effect(RegionEffect {
                    x,
                    y,
                    r,
                    kind: EffectKind::Storm,
                    until: self.step_count + duration_steps as u64,
                    pending_bolts: lightning_bolts,
                    next_bolt_at: self.step_count + BOLT_INTERVAL_STEPS as u64,
                });
                Ok(format!("storm r={r} for {duration_steps} steps, {lightning_bolts} bolts"))
            }
            GodEvent::Lightning { x, y } => {
                Self::sane_pos(x, y)?;
                let hit = self.do_lightning(x, y);
                Ok(format!("bolt: {hit} organisms hit, ash r={BOLT_ASH_RADIUS}"))
            }
            GodEvent::NutrientDump { x, y, r, energy } => {
                Self::sane_pos(x, y)?;
                if !energy.is_finite() || energy <= 0.0 {
                    return Err(format!("invalid nutrient energy {energy}"));
                }
                let r = Self::sane_radius(r, 2.0, 2500.0, "nutrient")?;
                let per = self.params.food_energy.max(1.0);
                let want = ((energy / per) as usize).max(1).min(100_000);
                let room = (self.params.max_food as usize).saturating_sub(self.foods.len());
                let n = want.min(room);
                for _ in 0..n {
                    let a = self.rng.range(0.0, TAU);
                    let rr = self.rng.next_f32().sqrt() * r;
                    let (fx, fy) = self.wrap_pos(x + a.cos() * rr, y + a.sin() * rr);
                    self.spawn_food_exact(fx, fy, per);
                }
                let unplaced = want.saturating_sub(n);
                if unplaced > 0 {
                    Ok(format!("+{n}/{want} food (maxFood cap; {unplaced} unplaced)"))
                } else {
                    Ok(format!("+{n} food"))
                }
            }
            GodEvent::SummonOrganism { x, y } => {
                Self::sane_pos(x, y)?;
                if self.organisms.len() >= self.params.max_population as usize {
                    return Err(format!(
                        "population at cap ({}/{}), summon rejected",
                        self.organisms.len(),
                        self.params.max_population
                    ));
                }
                let id = self.spawn_founder_at(x, y);
                Ok(format!("summoned organism #{id}"))
            }
            GodEvent::PurgeLineage { species_id } => {
                let exists = self.organisms.iter().any(|o| o.lineage == species_id)
                    || self.records.values().any(|r| r.species_id == species_id);
                if !exists {
                    return Err(format!("unknown species id {species_id}"));
                }
                // queen lifecycle rule: she is never removed by the purge
                let qid = self.queen_id().map(|q| q as u64);
                let (keep, purged): (Vec<Organism>, Vec<Organism>) =
                    std::mem::take(&mut self.organisms)
                        .into_iter()
                        .partition(|o| {
                            o.lineage != species_id
                                || qid == Some(o.id)
                        });
                self.organisms = keep;
                let n = purged.len();
                for o in purged {
                    if let Some(rec) = self.records.get_mut(&o.id) {
                        rec.death_step = Some(self.step_count);
                        rec.cause = Some(DeathCause::Hazard);
                    }
                    self.pending_deaths.push(DeathEvent {
                        step: self.step_count,
                        id: o.id,
                        species_id: o.lineage,
                        cause: DeathCause::Hazard,
                        x: o.x,
                        y: o.y,
                        generation: o.generation,
                        age: o.age,
                    });
                }
                Ok(format!("purged species {species_id}: {n} died"))
            }
            GodEvent::Catastrophe { kind } => match kind {
                CatastropheKind::Meteor => self.apply_meteor(),
                CatastropheKind::GlobalStorm => {
                    let r = (self.world.width * self.world.width
                        + self.world.height * self.world.height)
                        .sqrt()
                        * 0.6;
                    self.push_effect(RegionEffect {
                        x: self.world.width * 0.5,
                        y: self.world.height * 0.5,
                        r,
                        kind: EffectKind::Storm,
                        until: self.step_count + GLOBAL_STORM_STEPS,
                        pending_bolts: GLOBAL_STORM_BOLTS,
                        next_bolt_at: self.step_count + BOLT_INTERVAL_STEPS as u64,
                    });
                    Ok(format!("global storm: whole-world cloud, {GLOBAL_STORM_BOLTS} bolts over {GLOBAL_STORM_STEPS} steps"))
                }
                CatastropheKind::GlobalDrought => {
                    let r = (self.world.width * self.world.width
                        + self.world.height * self.world.height)
                        .sqrt()
                        * 0.6;
                    self.push_effect(RegionEffect {
                        x: self.world.width * 0.5,
                        y: self.world.height * 0.5,
                        r,
                        kind: EffectKind::Drought,
                        until: self.step_count + GLOBAL_DROUGHT_STEPS,
                        pending_bolts: 0,
                        next_bolt_at: 0,
                    });
                    Ok(format!("global drought: world-wide fertility - for {GLOBAL_DROUGHT_STEPS} steps"))
                }
            },
        }
    }

    /// Meteor: hard burn at a random impact (rng), r = METEOR_RADIUS:
    /// organisms in the zone die, food is gone, fertility is destroyed for a
    /// long burn window, and a fire hazard zone (dps = FIRE_DPS) rages.
    fn apply_meteor(&mut self) -> Result<String, String> {
        let margin = METEOR_RADIUS * 1.2;
        let (cx, cy) = if self.world.width > margin * 2.0 && self.world.height > margin * 2.0 {
            (
                self.rng.range(margin, self.world.width - margin),
                self.rng.range(margin, self.world.height - margin),
            )
        } else {
            (self.world.width * 0.5, self.world.height * 0.5)
        };
        let r2 = METEOR_RADIUS * METEOR_RADIUS;
        // organisms: burned (energy gone, Hazard death event next step).
        // queen lifecycle rule: she survives even a direct hit (she drains
        // through the same hazard fire afterwards and must recover)
        let qid = self.queen_id().map(|q| q as u64);
        let (remaining, burnt): (Vec<Organism>, Vec<Organism>) = std::mem::take(&mut self.organisms)
            .into_iter()
            .partition(|o| {
                if qid == Some(o.id) {
                    return true;
                }
                let dx = o.x - cx;
                let dy = o.y - cy;
                dx * dx + dy * dy > r2
            });
        let killed = burnt.len();
        self.organisms = remaining;
        for o in burnt {
            if let Some(rec) = self.records.get_mut(&o.id) {
                rec.death_step = Some(self.step_count);
                rec.cause = Some(DeathCause::Hazard);
            }
            self.pending_deaths.push(DeathEvent {
                step: self.step_count,
                id: o.id,
                species_id: o.lineage,
                cause: DeathCause::Hazard,
                x: o.x,
                y: o.y,
                generation: o.generation,
                age: o.age,
            });
        }
        // food: gone
        let mut burned_food = 0u32;
        for f in &mut self.foods {
            if f.alive {
                let dx = f.x - cx;
                let dy = f.y - cy;
                if dx * dx + dy * dy <= r2 {
                    f.alive = false;
                    burned_food += 1;
                }
            }
        }
        // fertility destruction (long-lived burn scar)
        if self.weather.fert_boosts.len() >= BOOSTS_CAP {
            self.weather.fert_boosts.remove(0);
        }
        self.weather.fert_boosts.push(FertBoost {
            x: cx,
            y: cy,
            r: METEOR_RADIUS,
            amount: METEOR_BURN_AMOUNT,
            until: self.step_count + METEOR_BURN_STEPS,
        });
        // long-lived fire hazard zone (weather effect, dps = FIRE_DPS)
        self.push_effect(RegionEffect {
            x: cx,
            y: cy,
            r: METEOR_RADIUS,
            kind: EffectKind::Fire,
            until: self.step_count + METEOR_FIRE_STEPS,
            pending_bolts: 0,
            next_bolt_at: 0,
        });
        Ok(format!(
            "meteor at ({cx:.0},{cy:.0}): {killed} organisms, {burned_food} food burned; fire for {} steps",
            METEOR_FIRE_STEPS
        ))
    }

    /// Toggle hazard immunity for a lineage. Returns the resulting state
    /// (true = protected). The engine skips hazard drain (world field, zones
    /// and weather fire) for protected organisms.
    pub fn protect_lineage(&mut self, species_id: u32, on: bool) -> bool {
        let has = self.protected.contains(&species_id);
        match (on, has) {
            (true, false) => {
                if self.protected.len() >= PROTECTED_CAP {
                    self.protected.remove(0);
                }
                self.protected.push(species_id);
                true
            }
            (false, true) => {
                self.protected.retain(|&l| l != species_id);
                false
            }
            _ => has,
        }
    }

    pub fn interventions(&self) -> &[InterventionRecord] {
        &self.interventions
    }

    pub fn protected_lines(&self) -> Vec<u32> {
        self.protected.clone()
    }

    // ------------------------------------------------------------- accessors

    /// Aggregate diagnostics for UI cards. Computed from existing stats +
    /// rolling cause counters (cheap, no world scans).
    pub fn diagnosis(&self) -> Diag {
        let s = &self.stats;
        let starvation_share_window = if s.deaths_window > 0 {
            (self.starve_window as f32 / s.deaths_window.max(1) as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // documented rule: population per remaining food item (clamped to 10);
        // > 1 means fewer food items than organisms (deficit risk).
        let food_deficit_ratio =
            (s.population as f32 / (s.food_count.saturating_add(1) as f32)).clamp(0.0, 10.0);
        Diag {
            population: s.population,
            births_window: s.births_window,
            deaths_window: s.deaths_window,
            food_count: s.food_count,
            avg_energy: s.avg_energy,
            avg_generation: s.avg_generation,
            starvation_share_window,
            food_deficit_ratio,
            wet_cover: self.weather.wet_cover(),
        }
    }

    /// Additive diagnostics (memory budget): backing storage sizes of the
    /// big collections. Read-only; never mutates simulation state.
    pub fn memory_probe(&self) -> MemProbe {
        MemProbe {
            organisms_capacity: self.organisms.capacity(),
            foods_capacity: self.foods.capacity(),
            records_len: self.records.len(),
            pending_deaths_capacity: self.pending_deaths.capacity(),
        }
    }

    /// Deaths in the last window by cause (history card).
    pub fn recent_deaths_breakdown(&self) -> Vec<(DeathCause, u32)> {
        vec![
            (DeathCause::Starvation, self.starve_window),
            (DeathCause::OldAge, self.age_window),
            (DeathCause::Hazard, self.hazard_window),
        ]
    }

    /// All-time death totals by cause: (starve, age, hazard).
    pub fn death_cause_totals(&self) -> (u64, u64, u64) {
        (self.total_starve, self.total_age, self.total_hazard)
    }

    /// Per-step energy ledger from the last `advance()`:
    /// (gain, drain, birth_cost, birth_credit, death_loss) in f64.
    /// Conservation identity:
    /// `delta_sum(organism energy) == gain - drain - birth_cost + birth_credit
    /// - death_loss` (death_loss = leftover energy of age/hazard corpses, not
    /// an engine leak; founder spawns and god events inject energy outside of
    /// stepped dynamics).
    pub fn step_energy_ledger(&self) -> (f64, f64, f64, f64, f64) {
        (
            self.step_gain,
            self.step_drain,
            self.step_birth_cost,
            self.step_birth_credit,
            self.step_death_loss,
        )
    }

    pub fn organism_by_id(&self, id: u64) -> Option<&Organism> {
        self.organisms.iter().find(|o| o.id == id)
    }

    pub fn record_by_id(&self, id: u64) -> Option<LineageRecord> {
        self.records.get(&id).cloned()
    }

    pub fn pick_nearest(&self, x: f32, y: f32, r: f32) -> Option<u64> {
        if !(x.is_finite() && y.is_finite() && r.is_finite()) {
            return None;
        }
        let mut best = None;
        let mut best_d2 = f32::MAX;
        for o in &self.organisms {
            if !o.alive {
                continue;
            }
            let (dx, dy) = self.wdelta(o.x - x, o.y - y);
            let d2 = dx * dx + dy * dy;
            let reach = r + organism_radius(&o.genotype.traits);
            if d2 <= reach * reach && d2 < best_d2 {
                best_d2 = d2;
                best = Some(o.id);
            }
        }
        best
    }

    /// Snapshot of the internal mulberry32 state (odd as consumed by Rng).
    pub fn current_rng_state(&self) -> u32 {
        let v = match serde_json::to_value(&self.rng) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        match v.get("s").and_then(|x| x.as_u64()) {
            Some(s) => (s as u32).max(1),
            None => 1,
        }
    }
}

fn fix_ring(v: &Vec<u32>) -> Vec<u32> {
    let mut v = v.clone();
    v.resize(WINDOW_LEN, 0);
    v
}

/// Finite [0, FOOD_ACC_CAP] accumulator value for restore paths.
fn sane_acc(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, FOOD_ACC_CAP)
    } else {
        0.0
    }
}

/// Read-only memory budget probe from [`Simulation::memory_probe`].
/// Additive-only diagnostics; never serialized into saves.
#[derive(Clone, Debug)]
pub struct MemProbe {
    /// backing capacity of the live organisms Vec
    pub organisms_capacity: usize,
    /// backing capacity of the foods Vec
    pub foods_capacity: usize,
    /// current lineage record count (bounded by RECORDS_CAP + live ancestry)
    pub records_len: usize,
    /// backing capacity of the deferred-death buffer
    pub pending_deaths_capacity: usize,
}

/// Cheap aggregate diagnostics snapshot (serde; see `Simulation::diagnosis`).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Diag {
    pub population: u32,
    pub births_window: u32,
    pub deaths_window: u32,
    pub food_count: u32,
    pub avg_energy: f32,
    pub avg_generation: f32,
    /// starvation deaths / deaths over the last window (0..1)
    pub starvation_share_window: f32,
    /// population / (food_count + 1), clamped 0..10 (>1 = deficit risk)
    pub food_deficit_ratio: f32,
    /// weather wet cover (last weather tick)
    pub wet_cover: f32,
}

/// Rehydrates a mulberry32 state. Rng keeps no public state getter, so the
/// state travels through serde (numbers only; cannot fail save for finite state).
fn set_rng_state(rng: &mut Rng, state: u32, fallback_seed: u32) {
    let v = serde_json::json!({ "s": state });
    *rng = match serde_json::from_value::<Rng>(v) {
        Ok(r) => r,
        Err(_) => Rng::new(fallback_seed | 1),
    };
}

fn normalize_org(o: &mut Organism, w: f32, h: f32) {
    if !o.x.is_finite() {
        o.x = w * 0.5;
    }
    if !o.y.is_finite() {
        o.y = h * 0.5;
    }
    if !o.dir.is_finite() {
        o.dir = 0.0;
    }
    o.speed = if o.speed.is_finite() { o.speed.max(0.0) } else { 0.0 };
    o.max_energy = if o.max_energy.is_finite() && o.max_energy > 0.0 {
        o.max_energy
    } else {
        max_energy_for(&o.genotype.traits)
    };
    o.energy = if o.energy.is_finite() {
        o.energy.clamp(0.0, o.max_energy)
    } else {
        0.0
    };
    if o.last_inputs.len() != NET_INPUTS {
        o.last_inputs.clear();
        o.last_inputs.resize(NET_INPUTS, 0.0);
    }
    if o.last_outputs.len() != NET_OUTPUTS {
        o.last_outputs.clear();
        o.last_outputs.resize(NET_OUTPUTS, 0.0);
    }
}

impl SimApi for Simulation {
    fn step(&mut self) -> StepEvents {
        self.advance()
    }
    fn stats(&self) -> &SimStats {
        &self.stats
    }
    fn step_count(&self) -> u64 {
        self.step_count
    }
    fn organisms(&self) -> &[Organism] {
        &self.organisms
    }
    fn foods(&self) -> &[Food] {
        &self.foods
    }
    fn edits(&self) -> &EnvironmentEdits {
        &self.edits
    }
    fn pick(&self, x: f32, y: f32, r: f32) -> Option<u64> {
        self.pick_nearest(x, y, r)
    }
    fn spawn_organisms(&mut self, n: usize) {
        self.spawn_organisms_n(n)
    }
    fn add_food(&mut self, x: f32, y: f32) -> bool {
        self.add_food_at(x, y)
    }
    fn add_obstacle(&mut self, x: f32, y: f32, r: f32) -> bool {
        self.add_obstacle_at(sanitize(x), sanitize(y), r)
    }
    fn add_patch(&mut self, x: f32, y: f32, r: f32) -> bool {
        self.add_patch_at(sanitize(x), sanitize(y), r)
    }
    fn add_hazard(&mut self, x: f32, y: f32, r: f32) -> bool {
        self.add_hazard_at(sanitize(x), sanitize(y), r)
    }
    fn remove_at(&mut self, x: f32, y: f32, r: f32) -> bool {
        self.remove_items_at(x, y, r)
    }
    fn undo_edit(&mut self) -> bool {
        self.pop_undo()
    }
    fn update_live_params(&mut self, patch: &LiveParams) {
        self.apply_live_params(patch)
    }
    fn organism(&self, id: u64) -> Option<&Organism> {
        self.organism_by_id(id)
    }
    fn get_record(&self, id: u64) -> Option<LineageRecord> {
        self.record_by_id(id)
    }
    fn rng_state(&self) -> u32 {
        self.current_rng_state()
    }
    fn snapshot(&self) -> SaveCore {
        Simulation::snapshot(self)
    }
    fn restore(&mut self, params: &SimParams, world: WorldModel, save: &SaveCore) {
        *self = Simulation::from_save_core(params, world, save);
    }
    fn trigger(&mut self, ev: &GodEvent) -> Result<(), String> {
        Simulation::trigger(self, ev)
    }
    fn interventions(&self) -> &[InterventionRecord] {
        Simulation::interventions(self)
    }
    fn protect_lineage(&mut self, species_id: u32, on: bool) -> bool {
        Simulation::protect_lineage(self, species_id, on)
    }
    fn protected_lines(&self) -> Vec<u32> {
        Simulation::protected_lines(self)
    }
    fn weather(&self) -> &WeatherState {
        &self.weather
    }
    fn diagnosis(&self) -> Diag {
        Simulation::diagnosis(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimApi;

    fn genesis_sim() -> Simulation {
        let world = alife_worldgen::generate(&alife_common::WorldGenConfig::default());
        Simulation::new(&SimParams::default(), world)
    }

    #[test]
    fn runs_and_stays_finite() {
        let mut sim = genesis_sim();
        for _ in 0..3000 {
            let ev = sim.advance();
            for b in &ev.births {
                assert!(b.child_id != b.parent_id);
            }
        }
        let s = sim.stats();
        assert_eq!(s.step, 3000);
        for o in sim.organisms() {
            assert!(o.x.is_finite() && o.y.is_finite() && o.energy.is_finite());
        }
        assert!(sim.foods().iter().all(|f| f.alive));
    }

    #[test]
    fn deterministic_trajectory() {
        let mut a = genesis_sim();
        let mut b = genesis_sim();
        for _ in 0..400 {
            a.advance();
            b.advance();
        }
        assert_eq!(a.step_count(), b.step_count());
        assert_eq!(a.stats().population, b.stats().population);
        assert_eq!(a.stats().species_counts, b.stats().species_counts);
        assert_eq!(a.organisms().len(), b.organisms().len());
        if let (Some(x), Some(y)) = (a.organism(1), b.organism(1)) {
            assert_eq!(x, y);
            assert_eq!(x.last_inputs, y.last_inputs);
        }
        assert_eq!(a.rng_state(), b.rng_state());
    }

    #[test]
    fn save_restore_exact() {
        let mut a = genesis_sim();
        for _ in 0..200 {
            a.advance();
        }
        let save = a.snapshot();
        let world = alife_worldgen::generate(&alife_common::WorldGenConfig::default());
        let mut b = Simulation::from_save_core(&SimParams::default(), world, &save);
        assert_eq!(b.rng_state(), a.rng_state());
        for _ in 0..300 {
            let ea = a.advance();
            let eb = b.advance();
            assert_eq!(ev_digest(&ea), ev_digest(&eb));
        }
        assert_eq!(a.stats().species_counts, b.stats().species_counts);
        assert_eq!(a.organisms().len(), b.organisms().len());
    }

    fn ev_digest(e: &StepEvents) -> Vec<u64> {
        let mut v: Vec<u64> = e.births.iter().map(|b| b.child_id).collect();
        v.extend(e.deaths.iter().map(|d| d.id));
        v.sort_unstable();
        v
    }

    #[test]
    fn editor_roundtrip() {
        let mut sim = genesis_sim();
        assert!(sim.add_food_at(100.0, 100.0));
        assert!(sim.add_obstacle_at(200.0, 200.0, 40.0));
        assert!(sim.add_patch_at(300.0, 300.0, 50.0));
        assert!(sim.add_hazard_at(400.0, 400.0, 50.0));
        assert!(sim.remove_items_at(200.0, 200.0, 50.0));
        assert!(sim.edits().obstacles.is_empty());
        assert!(sim.pop_undo());
        assert_eq!(sim.edits().obstacles.len(), 1);
        assert!(sim.pop_undo());
        assert_eq!(sim.edits().hazards.len(), 0);
    }

    #[test]
    fn redo_reapplies_undone_edits() {
        // food add -> undo -> redo restores the exact food object
        let mut sim = genesis_sim();
        let before = sim.stats().food_count;
        assert!(sim.add_food_at(150.0, 150.0));
        assert_eq!(sim.stats().food_count, before + 1);
        assert!(sim.pop_undo());
        assert_eq!(sim.stats().food_count, before);
        assert!(sim.redo_edit());
        assert_eq!(sim.stats().food_count, before + 1);
        assert!(matches!(
            sim.foods().iter().find(|f| (f.x - 150.0).abs() < 2.0 && (f.y - 150.0).abs() < 2.0),
            Some(f) if f.alive
        ));

        // erase -> undo -> redo re-removes
        let mut sim = genesis_sim();
        assert!(sim.add_obstacle_at(60.0, 60.0, 20.0));
        assert!(sim.remove_items_at(60.0, 60.0, 25.0));
        assert!(sim.edits().obstacles.is_empty());
        assert!(sim.pop_undo());
        assert_eq!(sim.edits().obstacles.len(), 1);
        assert!(sim.redo_edit());
        assert!(sim.edits().obstacles.is_empty());

        // a new edit clears the redo stack
        let mut sim = genesis_sim();
        assert!(sim.add_food_at(10.0, 10.0));
        assert!(sim.pop_undo());
        assert!(sim.add_food_at(20.0, 20.0));
        assert!(!sim.redo_edit());
    }

    #[test]
    fn empty_population_is_legal() {
        let world = alife_worldgen::generate(&alife_common::WorldGenConfig::default());
        let mut params = SimParams::default();
        params.initial_population = 0;
        let mut sim = Simulation::new(&params, world);
        for _ in 0..500 {
            sim.advance();
        }
        assert_eq!(sim.stats().population, 0);
        assert_eq!(sim.stats().food_count, sim.foods().len() as u32);
    }

    #[test]
    fn records_trim_bounded_and_ancestors_protected() {
        // Phase A (natural churn): a reproducing population where every step
        // must keep records bounded AND every living organism's full ancestor
        // chain must stay in the map (the trim protects mom-chains).
        let params = SimParams {
            initial_population: 120,
            max_population: 220,
            initial_food: 900,
            max_food: 1400,
            food_regen_per_sec: 30.0,
            min_repro_age_steps: 120,
            reproduction_cooldown_steps: 60,
            ..SimParams::default()
        };
        let world = alife_worldgen::generate(&alife_common::WorldGenConfig::default());
        let mut sim = Simulation::new(&params, world.clone());
        let mut saw_chain = false;
        for step in 1..=1600 {
            sim.advance();
            let p = sim.memory_probe();
            assert!(
                p.records_len <= 576 + 3 * 220,
                "records_len={} exceeded trim bound",
                p.records_len
            );
            let mut chain_hits = 0usize;
            for o in &sim.organisms {
                let mut cur = o.mom;
                while let Some(id) = cur {
                    let rec = sim.records.get(&id).unwrap_or_else(|| {
                        panic!("protected ancestor {} of living {} evicted", id, o.id)
                    });
                    chain_hits += 1;
                    cur = rec.mom;
                }
            }
            // founders have mom=None; real ancestor chains exist once the
            // population has been breeding for a while
            if step > 300 {
                saw_chain |= chain_hits > 0;
            }
        }
        assert!(saw_chain, "births must have produced ancestor chains");

        // Phase B (forced trim proof): mass-spawn + mass-purge to stack up
        // > cap+slack already-dead records, then the next birth must evict
        // them (oldest death first) back under the bound.
        let base = SimParams {
            initial_population: 40,
            max_population: 1600,
            initial_food: 400,
            max_food: 800,
            food_regen_per_sec: 10.0,
            min_repro_age_steps: 600,
            reproduction_cooldown_steps: 240,
            ..SimParams::default()
        };
        let mut sim2 = Simulation::new(&base, world.clone());
        for _ in 0..1000 {
            sim2.spawn_founder();
        }
        let high_water = sim2.memory_probe().records_len;
        assert!(high_water > 576 + 64, "must cross the trim mark (got {high_water})");
        while sim2.stats().population > 0 {
            let sid = sim2.stats().species_counts[0].species_id;
            assert!(Simulation::trigger(&mut sim2, &GodEvent::PurgeLineage { species_id: sid }).is_ok());
        }
        sim2.spawn_founder(); // record_born -> eval evict fires
        let after = sim2.memory_probe().records_len;
        assert!(after < high_water - 32, "trim must evict (after={after}, hw={high_water})");
        assert!(after <= 576 + 64 + 40, "post-trim len {after} over bound");
        // the trim keeps everything a snapshot needs: exact restore of the
        // trimmed state and an identical continued trajectory
        let save = sim2.snapshot();
        let mut b = Simulation::from_save_core(&base, world.clone(), &save);
        assert_eq!(b.rng_state(), sim2.rng_state());
        for _ in 0..60 {
            let ea = sim2.advance();
            let eb = b.advance();
            let ae: Vec<u64> = ea.births.iter().map(|x| x.child_id).collect();
            let be: Vec<u64> = eb.births.iter().map(|x| x.child_id).collect();
            assert_eq!(ae, be);
        }
    }

    #[test]
    fn perf_smoke() {
        let mut sim = genesis_sim();
        let t0 = std::time::Instant::now();
        for _ in 0..1800 {
            sim.advance();
        }
        let per_step = t0.elapsed() / 1800;
        eprintln!("1800 genesis steps: {:?} per step (debug build)", per_step);
    }
}
