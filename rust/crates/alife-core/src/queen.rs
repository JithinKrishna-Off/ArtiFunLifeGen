//! Queen organism layer: a single crowned organism (Agent 1).
//!
//! The queen is the same `Organism` body — she ages through the same
//! per-step path, eats through the same food pickup and dies through the
//! same death checks. Differences: extra metabolism drain
//! ([`QueenConfig::metabolism_mult`]), an extended lifespan
//! ([`QueenConfig::lifespan_mult`]), and an optional pluggable brain
//! ([`QueenBrain`]; wired by the coordinator through `queen_bridge` to the
//! `alife-neural` spiking runtime — this module keeps no brain type).
//!
//! Signals: `Simulation.signals` is the REAL diffusion field
//! (`crate::signals::SignalField`); QueenActivity is kind 4
//! ([`QUEEN_ACTIVITY_KIND`]), emitted by the plugged brain's
//! `QueenMotorOut::emit_signal` (see the wiring in `Simulation::advance`).

use serde::{Deserialize, Serialize};
use alife_common::DeathCause;

/// Signal kinds (pre-agreed contract with `crate::signals`).
pub const QUEEN_ACTIVITY_KIND: usize = 4;

// ============================================================ brain contract

/// Pluggable queen brain. The coordinator plugs the real runtime through
/// this trait; core never depends on alife-neural.
pub trait QueenBrain: Send {
    /// One brain tick with the queen's sensory snapshot under a tick budget.
    fn tick(&mut self, senses: &SensoryInputForQueen, budget: &NeuralTickBudget) -> QueenMotorOut;
    /// Opaque serializable brain state (round-trips through `restore`).
    fn snapshot(&self) -> Vec<u8>;
    /// Restore from opaque bytes. Returns false if unusable (state kept).
    fn restore(&mut self, bytes: &[u8]) -> bool;
    /// Cheap counters for the compute report / UI.
    fn stats(&self) -> QueenBrainStats;
    /// (connectome name, version)
    fn identity(&self) -> (String, String);
    /// Test/diagnostic injection into a sensory channel.
    fn sensory_injection(&mut self, channel: usize, value: f32);
}

/// Queen sensory snapshot built from the same per-organism sensory context
/// the cheap 11-input net gets. All values are plain f32 (unit notes inline).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct SensoryInputForQueen {
    /// unit vector toward the nearest food, WORLD frame (0,0 if none)
    pub food_dir: (f32, f32),
    /// distance to that food in world units (f32::INFINITY if none)
    pub food_dist: f32,
    /// heading (radians, wrapped to [-pi, pi])
    pub heading: f32,
    /// current forward speed (units/second; brain normalizes by traits)
    pub speed: f32,
    /// stored energy / max stored energy, clamped 0..1
    pub energy_frac: f32,
    /// world-hazard pressure at her cell, normalized 0..1 (same as net inp[6])
    pub hazard: f32,
    /// distance to the nearest non-wrapping edge, normalized 0..1
    /// (1.0 = far/no edge — wrapping worlds report 1.0)
    pub edge_dist: f32,
    /// QueenActivity signal sample at her position (raw field value)
    pub signal_sample: f32,
    /// local peer density (same normalization as net inp[5], 0..1)
    pub density: f32,
    /// unit vector toward the nearest hazard (WORLD frame, (0,0) when calm).
    /// Threat DIRECTION, distinct from the scalar `hazard` pressure.
    pub hazard_dir: (f32, f32),
    /// threat approach rate in [0, 1]: 1 when the hazard pressure under her
    /// is rising fast (closing in), 0 when steady/receding. Measured closure
    /// between consecutive brain advances; 0 right after restore.
    pub hazard_rate: f32,
}

/// Motor output applied through the ordinary movement/actuation code path.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct QueenMotorOut {
    /// desired movement direction, WORLD frame (atan2 input; (0,0) = keep dir)
    pub move_dir: (f32, f32),
    /// 0..1 how hard to push toward `move_dir` (scales target speed)
    pub move_intensity: f32,
    /// raw turn intent offered to the brain-side bookkeeping (rad-ish); the
    /// engine steers via `move_dir`, this value is recorded in last_outputs[0]
    pub turn: f32,
    /// 0..1 steering bias that pulls the heading toward the nearest food,
    /// exactly analog to the ordinary food-driven input mapping
    pub feed_seek: f32,
    /// 0..1 QueenActivity emission strength this tick (emit cost scaled)
    pub emit_signal: f32,
}

/// Per-advance brain tick budget handed to the brain each tick.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NeuralTickBudget {
    pub max_ticks_per_advance: u32,
}

/// Cheap brain counters (see [`QueenBrain::stats`]).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct QueenBrainStats {
    pub neuron_count: u32,
    pub synapse_count: u32,
    pub active_neurons: u32,
    pub spikes_last: u32,
    pub exec_ns: u64,
}

/// Rehydrates a `Box<dyn QueenBrain>` from snapshotted bytes on load.
/// The brain snapshot is stored in saves, but core cannot name the concrete
/// type — the coordinator registers a factory before round-tripping brains.
pub trait QueenBrainFactory: Send {
    fn build(&self) -> Box<dyn QueenBrain>;
}

// ============================================================ events / state

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueenDeath {
    Starvation,
    OldAge,
    Hazard,
    Other,
}

impl From<DeathCause> for QueenDeath {
    fn from(c: DeathCause) -> Self {
        match c {
            DeathCause::Starvation => QueenDeath::Starvation,
            DeathCause::OldAge => QueenDeath::OldAge,
            DeathCause::Hazard => QueenDeath::Hazard,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueenEventKind {
    Crowned,
    Died(QueenDeath),
    Fed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueenEvent {
    /// simulation step the event happened on
    pub tick: u64,
    pub kind: QueenEventKind,
    /// queen organism id (u32 per contract; ids are sequential and small)
    pub org_id: u32,
}

const EVENT_CAP: usize = 64;
const ENERGY_HIST_CAP: usize = 256;

/// The queen slot. Always present (alive=false before the first crown);
/// death drops the brain and never respawns her.
#[derive(Default)]
pub struct QueenState {
    pub alive: bool,
    /// queen organism id (u32 per contract)
    pub org_id: u32,
    /// step the current reign was crowned
    pub crowned_tick: u64,
    pub death_cause: Option<QueenDeath>,
    pub brain: Option<Box<dyn QueenBrain>>,
    /// per-step energy fraction ring (<= [ENERGY_HIST_CAP])
    pub energy_history: Vec<f32>,
    /// total brain ticks since load (diagnostic; additive to contract)
    pub brain_ticks: u64,
    /// true when a loaded brain snapshot failed to restore (honest telemetry)
    pub brain_restore_failed: bool,
    /// Transient (never serialized from a live sim): the snapshot bytes of a
    /// restored queen whose brain could not be rehydrated at load time
    /// because no factory was registered yet. `set_queen_brain_factory`
    /// attempts one honest, factory-built restore of these bytes; on
    /// success the flag clears and the brain plugs. Dead queens never
    /// reclaim (no respawn).
    pub reclaim_brain: Option<QueenBrainSnapshot>,
}

impl QueenState {
    pub(crate) fn push_energy_frac(&mut self, v: f32) {
        let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        self.energy_history.push(v);
        if self.energy_history.len() > ENERGY_HIST_CAP {
            self.energy_history.remove(0);
        }
    }

    /// Restore-path bound: keeps the last `ENERGY_HIST_CAP` samples only.
    pub(crate) fn cap_energy_history(&mut self) {
        if self.energy_history.len() > ENERGY_HIST_CAP {
            let drop = self.energy_history.len() - ENERGY_HIST_CAP;
            self.energy_history.drain(0..drop);
        }
    }
}

// ============================================================ config

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct QueenConfig {
    /// extra metabolism multiplier ON TOP of her ordinary drain (1.5 = she
    /// loses 1.5x the energy an identical body would lose each step)
    #[serde(default = "QueenConfig::def_metabolism")]
    pub metabolism_mult: f32,
    /// she ages via the SAME aging check, just longer (3.0 = 3x lifespan)
    #[serde(default = "QueenConfig::def_lifespan")]
    pub lifespan_mult: f32,
    /// brain ticks per advance (budget cap)
    #[serde(default = "QueenConfig::def_ticks")]
    pub brain_ticks_per_advance: u32,
    /// emit_signal strength scale applied to `SignalField::emit` kind 4
    #[serde(default = "QueenConfig::def_emit")]
    pub signal_emit_cost: f32,
    /// survival floor (fraction of max energy) below which the crisis state
    /// begins (drain scaled by `crisis_drain_scale`, recovery trickle starts)
    #[serde(default = "QueenConfig::def_floor")]
    pub survival_floor: f32,
    /// recovery trickle in fractions of max energy per second while she is
    /// below `survival_floor` (the documented recovery mechanism)
    #[serde(default = "QueenConfig::def_regen")]
    pub survival_regen_per_sec: f32,
    /// drain scale while in the crisis state (metabolic saving; 0.5 = half)
    #[serde(default = "QueenConfig::def_crisis")]
    pub crisis_drain_scale: f32,
}

impl QueenConfig {
    fn def_metabolism() -> f32 {
        1.5
    }
    fn def_lifespan() -> f32 {
        3.0
    }
    fn def_ticks() -> u32 {
        // 2 x 1ms brain integration per advance: enough temporal integration
        // for the rate encoders to deliver afferent pulses each advance
        // (16.7 sim-ms) without overwhelming the frame budget (~0.9ms/tick
        // measured on the 40k-neuron queen subnetwork).
        2
    }
    fn def_emit() -> f32 {
        0.05
    }
    fn def_floor() -> f32 {
        0.10
    }
    fn def_regen() -> f32 {
        0.03
    }
    fn def_crisis() -> f32 {
        0.5
    }
}

impl Default for QueenConfig {
    fn default() -> Self {
        Self {
            metabolism_mult: Self::def_metabolism(),
            lifespan_mult: Self::def_lifespan(),
            brain_ticks_per_advance: Self::def_ticks(),
            signal_emit_cost: Self::def_emit(),
            survival_floor: Self::def_floor(),
            survival_regen_per_sec: Self::def_regen(),
            crisis_drain_scale: Self::def_crisis(),
        }
    }
}

/// Renderer/UI flag set (mark the queen without touching organism traits).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct QueenVisualFlags {
    pub is_queen: bool,
    /// deterministic crown hue in 0..1 (golden-ratio sequence on org id)
    pub crown_hue: f32,
}

// ============================================================ save payload

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct QueenBrainSnapshot {
    pub identity: (String, String),
    pub stats: QueenBrainStats,
    pub bytes: Vec<u8>,
}

/// Additive v2 save payload (`SaveCore.queen: Option<...>`). A save without
/// this field (or `null`) deserializes to `None` and loads identically to
/// pre-queen builds.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct QueenSnapshot {
    pub alive: bool,
    pub org_id: u32,
    pub crowned_tick: u64,
    pub death_cause: Option<QueenDeath>,
    pub energy_history: Vec<f32>,
    pub brain_ticks: u64,
    pub events: Vec<QueenEvent>,
    pub config: QueenConfig,
    pub brain: Option<QueenBrainSnapshot>,
}

// ------------------------------------------------------------------ helpers

#[inline]
fn wrap_pi(a: f32) -> f32 {
    let a = if a.is_finite() { a } else { 0.0 };
    let mut a = a;
    while a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    while a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

#[inline]
fn san(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Trim kept-events to the last [EVENT_CAP], oldest dropped.
pub fn trim_events(events: &mut Vec<QueenEvent>) {
    if events.len() > EVENT_CAP {
        events.drain(0..events.len() - EVENT_CAP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_contract() {
        let c = QueenConfig::default();
        assert_eq!(c.metabolism_mult, 1.5);
        assert_eq!(c.lifespan_mult, 3.0);
        assert_eq!(c.brain_ticks_per_advance, 2);
        assert!(c.survival_floor > 0.0 && c.survival_floor < 0.5);
        assert!(c.survival_regen_per_sec > 0.0);
        assert!(c.crisis_drain_scale > 0.0 && c.crisis_drain_scale < 1.0);
    }

    #[test]
    fn event_ring_trims_to_cap() {
        let mut v: Vec<QueenEvent> = Vec::new();
        for i in 0..80u64 {
            v.push(QueenEvent { tick: i, kind: QueenEventKind::Crowned, org_id: 1 });
            trim_events(&mut v);
        }
        assert_eq!(super::EVENT_CAP, 64);
        assert_eq!(v.len(), 64);
        assert_eq!(v[0].tick, 80 - 64);
    }

    // tiny wiring smoke test: wrap_pi keeps angles in [-pi, pi] and is
    // periodic over TAU (correct for inputs needing any number of wraps)
    #[test]
    fn wrap_pi_basics() {
        let w = wrap_pi(10.0);
        assert!(w <= std::f32::consts::PI && w >= -std::f32::consts::PI);
        assert!((wrap_pi(10.0 + std::f32::consts::TAU) - w).abs() < 1e-6);
        assert!(wrap_pi(std::f32::consts::PI) <= std::f32::consts::PI);
    }
}







