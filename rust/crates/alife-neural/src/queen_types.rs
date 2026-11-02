//! Queen-facing trait TYPES (structs only), shaped after the coordinator's
//! `alife_core::queen` contract so the adapter can compile standalone while
//! Agent 1's `queen.rs` lands.
//!
//! STATUS: duplicated on landing — once `alife_core::queen` is in the tree and
//! the `coordinator` cargo feature is enabled, the REAL trait (`QueenBrain`)
//! and its types come from alife-core; this module stays as the adapter's
//! standalone fallback for builds without that feature. Field semantics:
//! - `food_dir`: unit-or-zero vector toward the nearest food.
//! - `food_dist`: normalized food distance (smaller = closer); encoded as
//!   `1 / (1 + dist)` (monotone, bounded).
//! - `heading`: organism heading in radians.
//! - `speed` / `energy_frac` / `hazard` / `edge_dist` / `density` /
//!   `signal_sample`: all in [0, 1].
//! Motor outputs: `move_dir` unit-or-zero; `move_intensity` / `feed_seek` /
//! `emit_signal` in [0, 1]; `turn` in [-1, 1].

/// Sensory sample delivered by the coordinator each advance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensoryInputForQueen {
    pub food_dir: (f32, f32),
    pub food_dist: f32,
    /// Radians.
    pub heading: f32,
    pub speed: f32,
    pub energy_frac: f32,
    pub hazard: f32,
    pub edge_dist: f32,
    /// Queen coordination signal (QueenActivity sample) in [0, 1].
    pub signal_sample: f32,
    pub density: f32,
    /// Unit vector toward the nearest hazard, WORLD frame (0,0 when calm).
    /// Threat DIRECTION (distinct from the scalar `hazard` pressure).
    pub hazard_dir: (f32, f32),
    /// Threat approach rate in [0, 1] (hazard pressure closing speed).
    pub hazard_rate: f32,
}

impl Default for SensoryInputForQueen {
    fn default() -> Self {
        Self {
            food_dir: (0.0, 0.0),
            food_dist: 1.0,
            heading: 0.0,
            speed: 0.0,
            energy_frac: 0.0,
            hazard: 0.0,
            edge_dist: 0.0,
            signal_sample: 0.0,
            density: 0.0,
            hazard_dir: (0.0, 0.0),
            hazard_rate: 0.0,
        }
    }
}

/// Bounded motor output returned by the queen brain each advance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueenMotorOut {
    /// Unit-or-zero movement direction.
    pub move_dir: (f32, f32),
    /// [0, 1]
    pub move_intensity: f32,
    /// [-1, 1]
    pub turn: f32,
    /// [0, 1]
    pub feed_seek: f32,
    /// [0, 1]
    pub emit_signal: f32,
}

impl Default for QueenMotorOut {
    fn default() -> Self {
        Self {
            move_dir: (0.0, 0.0),
            move_intensity: 0.0,
            turn: 0.0,
            feed_seek: 0.0,
            emit_signal: 0.0,
        }
    }
}

/// Aggregate brain statistics for the coordinator (measured only; never part
/// of simulation determinism).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QueenBrainStats {
    /// Total brain ticks advanced since construction.
    pub ticks: u64,
    /// Cells injected or refractory during the last tick.
    pub active_neurons: u64,
    /// Spikes during the last tick.
    pub spikes_last: u64,
    /// EMA of the mean per-neuron firing rate (Hz).
    pub avg_rate_ema: f32,
    /// Wall-clock cost of the last tick (ns; excluded from determinism).
    pub exec_ns: u64,
}

/// Per-advance tick budget for the queen brain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuralTickBudget {
    /// Max 1 ms brain ticks executed per `tick()` call; 0 = paused (the last
    /// motor output is replayed without advancing state).
    pub max_ticks_per_advance: u32,
}

impl Default for NeuralTickBudget {
    fn default() -> Self {
        Self {
            max_ticks_per_advance: 1,
        }
    }
}
