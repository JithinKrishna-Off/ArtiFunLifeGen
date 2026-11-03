//! Queen brain bridge (coordinator): adapts the coordinator's
//! [`crate::queen::QueenBrain`] contract to `alife_neural::QueenBrainAdapter`
//! (the spiking FlyCNS runtime). Both crates own disjoint halves — alife-core
//! owns this glue, alife-neural never depends on core.
//!
//! # Field mappings (core ⇄ neural, documented)
//! - core `SensoryInputForQueen.food_dir` unit-or-zero vector passes through
//!   unchanged (the adapter re-normalizes per tick via `safe_dir`).
//! - `food_dist` (world units) passes through; the adapter encodes it as
//!   `1 / (1 + dist)` (monotone inverse) — do NOT pre-encode here.
//! - `heading` radians pass through; adapter maps (cos, sin) onto its
//!   light-alias directional pair.
//! - `speed` / `energy_frac` / `hazard` / `edge_dist` / `density` /
//!   `signal_sample`: the ENGINE already fills everything in [0, 1] (speed is
//!   normalized by the queen body's max speed, see `Simulation::brain_senses`);
//!   the bridge only re-clamps defensively and passes through.
//! - `QueenMotorOut` comes back already bounded from the adapter's
//!   `decode_queen_motor`; the bridge re-clamps (`turn` [-1,1], the rest
//!   [0,1]) and unit-normalizes `move_dir` (zero stays zero). The engine
//!   steers from `move_dir` + `move_intensity`; `turn` is recorded into
//!   `last_outputs[0]`, `move_intensity` into `last_outputs[1]` (net output
//!   slots keep their renderer/UI semantics). `feed_seek` and `emit_signal`
//!   are consumed engine-side (food-intensity slot bias / kind-4 emission).
//!
//! # Stats mapping (honest source note)
//! `neuron_count` / `synapse_count` come from the runtime's measured
//! `FlyBrainRuntime::last_stats()` counters (populated at build time from the
//! connectome graph). `active_neurons` / `spikes_last` / `exec_ns` come from
//! the adapter's measured `stats()`. NOT surfaced (dropped, documented): the
//! adapter's `ticks` counter (core tracks `QueenState.brain_ticks`
//! separately) and `avg_rate_ema` (no core slot).
//!
//! # Save contract
//! `snapshot`/`restore` are the adapter's runtime bytes verbatim
//! (serde_json). Restoring requires the SAME connectome shape; `identity()`
//! is stored by the coordinator for provenance checking.

use std::path::Path;
use std::sync::Arc;

use alife_neural::connectome::Connectome;
use alife_neural::connectome_binary as cb;
use alife_neural::queen_adapter::QueenBrainAdapter;
use alife_neural::queen_types as qt;

use crate::queen;

/// Expected queen subnetwork artifact next to the coordinator's data dir.
pub const QUEEN_ARTIFACT_FILE: &str = "queen-subnetwork-166k.alifecns";

/// Boxed adapter implementing the coordinator's `QueenBrain` trait.
pub struct QueenBrainAdapterBridge(Box<QueenBrainAdapter>);

impl QueenBrainAdapterBridge {
    pub fn new(adapter: QueenBrainAdapter) -> Self {
        Self(Box::new(adapter))
    }

    pub fn inner(&self) -> &QueenBrainAdapter {
        &self.0
    }
}

#[inline]
fn san(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn unit_or_zero(d: (f32, f32)) -> (f32, f32) {
    let (dx, dy) = (san(d.0), san(d.1));
    let m = (dx * dx + dy * dy).sqrt();
    if m <= 1e-6 {
        (0.0, 0.0)
    } else {
        (dx / m, dy / m)
    }
}

impl queen::QueenBrain for QueenBrainAdapterBridge {
    fn tick(
        &mut self,
        senses: &queen::SensoryInputForQueen,
        budget: &queen::NeuralTickBudget,
    ) -> queen::QueenMotorOut {
        // core -> neural: straight struct map (see module doc for semantics)
        let qt_in = qt::SensoryInputForQueen {
            food_dir: (san(senses.food_dir.0), san(senses.food_dir.1)),
            food_dist: senses.food_dist.max(0.0),
            heading: san(senses.heading),
            speed: senses.speed.clamp(0.0, 1.0),
            energy_frac: senses.energy_frac.clamp(0.0, 1.0),
            hazard: senses.hazard.clamp(0.0, 1.0),
            edge_dist: senses.edge_dist.clamp(0.0, 1.0),
            signal_sample: senses.signal_sample.clamp(0.0, 1.0),
            density: senses.density.clamp(0.0, 1.0),
            hazard_dir: (san(senses.hazard_dir.0), san(senses.hazard_dir.1)),
            hazard_rate: senses.hazard_rate.clamp(0.0, 1.0),
        };
        let qt_budget =
            qt::NeuralTickBudget { max_ticks_per_advance: budget.max_ticks_per_advance };
        let out = self.0.tick(&qt_in, &qt_budget);
        // neural -> core: bounded back-map, clamps re-applied defensively
        queen::QueenMotorOut {
            move_dir: unit_or_zero(out.move_dir),
            move_intensity: out.move_intensity.clamp(0.0, 1.0),
            turn: out.turn.clamp(-1.0, 1.0),
            feed_seek: out.feed_seek.clamp(0.0, 1.0),
            emit_signal: out.emit_signal.clamp(0.0, 1.0),
        }
    }

    fn snapshot(&self) -> Vec<u8> {
        self.0.snapshot()
    }

    fn restore(&mut self, bytes: &[u8]) -> bool {
        self.0.restore(bytes)
    }

    fn stats(&self) -> queen::QueenBrainStats {
        let a = self.0.stats();
        let st = self.0.runtime().last_stats();
        let cap32 = u32::MAX as u64;
        queen::QueenBrainStats {
            neuron_count: (st.neuron_count.min(cap32)) as u32,
            synapse_count: (st.synapse_count.min(cap32)) as u32,
            active_neurons: (a.active_neurons.min(cap32)) as u32,
            spikes_last: (a.spikes_last.min(cap32)) as u32,
            exec_ns: a.exec_ns,
        }
    }

    fn identity(&self) -> (String, String) {
        self.0.identity()
    }

    fn sensory_injection(&mut self, channel: usize, value: f32) {
        self.0.sensory_injection(channel, value);
    }
}

/// Factory rehydrating a queen brain from the MaleCNS 166k queen-subnetwork
/// artifact ([`QUEEN_ARTIFACT_FILE`]). The connectome is loaded ONCE at
/// registration ([`register_default_factory`]) and kept behind an `Arc`, so
/// `build()` never re-reads the disk and cannot start failing silently.
pub struct MaleCnsQueenBrainFactory {
    connectome: Arc<Connectome>,
    /// Real sugar gustatory population (original MaleCNS body ids), read
    /// once from the artifact package dir. `None` = the package's population
    /// tables were unavailable; the adapter then binds food channels onto
    /// structurally-defensible `cb_sensory` slices (documented honest
    /// degradation).
    sugar_bodies: Option<Arc<Vec<u64>>>,
}

impl MaleCnsQueenBrainFactory {
    /// Build from an already-validated connectome (see
    /// [`register_default_factory`] for the disk-loading path). No sugar
    /// population — food channels bind structurally, not gustatorily.
    pub fn from_connectome(connectome: Arc<Connectome>) -> Self {
        Self { connectome, sugar_bodies: None }
    }

    /// [`MaleCnsQueenBrainFactory::from_connectome`] plus the sugar
    /// population (original body ids; see
    /// `alife_neural::malecns_bins::sugar_body_ids`).
    pub fn from_connectome_with_sugar(connectome: Arc<Connectome>, sugar: Arc<Vec<u64>>) -> Self {
        Self { connectome, sugar_bodies: Some(sugar) }
    }
}

impl queen::QueenBrainFactory for MaleCnsQueenBrainFactory {
    fn build(&self) -> Box<dyn queen::QueenBrain> {
        let sugar = self.sugar_bodies.as_ref().map(|a| a.as_slice());
        match QueenBrainAdapter::new_from_connectome_with_sugar(self.connectome.clone(), 0, sugar) {
            Ok(a) => Box::new(QueenBrainAdapterBridge::new(a)),
            // Practically unreachable: registration pre-verified the exact
            // same build call on this Arc. A `TooSmall` here would mean a
            // logic regression, not a bad artifact — fail loudly.
            Err(e) => Box::new(FailedQueenBrain(format!("connectome build failed: {e:?}"))),
        }
    }
}

/// Honest no-op degradation for `build()`-time failure (never a synthetic
/// substitute posing as the real subnetwork): zeroed stats, identity marks
/// the failure, motor output stays neutral so the queen falls back to her
/// ordinary evolved net steering. Only reachable through misregistration.
pub struct FailedQueenBrain(pub String);

impl queen::QueenBrain for FailedQueenBrain {
    fn tick(
        &mut self,
        _senses: &queen::SensoryInputForQueen,
        _budget: &queen::NeuralTickBudget,
    ) -> queen::QueenMotorOut {
        queen::QueenMotorOut::default()
    }
    fn snapshot(&self) -> Vec<u8> {
        Vec::new()
    }
    fn restore(&mut self, _bytes: &[u8]) -> bool {
        false
    }
    fn stats(&self) -> queen::QueenBrainStats {
        queen::QueenBrainStats::default()
    }
    fn identity(&self) -> (String, String) {
        ("failed".into(), self.0.clone())
    }
    fn sensory_injection(&mut self, _channel: usize, _value: f32) {}
}

/// Load + verify the queen subnetwork artifact from `artifact_dir` and, on
/// success, register the rehydration factory on the simulation. On missing or
/// corrupt artifact this is an honest [`Err`] and the sim keeps NO factory
/// (absent brain is a legal state). Never a synthetic-fixture fallback —
/// fixtures are tests-only ([`QueenBrainAdapter::new_for_fixture`]).
pub fn register_default_factory(
    sim: &mut crate::Simulation,
    artifact_dir: impl AsRef<Path>,
) -> Result<(), String> {
    let dir = artifact_dir.as_ref();
    let path = dir.join(QUEEN_ARTIFACT_FILE);
    let conn = cb::load(&path).map_err(|e| format!("queen artifact load failed: {e}"))?;
    // the population tables live alongside the artifact (the package dir);
    // their absence is honest degradation, not a failure
    let sugar = alife_neural::malecns_bins::sugar_body_ids(dir).map(Arc::new);
    let sugar_ref = sugar.as_ref().map(|a| a.as_slice());
    QueenBrainAdapter::new_from_connectome_with_sugar(conn.clone(), 0, sugar_ref)
        .map_err(|e| format!("queen artifact unusable by runtime: {e:?}"))?;
    sim.set_queen_brain_factory(Some(Box::new(match sugar {
        Some(s) => MaleCnsQueenBrainFactory::from_connectome_with_sugar(conn, s),
        None => MaleCnsQueenBrainFactory::from_connectome(conn),
    })));
    Ok(())
}
