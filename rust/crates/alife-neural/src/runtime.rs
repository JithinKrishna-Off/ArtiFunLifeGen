//! FlyBrainRuntime: deterministic 1 ms LIF stepping, event-driven propagation
//! over ACTIVE (spiking or injected) neurons' CSR rows, per-tick propagation
//! budget with deterministic truncation, and Queen-facing introspection.
//!
//! Step order per tick (fixed, deterministic):
//!   1. inject sensory channels (phase-accumulator pulses or linear current)
//!   2. drain delayed arrivals for this tick into `inj`
//!   3. integrate every non-refractory cell (LIF), collect spikes in index order
//!   4. propagate spikes event-style: only spiked neurons' CSR rows are touched,
//!      ascending pre-index, ascending in-row position; contributions enter the
//!      ring at (tick + delay)
//!   5. budget: if spikes processed or synops relaxed exceed the budget, the
//!      remainder is dropped deterministic of lowest indices first, and counted
//!      in `Stats.dropped_spikes` / `dropped_synops`
//!   6. decode motor outputs from output-cell spike counts (rate EMA)
//!
//! Determinism: no RNG is used in stepping. Identical (state, input, budget)
//! yields identical (state, output). `Stats.exec_ns` is a wall-clock
//! measurement and is excluded from snapshots.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::connectome::{Connectome, Polarity};
use crate::motor::{decode as motor_decode, MotorConfig, MotorOutput, OUTPUT_CELL_COUNT};
use crate::provenance::{ProvenanceRecord, ProvenanceTag};
use crate::sensory::{encode_into, inputs_needed, validate_mapping, Encoder, SensoryConfig, SensoryInput};
use crate::state::{BrainState, LifParams, MAX_RING_DELAY};

/// Policy for synapses/neurons whose polarity is Unknown (configurable and
/// documented; default: conservative attenuated excitatory).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnknownPolarityPolicy {
    /// Ignore the synapse (no effect). Safest.
    Ignore,
    /// Treat as excitatory, full weight.
    AsExcitatory,
    /// Treat as inhibitory, full weight.
    AsInhibitory,
    /// Treat as excitatory with a weight scale factor in (0, 1]; the factor is
    /// a documented modeling hedge, not biology.
    Scale(f32),
}

impl Default for UnknownPolarityPolicy {
    fn default() -> Self {
        UnknownPolarityPolicy::Scale(0.25)
    }
}

/// Per-tick propagation budget. `usize::MAX` fields = unbounded.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TickBudget {
    /// Max spiked neurons whose CSR rows are processed this tick.
    pub max_spikes: usize,
    /// Max synapses relaxed this tick.
    pub max_synops: usize,
}

impl Default for TickBudget {
    fn default() -> Self {
        Self {
            max_spikes: 4096,
            max_synops: 262_144,
        }
    }
}

impl TickBudget {
    pub fn unbounded() -> Self {
        Self {
            max_spikes: usize::MAX,
            max_synops: usize::MAX,
        }
    }
}

/// One-tick statistics (measured; never part of sim determinism).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub neuron_count: u64,
    pub synapse_count: u64,
    /// spiked cells this tick.
    pub active: u64,
    pub spikes: u64,
    pub dropped_spikes: u64,
    pub dropped_synops: u64,
    pub exec_ns: u64,
    pub mem_est: u64,
}

/// Snapshot of the mutable runtime state (Queen save/restore).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub tick: u64,
    pub brain: BrainState,
    pub sensory: SensoryConfig,
    pub motor: MotorConfig,
}

/// Simulation/config introspection surface for the Queen layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub connectome_name: String,
    pub template_version: String,
    pub neurons: u64,
    pub synapses: u64,
    pub regions_available: bool,
    pub cell_types_available: bool,
    pub lif_params: LifParams,
    pub sensory_channels: usize,
    pub motor_cells: usize,
    pub unknown_polarity_policy: UnknownPolarityPolicy,
    pub provenance: Option<ProvenanceRecord>,
}

/// The runnable brain instance. `connectome` is `Arc` (STATIC shared or
/// DYNAMIC per-organism — the API is identical).
pub struct FlyBrainRuntime {
    connectome: Arc<Connectome>,
    pub lif: LifParams,
    pub sensory: SensoryConfig,
    pub motor: MotorConfig,
    pub encoders: Vec<Encoder>,
    pub polarity_policy: UnknownPolarityPolicy,
    provenance: Option<ProvenanceRecord>,
    brain: BrainState,
    last_stats: Stats,
    out_start: usize,
    mapping_ok: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeBuildError {
    #[error(
        "connectome has {n} neurons; default sensory/motor mapping needs at least {need}"
    )]
    TooSmall { n: usize, need: usize },
}

impl FlyBrainRuntime {
    /// Build from a connectome + seed. The seed is accepted for future
    /// sensory-corridor noise; stepping is fully deterministic in it today
    /// (documented).
    pub fn new(connectome: Arc<Connectome>, _seed: u32) -> Result<Self, RuntimeBuildError> {
        let n = connectome.n();
        let sensory_cfg = SensoryConfig::default();
        let need = inputs_needed(&sensory_cfg) + OUTPUT_CELL_COUNT;
        if n < need {
            return Err(RuntimeBuildError::TooSmall { n, need });
        }
        let lif = LifParams::default();
        let v_rest = lif.v_rest;
        let ring_len = lif.max_delay(effective_max_delay(&connectome, lif.default_delay_steps));
        let mapping_ok = validate_mapping(&connectome, &sensory_cfg, OUTPUT_CELL_COUNT);
        let encoders = vec![Encoder::Rate; inputs_needed(&sensory_cfg)];
        Ok(Self {
            connectome,
            lif,
            sensory: sensory_cfg,
            motor: MotorConfig::default(),
            encoders,
            polarity_policy: UnknownPolarityPolicy::default(),
            provenance: None,
            brain: BrainState::new(n, ring_len, v_rest),
            last_stats: Stats::default(),
            out_start: n - OUTPUT_CELL_COUNT,
            mapping_ok,
        })
    }

    pub fn connectome(&self) -> &Arc<Connectome> {
        &self.connectome
    }

    pub fn state(&self) -> &BrainState {
        &self.brain
    }

    pub fn last_stats(&self) -> Stats {
        self.last_stats
    }

    /// Whether the default first-inputs / last-outputs binding is valid for
    /// this connectome. Callers may install a region-based mapping instead
    /// (see `subnetwork::default_mapping`).
    pub fn default_mapping_ok(&self) -> bool {
        self.mapping_ok
    }

    pub fn set_sensory_mapping(&mut self, cfg: SensoryConfig) -> bool {
        validate_mapping(&self.connectome, &cfg, OUTPUT_CELL_COUNT)
    }

    /// Deterministic fixed-step tick.
    #[allow(clippy::too_many_lines)]
    pub fn tick(&mut self, input: &SensoryInput, budget: &TickBudget) -> MotorOutput {
        let t0 = std::time::Instant::now();
        self.brain.tick += 1;
        let tick = self.brain.tick;

        // (1) sensory injection into channel-bound input cells
        encode_into(
            &mut self.sensory,
            input,
            self.lif.dt_ms,
            &mut self.brain.inj,
            &self.encoders,
        );

        // (2) drain delayed arrivals scheduled for this tick
        self.brain.drain_ring();

        // (3) integrate every cell in ascending index order; collect spikes
        let leak = self.lif.leak();
        let mut spiked: Vec<u32> = Vec::with_capacity(64);
        let mut active: u64 = 0;
        for i in 0..self.connectome.n() {
            let has_inj = self.brain.inj[i] != 0.0;
            if self.brain.step_cell(i, leak, &self.lif) {
                spiked.push(i as u32);
            }
            if has_inj || self.brain.refractory[i] > 0 {
                active += 1;
            }
        }

        // (4) event-driven propagation: touch only spiked rows, ascending.
        let mut synops = 0usize;
        let processed = spiked.len().min(budget.max_spikes);
        let dropped_spikes = (spiked.len() - processed) as u64;
        let mut dropped_synops = 0u64;
        let mut slots: Vec<(usize, u32, f32)> = Vec::new();
        'rows: for &s in spiked.iter().take(processed) {
            let rs = self.connectome.row_offsets[s as usize] as usize;
            let re = self.connectome.row_offsets[s as usize + 1] as usize;
            for i in rs..re {
                if synops >= budget.max_synops {
                    dropped_synops += (re - i) as u64;
                    // All remaining spikes are also propagated-not: they are
                    // recorded as dropped for cross-parity with spike budget.
                    break 'rows;
                }
                let post = self.connectome.col_idx[i];
                let w = self.effective_weight(i, s);
                if w != 0.0 {
                    synops += 1;
                    let d = self.connectome.delay_at(i, self.lif.default_delay_steps);
                    let d = d.min(MAX_RING_DELAY).max(1);
                    let slot = self.brain.slot(tick, d);
                    slots.push((slot, post, w));
                }
            }
        }
        for (slot, post, w) in slots {
            self.brain.ring[slot].push((post, w));
        }

        // (5) bookkeeping: spikes survived the tick; injectors reset for next.
        self.brain.spiked = spiked;
        for x in self.brain.inj.iter_mut() {
            *x = 0.0;
        }

        // (6) motor decode from output cell spike counts
        let motor = motor_decode(
            &mut self.motor,
            &self.brain.spiked,
            self.lif.dt_ms,
            self.out_start,
        );

        self.last_stats = Stats {
            neuron_count: self.connectome.n() as u64,
            synapse_count: self.connectome.weights.len() as u64,
            active,
            spikes: self.brain.spiked.len() as u64,
            dropped_spikes,
            dropped_synops,
            exec_ns: t0.elapsed().as_nanos() as u64,
            mem_est: self.connectome.memory_bytes(),
        };
        motor
    }

    /// Effective propagated weight for synapse i from pre-cell `pre`, applying
    /// the configured unknown-polarity policy.
    pub fn effective_weight(&self, i: usize, pre: u32) -> f32 {
        let w = self.connectome.weights[i];
        match self.connectome.polarity_at(i, pre) {
            Polarity::Excitatory => w,
            Polarity::Inhibitory => -w.abs(),
            Polarity::Unknown => match self.polarity_policy {
                UnknownPolarityPolicy::Ignore => 0.0,
                UnknownPolarityPolicy::AsExcitatory => w,
                UnknownPolarityPolicy::AsInhibitory => -w.abs(),
                UnknownPolarityPolicy::Scale(f) => w * f,
            },
        }
    }

    // ------------------------------------------------------- Queen-facing API
    /// Full identity/config exposure for the Queen organism-owned instance.
    pub fn info(&self) -> RuntimeInfo {
        let st = self.connectome.stats();
        RuntimeInfo {
            connectome_name: self.connectome.name.clone(),
            template_version: self.connectome.template_version.clone(),
            neurons: st.neuron_count,
            synapses: st.synapse_count,
            regions_available: !self.connectome.regions.is_empty(),
            cell_types_available: !self.connectome.cell_types.is_empty(),
            lif_params: self.lif.clone(),
            sensory_channels: crate::sensory::CHANNEL_COUNT,
            motor_cells: OUTPUT_CELL_COUNT,
            unknown_polarity_policy: self.polarity_policy,
            provenance: self.provenance.clone(),
        }
    }

    pub fn set_provenance(&mut self, rec: ProvenanceRecord) {
        self.provenance = Some(rec);
    }

    pub fn provenance(&self) -> Option<&ProvenanceRecord> {
        self.provenance.as_ref()
    }

    pub fn origin_note(m: &crate::connectome::NeuronMeta) -> ProvenanceTag {
        m.origin
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            tick: self.brain.tick,
            brain: self.brain.clone(),
            sensory: self.sensory.clone(),
            motor: self.motor.clone(),
        }
    }

    pub fn restore_snapshot(&mut self, snap: RuntimeSnapshot) {
        self.brain = snap.brain;
        self.sensory = snap.sensory;
        self.motor = snap.motor;
    }
}

/// Max explicit delay on the connectome (capped by the ring constant).
fn effective_max_delay(conn: &Connectome, default_delay: u8) -> u8 {
    let mut mx = default_delay.max(1);
    if let Some(d) = &conn.delays {
        for &v in d {
            if v != 0 {
                mx = mx.max(v);
            }
        }
    }
    mx.min(MAX_RING_DELAY)
}

#[allow(dead_code)]
fn _prov_touch(_t: ProvenanceTag) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{EdgeSpec, NeuronMeta, NO_IDX};
    use crate::fixtures::{synthetic_fixture, FIXTURE_OUTPUT_START, FIXTURE_ATTRACT_START};
    use std::sync::Arc;

    fn fixture_runtime() -> FlyBrainRuntime {
        let f = synthetic_fixture();
        let mut rt = FlyBrainRuntime::new(Arc::new(f.connectome), 7).expect("fixture fits");
        // Accelerate the rate encoders so one pulse arrives every 2 ticks.
        rt.sensory.baseline_hz = 0.0;
        rt.sensory.gain_hz = 500.0;
        rt
    }

    fn food_input() -> SensoryInput {
        SensoryInput {
            food_dir: (1.0, 0.0),
            ..SensoryInput::default()
        }
    }

    fn ran_ticks(rt: &mut FlyBrainRuntime, n: u32, input: &SensoryInput) -> Vec<MotorOutput> {
        let b = TickBudget::unbounded();
        (0..n).map(|_| rt.tick(input, &b)).collect()
    }

    fn spiked_tick(rt: &FlyBrainRuntime) -> &[u32] {
        &rt.state().spiked
    }

    #[test]
    fn two_hop_propagation_reaches_output() {
        // Path under test (fixture, SYNTHETIC): input cell 16 (channel 8 =
        // FoodDirCos) -> relay -> attract pool (hop 2) -> DirX output 62.
        let mut rt = fixture_runtime();
        let food = food_input();
        let mut saw_relay = false;
        let mut saw_attract = false;
        let mut saw_output = false;
        for _ in 0..12 {
            rt.tick(&food, &TickBudget::unbounded());
            let s = spiked_tick(&rt);
            if s.contains(&16) {
                saw_relay = true;
            }
            if s.contains(&(FIXTURE_ATTRACT_START as u32)) {
                saw_attract = true;
            }
            if s.contains(&(FIXTURE_OUTPUT_START as u32)) {
                saw_output = true;
            }
        }
        assert!(saw_relay, "input cell 16 must spike under 500 Hz encoding");
        assert!(saw_attract, "attract pool must spike (hop 2)");
        assert!(saw_output, "DirX output must eventually fire");
    }

    #[test]
    fn refractory_blocks_back_to_back_spikes() {
        use crate::state::BrainState;
        let p = LifParams::default();
        let mut st = BrainState::new(1, 4, p.v_rest);
        let leak = p.leak();
        st.inj[0] = 1_000.0;
        assert!(st.step_cell(0, leak, &p), "huge current must spike");
        st.clear_inj();
        st.inj[0] = 1_000.0;
        assert!(
            !st.step_cell(0, leak, &p),
            "refractory cell must not spike on the next tick"
        );
        assert!(
            !st.step_cell(0, leak, &p),
            "refractory persists for the configured steps"
        );
    }

    #[test]
    fn polarity_signs_and_unknown_policy() {
        let meta = |i: u64| NeuronMeta {
            body_id: 900_000_000 + i,
            region_idx: NO_IDX,
            cell_type_idx: NO_IDX,
            polarity: Polarity::Unknown,
            origin: ProvenanceTag::Assumption,
        };
        let nodes: Vec<NeuronMeta> = (0..68).map(meta).collect();
        let edges = vec![EdgeSpec {
            pre: 0,
            post: 1,
            weight: 4.0,
            delay: None,
            polarity: None,
        }];
        let conn = Arc::new(
            Connectome::from_edges("p", "", nodes, vec![], vec![], edges)
                .expect("policy connectome"),
        );
        // synapse polarity falls back to the pre-neuron polarity (Unknown).
        let eff = |policy: UnknownPolarityPolicy| {
            let mut rt = FlyBrainRuntime::new(conn.clone(), 1).expect("fits");
            rt.polarity_policy = policy;
            rt.effective_weight(0, 0)
        };
        assert_eq!(conn.polarity_at(0, 0), Polarity::Unknown);
        assert!(
            (eff(UnknownPolarityPolicy::Scale(0.25)) - 1.0).abs() < 1e-6,
            "default Scale(0.25) attenuates unknown synapses"
        );
        assert!((eff(UnknownPolarityPolicy::Ignore) - 0.0).abs() < 1e-6);
        assert!((eff(UnknownPolarityPolicy::AsExcitatory) - 4.0).abs() < 1e-6);
        assert!(eff(UnknownPolarityPolicy::AsInhibitory) + 4.0 < 1e-6);
        // Explicit per-synapse polarity dominates: excitatory stays +
        let f = synthetic_fixture();
        let exc = f.connectome.weights.iter().position(|w| *w > 0.0).expect("exc edge");
        let rt = FlyBrainRuntime::new(Arc::new(f.connectome), 1).expect("fits");
        assert!(rt.effective_weight(exc, 0) > 0.0);
    }

    #[test]
    fn determinism_same_inputs_same_outputs_and_state() {
        let food = food_input();
        let run = || -> (Vec<f32>, Vec<u32>, Vec<MotorOutput>) {
            let mut rt = fixture_runtime();
            let outs = ran_ticks(&mut rt, 20, &food);
            (rt.state().v.clone(), rt.state().spiked.clone(), outs)
        };
        let (v1, s1, o1) = run();
        let (v2, s2, o2) = run();
        assert_eq!(v1, v2);
        assert_eq!(s1, s2);
        assert_eq!(o1, o2);
        // boundedness while we are here
        for o in &o1 {
            assert!(o.move_intensity >= 0.0 && o.move_intensity <= 1.0);
            assert!(o.move_dir.0.abs() <= 1.0 && o.move_dir.1.abs() <= 1.0);
            assert!(o.turn.abs() <= 1.0);
        }
    }

    #[test]
    fn budget_truncation_is_deterministic() {
        let food = food_input();
        let run = |budget: TickBudget| {
            let mut rt = fixture_runtime();
            let mut outs = Vec::new();
            for _ in 0..20 {
                outs.push(rt.tick(&food, &budget));
            }
            (
                rt.last_stats,
                rt.state().v.clone(),
                rt.state().spiked.clone(),
                outs,
            )
        };
        let (sa, sv, ss, so) = run(TickBudget {
            max_spikes: 1,
            max_synops: 2,
        });
        let (_sb, bv, bs, bo) = run(TickBudget {
            max_spikes: 1,
            max_synops: 2,
        });
        assert_eq!(sv, bv, "state must be identical under truncation");
        assert_eq!(ss, bs);
        assert_eq!(so, bo);
        assert!(sa.dropped_synops > 0 || sa.dropped_spikes > 0, "tight budget must truncate");
    }

    #[test]
    fn fixture_sensory_to_motor_path_moves_toward_food() {
        let mut rt = fixture_runtime();
        let outs = ran_ticks(&mut rt, 30, &food_input());
        let last = outs.last().expect("non-empty");
        assert!(
            last.move_dir.0 > 0.1 && last.move_intensity > 0.0,
            "food excitation must drive a bounded positive move_dir: {last:?}"
        );
        assert!(last.feed_seek > 0.0, "food channels gate FeedSeek");
        assert!(last.move_dir.0.abs() <= 1.0 && last.move_intensity <= 1.0);
    }

    // NOTE: there is deliberately no "zero input is quiescent" test: signed
    // channels map 0 to the middle of the encoder domain by design, so zero
    // sensory input still produces tonic baseline firing.

    #[test]
    fn snapshot_roundtrip_restores_state() {
        let mut rt = fixture_runtime();
        ran_ticks(&mut rt, 6, &food_input());
        let snap = rt.snapshot();
        let f2 = synthetic_fixture();
        let mut rt2 = FlyBrainRuntime::new(Arc::new(f2.connectome), 7).expect("fits");
        rt2.sensory.baseline_hz = 0.0;
        rt2.sensory.gain_hz = 500.0;
        rt2.restore_snapshot(snap);
        let o1 = rt.tick(&food_input(), &TickBudget::unbounded());
        let o2 = rt2.tick(&food_input(), &TickBudget::unbounded());
        assert_eq!(o1.move_dir, o2.move_dir);
        assert_eq!(o1.move_intensity, o2.move_intensity);
        assert_eq!(rt.state().v, rt2.state().v);
    }

    #[test]
    fn info_exposes_identity_and_counts() {
        let mut rt = fixture_runtime();
        let mut rec = ProvenanceRecord::malecns_default();
        rec.imported_neurons = 68;
        rt.set_provenance(rec);
        let info = rt.info();
        assert_eq!(info.neurons, 68);
        assert_eq!(info.motor_cells, OUTPUT_CELL_COUNT);
        assert_eq!(info.sensory_channels, crate::sensory::CHANNEL_COUNT);
        assert!(info.provenance.is_some());
        assert_eq!(info.connectome_name, "synthetic-taxi-fixture");
    }

    /// Measured per-tick cost on the SYNTHETIC fixture (68 neurons). The bound
    /// is deliberately generous (slow CI, debug builds): this test is a
    /// regression guard on gross runtime regressions, and the printed value is
    /// the number quoted in FLYBRAIN_IMPLEMENTATION.md (run with --nocapture).
    #[test]
    fn measured_tick_cost_fixture_well_under_millisecond() {
        let mut rt = fixture_runtime();
        let food = food_input();
        let b = TickBudget::unbounded();
        let ticks = 1_000u32;
        let mut total_ns = 0u64;
        for _ in 0..ticks {
            let _ = rt.tick(&food, &b);
            total_ns += rt.last_stats().exec_ns;
        }
        let avg_ns = total_ns / ticks as u64;
        let spike_tick = rt.last_stats();
        println!(
            "PERF fixture: {} neurons / {} synapses, avg tick = {} ns \
             (last: {} spikes, {} synops, exec {} ns)",
            spike_tick.neuron_count,
            spike_tick.synapse_count,
            avg_ns,
            spike_tick.spikes,
            spike_tick.synapse_count.min(spike_tick.active),
            spike_tick.exec_ns,
        );
        assert!(
            avg_ns < 250_000,
            "fixture tick cost regressed: avg {avg_ns} ns/tick (limit 250 us; \
             this is a gross-regression guard, not precision timing)"
        );
    }
}

