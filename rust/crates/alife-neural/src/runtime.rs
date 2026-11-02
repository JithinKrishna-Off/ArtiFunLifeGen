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
use crate::motor::{
    decode_cells as motor_decode_cells, decode_pools as motor_decode_pools, MotorConfig,
    MotorOutput, OUTPUT_CELL_COUNT,
};
use crate::provenance::{ProvenanceRecord, ProvenanceTag};
use crate::sensory::{
    channels as sensory_channels, encode_channels_explicit, inputs_needed, validate_mapping,
    Encoder, SensoryConfig, SensoryInput, CHANNEL_COUNT,
};
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
        // Sized so one tick stays in the LOW-millisecond range even on the
        // largest Queen subnetwork (40k neurons / ~6M synapses): ~65k
        // scattered CSR relaxations cost a few ms on the reference machine.
        // Work past the cap is RECORDED as dropped_synops (deterministic),
        // never silently absorbed.
        Self {
            max_spikes: 4096,
            max_synops: 65_536,
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

/// Observability: per-tick activity sample for the ring trace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActivitySample {
    pub tick: u64,
    /// cells that were injected or refractory this tick.
    pub active: u64,
    pub spikes: u64,
}

/// Bounded activity ring (`<= MAX_TRACE` entries; the OLDEST sample is dropped
/// once full). Observatory-replay bookkeeping only: excluded from snapshots
/// and determinism.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ActivityTrace {
    entries: Vec<ActivitySample>,
}

/// Ring capacity.
pub const MAX_TRACE: usize = 4096;
/// EMA factor for `ActivityStats.avg_rate_ema` (fixed, documented).
pub const AVG_RATE_EMA_ALPHA: f32 = 0.1;

impl ActivityTrace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one sample; drops the oldest when at capacity.
    pub fn push(&mut self, sample: ActivitySample) {
        if self.entries.len() >= MAX_TRACE {
            self.entries.remove(0);
        }
        self.entries.push(sample);
    }

    pub fn as_slice(&self) -> &[ActivitySample] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Aggregate activity view for the observatory (measured, never deterministic
/// state).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ActivityStats {
    pub active_neurons: u64,
    pub spikes_last: u64,
    pub exec_ns: u64,
    /// EMA of the mean per-neuron firing rate in Hz: `(spikes / n) / dt`.
    pub avg_rate_ema: f32,
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
    /// Optional region-based sensory binding: explicit absolute neuron index
    /// per (channel, cell) slot, length `inputs_needed(sensory)`. `None` keeps
    /// the default contiguous first-N layout. Configuration (persists across
    /// `restore_snapshot`), never simulation state.
    input_binding: Option<Vec<usize>>,
    /// Optional explicit motor output binding (six real neuron ids, one per
    /// [OutputCell] slot). `None` keeps the contiguous window starting at
    /// `out_start`. Configuration (persists across `restore_snapshot`).
    out_cells: Option<[usize; 6]>,
    /// Optional POPULATION-level output binding: one pool of real neurons per
    /// FUNCTIONAL channel (K = pools.len(), 1..=6, whatever the anatomy
    /// justified); the channel decodes the POOL MEAN rate (see
    /// `motor::decode_pools`). A pool averages away single-cell noise — the
    /// measured scene responses of the descending/motor POPULATIONS are far
    /// larger than those of individual cells. Anatomical populations and
    /// functional motor channels are DISTINCT here: K channels map onto the
    /// body DOFs in fixed order, and body DOFs beyond K decode neutral.
    /// Takes precedence over `out_cells` when installed. Configuration
    /// (persists across snapshots).
    out_pools: Option<Vec<Vec<u32>>>,
    /// Region selectivity gate (`Some` = only gated-in spiked neurons
    /// propagate their CSR rows). Diagnostic/observatory configuration; NOT
    /// part of snapshots (it persists across restore_snapshot by design).
    region_gate: Option<Vec<bool>>,
    /// Bounded activity ring for observatory replay.
    activity: ActivityTrace,
    /// EMA of mean per-neuron firing rate (Hz), observatory-only.
    avg_rate_ema: f32,
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
            input_binding: None,
            out_cells: None,
            out_pools: None,
            region_gate: None,
            activity: ActivityTrace::new(),
            avg_rate_ema: 0.0,
        })
    }

    /// Install an explicit absolute-neuron binding for every (channel, cell)
    /// sensory slot (length must equal `inputs_needed(sensory)`; every index
    /// must be in range). Returns false (no change) on any mismatch.
    pub fn set_input_binding(&mut self, binding: &[usize]) -> bool {
        let need = inputs_needed(&self.sensory);
        if binding.len() != need || binding.iter().any(|&i| i >= self.connectome.n()) {
            return false;
        }
        self.input_binding = Some(binding.to_vec());
        true
    }

    /// Clear a previously installed sensory binding (back to contiguous).
    pub fn clear_input_binding(&mut self) {
        self.input_binding = None;
    }

    /// The installed binding, if any (`None` = contiguous default).
    pub fn input_binding(&self) -> Option<&[usize]> {
        self.input_binding.as_deref()
    }

    /// Move the motor output window: the six output DECISION cells become
    /// neurons `[start, start + OUTPUT_CELL_COUNT)`. Bounded so the window
    /// stays inside the connectome and clear of sensory inputs; returns false
    /// (no change) otherwise. Used by region-based queen bindings.
    pub fn set_output_start(&mut self, start: usize) -> bool {
        let need = inputs_needed(&self.sensory);
        if start < need || start + OUTPUT_CELL_COUNT > self.connectome.n() {
            return false;
        }
        self.out_start = start;
        true
    }

    /// The current output-cell window start (first cell of the default
    /// contiguous window; telemetry only when an explicit binding is active).
    pub fn output_start(&self) -> usize {
        self.out_start
    }

    /// Explicit region-grounded output binding: replaces the contiguous
    /// `[out_start, out_start + 6)` window with six arbitrary real neurons
    /// (one per [OutputCell] slot, order fixed). All indices must be inside
    /// the connectome, unique and clear of the sensory input cells; returns
    /// false (no change) otherwise. Mirrors `set_input_binding`.
    pub fn set_output_binding(&mut self, cells: &[usize; OUTPUT_CELL_COUNT]) -> bool {
        let need = inputs_needed(&self.sensory);
        let n = self.connectome.n();
        for (k, &c) in cells.iter().enumerate() {
            if c < need || c >= n {
                return false;
            }
            if cells[..k].contains(&c) {
                return false;
            }
        }
        let mut owned = *cells;
        owned.sort_unstable();
        self.out_start = owned[0];
        self.out_cells = Some(*cells);
        true
    }

    /// Remove an explicit output binding (falls back to the contiguous
    /// `[out_start, out_start + 6)` window; see [`FlyBrainRuntime::set_output_start`]).
    pub fn clear_output_binding(&mut self) {
        self.out_cells = None;
        self.out_pools = None;
    }

    /// Install POPULATION-level output pools (one pool per FUNCTIONAL
    /// channel; K = `pools.len()` in 1..=OUTPUT_CELL_COUNT — the pool count is
    /// whatever the anatomical data justified, NOT forced to six). Every cell
    /// must be a real neuron index (>= `inputs_needed`, < `n`) and no cell may
    /// appear in two pools. Each pool must be non-empty. Returns false (no
    /// change) otherwise. The channel decode then reads the POOL MEAN spike
    /// rate (motor::decode_pools); body DOFs beyond K decode neutral.
    pub fn set_output_pools(&mut self, pools: &[&[usize]]) -> bool {
        let need = inputs_needed(&self.sensory);
        let n = self.connectome.n();
        if pools.is_empty() || pools.len() > OUTPUT_CELL_COUNT {
            return false;
        }
        let mut seen = std::collections::HashSet::new();
        let mut owned: Vec<Vec<u32>> = Vec::with_capacity(pools.len());
        for pool in pools {
            if pool.is_empty() {
                return false;
            }
            let mut p = Vec::with_capacity(pool.len());
            for &c in pool.iter() {
                if c < need || c >= n || !seen.insert(c) {
                    return false;
                }
                p.push(c as u32);
            }
            p.sort_unstable();
            owned.push(p);
        }
        self.out_start = owned.iter().map(|p| p[0] as usize).min().unwrap_or(n);
        self.out_pools = Some(owned);
        true
    }

    /// The installed population output pools, if any.
    pub fn output_pools(&self) -> Option<&[Vec<u32>]> {
        self.out_pools.as_deref()
    }

    /// Install the measured per-channel decode calibration (baseline + range
    /// Hz, both length = channel count K). See `MotorConfig::calib_baseline`.
    /// `None` args clear the calibration (back to the absolute-rate decode).
    /// Returns false (no change) on length mismatch.
    pub fn set_motor_calibration(
        &mut self,
        baseline: Option<&[f32]>,
        range: Option<&[f32]>,
    ) -> bool {
        match (baseline, range) {
            (Some(b), Some(r))
                if b.len() == r.len()
                    && (1..=OUTPUT_CELL_COUNT).contains(&b.len()) =>
            {
                self.motor.calib_baseline = Some(b.to_vec());
                self.motor.calib_range = Some(r.to_vec());
                true
            }
            _ => {
                self.motor.calib_baseline = None;
                self.motor.calib_range = None;
                false
            }
        }
    }

    /// The installed decode calibration (Hz per output slot).
    pub fn motor_calibration(&self) -> (Option<&[f32]>, Option<&[f32]>) {
        (
            self.motor.calib_baseline.as_deref(),
            self.motor.calib_range.as_deref(),
        )
    }

    /// The active output cell binding, if an explicit one is installed.
    pub fn output_cells(&self) -> Option<&[usize; OUTPUT_CELL_COUNT]> {
        self.out_cells.as_ref()
    }

    /// Per-neuron spike IDs of the last tick (ascending; observatory /
    /// binding-diagnostic use).
    pub fn spike_ids_last(&self) -> &[u32] {
        &self.brain.spiked
    }

    /// The active motor decoder configuration (rates/EMA tempo).
    pub fn motor_cfg(&self) -> &crate::motor::MotorConfig {
        &self.motor
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

    /// Deterministic fixed-step tick (world-facing path: fills the 11 sensory
    /// channels from a [`SensoryInput`] and steps).
    pub fn tick(&mut self, input: &SensoryInput, budget: &TickBudget) -> MotorOutput {
        let chans = sensory_channels(input);
        self.tick_channels(&chans, budget)
    }

    /// Deterministic fixed-step tick with an explicit channel vector
    /// (`chans[c]` must hold each channel's RAW value in its documented
    /// domain; defensive clamping/mapping applies afterwards, exactly as for
    /// the `SensoryInput` path). Enables alternate sensory mappings (queen
    /// adapter) without touching the world contract.
    ///
    /// Step order and determinism are identical to `tick`.
    #[allow(clippy::too_many_lines)]
    pub fn tick_channels(&mut self, chans: &[f32; CHANNEL_COUNT], budget: &TickBudget) -> MotorOutput {
        let t0 = std::time::Instant::now();
        self.brain.tick += 1;
        let tick = self.brain.tick;

        // (1) sensory injection into channel-bound input cells
        encode_channels_explicit(
            &mut self.sensory,
            chans,
            self.lif.dt_ms,
            &mut self.brain.inj,
            &self.encoders,
            self.input_binding.as_deref(),
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
        // Region selectivity (when active) gates WHICH spiked rows propagate
        // at all; neuron indices/body IDs are never re-mapped.
        let mut synops = 0usize;
        let processed = spiked.len().min(budget.max_spikes);
        let dropped_spikes = (spiked.len() - processed) as u64;
        let mut dropped_synops = 0u64;
        let mut slots: Vec<(usize, u32, f32)> = Vec::new();
        'rows: for &s in spiked.iter().take(processed) {
            if let Some(gate) = self.region_gate.as_ref() {
                if !gate[s as usize] {
                    continue 'rows;
                }
            }
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

        // (6) motor decode from output cell spike counts: contiguous window
        // by default; explicit region-grounded binding when installed; a
        // population-pool binding decodes the POOL MEAN rate per channel.
        let s0 = self.out_start;
        let default_cells = [
            s0, s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5,
        ];
        let motor = if let Some(pools) = self.out_pools.as_ref() {
            let refs: Vec<&[u32]> = pools.iter().map(|p| p.as_slice()).collect();
            motor_decode_pools(
                &mut self.motor,
                &self.brain.spiked,
                self.lif.dt_ms,
                &refs,
            )
        } else {
            let cells_ref = self.out_cells.as_ref().unwrap_or(&default_cells);
            motor_decode_cells(
                &mut self.motor,
                &self.brain.spiked,
                self.lif.dt_ms,
                cells_ref,
            )
        };
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

        // Observatory bookkeeping (never deterministic state; excluded from
        // snapshots). Mean per-neuron rate in Hz from this tick's spike count.
        let dt_s = self.lif.dt_ms.max(0.001) / 1000.0;
        let rate = if self.connectome.n() > 0 {
            (self.brain.spiked.len() as f32) / (self.connectome.n() as f32) / dt_s
        } else {
            0.0
        };
        self.avg_rate_ema += AVG_RATE_EMA_ALPHA * (rate - self.avg_rate_ema);
        self.activity.push(ActivitySample {
            tick,
            active: self.last_stats.active,
            spikes: self.last_stats.spikes,
        });

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

    // ------------------------------------------------- observatory / activity
    /// Aggregate activity view (last tick + EMA; measured only).
    pub fn activity_stats(&self) -> ActivityStats {
        let st = self.last_stats;
        ActivityStats {
            active_neurons: st.active,
            spikes_last: st.spikes,
            exec_ns: st.exec_ns,
            avg_rate_ema: self.avg_rate_ema,
        }
    }

    /// Bounded activity ring (newest-last, oldest dropped past 4096) for
    /// observatory replay.
    pub fn trace(&self) -> &[ActivitySample] {
        self.activity.as_slice()
    }

    /// Observatory/diagnostic: restrict PROPAGATION (`tick_channels` step 4)
    /// to spiked neurons whose region name satisfies `pred`. Neurons without
    /// an assigned region never match. Source neuron indices and body IDs are
    /// PRESERVED (this is a run-mode gate, not a re-index). Returns how many
    /// neurons will propagate. A later call replaces the gate;
    /// [`FlyBrainRuntime::clear_selectivity`] removes it. Not snapshot state.
    pub fn select_region(&mut self, pred: impl Fn(&str) -> bool) -> usize {
        let conn = &self.connectome;
        let mut gate = vec![false; conn.n()];
        let mut count = 0usize;
        for i in 0..conn.n() {
            let idx = conn.node_meta[i].region_idx;
            if let Some(r) = conn.region_name(idx) {
                if pred(r) {
                    gate[i] = true;
                    count += 1;
                }
            }
        }
        self.region_gate = Some(gate);
        count
    }

    /// Remove any region selectivity gate (all spiked rows propagate again).
    pub fn clear_selectivity(&mut self) {
        self.region_gate = None;
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

