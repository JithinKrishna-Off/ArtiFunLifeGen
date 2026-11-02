//! `QueenBrainAdapter`: queen-brain implementation ON TOP of a
//! [`FlyBrainRuntime`] for the alife-core coordinator.
//!
//! It is the ONLY piece of glue between the coordinator's queen contract and
//! this crate's spiking runtime; the runtime itself never knows about
//! coordinator types.
//!
//! Mapping (documented model decisions, all channels in use â€” no zeroed
//! islands; see [`QueenSensoryMapping`]):
//! - `food_dir`            -> food cos/sin channel pair (chemical-taxis pair)
//! - `food_dist`           -> food-proximity channel, linear rate encoding of
//!                            `1 / (1 + dist)` (bounded monotone inverse)
//! - `heading`             -> (cos, sin) pair on the temperature-proxy
//!                            (light-alias) directional channels
//! - `speed`               -> optic-speed channel (rate)
//! - `energy_frac`         -> ground-contact channel (rate)
//! - `hazard`              -> obstacle-proximity channel (rate)
//! - `edge_dist` / `density` -> the two slots of the formerly directional
//!                            optic pool, reused as scalar rate slots (the
//!                            optic-flow direction is NOT represented)
//! - `signal_sample`       -> queen-signal channel (rate)
//!
//! Motor decoding: `move_dir` normalized to a unit-or-zero vector; and the
//! rest are hard-clamped to the documented bounds. Behavior there is
//! `FlyBrainRuntime`'s bounded motor decode (EI decoding from output-cell
//! rate EMAs) â€” no re-interpretation, only normalizing + clamping.
//!
//! Determinism: the runtime is RNG-free; the adapter adds none, so repeated
//! identical advances produce identical outputs.
//!
//! Save format: `runtime.snapshot()` serialized with `serde_json` (bytes).
//! Restoring requires the SAME connectome (neuron-count check is a safety
//! net, NOT a proof â€” compare `identity()` when switching organisms).

use std::sync::Arc;

use serde_json::Value;

use crate::connectome::Connectome;
use crate::connectome_binary::connectome_identity;
use crate::fixtures::synthetic_fixture;
use crate::motor::MotorOutput;
use crate::runtime::{FlyBrainRuntime, RuntimeBuildError};
use crate::sensory::{self, CHANNEL_COUNT};
use crate::queen_types::{
    NeuralTickBudget, QueenBrainStats, QueenMotorOut, SensoryInputForQueen,
};

/// Channel assignment for each queen sense (default: every one of the 11
/// sensory channels is used exactly once â€” configurable).
///
/// Threat-channel decision (documented model choice): the optic DIRECTION
/// pair is reused for the THREAT direction (the optic-flow concept â€” visual
/// detection of approaching motion â€” is the closest sensory analogy in the
/// fixed 11-channel contract), and the optic-speed channel carries the threat
/// approach-rate contribution. `edge_dist`/`density` (wrap worlds make edge
/// distance constant; density adds little) are no longer mapped by the
/// default mapping but stay optional (`None` channels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueenSensoryMapping {
    pub food_dir_cos: usize,
    pub food_dir_sin: usize,
    pub food_prox: usize,
    pub heading_cos: usize,
    pub heading_sin: usize,
    pub speed: usize,
    pub energy_frac: usize,
    pub hazard: usize,
    pub edge_dist: Option<usize>,
    pub density: Option<usize>,
    pub signal: usize,
    /// threat direction cos/sin (optic channels, WORLD frame)
    pub hazard_cos: usize,
    pub hazard_sin: usize,
    /// threat approach rate folded onto the optic-speed channel
    pub hazard_rate: usize,
}

impl Default for QueenSensoryMapping {
    fn default() -> Self {
        Self {
            food_dir_cos: sensory::CH_FOOD_COS,
            food_dir_sin: sensory::CH_FOOD_SIN,
            food_prox: sensory::CH_FOOD_PROX,
            heading_cos: sensory::CH_LIGHT_COS,
            heading_sin: sensory::CH_LIGHT_SIN,
            speed: sensory::CH_OPTIC_SPEED,
            energy_frac: sensory::CH_GROUND,
            hazard: sensory::CH_OBSTACLE,
            edge_dist: None,
            density: None,
            signal: sensory::CH_QUEEN,
            hazard_cos: sensory::CH_OPTIC_COS,
            hazard_sin: sensory::CH_OPTIC_SIN,
            hazard_rate: sensory::CH_OPTIC_SPEED,
        }
    }
}

pub struct QueenBrainAdapter {
    runtime: FlyBrainRuntime,
    mapping: QueenSensoryMapping,
    /// Diagnostic channel overrides (`sensory_injection`); raw channel slot
    /// -> forced value for subsequent ticks, cleared by
    /// [`QueenBrainAdapter::clear_sensory_injections`].
    overrides: Vec<Option<f32>>,
    ticks_run: u64,
    last_out: QueenMotorOut,
}

impl QueenBrainAdapter {
    /// Wrap an existing runtime (mapping is the defaults unless overridden).
    pub fn new(runtime: FlyBrainRuntime) -> Self {
        Self::with_mapping(runtime, QueenSensoryMapping::default())
    }

    pub fn with_mapping(runtime: FlyBrainRuntime, mapping: QueenSensoryMapping) -> Self {
        Self {
            runtime,
            mapping,
            overrides: vec![None; CHANNEL_COUNT],
            ticks_run: 0,
            last_out: QueenMotorOut::default(),
        }
    }

    /// Default fixture brain (SYNTHETIC 68-neuron test circuit â€” never
    /// presented as real fly anatomy). Keeps the contiguous binding the
    /// fixture was designed for.
    pub fn new_for_fixture(seed: u32) -> Self {
        let f = synthetic_fixture();
        let rt = FlyBrainRuntime::new(Arc::new(f.connectome), seed).expect("fixture fits runtime");
        Self::new(rt)
    }

    /// Real circuits: subnetwork / full connectome path. Installs the
    /// region-grounded binding of [`propose_region_binding`] (real sensory
    /// subpopulations for the afferent channels, real descending/motor
    /// subpopulations for the output window) when the connectome carries
    /// region labels; otherwise the contiguous default binding applies
    /// unchanged (no fabricated mapping). Fails with
    /// [`RuntimeBuildError::TooSmall`] when the connectome cannot host the
    /// current sensory/motor binding.
    pub fn new_for_subnetwork(
        connectome: Arc<Connectome>,
        seed: u32,
    ) -> Result<Self, RuntimeBuildError> {
        Self::new_for_subnetwork_with_cells(connectome, seed, 2)
    }

    /// [`new_for_subnetwork`] with an explicit afferent cells-per-channel
    /// count. Population encoding of each sensory channel over N real cells
    /// (of the annotated subpopulation) is the honest way to raise signal
    /// contrast against the network's intrinsic background; N is a documented
    /// model decision.
    pub fn new_for_subnetwork_with_cells(
        connectome: Arc<Connectome>,
        seed: u32,
        cells_per_channel: usize,
    ) -> Result<Self, RuntimeBuildError> {
        let mut rt = FlyBrainRuntime::new(connectome, seed)?;
        // Queen-channel calibration (measured operating point; the LIF engine
        // is untouched): the 40k-cell network self-runs at ~19 Hz mean, so the
        // stock 4..30 Hz afferent rate window is drowned out. The queen's
        // afferent cells encode [0,1] channels over a wider 3..63 Hz rate
        // window (rate encoder gain), giving scene changes enough contrast to
        // be visible through the tonic intrinsic background.
        rt.sensory.gain_hz = 60.0;
        rt.sensory.baseline_hz = 3.0;
        rt.sensory.channel_cells = cells_per_channel.max(1);
        // re-derive the encoders vector for the new slot count
        rt.encoders
            .resize(crate::sensory::inputs_needed(&rt.sensory), crate::sensory::Encoder::Rate);
        if let Some((binding, motor_candidates)) = propose_region_binding(rt.connectome(), rt.sensory.channel_cells) {
            rt.set_input_binding(&binding);
            auto_tune_output_binding(&mut rt, &motor_candidates);
        }
        Ok(Self::new(rt))
    }

    /// Build the Queen brain over a REAL imported connectome (e.g. the
    /// MaleCNS v1.0 Queen subnetwork from `malecns_bins::subnetwork_for_queen`,
    /// loaded via `connectome_binary`). Identical contract to
    /// [`new_for_subnetwork`]: the default first-inputs / last-outputs
    /// sensory/motor binding applies (a documented structural placeholder for
    /// imported anatomical graphs â€” the runtime flags it via
    /// `default_mapping_ok()` unless a region-based mapping is installed).
    pub fn new_from_connectome(
        connectome: Arc<Connectome>,
        seed: u32,
    ) -> Result<Self, RuntimeBuildError> {
        Self::new_for_subnetwork(connectome, seed)
    }

    pub fn runtime(&self) -> &FlyBrainRuntime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut FlyBrainRuntime {
        &mut self.runtime
    }

    pub fn mapping(&self) -> &QueenSensoryMapping {
        &self.mapping
    }

    /// One coordinator advance: map senses into the channel vector, run up to
    /// `budget.max_ticks_per_advance` 1 ms brain ticks (senses held constant
    /// during the burst; 0 = paused, replays the last output unchanged), and
    /// decode the bounded motor output from the LAST tick.
    pub fn tick(
        &mut self,
        senses: &SensoryInputForQueen,
        budget: &NeuralTickBudget,
    ) -> QueenMotorOut {
        let chans = self.map_channels(senses);
        let mut out = self.last_out;
        for _ in 0..budget.max_ticks_per_advance {
            let m = self
                .runtime
                .tick_channels(&chans, &crate::runtime::TickBudget::default());
            out = decode_queen_motor(m);
        }
        if budget.max_ticks_per_advance > 0 {
            self.ticks_run += budget.max_ticks_per_advance as u64;
            self.last_out = out;
        }
        out
    }

    /// Diagnostic (observatory) only: force one raw sensory channel from the
    /// next ticks on. Channel indices are the crate's `CH_*` constants;
    /// values are defensively clamped to the channel's documented domain.
    pub fn sensory_injection(&mut self, channel: usize, value: f32) {
        if channel >= CHANNEL_COUNT {
            return;
        }
        self.overrides[channel] = Some(value);
    }

    /// Remove all sensory channel overrides.
    pub fn clear_sensory_injections(&mut self) {
        self.overrides = vec![None; CHANNEL_COUNT];
    }

    pub fn has_sensory_injections(&self) -> bool {
        self.overrides.iter().any(Option::is_some)
    }

    /// Serialize the underlying runtime state (brain + encoders' config).
    /// Override injections and `ticks_run` are deliberately NOT included.
    pub fn snapshot(&self) -> Vec<u8> {
        serde_json::to_vec(&self.runtime.snapshot()).unwrap_or_default()
    }

    /// Restore from [`QueenBrainAdapter::snapshot`] bytes. `false` on any
    /// parse/shape mismatch (state left untouched on failure).
    pub fn restore(&mut self, bytes: &[u8]) -> bool {
        let Ok(v) = serde_json::from_slice::<Value>(bytes) else {
            return false;
        };
        let Ok(snap) = serde_json::from_value::<crate::runtime::RuntimeSnapshot>(v) else {
            return false;
        };
        if snap.brain.v.len() != self.runtime.connectome().n() {
            return false;
        }
        self.runtime.restore_snapshot(snap);
        true
    }

    /// Aggregate stats for the coordinator (measured only).
    pub fn stats(&self) -> QueenBrainStats {
        let st = self.runtime.last_stats();
        let act = self.runtime.activity_stats();
        QueenBrainStats {
            ticks: self.ticks_run,
            active_neurons: act.active_neurons,
            spikes_last: act.spikes_last,
            avg_rate_ema: act.avg_rate_ema,
            exec_ns: act.exec_ns.max(st.exec_ns),
        }
    }

    /// `(name, template_version)` from the connectome asset identity.
    pub fn identity(&self) -> (String, String) {
        let (name, version, _, _, _) = connectome_identity(self.runtime.connectome());
        (name, version)
    }

    /// Raw channel vector for one queen sample (diagnostics/tests).
    pub fn map_channels(&self, senses: &SensoryInputForQueen) -> [f32; CHANNEL_COUNT] {
        let mut ch = [0.0f32; CHANNEL_COUNT];
        let m = self.mapping;
        ch[m.food_dir_cos] = safe_dir(senses.food_dir).0;
        ch[m.food_dir_sin] = safe_dir(senses.food_dir).1;
        ch[m.food_prox] = (1.0 / (senses.food_dist.abs() + 1.0)).clamp(0.0, 1.0);
        let (hc, hs) = (senses.heading.cos(), senses.heading.sin());
        ch[m.heading_cos] = hc.clamp(-1.0, 1.0);
        ch[m.heading_sin] = hs.clamp(-1.0, 1.0);
        // optic-speed channel: own motion + threat approach (max = dominance)
        ch[m.speed] = senses.speed.clamp(0.0, 1.0).max(senses.hazard_rate.clamp(0.0, 1.0));
        ch[m.energy_frac] = senses.energy_frac.clamp(0.0, 1.0);
        ch[m.hazard] = senses.hazard.clamp(0.0, 1.0);
        if let Some(c) = m.edge_dist {
            ch[c] = senses.edge_dist.clamp(0.0, 1.0);
        }
        if let Some(c) = m.density {
            ch[c] = senses.density.clamp(0.0, 1.0);
        }
        // threat DIRECTION (signed optic pool) â€” the approaching hazard's
        // world-frame bearing; (0,0) when no threat is sensed
        ch[m.hazard_cos] = safe_dir(senses.hazard_dir).0;
        ch[m.hazard_sin] = safe_dir(senses.hazard_dir).1;
        ch[m.signal] = senses.signal_sample.clamp(0.0, 1.0);
        for (c, ov) in self.overrides.iter().enumerate() {
            if let Some(v) = ov {
                ch[c] = crate::sensory::mapped_value(c, *v);
            }
        }
        ch
    }
}

/// Normalize (or zero) a direction vector (NaN/zero-safe).
fn safe_dir(d: (f32, f32)) -> (f32, f32) {
    let m = (d.0 * d.0 + d.1 * d.1).sqrt();
    if m <= 1e-6 || !m.is_finite() {
        (0.0, 0.0)
    } else {
        (d.0 / m, d.1 / m)
    }
}

/// Unit-normalize runtime motor output into the bounded queen output.
pub fn decode_queen_motor(m: MotorOutput) -> QueenMotorOut {
    QueenMotorOut {
        move_dir: safe_dir(m.move_dir),
        move_intensity: m.move_intensity.clamp(0.0, 1.0),
        turn: m.turn.clamp(-1.0, 1.0),
        feed_seek: m.feed_seek.clamp(0.0, 1.0),
        emit_signal: m.emit_signal.clamp(0.0, 1.0),
    }
}

// ==================================================== region-based binding

/// One ANNOTATED motor class proposed for output pooling: real neurons of the
/// named MaleCNS superclass (e.g. `cb_motor`, `efferent_ascending`,
/// `vnc_motor`, `descending_neuron`), highest in-degree first (one CSR scan)
/// so the probe reads well-driven members of the class.
#[derive(Clone, Debug)]
pub struct MotorClassCandidate {
    pub label: String,
    pub cells: Vec<usize>,
}

/// Region-grounded sensory binding for an IMPORTED connectome
/// (documented model decision — the connectivity itself is untouched):
///
/// - Afferent channels are pooled onto real sensory subpopulations by their
///   superclass label — limited by the import's MEASURED census: there are
///   no gustatory receptors in this subnetwork, so food cues ride disjoint
///   `ol_sensory` slices as odor streams (documented, not invented
///   anatomy): visual channels onto visual/`ol_sensory` cells, threat
///   channel onto another `ol_sensory` slice, ground contact onto
///   `vnc_sensory` mechanoreceptors, food channels onto further
///   `ol_sensory` slices, and the queen-signal channel onto ascending
///   sensory cells.
/// - Motor output decodes POOL MEAN rates over FUNCTIONAL channels mined from
///   the annotated motor classes (see [`MotorClassCandidate`] and
///   [`auto_tune_output_binding`] — anatomical populations and motor channels
///   are distinct; the channel count is however many the data defends).
///
/// Every cell is a real, labeled neuron — nothing is fabricated. Deterministic.
/// Returns `None` when the connectome lacks labels (or is too small) so the
/// caller keeps the contiguous default binding unchanged.
pub fn propose_region_binding(
    conn: &Connectome,
    cells_per_channel: usize,
) -> Option<(Vec<usize>, Vec<MotorClassCandidate>)> {
    if conn.regions.is_empty() || conn.n() < 120 {
        return None;
    }
    let n = conn.n();
    let class_of = |i: usize| -> Option<&str> { conn.region_name(conn.node_meta[i].region_idx) };

    // Motor candidates grouped by their real superclass label
    // ("sensory_descending" stays afferent — it is a sensory class).
    let mut indeg = vec![0u16; n];
    for i in 0..n {
        let (rs, re) = (conn.row_offsets[i] as usize, conn.row_offsets[i + 1] as usize);
        for p in rs..re {
            let post = conn.col_idx[p] as usize;
            indeg[post] = indeg[post].saturating_add(1);
        }
    }
    const PER_CLASS_CAP: usize = 24;
    // the descending class is data-mined for response-profile ENSEMBLES, so
    // it keeps a larger candidate set; all other classes stay small
    const PER_CLASS_CAP_DESC: usize = 96;
    let mut classes: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for i in 0..n {
        let Some(r) = class_of(i) else { continue };
        let motor_like = (r.contains("motor") && !r.contains("sensory"))
            || (r.contains("descending") && !r.contains("sensory"))
            || (r.contains("efferent"));
        if !motor_like {
            continue;
        }
        let entry = classes.entry(r.to_string()).or_default();
        entry.push(i);
    }
    // keep motor-like sets that can form a sensible pool, sorted by indeg
    let classes: Vec<MotorClassCandidate> = {
        let mut out: Vec<MotorClassCandidate> = classes
            .into_iter()
            .map(|(label, mut cells)| {
                cells.sort_unstable_by(|&a, &b| indeg[b].cmp(&indeg[a]).then(a.cmp(&b)));
                let cap = if label.contains("descending") { PER_CLASS_CAP_DESC } else { PER_CLASS_CAP };
                cells.truncate(cap);
                MotorClassCandidate { label, cells }
            })
            .filter(|c| !c.cells.is_empty())
            .collect();
        // deterministic ordering: big classes first (decls below)
        out.sort_unstable_by(|a, b| b.cells.len().cmp(&a.cells.len()).then(a.label.cmp(&b.label)));
        out
    };

    // Afferent pools: RAW candidate lists (first untaken members fill slots;
    // the motor-class candidates are marked taken up front so they cannot
    // leak into the afferent binding).
    let mut taken = vec![false; n];
    for cls in &classes {
        for &c in &cls.cells {
            taken[c] = true;
        }
    }
    // Measured anatomical census of the imported subnetwork (label_census):
    // REAL peripheral chemoreceptors (sugar/gustatory/taste/odor naming) DO
    // NOT EXIST — the only chemosensory class present is `ol_sensory`
    // (n=3335), plus mechanosensory `vnc_sensory` (n=301) and visual
    // populations. Bindings therefore say what the data supports and no
    // more:
    // - optical-flow / heading channels (CH 0..=4) -> visual populations;
    // - the threat-alarm channel (CH 5) -> an `ol_sensory` slice (odor
    //   stream; no dedicated nociceptor population exists);
    // - ground contact (CH 6) -> `vnc_sensory` (leg mechanoreceptors);
    // - food channels (CH 7..=9) -> further disjoint `ol_sensory` slices
    //   (the food cue is carried as an odor stream — NOT as gustation, and
    //   never as mechanosensation);
    // - internal coordination signal (CH 10) -> ascending relay cells.
    // Fallbacks ride the umbrella `sensory` classes only to satisfy the
    // runtime's binding-length contract, always AFTER the semantic pools.
    let pool = |pred: &dyn Fn(&str) -> bool| -> Vec<usize> {
        (0..n)
            .filter(|&i| !taken[i] && class_of(i).map_or(false, |r| pred(r)))
            .collect()
    };
    let visual = pool(&|r| {
        r.contains("visual") || r.contains("ol_sensory") || r.contains("ol_intrinsic")
    });
    let odor = pool(&|r| r.contains("ol_sensory"));
    let central_sensory =
        pool(&|r| r.contains("sensory") && !r.contains("vnc") && !r.contains("descending"));
    let vnc_sensory = pool(&|r| r.contains("vnc_sensory"));
    let ascending = pool(&|r| r.contains("ascending") && r.contains("sensory"));
    let sensory_like = pool(&|r| r.contains("sensory"));
    let ascending_any = pool(&|r| r.contains("ascending"));

    let mut binding: Vec<usize> = Vec::new();
    let cells_per_channel = cells_per_channel.max(1);
    // channel order follows `crate::sensory::CH_*`:
    // 0 optic speed, 1..=2 optic dirs, 3..=4 heading(light) -> visual pool
    for _ in 0..5 {
        assign_slots(&mut taken, &mut binding, cells_per_channel, &[&visual], &[&sensory_like, &ascending_any])?;
    }
    // 5 threat alarm -> ol_sensory slice (odor stream), central sensory fill
    assign_slots(&mut taken, &mut binding, cells_per_channel, &[&odor, &central_sensory], &[&sensory_like])?;
    // 6 ground contact -> vnc sensory (mechanosensation), central fill
    assign_slots(&mut taken, &mut binding, cells_per_channel, &[&vnc_sensory, &central_sensory], &[&sensory_like])?;
    // 7..=9 food proximity/direction -> further ol_sensory slices (odor)
    for _ in 0..3 {
        assign_slots(&mut taken, &mut binding, cells_per_channel, &[&odor, &central_sensory], &[&sensory_like])?;
    }
    // 10 queen coordination signal -> ascending sensory
    assign_slots(&mut taken, &mut binding, cells_per_channel, &[&ascending, &ascending_any], &[&sensory_like, &ascending_any])?;
    // safety: exact-length filler (the runtime requires len == inputs_needed)
    let target = cells_per_channel * CHANNEL_COUNT;
    while binding.len() < target {
        assign_slots(&mut taken, &mut binding, 1, &[&sensory_like], &[&ascending_any])?;
    }

    Some((binding, classes))
}

/// First `slots` untaken pool neurons (primary pools first, then the shared
/// filler pools), marking them taken. `None` when the pools run dry.
fn assign_slots(
    taken: &mut [bool],
    binding: &mut Vec<usize>,
    slots: usize,
    pools: &[&[usize]],
    filler: &[&[usize]],
) -> Option<()> {
    for _ in 0..slots {
        let mut got: Option<usize> = None;
        'find: for src in pools.iter().chain(filler.iter()) {
            for &c in *src {
                if !taken[c] {
                    got = Some(c);
                    break 'find;
                }
            }
        }
        let c = got?;
        taken[c] = true;
        binding.push(c);
    }
    Some(())
}

/// FUNCTIONAL channel assembly from annotated classes for the region-grounded
/// queen run mode (documented, all-offline, deterministic).
///
/// ANATOMICAL POPULATIONS and MOTOR CHANNELS are distinct here, and the
/// decoder installs exactly as many channels as the data defends (K in
/// 1..=`OUTPUT_CELL_COUNT`); body DOFs beyond K decode neutral.
///
/// Pipeline:
/// 1. probe: alternate three scene vectors over the just-installed afferent
///    binding (FOOD scene, HAZARD scene, OBSTACLE-ONLY scene — deliberately
///    different from the later validation scenarios); collect per-member
///    per-phase mean rates over 12 phases.
/// 2. per-member stats: scene means + WITHIN-scene noise sigma (correlated
///    members mean honest noise is the pooled sigma average, NOT the
///    1/sqrt(K)-reduced value).
/// 3. channel assembly, per class, from MEASURED RESPONSE PROFILES and REAL
///    CONNECTIVITY:
///    - `descending_neuron`: greedy correlation clustering of the three-scene
///      response vectors; an ENSEMBLE qualifies when it holds >= 4 neurons
///      with at least one internal synapse and a pooled scene response above
///      the pooled within-scene noise floor.
///    - `cb_motor`: evaluated as a separate motor-related population — one
///      channel when its pooled response clears the same noise floor.
///    - `vnc_motor`: a SINGLE-CELL signal (pool of that one neuron's own
///      calibrated rate, not a population average).
///    - `efferent_ascending`: deliberately NOT an output population (afferent
///      transport); reported in diagnostics only.
///    Channel-to-DOF assignment is STRUCTURAL — the dataset provides no
///    per-neuron semantic for specific fly behaviours; this limitation is
///    documented rather than inventing a biological mapping. The
///    CALIBRATION (measured baseline/normalize ranges) makes each channel an
///    honest relative readout of its own scene response.
/// 4. install via `set_output_pools` + `set_motor_calibration` (K channels).
///    When NO channel is defensible, the SNR-ranked SINGLE-CELL decoder
///    remains active so the pool decoder can be compared against it.
fn auto_tune_output_binding(rt: &mut FlyBrainRuntime, classes: &[MotorClassCandidate]) {
    const PHASE_TICKS: usize = 150;
    const PHASES: usize = 12; // 4 rounds x 3 calibration scenes
    let members: Vec<usize> = classes.iter().flat_map(|c| c.cells.iter().copied()).collect();
    let n = rt.connectome().n();
    if members.is_empty() || members.iter().any(|&m| m >= n) {
        return;
    }
    use crate::sensory::{CH_FOOD_COS, CH_FOOD_PROX, CH_GROUND, CH_LIGHT_COS, CH_OBSTACLE, CH_QUEEN};
    let budget = crate::runtime::TickBudget::default();
    let n_mem = members.len();
    let mut slot_of = vec![usize::MAX; n];
    for (k, &c) in members.iter().enumerate() {
        slot_of[c] = k;
    }
    // per-member rates across 3 calibration scenes (phase p: scene p % 3)
    let mut phase_rates = vec![0.0f32; PHASES * n_mem];
    for phase in 0..PHASES {
        let mut chans = [0.0f32; CHANNEL_COUNT];
        match phase % 3 {
            0 => {
                // FOOD scene
                chans[CH_FOOD_COS] = 1.0;
                chans[CH_FOOD_PROX] = 0.9;
                chans[CH_GROUND] = 0.1;
                chans[CH_QUEEN] = 0.2;
            }
            1 => {
                // HAZARD scene (threat present, heading cues active)
                chans[CH_OBSTACLE] = 0.8;
                chans[CH_LIGHT_COS] = 0.7;
                chans[CH_GROUND] = 0.8;
            }
            _ => {
                // OBSTACLE-ONLY scene (threat, no heading cues)
                chans[CH_OBSTACLE] = 0.9;
            }
        }
        for _ in 0..PHASE_TICKS {
            let _ = rt.tick_channels(&chans, &budget);
            for &s in rt.spike_ids_last() {
                let k = slot_of.get(s as usize).copied().unwrap_or(usize::MAX);
                if k < n_mem {
                    phase_rates[phase * n_mem + k] += 1.0;
                }
            }
        }
        for k in 0..n_mem {
            let tag = phase * n_mem + k;
            phase_rates[tag] /= PHASE_TICKS as f32 * 0.001;
        }
    }
    // per-member stats: scene means, within-scene sigma, scene spread, SNR
    let mut m_mean = vec![[0.0f32; 3]; n_mem];
    let mut m_sigma = vec![0.0f32; n_mem];
    let mut m_spread = vec![0.0f32; n_mem];
    for k in 0..n_mem {
        let mut sums = [0.0f64; 3];
        let mut mo = [0.0f64; 3];
        for phase in 0..PHASES {
            let hz = phase_rates[phase * n_mem + k] as f64;
            let tag = phase % 3;
            sums[tag] += hz;
            mo[tag] += hz * hz;
        }
        let per = (PHASES / 3) as f64;
        let mut means = [0.0f32; 3];
        let mut var = 0.0f64;
        for t in 0..3 {
            means[t] = (sums[t] / per) as f32;
            let m2 = means[t] as f64;
            var += (mo[t] / per - m2 * m2).max(0.0);
        }
        m_mean[k] = means;
        m_sigma[k] = ((var / 3.0).max(1.0) as f32).sqrt().max(1.0);
        m_spread[k] = (means[0] - means[1])
            .abs()
            .max((means[1] - means[2]).abs())
            .max((means[0] - means[2]).abs());
    }
    let m_snr: Vec<f32> = (0..n_mem).map(|k| m_spread[k] / m_sigma[k]).collect();

    struct Chan {
        label: String,
        members: Vec<usize>,
        base: f32,
        spread: f32,
        sigma: f32,
    }
    // MEASURED pooled stats of a member set (member indices into the flat
    // member list): the pool's actual per-phase mean-rate series is scored
    // directly, so member correlations are fully accounted for — this is not
    // the conservative fully-correlated noise model.
    let pooled = |ks: &[usize]| -> ([f32; 3], f32, f32, f32) {
        let mut sums = [0.0f64; 3];
        let mut mo = [0.0f64; 3];
        for phase in 0..PHASES {
            let m = ks
                .iter()
                .map(|&k| phase_rates[phase * n_mem + k])
                .sum::<f32>()
                / ks.len() as f32;
            let tag = phase % 3;
            sums[tag] += m as f64;
            mo[tag] += (m as f64) * (m as f64);
        }
        let per = (PHASES / 3) as f64;
        let mut mean = [0.0f32; 3];
        let mut var = 0.0f64;
        for t in 0..3 {
            mean[t] = (sums[t] / per) as f32;
            let m2 = sums[t] / per;
            var += (mo[t] / per - m2 * m2).max(0.0);
        }
        let base = (mean[0] + mean[1] + mean[2]) / 3.0;
        let spread =
            (mean[0] - mean[1]).abs().max((mean[1] - mean[2]).abs()).max((mean[0] - mean[2]).abs());
        (mean, base, spread, ((var / 3.0).max(1.0) as f32).sqrt())
    };
    // a pooled response above this is a real scene response, not drift
    let noise_floor = |sigma: f32| -> f32 { (2.5 * sigma).max(3.0) };

    let conn = rt.connectome().clone();
    let edge_between = |a: usize, b: usize| -> bool {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        for p in conn.row_offsets[lo] as usize..conn.row_offsets[lo + 1] as usize {
            if conn.col_idx[p] as usize == hi {
                return true;
            }
        }
        false
    };

    let mut chans: Vec<Chan> = Vec::new();

    // --- (a) descending ensembles: response-profile clustering -------------
    if let Some(desc) = classes.iter().find(|c| c.label.contains("descending")) {
        let idx: Vec<usize> = desc.cells.iter().map(|c| slot_of[*c]).collect();
        // candidates: living members with some scene modulation; ordered by
        // measured SNR. The hard gate is applied later on MEASURED pooled
        // stats, where pool averaging has already suppressed independent
        // single-cell noise.
        let mut cand: Vec<usize> = idx
            .into_iter()
            .filter(|&k| {
                m_mean[k].iter().sum::<f32>() / 3.0 > 3.0 && m_spread[k] >= 1.0
            })
            .collect();
        cand.sort_unstable_by(|&a, &b| m_snr[b].partial_cmp(&m_snr[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        eprintln!(
            "queen channels: descending candidates {}/{} (living+modulated)",
            cand.len(),
            desc.cells.len()
        );
        // greedy clustering: join the first ensemble whose centroid response
        // vector is collinear with this member's scene response vector
        let cos = |u: &[f32; 3], v: &[f32; 3]| -> f32 {
            let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if nu <= 0.0 || nv <= 0.0 { 0.0 } else { dot / (nu * nv) }
        };
        let mut clusters: Vec<(Vec<usize>, [f32; 3])> = Vec::new();
        for &k in &cand {
            let mut joined = false;
            for cluster in clusters.iter_mut() {
                if cos(&m_mean[k], &cluster.1) >= 0.70 {
                    cluster.0.push(k);
                    let len = cluster.0.len() as f32;
                    for t in 0..3 {
                        cluster.1[t] = (cluster.1[t] * (len - 1.0) + m_mean[k][t]) / len;
                    }
                    joined = true;
                    break;
                }
            }
            if !joined {
                clusters.push((vec![k], m_mean[k]));
            }
        }
        // an ensemble is defensible when it is a real POPULATION: >= 4
        // neurons, internally connected, pooled response above its noise floor
        let mut desc_chans: Vec<Chan> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(ci, (mut ks, _))| {
                let size = ks.len();
                let (_mean, base, spread, sigma) = pooled(&ks);
                let connected = ks
                    .iter()
                    .enumerate()
                    .any(|(x, &a)| ks[x + 1..].iter().any(|&b| edge_between(members[a], members[b])));
                let gate = base > 3.0 && spread >= noise_floor(sigma);
                if size >= 4 {
                    eprintln!(
                        "queen channels: descending cluster#{ci} n={size} base={base:.1} spread={spread:.2} sigma={sigma:.2} floor={:.2} connected={connected} gate={gate}",
                        noise_floor(sigma)
                    );
                }
                if size < 4 || !connected || !gate {
                    return None;
                }
                ks.sort_unstable();
                let ks: Vec<usize> = ks;
                let label = desc.label.clone();
                Some(Chan { label, members: ks, base, spread, sigma })
            })
            .collect();
        desc_chans.sort_unstable_by(|a, b| {
            b.spread.partial_cmp(&a.spread).unwrap_or(std::cmp::Ordering::Equal).then(a.label.cmp(&b.label))
        });
        // leave DOF room for the other classes' defensible channels
        chans.extend(desc_chans.into_iter().take(4));
    }

    // --- (b) cb_motor: a separate motor-related population -----------------
    if let Some(cb) = classes.iter().find(|c| c.label.contains("cb_motor")) {
        let ks: Vec<usize> = cb.cells.iter().map(|c| slot_of[*c]).collect();
        if !ks.is_empty() {
            let (_, base, spread, sigma) = pooled(&ks);
            eprintln!(
                "queen channels: cb_motor n={} base={base:.1} spread={spread:.2} sigma={sigma:.2} floor={:.2}",
                ks.len(),
                noise_floor(sigma)
            );
            if base > 3.0 && spread >= noise_floor(sigma) {
                chans.push(Chan {
                    label: cb.label.clone(),
                    members: ks,
                    base,
                    spread,
                    sigma,
                });
            }
        }
    }

    // --- (c) vnc_motor: single-cell signal (not a population average) ------
    if let Some(vnc) = classes.iter().find(|c| c.label.contains("vnc_motor")) {
        for &cell in &vnc.cells {
            let k = slot_of[cell];
            if k >= n_mem {
                continue;
            }
            let base = m_mean[k].iter().sum::<f32>() / 3.0;
            if base > 3.0 && m_spread[k] >= noise_floor(m_sigma[k]) {
                chans.push(Chan {
                    label: format!("{}#cell{cell}", vnc.label),
                    members: vec![k],
                    base,
                    spread: m_spread[k],
                    sigma: m_sigma[k],
                });
            }
        }
    }

    // NOTE: efferent_ascending is intentionally NOT installed as a motor
    // output (afferent transport). Its measured response is reported below
    // for the record.
    if let Some(ef) = classes.iter().find(|c| c.label.contains("efferent")) {
        let ks: Vec<usize> = ef.cells.iter().map(|c| slot_of[*c]).collect();
        let (_, base, spread, sigma) = pooled(&ks);
        eprintln!(
            "queen channels: efferent class {:<22} NOT a motor output (report-only: base={base:.1}Hz spread={spread:.2}Hz sigma={sigma:.2}Hz)",
            ef.label
        );
    }

    chans.truncate(crate::motor::OUTPUT_CELL_COUNT);

    if chans.is_empty() {
        // SINGLE-CELL decoder (documented comparison baseline): the
        // best-SNR members across all classes, calibrated per cell. Also
        // the state whenever no anatomical pool clears the noise floor.
        let mut ord: Vec<usize> = (0..n_mem).collect();
        ord.sort_by(|&x, &y| {
            m_snr[y].partial_cmp(&m_snr[x]).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y))
        });
        ord.truncate(crate::motor::OUTPUT_CELL_COUNT);
        if ord.len() < crate::motor::OUTPUT_CELL_COUNT {
            return;
        }
        let mut cells = [0usize; crate::motor::OUTPUT_CELL_COUNT];
        let mut bl = vec![0f32; crate::motor::OUTPUT_CELL_COUNT];
        let mut rg = vec![0f32; crate::motor::OUTPUT_CELL_COUNT];
        for (slot, &k) in ord.iter().enumerate() {
            cells[slot] = members[k];
            bl[slot] = m_mean[k].iter().sum::<f32>() / 3.0;
            rg[slot] = m_spread[k].max(2.5 * m_sigma[k]).max(2.0)
                .min(crate::motor::OUTPUT_CALIB_RANGE_MAX_HZ);
        }
        eprintln!(
            "queen decoder: SINGLE-CELL (comparison mode; no defensible population channel) cells={cells:?} SNRs={:?}",
            ord.iter().map(|&k| m_snr[k]).collect::<Vec<_>>()
        );
        if rt.set_output_binding(&cells) {
            rt.set_motor_calibration(Some(&bl), Some(&rg));
        }
        return;
    }

    // install the measured calibrated functional channels
    eprintln!("queen decoder: POPULATION ({} functional channels):", chans.len());
    let refs: Vec<&[usize]> = chans.iter().map(|c| c.members.as_slice()).collect();
    for (slot, p) in chans.iter().enumerate() {
        eprintln!(
            "  [{slot}] {:<24} n={:<3} base={:>7.1}Hz spread={:>6.2}Hz sigma={:>5.2}Hz",
            p.label,
            p.members.len(),
            p.base,
            p.spread,
            p.sigma
        );
    }
    if rt.set_output_pools(&refs) {
        let calib: Vec<f32> = chans
            .iter()
            .map(|c| {
                c.spread.max(2.5 * c.sigma).max(2.0).min(crate::motor::OUTPUT_CALIB_RANGE_MAX_HZ)
            })
            .collect();
        let base: Vec<f32> = chans.iter().map(|c| c.base).collect();
        if !rt.set_motor_calibration(Some(&base), Some(&calib)) {
            // length mismatch cannot happen (base/calib both K) — defensive
            rt.set_motor_calibration(None, None);
        }
    }
}
