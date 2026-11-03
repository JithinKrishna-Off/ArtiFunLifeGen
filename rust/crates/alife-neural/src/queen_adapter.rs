//! `QueenBrainAdapter`: queen-brain implementation ON TOP of a
//! [`FlyBrainRuntime`] for the alife-core coordinator.
//!
//! It is the ONLY piece of glue between the coordinator's queen contract and
//! this crate's spiking runtime; the runtime itself never knows about
//! coordinator types.
//!
//! Mapping (documented model decisions, all channels in use — no zeroed
//! islands; see [`QueenSensoryMapping`] and [`propose_region_binding`]):
//! - `food_dir` / `food_dist` -> the food cos/sin/prox channels, bound
//!   (when the sugar population is available) inside the REAL sugar
//!   gustatory cells, else the structurally-healthiest `cb_sensory` slices
//!   — the food cue is carried as a CHEMOSENSORY stream, never invented
//!   anatomy.
//! - `heading` -> (cos, sin) pair on the temperature-proxy (light-alias)
//!   directional channels.
//! - `speed` -> optic-speed channel (rate); `energy_frac` -> ground contact
//!   (`vnc_sensory` mechanoreceptors); `hazard` -> obstacle-proximity
//!   channel (`sensory_ascending`/`ascending_neuron` relays).
//! - `edge_dist` / `density` -> unmapped (`None` channels).
//! - `signal_sample` -> queen-signal channel (ascending relay cells).
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
        Self::new_for_subnetwork_with_sugar(connectome, seed, cells_per_channel, None)
    }

    /// [`new_for_subnetwork_with_cells`] plus the OPTIONAL real sugar sensory
    /// population (original MaleCNS body ids). When provided, the food
    /// channels (7..=9) bind inside the ACTUAL sugar-gustatory cells (mapped
    /// through the preserved `body_id` on the subnetwork); when absent, they
    /// fall back to the structurally-defensible `cb_sensory` class — never a
    /// fabricated anatomy.
    ///
    /// Calibrated operating point (measured against the 166k MaleCNS queen
    /// subnetwork; all runtime CALIBRATION knobs that persist across
    /// snapshots — the LIF engine itself is untouched):
    /// - afferent rate window 3..63 Hz with 30 (mV-equivalent) injected
    ///   pulses, so [0,1] channels clear the network's tonic self-run;
    /// - `weight_scale` 0.45 + 20 event-current clamp: raw MaleCNS weights
    ///   are signed synaptic COUNTS; at 1.0 they push the LIF into an
    ///   inhibitory runaway (measured: mean membrane -> -300 mV, sustained
    ///   >1000 spikes/tick), while < 0.15 leaves it inert. 0.45 sits in the
    ///   measured stable band (~640 spikes/tick warm steady state, bounded
    ///   membrane envelope) where afferent scene contrast survives to decode
    ///   populations (measured per-cell reproducible deltas 45..122 Hz at a
    ///   ~1 Hz run-to-run sigma).
    /// Fixtures/tests that do not use this constructor keep runtime defaults
    /// (weight_scale 1.0 / clamp 40).
    pub fn new_for_subnetwork_with_sugar(
        connectome: Arc<Connectome>,
        seed: u32,
        cells_per_channel: usize,
        sugar_bodies: Option<&[u64]>,
    ) -> Result<Self, RuntimeBuildError> {
        let mut rt = FlyBrainRuntime::new(connectome, seed)?;
        rt.sensory.gain_hz = 60.0;
        rt.sensory.baseline_hz = 3.0;
        rt.sensory.pulse_current = 30.0;
        rt.weight_scale = 0.45;
        rt.event_current_clamp = 20.0;
        rt.sensory.channel_cells = cells_per_channel.max(1);
        // re-derive the encoders vector for the new slot count
        rt.encoders
            .resize(crate::sensory::inputs_needed(&rt.sensory), crate::sensory::Encoder::Rate);
        if let Some((binding, motor_candidates)) =
            propose_region_binding(rt.connectome(), rt.sensory.channel_cells, sugar_bodies)
        {
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

    /// [`new_from_connectome`] with the sugar population available (see
    /// [`QueenBrainAdapter::new_for_subnetwork_with_sugar`]).
    pub fn new_from_connectome_with_sugar(
        connectome: Arc<Connectome>,
        seed: u32,
        sugar_bodies: Option<&[u64]>,
    ) -> Result<Self, RuntimeBuildError> {
        Self::new_for_subnetwork_with_sugar(connectome, seed, 2, sugar_bodies)
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
/// - AFFERENT channels are bound onto real sensory subpopulations by their
///   superclass label, and the members of each pool are ordered by their
///   CONNECTIVITY HEALTH: positive outgoing-weight sum (the injected pulse
///   must be able to leave the cell through a real synapse). Measured
///   findings this encoding follows (audit of the 166k queen subnetwork):
///   the default binding's first-N slices of `ol_sensory` were dead afferents
///   (zero positive out-edges across the whole n=3335 class), so a food or
///   threat cue injected there never propagated.
///   Channel assignments (nothing fabricated; census in `label_census`):
///   visual/heading channels (CH 0..=4) -> `visual_projection`, then
///   `visual_centrifugal`/`ol_intrinsic` cells; threat alarm (CH 5) and the
///   queen-signal channel (CH 10) -> `sensory_ascending`/`ascending_neuron`
///   relays; ground contact (CH 6) -> `vnc_sensory` mechanoreceptors; food
///   channels (CH 7..=9) -> the REAL sugar gustatory cells (when the caller
///   supplies the sugar population's original MaleCNS body ids, mapped into
///   the subnetwork via the preserved `body_id`) else `cb_sensory` slices
///   (central sensory class — the food cue as a chemosensory stream, clearly
///   NOT gustation when the population file is absent);
/// - Motor output decodes POOL MEAN rates over FUNCTIONAL channels mined from
///   the annotated motor classes (see [`MotorClassCandidate`] and
///   [`auto_tune_output_binding`]).
///
/// Every cell is a real, labeled neuron — nothing is fabricated. Deterministic.
/// Returns `None` when the connectome lacks labels (or is too small) so the
/// caller keeps the contiguous default binding unchanged.
pub fn propose_region_binding(
    conn: &Connectome,
    cells_per_channel: usize,
    sugar_bodies: Option<&[u64]>,
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
    // Connectivity health: a cell can only CARRY an injected pulse forward
    // through positive outgoing synapses, so pool members are ordered by
    // their positive out-weight sum (ties: ascending index). Deterministic.
    let pos_out = |i: usize| -> f32 {
        let (rs, re) = (conn.row_offsets[i] as usize, conn.row_offsets[i + 1] as usize);
        (rs..re).filter(|&p| conn.weights[p] > 0.0).map(|p| conn.weights[p]).sum()
    };
    let health_sorted = |mut v: Vec<usize>| -> Vec<usize> {
        v.sort_unstable_by(|&a, &b| {
            pos_out(b).partial_cmp(&pos_out(a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        v
    };
    let pool = |pred: &dyn Fn(&str) -> bool| -> Vec<usize> {
        health_sorted(
            (0..n)
                .filter(|&i| !taken[i] && class_of(i).map_or(false, |r| pred(r)))
                .collect(),
        )
    };
    let visual = pool(&|r| r == "visual_projection");
    let visual_secondary = pool(&|r| r == "visual_centrifugal" || r == "ol_intrinsic");
    let ascending = pool(&|r| r == "sensory_ascending" || r == "ascending_neuron");
    let vnc_sensory = pool(&|r| r == "vnc_sensory");
    let central_sensory = pool(&|r| r == "cb_sensory");
    let sensory_like = pool(&|r| r.contains("sensory"));
    let ascending_any = pool(&|r| r.contains("ascending"));

    // The REAL sugar gustatory population when its original body ids are
    // supplied (package `populations/sugar.i32` + `neurons/body_id.i64`),
    // mapped into this subnetwork through the preserved body ids and then
    // health-ordered itself.
    let sugar: Vec<usize> = sugar_bodies
        .map(|bodies| {
            let by_body: std::collections::HashMap<u64, usize> =
                (0..n).map(|i| (conn.node_meta[i].body_id, i)).collect();
            health_sorted(bodies.iter().filter_map(|&b| by_body.get(&b).copied()).collect())
        })
        .unwrap_or_default();

    let mut binding: Vec<usize> = Vec::new();
    let cells_per_channel = cells_per_channel.max(1);
    // channel order follows `crate::sensory::CH_*`. Primary pools first, then
    // the same-class fallbacks, then umbrella fillers — semantics degrade
    // honestly (same-cell-class relays), never onto fabricated anatomy:
    let visual_pools = [visual.as_slice(), visual_secondary.as_slice(), sensory_like.as_slice()];
    let heading_pools =
        [visual_secondary.as_slice(), visual.as_slice(), sensory_like.as_slice()];
    let relay_pools = [ascending.as_slice(), ascending_any.as_slice(), sensory_like.as_slice()];
    let ground_pools = [vnc_sensory.as_slice(), sensory_like.as_slice()];
    let food_pools = [sugar.as_slice(), central_sensory.as_slice(), sensory_like.as_slice()];
    let channel_pools: [&[&[usize]]; CHANNEL_COUNT] = [
        &visual_pools,    // 0 optic speed
        &visual_pools,    // 1 optic dir cos
        &visual_pools,    // 2 optic dir sin
        &heading_pools,   // 3 heading cos (light-alias)
        &heading_pools,   // 4 heading sin
        &relay_pools,     // 5 threat alarm
        &ground_pools,    // 6 ground contact
        &food_pools,      // 7 food proximity
        &food_pools,      // 8 food dir cos
        &food_pools,      // 9 food dir sin
        &relay_pools,     // 10 queen coordination signal
    ];
    for pools in channel_pools {
        assign_slots(&mut taken, &mut binding, cells_per_channel, pools)?;
    }
    // safety: exact-length filler (the runtime requires len == inputs_needed)
    let target = cells_per_channel * CHANNEL_COUNT;
    while binding.len() < target {
        assign_slots(
            &mut taken,
            &mut binding,
            1,
            &[&[sensory_like.as_slice(), ascending_any.as_slice()].concat()],
        )?;
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
) -> Option<()> {
    for _ in 0..slots {
        let mut got: Option<usize> = None;
        'find: for src in pools {
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

/// Runtime diagnostics switch: the queen path is SILENT by default (no
/// per-build/per-frame stderr); set `ALIFE_QUEEN_BRAIN_DIAG=1` to observe the
/// calibration log. Documented policy, enforced at every queen print site.
fn queen_diag() -> bool {
    std::env::var("ALIFE_QUEEN_BRAIN_DIAG").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Probe schedule per scene: warm-up (state settles on the scene), then a
/// count window split into two halves for the within-scene sigma estimate.
const AUTO_TUNE_WARM_TICKS: usize = 100;
const AUTO_TUNE_HALF_TICKS: usize = 100;
const AUTO_TUNE_COUNT_TICKS: usize = AUTO_TUNE_HALF_TICKS * 2;

/// FUNCTIONAL channel assembly from annotated classes for the region-grounded
/// queen run mode (documented, all-offline, deterministic).
///
/// ANATOMICAL POPULATIONS and MOTOR CHANNELS are distinct: the decoder
/// installs exactly as many channels as the measured data defends (K in
/// 1..=`OUTPUT_CELL_COUNT`), and body DOFs beyond K decode neutral. The
/// assembly is a PREFIX: DOFs pick their measured pools in fixed
/// decode-slot order, and the first DOF without a defensible pool ends the
/// decoder (a hole mid-chain Would mis-map every later slot onto the wrong
/// DOF).
///
/// Pipeline (v2, replaced the 3-scene clustering after the audit):
/// 1. probe every DECODE slot's focal sensory channel directly — one scene
///    per sign per slot, plus one rest scene:
///    DirX <- CH_FOOD_COS(±1), DirY <- CH_FOOD_SIN(±1), Intensity <-
///    CH_OBSTACLE(+1), Turn <- CH_LIGHT_SIN(±1), FeedSeek <-
///    CH_FOOD_PROX(+1), Emit <- CH_QUEEN(+1). Every driven scene starts from
///    a snapshot-restored common state, so a scene's measured rate is its
///    MARGINAL effect, not a carryover.
///    Measured justification (snr/rep probes on the 166k subnetwork): the
///    network's per-cell scene responses are deterministic (run-to-run sigma
///    ~1 Hz) with reproducible deltas up to ~122 Hz on descending/motor/relay
///    cells — real channel-locked contrast exists to be calibrated against.
/// 2. per-DOF pool assembly from MEASURED member responses: members whose
///    drift direction matches the pool's sign (signed slots) / a positive
///    response (gate slots), ordered by |delta| descending, capped per pool;
///    the gate is the pooled scene contrast clearing the pooled within-scene
///    noise floor (2.5 × sigma, min 3 Hz). This is the same statutory noise
///    floor criterion the previous version used on 3-scene clusters, applied
///    where the semantics now point: at a specific channel.
/// 3. calibration per installed channel: baseline = MEASURED pooled rest
///    rate; range = MEASURED pooled scene contrast (max over ± scenes),
///    clamped to `[2.0, OUTPUT_CALIB_RANGE_MAX_HZ]`; the decoder's calibrated
///    `(ema - base)/range` readout then reproduces ±1 exactly at the probe's
///    drive extremes.
/// 4. install via `set_output_pools` + `set_motor_calibration` (K channels).
///    When NO DOF is defensible, the SNR-ranked SINGLE-CELL decoder remains
///    active as the documented comparison baseline.
fn auto_tune_output_binding(rt: &mut FlyBrainRuntime, classes: &[MotorClassCandidate]) {
    let members: Vec<usize> = classes.iter().flat_map(|c| c.cells.iter().copied()).collect();
    let n = rt.connectome().n();
    if members.is_empty() || members.iter().any(|&m| m >= n) {
        return;
    }
    use crate::sensory::{CH_FOOD_COS, CH_FOOD_PROX, CH_FOOD_SIN, CH_LIGHT_SIN, CH_OBSTACLE, CH_QUEEN};
    // decode slot <- focal channel (documented contract, see the fn doc)
    const FOCAL: [usize; crate::motor::OUTPUT_CELL_COUNT] =
        [CH_FOOD_COS, CH_FOOD_SIN, CH_OBSTACLE, CH_LIGHT_SIN, CH_FOOD_PROX, CH_QUEEN];
    const SIGNED_DOF: [bool; crate::motor::OUTPUT_CELL_COUNT] = [true, true, false, true, false, false];

    let n_mem = members.len();
    let mut slot_of = vec![usize::MAX; n];
    for (k, &c) in members.iter().enumerate() {
        slot_of[c] = k;
    }

    // scene vectors: [rest(0)] + per-DOF ± drive
    let scene_chan = |scene: usize| -> [f32; CHANNEL_COUNT] {
        let mut ch = [0.0f32; CHANNEL_COUNT];
        if scene > 0 {
            // scene s drives dof ( (s-1) ... ): dof emits 1 (+) or 2 (−) scenes
            let mut s = scene - 1;
            for dof in 0..crate::motor::OUTPUT_CELL_COUNT {
                let drive = if SIGNED_DOF[dof] { 2 } else { 1 };
                if s < drive {
                    ch[FOCAL[dof]] = if s == 0 { 1.0 } else { -1.0 };
                    break;
                }
                s -= drive;
            }
        }
        ch
    };
    let scene_count =
        1 + (0..crate::motor::OUTPUT_CELL_COUNT).map(|d| if SIGNED_DOF[d] { 2 } else { 1 }).sum::<usize>();

    // per-scene member mean rates (Hz) and within-scene sigmas; every
    // non-rest scene measures on the restored common state.
    let budget = crate::runtime::TickBudget::default();
    let home = rt.snapshot();
    let mut rates: Vec<Vec<f32>> = Vec::with_capacity(scene_count);
    let mut sigmas: Vec<Vec<f32>> = Vec::with_capacity(scene_count);
    for scene in 0..scene_count {
        if scene > 0 {
            rt.restore_snapshot(home.clone());
        }
        let chans = scene_chan(scene);
        let mut acc1 = vec![0.0f32; n_mem];
        let mut acc2 = vec![0.0f32; n_mem];
        for t in 0..AUTO_TUNE_WARM_TICKS + AUTO_TUNE_COUNT_TICKS {
            let _ = rt.tick_channels(&chans, &budget);
            if t < AUTO_TUNE_WARM_TICKS {
                continue;
            }
            let half = t - AUTO_TUNE_WARM_TICKS < AUTO_TUNE_HALF_TICKS;
            for &s in rt.spike_ids_last() {
                let k = slot_of.get(s as usize).copied().unwrap_or(usize::MAX);
                if k < n_mem {
                    if half {
                        acc1[k] += 1.0;
                    } else {
                        acc2[k] += 1.0;
                    }
                }
            }
        }
        let per_half = AUTO_TUNE_HALF_TICKS as f32 * 0.001;
        rates.push(
            acc1.iter().zip(&acc2).map(|(&a, &b)| (a + b) / (2.0 * per_half)).collect::<Vec<f32>>(),
        );
        sigmas.push(
            acc1.iter().zip(&acc2).map(|(&a, &b)| {
                // two-sample sigma estimate, honest floor at 1 Hz
                ((a - b) / (std::f32::consts::SQRT_2 * per_half)).abs().max(1.0)
            }).collect::<Vec<f32>>(),
        );
    }
    rt.restore_snapshot(home.clone());

    // pooled statistics of a member set: pooled mean rate per scene +
    // CONSERVATIVE pooled sigma (fully-correlated members — the mean of the
    // member sigmas, NOT the sqrt(K)-reduced independent estimate).
    let pooled_rate = |ks: &[usize], scene: usize| -> f32 {
        ks.iter().map(|&k| rates[scene][k]).sum::<f32>() / ks.len() as f32
    };
    let pooled_sigma = |ks: &[usize], scene: usize| -> f32 {
        ks.iter().map(|&k| sigmas[scene][k]).sum::<f32>() / ks.len() as f32
    };

    struct Chan {
        members: Vec<usize>,
        base: f32,
        range: f32,
    }
    const POOL_CAP: usize = 12;
    let mut chans: Vec<Chan> = Vec::new();
    for dof in 0..crate::motor::OUTPUT_CELL_COUNT {
        // scenes for this dof: from the scene layout (rest, then dof order)
        let mut s = 1; // skip rest
        for d in 0..dof {
            s += if SIGNED_DOF[d] { 2 } else { 1 };
        }
        let plus = s;
        let minus = if SIGNED_DOF[dof] { Some(s + 1) } else { None };
        let focal = FOCAL[dof];
        // per-member contrast vs rest
        let mut cand: Vec<(f32 /*delta*/, usize /*member slot, sorted later*/)> = (0..n_mem)
            .map(|k| (rates[plus][k] - rates[0][k], k))
            .collect();
        if cand.is_empty() {
            break;
        }
        // pool sign: does the class respond up or down to the drive?
        let all_deltas_mean = cand.iter().map(|(d, _)| *d).sum::<f32>() / n_mem as f32;
        // candidates: members in the SIGN-CONSISTENT direction (signed slots
        // take the pooled majority sign; gate slots (Int/prox/emit) take a
        // positive response by definition of their decode contract).
        let pool_sign = if SIGNED_DOF[dof] {
            if all_deltas_mean >= 0.0 { 1.0 } else { -1.0 }
        } else {
            1.0
        };
        cand.retain(|(d, _)| *d * pool_sign > 0.0);
        cand.sort_unstable_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        cand.truncate(POOL_CAP);
        if cand.len() < 3 {
            break;
        }
        let ks: Vec<usize> = cand.iter().map(|(_, k)| *k).collect();
        let dplus = pooled_rate(&ks, plus) - pooled_rate(&ks, 0);
        let dminus = minus.map(|m| pooled_rate(&ks, m) - pooled_rate(&ks, 0));
        let sigma =
            (pooled_sigma(&ks, plus).powi(2) + pooled_sigma(&ks, 0).powi(2)).sqrt().max(1.0);
        let contrast = dplus.abs().max(dminus.map(|d| d.abs()).unwrap_or(0.0));
        let floor = (2.5 * sigma).max(3.0);
        let sign_ok = if SIGNED_DOF[dof] {
            // signed slots need BOTH directions present (or at least the +
            // scene contrast in the pool's direction)
            dplus * pool_sign > floor
                && dminus.map_or(true, |d| d * pool_sign < 0.0)
        } else {
            dplus > floor
        };
        let ok = sign_ok && dplus * pool_sign > floor;
        if queen_diag() {
            eprintln!(
                "queen channels: dof {dof} ch{focal} n={} dplus={dplus:.1} dminus={} sigma={sigma:.1} floor={floor:.1} gate={ok}",
                ks.len(),
                dminus.map(|d| format!("{d:.1}")).unwrap_or_else(|| "n/a".into()),
            );
        }
        if !ok {
            break;
        }
        let base = pooled_rate(&ks, 0);
        let range = contrast.clamp(2.0, crate::motor::OUTPUT_CALIB_RANGE_MAX_HZ);
        chans.push(Chan { members: ks, base, range });
    }

    if chans.is_empty() {
        // SINGLE-CELL decoder (documented comparison baseline): the
        // best-SNR members across all classes, calibrated per cell.
        single_cell_fallback(rt, &members, &rates, &sigmas);
        return;
    }

    // install the measured calibrated functional channels
    if queen_diag() {
        eprintln!("queen decoder: POPULATION ({} functional channels):", chans.len());
        for (slot, p) in chans.iter().enumerate() {
            eprintln!(
                "  [{slot}] {:<24} n={:<3} base={:>7.1}Hz range={:>6.2}Hz",
                dof_label(slot),
                p.members.len(),
                p.base,
                p.range
            );
        }
    }
    let refs: Vec<&[usize]> = chans.iter().map(|c| c.members.as_slice()).collect();
    if rt.set_output_pools(&refs) {
        let base: Vec<f32> = chans.iter().map(|c| c.base).collect();
        let range: Vec<f32> = chans.iter().map(|c| c.range).collect();
        if !rt.set_motor_calibration(Some(&base), Some(&range)) {
            // length mismatch cannot happen (both are K) — defensive reset
            rt.set_motor_calibration(None, None);
        }
    }
}

// decode-slot display names (fixed order = the motor contract).
fn dof_label(slot: usize) -> &'static str {
    const DOF_LABELS: [&str; crate::motor::OUTPUT_CELL_COUNT] =
        ["DirX", "DirY", "Intensity", "Turn", "FeedSeek", "Emit"];
    DOF_LABELS.get(slot).copied().unwrap_or("dof")
}

/// SINGLE-CELL decoder (documented comparison baseline): the best-SNR members
/// across the annotated motor classes, calibrated per cell from the same
/// measured probe rates. Installed when no anatomical pool clears the noise
/// floor, so the queen always has a real-measured decoder.
fn single_cell_fallback(
    rt: &mut FlyBrainRuntime,
    members: &[usize],
    rates: &[Vec<f32>],
    sigmas: &[Vec<f32>],
) {
    const K: usize = crate::motor::OUTPUT_CELL_COUNT;
    let n_mem = members.len();
    let mut snr: Vec<(f32, usize)> = (0..n_mem)
        .map(|k| {
            let sig = sigmas.iter().map(|s| s[k]).sum::<f32>() / sigmas.len().max(1) as f32;
            let r0 = rates[0][k];
            let spread = rates.iter().map(|r| (r[k] - r0).abs()).fold(0.0f32, f32::max);
            (spread / sig.max(1.0), k)
        })
        .collect();
    snr.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    snr.truncate(K);
    if snr.len() < K {
        return;
    }
    if queen_diag() {
        eprintln!(
            "queen decoder: SINGLE-CELL (comparison mode; no defensible population channel) cells={:?}",
            snr.iter().map(|&(_, k)| members[k]).collect::<Vec<_>>()
        );
    }
    let mut cells = [0usize; K];
    let mut bl = vec![0f32; K];
    let mut rg = vec![0f32; K];
    for (slot, &(_, k)) in snr.iter().enumerate() {
        let mean_sigma = sigmas.iter().map(|s| s[k]).sum::<f32>() / sigmas.len().max(1) as f32;
        let r0 = rates[0][k];
        let spread = rates.iter().map(|r| (r[k] - r0).abs()).fold(0.0f32, f32::max);
        cells[slot] = members[k];
        bl[slot] = r0;
        rg[slot] = spread.max(2.5 * mean_sigma).max(2.0).min(crate::motor::OUTPUT_CALIB_RANGE_MAX_HZ);
    }
    if rt.set_output_binding(&cells) {
        rt.set_motor_calibration(Some(&bl), Some(&rg));
    }
}
