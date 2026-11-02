//! Sensory input struct + documented, configurable sensory -> neuron mapping
//! with rate/linear encoders.
//!
//! EVERY mapping in this file is a documented model interface decision, not a
//! claim about real fly anatomy:
//! - "light direction" is a TEMPERATURE-GRADIENT PROXY (the world has
//!   temperature fields, no light fields). It is described as light only as a
//!   behavioral-level abstraction and must never be presented as biological
//!   phototaxis wiring.
//! - Default neuron bindings (channels occupy the first `channel_cells * n_ch`
//!   neurons, motor cells the last `u_count`) are structural ASSUMPTIONS used
//!   by the synthetic fixture; for imported anatomical connectomes a real
//!   mapping has to be provided by future region-based assignment (see
//!   `subnetwork.rs`), and until then the default binding is flagged as an
//!   `Assumption` placeholder (documented, configurable, and exposed to the
//!   Queen layer via runtime configuration).

use serde::{Deserialize, Serialize};

use crate::connectome::Connectome;

/// Sensory value coming from the world layer. Callers (Queen/coordinator)
/// fill this from organism + field data; this crate defines only the format,
/// bounds contract and neuron mapping.
///
/// Bounds: `*_dist`/`*_prox`/`speed` values are in [0, 1]; direction vectors
/// are unit or zero vectors; `queen_signal` in [0, 1] (default 0.0, unused
/// until coordination signals exist).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SensoryInput {
    /// Normalized speed of nearby organisms (optic-flow proxy in [0, 1]).
    pub optic_flow_speed: f32,
    /// Mean heading direction offset of nearby organisms (cos, sin).
    pub optic_flow_dir: (f32, f32),
    /// TEMPERATURE-GRADIENT PROXY direction (cos, sin). Documented; not a
    /// biological light sensor.
    pub light_dir: (f32, f32),
    /// Hazard + logistic-edge proximity in [0, 1].
    pub obstacle_proximity: f32,
    /// Ground contact quality in [0, 1] (elevation/biome suitability).
    pub ground_contact: f32,
    /// Normalized proximity of nearest food in [0, 1].
    pub food_proximity: f32,
    /// Direction to nearest food (cos, sin), zero when no food known.
    pub food_dir: (f32, f32),
    /// Queen coordination signal, default 0.0 (no signal yet).
    pub queen_signal: f32,
}

impl Default for SensoryInput {
    fn default() -> Self {
        Self {
            optic_flow_speed: 0.0,
            optic_flow_dir: (0.0, 0.0),
            light_dir: (0.0, 0.0),
            obstacle_proximity: 0.0,
            ground_contact: 0.0,
            food_proximity: 0.0,
            food_dir: (0.0, 0.0),
            queen_signal: 0.0,
        }
    }
}

/// Scalar sensory channels, in fixed order. Values must already be within
/// their documented ranges ([0,1] for magnitude channels, [-1,1] for the
/// signed directional cos/sin channels).
pub const CHANNEL_COUNT: usize = 11;

pub const CH_OPTIC_SPEED: usize = 0;
pub const CH_OPTIC_COS: usize = 1;
pub const CH_OPTIC_SIN: usize = 2;
pub const CH_LIGHT_COS: usize = 3;
pub const CH_LIGHT_SIN: usize = 4;
pub const CH_OBSTACLE: usize = 5;
pub const CH_GROUND: usize = 6;
pub const CH_FOOD_PROX: usize = 7;
pub const CH_FOOD_COS: usize = 8;
pub const CH_FOOD_SIN: usize = 9;
pub const CH_QUEEN: usize = 10;

/// Signed channel flag by index.
pub const SIGNED_CHANNELS: [bool; CHANNEL_COUNT] = [
    false, // optic speed
    true, true, // optic flow dir
    true, true, // light (temperature-proxy) dir
    false, // obstacle
    false, // ground
    false, // food proximity
    true, true, // food dir
    false, // queen
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensoryConfig {
    /// Neurons bound per channel (>= 1). Default binding: channel c, cell k
    /// -> neuron `c * channel_cells + k`. Fixture uses 2.
    pub channel_cells: usize,
    /// Injected current per encoded pulse (mV-scale LIF units).
    pub pulse_current: f32,
    /// Encoder rate at mapped value 0 (Hz).
    pub baseline_hz: f32,
    /// Additional encoder rate at mapped value 1 (Hz).
    pub gain_hz: f32,
    /// Encoder phase accumulators (max `CHANNEL_COUNT * channel_cells`).
    pub accum: Vec<f32>,
}

impl Default for SensoryConfig {
    fn default() -> Self {
        Self {
            channel_cells: 2,
            pulse_current: 15.0,
            baseline_hz: 4.0,
            gain_hz: 26.0,
            accum: Vec::new(),
        }
    }
}

/// Encoder mode. Both are deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoder {
    /// Phase-accumulator pulse train: `rate_hz * dt` accumulates; whenever the
    /// accumulator passes an integer, that many pulses are injected this tick
    /// (fractional remainder carried). Pulse -> suprathreshold current.
    Rate,
    /// Linear: direct analogue current into the cell every tick.
    Linear,
}

/// Default neuron binding (see SensoryConfig).
pub fn input_neuron(cfg: &SensoryConfig, channel: usize, cell: usize) -> usize {
    cfg.channel_cells * channel + cell
}

/// Map a raw channel value to the encoder's [0, 1] input domain:
/// signed channels use `0.5 + 0.5*v` so both signs are expressed as magnitude
/// of firing; magnitude channels are clamped to [0, 1].
pub fn mapped_value(channel: usize, raw: f32) -> f32 {
    if SIGNED_CHANNELS[channel] {
        (0.5 + 0.5 * raw.clamp(-1.0, 1.0)) as f32
    } else {
        raw.clamp(0.0, 1.0)
    }
}

/// The full scalar channel vector for one input sample (already bounded).
pub fn channels(s: &SensoryInput) -> [f32; CHANNEL_COUNT] {
    let dir3 = |d: (f32, f32)| safe_unit(d);
    let of = dir3(s.optic_flow_dir);
    let ld = dir3(s.light_dir);
    let fd = dir3(s.food_dir);
    [
        s.optic_flow_speed.clamp(0.0, 1.0),
        of.0,
        of.1,
        ld.0,
        ld.1,
        s.obstacle_proximity.clamp(0.0, 1.0),
        s.ground_contact.clamp(0.0, 1.0),
        s.food_proximity.clamp(0.0, 1.0),
        fd.0,
        fd.1,
        s.queen_signal.clamp(0.0, 1.0),
    ]
}

fn safe_unit(d: (f32, f32)) -> (f32, f32) {
    let m = (d.0 * d.0 + d.1 * d.1).sqrt();
    if m <= 1e-6 || !m.is_finite() {
        (0.0, 0.0)
    } else {
        (d.0 / m, d.1 / m)
    }
}

/// Encode one sensory sample into the per-tick injected current vector.
/// Deterministic: same state + same input => same injections.
/// `inj` must have length `connectome.n()`.
pub fn encode_into(
    cfg: &mut SensoryConfig,
    input: &SensoryInput,
    dt_ms: f32,
    inj: &mut Vec<f32>,
    encoders: &[Encoder],
) {
    let ch = channels(input);
    encode_channels(cfg, &ch, dt_ms, inj, encoders);
}

/// Encode an EXPLICIT channel vector (alternate sensory mappings, e.g. the
/// queen adapter). `chans[c]` is each channel's RAW value in its documented
/// domain ([0,1] magnitude channels; [-1,1] signed channels); the same
/// defensive `mapped_value` clamping applies as for the `SensoryInput`
/// path. Deterministic.
pub fn encode_channels(
    cfg: &mut SensoryConfig,
    chans: &[f32; CHANNEL_COUNT],
    dt_ms: f32,
    inj: &mut Vec<f32>,
    encoders: &[Encoder],
) {
    encode_channels_explicit(cfg, chans, dt_ms, inj, encoders, None);
}

/// Same encoding with an OPTIONAL explicit absolute-neuron binding table
/// (length `inputs_needed(cfg)`, one target neuron per (channel, cell)
/// slot). `None` keeps the default contiguous layout. Each targeted neuron
/// must be < `inj.len()` (defensively skipped otherwise). Deterministic.
pub fn encode_channels_explicit(
    cfg: &mut SensoryConfig,
    chans: &[f32; CHANNEL_COUNT],
    dt_ms: f32,
    inj: &mut Vec<f32>,
    encoders: &[Encoder],
    binding: Option<&[usize]>,
) {
    if cfg.accum.len() != cfg.channel_cells * CHANNEL_COUNT {
        cfg.accum.resize(cfg.channel_cells * CHANNEL_COUNT, 0.0);
    }
    for (c, &val) in chans.iter().enumerate() {
        let v = mapped_value(c, val);
        for k in 0..cfg.channel_cells {
            let ai = c * cfg.channel_cells + k;
            let idx = match binding {
                Some(b) => {
                    let i = b[ai];
                    if i >= inj.len() {
                        continue;
                    }
                    i
                }
                None => input_neuron(cfg, c, k),
            };
            if idx >= inj.len() {
                continue;
            }
            match encoders[ai] {
                Encoder::Rate => {
                    // Phase-accumulator pulse train (deterministic): accumulate
                    // pulse fraction, inject whenever a whole pulse accrues.
                    cfg.accum[ai] = (cfg.accum[ai]
                        + rate_hz(v, cfg) * dt_ms / 1000.0)
                        .clamp(0.0, 4.0);
                    let pulses = cfg.accum[ai].floor() as i32;
                    if pulses > 0 {
                        cfg.accum[ai] -= pulses as f32;
                        inj[idx] += cfg.pulse_current * pulses as f32;
                    }
                }
                Encoder::Linear => {
                    inj[idx] += cfg.pulse_current * v * (cfg.gain_hz / 30.0);
                }
            }
        }
    }
}

fn rate_hz(v: f32, cfg: &SensoryConfig) -> f32 {
    (cfg.baseline_hz + cfg.gain_hz * v).max(0.0)
}

/// Validated default mapping for a connectome. Shrinks (documented) when the
/// connectome is smaller than the full default layout.
/// Returns (input_neuron_index_fn_closure-data, output_start) plus a
/// provenance-style marker describing which fields were derived.
pub fn validate_mapping(conn: &Connectome, cfg: &SensoryConfig, outputs: usize) -> bool {
    inputs_needed(cfg) <= conn.n()
        && cfg.channel_cells >= 1
        && outputs > 0
        && conn.n() >= outputs + inputs_needed(cfg)
}

pub fn inputs_needed(cfg: &SensoryConfig) -> usize {
    cfg.channel_cells * CHANNEL_COUNT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_mapping_symmetric() {
        assert!((mapped_value(CH_FOOD_COS, -1.0) - 0.0).abs() < 1e-6);
        assert!((mapped_value(CH_FOOD_COS, 1.0) - 1.0).abs() < 1e-6);
        assert!((mapped_value(CH_FOOD_COS, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_bounded() {
        let mut cfg = SensoryConfig::default();
        cfg.channel_cells = 2;
        let mut inj = vec![0.0; 68];
        let enc = vec![Encoder::Rate; 22];
        encode_into(&mut cfg, &SensoryInput::default(), 1.0, &mut inj, &enc);
        for x in inj.iter().take(22) {
            assert!(*x >= 0.0 && x.is_finite());
        }
        assert!(inj[22..].iter().all(|&x| x == 0.0));
    }
}

