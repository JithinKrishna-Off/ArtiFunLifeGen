//! Motor decoding from spiking output cells.
//!
//! Output cells (default binding: the LAST `u_count` neurons of the
//! connectome, order documented in `OutputCell`) are decoded by running a
//! spike-rate EMA per cell, then converting Hz rates to bounded behavioural
//! scalars. All outputs are clamped: `move_dir` components to [-1, 1] unit
//! behavior, others to [0, 1]. The mapping from pools to movement scalars is
//! A MODEL DECISION for the synthetic circuit, not a neural-encoding claim.

use serde::{Deserialize, Serialize};

pub const OUTPUT_CELL_COUNT: usize = 6;

/// Last-six-neuron default binding order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputCell {
    /// movement direction X (cos component)
    DirX = 0,
    /// movement direction Y (sin component)
    DirY = 1,
    /// locomotion intensity
    Intensity = 2,
    /// turning (0.5 = straight, >0.5 one way, <0.5 the other)
    Turn = 3,
    /// food-seeking gate
    FeedSeek = 4,
    /// signal emission gate
    Emit = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotorOutput {
    /// Bounded movement direction, components in [-1, 1] (not force-normalized
    /// to a unit vector; caller may normalize).
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

impl Default for MotorOutput {
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

impl MotorOutput {
    /// Hard clamp (defensive; decode path already clamps).
    pub fn clamp_bounded(&mut self) {
        self.move_dir.0 = self.move_dir.0.clamp(-1.0, 1.0);
        self.move_dir.1 = self.move_dir.1.clamp(-1.0, 1.0);
        self.move_intensity = self.move_intensity.clamp(0.0, 1.0);
        self.turn = self.turn.clamp(-1.0, 1.0);
        self.feed_seek = self.feed_seek.clamp(0.0, 1.0);
        self.emit_signal = self.emit_signal.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotorConfig {
    /// Output neuron binding: the last `u_count` neurons, index
    /// `n - OUTPUT_CELL_COUNT + cell`, order per [OutputCell].
    pub u_count: usize,
    /// Rate EMA time constant (ms).
    pub ema_tau_ms: f32,
    /// Rate (Hz) considered "half of full deflection" for [0,1] channels.
    pub rate_mid_hz: f32,
    /// Rate (Hz) mapping to full deflection.
    pub rate_max_hz: f32,
    /// Internal EMA state snapshot.
    pub ema: Vec<f32>,
    /// Per-channel calibration (K entries, one per DRIVEN channel, K =
    /// channel count; from the measured calibration probe of a region-bound
    /// queen run: `baseline` Hz (mean rate across scenes) and `range` Hz
    /// (measured scene modulation). When present, decode RELATIVE rates:
    /// channel value = (ema - baseline) / range — baseline subtraction removes
    /// each real population's tonic offset so well-driven channels decode near
    /// 0 at rest and open only when the scene drives them (documented measured
    /// calibration, no fabricated biology). `None` keeps the absolute
    /// `rate_mid_hz`/`rate_max_hz` decode.
    #[serde(default)]
    pub calib_baseline: Option<Vec<f32>>,
    #[serde(default)]
    pub calib_range: Option<Vec<f32>>,
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            u_count: OUTPUT_CELL_COUNT,
            ema_tau_ms: 40.0,
            rate_mid_hz: 12.0,
            rate_max_hz: 30.0,
            ema: Vec::new(),
            calib_baseline: None,
            calib_range: None,
        }
    }
}

/// Advance per-cell EMAs given spike counts of this tick, then decode all six
/// output scalars.
///
/// Encoding convention (documented model decision):
/// - SIGNED cells (DirX, DirY, Turn) carry a tonic baseline: with zero spikes
///   their EMA target is `rate_mid_hz`, so the decoded deviation is 0 exactly
///   at silence; excitation above baseline gives positive deviation and
///   inhibition below baseline gives negative deviation.
/// - GATE cells (Intensity, FeedSeek, Emit) are silence-null: zero target at
///   zero spikes; decoded value grows from 0 as rate exceeds `rate_mid_hz`,
///   saturating at `rate_max_hz` -> 1.
/// `out_start` = DirX cell index; cell order follows [OutputCell]. The
/// default call decodes the CONTIGUOUS window `[out_start, out_start + 6)`；
/// region-grounded bindings decode an EXPLICIT cell list (one real neuron
/// per slot) through [`decode_cells`].
pub fn decode(
    cfg: &mut MotorConfig,
    spiked: &[u32],
    dt_ms: f32,
    out_start: usize,
) -> MotorOutput {
    let cells = [
        out_start,
        out_start + 1,
        out_start + 2,
        out_start + 3,
        out_start + 4,
        out_start + 5,
    ];
    decode_cells(cfg, spiked, dt_ms, &cells)
}

/// Upper bound (Hz) for an installed per-slot decode calibration range, so a
/// pathological probe result cannot stretch the decoder into permanent
/// silence.
pub const OUTPUT_CALIB_RANGE_MAX_HZ: f32 = 120.0;

/// Decode up to six output slots from POPULATION pools (one pool of real
/// neurons per FUNCTIONAL channel; the channel rate is the POOL MEAN spike
/// rate, in Hz/cell). Pool averaging suppresses single-cell noise. The
/// channel count K = `pools.len()` is whatever the anatomical data justified
/// (1..=6); body DOFs beyond K decode NEUTRAL (0). Same EMA and calibration
/// machinery as [`decode_cells`].
pub fn decode_pools(
    cfg: &mut MotorConfig,
    spiked: &[u32],
    dt_ms: f32,
    pools: &[&[u32]],
) -> MotorOutput {
    let k = pools.len().min(OUTPUT_CELL_COUNT);
    let mut counts = [0u32; OUTPUT_CELL_COUNT];
    for &s in spiked {
        for (j, pool) in pools.iter().take(k).enumerate() {
            if pool.contains(&s) {
                counts[j] += 1;
                break;
            }
        }
    }
    let rates: Vec<f32> = (0..k)
        .map(|j| counts[j] as f32 / pools[j].len().max(1) as f32 / dt_ms.max(0.001) * 1000.0)
        .collect();
    finish_decode(cfg, &rates, dt_ms)
}

/// Decode the six output cells from an EXPLICIT binding (region-grounded
/// queen run mode). `cells[k]` is the real connectome neuron carrying output
/// channel k; order follows [OutputCell]. Same EMA/rate encoding as [`decode`].
pub fn decode_cells(
    cfg: &mut MotorConfig,
    spiked: &[u32],
    dt_ms: f32,
    cells: &[usize; OUTPUT_CELL_COUNT],
) -> MotorOutput {
    let mut counts = [0u32; OUTPUT_CELL_COUNT];
    for &s in spiked {
        for (k, &c) in cells.iter().enumerate() {
            if s as usize == c {
                counts[k] += 1;
                break;
            }
        }
    }
    let rates: [f32; OUTPUT_CELL_COUNT] = std::array::from_fn(|k| {
        counts[k] as f32 / dt_ms.max(0.001) * 1000.0
    });
    finish_decode(cfg, &rates, dt_ms)
}

/// Final EMA advance + bounded decode from per-channel rates in Hz.
/// `rates.len()` = the number of DRIVEN body DOFs K (1..=6); DOFs beyond K
/// decode neutral (0) and their EMA is untouched. Single-cell bindings pass
/// exactly six rates; population bindings pass K = channel count.
fn finish_decode(cfg: &mut MotorConfig, rates: &[f32], dt_ms: f32) -> MotorOutput {
    let k = rates.len().min(OUTPUT_CELL_COUNT);
    if cfg.ema.len() != OUTPUT_CELL_COUNT {
        cfg.ema.clear();
        cfg.ema.resize(OUTPUT_CELL_COUNT, cfg.rate_mid_hz);
    }
    let alpha = (dt_ms / cfg.ema_tau_ms.max(1e-6)).clamp(0.0, 1.0);
    for (i, &r) in rates.iter().take(k).enumerate() {
        // signed body DOFs are the fixed slots DirX(0), DirY(1), Turn(3):
        // full silence carries the tonic baseline (same semantics as the
        // single-cell window decode); otherwise the rate is the real signal
        let signed = matches!(i, 0 | 1 | 3);
        let target = if signed && r == 0.0 {
            cfg.rate_mid_hz
        } else {
            r
        };
        cfg.ema[i] += alpha * (target - cfg.ema[i]);
    }
    let mut out = MotorOutput::default();
    if let (Some(base), Some(range)) = (&cfg.calib_baseline, &cfg.calib_range) {
        if base.len() == k && range.len() == k {
            // measured calibration decode: relative rates (baseline-subtracted,
            // range-normalized). Signed channels may go NEGATIVE (relative
            // inhibition); silence-null channels clamp at 0. Undriven body
            // DOFs (i >= k) stay at their zero/neutral default.
            let rel = |i: usize| -> f32 { (cfg.ema[i] - base[i]) / range[i].max(1.0) };
            if k > 0 {
                out.move_dir.0 = rel(0).clamp(-1.0, 1.0);
            }
            if k > 1 {
                out.move_dir.1 = rel(1).clamp(-1.0, 1.0);
            }
            if k > 2 {
                out.move_intensity = rel(2).clamp(0.0, 1.0);
            }
            if k > 3 {
                out.turn = rel(3).clamp(-1.0, 1.0);
            }
            if k > 4 {
                out.feed_seek = rel(4).clamp(0.0, 1.0);
            }
            if k > 5 {
                out.emit_signal = rel(5).clamp(0.0, 1.0);
            }
            out.clamp_bounded();
            return out;
        }
    }
    // absolute-rate decode (driven channels only; undriven DOFs 0)
    if k > 0 {
        out.move_dir.0 = signed_dev(cfg.ema[0], cfg);
    }
    if k > 1 {
        out.move_dir.1 = signed_dev(cfg.ema[1], cfg);
    }
    if k > 2 {
        out.move_intensity = gate(cfg.ema[2], cfg);
    }
    if k > 3 {
        out.turn = signed_dev(cfg.ema[3], cfg);
    }
    if k > 4 {
        out.feed_seek = gate(cfg.ema[4], cfg);
    }
    if k > 5 {
        out.emit_signal = gate(cfg.ema[5], cfg);
    }
    out.clamp_bounded();
    out
}

/// Signed deviation from tonic baseline, clamped to [-1, 1].
pub fn signed_dev(ema_hz: f32, cfg: &MotorConfig) -> f32 {
    let span = (cfg.rate_max_hz - cfg.rate_mid_hz).max(1e-6);
    ((ema_hz - cfg.rate_mid_hz) / span).clamp(-1.0, 1.0)
}

/// Gate value: 0 at/below `rate_mid_hz`, saturating to 1 at `rate_max_hz`.
pub fn gate(ema_hz: f32, cfg: &MotorConfig) -> f32 {
    let span = (cfg.rate_max_hz - cfg.rate_mid_hz).max(1e-6);
    ((ema_hz - cfg.rate_mid_hz) / span).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_bounded_under_saturating_rates() {
        let mut cfg = MotorConfig::default();
        let mut out = decode(&mut cfg, &[0, 0, 0, 0, 0, 0], 1.0, 0);
        out.clamp_bounded();
        assert!(out.move_dir.0.abs() <= 1.0 && out.move_dir.1.abs() <= 1.0);
        assert!((0.0..=1.0).contains(&out.move_intensity));
        // Saturate: high rates for many ticks stay in bounds.
        for _ in 0..200 {
            out = decode(&mut cfg, &[0, 1, 2, 3, 4, 5], 1.0, 0);
            out.clamp_bounded();
        }
        assert!(out.move_dir.0.abs() <= 1.0 && out.move_dir.1.abs() <= 1.0);
        assert!((0.0..=1.0).contains(&out.move_intensity));
        assert!(out.turn.abs() <= 1.0);
        assert!((0.0..=1.0).contains(&out.feed_seek));
        assert!((0.0..=1.0).contains(&out.emit_signal));
    }
}
