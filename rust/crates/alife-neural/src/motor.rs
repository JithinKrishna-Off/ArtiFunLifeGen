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
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            u_count: OUTPUT_CELL_COUNT,
            ema_tau_ms: 40.0,
            rate_mid_hz: 12.0,
            rate_max_hz: 30.0,
            ema: Vec::new(),
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
/// `out_start` = DirX cell index; cell order follows [OutputCell].
pub fn decode(
    cfg: &mut MotorConfig,
    spiked: &[u32],
    dt_ms: f32,
    out_start: usize,
) -> MotorOutput {
    if cfg.ema.len() != OUTPUT_CELL_COUNT {
        cfg.ema.clear();
        cfg.ema.resize(OUTPUT_CELL_COUNT, cfg.rate_mid_hz);
    }
    let alpha = (dt_ms / cfg.ema_tau_ms.max(1e-6)).clamp(0.0, 1.0);
    let mut counts = [0u32; OUTPUT_CELL_COUNT];
    for &s in spiked {
        if (s as usize) >= out_start && s < (out_start + OUTPUT_CELL_COUNT) as u32 {
            counts[s as usize - out_start] += 1;
        }
    }
    for (i, &c) in counts.iter().enumerate() {
        // signed cells are the enum discriminants DirX(0), DirY(1), Turn(3)
        let signed = matches!(i, 0 | 1 | 3);
        let target = if signed && c == 0 {
            cfg.rate_mid_hz
        } else {
            c as f32 * 1000.0 / dt_ms.max(0.001)
        };
        cfg.ema[i] += alpha * (target - cfg.ema[i]);
    }
    let mut out = MotorOutput::default();
    out.move_dir.0 = signed_dev(cfg.ema[OutputCell::DirX as usize], cfg);
    out.move_dir.1 = signed_dev(cfg.ema[OutputCell::DirY as usize], cfg);
    out.move_intensity = gate(cfg.ema[OutputCell::Intensity as usize], cfg);
    out.turn = signed_dev(cfg.ema[OutputCell::Turn as usize], cfg);
    out.feed_seek = gate(cfg.ema[OutputCell::FeedSeek as usize], cfg);
    out.emit_signal = gate(cfg.ema[OutputCell::Emit as usize], cfg);
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
