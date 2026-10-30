//! LIF (leaky integrate-and-fire) per-organism brain state and the
//! delayed-spike ring buffer.
//!
//! Cell model (1 ms fixed step, deterministic):
//!   v(t+1) = leak*(v(t) - v_rest) + v_rest + I(t)      when not refractory
//!   spike(v >= v_thresh): v := v_reset, refractory := period, propagate after
//!   delay steps via the ring. `I(t)` combines: arrivals drained from the ring
//!   at the start of the tick + sensory pulses injected during the tick.
//!   `leak = exp(-dt/tau)` is precomputed once (single `exp`, no per-tick float
//!   re-derivation) so stepping is a pure multiply-add chain.
//!
//! There are two connectome ownership modes by design:
//! - STATIC: `FlyBrainRuntime { connectome: Arc<Connectome> }` shared across
//!   many organisms (the common case; state below is the per-organism part).
//! - DYNAMIC: a per-organism `Arc<Connectome>` from a per-organism genotype
//!   file; the runtime API is identical either way. This crate never stores a
//!   per-organism copy of a shared static connectome.
//!
//! No RNG is used anywhere in stepping: sensory channels use deterministic
//! phase-accumulator rate encoders (see `sensory.rs`), so identical inputs
//! always produce identical spikes and motor outputs.

use serde::{Deserialize, Serialize};

/// LIF + stepping parameters (all documented in FLYBRAIN_IMPLEMENTATION.md).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifParams {
    /// Integration step in milliseconds (spec: deterministic 1 ms step).
    pub dt_ms: f32,
    /// Membrane time constant in ms.
    pub tau_ms: f32,
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_thresh: f32,
    /// Refractory period in steps.
    pub refractory_steps: u8,
    /// Default propagation delay in steps for synapses without an explicit
    /// one. Always >= 1 (a synapse never acts on the same tick it spiked).
    pub default_delay_steps: u8,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            dt_ms: 1.0,
            tau_ms: 20.0,
            v_rest: -70.0,
            v_reset: -75.0,
            v_thresh: -57.0,
            refractory_steps: 2,
            default_delay_steps: 1,
        }
    }
}

impl LifParams {
    /// Precomputed exponential leak.
    pub fn leak(&self) -> f32 {
        (-self.dt_ms.max(0.001) / self.tau_ms.max(0.001)).exp()
    }

    pub fn max_delay(&self, connectome_max_delay: u8) -> u8 {
        self.default_delay_steps
            .max(1)
            .max(connectome_max_delay)
    }
}

/// Max explicit delay supported by the ring (u8, so 255 max by construction).
pub const MAX_RING_DELAY: u8 = 16;

/// Per-organism spiking state. Snapshot-serializable for save/restore.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainState {
    pub tick: u64,
    pub v: Vec<f32>,
    pub refractory: Vec<u8>,
    /// Current injected into each cell for THIS tick (ring arrivals + sensory).
    pub inj: Vec<f32>,
    /// Neurons that spiked during the last tick (ascending order).
    pub spiked: Vec<u32>,
    /// Delayed-spike ring: `ring[t % len]` holds (target, current) pairs that
    /// arrive at tick t.
    pub ring: Vec<Vec<(u32, f32)>>,
    /// Ring length (max delay + 1, >= 2).
    pub ring_len: u8,
}

impl BrainState {
    /// `v_rest` initializes the membrane at rest (NOT 0.0 — with the default
    /// `v_thresh = -57` a 0.0 initialize would spike every cell on tick 1).
    pub fn new(n: usize, ring_len: u8, v_rest: f32) -> Self {
        let ring_len = ring_len.max(2);
        Self {
            tick: 0,
            v: vec![v_rest; n],
            refractory: vec![0; n],
            inj: vec![0.0; n],
            spiked: Vec::new(),
            ring: (0..ring_len).map(|_| Vec::new()).collect(),
            ring_len,
        }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    /// Ring slot for an arrival at `tick + delay`.
    pub fn slot(&self, tick: u64, delay: u8) -> usize {
        ((tick + delay as u64) % self.ring_len as u64) as usize
    }

    /// Drain the arrivals scheduled for `self.tick` into `inj`.
    pub fn drain_ring(&mut self) {
        let slot = (self.tick % self.ring_len as u64) as usize;
        for (target, cur) in std::mem::take(&mut self.ring[slot]) {
            if (target as usize) < self.inj.len() {
                self.inj[target as usize] += cur;
            }
        }
    }

    /// Apply the LIF update to one cell. Returns true when it spikes.
    pub fn step_cell(&mut self, i: usize, leak: f32, p: &LifParams) -> bool {
        if self.refractory[i] > 0 {
            self.refractory[i] -= 1;
            self.v[i] = p.v_reset;
            return false;
        }
        let v = leak * (self.v[i] - p.v_rest) + p.v_rest + self.inj[i];
        if v >= p.v_thresh {
            self.v[i] = p.v_reset;
            self.refractory[i] = p.refractory_steps;
            true
        } else {
            self.v[i] = v;
            false
        }
    }

    pub fn clear_inj(&mut self) {
        for x in self.inj.iter_mut() {
            *x = 0.0;
        }
        self.spiked.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_drain_deterministic() {
        let mut st = BrainState::new(2, 4, -70.0);
        let slot = st.slot(0, 3);
        st.ring[slot].push((1, 5.0));
        st.tick = 3;
        st.drain_ring();
        assert_eq!(st.inj[1], 5.0);
        st.tick = 4;
        st.clear_inj();
        st.drain_ring();
        assert_eq!(st.inj[1], 0.0);
    }
}
