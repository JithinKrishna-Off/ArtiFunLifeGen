//! Type-safe unit spaces for the governor.
//!
//! The governor juggles three fundamentally different quantities. Mixing them
//! historically produced nonsense (e.g. "the queen spent 40 MB of hunger").
//! These newtypes make cross-space arithmetic a *compile error*; only
//! conversions that are actually meaningful exist, and none of them are
//! blind `From` impls.
//!
//! | Space | Newtype | Unit | Source |
//! |---|---|---|---|
//! | (a) Real hardware | [`HardwareMemory`] | bytes (process working set) | measured (sysinfo / mock) |
//! | (a) Real hardware | `f32` CPU percent | % of one core, EMA-smoothed | measured |
//! | (b) Abstract compute | [`ComputeUnits`] | CU (fictional, governor-defined) | scheduler accounting |
//! | (c) Biology | [`BioEnergy`] | core's native `f32` energy | `Organism::energy` |
//!
//! Rules enforced at compile time:
//! * `HardwareMemory + ComputeUnits` — does not compile.
//! * `ComputeUnits + BioEnergy` — does not compile (see the crate doctest).
//! * A policy share (fraction) is a plain `f32`; it is consumed by the
//!   scheduler and never handed to core's energy ledger as-is.

use serde::{Deserialize, Serialize};

/// Real, *measured* process memory in bytes (app-only footprint, never
/// total system memory). Canonical serialized form: a `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct HardwareMemory(u64);

impl HardwareMemory {
    pub const fn from_bytes(b: u64) -> Self {
        Self(b)
    }
    pub const fn bytes(self) -> u64 {
        self.0
    }
    /// Saturating addition (hardware counters never go negative, and an
    /// overflowed reading is still just "a lot").
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
    /// Ratio against a budget clamp in `[0.0, +inf)`; callers compare
    /// against threshold *fractions*, never against raw units.
    pub fn ratio_over(self, budget: Self) -> f32 {
        if budget.0 == 0 {
            return 0.0;
        }
        self.0 as f32 / budget.0 as f32
    }
}

/// Abstract computational cost in fictional units (CU). One CU is whatever
/// the coordinator's accounting says it is (e.g. "one neural forward pass in
/// the observatory costs 2 CU"); the governor only sums and budgets them.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ComputeUnits(f32);

impl ComputeUnits {
    pub const ZERO: Self = Self(0.0);
    pub const fn new(cu: f32) -> Self {
        Self(if cu.is_finite() && cu > 0.0 { cu } else { 0.0 })
    }
    pub const fn value(self) -> f32 {
        self.0
    }
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self((self.0 - rhs.0).max(0.0))
    }
    pub fn scale(self, k: f32) -> Self {
        Self((self.0 * k).max(0.0))
    }
}

impl std::ops::Add for ComputeUnits {
    type Output = ComputeUnits;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}

impl std::iter::Sum for ComputeUnits {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a.saturating_add(b))
    }
}

/// Biological energy in the simulation core's native `f32` space. Carried
/// through the governor *only* as metadata (e.g. average population energy
/// reported into `PolicyInput`); it never participates in hardware or CU
/// arithmetic, and the governor never writes into it.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BioEnergy(f32);

impl BioEnergy {
    pub fn new(e: f32) -> Self {
        Self(if e.is_finite() { e.max(0.0) } else { 0.0 })
    }
    pub const fn value(self) -> f32 {
        self.0
    }
    /// Only cross-space op that exists: a *read-only* ratio for heuristics
    /// (how starved is the colony, in CU terms) — it produces a fresh CU
    /// value and consumes neither space into the other.
    pub fn to_compute_pressure(self, full_scale: Self) -> ComputeUnits {
        if full_scale.0 <= 0.0 {
            return ComputeUnits::ZERO;
        }
        ComputeUnits::new((1.0 - (self.0 / full_scale.0).clamp(0.0, 1.0)) * 10.0)
    }
}
