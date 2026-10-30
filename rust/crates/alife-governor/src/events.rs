//! Bounded event ring for UI observatory consumption.
//!
//! Newest first (`events()[0]` = most recent). Capacity is clamped to at
//! most [`EVENT_RING_MAX`] (= 128) so a long session cannot grow the log.

use serde::{Deserialize, Serialize};

use crate::hardware::AdvisoryAction;

/// Hard upper bound of the ring (spec: <= 128).
pub const EVENT_RING_MAX: usize = 128;

pub(crate) struct EventLog {
    max: usize,
    ring: Vec<crate::GovernorEvent>,
}

impl EventLog {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            max: max.clamp(1, EVENT_RING_MAX),
            ring: Vec::with_capacity(max.clamp(1, EVENT_RING_MAX)),
        }
    }

    pub(crate) fn push(&mut self, event: crate::GovernorEvent) {
        self.ring.insert(0, event);
        if self.ring.len() > self.max {
            self.ring.truncate(self.max);
        }
    }

    pub(crate) fn as_slice(&self) -> &[crate::GovernorEvent] {
        &self.ring
    }
}

/// What happened, at which tick, quoting the advisory action that was
/// "taken" (advised) and a human-readable reason. Serializable so the UI
/// observatory and future save-v3 diagnostics can replay it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernorEvent {
    /// Simulation tick at emission time.
    pub tick: u64,
    pub kind: EventKind,
    /// The advisory action advised (never a lethal/irreversible operation).
    pub action_taken: AdvisoryAction,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    /// Hardware guardian state transitions.
    Hardware,
    /// Computational-selection CU starvation review.
    ComputeSelection,
}
