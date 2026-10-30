//! SYNTHETIC TEST FIXTURE — guaranteed deterministic circuit for tests.
//!
//! THIS IS SYNTHETIC TEST DATA. It is NOT the MaleCNS connectome and must
//! never be presented as a real fly brain. All neurons carry
//! `ProvenanceTag::Assumption` and synthetic body IDs (>= SYNTHETIC_BODY_BASE).
//!
//! Layout (68 neurons; matches the runtime's default sensory/motor binding):
//!   inputs   0..22  : 11 sensory channels x 2 cells (rate encoders)
//!   relays   22..44 : one per channel cell, strong feedforward excitation
//!   inhib    44..50 : inhibitory interneurons (avoid-driven)
//!   attract  50..54 : attract-taxis pool
//!   avoid    54..58 : avoid/escape pool
//!   feedback 58..62 : delayed feedback loop neurons (back to attract)
//!   outputs  62..68 : DirX, DirY, Intensity, Turn, FeedSeek, Emit
//!
//! Guaranteed sensory->motor path: FoodDirCos input (8) -> relay (30) ->
//! attract (50) -> DirX output (62).

use serde::{Deserialize, Serialize};

use crate::connectome::{Connectome, EdgeSpec, NeuronMeta, Polarity, NO_IDX};
use crate::provenance::ProvenanceTag;

/// Synthetic body IDs start here (upstream body IDs are far above this).
pub const SYNTHETIC_BODY_BASE: u64 = 900_000_000;

pub const FIXTURE_N: usize = 68;
pub const FIXTURE_INPUT_CELLS: usize = 22;
pub const FIXTURE_RELAY_START: usize = 22;
pub const FIXTURE_INHIB_START: usize = 44;
pub const FIXTURE_ATTRACT_START: usize = 50;
pub const FIXTURE_AVOID_START: usize = 54;
pub const FIXTURE_FEEDBACK_START: usize = 58;
pub const FIXTURE_OUTPUT_START: usize = 62;

/// Strong excitatory synapse weight (+1 spike suprathreshold at defaults).
pub const W_EXC: f32 = 14.0;
/// Weak excitatory weight.
pub const W_EXC_WEAK: f32 = 8.0;
/// Inhibitory magnitude (effective -18 via Inhibitory polarity).
pub const W_INH: f32 = 18.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fixture {
    pub connectome: Connectome,
    /// Neuron index starts for quick assertions.
    pub layout: FixtureLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureLayout {
    pub inputs: usize,
    pub relays: usize,
    pub inhib: usize,
    pub attract: usize,
    pub avoid: usize,
    pub feedback: usize,
    pub outputs: usize,
}

fn meta(body_id: u64) -> NeuronMeta {
    NeuronMeta {
        body_id,
        region_idx: NO_IDX,
        cell_type_idx: NO_IDX,
        polarity: Polarity::Unknown,
        origin: ProvenanceTag::Assumption,
    }
}

/// Build the 68-neuron deterministic synthetic circuit.
pub fn synthetic_fixture() -> Fixture {
    let mut nodes: Vec<NeuronMeta> = Vec::with_capacity(FIXTURE_N);
    for i in 0..FIXTURE_N {
        nodes.push(meta(SYNTHETIC_BODY_BASE + i as u64));
    }
    // Structural polarity notes (all Assumption): inhib interneurons and the
    // avoid pool marked Inhibitory source cells for clarity; actual effect is
    // driven by per-synapse polarity flags below.
    for i in FIXTURE_INHIB_START..FIXTURE_OUTPUT_START {
        nodes[i].polarity = Polarity::Inhibitory;
    }

    let mut edges: Vec<EdgeSpec> = Vec::new();
    let exc = |pre: u32, post: u32, w: f32| EdgeSpec {
        pre,
        post,
        weight: w,
        delay: None,
        polarity: Some(Polarity::Excitatory),
    };
    let inh = |pre: u32, post: u32, w: f32| EdgeSpec {
        pre,
        post,
        weight: w,
        delay: None,
        polarity: Some(Polarity::Inhibitory),
    };

    // 1) sensory inputs -> relays (strong, one-to-one)
    for c in 0..11usize {
        for k in 0..2usize {
            let i = (c * 2 + k) as u32;
            let r = (FIXTURE_RELAY_START + c * 2 + k) as u32;
            edges.push(exc(i, r, W_EXC));
        }
    }

    // 2) food relays (channels 8,9 = FoodDirCos/Sin) -> attract pool
    for pr in [8usize, 9usize] {
        for k in 0..2usize {
            let r = (FIXTURE_RELAY_START + pr * 2 + k) as u32;
            for a in FIXTURE_ATTRACT_START..FIXTURE_ATTRACT_START + 4 {
                edges.push(exc(r, a as u32, W_EXC));
            }
        }
    }
    // optic relays (channels 1,2 = OpticCos/Sin) -> attract pool (weaker)
    for pr in [1usize, 2usize] {
        for k in 0..2usize {
            let r = (FIXTURE_RELAY_START + pr * 2 + k) as u32;
            for a in FIXTURE_ATTRACT_START..FIXTURE_ATTRACT_START + 4 {
                edges.push(exc(r, a as u32, W_EXC_WEAK));
            }
        }
    }

    // 3) obstacle (ch 5) + optic speed (ch 0) relays -> avoid pool (strong)
    for pr in [0usize, 5usize] {
        for k in 0..2usize {
            let r = (FIXTURE_RELAY_START + pr * 2 + k) as u32;
            for a in FIXTURE_AVOID_START..FIXTURE_AVOID_START + 4 {
                edges.push(exc(r, a as u32, W_EXC));
            }
        }
    }

    // 4) avoid-driven inhibition: obstacle/speed relays -> inhib interneurons
    //    -> attract pool (cross-inhibition: hazard suppresses attract drive)
    for pr in [0usize, 5usize] {
        for k in 0..2usize {
            let r = (FIXTURE_RELAY_START + pr * 2 + k) as u32;
            for h in FIXTURE_INHIB_START..FIXTURE_INHIB_START + 4 {
                edges.push(exc(r, h as u32, W_EXC));
            }
        }
    }
    for h in FIXTURE_INHIB_START..FIXTURE_INHIB_START + 4 {
        for a in FIXTURE_ATTRACT_START..FIXTURE_ATTRACT_START + 4 {
            edges.push(inh(h as u32, a as u32, W_INH));
        }
    }
    // avoid pool -> attract pool (direct cross-inhibition) keeps taxis swaps
    // mutually exclusive; documented circular-circuit design choice.
    for v in FIXTURE_AVOID_START..FIXTURE_AVOID_START + 4 {
        for a in FIXTURE_ATTRACT_START..FIXTURE_ATTRACT_START + 4 {
            edges.push(inh(v as u32, a as u32, W_INH));
        }
    }

    // 5) pools -> outputs
    for a in FIXTURE_ATTRACT_START..FIXTURE_ATTRACT_START + 4 {
        edges.push(exc(a as u32, 62 /* DirX */, W_EXC));
        edges.push(exc(a as u32, 63 /* DirY */, W_EXC));
        edges.push(exc(a as u32, 64 /* Intensity */, W_EXC_WEAK));
    }
    for v in FIXTURE_AVOID_START..FIXTURE_AVOID_START + 4 {
        edges.push(inh(v as u32, 62, W_INH)); // hazard pushes X negative
        edges.push(exc(v as u32, 65 /* Turn */, W_EXC));
    }
    // food proximity relays (ch 7) + food dir relays -> FeedSeek
    for pr in [7usize, 8usize, 9usize] {
        for k in 0..2usize {
            let r = (FIXTURE_RELAY_START + pr * 2 + k) as u32;
            edges.push(exc(r, 66, W_EXC));
        }
    }
    // queen relay (ch 10) -> Emit
    for k in 0..2usize {
        let r = (FIXTURE_RELAY_START + 10 * 2 + k) as u32;
        edges.push(exc(r, 67, W_EXC));
    }

    // 6) motor -> feedback -> attract loop (delayed, stabilizing/path-completing)
    for (o, f) in [(62u32, 58u32), (63, 59), (64, 60), (65, 61)] {
        edges.push(EdgeSpec {
            pre: o,
            post: f,
            weight: W_EXC_WEAK,
            delay: Some(1),
            polarity: Some(Polarity::Excitatory),
        });
    }
    for (f, a) in [(58u32, 50u32), (59, 51), (60, 52), (61, 53)] {
        edges.push(EdgeSpec {
            pre: f,
            post: a,
            weight: W_EXC_WEAK,
            delay: Some(2),
            polarity: Some(Polarity::Excitatory),
        });
    }

    let connectome = Connectome::from_edges(
        "synthetic-taxi-fixture",
        "fixture-v1",
        nodes,
        Vec::new(),
        Vec::new(),
        edges,
    )
    .expect("fixture must build");
    Fixture {
        connectome,
        layout: FixtureLayout {
            inputs: 0,
            relays: FIXTURE_RELAY_START,
            inhib: FIXTURE_INHIB_START,
            attract: FIXTURE_ATTRACT_START,
            avoid: FIXTURE_AVOID_START,
            feedback: FIXTURE_FEEDBACK_START,
            outputs: FIXTURE_OUTPUT_START,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_valid_and_labeled_synthetic() {
        let f = synthetic_fixture();
        assert_eq!(f.connectome.n(), FIXTURE_N);
        assert!(f.connectome.validate().is_ok());
        for m in &f.connectome.node_meta {
            assert_eq!(m.origin, ProvenanceTag::Assumption);
            assert!(m.body_id >= SYNTHETIC_BODY_BASE);
        }
        let st = f.connectome.stats();
        assert!(st.synapse_count > 100);
        // 62..68 free slots: guaranteed motor path flood sanity below.
        assert_eq!(f.connectome.col_idx.iter().max().copied(), Some(67u32));
    }
}
