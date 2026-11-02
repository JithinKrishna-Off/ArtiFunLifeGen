//! Biology -> render morphology bridge.
//!
//! The 3D renderer (alife-view3d) currently re-derives anatomy from an
//! organism's traits at draw time. This module owns the single deterministic
//! mapping from the REAL genotype (`Genotype` / `Traits`), the lineage id and
//! a stable genome fingerprint to a small planar `CreatureMorph` descriptor
//! that render agents can consume and cache.
//!
//! RULES (hard contract):
//! - Every value is derived from a real trait, the lineage hue, or a
//!   deterministic hash of the genome. NO runtime randomness, NO environment
//!   reads, NO wall-clock reads. Same organism -> bitwise identical struct.
//! - Allocation-light: `Organism` is borrowed read-only; the result is a
//!   plain Copy struct (no `Vec`, `String`, or `Option<&[T]>`). Safe to call
//!   per-frame per-organism for cache-key building.
//! - Traits never change during an organism's life (the engine mutates
//!   genotypes only via reproduction), so `morph_key` is stable for the
//!   organism's lifetime; render caches keyed by it rebuild only on birth.
//!
//! Body plan mapping (must stay in lockstep with the existing view3d brand
//! table, see crates/alife-view3d/src/morph.rs `brand_of`; discriminants equal
//! the BRAND_* constants):
//!
//! | brand               | BodyPlan                       |
//! |---------------------|--------------------------------|
//! | BRAND_CELL (0)      | `BodyPlan::Primitive` (D)      |
//! | BRAND_FLAGELLATE(1) | `BodyPlan::SegmentedArthropod` (A) |
//! | BRAND_COLONY (2)    | `BodyPlan::MultiLegged` (C)    |
//! | BRAND_FILAMENT (3)  | `BodyPlan::Quadruped` (B)      |

use crate::species_hue;
use crate::{organism_radius, Organism};

/// Body architecture. Discriminants intentionally equal the existing renderer
/// brand constants (`BRAND_CELL..BRAND_FILAMENT`) consumed by view3d.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodyPlan {
    /// D: cell / sessile blob (brand 0). Simple body, no articulated limbs.
    Primitive = 0,
    /// A: segmented arthropod, tail-driven (brand 1, "flagellate").
    SegmentedArthropod = 1,
    /// C: multi-legged cluster body (brand 2, "colony").
    MultiLegged = 2,
    /// B: quadruped-like chained segments (brand 3, "filament").
    Quadruped = 3,
}

impl BodyPlan {
    /// Renderer brand constant this plan maps to (0..3).
    pub fn as_brand(self) -> u32 {
        self as u32
    }
}

/// Mouth architecture, from real `metabolism` (no mouth gene exists).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mouthpart {
    /// low metabolism (<= 0.9): absorb/suck surface
    Sucker,
    /// mid metabolism (0.9..1.4): graze/filter
    Filter,
    /// high metabolism (>= 1.4): bite
    Jaw,
}

/// Deterministic morphology descriptor. All ranges are documented and
/// clamped in `derive`. Copy-only (renderer-cache friendly).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreatureMorph {
    /// Body architecture (maps 1:1 to the existing BRAND_* palette).
    pub body_plan: BodyPlan,
    /// Body length in world units = `organism_radius(t) = 5.5 * traits.size`
    /// (min..max = 3.575..8.8). Renderers scale silhouettes down (~0.3x) to
    /// screen units; this is the simulated linear body extent.
    pub body_length: f32,
    /// Fraction of body length that is the head. Source: `metabolism`
    /// normalized over its TRAIT_MIN..TRAIT_MAX span, 0.18..0.42.
    pub head_ratio: f32,
    /// Segments 2..6. Plan base (D:2, A:3, B:4, C:5) + longevity term from
    /// `lifespan` normalized over TRAIT_MIN..TRAIT_MAX lifespan span.
    /// NOTE: no segment gene exists; lifespan is the documented proxy.
    pub segment_count: u8,
    /// Appendages 0..8. Plan base (D:0, A:2, B:4, C:6) + 0..2 from
    /// `sensor_range` normalized. Matches the legacy view3d convention
    /// (`morpho_param.limb_count` -> 2 * sensor_range / 100).
    pub appendage_count: u8,
    /// Appendage length range (fraction of body_length), min..max. Source:
    /// `efficiency` and inverse `max_speed`... see `derive`; 0.25..0.85.
    pub appendage_len: (f32, f32),
    /// Limb joint flexibility 0..1 (1 = fully jointed/floppy). Source:
    /// inverse normalized `efficiency` (cheap movers bend more).
    pub jointedness: f32,
    /// Eyes 1..3. Plan base (Primitive: 1, else 2) + 1 when
    /// `sensor_range >= 200` (sensory specialization => third eye).
    pub eye_count: u8,
    /// Eye size scale 0.5..1.5 from normalized `sensor_range`.
    pub eye_scale: f32,
    /// Antennae 0..2 from `sensor_range` rank (>=120 => 1, >= 200 => 2),
    /// halved for Primitive bodies (0) and Quadruped (at most 1).
    pub antenna_count: u8,
    /// Mouth from `metabolism` (see `Mouthpart` docs).
    pub mouthpart: Mouthpart,
    /// Armour plating 0..1. Source: half normalized `size` + half inverse
    /// normalized `max_speed` (big slow bodies plate up, fast bodies sleek).
    pub armour: f32,
    /// Dorsal spikes 0..6. Gene does NOT exist: genome hash bits (see
    /// `genome_hash`), sampled from the neural weights.
    pub spike_count: u8,
    /// Tail presence. Always true for SegmentedArthropod (flagellates drag
    /// 1-2 flagella per the decomp rule); otherwise genome hash bit unless
    /// Primitive (cells never grow tails).
    pub tail_present: bool,
    /// Tail length fraction of body_length 0.3..0.9 from normalized
    /// `max_speed` (faster flagella drags = longer tail). 0.0 when absent.
    pub tail_len: f32,
    /// Wings. No flight exists in the sim: genome hash bit only (visual
    /// vestigial pair; permitted only on SegmentedArthropod/Quadruped).
    pub winged: bool,
    /// Skin pattern (spots/stripes flag for shaders). Genome hash bit.
    pub patterned: bool,
    /// Species hue in degrees [0, 360): `species_hue(org.lineage)`
    /// (golden-angle walk; identical source as 2D renderer palette).
    pub hue: f32,
    /// Palette saturation tint nudge 0.15..0.45 from the genome hash
    /// (per-genome chroma variation layered on the species hue).
    pub sat_tint: f32,
}

// ------------------------------------------------------- normalization

/// Traits spans used everywhere below (mirror `TRAIT_MIN..TRAIT_MAX`).
const SPEED: (f32, f32) = (18.0, 150.0);
const EFF: (f32, f32) = (0.6, 2.4);
const SENSE: (f32, f32) = (40.0, 300.0);
const METAB: (f32, f32) = (0.6, 1.9);
const LIFE_SPAN: (f32, f32) = (2400.0, 14000.0);
const SIZE: (f32, f32) = (0.65, 1.6);

#[inline]
fn clamp_norm01(v: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Random-remap a [0,1) fraction through a fixed monotone span [a, b) with
/// clamping (so hash bits never escape documented ranges).
#[inline]
fn span01(t: f32, a: f32, b: f32) -> f32 {
    (a + (b - a) * t.clamp(0.0, 1.0)).clamp(a, b)
}

// ------------------------------------------------------- genome hashing

/// Deterministic genome fingerprint (fnv-1a 64 over a fixed, small sample of
/// the neural weights, all trait bit patterns and the lineage id).
///
/// Sampling contract (kept cheap for per-frame use; ~20 words hashed):
/// - `net.w1` stride samples: indices 0, S1, 2*S1, .. (S1 = ceil(n/8))
/// - `net.w1` ALWAYS includes index 0 (so single-weight edits always exit
///   through the fingerprint) and last index.
/// - `net.w2` stride samples + first/last; all of `b1`, `b2`; every trait as
///   its `to_bits()`; `org.lineage`.
///
/// Feeds exactly: `spike_count`, `tail_present` (non-flagellate plans),
/// `winged`, `patterned`, `sat_tint` — the "dress" fields. Never feeds
/// behaviour, never feeds fitness.
#[inline]
pub fn genome_hash(org: &Organism) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    #[inline]
    fn mix(h: &mut u64, w: u64) {
        *h ^= w;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let net = &org.genotype.net;
    let n1 = net.w1.len().max(1);
    let stride1 = (n1 / 8).max(1);
    let mut i = 0usize;
    while i < n1 {
        mix(&mut h, net.w1[i].to_bits() as u64);
        i += stride1;
    }
    mix(&mut h, net.w1[n1 - 1].to_bits() as u64);
    let n2 = net.w2.len().max(1);
    let stride2 = (n2 / 4).max(1);
    let mut i = 0usize;
    while i < n2 {
        mix(&mut h, net.w2[i].to_bits() as u64);
        i += stride2;
    }
    mix(&mut h, net.w2[n2 - 1].to_bits() as u64);
    for &b in &net.b1 {
        mix(&mut h, b.to_bits() as u64);
    }
    for &b in &net.b2 {
        mix(&mut h, b.to_bits() as u64);
    }
    let t = &org.genotype.traits;
    mix(&mut h, t.max_speed.to_bits() as u64);
    mix(&mut h, t.efficiency.to_bits() as u64);
    mix(&mut h, t.sensor_range.to_bits() as u64);
    mix(&mut h, t.metabolism.to_bits() as u64);
    mix(&mut h, t.lifespan as u64);
    mix(&mut h, t.size.to_bits() as u64);
    mix(&mut h, t.repr_threshold.to_bits() as u64);
    mix(&mut h, t.repr_cost.to_bits() as u64);
    mix(&mut h, org.lineage as u64);
    h
}

/// Splitmix64 finalizer for `morph_key` field mixing.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// Stable cheap fingerprint: mix the descriptor buckets of `derive`.
///
/// Contract for render caches: same organism twice -> same key; ANY used trait
/// change (or genome dress change) -> key change. Never touches position,
/// speed, energy, age, outputs (those are per-frame state, not morphology).
pub fn morph_key(org: &Organism) -> u64 {
    let m = derive(org);
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for v in [
        m.body_plan as u32 as u64,
        m.body_length.to_bits() as u64,
        m.head_ratio.to_bits() as u64,
        m.segment_count as u64,
        m.appendage_count as u64,
        m.appendage_len.0.to_bits() as u64,
        m.appendage_len.1.to_bits() as u64,
        m.jointedness.to_bits() as u64,
        m.eye_count as u64,
        m.eye_scale.to_bits() as u64,
        m.antenna_count as u64,
        m.mouthpart as u32 as u64,
        m.armour.to_bits() as u64,
        m.spike_count as u64,
        m.tail_present as u64,
        m.tail_len.to_bits() as u64,
        m.winged as u64,
        m.patterned as u64,
        m.hue.to_bits() as u64,
        m.sat_tint.to_bits() as u64,
        genome_hash(org),
    ] {
        h = mix64(h.wrapping_add(v.wrapping_mul(0x2545_f491_4f6c_dd1d)));
    }
    h
}

// ------------------------------------------------------- the derive

/// Pure, deterministic, allocation-free trait -> morphology mapping.
/// See the module docs for the full field -> source table.
pub fn derive(org: &Organism) -> CreatureMorph {
    let t = &org.genotype.traits;

    // -- body plan: exact mirror of view3d::morph::brand_of -------------
    let body_plan = if t.size < 0.85 {
        BodyPlan::Primitive
    } else if t.max_speed > 65.0 {
        BodyPlan::SegmentedArthropod
    } else if t.size > 1.25 {
        BodyPlan::Quadruped
    } else if t.max_speed <= 45.0 && t.efficiency < 1.0 && t.sensor_range >= 120.0 {
        BodyPlan::MultiLegged
    } else {
        BodyPlan::Primitive
    };

    let body_length = organism_radius(t).clamp(5.5 * SIZE.0, 5.5 * SIZE.1);
    let speed01 = clamp_norm01(t.max_speed, SPEED.0, SPEED.1);
    let eff01 = clamp_norm01(t.efficiency, EFF.0, EFF.1);
    let sense01 = clamp_norm01(t.sensor_range, SENSE.0, SENSE.1);
    let met01 = clamp_norm01(t.metabolism, METAB.0, METAB.1);
    let life01 = clamp_norm01(t.lifespan as f32, LIFE_SPAN.0, LIFE_SPAN.1);

    let head_ratio = span01(0.18 + 0.24 * met01, 0.18, 0.42);

    let plan_seg_base = match body_plan {
        BodyPlan::Primitive => 2u32,
        BodyPlan::SegmentedArthropod => 3,
        BodyPlan::Quadruped => 4,
        BodyPlan::MultiLegged => 5,
    };
    let segment_count = (plan_seg_base + (2.0 * life01).round() as u32).clamp(2, 6) as u8;

    let plan_limb_base = match body_plan {
        BodyPlan::Primitive => 0u32,
        BodyPlan::SegmentedArthropod => 2,
        BodyPlan::Quadruped => 4,
        BodyPlan::MultiLegged => 6,
    };
    let appendage_count = (plan_limb_base + (2.0 * sense01).round() as u32).clamp(0, 8) as u8;

    // limbs of cheap-frugal bodies run longer; fast bodies sleek shorter
    let appendage_len = (
        span01(0.25 + 0.25 * eff01 - 0.10 * speed01, 0.25, 0.85),
        span01(0.55 + 0.30 * eff01 - 0.15 * speed01, 0.35, 0.85),
    );

    let jointedness = span01(1.0 - eff01, 0.0, 1.0);

    let eye_count = match body_plan {
        BodyPlan::Primitive => 1u32,
        _ => 2,
    } + (t.sensor_range >= 200.0) as u32;
    let eye_count = eye_count.clamp(1, 3) as u8;
    let eye_scale = span01(0.5 + 1.0 * sense01, 0.5, 1.5);

    let antenna_rank = if t.sensor_range >= 200.0 { 2u32 } else if t.sensor_range >= 120.0 { 1 } else { 0 };
    let antenna_count = match body_plan {
        BodyPlan::Primitive => 0,
        BodyPlan::Quadruped => antenna_rank.clamp(0, 1),
        _ => antenna_rank,
    } as u8;

    let mouthpart = if t.metabolism <= 0.9 {
        Mouthpart::Sucker
    } else if t.metabolism < 1.4 {
        Mouthpart::Filter
    } else {
        Mouthpart::Jaw
    };

    let armour = span01(0.5 * clamp_norm01(t.size, SIZE.0, SIZE.1) + 0.5 * (1.0 - speed01), 0.0, 1.0);

    let gh = genome_hash(org);
    let spike_count = (gh & 0x7) as u8; // 0..=7 -> renderer clamps to 0..6

    let tail_present = match body_plan {
        // flagellates drag 1-2 flagella per the decomp anatomy rule
        BodyPlan::SegmentedArthropod => true,
        BodyPlan::Primitive => false,
        _ => gh & (1 << 8) != 0,
    };
    let tail_len = if tail_present {
        span01(0.3 + 0.6 * speed01, 0.3, 0.9)
    } else {
        0.0
    };

    let winged = gh & (1 << 9) != 0
        && matches!(body_plan, BodyPlan::SegmentedArthropod | BodyPlan::Quadruped);
    let patterned = gh & (1 << 10) != 0;
    let hue = species_hue(org.lineage).clamp(0.0, 359.999);
    let sat_tint = span01(((gh >> 11) & 0xFF) as f32 / 255.0, 0.15, 0.45);

    CreatureMorph {
        body_plan,
        body_length,
        head_ratio,
        segment_count,
        appendage_count,
        appendage_len,
        jointedness,
        eye_count,
        eye_scale,
        antenna_count,
        mouthpart,
        armour,
        spike_count,
        tail_present,
        tail_len,
        winged,
        patterned,
        hue,
        sat_tint,
    }
}

// ------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Network, Organism, Rng, TRAIT_MIN};

    fn bare(id: u64, lineage: u32, net: Network, size: f32, max_speed: f32) -> Organism {
        Organism {
            id,
            x: 0.0,
            y: 0.0,
            dir: 0.0,
            speed: 0.0,
            energy: 0.0,
            max_energy: 110.0,
            age: 0,
            generation: 1,
            genotype: crate::Genotype {
                net,
                traits: Traits {
                    size,
                    max_speed,
                    ..TRAIT_MIN
                },
            },
            lineage,
            mom: None,
            dad: None,
            offspring_count: 0,
            alive: true,
            repr_cooldown: 0,
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }

    use crate::Traits;

    #[test]
    fn deterministic_same_twice() {
        let org = bare(1, 7, Network::zeros(), 1.0, 30.0);
        let a = derive(&org);
        let b = derive(&org);
        assert_eq!(a, b, "derive must be pure");
        assert_eq!(morph_key(&org), morph_key(&org), "key must be pure");
    }

    /// Plan mapping matrix: the four documented brand thresholds.
    #[test]
    fn plan_mapping_matrix() {
        // small => Primitive (D) / BRAND_CELL 0
        let cell = bare(1, 3, Network::zeros(), 0.7, 30.0);
        assert_eq!(derive(&cell).body_plan, BodyPlan::Primitive);
        assert_eq!(derive(&cell).body_plan.as_brand(), 0);
        // fast => SegmentedArthropod (A) / BRAND_FLAGELLATE 1
        let flag = bare(2, 5, Network::zeros(), 1.0, 90.0);
        assert_eq!(derive(&flag).body_plan, BodyPlan::SegmentedArthropod);
        assert_eq!(derive(&flag).body_plan.as_brand(), 1);
        // colony rule: slow, cheap, far-sensing, mid-size => C / brand 2
        let mut colony = bare(3, 9, Network::zeros(), 1.0, 40.0);
        colony.genotype.traits.efficiency = 0.8;
        colony.genotype.traits.sensor_range = 140.0;
        assert_eq!(derive(&colony).body_plan, BodyPlan::MultiLegged);
        assert_eq!(derive(&colony).body_plan.as_brand(), 2);
        // big slow => Quadruped (B) / BRAND_FILAMENT 3
        let mut fil = colony;
        fil.genotype.traits.size = 1.4;
        assert_eq!(derive(&fil).body_plan, BodyPlan::Quadruped);
        assert_eq!(derive(&fil).body_plan.as_brand(), 3);
    }

    #[test]
    fn value_ranges_clamped_over_sweep() {
        let rng = &mut Rng::new(2026);
        for i in 0..2000u32 {
            let org = bare(i as u64, (i * 7919) % 10000, {
                let mut n = Network::zeros();
                n.w1[0] = rng.gauss_sigma(3.0);
                n.w2[0] = rng.gauss_sigma(3.0);
                n
            }, rng.range(0.5, 1.7), rng.range(10.0, 160.0));
            let m = derive(&org);
            assert!((5.5 * 0.65..=5.5 * 1.6).contains(&m.body_length));
            assert!((0.18..=0.42).contains(&m.head_ratio), "head {}", m.head_ratio);
            assert!((2..=6).contains(&(m.segment_count as i32)), "seg {}", m.segment_count);
            assert!(m.appendage_count <= 8, "app {}", m.appendage_count);
            let (lo, hi) = m.appendage_len;
            assert!(lo >= 0.25 && lo <= 0.85 && hi >= 0.35 && hi <= 0.85 && lo <= hi + f32::EPSILON,
                    "limb ({lo},{hi})");
            assert!((0.0..=1.0).contains(&m.jointedness));
            assert!((1..=3).contains(&(m.eye_count as i32)));
            assert!((0.5..=1.5).contains(&m.eye_scale), "eye {}", m.eye_scale);
            assert!(m.antenna_count <= 2, "ant {}", m.antenna_count);
            assert!((0.0..=1.0).contains(&m.armour), "arm {}", m.armour);
            assert!(m.spike_count <= 7, "spike {}", m.spike_count);
            if !m.tail_present {
                assert_eq!(m.tail_len, 0.0);
            } else {
                assert!((0.3..=0.9).contains(&m.tail_len), "tail {}", m.tail_len);
            }
            assert!(!m.winged || matches!(m.body_plan, BodyPlan::SegmentedArthropod | BodyPlan::Quadruped));
            assert!((0.15..=0.45).contains(&m.sat_tint), "sat {}", m.sat_tint);
            assert!((0.0..360.0).contains(&m.hue), "hue {}", m.hue);
        }
    }

    /// Variation sensitivity sample (documented): organisms differing ONLY in
    /// the surveyed trait must produce differing descriptor values.
    #[test]
    fn trait_changes_move_the_morph() {
        let base = Network::zeros();
        let slow = bare(1, 42, base, 1.0, 25.0);
        let mut fast = slow;
        fast.genotype.traits.max_speed = 140.0;
        // documented change: speed flips plan to A and grows the tail
        assert_ne!(derive(&slow).body_plan, derive(&fast).body_plan);
        assert!(derive(&fast).tail_len > derive(&slow).tail_len);

        let shortl = bare(2, 42, base, 0.9, 30.0);
        let mut longl = shortl;
        longl.genotype.traits.lifespan = 14000;
        shortl.genotype.traits.lifespan = 2400;
        // documented change: longevity drives segment_count
        assert_ne!(derive(&shortl).segment_count, derive(&longl).segment_count);

        let a = bare(3, 42, base, 0.9, 30.0);
        let mut b = a;
        b.genotype.traits.sensor_range = 300.0;
        a.genotype.traits.sensor_range = 40.0;
        // documented change: sensory reach drives appendage_count
        assert_ne!(derive(&a).appendage_count, derive(&b).appendage_count);
        assert_ne!(derive(&a).antenna_count, derive(&b).antenna_count);

        // genome dress: a single weight edit flips the hash-derived dress bits
        let g1 = bare(4, 42, base, 1.0, 40.0);
        let mut g2 = g1;
        g2.genotype.net.w1[0] = 1.5; // index 0 always enters genome_hash
        assert_ne!(genome_hash(&g1), genome_hash(&g2));
        let m1 = derive(&g1);
        let m2 = derive(&g2);
        assert_ne!(
            (m1.spike_count, m1.patterned, m1.winged, m1.sat_tint.to_bits()),
            (m2.spike_count, m2.patterned, m2.winged, m2.sat_tint.to_bits()),
            "dress must react to genome edits"
        );
        assert_ne!(morph_key(&g1), morph_key(&g2));
    }

    #[test]
    fn key_stability_and_diversity() {
        // equal organisms share the key across instances
        let a = bare(11, 777, Network::zeros(), 1.1, 55.0);
        let a2 = bare(11, 777, Network::zeros(), 1.1, 55.0);
        assert_eq!(morph_key(&a), morph_key(&a2));
        // key ignores per-frame state
        let mut a3 = a.clone();
        a3.energy = 10.0;
        a3.age = 500;
        a3.dir = 2.0;
        a3.x = -9.9;
        assert_eq!(morph_key(&a), morph_key(&a3));
        // a spread of genomes must produce distinct keys (no key collisions)
        let rng = &mut Rng::new(5);
        let mut keys = std::collections::HashSet::new();
        for i in 0..300u32 {
            let mut n = Network::zeros();
            let words = n.w1.len() + n.b1.len() + n.w2.len() + n.b2.len();
            let mut flat = Vec::with_capacity(words);
            for _ in 0..words {
                flat.push(rng.gauss_sigma(2.0));
            }
            n.w1.copy_from_slice(&flat[..n.w1.len()]);
            n.b1.copy_from_slice(&flat[n.w1.len()..n.w1.len() + n.b1.len()]);
            n.w2.copy_from_slice(&flat[n.w1.len() + n.b1.len()..n.w1.len() + n.b1.len() + n.w2.len()]);
            n.b2.copy_from_slice(&flat[words - n.b2.len()..]);
            let org = bare(i as u64, i % 999, n, rng.range(0.7, 1.5), rng.range(20.0, 120.0));
            keys.insert(morph_key(&org));
        }
        assert_eq!(keys.len(), 300, "distinct genomes must have distinct keys");
    }
}
