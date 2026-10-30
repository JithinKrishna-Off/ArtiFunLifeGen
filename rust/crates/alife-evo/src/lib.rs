//! Genetics controllers, mutation, and evolution machinery.
//!
//! Selection is natural only: survival and reproduction are decided by the
//! engine (alife-core); nothing here applies a hardcoded fitness function.
//! Species are a documented genotype-distance criterion — a child within
//! [`SPECIES_DISTANCE_THRESHOLD`] (mean-squared metric, see
//! [`genotype_distance`]) of its species centroid stays in that species,
//! otherwise it founds a new species with a fresh stable id. Mutation rules:
//! per gene at `mutation_rate`, network weights take additive gaussian steps
//! (sigma `0.5 * mutation_magnitude`) and traits take multiplicative gaussian
//! steps (sigma `6% * mutation_magnitude`); a mutating gene jumps 10x farther
//! with probability `big_jump_chance`; mutants are always clamped to viable
//! bounds via `clamp_traits` / weight clamping.

use serde::{Deserialize, Serialize};

use alife_common::{
    clamp_traits, sanitize, Genotype, Network, Rng, SimParams, Traits, NET_HIDDEN, NET_INPUTS,
    NET_OUTPUTS, TRAIT_MAX, TRAIT_MIN,
};

/// Baseline network "compactness" for fresh genotypes.
const BASE_COMPACT: f32 = 0.7;

/// Local Box-Muller gaussian (sigma `sigma`). We do not call
/// `alife_common::Rng::gauss`: that fn currently returns NaN for most draws
/// (its `.max(1e-20)` is applied to a negative log, forcing positive epsilon
/// into `sqrt(-2e)`). This local version consumes the identical two-draw
/// sequence (`next_f32` twice), keeping `Rng` streams deterministic.
#[inline]
fn gauss(rng: &mut Rng, sigma: f32) -> f32 {
    let u = 1.0 - rng.next_f32();
    let theta = 2.0 * std::f32::consts::TAU * rng.next_f32();
    (-2.0 * u.max(f32::MIN_POSITIVE).ln()).sqrt() * theta.cos() * sigma
}

/// Mean-squared genotype distance above which a child is considered a new
/// species/lineage founder (see [`genotype_distance`] and [`assign_species`]).
pub const SPECIES_DISTANCE_THRESHOLD: f32 = 0.62;

/// Species centroids tracked at most; beyond this the least-recently-matched
/// slot is recycled at the next founder event, so the book never grows unbound.
const MAX_TRACKED_SPECIES: usize = 64;

/// Centroid exponential-moving-average factor on each assignment.
const CENTROID_ALPHA: f32 = 0.05;

/// Weight bounds for mutated networks (symmetric, keeps activations sane).
const WEIGHT_BOUND: f32 = 6.0;

// ------------------------------------------------------------- fresh genomes

/// A randomized founder genotype. Three mixed strategy archetypes
/// (fast-burst, slow-efficient, sensor-heavy) are chosen by `rng`, plus a
/// uniform generalist spread — mixed seeds make early evolution richer while
/// staying fully deterministic for a given `Rng` seed.
pub fn random_genotype(rng: &mut Rng) -> Genotype {
    let t = match rng.int(3) {
        // fast-burst: aggressive speed, pays for it in metabolism
        0 => Traits {
            max_speed: rng.range(95.0, 150.0),
            efficiency: rng.range(1.0, 2.2),
            sensor_range: rng.range(60.0, 140.0),
            metabolism: rng.range(1.0, 1.9),
            lifespan: rng.range(3600.0, 7200.0) as u32,
            size: rng.range(0.7, 1.2),
            repr_threshold: rng.range(0.55, 0.8),
            repr_cost: rng.range(0.25, 0.45),
        },
        // slow-efficient: frugal grazer, long lifespan, small body budget
        1 => Traits {
            max_speed: rng.range(20.0, 60.0),
            efficiency: rng.range(0.6, 1.0),
            sensor_range: rng.range(70.0, 160.0),
            metabolism: rng.range(0.6, 1.0),
            lifespan: rng.range(7000.0, 12000.0) as u32,
            size: rng.range(0.9, 1.4),
            repr_threshold: rng.range(0.55, 0.8),
            repr_cost: rng.range(0.24, 0.4),
        },
        // sensor-heavy: long vision, moderate body, higher sensing cost
        _ => Traits {
            max_speed: rng.range(30.0, 90.0),
            efficiency: rng.range(0.8, 1.6),
            sensor_range: rng.range(180.0, 300.0),
            metabolism: rng.range(0.8, 1.4),
            lifespan: rng.range(5000.0, 10000.0) as u32,
            size: rng.range(0.75, 1.6),
            repr_threshold: rng.range(0.6, 0.85),
            repr_cost: rng.range(0.26, 0.5),
        },
    };
    Genotype { net: create_network(rng, BASE_COMPACT), traits: t }
}

/// Random feed-forward network; `compact` scales initial weight spread.
pub fn create_network(rng: &mut Rng, compact: f32) -> Network {
    let w1: Vec<f32> = (0..NET_HIDDEN * NET_INPUTS).map(|_| gauss(rng, compact)).collect();
    let b1: Vec<f32> = (0..NET_HIDDEN).map(|_| gauss(rng, compact)).collect();
    let w2: Vec<f32> = (0..NET_OUTPUTS * NET_HIDDEN).map(|_| gauss(rng, compact)).collect();
    let b2: Vec<f32> = (0..NET_OUTPUTS).map(|_| gauss(rng, compact)).collect();
    Network { w1, b1, w2, b2 }
}

/// Inherit-and-vary: returns a new genotype derived from `parent`.
/// Weights: additive gaussian step ~ N(0, `0.5 * mag`), 10x jump at
/// `big_jump_chance`. Traits: multiplicative ~ N(1, `0.06 * mag`), 10x jump at
/// `big_jump_chance`, relative factor clamped to [0.2, 5.0]. Lifespan: same
/// multiplicative rule with sigma `0.05 * mag`. All traits are clamped to
/// `TRAIT_MIN..TRAIT_MAX`, so even wild mutants remain viable.
pub fn mutate_genotype(parent: &Genotype, rng: &mut Rng, params: &SimParams) -> Genotype {
    let mut g = parent.clone();
    let rate = params.mutation_rate.clamp(0.0, 1.0);
    let mag = params.mutation_magnitude.max(0.05);
    let big = params.big_jump_chance.clamp(0.0, 1.0);

    for arr in [&mut g.net.w1, &mut g.net.b1, &mut g.net.w2, &mut g.net.b2] {
        for v in arr.iter_mut() {
            if rng.chance(rate) {
                let sigma = 0.5 * mag * if rng.chance(big) { 10.0 } else { 1.0 };
                let step = gauss(rng, sigma);
                *v = sanitize(*v + step).clamp(-WEIGHT_BOUND, WEIGHT_BOUND);
            }
        }
    }

    let t = &mut g.traits;
    let maybe_mutate = |rng: &mut Rng, v: &mut f32| {
        if rng.chance(rate) {
            let sigma = 0.06 * mag * if rng.chance(big) { 10.0 } else { 1.0 };
            let rel = (1.0 + gauss(rng, sigma).clamp(-0.9, 0.9)).clamp(0.2, 5.0);
            *v = sanitize(*v * rel);
        }
    };
    maybe_mutate(rng, &mut t.max_speed);
    maybe_mutate(rng, &mut t.efficiency);
    maybe_mutate(rng, &mut t.sensor_range);
    maybe_mutate(rng, &mut t.metabolism);
    maybe_mutate(rng, &mut t.size);
    maybe_mutate(rng, &mut t.repr_threshold);
    maybe_mutate(rng, &mut t.repr_cost);
    if rng.chance(rate) {
        let sigma = 0.05 * mag * if rng.chance(big) { 10.0 } else { 1.0 };
        let rel = (1.0 + gauss(rng, sigma).clamp(-0.9, 0.9)).clamp(0.2, 5.0);
        t.lifespan = ((t.lifespan as f32 * rel).round() as i64).clamp(1, u32::MAX as i64) as u32;
    }

    clamp_traits(t);
    g
}

// ------------------------------------------------------------- inference

/// Feed-forward controller: tanh hidden, tanh outputs.
/// Fixed-shape matrix-vector; chunks-driven rows compile to bounds-checked-free
/// tight loops and there are no heap allocations. Typical cost < 300ns.
#[inline]
pub fn think(net: &Network, inputs: &[f32; NET_INPUTS], out: &mut [f32; NET_OUTPUTS]) {
    let mut hidden = [0.0f32; NET_HIDDEN];

    for (h, (row, &b)) in net.w1.chunks_exact(NET_INPUTS).zip(net.b1.iter()).enumerate() {
        let mut s = b;
        for (&w, &x) in row.iter().zip(inputs.iter()) {
            s += w * x;
        }
        hidden[h] = s.tanh();
    }

    for (o, (row, &b)) in net.w2.chunks_exact(NET_HIDDEN).zip(net.b2.iter()).enumerate() {
        let mut s = b;
        for (&w, &x) in row.iter().zip(hidden.iter()) {
            s += w * x;
        }
        out[o] = s.tanh();
    }
}

// ------------------------------------------------------------- distance

fn norm01(v: f32, min: f32, max: f32) -> f32 {
    (v - min) / (max - min)
}

/// Mean squared deviation over all genome floats; traits normalized by their
/// `TRAIT_MIN..TRAIT_MAX` span and weighted 4x vs each net float. Used both
/// for species assignment and (optionally) UI similarity readouts.
pub fn genotype_distance(a: &Genotype, b: &Genotype) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (x, y) in [
        (&a.net.w1, &b.net.w1),
        (&a.net.b1, &b.net.b1),
        (&a.net.w2, &b.net.w2),
        (&a.net.b2, &b.net.b2),
    ] {
        for (&xx, &yy) in x.iter().zip(y.iter()) {
            let d = (xx - yy) as f64;
            sum += d * d;
            n += 1;
        }
    }
    let (t, t2) = (&a.traits, &b.traits);
    for (la, lb, kmin, kmax) in [
        (t.max_speed, t2.max_speed, TRAIT_MIN.max_speed, TRAIT_MAX.max_speed),
        (t.efficiency, t2.efficiency, TRAIT_MIN.efficiency, TRAIT_MAX.efficiency),
        (t.sensor_range, t2.sensor_range, TRAIT_MIN.sensor_range, TRAIT_MAX.sensor_range),
        (t.metabolism, t2.metabolism, TRAIT_MIN.metabolism, TRAIT_MAX.metabolism),
        (t.size, t2.size, TRAIT_MIN.size, TRAIT_MAX.size),
        (t.repr_threshold, t2.repr_threshold, TRAIT_MIN.repr_threshold, TRAIT_MAX.repr_threshold),
        (t.repr_cost, t2.repr_cost, TRAIT_MIN.repr_cost, TRAIT_MAX.repr_cost),
    ] {
        let da = norm01(la, kmin, kmax) as f64;
        let db = norm01(lb, kmin, kmax) as f64;
        let d = da - db;
        sum += d * d * 4.0;
        n += 4;
    }
    ((sum / n.max(1) as f64).sqrt()) as f32
}

// ------------------------------------------------------------- speciation

/// Book of live species centroids. Species ids are stable and monotonic; ids
/// start at 1 (0 means "unassigned"). Slot order is the deterministic scan
/// order — no HashMap dependence anywhere. Serializable so persistence can
/// restore ids exactly across saves.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SpeciesBook {
    /// stable species ids, slot order == scan order
    pub ids: Vec<u32>,
    /// rolling-mean centroid genotype per slot (EMA alpha 0.05)
    pub centroids: Vec<Genotype>,
    /// global recency counter stamp per slot, drives eviction past the cap
    pub last_touch: Vec<u64>,
    /// next fresh species id
    pub next_id: u32,
    /// global assignment counter
    touch_clock: u64,
}

/// Empty book; first assignment gets species id 1.
pub fn species_book_new() -> SpeciesBook {
    SpeciesBook {
        ids: Vec::new(),
        centroids: Vec::new(),
        last_touch: Vec::new(),
        next_id: 1,
        touch_clock: 0,
    }
}

/// Centroid EMA pull toward `child`, applied to net floats and traits alike.
fn ema_toward(centroid: &mut Genotype, child: &Genotype, alpha: f32) {
    let a = alpha;
    let pull = |src: &mut Vec<f32>, dst: &[f32]| {
        for (c, &k) in src.iter_mut().zip(dst.iter()) {
            *c += (k - *c) * a;
        }
    };
    pull(&mut centroid.net.w1, &child.net.w1);
    pull(&mut centroid.net.b1, &child.net.b1);
    pull(&mut centroid.net.w2, &child.net.w2);
    pull(&mut centroid.net.b2, &child.net.b2);
    let (c, k) = (&mut centroid.traits, &child.traits);
    c.max_speed += (k.max_speed - c.max_speed) * a;
    c.efficiency += (k.efficiency - c.efficiency) * a;
    c.sensor_range += (k.sensor_range - c.sensor_range) * a;
    c.metabolism += (k.metabolism - c.metabolism) * a;
    c.lifespan += ((k.lifespan as f32 - c.lifespan as f32) * a) as u32;
    c.size += (k.size - c.size) * a;
    c.repr_threshold += (k.repr_threshold - c.repr_threshold) * a;
    c.repr_cost += (k.repr_cost - c.repr_cost) * a;
}

/// Assign `child` to a species, given the species id of its parent.
/// Behavior: `parent_species` is seeded as the initial candidate (so the parent
/// lineage wins distance ties), then all centroids are scanned in slot order
/// (deterministic); the nearest centroid within
/// <= [`SPECIES_DISTANCE_THRESHOLD`] takes the child, its centroid is
/// EMA-updated (alpha 0.05) and the child keeps that species id. Otherwise the
/// child founds a new species with a fresh stable id (`book.next_id`), either
/// as a new slot or, past the 64-slot cap, by recycling the
/// least-recently-matched slot. Cost is O(#species) distance evals, typically
/// 1-2 because the parent species usually matches.
pub fn assign_species(book: &mut SpeciesBook, child: &Genotype, parent_species: u32) -> (u32, bool) {
    let mut best: Option<usize> = None;
    let mut best_d = f32::INFINITY;
    if let Some(p) = book.ids.iter().position(|&id| id == parent_species) {
        best_d = genotype_distance(&book.centroids[p], child);
        best = Some(p);
    }
    book.touch_clock += 1;

    if let Some(i) = best {
        if best_d <= SPECIES_DISTANCE_THRESHOLD {
            ema_toward(&mut book.centroids[i], child, CENTROID_ALPHA);
            book.last_touch[i] = book.touch_clock;
            let id = book.ids[i];
            return (id, false);
        }
    }

    // new founder
    book.next_id += 1;
    let id = book.next_id - 1;
    if book.ids.len() < MAX_TRACKED_SPECIES {
        book.ids.push(id);
        book.centroids.push(child.clone());
        book.last_touch.push(book.touch_clock);
    } else {
        let mut slot = 0;
        let mut min = book.last_touch[0];
        for (i, &t) in book.last_touch.iter().enumerate() {
            if t < min {
                min = t;
                slot = i;
            }
        }
        book.ids[slot] = id;
        book.centroids[slot] = child.clone();
        book.last_touch[slot] = book.touch_clock;
    }
    (id, true)
}

/// Per-species `(id, centroid diversity proxy)`: mean distance from that
/// species centroid to every other centroid (0.0 for a lone species). Higher
/// values mean a more isolated, more distinct lineage.
pub fn species_stats(book: &SpeciesBook) -> Vec<(u32, f32)> {
    let n = book.centroids.len();
    let mut out = Vec::with_capacity(n);
    if n == 1 {
        out.push((book.ids[0], 0.0));
        return out;
    }
    for i in 0..n {
        let mut s = 0.0f64;
        for j in 0..n {
            if i != j {
                s += genotype_distance(&book.centroids[i], &book.centroids[j]) as f64;
            }
        }
        out.push((book.ids[i], (s / (n - 1) as f64) as f32));
    }
    out
}

// ------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::{INPUT_LABELS, OUTPUT_LABELS, TRAIT_MIN};

    fn params() -> SimParams {
        SimParams::default()
    }

    #[test]
    fn labels_reexported_not_duplicated() {
        assert_eq!(INPUT_LABELS.len(), NET_INPUTS);
        assert_eq!(OUTPUT_LABELS.len(), NET_OUTPUTS);
        assert_eq!(OUTPUT_LABELS[NET_OUTPUTS - 1], "reproduce gate");
    }

    #[test]
    fn mutation_stays_viable_and_pairs_split() {
        let rng = &mut Rng::new(42);
        let mut root = random_genotype(rng);
        root.net.w1[3] = 2.0;
        let p = params();
        for _ in 0..2000 {
            let child = mutate_genotype(&root, rng, &p);
            let t = &child.traits;
            assert!(t.max_speed >= TRAIT_MIN.max_speed && t.max_speed <= TRAIT_MAX.max_speed);
            assert!(t.repr_cost >= TRAIT_MIN.repr_cost && t.repr_cost <= TRAIT_MAX.repr_cost);
            for v in child.net.w1.iter().chain(&child.net.b1).chain(&child.net.w2) {
                assert!(v.is_finite() && v.abs() <= WEIGHT_BOUND);
            }
            root = child;
        }
        // deep lineage must have drifted apart from the fresh root
        let fresh = random_genotype(rng);
        assert!(genotype_distance(&root, &fresh) > SPECIES_DISTANCE_THRESHOLD);
    }

    #[test]
    fn species_assignment_stable_then_new() {
        let book = &mut species_book_new();
        let rng = &mut Rng::new(7);
        let a = random_genotype(rng);
        let (id_a, new_a) = assign_species(book, &a, 0);
        assert_eq!(id_a, 1);
        assert!(new_a);
        // near parent: stays
        let (id_b, new_b) = assign_species(book, &a, id_a);
        assert_eq!((id_b, new_b), (id_a, false));
        // uncorrelated genome: founders a new species
        let c = random_genotype(rng);
        let (id_c, new_c) = assign_species(book, &c, id_b);
        assert!(new_c);
        assert!(id_c > id_b);
        let stats = species_stats(book);
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn species_ids_deterministic() {
        let run = |seed: u32| {
            let book = &mut species_book_new();
            let rng = &mut Rng::new(seed);
            let mut ids = Vec::new();
            let mut lineage = 0u32;
            for _ in 0..64 {
                let g = random_genotype(rng);
                let (id, _) = assign_species(book, &g, lineage);
                lineage = id;
                ids.push(id);
            }
            ids
        };
        assert_eq!(run(99), run(99));
    }

    #[test]
    fn think_zero_net_is_zero_and_fast_path_shapes() {
        let zero = Network::zeros();
        let mut out = [1.0f32; NET_OUTPUTS];
        think(&zero, &[0.0; NET_INPUTS], &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
        let mut out2 = [0.0f32; NET_OUTPUTS];
        let net = create_network(&mut Rng::new(3), 1.7);
        think(&net, &[0.5; NET_INPUTS], &mut out2);
        assert!(out2.iter().all(|&v| v.is_finite() && v.abs() <= 1.0));
    }
}
