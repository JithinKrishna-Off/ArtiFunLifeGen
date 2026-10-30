//! Bounded history buffer: aggregate `interval` steps per sample, then
//! downsampling merges adjacent pair samples (interval doubles) so charts stay
//! bounded (default max 2048 samples).

use alife_common::{
    persist::{HistoryBuffer, HistorySample},
    SimStats, StepEvents,
};

const DEFAULT_INTERVAL: u32 = 20;

fn top_species(counts: &[alife_common::SpeciesCount], n: usize) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = counts.iter().map(|c| (c.species_id, c.count)).collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(n);
    v
}

/// Count heap-style downsample of two adjacent samples.
fn merge_pair(a: &HistorySample, b: &HistorySample) -> HistorySample {
    let avg2 = |x: f32, y: f32| (x + y) * 0.5;
    // species_top: keep the larger count seen for each species across the pair
    let mut merged: Vec<(u32, u32)> = a.species_top.clone();
    for &(id, count) in &b.species_top {
        if let Some(slot) = merged.iter_mut().find(|(sid, _)| *sid == id) {
            slot.1 = slot.1.max(count);
        } else {
            merged.push((id, count));
        }
    }
    merged.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    merged.truncate(5);
    HistorySample {
        step: a.step,
        population: ((a.population as u64 + b.population as u64) / 2) as u32,
        births: a.births + b.births,
        deaths: a.deaths + b.deaths,
        avg_energy: avg2(a.avg_energy, b.avg_energy),
        avg_age: avg2(a.avg_age, b.avg_age),
        avg_generation: avg2(a.avg_generation, b.avg_generation),
        food_count: ((a.food_count as u64 + b.food_count as u64) / 2) as u32,
        species_top: merged,
        diversity: avg2(a.diversity, b.diversity),
        wet_cover: avg2(a.wet_cover, b.wet_cover),
        merged: true,
    }
}

/// Merge all adjacent pairs; buffer length roughly halves, interval doubles.
/// Done in place: merged samples overwrite the buffer front, the tail is
/// truncated — no full-size temporary Vec per downsample.
fn downsample(h: &mut HistoryBuffer) {
    let mut w = 0usize;
    let mut i = 0usize;
    let len = h.samples.len();
    while i < len {
        let merged = if i + 1 < len {
            merge_pair(&h.samples[i], &h.samples[i + 1])
        } else {
            h.samples[i].clone() // odd tail sample is kept as-is
        };
        h.samples[w] = merged;
        w += 1;
        i += 2;
    }
    h.samples.truncate(w);
    h.interval = h.interval.saturating_mul(2).max(1);
}

pub fn step_history(h: &mut HistoryBuffer, stats: &SimStats, events: &StepEvents) {
    if h.max == 0 {
        h.max = 2048;
    }
    if h.interval == 0 {
        h.interval = DEFAULT_INTERVAL;
    }

    let births = events.births.len() as u32;
    let deaths = events.deaths.len() as u32;
    h.pending_births = h.pending_births.saturating_add(births);
    h.pending_deaths = h.pending_deaths.saturating_add(deaths);

    // Emit when `stats.step` crosses a multiple of the current interval.
    if stats.step == 0 || !stats.step.is_multiple_of(h.interval as u64) {
        return;
    }

    let sample = HistorySample {
        step: stats.step,
        population: stats.population,
        births: h.pending_births,
        deaths: h.pending_deaths,
        avg_energy: stats.avg_energy,
        avg_age: stats.avg_age,
        avg_generation: stats.avg_generation,
        food_count: stats.food_count,
        species_top: top_species(&stats.species_counts, 5),
        diversity: stats.diversity_index,
        wet_cover: stats.wet_cover,
        merged: false,
    };
    h.pending_births = 0;
    h.pending_deaths = 0;

    if h.samples.len() >= h.max {
        downsample(h);
    }
    h.samples.push(sample);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::SimStats;

    fn stats(step: u64) -> SimStats {
        SimStats {
            step,
            population: 100,
            avg_energy: 50.0,
            avg_age: 10.0,
            avg_generation: 1.0,
            food_count: 200,
            diversity_index: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn interval_sampling_and_downsample() {
        let mut h = HistoryBuffer::new(20, 4);
        let ev = StepEvents::default();
        for step in 1..=200u64 {
            step_history(&mut h, &stats(step), &ev);
        }
        assert!(h.samples.len() <= 4);
        assert_eq!(h.interval, 80);
        // after interval doubled to 80, emission points are multiples of 80
        assert_eq!(h.samples.last().unwrap().step, 160);
        step_history(&mut h, &stats(240), &ev);
        assert_eq!(h.samples.last().unwrap().step, 240);
        assert_eq!(h.pending_births, 0);
    }

    #[test]
    fn pending_counters_reset_per_sample() {
        let mut h = HistoryBuffer::new(20, 2048);
        step_history(&mut h, &stats(20), &StepEvents::default());
        assert_eq!(h.pending_births, 0);
        assert_eq!(h.pending_deaths, 0);
    }
}
