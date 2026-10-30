//! Renderer-side fading position trails for tracked organisms.
//! Bounded ring buffers with global point cap for graceful degradation.

use std::collections::HashMap;

const TRAIL_POINTS_PER_ORG: usize = 20;
const TRAIL_TOTAL_CAP: usize = 4000;
/// Full retain sweeps this often (frames-ish) even when the population-count
/// heuristic would not trigger: guarantees dead ids expire deterministically.
const PRUNE_EVERY: u64 = 128;

#[derive(Default)]
pub struct Trails {
    map: HashMap<u64, std::collections::VecDeque<(f32, f32)>>,
    total: usize,
    enabled: bool,
    track_calls: u64,
}

impl Trails {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            total: 0,
            enabled: false,
            track_calls: 0,
        }
    }

    pub fn set_enabled(&mut self, on: bool) {
        if on == self.enabled {
            return;
        }
        self.enabled = on;
        if !on {
            self.clear();
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Record latest organism positions (call once per frame while running).
    pub fn track(&mut self, organisms: &[alife_common::Organism]) {
        if !self.enabled {
            return;
        }
        for o in organisms {
            let q = self.map.entry(o.id).or_default();
            q.push_back((o.x, o.y));
            if q.len() > TRAIL_POINTS_PER_ORG {
                q.pop_front();
            } else {
                self.total += 1;
            }
        }
        // enforce global cap: drop oldest points round-robin
        while self.total > TRAIL_TOTAL_CAP && !self.map.is_empty() {
            let mut any = false;
            for q in self.map.values_mut() {
                if !q.is_empty() {
                    q.pop_front();
                    self.total -= 1;
                    any = true;
                    if self.total <= TRAIL_TOTAL_CAP {
                        break;
                    }
                }
            }
            if !any {
                break;
            }
        }
        // prune organisms no longer present (checked on frame boundaries)
        if organisms.len() < self.map.len() {
            let alive: std::collections::HashSet<u64> =
                organisms.iter().map(|o| o.id).collect();
            self.map.retain(|k, _| alive.contains(k));
        }
        // deterministic expiry: even when population size stays constant with
        // id churn, sweep periodically so dead ids never linger
        self.track_calls = self.track_calls.wrapping_add(1);
        if self.track_calls % PRUNE_EVERY == 0 {
            let alive: std::collections::HashSet<u64> =
                organisms.iter().map(|o| o.id).collect();
            self.map.retain(|k, _| alive.contains(k));
            self.total = self.map.values().map(|q| q.len()).sum();
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.total = 0;
    }

    pub fn points_of(&self, id: u64) -> Option<&std::collections::VecDeque<(f32, f32)>> {
        if !self.enabled {
            return None;
        }
        self.map.get(&id)
    }
}
