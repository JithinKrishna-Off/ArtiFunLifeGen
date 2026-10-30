//! The ten experiment presets (Genesis is index 0, Primordial index 1). Each
//! preset pairs a `WorldGenConfig` with tuned `SimParams`. NOTE: `alife-common`'s
//! `WorldGenConfig` derives a zero-default; `base_world` reproduces the
//! engine's own sensible defaults (seed, 2400x1600, cell 8) so every preset
//! starts from a coherent world, then applies its deltas.

use alife_common::{presets::Preset, SimParams, WorldGenConfig};
use std::sync::OnceLock;

/// Engine-equivalent world baseline (matches alife-common's Default impl).
fn base_world() -> WorldGenConfig {
    WorldGenConfig::default()
}

fn presets() -> &'static [Preset] {
    static LOCK: OnceLock<Vec<Preset>> = OnceLock::new();
    LOCK.get_or_init(|| {
        vec![
            // 0 — Genesis: the balanced default ecosystem.
            Preset::from("genesis", "Genesis", "Balanced default ecosystem — immediately interesting on fresh worlds.")
                .with_world(|w| *w = base_world()),
            // 1 — Primordial: one cell in a young, broadly fertile ocean.
            Preset::from("primordial", "Primordial Ocean", "Start with a single primordial cell in a young fertile ocean — watch life diversify.")
                .with_world(|w| {
                    *w = base_world();
                    w.fertility_contrast = 0.35;
                    w.resource_density = 1.3;
                    w.hazard_amount = 0.05;
                    w.terrain_scale = 0.7;
                    w.detail = 3;
                    w.warp = 0.25;
                })
                .with_sim(|s| {
                    s.seed = 20260;
                    s.initial_population = 1;
                    s.initial_food = 700;
                    s.food_regen_per_sec = 22.0;
                    s.max_population = 1500;
                }),
            // 1 — Eden: lush, gentle, room to grow big populations.
            Preset::from("eden", "Eden", "Lush gardens with rare hazard zones — easy living, big populations.")
                .with_world(|w| {
                    *w = base_world();
                    w.fertility_contrast = 0.6;
                    w.resource_density = 1.6;
                    w.hazard_amount = 0.08;
                })
                .with_sim(|s| {
                    s.food_regen_per_sec = 40.0;
                    s.max_population = 1500;
                }),
            // 2 — Arid: scarce food, dangerous sunbelts, punishing metabolism.
            Preset::from("arid", "Arid Expanse", "Scarce food and hostile hazards — evolution under starvation.")
                .with_world(|w| {
                    *w = base_world();
                    w.resource_density = 0.45;
                    w.hazard_amount = 0.15;
                })
                .with_sim(|s| {
                    s.food_regen_per_sec = 14.0;
                    s.max_population = 700;
                    s.metabolism_base = 0.7;
                }),
            // 3 — Fractured: large broken terrain isolates populations.
            Preset::from("fractured", "Fractured World", "Large-scale broken terrain that fragments populations into niches.")
                .with_world(|w| {
                    *w = base_world();
                    w.terrain_scale = 1.6;
                    w.detail = 6;
                    w.warp = 0.55;
                    w.cell = 6;
                }),
            // 4 — Toxic: deadly pulsing hazards carve hard selection pressure.
            Preset::from("toxic", "Toxic Frontier", "Pulsing toxic zones nearly everywhere — survival of the lucky and the fast.")
                .with_world(|w| {
                    *w = base_world();
                    w.hazard_amount = 0.9;
                    w.resource_density = 1.0;
                    w.hazard_pulse_period = 4.0;
                })
                .with_sim(|s| {
                    s.max_population = 900;
                    s.initial_population = 200;
                }),
            // 5 — Archipelago: island-like blobs via broad features + strong warp.
            Preset::from("archipelago", "Archipelago", "Island-like habitats linked by open water gaps — endemic evolution.")
                .with_world(|w| {
                    *w = base_world();
                    w.terrain_scale = 2.2;
                    w.moisture_scale = 1.2;
                    w.fertility_contrast = 1.4;
                    w.warp = 0.6;
                }),
            // 6 — Labyrinth: tight small-scale structure with heavy shaping.
            Preset::from("labyrinth", "Living Labyrinth", "Maze-like tight terrain with heavy shaping and verging hazards.")
                .with_world(|w| {
                    *w = base_world();
                    w.terrain_scale = 0.8;
                    w.detail = 5;
                    w.warp = 0.3;
                    w.hazard_amount = 0.3;
                }),
            // 7 — Bloom: extreme fertility contrast and hot mutation.
            Preset::from("bloom", "Chaotic Bloom", "Extreme fertility swings and hot mutation — species churn fast.")
                .with_world(|w| {
                    *w = base_world();
                    w.fertility_contrast = 2.0;
                    w.resource_density = 1.5;
                })
                .with_sim(|s| {
                    s.food_regen_per_sec = 45.0;
                    s.mutation_rate = 0.12;
                    s.mutation_magnitude = 1.6;
                }),
            // 8 — Blank: a near-flat sandbox for hand-built scenarios.
            Preset::from("blank", "Blank Canvas", "Near-flat empty sandbox — build the world by hand from zero.")
                .with_world(|w| {
                    *w = base_world();
                    w.fertility_contrast = 0.2;
                    w.resource_density = 0.02;
                    w.hazard_amount = 0.0;
                    w.cell = 16;
                })
                .with_sim(|s| {
                    s.initial_population = 0;
                    s.initial_food = 0;
                    s.max_population = 2000;
                }),
        ]
    })
    .as_slice()
}

/// All ten presets (Genesis is index 0, Primordial is index 1).
pub fn preset_list() -> &'static [Preset] {
    presets()
}

/// Match a preset id and return a copy of its world + sim configuration.
pub fn apply_preset(id: &str) -> Result<(WorldGenConfig, SimParams), String> {
    let list = presets();
    list.iter()
        .find(|p| p.id == id)
        .map(|p| (p.world, p.sim.clone()))
        .ok_or_else(|| {
            let ids: Vec<&str> = list.iter().map(|p| p.id).collect();
            format!("unknown preset id '{id}' — available: {}", ids.join(", "))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_presets_unique_ids() {
        let list = preset_list();
        assert_eq!(list.len(), 10);
        assert_eq!(list[0].id, "genesis");
        assert_eq!(list[1].id, "primordial");
        let mut ids: Vec<&str> = list.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n);
    }

    #[test]
    fn apply_matches_and_rejects() {
        for p in preset_list() {
            let (w, s) = apply_preset(p.id).unwrap();
            assert!(w.width > 0 && w.cell > 0, "{} has degenerate world config", p.id);
            match p.id {
                "blank" => {
                    assert_eq!(s.initial_population, 0);
                    assert_eq!(s.initial_food, 0);
                }
                "primordial" => {
                    assert_eq!(s.initial_population, 1);
                    assert_eq!(s.initial_food, 700);
                    assert!((s.food_regen_per_sec - 22.0).abs() < 1e-6);
                    assert_eq!(s.max_population, 1500);
                    assert_ne!(s.seed, SimParams::default().seed);
                    assert!((w.fertility_contrast - 0.35).abs() < 1e-6);
                    assert!((w.resource_density - 1.3).abs() < 1e-6);
                    assert!((w.hazard_amount - 0.05).abs() < 1e-6);
                    assert!((w.terrain_scale - 0.7).abs() < 1e-6);
                    assert_eq!(w.detail, 3);
                    assert!((w.warp - 0.25).abs() < 1e-6);
                }
                "eden" => {
                    assert!((w.resource_density - 1.6).abs() < 1e-6);
                    assert_eq!(s.max_population, 1500);
                }
                "arid" => assert!((s.metabolism_base - 0.7).abs() < 1e-6),
                "fractured" => assert_eq!(w.cell, 6),
                "toxic" => assert!((w.hazard_amount - 0.9).abs() < 1e-6),
                "archipelago" => assert!(w.terrain_scale > 2.0),
                "labyrinth" => assert!(w.hazard_amount > 0.2),
                "bloom" => assert!(s.mutation_rate > 0.1),
                _ => {}
            }
        }
        assert!(apply_preset("nope").is_err());
    }
}
