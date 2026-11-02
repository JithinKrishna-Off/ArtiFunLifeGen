//! Save/load plumbing, preset fallbacks and lineage record bookkeeping.
//! Isolated here so engine-contract reconstruction (Simulation::from_save)
//! lives in exactly one file.

use alife_common::persist::{HistoryBuffer, WorldSave, FORMAT_ID, FORMAT_VERSION};
use alife_common::{presets::Preset, LineageRecord, SimParams, WorldGenConfig};
use alife_core::queen::QueenSnapshot;
use alife_core::{SaveCore, SimApi, Simulation};
use std::collections::HashMap;
use std::path::Path;

/// Fallback presets so the app ships real content even while alife-data is a
/// stub; merged with (and shadowed by) `alife_data::preset_list()`.
pub fn fallback_presets() -> Vec<Preset> {
    vec![
        Preset::from("genesis", "Genesis", "Balanced baseline world"),
        Preset::from("petri", "Petri Dish", "Dense nutrients, crowded start")
            .with_world(|w| {
                w.width = 1200;
                w.height = 900;
                w.resource_density = 1.6;
                w.hazard_amount = 0.15;
            })
            .with_sim(|s| {
                s.initial_population = 420;
                s.max_population = 4000;
                s.initial_food = 1400;
                s.max_food = 2600;
                s.food_regen_per_sec = 52.0;
            }),
        Preset::from("harsh", "Harsh Frontier", "Low food, strong hazards")
            .with_world(|w| {
                w.hazard_amount = 1.1;
                w.fertility_contrast = 1.7;
                w.resource_density = 0.55;
            })
            .with_sim(|s| {
                s.initial_population = 90;
                s.max_food = 600;
                s.food_regen_per_sec = 9.0;
                s.metabolism_base = 0.8;
            }),
        Preset::from("mutation", "Mutation Storm", "High mutation pressure")
            .with_world(|w| {
                w.warp = 0.7;
                w.detail = 5;
            })
            .with_sim(|s| {
                s.mutation_rate = 0.3;
                s.mutation_magnitude = 2.4;
                s.big_jump_chance = 0.12;
                s.initial_population = 300;
            }),
        Preset::from("longevity", "Longevity Lab", "Slow generations, long lives")
            .with_world(|w| {
                w.terrain_scale = 1.8;
                w.moisture_scale = 1.2;
            })
            .with_sim(|s| {
                s.reproduction_cooldown_steps = 480;
                s.min_repro_age_steps = 1200;
                s.mutation_rate = 0.02;
                s.mutation_magnitude = 0.4;
                s.initial_population = 160;
            }),
        Preset::from("explorer", "Wide Worlds", "Large landscape, spread out")
            .with_world(|w| {
                w.width = 4800;
                w.height = 3200;
                w.cell = 10;
                w.terrain_scale = 2.4;
            })
            .with_sim(|s| {
                s.initial_population = 300;
                s.max_population = 2000;
                s.initial_food = 1800;
                s.max_food = 3400;
                s.food_regen_per_sec = 70.0;
            }),
    ]
}

/// Merged preset catalog (data crate wins over fallback per id).
pub fn preset_catalog() -> Vec<Preset> {
    let mut out: Vec<Preset> = alife_data::preset_list().to_vec();
    for p in fallback_presets() {
        if !out.iter().any(|x| x.id == p.id) {
            out.push(p);
        }
    }
    out.sort_by_key(|p| p.id);
    out
}

/// Resolve a preset through the data crate with a local fallback.
pub fn apply_preset_with_fallback(id: &str) -> (WorldGenConfig, SimParams) {
    match alife_data::apply_preset(id) {
        Ok(pair) => pair,
        Err(_) => {
            let cat = preset_catalog();
            if let Some(p) = cat.iter().find(|p| p.id == id) {
                (p.world.clone(), p.sim.clone())
            } else {
                (WorldGenConfig::default(), SimParams::default())
            }
        }
    }
}

/// Ensure the lineage bookkeeping map carries records for all live organisms.
pub fn collect_records(seen: &mut HashMap<u64, LineageRecord>, sim: &Simulation) {
    if seen.len() > 5usize << 5 {
        seen.clear();
    }
    for o in sim.organisms() {
        if !seen.contains_key(&o.id) {
            if let Some(r) = sim.get_record(o.id) {
                seen.insert(o.id, r);
            }
        }
    }
}

/// Assemble a WorldSave from live state. Uses the data crate's builder; if
/// the crate is still a stub (fills nothing), the fields are written inline.
#[allow(clippy::too_many_arguments)]
pub fn assemble_save(
    name: &str,
    preset: Option<String>,
    params: &SimParams,
    cfg: &WorldGenConfig,
    sim: &Simulation,
    records: &[LineageRecord],
    history: &HistoryBuffer,
) -> WorldSave {
    let records_vec = records.to_vec();
    let core = sim.snapshot();
    let mut ws = alife_data::build_save_v2(
        name,
        preset.clone(),
        params,
        cfg,
        sim.step_count(),
        sim.organisms(),
        sim.foods(),
        sim.edits(),
        sim.rng_state(),
        records_vec,
        history,
        &core,
    );
    if ws.name != name || ws.format_id != FORMAT_ID || ws.core.is_none() {
        ws.name = name.to_string();
        ws.preset = preset;
        ws.params = params.clone();
        ws.world_config = *cfg;
        ws.step_count = sim.step_count();
        ws.organisms = sim.organisms().to_vec();
        ws.foods = sim.foods().to_vec();
        ws.edits = sim.edits().clone();
        ws.rng_state = sim.rng_state();
        ws.history = history.clone();
    }
    ws
}

/// What a successful load hands back to the app shell: the restored
/// simulation plus the queen-reign payload (v2 `SaveCore.queen`) so the
/// crown-once flags and (where possible) the reign itself can be restored.
pub struct RestoredSave {
    pub sim: Simulation,
    /// parse of `SaveCore.queen` (None = no reign in the save / v1 file).
    pub queen_saved: Option<QueenSnapshot>,
}

impl RestoredSave {
    /// The saved world already had a reign — alive queen (with a restorable
    /// brain) or a dead one. Both block any future auto-crown/respawn.
    pub fn queen_was_crowned(&self) -> bool {
        self.queen_saved
            .as_ref()
            .is_some_and(|q| q.alive || q.death_cause.is_some())
    }
}

/// Rebuild a Simulation from a loaded WorldSave. Prefers the v2 core blob
/// (`SaveCore`, exact engine restore via `Simulation::from_save_core`); v1
/// files (and a corrupt blob) fall back to the contracted individual-field
/// restore path (`Simulation::from_save`) — unchanged behavior.
pub fn restore_from_save(ws: &WorldSave) -> Result<RestoredSave, String> {
    if ws.format_id != FORMAT_ID {
        return Err(format!("not a save file (format \"{}\")", ws.format_id));
    }
    if ws.version < 1 || ws.version > FORMAT_VERSION {
        return Err(format!(
            "unsupported save version {} (supported: 1..={})",
            ws.version, FORMAT_VERSION
        ));
    }
    let world = alife_worldgen::generate(&ws.world_config);
    if let Some(core_blob) = alife_data::core_blob(ws) {
        if !core_blob.is_null() {
            if let Ok(core) = serde_json::from_value::<SaveCore>(core_blob.clone()) {
                return Ok(RestoredSave {
                    sim: Simulation::from_save_core(&ws.params, world, &core),
                    queen_saved: core.queen,
                });
            }
            // corrupt blob: v1 best-effort restore below (honest fallback)
        }
    }
    Ok(RestoredSave {
        sim: Simulation::from_save(
            &ws.params,
            world,
            ws.step_count,
            ws.organisms.clone(),
            ws.foods.clone(),
            ws.edits.clone(),
            ws.rng_state,
            ws.records.clone(),
        ),
        queen_saved: None,
    })
}

/// Export experiment config JSON into `<root>/experiments/<name>.json`.
pub fn export_config_json(
    store_root: &Path,
    name: &str,
    preset: &str,
    params: &SimParams,
    cfg: &WorldGenConfig,
) -> Result<String, String> {
    let dir = store_root.join("experiments");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let safe = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>();
    let file = dir.join(format!("{}.json", safe));
    let doc = serde_json::json!({
        "preset": preset,
        "sim_params": params,
        "world_config": cfg,
    });
    std::fs::write(
        &file,
        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(file.display().to_string())
}
