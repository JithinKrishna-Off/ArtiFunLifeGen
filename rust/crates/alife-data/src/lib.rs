//! alife-data — persistence (WorldSave IO) + analytics (bounded history,
//! presets, CSV export, error log). Public signatures match the stub API
//! contract; bodies are real implementations.

use alife_common::{
    persist::{HistoryBuffer, WorldSave},
    EnvironmentEdits, Food, LineageRecord, Organism, SimParams, WorldGenConfig,
};

mod history;
mod log;
mod presets;
mod store;

pub use log::log_error;
pub use presets::{apply_preset, preset_list};
pub use store::{DataStore, SaveSummary};

/// Save format version written by this crate. The canonical contract lives
/// in `alife-common::persist` (`FORMAT_VERSION`); format bump request for
/// the coordinator: bump `persist::FORMAT_VERSION` 1 => 2.
///
/// v1 => v2 differences:
/// - v2 files carry an optional top-level `core` field: the engine's
///   `SaveCore` blob serialized to JSON (weather state, interventions and
///   everything alife-core needs for exact restoration).
/// - v1 files (`core` absent) still load forever; restore is best-effort.
pub const SAVE_FORMAT_VERSION: u32 = 2;

/// Construct a full v1-style save snapshot (no engine blob); all inputs are
/// copied verbatim. Signature kept stable — `build_save_v2` is the v2 writer.
#[allow(clippy::too_many_arguments)]
pub fn build_save(
    name: &str,
    preset: Option<String>,
    params: &SimParams,
    world_config: &WorldGenConfig,
    step_count: u64,
    organisms: &[Organism],
    foods: &[Food],
    edits: &EnvironmentEdits,
    rng_state: u32,
    records: Vec<LineageRecord>,
    history: &HistoryBuffer,
) -> WorldSave {
    world_save(
        name,
        preset,
        params,
        world_config,
        step_count,
        organisms,
        foods,
        edits,
        rng_state,
        records,
        history,
        None,
    )
}

/// Construct a full v2 save snapshot with the engine `SaveCore` blob
/// (serialized to JSON) embedded. `SaveCore` is opaque from the data layer's
/// perspective: anything serializable is embedded verbatim. If serialization
/// fails the save degrades to a `core: None` v2 file (best-effort restore).
#[allow(clippy::too_many_arguments)]
pub fn build_save_v2<S: serde::Serialize>(
    name: &str,
    preset: Option<String>,
    params: &SimParams,
    world_config: &WorldGenConfig,
    step_count: u64,
    organisms: &[Organism],
    foods: &[Food],
    edits: &EnvironmentEdits,
    rng_state: u32,
    records: Vec<LineageRecord>,
    history: &HistoryBuffer,
    core: &S,
) -> WorldSave {
    let core_blob = serde_json::to_value(core)
        .map_err(|e| format!("cannot serialize engine core blob for save '{name}': {e}"))
        .ok();
    world_save(
        name,
        preset,
        params,
        world_config,
        step_count,
        organisms,
        foods,
        edits,
        rng_state,
        records,
        history,
        core_blob,
    )
}

#[allow(clippy::too_many_arguments)]
fn world_save(
    name: &str,
    preset: Option<String>,
    params: &SimParams,
    world_config: &WorldGenConfig,
    step_count: u64,
    organisms: &[Organism],
    foods: &[Food],
    edits: &EnvironmentEdits,
    rng_state: u32,
    records: Vec<LineageRecord>,
    history: &HistoryBuffer,
    core: Option<serde_json::Value>,
) -> WorldSave {
    WorldSave {
        format_id: alife_common::persist::FORMAT_ID.to_string(),
        version: SAVE_FORMAT_VERSION,
        name: name.to_string(),
        preset,
        params: params.clone(),
        world_config: *world_config,
        step_count,
        organisms: organisms.to_vec(),
        foods: foods.to_vec(),
        edits: edits.clone(),
        rng_state,
        records,
        history: history.clone(),
        core,
    }
}

/// Extract the engine `SaveCore` JSON blob from a loaded save, if it carried
/// one (v2 saves only; v1 saves return None => best-effort restore).
pub fn core_blob(ws: &WorldSave) -> Option<&serde_json::Value> {
    ws.core.as_ref()
}

/// Push one step into the history buffer, aggregating `interval` steps per
/// sample and downsampling by pair-merge when bounded.
pub fn step_history(h: &mut HistoryBuffer, stats: &alife_common::SimStats, events: &alife_common::StepEvents) {
    crate::history::step_history(h, stats, events)
}
