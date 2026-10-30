//! Environment editor operation log. Every applied editor action may push one
//! entry describing how to revert it; `undo_edit` pops and applies entries
//! (multi-level, bounded stack).

use alife_common::{Food, FoodPatch, HazardZone, InteractionMode, Obstacle};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditEntry {
    pub mode: InteractionMode,
    /// brush anchor point (for reference / later analysis)
    pub x: f32,
    pub y: f32,
    pub r: f32,
    /// foods added by this op (revert = remove by id)
    pub added_food_ids: Vec<u64>,
    /// the food objects added by this op (redo = re-insert; `#[serde(default)]`
    /// keeps pre-existing saves loading: older entries lose food redo only)
    #[serde(default)]
    pub added_foods: Vec<Food>,
    pub added_obstacles: Vec<Obstacle>,
    pub added_hazards: Vec<HazardZone>,
    pub added_patches: Vec<FoodPatch>,
    /// prior objects removed by this op (revert = re-insert)
    pub removed_obstacles: Vec<Obstacle>,
    pub removed_hazards: Vec<HazardZone>,
    pub removed_patches: Vec<FoodPatch>,
    pub removed_foods: Vec<Food>,
}

impl Default for EditEntry {
    fn default() -> Self {
        Self {
            mode: InteractionMode::Inspect,
            x: 0.0,
            y: 0.0,
            r: 0.0,
            added_food_ids: Vec::new(),
            added_foods: Vec::new(),
            added_obstacles: Vec::new(),
            added_hazards: Vec::new(),
            added_patches: Vec::new(),
            removed_obstacles: Vec::new(),
            removed_hazards: Vec::new(),
            removed_patches: Vec::new(),
            removed_foods: Vec::new(),
        }
    }
}
