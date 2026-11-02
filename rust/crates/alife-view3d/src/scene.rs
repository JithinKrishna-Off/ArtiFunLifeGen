//! Per-frame rendering snapshot + camera input. Built by the UI layer every
//! frame; the renderer treats it as immutable. Owned by the coordinator
//! (fields may grow, but breaking changes must be coordinated).

#[path = "selection_markers.rs"]
pub mod markers;

use alife_common::Organism;
use alife_worldgen::WorldModel;
use std::sync::Arc;

/// Compact per-organism render instance. Derived from the real organism by
/// the UI via `morph` (a signature index, not raw genome).
///
/// Pod byte layout (44 bytes total; all fields are 4-byte scalars in
/// declaration order; the size is already 8-byte aligned): `is_queen` is a
/// `u32` flag (1 = queen, 0 = plain) so the WGSL path sees a clean f32/u32
/// float compare; it is consumed by the Queen morph (AGENT: morph_queen) to
/// route the flagged instance to the reserved queen visual, and is ignored
/// by every other visual pipeline. `visual_radius_mul` is render/pick-side
/// only (no GPU stride reads it).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OrgInst {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    /// brand from morphology signature: 0=cell 1=flagellate 2=colony
    /// 3=filament 4=queen (override when `is_queen` is set)
    pub brand: u32,
    /// species hue in [0, 1)
    pub hue: f32,
    /// normalized energy 0..1
    pub energy: f32,
    /// normalized age 0..1
    pub age: f32,
    /// expressed trait size (body scale factor; visual snapshot only —
    /// the simulation's collision radius is never touched by the queen flag)
    pub size: f32,
    /// 1.0 while alive (0 only when a corpse passes the brush briefly)
    pub alive: f32,
    /// 1 = the queen (appointed by the sim, not inferred here). Visual-only
    pub is_queen: u32,
    /// Per-candidate *visual* radius override for picking + markers:
    /// multiplies the organism's rendered body radius (ordinary base
    /// [`crate::morph::decomp::ORGANISM_UNIT_RADIUS`] * size * turgor).
    /// Default/ordinary = 1.0 (the `OrgInst::zeroed` 0.0 is treated as 1.0
    /// everywhere); the queen carries
    /// [`crate::morph::decomp::QUEEN_SCALE`] (~= 0.4435, mirroring the 5.5
    /// legacy base: 5.5 * QUEEN_SCALE == QUEEN_UNIT_RADIUS) so picking and
    /// markers see her true visible crown-to-body scale instead of the
    /// plain `size` clamp. Purely visual: the sim never reads this.
    pub visual_radius_mul: f32,
}

/// One organism -> render instance (read-only view of the simulation).
/// `is_queen` comes from the app's own queen tracking (e.g. the appointed
/// organism id), NOT from traits: the renderer must never guess royalty.
/// Queens also carry the QUEEN_SCALE visual-radius multiplier so picking and
/// selection markers widen around her actual rendered silhouette.
pub fn org_instance(o: &Organism, brand: u32, hue: f32, is_queen: bool) -> OrgInst {
    let t = &o.genotype.traits;
    OrgInst {
        x: o.x,
        y: o.y,
        heading: o.dir,
        // queen flag overrides the trait-derived brand: royalty renders as
        // the reserved queen template regardless of body class
        brand: if is_queen { crate::morph::BRAND_QUEEN } else { brand },
        hue,
        energy: (o.energy / o.max_energy).clamp(0.0, 1.0),
        age: (o.age as f32 / t.lifespan.max(1) as f32).clamp(0.0, 1.0),
        size: t.size.clamp(0.3, 2.0),
        alive: if o.alive { 1.0 } else { 0.0 },
        is_queen: if is_queen { 1 } else { 0 },
        visual_radius_mul: if is_queen {
            crate::morph::decomp::QUEEN_SCALE
        } else {
            1.0
        },
    }
}

/// Convenience for callers that do not track a queen (non-queen callers =
/// every plain organism). Keeps the old two-flag call shape intact.
pub fn org_instance_default(o: &Organism, brand: u32, hue: f32) -> OrgInst {
    org_instance(o, brand, hue, false)
}

/// Index of the first queen instance in a snapshot, if the app flagged one.
pub fn find_queen(orgs: &[OrgInst]) -> Option<usize> {
    orgs.iter().position(|o| o.is_queen != 0)
}

/// Camera operation requested by the user this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraMode {
    /// free orbit around a world point (habitat view)
    Orbit { focus: (f32, f32) },
    /// locked follow of one organism id
    Follow { id: u64 },
    /// high overview over a region
    Planet,
}

/// Mouse/keys produced by the UI for this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct CamInput {
    pub drag: (f32, f32),
    pub wheel: f32,
    pub back: bool,
    pub forward: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub reset_pressed: bool,
    /// Queen POV alpha channel, streamed per frame by the UI while the POV
    /// sub-mode is engaged: `(x, z, heading, body_len)`. While `Some`, the
    /// camera overrides the final eye/look with the POV gaze (orbit drag is
    /// ignored) — `cam_mode` stays `Orbit { focus = queen pos }` so every
    /// grass/morph/terrain consumer keeps its existing window math (see
    /// `camera::CameraState::set_queen_pov`). `None` holds the last look; all
    /// four values must be finite (non-finite are rejected per-frame).
    pub pov_target: Option<(f32, f32, f32, f32)>,
}

impl CamInput {
    pub fn clear(&mut self) { *self = CamInput::default(); }
}

/// Everything the renderer needs for one frame.
pub struct Frame3D {
    /// Final camera matrices for THIS frame (must be computed by
    /// `camera::CameraState::update` + `matrices()` before building).
    pub cam: CamUniform,
    /// world fields (read-only shared model; the renderer never mutates it)
    pub world: Arc<WorldModel>,
    pub cam_mode: CameraMode,
    pub dt: f32,
    pub time: f32,
    pub viewport_px: (f32, f32),
    pub follow_target: Option<(f32, f32)>,
    pub orgs: Vec<OrgInst>,
    /// atmosphere from the real weather layer: (0..1 coverage values)
    pub rain_cover: f32,
    pub cloud_cover: f32,
    pub flash: f32,
}

impl Frame3D {
    /// Rough vector of organisms (bounded): degenerates to small population.
    /// No queen tracking here — plain `org_instance_default` for all; the
    /// coordinator passes `is_queen` at its own build site.
    pub fn build(orgs: &[Organism], brand_of: impl Fn(&Organism) -> u32, hue_of: impl Fn(&Organism) -> f32) -> Vec<OrgInst> {
        orgs.iter().take(1600).map(|o| org_instance_default(o, brand_of(o), hue_of(o))).collect()
    }
}

/// Combined camera + projection block for the shader (Pod-friendly layout;
/// glam matrices are converted with `to_cols_array_2d()`).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CamUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_pos: [f32; 4],
    /// (world_w, world_h, unif_TIME, lightning flash)
    pub world_wh_time_flash: [f32; 4],
}
