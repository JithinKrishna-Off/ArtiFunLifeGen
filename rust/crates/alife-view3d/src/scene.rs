//! Per-frame rendering snapshot + camera input. Built by the UI layer every
//! frame; the renderer treats it as immutable. Owned by the coordinator
//! (fields may grow, but breaking changes must be coordinated).

use alife_common::Organism;
use alife_worldgen::WorldModel;
use std::sync::Arc;

/// Compact per-organism render instance. Derived from the real organism by
/// the UI via `morph` (a signature index, not raw genome).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OrgInst {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    /// brand from morphology signature: 0=cell 1=flagellate 2=colony 3=filament
    pub brand: u32,
    /// species hue in [0, 1)
    pub hue: f32,
    /// normalized energy 0..1
    pub energy: f32,
    /// normalized age 0..1
    pub age: f32,
    /// expressed trait size (body scale factor)
    pub size: f32,
    /// 1.0 while alive (0 only when a corpse passes the brush briefly)
    pub alive: f32,
}

/// One organism -> render instance (read-only view of the simulation).
pub fn org_instance(o: &Organism, brand: u32, hue: f32) -> OrgInst {
    let t = &o.genotype.traits;
    OrgInst {
        x: o.x,
        y: o.y,
        heading: o.dir,
        brand,
        hue,
        energy: (o.energy / o.max_energy).clamp(0.0, 1.0),
        age: (o.age as f32 / t.lifespan.max(1) as f32).clamp(0.0, 1.0),
        size: t.size.clamp(0.3, 2.0),
        alive: if o.alive { 1.0 } else { 0.0 },
    }
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
    pub fn build(orgs: &[Organism], brand_of: impl Fn(&Organism) -> u32, hue_of: impl Fn(&Organism) -> f32) -> Vec<OrgInst> {
        orgs.iter().take(1600).map(|o| org_instance(o, brand_of(o), hue_of(o))).collect()
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
