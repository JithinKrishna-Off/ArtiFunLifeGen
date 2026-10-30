//! Procedural organism morphology. Maps expressed phenotype traits to
//! morphology brands and parameters. (M1 stub: brands + hue; the mesh
//! generator and membrane animation pipeline is Agent C's deliverable.)

use alife_common::{Organism, Traits};

/// Morphology signature classes (uint brand):
pub const BRAND_CELL: u32 = 0;
pub const BRAND_FLAGELLATE: u32 = 1;
pub const BRAND_COLONY: u32 = 2;
pub const BRAND_FILAMENT: u32 = 3;

/// Deterministic expressed-trait plumbing -> morphology brand. Pure function
/// of the organism's contemporary traits (no hashing, no randomness; the
/// renderer must never invent anatomy a trait table cannot support).
/// Fallback: filaments/colonies when lineages grow large/life stages long.
pub fn brand_of(o: &Organism) -> u32 {
    let t = &o.genotype.traits;
    if t.size < 0.85 {
        BRAND_CELL
    } else if t.max_speed > 65.0 {
        BRAND_FLAGELLATE
    } else if t.size > 1.25 {
        BRAND_FILAMENT
    } else {
        BRAND_CELL
    }
}

/// Golden-angle species hue (matches the 2D renderer palette walk).
pub fn hue_of(species: u32) -> f32 {
    ((species as f64 * 0.618_033_988_749_894_9) % 1.0) as f32
}

/// Trait -> visual shape knobs the agents use to build meshes.
pub struct MorphoParam {
    pub body_radius: f32,
    pub limb_count: u32,
    pub length_factor: f32,
}

pub fn morpho_param(t: &Traits) -> MorphoParam {
    MorphoParam {
        body_radius: 0.5 + 0.5 * t.size.clamp(0.3, 1.8) / 1.6,
        limb_count: (2.0 * t.sensor_range / 100.0).clamp(0.0, 6.0) as u32,
        length_factor: 0.6 + 0.8 * (t.metabolism / 1.9).clamp(0.2, 1.2),
    }
}

pub struct Morph {
    _p: (),
}

impl Morph {
    pub fn new(_device: &wgpu::Device) -> Option<Morph> {
        Some(Morph { _p: () })
    }

    /// Empty draw placeholder until Agent C lands the morph pipeline.
    pub fn draw_with<'a>(&self, _pass: &mut wgpu::RenderPass<'a>, _frame: Option<&crate::scene::Frame3D>) {}
}
