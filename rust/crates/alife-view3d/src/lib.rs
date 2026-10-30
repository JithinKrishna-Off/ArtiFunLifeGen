//! alife-view3d -- in-process wgpu 3D renderer for the Artificial Life
//! desktop app. Read-only consumer of simulation state: the Rust engine owns
//! organisms, genomes, weather and ecology; this crate only renders snapshots
//! handed to it every frame by the UI layer.
//!
//! Architecture:
//! - One persistent [`View3D`] per app holding the wgpu pipelines and caches
//!   (no per-frame GPU resource churn); registered once in egui's
//!   `CallbackResources` via `render::install`.
//! - The UI builds a compact [`Frame3D`] snapshot each frame (organisms,
//!   camera matrices, selection) -- never the raw simulation structures.
//! - Rendering happens inside an egui paint callback so the 3D viewport
//!   lives inside the normal egui canvas; eframe must run `Renderer::Wgpu`.
//!
//! Module ownership (do not cross-edit):
//! - `terrain.rs`  terrain window mesh, LOD, caching       (Agent A)
//! - `grass.rs`    vegetation instancing, wind, bending    (Agent B)
//! - `morph.rs`    procedural organism morphology          (Agent C)
//! - `camera.rs`   multi-scale camera + transitions        (Agent D)
//! - `render.rs`   pipelines + callback glue               (coordinator)
//! - `scene.rs`    snapshot types                          (coordinator)

pub mod camera;
pub mod grass;
pub mod morph;
pub mod render;
pub mod scene;
pub mod terrain;

use std::sync::Arc;

pub use render::{enqueue, install, View3DCallback};

/// Long-lived 3D renderer state; stored inside egui's CallbackResources so
/// callbacks can reach it. Falls back gracefully to 2D-only when the wgpu
/// device is not usable.
pub struct View3D {
    pub terrain: terrain::Terrain,
    pub grass: grass::Grass,
    pub morph: morph::Morph,
    /// the most recent per-frame snapshot (set in `prepare`, read in `paint`)
    pub frame_of: std::sync::Mutex<Option<Arc<scene::Frame3D>>>,
}

impl View3D {
    /// Creates all GPU resources; None only if something fundamental fails.
    pub fn new(rs: &egui_wgpu::RenderState) -> Option<View3D> {
        let terrain = terrain::Terrain::new(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float)?;
        let mut grass = grass::Grass::new(&rs.device)?;
        grass.ensure_pipeline(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float);
        let morph = morph::Morph::new(&rs.device)?;
        Some(View3D { terrain, grass, morph, frame_of: Default::default() })
    }
}
