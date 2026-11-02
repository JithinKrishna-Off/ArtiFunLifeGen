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
pub mod grass_clump;
pub mod morph;
pub mod render;
pub mod scene;
pub mod terrain;
pub mod tree;
// trees (Agent: trees)
pub mod tree_mesh;
pub mod tree_place;

use std::sync::Arc;

pub use render::{enqueue, install, View3DCallback};

/// Long-lived 3D renderer state; stored inside egui's CallbackResources so
/// callbacks can reach it. Falls back gracefully to 2D-only when the wgpu
/// device is not usable.
pub struct View3D {
    pub terrain: terrain::Terrain,
    pub grass: grass::Grass,
    pub morph: morph::Morph,
    /// trees + bushes (Agent: trees)
    pub tree: tree::Tree,
    /// the most recent per-frame snapshot (set in `prepare`, read in `paint`)
    pub frame_of: std::sync::Mutex<Option<Arc<scene::Frame3D>>>,
}

impl View3D {
    /// Creates all GPU resources; None only if something fundamental fails.
    pub fn new(rs: &egui_wgpu::RenderState) -> Option<View3D> {
        // wgpu validation errors must NEVER kill the app: log them honestly
        // (the offending subsystem is named by the per-draw debug groups) and
        // keep rendering; the aborted draw is skipped, the simulation lives.
        let crash_log = std::path::PathBuf::from("appdata").join("render.log");
        rs.device.on_uncaptured_error(Box::new(move |err| {
            let payload = format!("WGPU ERROR {err:?}\n");
            eprintln!("{payload}");
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&crash_log) {
                use std::io::Write as _;
                let _ = f.write_all(payload.as_bytes());
            }
        }));
        let terrain = terrain::Terrain::new(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float)?;
        let mut grass = grass::Grass::new(&rs.device)?;
        grass.ensure_pipeline(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float);
        // trees (Agent: trees) — instantiated after Grass, pipeline with the
        // same color/depth formats the terrain was created with
        let mut tree = tree::Tree::new(&rs.device)?;
        tree.ensure_pipeline(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float);
        let morph = morph::Morph::new(&rs.device)?;
        let mut morph = morph;
        morph.ensure_pipeline(&rs.device, rs.target_format, wgpu::TextureFormat::Depth32Float);
        Some(View3D { terrain, grass, morph, tree, frame_of: Default::default() })
    }
}
