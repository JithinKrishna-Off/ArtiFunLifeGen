//! Camera / view controller: pan, zoom, follow, coordinate transforms,
//! terrain texture lifecycle and dynamic raster caches.

use std::time::Instant;

use alife_worldgen::WorldModel;
use egui::{Pos2, Rect, Sense, TextureFilter, TextureHandle, TextureOptions, TextureWrapMode, Ui};

use crate::terrain::{bake_terrain_scaled, MAX_BAKE_PIXELS};
use crate::trails::Trails;
use crate::weather::WeatherLayer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewMode {
    Natural,
    Elevation,
    Moisture,
    Temperature,
    Fertility,
    Hazard,
    Density,
    Species,
    Energy,
}

impl ViewMode {
    pub const ALL: [ViewMode; 9] = [
        ViewMode::Natural,
        ViewMode::Elevation,
        ViewMode::Moisture,
        ViewMode::Temperature,
        ViewMode::Fertility,
        ViewMode::Hazard,
        ViewMode::Density,
        ViewMode::Species,
        ViewMode::Energy,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Natural => "Natural terrain",
            ViewMode::Elevation => "Elevation",
            ViewMode::Moisture => "Moisture",
            ViewMode::Temperature => "Temperature",
            ViewMode::Fertility => "Fertility",
            ViewMode::Hazard => "Hazard",
            ViewMode::Density => "Density",
            ViewMode::Species => "Species",
            ViewMode::Energy => "Energy",
        }
    }

    /// Modes whose texture depends on live sim state (rebuild throttled).
    pub fn dynamic(self) -> bool {
        matches!(self, ViewMode::Density | ViewMode::Species | ViewMode::Energy)
    }
}

/// Per-cell dynamic rasters (food + organism density, dominant species,
/// energy). Owned by WorldView; recomputed by the paint layer.
#[derive(Default, Debug, Clone)]
pub struct DynFields {
    pub w: u32,
    pub h: u32,
    pub density: Vec<u32>,
    pub dominant: Vec<u32>,
    pub energy: Vec<f32>,
}

pub struct CanvasResponse {
    /// World-space position of a click (press + release without real drag).
    pub click: Option<(f32, f32)>,
    /// True when the user dragged far enough to count as a pan (app cancels
    /// camera follow based on this).
    pub dragged: bool,
    /// World-space cursor position while the viewport is hovered.
    pub hover: (f32, f32),
    pub hovering: bool,
    pub organisms_drawn: u32,
    pub organisms_total: u32,
    pub scale: f32,
}

impl Default for CanvasResponse {
    fn default() -> Self {
        Self {
            click: None,
            dragged: false,
            hover: (0.0, 0.0),
            hovering: false,
            organisms_drawn: 0,
            organisms_total: 0,
            scale: 1.0,
        }
    }
}

const TERRAIN_OPTIONS: TextureOptions = TextureOptions {
    magnification: TextureFilter::Linear,
    minification: TextureFilter::Linear,
    mipmap_mode: None,
    wrap_mode: TextureWrapMode::ClampToEdge,
};

/// Rebuild dynamic-mode textures at most this often.
const DYN_REBAKE_FRAMES: u32 = 10;

/// synchronous terrain bakes commit only when the estimated cost fits inside
/// this wall-clock budget; larger bakes keep the old texture and retry next
/// frame as the estimate adapts.
const BAKE_MS_BUDGET: f32 = 8.0;
/// supersampling floor (bake no more than this times the source grid).
const MAX_UPSCALE: f32 = 4.0;

/// Viewport pixel density bucketed at 128 px: part of the bake key so modest
/// resizes reuse the cached supersampled texture.
pub fn viewport_bucket(viewport: Rect, ppp: f32) -> (u32, u32) {
    let ppp = ppp.max(0.1);
    let w = (viewport.width() * ppp / 128.0).round().max(1.0) as u32;
    let h = (viewport.height() * ppp / 128.0).round().max(1.0) as u32;
    (w, h)
}

/// Deterministic bake identity: world source (dims + generation tag), view
/// mode and the 128 px viewport-pixel-density bucket. Equal inputs must give
/// an equal key (stability test in this module).
pub fn terrain_bake_key(world: &WorldModel, mode: ViewMode, viewport: Rect, ppp: f32) -> u64 {
    let (bw, bh) = viewport_bucket(viewport, ppp);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(mode as u8 as u64);
    mix(world.grid_w() as u64);
    mix(world.grid_h() as u64);
    mix(world.generation_tag as u64);
    mix(bw as u64);
    mix(bh as u64);
    if h == u64::MAX { 1 } else { h }
}

/// Target bake dimensions: ~2x the displayed viewport pixel density in the
/// world's aspect ratio, capped at [`MAX_BAKE_PIXELS`], capped at
/// [`MAX_UPSCALE`]x the source grid, and shrunk further to respect the
/// measured synchronous-bake cost model (`cost_ms_per_mp`, measured).
pub fn terrain_target_dims(
    world: &WorldModel,
    viewport: Rect,
    ppp: f32,
    cost_ms_per_mp: f32,
) -> (usize, usize) {
    let ppp = ppp.max(0.1);
    let gw = (world.grid_w() as f32).max(2.0);
    let gh = (world.grid_h() as f32).max(2.0);
    let disp_w = (viewport.width() * ppp).max(1.0);
    let disp_h = (viewport.height() * ppp).max(1.0);
    // 2x linear supersampling of what the viewport displays
    let mut ow = (disp_w * 2.0).clamp(gw, gw * MAX_UPSCALE);
    let mut oh = (disp_h * 2.0).clamp(gh, gh * MAX_UPSCALE);
    // respect the world aspect ratio (viewport and world may disagree)
    let aspect = world.width.max(1.0) / world.height.max(1.0);
    if ow / oh > aspect {
        ow = oh * aspect;
    } else {
        oh = ow / aspect;
    }
    // absolute pixel cap
    let area = ow * oh;
    if area > MAX_BAKE_PIXELS as f32 {
        let s = (MAX_BAKE_PIXELS as f32 / area).sqrt();
        ow *= s;
        oh *= s;
    }
    // 64px quantization: the adaptive cost-model estimate otherwise nudges
    // the bake dims every bake, and a differently-sized `TextureHandle::set`
    // frees/reallocs the GPU texture per frame — an in-flight submit can then
    // reference the destroyed texture (wgpu "texture destroyed" panic).
    // Quantized dims keep repeated bakes same-size (pure in-place upload).
    let q = 64.0_f32;
    ow = (ow / q).floor().max(1.0) * q;
    oh = (oh / q).floor().max(1.0) * q;
    // synchronous-bake cost gate; floor at 1x the source grid area — if even
    // a grid-res bake cannot make the budget, the caller still commits it on
    // a fresh view (must show *something*) and keeps it otherwise.
    let est = (ow * oh) / 1_000_000.0 * cost_ms_per_mp.max(1e-4);
    if est > BAKE_MS_BUDGET {
        let grid_area = (gw * gh) as f32;
        let s_floor = (grid_area / (ow * oh)).sqrt().min(1.0);
        let s = ((BAKE_MS_BUDGET / est).sqrt().min(1.0)).max(s_floor);
        ow *= s;
        oh *= s;
    }
    (ow.max(16.0) as usize, oh.max(16.0) as usize)
}

pub struct WorldView {
    /// Screen pixels per world unit.
    scale: f32,
    /// Camera center in world coordinates.
    center: (f32, f32),
    viewport: Rect,
    world_tag: u32,
    mode: ViewMode,
    terrain: Option<TextureHandle>,
    terrain_mode: ViewMode,
    terrain_tag: u32,
    terrain_baked_frame: u32,
    /// identity (hash) of the currently committed synchronous bake
    bake_key: u64,
    /// committed bake dimensions
    bake_dims: (u32, u32),
    /// measured terrain bake cost, ms per baked megapixel (adaptive)
    bake_cost_ms_per_mp: f32,
    /// latest weather layer snapshot (minimap tint); set by the UI per frame
    pub weather: WeatherLayer,
    pub(crate) dyn_: DynFields,
    pub trails: Trails,
    pub show_grid: bool,
    hover_world: (f32, f32),
    frame: u32,
    fit_pending: bool,
    drag_origin: Option<Pos2>,
    drag_moved: bool,
    drag_total: f32,
    /// cached detail label: rebuilt on change, but at most at ~10 Hz
    info_key: (u64, usize, usize),
    info_cache: String,
}

impl Default for WorldView {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldView {
    pub fn new() -> Self {
        Self {
            scale: 0.5,
            center: (0.0, 0.0),
            viewport: Rect::from_min_size(Pos2::ZERO, egui::Vec2::new(800.0, 600.0)),
            world_tag: 0,
            mode: ViewMode::Natural,
            terrain: None,
            terrain_mode: ViewMode::Natural,
            terrain_tag: 0,
            terrain_baked_frame: 0,
            bake_key: u64::MAX,
            bake_dims: (0, 0),
            bake_cost_ms_per_mp: 24.0,
            weather: WeatherLayer::default(),
            dyn_: DynFields::default(),
            trails: Trails::new(),
            show_grid: false,
            hover_world: (0.0, 0.0),
            frame: 0,
            fit_pending: true,
            drag_origin: None,
            drag_moved: false,
            drag_total: 0.0,
            info_key: (u64::MAX, 0, 0),
            info_cache: String::new(),
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn center(&self) -> (f32, f32) {
        self.center
    }
    pub fn set_center(&mut self, x: f32, y: f32) {
        self.center = (x, y);
    }
    pub fn viewport(&self) -> Rect {
        self.viewport
    }
    pub fn view_mode(&self) -> ViewMode {
        self.mode
    }
    pub fn set_view_mode(&mut self, m: ViewMode) {
        if self.mode != m {
            self.mode = m;
            self.terrain_mode = m;
            self.terrain_baked_frame = 0;
        }
    }
    pub fn view_mode_mut(&mut self) -> &mut ViewMode {
        &mut self.mode
    }

    /// Call when a new world is installed: tag mismatch forces a terrain
    /// rebuild and a camera refit on the next frame.
    pub fn invalidate_world(&mut self, tag: u32) {
        self.world_tag = tag;
        self.terrain_tag = tag.wrapping_sub(1);
        self.terrain_baked_frame = 0;
        self.fit_pending = true;
        self.trails.clear();
        self.dyn_ = DynFields::default();
    }

    pub fn fit_world(&mut self, world: &WorldModel, viewport: Rect) {
        if world.width < 1.0 || world.height < 1.0 {
            return;
        }
        self.viewport = viewport;
        let margin = 0.955;
        let s = (viewport.width() / world.width).min(viewport.height() / world.height) * margin;
        self.scale = s.max(0.01);
        self.center = (world.width * 0.5, world.height * 0.5);
        self.fit_pending = false;
    }

    /// Fit using the last known viewport (when no fresh rect is at hand).
    pub fn fit(&mut self, world: &WorldModel) {
        self.fit_world(world, self.viewport);
    }

    /// Record the viewport each frame before drawing; performs a pending fit.
    pub fn begin_viewport(&mut self, world: &WorldModel, viewport: Rect) {
        self.viewport = viewport;
        self.frame = self.frame.wrapping_add(1);
        if self.fit_pending {
            self.fit_world(world, viewport);
        }
    }

    pub fn world_to_screen(&self, x: f32, y: f32) -> Pos2 {
        let size = self.viewport.size();
        Pos2::new(
            self.viewport.min.x + size.x * 0.5 + (x - self.center.0) * self.scale,
            self.viewport.min.y + size.y * 0.5 + (y - self.center.1) * self.scale,
        )
    }

    pub fn screen_to_world(&self, p: Pos2) -> (f32, f32) {
        let size = self.viewport.size();
        (
            self.center.0 + (p.x - self.viewport.min.x - size.x * 0.5) / self.scale,
            self.center.1 + (p.y - self.viewport.min.y - size.y * 0.5) / self.scale,
        )
    }

    fn clamp_center(&mut self, world: &WorldModel) {
        let mx = world.width * 0.5 + 60.0;
        let my = world.height * 0.5 + 60.0;
        self.center.0 = self.center.0.clamp(-mx, world.width + mx);
        self.center.1 = self.center.1.clamp(-my, world.height + my);
    }

    pub fn zoom_global(&mut self, factor: f32) {
        self.scale = (self.scale * factor).clamp(0.03, 160.0);
    }

    pub fn zoom_at_screen(&mut self, p: Pos2, factor: f32) {
        let (wx, wy) = self.screen_to_world(p);
        self.scale = (self.scale * factor).clamp(0.03, 160.0);
        let size = self.viewport.size();
        self.center = (
            wx - (p.x - self.viewport.min.x - size.x * 0.5) / self.scale,
            wy - (p.y - self.viewport.min.y - size.y * 0.5) / self.scale,
        );
    }

    /// Pan / zoom / follow handling. When `follow_target` is Some((x, y)) the
    /// camera re-centers on that world point each frame; dragging cancels it.
    pub fn handle_interaction(
        &mut self,
        ui: &mut Ui,
        world: &WorldModel,
        follow_target: Option<(f32, f32)>,
    ) -> CanvasResponse {
        let mut out = CanvasResponse::default();
        out.scale = self.scale;

        if let Some(t) = follow_target {
            self.center = t;
        }

        let resp = ui.allocate_rect(self.viewport, Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        let ctx = &ctx;

        // -------------------------------------------- wheel/pinch zoom at cursor
        if resp.hovered() {
            let scroll: f32 = ctx.input(|i| i.raw_scroll_delta.y);
            let pinch = ctx.input(|i| i.zoom_delta());
            let mut factor = 1.0f32;
            if scroll.abs() > 0.05 {
                factor = if scroll > 0.0 { 1.12 } else { 1.0 / 1.12 };
            }
            if (pinch - 1.0).abs() > 0.001 {
                factor *= pinch;
            }
            if (factor - 1.0).abs() > 0.0001 {
                let mouse = ctx.input(|i| i.pointer.interact_pos());
                match mouse {
                    Some(m) if self.viewport.contains(m) => self.zoom_at_screen(m, factor),
                    _ => self.zoom_global(factor),
                }
            }
        }

        // ------------------------------------------------------------- drag pan
        if resp.drag_started() {
            self.drag_origin = resp.interact_pointer_pos();
            self.drag_moved = false;
            self.drag_total = 0.0;
        }
        if resp.dragged() && self.drag_origin.is_some() {
            let d = resp.drag_delta();
            self.drag_total += d.length();
            if self.drag_total > 6.0 {
                self.drag_moved = true;
                self.center.0 -= d.x / self.scale;
                self.center.1 -= d.y / self.scale;
                self.clamp_center(world);
                out.dragged = true;
            }
        }
        if resp.drag_stopped() {
            if !self.drag_moved {
                if let Some(p) = self.drag_origin {
                    out.click = Some(self.screen_to_world(p));
                }
            }
            self.drag_origin = None;
            self.drag_moved = false;
            self.drag_total = 0.0;
        }

        // -------------------------------------------------------- hover coords
        out.hovering = resp.hovered() || resp.dragged();
        if out.hovering {
            let mouse = ctx.input(|i| i.pointer.interact_pos());
            if let Some(p) = mouse {
                if self.viewport.contains(p) {
                    self.hover_world = self.screen_to_world(p);
                }
            }
        }
        out.hover = self.hover_world;
        out
    }

    /// Rebuild / reuse the terrain texture respecting mode + world tag,
    /// throttling dynamic modes to every DYN_REBAKE_FRAMES frames. Static
    /// bakes are keyed by (world source, view mode, 128px viewport density
    /// bucket) and are supersampled ~2x the viewport pixel density, capped at
    /// MAX_BAKE_PIXELS. A bake is synchronous only while the measured cost
    /// model predicts < BAKE_MS_BUDGET; otherwise the existing (possibly
    /// lower-res) texture stays visible to the next frame.
    pub fn terrain_texture(
        &mut self,
        ctx: &egui::Context,
        world: &WorldModel,
    ) -> Option<&TextureHandle> {
        let mode_changed = self.terrain_mode != self.mode;
        let tag_changed = self.terrain_tag != self.world_tag;
        let dyn_due = self.mode.dynamic()
            && self.frame.wrapping_sub(self.terrain_baked_frame) >= DYN_REBAKE_FRAMES;
        let ppp = ctx.pixels_per_point();
        let key = if self.mode.dynamic() {
            self.bake_key // dyn modes re-bake on their own cadence
        } else {
            terrain_bake_key(world, self.mode, self.viewport, ppp)
        };
        let key_due = !self.mode.dynamic() && key != self.bake_key;
        if self.terrain.is_none() || mode_changed || tag_changed || dyn_due || key_due {
            let target = self.committed_target(world, ppp);
            let est_px = target.0.max(1) as f32 * target.1.max(1) as f32;
            let est_ms = est_px / 1_000_000.0 * self.bake_cost_ms_per_mp.max(1e-4);
            // explicit user/world changes bake immediately even when the
            // spend model recommends waiting (the stale view would show the
            // wrong mode/world); steady-state dyn refreshes keep the budget
            let force = mode_changed || tag_changed || key_due;
            if force || self.terrain.is_none() || est_ms <= BAKE_MS_BUDGET {
                let t0 = Instant::now();
                let img = bake_terrain_scaled(
                    world,
                    self.mode,
                    target.0,
                    target.1,
                    &self.dyn_.density,
                    &self.dyn_.dominant,
                    &self.dyn_.energy,
                );
                let ms = (t0.elapsed().as_secs_f32() * 1000.0).max(0.05);
                let mp = est_px / 1_000_000.0;
                if mp > 0.0 {
                    let measured = ms / mp;
                    // smooth the cost model toward the fresh measurement
                    self.bake_cost_ms_per_mp = (self.bake_cost_ms_per_mp * 0.65 + measured * 0.35)
                        .clamp(0.02, 400.0);
                }
                // ONE persistent texture for the whole session (first bake
                // creates it, every later bake uploads in place via `set`):
                // dropping/reloading handles would let egui free a GPU
                // texture while an in-flight submission still references it
                // ("texture destroyed" panic), and renaming reallocs for the
                // same reason. Same-handle updates never destroy.
                if let Some(tex) = self.terrain.as_mut() {
                    tex.set(img, TERRAIN_OPTIONS);
                } else {
                    let name = format!("alife.terrain.{}", self.mode.label());
                    self.terrain = Some(ctx.load_texture(&name, img, TERRAIN_OPTIONS));
                }
                self.bake_dims = (target.0 as u32, target.1 as u32);
                self.bake_key = key;
                self.terrain_mode = self.mode;
                self.terrain_tag = self.world_tag;
                self.terrain_baked_frame = self.frame;
            }
        }
        self.terrain.as_ref()
    }

    /// Current target bake dims under the cost model (also the stored dims
    /// for dynamic modes; they follow the same policy).
    fn committed_target(&self, world: &WorldModel, ppp: f32) -> (usize, usize) {
        terrain_target_dims(world, self.viewport, ppp, self.bake_cost_ms_per_mp)
    }

    /// Install freshly computed dynamic rasters (from the paint layer).
    pub(crate) fn set_dyn(&mut self, d: DynFields) {
        self.dyn_ = d;
    }
    pub(crate) fn frame_number(&self) -> u32 {
        self.frame
    }

    /// Swap in the per-frame weather snapshot (Arc clones, used by the
    /// minimap pass; the main weather overlay draws from DrawConfig).
    pub fn set_weather_layer(&mut self, w: WeatherLayer) {
        self.weather = w;
    }

    /// "N organisms, M food bits" label with cached allocation: rebuilt only
    /// when the counts change AND the 60-frame bucket (≈10 Hz) turns.
    pub fn cached_info_label(&mut self, pop: usize, food: usize) -> &str {
        let slot = self.frame as u64 / 6;
        let key = (slot, pop, food);
        if self.info_key != key {
            self.info_key = key;
            self.info_cache = format!("{pop} organisms, {food} food bits");
        }
        self.info_cache.as_str()
    }

    /// Committed bake dimensions (diagnostics/tests).
    pub fn baked_dims(&self) -> (u32, u32) {
        self.bake_dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Rect, Vec2};

    fn world() -> WorldModel {
        alife_worldgen::generate(&alife_common::WorldGenConfig::default())
    }

    fn vp(w: f32, h: f32) -> Rect {
        Rect::from_min_size(Pos2::ZERO, Vec2::new(w, h))
    }

    #[test]
    fn bake_key_stable_per_inputs() {
        let w = world();
        let a = terrain_bake_key(&w, ViewMode::Natural, vp(800.0, 600.0), 1.0);
        let b = terrain_bake_key(&w, ViewMode::Natural, vp(800.0, 600.0), 1.0);
        assert_eq!(a, b, "same inputs must map to the same bake key");
        let c = terrain_bake_key(&w, ViewMode::Natural, vp(820.0, 600.0), 1.0);
        assert_eq!(a, c, "80px resize stays inside the 128px bucket");
        let d = terrain_bake_key(&w, ViewMode::Natural, vp(1200.0, 600.0), 1.0);
        assert_ne!(a, d, "bucket change must change the key");
        let e = terrain_bake_key(&w, ViewMode::Elevation, vp(800.0, 600.0), 1.0);
        assert_ne!(a, e, "mode change must change the key");
    }

    #[test]
    fn bake_dims_respect_caps_and_cost() {
        let w = world();
        // cheap cost: target should be ~2x the viewport density, capped
        let (ow, oh) = terrain_target_dims(&w, vp(300.0, 200.0), 1.0, 1.0);
        assert!(ow <= (w.grid_w() as f32 * MAX_UPSCALE) as usize);
        assert!(ow * oh <= MAX_BAKE_PIXELS);
        // expensive cost: bake is shrunk but never below the source grid
        let (ow2, oh2) = terrain_target_dims(&w, vp(1000.0, 700.0), 1.0, 500.0);
        assert!(ow2 as f32 >= w.grid_w() as f32 - 0.5);
        assert!(ow2 * oh2 <= MAX_BAKE_PIXELS + 1);
    }

    #[test]
    fn bucket_is_rounded() {
        assert_eq!(viewport_bucket(vp(128.0, 128.0), 1.0), (1, 1));
        assert_eq!(viewport_bucket(vp(770.0, 640.0), 1.0), (6, 5));
    }

    #[test]
    fn info_label_is_cached() {
        let mut v = WorldView::new();
        v.frame = 6;
        let a = v.cached_info_label(10, 20).to_string();
        let b = v.cached_info_label(10, 20).to_string();
        assert_eq!(a, b, "same key same string object path");
        assert!(a.contains("10 organisms"));
        let c = v.cached_info_label(11, 20).to_string();
        assert!(c.contains("11 organisms"));
    }
}
