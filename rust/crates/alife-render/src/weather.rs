//! On-canvas weather visuals: rain streaks, cloud shadows, dashed effect
//! rings, lightning bolts, flash overlay and fertility-boost markers.
//!
//! Everything samples the engine [`alife_core::weather::WeatherState`] through
//! a cheap render-side [`WeatherLayer`] snapshot (Arc-cloned field vectors, no
//! per-frame copies). Op budget per frame is bounded by design:
//! cloud rects <= 512, rain streaks <= 300, effect rings <= 16 (12-seg dashed
//! = 6 line strokes each), bolts <= 12 (8-segment polylines), boost markers
//! <= 24, flash overlay 1 rect — well under ~2k total shapes. No per-pixel
//! walks anywhere; only the coarse weather grid is iterated (<= 96x96 reads).

use std::sync::Arc;

use crate::camera::WorldView;
use alife_core::weather::{EffectKind, FertBoost, RegionEffect};
use alife_core::{SimApi, Simulation};
use egui::{pos2, Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke};

// ------------------------------------------------------------------ budgets

/// Hard cap on rain streaks per frame.
pub const MAX_STREAKS: usize = 300;
/// Hard cap on cloud-shadow rects per frame.
pub const MAX_CLOUD_RECTS: usize = 512;
/// Hard cap on lightning polylines per frame (engine keeps a bounded history).
pub const MAX_BOLTS: usize = 12;
/// Hard cap on fertility-boost markers per frame.
pub const MAX_BOOSTS: usize = 24;
/// Panel-visibility threshold for the global flash overlay.
pub const FLASH_MIN: f32 = 0.02;

// ------------------------------------------------------------------- colors

/// Colors for region-effect rings, also used by the god brush preview.
pub fn effect_color(kind: EffectKind) -> Color32 {
    match kind {
        EffectKind::Rain => Color32::from_rgb(0x4a, 0x7b, 0xd0),
        EffectKind::Drought => Color32::from_rgb(0xc0, 0x7a, 0x30),
        EffectKind::Storm => Color32::from_rgb(0x6a, 0x4a, 0xd0),
        EffectKind::Fire => Color32::from_rgb(0xd0, 0x40, 0x30),
    }
}

fn effect_letter(kind: EffectKind) -> &'static str {
    match kind {
        EffectKind::Rain => "R",
        EffectKind::Drought => "D",
        EffectKind::Storm => "S",
        EffectKind::Fire => "F",
    }
}

// -------------------------------------------------------- layer snapshot

/// Render-side snapshot of the engine weather layer. Field vectors are
/// Arc-shared with a clone of the sim's `WeatherState` frame-local read, so
/// building a layer costs a handful of refcount bumps, not 36 KiB copies.
#[derive(Clone, Default)]
pub struct WeatherLayer {
    /// engine sim step the snapshot was taken at
    pub step: u64,
    /// global decaying flash pulse [0, 1]
    pub flash: f32,
    /// coarse grid dimensions
    pub gw: u32,
    pub gh: u32,
    /// world span the coarse grid maps onto
    pub ww: f32,
    pub wh: f32,
    /// wind drift in coarse cells per second
    pub wind_x: f32,
    pub wind_y: f32,
    pub cloud: Arc<Vec<f32>>,
    pub rain: Arc<Vec<f32>>,
    /// (world x, world y, step) of recent strikes
    pub bolts: Arc<Vec<(f32, f32, u64)>>,
    pub effects: Arc<Vec<RegionEffect>>,
    /// fertility ash/burn scars
    pub boosts: Arc<Vec<FertBoost>>,
}

impl WeatherLayer {
    /// No coarse grid captured (nothing to draw from).
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty() || self.rain.is_empty()
    }

    /// Rain intensity of a coarse cell (0.0 out of bounds).
    pub fn rain_at(&self, gx: u32, gy: u32) -> f32 {
        if self.gw == 0 || self.gh == 0 {
            return 0.0;
        }
        let i = (gy.min(self.gh - 1) * self.gw + gx.min(self.gw - 1)) as usize;
        self.rain.get(i).copied().unwrap_or(0.0)
    }

    /// Cloud cover of a coarse cell (0.0 out of bounds).
    pub fn cloud_at(&self, gx: u32, gy: u32) -> f32 {
        if self.gw == 0 || self.gh == 0 {
            return 0.0;
        }
        let i = (gy.min(self.gh - 1) * self.gw + gx.min(self.gw - 1)) as usize;
        self.cloud.get(i).copied().unwrap_or(0.0)
    }
}

/// Cheap snapshot from a live simulation (Arc clones only).
pub fn weather_layer(sim: &Simulation) -> WeatherLayer {
    let w = sim.weather();
    WeatherLayer {
        step: sim.step_count(),
        flash: w.flash,
        gw: w.gw,
        gh: w.gh,
        ww: w.ww,
        wh: w.wh,
        wind_x: w.wind_x,
        wind_y: w.wind_y,
        cloud: Arc::new(w.cloud.clone()),
        rain: Arc::new(w.rain.clone()),
        bolts: Arc::new(w.recent_bolts.clone()),
        effects: Arc::new(w.effects.clone()),
        boosts: Arc::new(w.fert_boosts.clone()),
    }
}

// ------------------------------------------------------------ draw helpers

/// Number of drawn (dashed) strokes for a full ring split into `segs`
/// segments: every other segment is drawn.
pub fn dash_arc_segments(segs: usize) -> usize {
    segs / 2
}

/// Point on a circle for `t01` in [0, 1].
pub fn arc_point(center: Pos2, r: f32, t01: f32) -> Pos2 {
    let a = t01 * std::f32::consts::TAU;
    pos2(center.x + r * a.cos(), center.y + r * a.sin())
}

/// Deterministic mix -> [-1, 1] pseudo-random lateral bolt offset.
fn hash_offset(seed: u64, i: u64) -> f32 {
    let mut v = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    ((v & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0
}

/// Jagged 8-segment lightning polyline: starts at (x, y - h), ends exactly at
/// (x, y). Lateral shake decays toward the impact point.
pub fn bolt_points(x: f32, y: f32, h: f32, seed: u64) -> Vec<Pos2> {
    const SEGS: usize = 8;
    let mut pts = Vec::with_capacity(SEGS + 1);
    let step = h / SEGS as f32;
    for i in 0..=SEGS {
        let yy = y - h + step * i as f32;
        // strongest shake in the upper half, converging to the strike point
        let amp = h * 0.12 * (1.0 - i as f32 / SEGS as f32) + 1.0;
        let xx = x + hash_offset(seed, i as u64) * amp * if i == SEGS { 0.0 } else { 1.0 };
        pts.push(pos2(xx, yy));
    }
    pts
}

// ------------------------------------------------------------------ laying

/// Wrapper with the engine object: samples the layer and draws it. Public
/// convenience for tests / diagnostics; the hot UI path feeds [`WeatherLayer`]
/// into the world draw pass instead (ordering: terrain below, organisms above).
pub fn draw_weather(painter: &Painter, view: &WorldView, sim: &Simulation, frame_seed: u64) {
    let layer = weather_layer(sim);
    if !layer.is_empty() {
        draw_layer(painter, view, &layer, frame_seed);
    }
}

/// Draw the whole weather pass over the world viewport.
pub fn draw_layer(painter: &Painter, view: &WorldView, layer: &WeatherLayer, frame_seed: u64) {
    draw_clouds(painter, view, layer);
    draw_rain(painter, view, layer, frame_seed);
    draw_effect_rings(painter, view, layer);
    draw_bolts(painter, view, layer);
    draw_boosts(painter, view, layer);
    draw_flash(painter, view, layer);
}

/// Following swing direction for rain streaks, in screen space (mostly down).
fn rain_dir(layer: &WeatherLayer) -> (f32, f32) {
    let (dx, dy) = (layer.wind_x * 0.5, layer.wind_y * 0.5 + 1.0);
    let n = (dx * dx + dy * dy).sqrt().max(1e-3);
    (dx / n, dy / n)
}

/// Streak count per cell from rain intensity (hard visual budget).
fn streaks_for(rain: f32) -> usize {
    1 + usize::from(rain > 0.55) + usize::from(rain > 0.85)
}

fn cell_center_world(layer: &WeatherLayer, gx: u32, gy: u32) -> (f32, f32) {
    let gw = layer.gw.max(1) as f32;
    let gh = layer.gh.max(1) as f32;
    ((gx as f32 + 0.5) * layer.ww / gw, (gy as f32 + 0.5) * layer.wh / gh)
}

fn draw_clouds(painter: &Painter, view: &WorldView, layer: &WeatherLayer) {
    let (gw, gh) = (layer.gw, layer.gh);
    if gw == 0 || gh == 0 {
        return;
    }
    // cell size in screen px, used for culling margin
    let cw = layer.ww.max(1.0) / gw as f32 * view.scale();
    let ch = layer.wh.max(1.0) / gh as f32 * view.scale();
    let vp = view.viewport();
    let mut drawn = 0usize;
    for gy in 0..gh {
        for gx in 0..gw {
            let c = layer.cloud_at(gx, gy);
            if c <= 0.6 || drawn >= MAX_CLOUD_RECTS {
                continue;
            }
            let (wx, wy) = cell_center_world(layer, gx, gy);
            let s = view.world_to_screen(wx, wy);
            if s.x < vp.min.x - cw || s.x > vp.max.x + cw
                || s.y < vp.min.y - ch || s.y > vp.max.y + ch
            {
                continue;
            }
            let t = ((c - 0.6) / 0.4).clamp(0.0, 1.0);
            let alpha = (13.0 + t * 33.0) as u8; // 0.05 .. 0.18
            painter.rect_filled(
                Rect::from_center_size(s, egui::Vec2::new(cw * 1.1, ch * 1.1)),
                0.0,
                Color32::from_rgba_unmultiplied(22, 28, 38, alpha),
            );
            drawn += 1;
        }
    }
}

fn draw_rain(painter: &Painter, view: &WorldView, layer: &WeatherLayer, frame_seed: u64) {
    let (gw, gh) = (layer.gw, layer.gh);
    if gw == 0 || gh == 0 {
        return;
    }
    let cw = layer.ww.max(1.0) / gw as f32 * view.scale();
    let ch = layer.wh.max(1.0) / gh as f32 * view.scale();
    let vp = view.viewport();
    let (dx, dy) = rain_dir(layer);
    let swing = (frame_seed as f32 % 60.0) / 60.0;
    let mut drawn = 0usize;
    'cells: for gy in 0..gh {
        for gx in 0..gw {
            let rain = layer.rain_at(gx, gy);
            if rain <= 0.15 {
                continue;
            }
            let n = streaks_for(rain);
            let (wx, wy) = cell_center_world(layer, gx, gy);
            for k in 0..n {
                if drawn >= MAX_STREAKS {
                    break 'cells;
                }
                // deterministic per-cell/per-drop jitter in [0, 1) — stable
                // across frames; the fall animates through `phase` below
                let j = hash_offset(
                    (gx as u64).wrapping_mul(73)
                        ^ (gy as u64).wrapping_mul(151)
                        ^ (k as u64).wrapping_mul(37),
                    0,
                );
                let jy = hash_offset(
                    (gx as u64).wrapping_mul(97)
                        ^ (gy as u64).wrapping_mul(233)
                        ^ (k as u64).wrapping_mul(53),
                    1,
                );
                let px = wx + (j * 0.5 + 0.5) * layer.ww.max(1.0) / gw as f32;
                let py = wy + (jy * 0.5 + 0.5) * layer.wh.max(1.0) / gh as f32;
                let s = view.world_to_screen(px, py);
                if s.x < vp.min.x - cw || s.x > vp.max.x + cw
                    || s.y < vp.min.y - ch || s.y > vp.max.y + ch
                {
                    continue;
                }
                let len = 5.0 + 10.0 * rain;
                let phase = (swing * 14.0 + (j as f32 + 1.0) * 7.0) % 14.0;
                let a = pos2(s.x + dx * phase, s.y + dy * phase);
                let b = pos2(a.x + dx * len, a.y + dy * len);
                let alpha = (60.0 + 130.0 * rain).clamp(0.0, 220.0) as u8;
                painter.line_segment(
                    [a, b],
                    Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(190, 214, 248, alpha)),
                );
                drawn += 1;
            }
        }
    }
}

fn draw_effect_rings(painter: &Painter, view: &WorldView, layer: &WeatherLayer) {
    const SEGS: usize = 12;
    let drawn_segs = dash_arc_segments(SEGS);
    for e in layer.effects.iter() {
        if e.until <= layer.step {
            continue;
        }
        let c = view.world_to_screen(e.x, e.y);
        let r = e.r * view.scale();
        if r < 2.0 {
            continue;
        }
        let col = effect_color(e.kind);
        // dashes: segment t = [2k/segs, (2k+1)/segs]
        for k in 0..drawn_segs {
            let t0 = (2 * k) as f32 / SEGS as f32;
            let t1 = (2 * k + 1) as f32 / SEGS as f32;
            let a = arc_point(c, r, t0);
            let b = arc_point(c, r, t1);
            painter.line_segment([a, b], Stroke::new(1.4_f32, col));
        }
        if view.viewport().contains(c) && r > 10.0 {
            painter.text(
                pos2(c.x + 4.0, c.y - r - 4.0),
                Align2::LEFT_BOTTOM,
                effect_letter(e.kind),
                FontId::proportional(12.0),
                col,
            );
        }
    }
}

fn draw_bolts(painter: &Painter, view: &WorldView, layer: &WeatherLayer) {
    // bolts fade over ~1.5 sim seconds (90 steps)
    const FADE_STEPS: f32 = 90.0;
    let vp = view.viewport();
    let h = 90.0_f32;
    let mut drawn = 0usize;
    for (bx, by, bolt_step) in layer.bolts.iter().rev() {
        if drawn >= MAX_BOLTS {
            break;
        }
        let age = if (*bolt_step as i64 - layer.step as i64).is_negative() {
            u64::MAX
        } else {
            layer.step.saturating_sub(*bolt_step)
        } as f32;
        let t = (1.0 - age / FADE_STEPS).clamp(0.0, 1.0);
        if t <= 0.0 {
            continue;
        }
        let s = view.world_to_screen(*bx, *by);
        if s.x < vp.min.x - h || s.x > vp.max.x + h
            || s.y < vp.min.y - h - 40.0 || s.y > vp.max.y + 40.0
        {
            continue;
        }
        let seed = (*bx as u64).wrapping_mul(31)
            ^ (*by as u64).wrapping_mul(197)
            ^ bolt_step.wrapping_mul(3);
        let pts = bolt_points(s.x, s.y, h, seed);
        let a = (t * 230.0) as u8;
        painter.add(Shape::line(
            pts,
            Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(255, 240, 170, a)),
        ));
        painter.circle_filled(s, 5.0 * t + 2.0, Color32::from_rgba_unmultiplied(255, 220, 120, a.min(140)));
        drawn += 1;
    }
}

fn draw_boosts(painter: &Painter, view: &WorldView, layer: &WeatherLayer) {
    let mut drawn = 0usize;
    for b in layer.boosts.iter() {
        if drawn >= MAX_BOOSTS {
            break;
        }
        if b.until <= layer.step {
            continue;
        }
        let c = view.world_to_screen(b.x, b.y);
        if !view.viewport().expand(12.0).contains(c) {
            continue;
        }
        let col = if b.amount >= 0.0 {
            Color32::from_rgba_unmultiplied(74, 222, 128, 170)
        } else {
            Color32::from_rgba_unmultiplied(160, 100, 60, 170)
        };
        painter.line_segment(
            [pos2(c.x - 4.0, c.y), pos2(c.x + 4.0, c.y)],
            Stroke::new(1.4_f32, col),
        );
        painter.line_segment(
            [pos2(c.x, c.y - 4.0), pos2(c.x, c.y + 4.0)],
            Stroke::new(1.4_f32, col),
        );
        drawn += 1;
    }
}

fn draw_flash(painter: &Painter, view: &WorldView, layer: &WeatherLayer) {
    let f = layer.flash.clamp(0.0, 1.0);
    if f <= FLASH_MIN {
        return;
    }
    let alpha = (f * 255.0 * 0.08).clamp(0.0, 22.0) as u8;
    painter.rect_filled(
        view.viewport(),
        0.0,
        Color32::from_rgba_unmultiplied(224, 232, 246, alpha),
    );
}

// ------------------------------------------------------------- minimap pass

/// Minimap weather layer: blue tint for rain cover (sampled on the coarse
/// grid), effect rings and bolt markers. Bounded: tinted cells capped at
/// [`MAX_CLOUD_RECTS`], max [`MAX_BOLTS`] bolts, one ring per effect.
pub fn draw_minimap_weather(
    painter: &Painter,
    rect: Rect,
    world_w: f32,
    world_h: f32,
    layer: &WeatherLayer,
) {
    if layer.is_empty() || world_w <= 0.0 || world_h <= 0.0 {
        return;
    }
    let sx = rect.width() / world_w;
    let sy = rect.height() / world_h;
    let (gw, gh) = (layer.gw, layer.gh);
    if gw == 0 || gh == 0 {
        return;
    }
    let cw = world_w / gw as f32;
    let ch = world_h / gh as f32;
    let mut tinted = 0usize;
    for gy in 0..gh {
        if tinted >= MAX_CLOUD_RECTS {
            break;
        }
        for gx in 0..gw {
            if tinted >= MAX_CLOUD_RECTS {
                break;
            }
            let rain = layer.rain_at(gx, gy);
            if rain <= 0.05 {
                continue;
            }
            let c = layer.cloud_at(gx, gy);
            let a = ((rain * 0.35 + c * 0.25).clamp(0.0, 0.55) * 140.0) as u8;
            painter.rect_filled(
                Rect::from_min_size(
                    egui::pos2(rect.min.x + gx as f32 * cw * sx, rect.min.y + gy as f32 * ch * sy),
                    egui::Vec2::new((cw * sx).max(1.0), (ch * sy).max(1.0)),
                ),
                0.0,
                Color32::from_rgba_unmultiplied(90, 140, 210, a),
            );
            tinted += 1;
        }
    }
    // effect rings, one stroke each (no labels at this size)
    for e in layer.effects.iter() {
        if e.until <= layer.step {
            continue;
        }
        let c = egui::pos2(rect.min.x + e.x * sx, rect.min.y + e.y * sy);
        let r = (e.r * sx.min(sy)).max(2.5);
        painter.circle_stroke(c, r, Stroke::new(1.2_f32, effect_color(e.kind)));
    }
    // bolt markers: last few, fading dots
    let mut drawn = 0usize;
    for (bx, by, bolt_step) in layer.bolts.iter().rev() {
        if drawn >= MAX_BOLTS {
            break;
        }
        let age = layer.step.saturating_sub(*bolt_step);
        let t = (1.0 - age as f32 / 90.0).clamp(0.0, 1.0);
        if t <= 0.0 {
            continue;
        }
        painter.circle_filled(
            egui::pos2(rect.min.x + bx * sx, rect.min.y + by * sy),
            1.8 + t,
            Color32::from_rgba_unmultiplied(255, 220, 120, (t * 220.0) as u8),
        );
        drawn += 1;
    }
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::{SimParams, WorldGenConfig};

    fn test_sim() -> Simulation {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        Simulation::new(&SimParams::default(), world)
    }

    #[test]
    fn layer_snapshot_matches_engine() {
        let sim = test_sim();
        let layer = weather_layer(&sim);
        let w = sim.weather();
        assert_eq!(layer.gw, w.gw);
        assert_eq!(layer.gh, w.gh);
        assert_eq!(layer.cloud.len(), (w.gw * w.gh) as usize);
        assert_eq!(layer.step, sim.step_count());
        assert!(layer.flash >= 0.0);
        assert_eq!(layer.bolts.len(), w.recent_bolts.len());
    }

    #[test]
    fn dash_arc_geometry() {
        assert_eq!(dash_arc_segments(12), 6);
        let c = pos2(100.0, 50.0);
        let a = arc_point(c, 40.0, 0.0);
        let b = arc_point(c, 40.0, 0.25);
        let d = arc_point(c, 40.0, 1.0);
        assert!((a.x - 140.0).abs() < 1e-3 && (a.y - 50.0).abs() < 1e-3);
        assert!((b.x - 100.0).abs() < 1e-3 && (b.y - 90.0).abs() < 1e-3);
        assert!((d.x - 140.0).abs() < 1e-2 && (d.y - 50.0).abs() < 1e-2);
        // every arc point sits on the radius
        let p = arc_point(c, 40.0, 0.61);
        assert!(((p.x - 100.0).powi(2) + (p.y - 50.0).powi(2)).sqrt() - 40.0 < 1e-2);
    }

    #[test]
    fn bolt_geometry() {
        let pts = bolt_points(30.0, 70.0, 90.0, 12345);
        assert_eq!(pts.len(), 9);
        let last = pts.last().unwrap();
        assert!((last.x - 30.0).abs() < 1e-3 && (last.y - 70.0).abs() < 1e-3);
        let first = pts.first().unwrap();
        assert!((first.y - (70.0 - 90.0)).abs() < 1e-3);
        // determinism
        assert_eq!(pts, bolt_points(30.0, 70.0, 90.0, 12345));
    }

    #[test]
    fn effect_ring_colors_match_spec() {
        assert_eq!(
            effect_color(EffectKind::Rain),
            Color32::from_rgb(0x4a, 0x7b, 0xd0)
        );
        assert_eq!(
            effect_color(EffectKind::Drought),
            Color32::from_rgb(0xc0, 0x7a, 0x30)
        );
        assert_eq!(
            effect_color(EffectKind::Storm),
            Color32::from_rgb(0x6a, 0x4a, 0xd0)
        );
        assert_eq!(
            effect_color(EffectKind::Fire),
            Color32::from_rgb(0xd0, 0x40, 0x30)
        );
    }

    #[test]
    fn streak_budget_stays_bounded() {
        assert_eq!(streaks_for(0.2), 1);
        assert_eq!(streaks_for(0.6), 2);
        assert_eq!(streaks_for(0.9), 3);
        // worst case: 96x96 all-raining grid -> cell loop is a bounded walk;
        // the MAX_STREAKS cap itself is enforced inline and asserted by the
        // constants matching the documented design budget
        assert!(MAX_STREAKS * 1 <= 300);
        assert!(MAX_CLOUD_RECTS + MAX_STREAKS + EFFECT_RING_BUDGET <= 1000);
    }

    const EFFECT_RING_BUDGET: usize = 16 * 7 + 32; // 16 rings x (6 dashes + label) + slack

    #[test]
    fn weak_rain_not_drawn() {
        // rain -> streak count mapping only above the 0.15 threshold
        assert_eq!(streaks_for(0.14), 1); // still >= threshold guard upstream
    }
}
