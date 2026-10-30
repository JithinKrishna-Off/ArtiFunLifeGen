//! Entity painting layer: terrain blit, obstacles/hazards/patches, food dots,
//! organism bodies with species colors, trails, selection marks, grid overlay
//! and the corner minimap. Pure egui painter with manual transform math.

use crate::camera::{CanvasResponse, DynFields, WorldView};
use crate::trails::Trails;
use crate::weather::{draw_layer as draw_weather_layer, draw_minimap_weather, WeatherLayer};
use alife_common::{EnvironmentEdits, Food, FoodPatch, HazardZone, Obstacle, Organism, organism_radius, species_hue};
use alife_worldgen::WorldModel;
use egui::{pos2, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Shape, Stroke, Vec2};

pub struct DrawConfig {
    pub show_food: bool,
    pub show_patches: bool,
    pub show_hazards: bool,
    pub show_obstacles: bool,
    /// record trails this frame (running state)
    pub trail_capture: bool,
    /// rain-bow sensor circles for the selected organism
    pub show_sensor: bool,
    /// free camera motion allowed (editor mode may still need it)
    pub interactions: bool,
    /// organism id the UI hovered (throttled pick): white hover ring
    pub hover_id: Option<u64>,
    /// animation clock for weather / selection pulse (often the frame number)
    pub frame_seed: u64,
    /// when Some, a weather pass (rain/clouds/effects/bolts/flash) draws
    /// between terrain and organisms. None = weather idle: zero cost.
    pub weather: Option<WeatherLayer>,
}

impl Default for DrawConfig {
    fn default() -> Self {
        Self {
            show_food: true,
            show_patches: true,
            show_hazards: true,
            show_obstacles: true,
            trail_capture: false,
            show_sensor: true,
            interactions: true,
            hover_id: None,
            frame_seed: 0,
            weather: None,
        }
    }
}

// ---------------------------------------------------------------- color utils

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color32 {
    let h = ((h % 360.0).rem_euclid(360.0)) / 360.0;
    let hh = h.clamp(0.0, 1.0);
    let a = s.min(1.0) * l.clamp(0.0, 1.0).min(1.0 - l);
    let p = l + a;
    let q = l - a;
    let conv = |raw: f32| -> f32 {
        let mut t = (raw + hh) % 1.0;
        if t < 0.0 {
            t += 1.0;
        }
        if t < 1.0 / 6.0 {
            q + (p - q) * 6.0 * t
        } else if t < 0.5 {
            p
        } else if t < 2.0 / 3.0 {
            q + (p - q) * (2.0 / 3.0 - t) * 6.0
        } else {
            q
        }
    };
    let r = (conv(1.0 / 3.0) * 255.0) as u8;
    let g = (conv(0.0) * 255.0) as u8;
    let b = (conv(-1.0 / 3.0) * 255.0) as u8;
    Color32::from_rgb(r, g, b)
}

pub fn species_color(species_id: u32) -> Color32 {
    hsl_to_rgb(species_hue(species_id), 0.62, 0.56)
}

fn species_body_color(species_id: u32, energy: f32, max_energy: f32) -> Color32 {
    let level = (energy / max_energy.max(1.0)).clamp(0.0, 1.0);
    let light = 0.40 + 0.28 * level;
    hsl_to_rgb(species_hue(species_id), 0.60, light)
}

// ---------------------------------------------------------------- main entry

#[allow(clippy::too_many_arguments)]
pub fn draw_world(
    ui: &mut egui::Ui,
    view: &mut WorldView,
    rect: Rect,
    world: &WorldModel,
    organisms: &[Organism],
    foods: &[Food],
    edits: &EnvironmentEdits,
    selected: Option<u64>,
    hover: (f32, f32),
    cfg: &DrawConfig
) -> CanvasResponse {
    view.begin_viewport(world, rect);
    let mut out = CanvasResponse {
        organisms_total: organisms.len() as u32,
        ..Default::default()
    };

    // interaction first so pan/zoom apply this very frame
    let resp = if cfg.interactions {
        view.handle_interaction(ui, world, None)
    } else {
        ui.allocate_rect(rect, Sense::hover());
        CanvasResponse::default()
    };
    out.scale = view.scale();

    // dynamic rasters for Density/Species/Energy terrain modes
    if view.view_mode().dynamic() && view.frame_number() % 8 == 0 {
        let dyn_fields = compute_dyn(world, organisms, foods);
        view.set_dyn(dyn_fields);
    }

    let painter = ui.painter().with_clip_rect(rect);

    // -------------------------------------------------------------- terrain
    let tex_id = view.terrain_texture(ui.ctx(), world).map(|t| t.id());
    if let Some(tid) = tex_id {
        let a = view.world_to_screen(0.0, 0.0);
        let b = view.world_to_screen(world.width, world.height);
        let world_rect = Rect::from_two_pos(a, b);
        painter.image(
            tid,
            world_rect.expand(0.5),
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        painter.rect_stroke(world_rect, 0.0, Stroke::new(1.5_f32, Color32::from_rgb(46, 56, 66)));
    }

    // -------------------------------------------------------------- weather
    // order: terrain below -> weather -> editor overlays -> organisms above
    if let Some(weather) = &cfg.weather {
        if !weather.is_empty() {
            draw_weather_layer(&painter, view, weather, cfg.frame_seed);
        }
    }

    // ---------------------------------------------------------- grid overlay
    if view.show_grid {
        draw_grid(&painter, view, world);
    }

    // ------------------------------------------------------------ editor info
    // dynamic overlays first (under entities)
    if cfg.show_patches {
        for p in &edits.patches {
            draw_patch(&painter, view, p);
        }
    }
    if cfg.show_hazards {
        for h in &edits.hazards {
            draw_hazard(&painter, view, h);
        }
    }
    if cfg.show_obstacles {
        for o in &edits.obstacles {
            draw_obstacle(&painter, view, o);
        }
    }

    // ----------------------------------------------------------------- food
    if cfg.show_food {
        let stride = food_stride(view.scale(), foods.len());
        for (i, f) in foods.iter().enumerate() {
            if f.alive && i % stride == 0 {
                draw_food_dot(&painter, view, f);
            }
        }
    }

    // --------------------------------------------------------------- trails
    if cfg.trail_capture {
        view.trails.track(organisms);
    }
    let pop = organisms.len();
    let trails_on = view.trails.enabled() && pop <= 900 && view.scale() > 0.22;
    if trails_on {
        for o in organisms {
            draw_trail_points(&painter, view, &view.trails, o);
        }
    }

    // ------------------------------------------------------------ organisms
    let dot_only = pop > 2000 || view.scale() < 0.22;
    let mut drawn = 0u32;
    let hover_picker = if out.hovering { Some(hover) } else { None };
    for o in organisms {
        let is_sel = selected == Some(o.id);
        let geometric_hover = o.alive && hover_picker.map(|(hx, hy)| {
            let r = organism_radius(&o.genotype.traits);
            (o.x - hx).powi(2) + (o.y - hy).powi(2) < r * r + 40.0
        }) == Some(true);
        // white hover ring also honors the throttled pick id from the UI
        let is_hover = geometric_hover || (o.alive && cfg.hover_id == Some(o.id));
        if dot_only && !is_sel {
            draw_dot_org(&painter, view, o);
        } else {
            draw_org(&painter, view, o, is_hover);
        }
        if is_sel {
            draw_selection(&painter, view, o, cfg.show_sensor, cfg.frame_seed);
        } else if is_hover {
            let c = view.world_to_screen(o.x, o.y);
            let r = organism_radius(&o.genotype.traits) * view.scale();
            painter.circle_stroke(c, r + 4.0, Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(238, 243, 248, 190)));
        }
        drawn += 1;
    }
    out.organisms_drawn = drawn;

    // detail-of-demand label: mini org count at current zoom coherence.
    // String is cached on the view (rebuilt on change + at most at the 10 Hz
    // frame-bucket rate) — no per-frame format allocation.
    if view.scale() > 0.3 && pop > 0 {
        let label = view.cached_info_label(pop, foods.len());
        painter.text(
            rect.min + egui::vec2(8.0, rect.height() - 20.0),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(11.0),
            Color32::from_rgba_unmultiplied(139, 150, 162, 200),
        );
    }
    out.hover = resp.hover;
    out.hovering = out.hovering || resp.hovering;
    out
}

// ------------------------------------------------------------------ layers

fn draw_grid(p: &Painter, view: &WorldView, world: &WorldModel) {
    let step_units = if view.scale() > 2.0 {
        50.0
    } else if view.scale() > 0.7 {
        200.0
    } else {
        800.0
    };
    let col = Color32::from_rgba_unmultiplied(143, 160, 178, 28);
    let (cx, cy) = view.center();
    let hw = view.viewport().width() / view.scale() * 0.5;
    let hh = view.viewport().height() / view.scale() * 0.5;
    let x0 = (cx - hw).max(0.0);
    let x1 = (cx + hw).min(world.width);
    let y0 = (cy - hh).max(0.0);
    let y1 = (cy + hh).min(world.height);
    let mut x = (x0 / step_units).floor() * step_units;
    while x <= x1 {
        let a = view.world_to_screen(x, y0);
        let b = view.world_to_screen(x, y1);
        p.line_segment([a, b], Stroke::new(1.0_f32, col));
        x += step_units;
    }
    let mut y = (y0 / step_units).floor() * step_units;
    while y <= y1 {
        let a = view.world_to_screen(x0, y);
        let b = view.world_to_screen(x1, y);
        p.line_segment([a, b], Stroke::new(1.0_f32, col));
        y += step_units;
    }
}

fn draw_obstacle(p: &Painter, view: &WorldView, o: &Obstacle) {
    let c = view.world_to_screen(o.x, o.y);
    let r = (o.r * view.scale()).max(1.5);
    let gray = Color32::from_rgb(108, 116, 124);
    if r > 5.0 {
        p.circle_filled(c, r + 2.0, Color32::from_rgba_unmultiplied(120, 128, 136, 60));
    }
    p.circle_filled(c, r, Color32::from_rgb(58, 64, 71));
    p.circle_stroke(c, r, Stroke::new(1.4_f32, gray));
}

fn draw_hazard(p: &Painter, view: &WorldView, h: &HazardZone) {
    let c = view.world_to_screen(h.x, h.y);
    let r = (h.r * view.scale()).max(2.0);
    let base = Color32::from_rgba_unmultiplied(248, 113, 113, 34);
    p.circle_filled(c, r, base);
    p.circle_filled(c, r * 0.62, Color32::from_rgba_unmultiplied(248, 113, 113, 36));
    // dashed ring
    let ring = Color32::from_rgba_unmultiplied(248, 113, 113, 190);
    let segs = 26.max((r * 0.8) as usize).min(72);
    for i in 0..segs {
        if i % 2 != 0 {
            continue;
        }
        let a0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        p.line_segment(
            [
                Pos2::new(c.x + r * a0.cos(), c.y + r * a0.sin()),
                Pos2::new(c.x + r * a1.cos(), c.y + r * a1.sin()),
            ],
            Stroke::new(1.3_f32, ring),
        );
    }
    // pulsing core
    let t = (h.phase * std::f32::consts::TAU / h.pulse_period.max(0.001)).sin();
    let _ = t;
}

fn draw_patch(p: &Painter, view: &WorldView, patch: &FoodPatch) {
    let c = view.world_to_screen(patch.x, patch.y);
    let r = (patch.r * view.scale()).max(2.0);
    let green = Color32::from_rgba_unmultiplied(52, 211, 153, 90);
    p.circle_filled(c, r, Color32::from_rgba_unmultiplied(52, 211, 153, 26));
    p.circle_stroke(c, r, Stroke::new(1.0_f32, green));
    p.circle_filled(c, 2.0, green);
}

fn food_stride(scale: f32, count: usize) -> usize {
    let lod = if scale < 0.16 {
        7
    } else if scale < 0.3 {
        3
    } else if scale < 0.55 {
        2
    } else {
        1
    };
    let pop_lod = if count > 2500 {
        4
    } else if count > 1400 {
        2
    } else {
        1
    };
    lod.max(pop_lod)
}

fn draw_food_dot(p: &Painter, view: &WorldView, f: &Food) {
    let scr = view.world_to_screen(f.x, f.y);
    if !view.viewport().contains(scr) {
        return;
    }
    let s = (f.energy / 44.0).clamp(0.35, 1.0);
    let col = Color32::from_rgb(126, 226, 148);
    p.add(Shape::rect_filled(Rect::from_center_size(scr, Vec2::new(2.6 * s, 2.6 * s)), 0.6, col));
}

fn draw_dot_org(p: &Painter, view: &WorldView, o: &Organism) {
    let c = view.world_to_screen(o.x, o.y);
    if !view.viewport().contains(c) {
        return;
    }
    let r = (organism_radius(&o.genotype.traits) * view.scale()).max(1.4);
    p.circle_filled(c, r, species_body_color(o.lineage, o.energy, o.max_energy));
}

fn draw_org(p: &Painter, view: &WorldView, o: &Organism, hover: bool) {
    let c = view.world_to_screen(o.x, o.y);
    if !view.viewport().expand(30.0).contains(c) {
        return;
    }
    let r = (organism_radius(&o.genotype.traits) * view.scale()).max(2.6);
    let dir = o.dir;
    let ca = dir.cos();
    let sa = dir.sin();
    let tip = pos2(c.x + ca * r, c.y + sa * r);
    let base = pos2(c.x - ca * r * 0.62, c.y - sa * r * 0.62);
    let side = Vec2::new(-sa, ca) * (r * 0.78);
    let left = pos2(base.x + side.x, base.y + side.y);
    let right = pos2(base.x - side.x, base.y - side.y);
    let fill = species_body_color(o.lineage, o.energy, o.max_energy);
    // darker outline ring: scales with on-screen size within sane bounds
    let stroke = Stroke::new(
        (0.6 + view.scale().sqrt() * 0.55).min(1.8),
        Color32::from_rgba_unmultiplied(10, 13, 16, 216),
    );
    p.add(Shape::convex_polygon(vec![tip, left, right], fill, stroke));
    // small heading tick just ahead of the nose
    let tick = pos2(c.x + ca * (r + 3.5), c.y + sa * (r + 3.5));
    p.line_segment(
        [tip, tick],
        Stroke::new(1.4_f32, Color32::from_rgba_unmultiplied(238, 243, 248, 150)),
    );
    // white hover ring when the throttled pick (or geometric hover) hits
    if hover {
        p.circle_stroke(
            c,
            r + 3.0,
            Stroke::new(1.4_f32, Color32::from_rgba_unmultiplied(240, 244, 249, 200)),
        );
    }
    // reach hint only when very zoomed in
    if r > 9.0 {
        let kb = pos2(c.x + ca * r * 1.7, c.y + sa * r * 1.7);
        p.line_segment([tip, kb], Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(221, 228, 234, 40)));
    }
}

fn draw_selection(p: &Painter, view: &WorldView, o: &Organism, sensor: bool, frame_seed: u64) {
    let c = view.world_to_screen(o.x, o.y);
    let r = (organism_radius(&o.genotype.traits) * view.scale()).max(3.0);
    let accent = Color32::from_rgb(45, 212, 191);
    // pulsing outer ring: +8 px, breathing 3*|sin(seed*0.05)|
    let pulse = (frame_seed as f32 * 0.05).sin().abs();
    let outer = r + 8.0 + 3.0 * pulse;
    if sensor {
        let sr = o.genotype.traits.sensor_range * view.scale();
        if sr > 8.0 && sr < 4000.0 {
            p.circle_stroke(c, sr, Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(45, 212, 191, 36)));
        }
    }
    p.circle_stroke(c, r + 5.0, Stroke::new(2.0_f32, accent));
    p.circle_stroke(c, outer, Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(45, 212, 191, 90)));
}

fn draw_trail_points(p: &Painter, view: &WorldView, trails: &Trails, o: &Organism) {
    let Some(q) = trails.points_of(o.id) else { return };
    let col = species_color(o.lineage);
    let n = q.len();
    for (k, (x, y)) in q.iter().enumerate() {
        let a = ((k + 1) as f32 / n.max(1) as f32 * 0.30).clamp(0.03, 0.3);
        let c = view.world_to_screen(*x, *y);
        if !view.viewport().contains(c) {
            continue;
        }
        p.circle_filled(c, 1.4, col.gamma_multiply(a));
    }
}

// ------------------------------------------------------- dynamic raster calc

fn compute_dyn(world: &WorldModel, orgs: &[Organism], foods: &[Food]) -> DynFields {
    let w = world.grid_w();
    let h = world.grid_h();
    let n = (w * h) as usize;
    let mut d = DynFields {
        w,
        h,
        density: vec![0; n],
        dominant: vec![0; n],
        energy: vec![0.0; n],
    };
    if w == 0 || h == 0 {
        return d;
    }
    let cw = world.width / w as f32;
    let ch = world.height / h as f32;
    let cell = |x: f32, y: f32| -> Option<usize> {
        if x < 0.0 || y < 0.0 || x >= world.width || y >= world.height {
            return None;
        }
        let cx = ((x / cw) as u32).min(w - 1);
        let cy = ((y / ch) as u32).min(h - 1);
        Some((cy * w + cx) as usize)
    };
    for f in foods.iter().filter(|f| f.alive) {
        if let Some(i) = cell(f.x, f.y) {
            d.density[i] += 1;
        }
    }
    for o in orgs {
        if let Some(i) = cell(o.x, o.y) {
            d.density[i] += 1;
            d.energy[i] += o.energy;
        }
    }
    // majority species per cell
    let mut counts: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
    for o in orgs {
        if let Some(i) = cell(o.x, o.y) {
            *counts.entry((i, o.lineage)).or_insert(0) += 1;
        }
    }
    let mut best: std::collections::HashMap<usize, (u32, u32)> = std::collections::HashMap::new();
    for ((cell, sp), count) in counts {
        let e = best.entry(cell).or_insert((sp, count));
        if count > e.1 {
            *e = (sp, count);
        }
    }
    for (i, (sp, _)) in best {
        if i < n {
            d.dominant[i] = sp;
        }
    }
    let max_e = d.energy.iter().copied().fold(0.0f32, f32::max);
    if max_e > 1.0 {
        for v in d.energy.iter_mut() {
            *v /= max_e;
        }
    }
    d
}

// ------------------------------------------------------------------ minimap

/// Small corner widget: terrain + organism dots + viewport rectangle.
/// Returns the world-space target the user clicked (for camera re-centering).
pub fn draw_minimap(
    ui: &mut egui::Ui,
    view: &mut WorldView,
    world: &WorldModel,
    organisms: &[Organism],
    max_size: Vec2,
) -> Option<(f32, f32)> {
    if world.width <= 0.0 || world.height <= 0.0 {
        return None;
    }
    let aspect = world.width / world.height;
    let mut size = max_size;
    if aspect > max_size.x / max_size.y {
        size.y = max_size.x / aspect;
    } else {
        size.x = max_size.y * aspect;
    }
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if !ui.is_rect_visible(rect) && ui.available_rect_before_wrap().intersects(rect) {
        // still draw: minimaps may be far into the scroll area
    }
    if let Some(tex) = view.terrain_texture(ui.ctx(), world) {
        ui.painter().image(
            tex.id(),
            rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    ui.painter().rect_stroke(rect, 0.0, Stroke::new(1.0_f32, Color32::from_rgb(35, 44, 53)));

    // weather tint: rain cover (coarse blue wash), effect rings, bolt markers
    draw_minimap_weather(&ui.painter(), rect, world.width, world.height, &view.weather);

    // organism dots (px-limited)

    // organism dots (px-limited)
    let p = ui.painter();
    let stride = (organisms.len() / 700).max(1);
    for (i, o) in organisms.iter().enumerate() {
        if i % stride != 0 {
            continue;
        }
        let px = rect.min.x + rect.width() * (o.x / world.width);
        let py = rect.min.y + rect.height() * (o.y / world.height);
        p.circle_filled(
            pos2(px, py),
            1.3,
            species_color(o.lineage),
        );
    }

    // viewport rectangle
    let (cx, cy) = view.center();
    let half = view.viewport().size() * 0.5 / view.scale();
    let (x0, y0) = (cx - half.x, cy - half.y);
    let (x1, y1) = (cx + half.x, cy + half.y);
    let a = pos2(
        rect.min.x + rect.width() * (x0 / world.width),
        rect.min.y + rect.height() * (y0 / world.height),
    );
    let b = pos2(
        rect.min.x + rect.width() * (x1 / world.width),
        rect.min.y + rect.height() * (y1 / world.height),
    );
    let vp = Rect::from_two_pos(a, b).intersect(rect);
    p.rect_stroke(vp, 0.0, Stroke::new(1.2_f32, Color32::from_rgb(45, 212, 191)));

    if resp.clicked() {
        let pos = (resp.interact_pointer_pos())?;
        let wx = ((pos.x - rect.min.x) / rect.width() * world.width).clamp(0.0, world.width);
        let wy = ((pos.y - rect.min.y) / rect.height() * world.height).clamp(0.0, world.height);
        return Some((wx, wy));
    }
    None
}


