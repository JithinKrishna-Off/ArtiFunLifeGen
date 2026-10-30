//! The world canvas: shared draw path for World + World Editor views,
//! follow-cam, editor tool application, minimap, extinction banner and the
//! first-run hint overlay.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets::{self, ToastKind};
use alife_common::InteractionMode;
use alife_core::SimApi;
use alife_render::{
    draw_minimap, draw_world, BrushKind, CanvasResponse, DrawConfig, WeatherLayer,
};
use egui::{Align2, Context, Id, Order, Rect, Stroke, Ui, Vec2};

/// God armed tool -> render brush preview kind (colors mirror effect rings).
fn brush_kind(tool: crate::god_api::GodTool) -> BrushKind {
    use crate::god_api::GodTool as T;
    match tool {
        T::Rain => BrushKind::Rain,
        T::Drought => BrushKind::Drought,
        T::Storm | T::Lightning => BrushKind::Lightning,
        T::NutrientDump => BrushKind::Dump,
        T::Summon => BrushKind::Summon,
    }
}

/// Shared canvas draw path. `draw_world` internally records the viewport
/// (`begin_viewport`) and runs pan/zoom/click interaction (`handle_interaction`).
pub fn canvas(app: &mut AlifeApp, ui: &mut Ui, rect: Rect) -> CanvasResponse {
    // 3D layer replaces the 2D canvas visuals while active
    if app.view3d_on {
        crate::panels_3d::canvas3d(app, ui, rect);
        return CanvasResponse::default();
    }

    // follow camera: re-center on the selected organism before painting
    if app.follow_selected {
        if let Some(id) = app.selected {
            if let Some(o) = app.sim.organism(id) {
                app.view.set_center(o.x, o.y);
            }
        }
    }

    let pop = app.sim.stats().population as usize;
    let every = app.quality.trail_every();
    let capture = app.running && every > 0 && pop < 900 && app.frame_no % every == 0;

    // weather pass only when the 10 Hz HUD snapshot says it is visible; the
    // render-side layer snapshot is refreshed by the same 10 Hz clock in app.rs
    let weather_idle = app
        .hud_weather
        .as_ref()
        .map_or(true, |w| w.idle());
    let weather: Option<WeatherLayer> = if weather_idle {
        None
    } else {
        app.weather_layer.clone()
    };
    // hand the shared snapshot to the minimap pass as well
    if let Some(w) = &weather {
        app.view.set_weather_layer(w.clone());
    } else {
        app.view.set_weather_layer(WeatherLayer::default());
    }

    // hover id fed into the renderer (white hover ring); reuses the existing
    // ~10 Hz throttled pick below (refresh cadence frame_no % 6)
    let hover_id = if app.hud_weather.is_some() || app.last_hover.is_some() {
        app.hover_pick.map(|(id, _, _)| id)
    } else {
        None
    };

    let cfg = DrawConfig {
        trail_capture: capture,
        interactions: true,
        hover_id,
        frame_seed: app.frame_no,
        weather,
        ..Default::default()
    };

    let resp = draw_world(
        ui,
        &mut app.view,
        rect,
        &app.sim.world,
        app.sim.organisms(),
        app.sim.foods(),
        app.sim.edits(),
        app.selected,
        (0.0, 0.0),
        &cfg,
    );

    app.canvas_rect = Some(rect);
    if resp.hovering {
        app.last_hover = Some(resp.hover);
    }
    if resp.dragged {
        app.follow_selected = false; // manual pan cancels follow
    }

    // god click-target mode: armed brush preview (render crate draws it)
    if app.god_tool.is_some() && resp.hovering {
        if let Some(tool) = app.god_tool {
            let preview_painter = ui.painter().clone();
            alife_render::draw_brush_preview(
                &preview_painter,
                &app.view,
                resp.hover.0,
                resp.hover.1,
                app.god_radius,
                brush_kind(tool),
            );
        }
    }

    // editor tool brush preview
    if app.editor_mode != InteractionMode::Inspect && resp.hovering {
        let c = app.view.world_to_screen(resp.hover.0, resp.hover.1);
        let r = (app.editor_brush * resp.scale).max(3.0);
        ui.painter().circle_stroke(
            c,
            r,
            Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(45, 212, 191, 120)),
        );
    }

    // forgiving hover pick, throttled to ~10 Hz: feeds both the renderer's
    // hover ring (next frame) and the small "id #N" chip near the organism
    if resp.hovering && app.frame_no % 6 == 0 {
        let r = app.pick_radius_px / resp.scale.max(1e-3);
        let picked = app.sim.pick(resp.hover.0, resp.hover.1, r);
        app.hover_pick = picked.map(|id| (id, resp.hover.0, resp.hover.1));
    }
    if let Some((id, _, _)) = app.hover_pick {
        if app.hover_chip.0 != id {
            app.hover_chip = (id, format!("id #{id}"));
        }
        if let Some(o) = app.sim.organism(id) {
            let c = app.view.world_to_screen(o.x, o.y) + egui::vec2(9.0, -15.0);
            let text = &app.hover_chip.1;
            let p = ui.painter();
            p.text(c, egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(10.5), egui::Color32::from_rgba_unmultiplied(10, 13, 17, 230));
            p.text(c + egui::vec2(-0.5, -0.5), egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(10.5), crate::theme::TEXT);
        }
    }

    if let Some((x, y)) = resp.click {
        apply_click(app, x, y, resp.scale.max(1e-3));
    }
    resp
}

#[allow(clippy::too_many_lines)]
fn apply_click(app: &mut AlifeApp, x: f32, y: f32, scale: f32) {
    // armed god act takes precedence (fire there, then disarm)
    if let Some(tool) = app.god_tool {
        let ev = crate::god_api::build_event(tool, x, y, app.god_radius);
        app.god_tool = None;
        crate::panels_god::fire_god_event(app, ev);
        return;
    }
    match app.editor_mode {
        InteractionMode::Inspect => {
            let r = app.pick_radius_px / scale;
            let picked = app.sim.pick(x, y, r);
            if picked.is_some() {
                app.selected = picked;
                app.show_inspector = true;
            } else {
                app.selected = None;
            }
        }
        InteractionMode::Food => {
            if !app.sim.add_food(x, y) {
                app.toasts
                    .push(ToastKind::Warn, "Food limit reached (max_food)");
            }
        }
        InteractionMode::Obstacle => {
            if !app.sim.add_obstacle(x, y, app.editor_brush) {
                app.toasts.push(ToastKind::Warn, "Could not place obstacle");
            }
        }
        InteractionMode::Patch => {
            if !app.sim.add_patch(x, y, app.editor_brush) {
                app.toasts.push(ToastKind::Warn, "Could not place food patch");
            }
        }
        InteractionMode::Hazard => {
            if !app.sim.add_hazard(x, y, app.editor_brush) {
                app.toasts.push(ToastKind::Warn, "Could not place hazard");
            }
        }
        InteractionMode::Erase => {
            app.sim.remove_at(x, y, app.editor_brush);
        }
    }
}

// -------------------------------------------------------------- World view

pub fn world_view(app: &mut AlifeApp, ctx: &Context) {
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(theme::BG_WINDOW))
        .show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            canvas(app, ui, rect);
        });

    overlays(app, ctx);
}

fn overlays(app: &mut AlifeApp, ctx: &Context) {
    let Some(rect) = app.canvas_rect else { return };
    let sr = ctx.screen_rect();

    // armed god tool chip + weather HUD over the viewport
    crate::panels_god::armed_strip(app, ctx);
    crate::panels_god::weather_hud(app, ctx);

    // minimap, bottom-right of the canvas
    let offset = egui::vec2(
        (rect.max.x - sr.max.x) - 12.0,
        -((sr.max.y - rect.max.y) + 12.0),
    );
    egui::Area::new(Id::new("alife.minimap"))
        .order(Order::Middle)
        .anchor(Align2::RIGHT_BOTTOM, offset)
        .show(ctx, |ui| {
            if let Some((wx, wy)) =
                draw_minimap(ui, &mut app.view, &app.sim.world, app.sim.organisms(), Vec2::new(200.0, 150.0))
            {
                app.view.set_center(wx, wy);
                app.follow_selected = false;
            }
        });

    // extinction banner, top-center of the canvas
    if app.sim.stats().population == 0 && app.was_populated {
        let offset = egui::vec2(rect.center().x - sr.center().x, rect.min.y - sr.min.y + 10.0);
        egui::Area::new(Id::new("alife.extinct"))
            .order(Order::Middle)
            .anchor(Align2::CENTER_TOP, offset)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(theme::BG_PANEL2)
                    .rounding(egui::Rounding::same(6.0))
                    .stroke(egui::Stroke::new(1.0_f32, theme::WARN))
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.label(
                                widgets::strong("Population collapsed — your ecosystem went extinct"),
                            );
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if widgets::mini_btn(ui, "Reseed 50 organisms", "Spawn 50 fresh founder organisms")
                                    .clicked()
                                {
                                    app.sim.spawn_organisms(50);
                                    app.was_populated = true;
                                    app.toasts
                                        .push(ToastKind::Ok, "Reseeded 50 founder organisms");
                                }
                                if widgets::mini_btn(ui, "New world", "Open the New World dialog").clicked()
                                {
                                    app.dialog = Some(crate::dialogs::Dialog::new_world_scratch(app));
                                }
                            });
                        });
                    });
            });
    }
}

// --------------------------------------------------------- hint overlay

pub fn hint_overlay(app: &mut AlifeApp, ctx: &Context) {
    if !app.show_hint {
        return;
    }
    egui::Area::new(Id::new("alife.hint"))
        .order(Order::Foreground)
        .anchor(Align2::CENTER_CENTER, [0.0, -40.0])
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(8.0))
                .stroke(egui::Stroke::new(1.0_f32, theme::ACCENT.gamma_multiply(0.7)))
                .inner_margin(18.0)
                .show(ui, |ui| {
                    ui.set_max_width(420.0);
                    ui.vertical(|ui| {
                        ui.label(widgets::strong("Welcome to Artificial Life"));
                        ui.add_space(8.0);
                        ui.label(widgets::weak("Space to play / pause"));
                        ui.label(widgets::weak("Click an organism to select and inspect it"));
                        ui.label(widgets::weak("Wheel to zoom, drag to pan"));
                        ui.label(widgets::weak("I toggles the inspector panel"));
                        ui.label(widgets::weak("? shows all shortcuts"));
                        ui.add_space(12.0);
                        ui.horizontal(|ui| {
                            if widgets::mini_btn(ui, "Got it", "Dismiss this hint (30 s auto-dismiss)")
                                .clicked()
                            {
                                app.show_hint = false;
                            }
                            if widgets::mini_btn(ui, "All shortcuts", "Open the shortcut cheatsheet")
                                .clicked()
                            {
                                app.show_shortcuts = true;
                                app.show_hint = false;
                            }
                        });
                        ui.add_space(4.0);
                        ui.label(widgets::mono(
                            format!("auto-dismiss in {:.0} s", (30.0 - app.hint_age).max(0.0)),
                            theme::WEAK,
                        ));
                    });
                });
        });
}

// --------------------------------------------------------- hint overlay