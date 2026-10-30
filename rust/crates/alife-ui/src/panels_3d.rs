//! 3D living-world canvas (Mode B/C of the multi-scale camera). Agent D owns
//! this file for transition/follow UX. The M1 wire-up is complete:
//! `3` toggles 3D on, selecting an organism + Enter descends (follow cam),
//! Esc pops one level (Organism -> Habitat -> Planet -> back to 2D).
//! A habitat free-fly (`C`: WASD move, drag look, Q/E or wheel altitude),
//! click-to-select through the shared 2D-canvas selection state and a corner
//! HUD round out the experience.

use crate::app::AlifeApp;
use crate::theme;
use alife_core::SimApi;
use alife_view3d::camera::picking::{ground_height_near, pick_org};
use alife_view3d::camera::CameraMode as CamMode;
use alife_view3d::scene::{CamInput, Frame3D, OrgInst};
use alife_view3d::terrain::terrain_height_at;
use egui::{Align2, FontId, Key, Rect, Pos2};
use std::sync::Arc;

/// Toggle-safe 3D switch (only when the wgpu renderer is attached).
pub fn toggle(app: &mut AlifeApp) {
    if !app.view3d_ready {
        app.toasts.push(crate::widgets::ToastKind::Info, "3D renderer unavailable in this session (2D only)");
        return;
    }
    app.view3d_on = !app.view3d_on;
}

/// Descend to the currently selected organism (follow cam).
pub fn descend_to_selected(app: &mut AlifeApp) {
    if !app.view3d_ready {
        toggle(app);
        if !app.view3d_on {
            return;
        }
    }
    let id = match app.selected {
        Some(id) => id,
        None => {
            app.toasts.push(crate::widgets::ToastKind::Info, "Select an organism first (click one in world view)");
            return;
        }
    };
    if let Some(o) = app.sim.organism(id) {
        let at = (o.x, o.y);
        let th = terrain_height_at(&app.sim.world, at.0, at.1);
        app.arc_world(); // refresh the shared world snapshot cache
        app.cam3d.request_follow(id, at, th);
        app.mode3d = CamMode::Follow { id };
        app.view3d_on = true;
    } else {
        app.toasts.push(crate::widgets::ToastKind::Info, "That organism is no longer alive — select a living one");
    }
}

/// Step back one camera level. Returns true when 3D is exited.
pub fn pop_level(app: &mut AlifeApp) -> bool {
    match app.mode3d {
        CamMode::Follow { id } => {
            let f = app
                .sim
                .organism(id)
                .map(|o| (o.x, o.y))
                .unwrap_or((app.sim.world.width * 0.5, app.sim.world.height * 0.5));
            let th = terrain_height_at(&app.sim.world, f.0, f.1);
            app.cam3d.request_habitat(f, th);
            app.mode3d = CamMode::Orbit { focus: f };
            false
        }
        CamMode::Orbit { .. } => {
            app.mode3d = CamMode::Planet;
            false
        }
        CamMode::Planet => {
            app.view3d_on = false;
            true
        }
    }
}

fn cam_input(ui: &egui::Ui, free_cam: bool) -> CamInput {
    let mut ci = CamInput::default();
    ui.ctx().input(|i| {
        if i.pointer.primary_down() {
            ci.drag = (i.pointer.delta().x, i.pointer.delta().y);
        }
        ci.wheel += i.smooth_scroll_delta.y * 0.35;
        if i.key_down(Key::W) { ci.forward = true; }
        if i.key_down(Key::S) { ci.back = true; }
        if i.key_down(Key::A) { ci.left = true; }
        if i.key_down(Key::D) { ci.right = true; }
        // free-fly sub-set: Q descends (`down`), E climbs (a small wheel push
        // shares the existing altitude channel; CamInput has no `up` slot)
        if i.key_down(Key::Q) { ci.down = true; }
        if free_cam && i.key_down(Key::E) {
            ci.wheel += 2.6 * (if i.stable_dt.is_finite() && i.stable_dt > 0.0 { i.stable_dt } else { 1.0 / 60.0 });
        }
    });
    ci
}

pub fn canvas3d(app: &mut AlifeApp, ui: &mut egui::Ui, rect: Rect) {
    if !app.view3d_ready {
        ui.painter().rect_filled(rect, 6.0, theme::BG_PANEL);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "3D renderer unavailable (wgpu backend not active) — press 3 to return to 2D",
            egui::FontId::proportional(15.0),
            theme::WEAK,
        );
        return;
    }

    // habitat free-fly toggle (habitat view only; the camera parks the fly
    // state itself when the view changes)
    if matches!(app.mode3d, CamMode::Orbit { .. })
        && ui.ctx().input(|i| i.key_pressed(Key::C))
    {
        let on = !app.cam3d.free_cam();
        app.cam3d.set_free_cam(on);
        app.toasts.push(
            crate::widgets::ToastKind::Info,
            if on { "Free fly: WASD move · drag look · Q/E altitude" } else { "Free fly off — orbit controls" },
        );
    }

    let world = app.arc_world();
    let dt = app.dt_last;
    app.cam_t += dt;

    let mut ci = cam_input(ui, app.cam3d.free_cam());
    if !ui.rect_contains_pointer(rect) {
        ci.drag = (0.0, 0.0);
        ci.wheel = 0.0;
    }

    // target position of the followed organism this frame (or lose-on-death)
    let follow_target = match app.mode3d {
        CamMode::Follow { id } => app.sim.organism(id).map(|o| (o.x, o.y)),
        _ => None,
    };
    // dead follow falls back to orbit at the last known position
    if let CamMode::Follow { id } = app.mode3d {
        if follow_target.is_none() {
            let anchor = app.last_follow_pos.unwrap_or((app.sim.world.width * 0.5, app.sim.world.height * 0.5));
            let th = terrain_height_at(world.as_ref(), anchor.0, anchor.1);
            app.toasts.push(crate::widgets::ToastKind::Info, format!("Followed organism died (id {id}) — orbiting its last position"));
            app.cam3d.request_habitat(anchor, th);
            app.mode3d = CamMode::Orbit { focus: anchor };
        }
    }
    if let (CamMode::Follow { .. }, Some(p)) = (app.mode3d, follow_target) {
        app.last_follow_pos = Some(p);
    }

    let mode = app.mode3d;
    let default_focus = (world.width * 0.5, world.height * 0.5);
    let mode_focus = match mode {
        CamMode::Follow { .. } => follow_target.unwrap_or(default_focus),
        CamMode::Orbit { .. } => app.cam3d.look_ground(),
        CamMode::Planet => default_focus,
    };
    let th = terrain_height_at(world.as_ref(), mode_focus.0, mode_focus.1);
    app.cam3d.aspect = rect.width() / rect.height().max(1.0);
    app.cam3d.update(mode, &ci, (world.width, world.height), follow_target, th, dt);

    // organism instances bounded around the camera focus; ids kept in slice
    // order so a pick index maps straight back to a real organism id
    let radius = 220.0;
    let mut org_ids: Vec<u64> = Vec::new();
    let orgs: Vec<OrgInst> = app
        .sim
        .organisms()
        .iter()
        .filter(|o| o.alive)
        .filter(|o| (o.x - mode_focus.0).abs() + (o.y - mode_focus.1).abs() < radius)
        .take(1600)
        .map(|o| {
            org_ids.push(o.id);
            alife_view3d::scene::org_instance(
                o,
                alife_view3d::morph::brand_of(o),
                alife_view3d::morph::hue_of(o.lineage),
            )
        })
        .collect();

    // click-to-select, routed through the SAME selection state the 2D canvas
    // and inspector use (app.selected). Picking only while the pointer is
    // over the viewport and this is a clean click (not the end of a drag).
    let clicked: Option<Pos2> = if ui.rect_contains_pointer(rect) {
        ui.input(|i| {
            if i.pointer.primary_clicked() {
                i.pointer.latest_pos().filter(|p| rect.contains(*p))
            } else {
                None
            }
        })
    } else {
        None
    };
    if let Some(px) = clicked {
        let focus = app.cam3d.look_ground();
        let gh = ground_height_near(world.as_ref(), focus.0, focus.1);
        let (view, proj) = app.cam3d.matrices();
        let view_proj = (proj * view).to_cols_array_2d();
        let eye = app.cam3d.eye();
        match pick_org(&view_proj, [eye.x, eye.y, eye.z], rect, px, focus, gh, &orgs) {
            Some(idx) => {
                if let Some(id) = org_ids.get(idx as usize) {
                    app.selected = Some(*id);
                    app.show_inspector = true; // same open-inspector path as 2D
                }
            }
            None => {
                app.selected = None; // clicked empty ground: clear (as in 2D)
            }
        }
    }

    let (rain, cloud, flash) = app
        .hud_weather
        .as_ref()
        .map(|w| (w.rain.max(w.cloud), w.cloud, w.flash_age.map_or(0.0, |_| 0.8)))
        .unwrap_or((0.0, 0.0, 0.0));

    let cam = app.cam3d.uniform((world.width, world.height), app.cam_t, flash);
    let frame = Frame3D {
        cam,
        world: Arc::clone(&world),
        cam_mode: mode,
        dt,
        time: app.cam_t,
        viewport_px: (rect.width(), rect.height()),
        follow_target,
        orgs,
        rain_cover: rain,
        cloud_cover: cloud,
        flash,
    };
    alife_view3d::enqueue(ui, rect, frame);
    app.canvas_rect = Some(rect);

    paint_hud(app, ui, rect);
}

/// Corner HUD: camera mode, ground coords under the crosshair and the active
/// key hints. Bounded to <= 220 x 140 px, pass-through (no interaction).
fn paint_hud(app: &AlifeApp, ui: &egui::Ui, rect: Rect) {
    let pad = 8.0;
    let hud_w = 220.0_f32.min(rect.width() - 2.0 * pad).max(0.0);
    let free = app.cam3d.free_cam();
    let (mode_name, accent) = match app.mode3d {
        CamMode::Follow { id } => (format!("FOLLOW  #{id}"), theme::WARN),
        CamMode::Orbit { .. } => if free { ("HABITAT  FREE FLY".to_string(), theme::OK) } else { ("HABITAT".to_string(), theme::ACCENT) },
        CamMode::Planet => ("PLANET".to_string(), theme::ACCENT),
    };
    let (gx, gy) = app.cam3d.look_ground();
    let coords = format!("at ({:.0}, {:.0})", gx, gy);
    let hints: [&str; 3] = match app.mode3d {
        CamMode::Planet => ["Drag / Wheel orbit", "Enter: descend to select", "Esc: back to 2D"],
        CamMode::Orbit { .. } => if free {
            ["WASD move · Q/E alt", "Drag look · C exit fly", "Click select · Esc back"]
        } else {
            ["Drag / Wheel orbit", "WASD pan · C free fly", "Click select · Esc back"]
        },
        CamMode::Follow { .. } => ["Drag / Wheel zoom", "Enter: back to habitat", "Esc: pop one level"],
    };

    let line_h = 15.0;
    let rows = 3.0_f32;
    let hud_h = (rows * line_h + pad).min(140.0);
    let min = rect.min + egui::vec2(pad, pad);
    let p = ui.painter();
    p.rect_filled(
        Rect::from_min_size(min, egui::vec2(hud_w, hud_h)),
        4.0,
        theme::BG_PANEL2.gamma_multiply(0.88),
    );
    p.rect(
        Rect::from_min_size(min, egui::vec2(hud_w, hud_h)),
        4.0,
        theme::STROKE,
        egui::Stroke::new(1.0_f32, theme::STROKE),
    );
    let mono = FontId::monospace(11.0);
    p.text(
        min + egui::vec2(8.0, 11.0),
        Align2::LEFT_TOP,
        &mode_name,
        FontId::monospace(13.0),
        accent,
    );
    p.text(min + egui::vec2(8.0 + hud_w * 0.62, 11.0), Align2::LEFT_TOP, &coords, mono.clone(), theme::TEXT);
    for (n, h) in hints.iter().enumerate() {
        p.text(
            min + egui::vec2(8.0, 34.0 + n as f32 * line_h),
            Align2::LEFT_TOP,
            *h,
            mono.clone(),
            theme::WEAK,
        );
    }
}
