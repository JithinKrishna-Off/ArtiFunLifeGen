//! 3D living-world canvas (Mode B/C of the multi-scale camera). Agent D owns
//! this file for transition/follow UX. The M1 wire-up below is complete:
//! `3` toggles 3D on, selecting an organism + Enter descends (follow cam),
//! Esc pops one level (Organism -> Habitat -> Planet -> back to 2D).

use crate::app::AlifeApp;
use crate::theme;
use alife_core::SimApi;
use alife_view3d::camera::CameraMode as CamMode;
use alife_view3d::scene::{CamInput, Frame3D, OrgInst};
use alife_view3d::terrain::terrain_height_at;
use egui::{Key, Rect};
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

fn cam_input(ui: &egui::Ui) -> CamInput {
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

    let world = app.arc_world();
    let dt = app.dt_last;
    app.cam_t += dt;

    let mut ci = cam_input(ui);
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

    // organism instances bounded around the camera focus
    let radius = 220.0;
    let orgs: Vec<OrgInst> = app
        .sim
        .organisms()
        .iter()
        .filter(|o| o.alive)
        .filter(|o| (o.x - mode_focus.0).abs() + (o.y - mode_focus.1).abs() < radius)
        .take(1600)
        .map(|o| {
            alife_view3d::scene::org_instance(
                o,
                alife_view3d::morph::brand_of(o),
                alife_view3d::morph::hue_of(o.lineage),
            )
        })
        .collect();

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
}
