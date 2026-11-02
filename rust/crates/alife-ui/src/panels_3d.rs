//! 3D living-world canvas (Mode B/C of the multi-scale camera). Agent D owns
//! this file for transition/follow UX. The M1 wire-up is complete:
//! `3` toggles 3D on, selecting an organism + Enter descends (follow cam),
//! Esc pops one level (Organism -> Habitat -> Planet -> back to 2D).
//! A habitat free-fly (`C`: WASD move, drag look, Q/E or wheel altitude),
//! click-to-select through the shared 2D-canvas selection state and a corner
//! HUD round out the experience.
//!
//! Selection polish (this pass): every-other-frame hover pick through
//! VIEW3D's allocation-free `hover_org` (cursor ring + "id #N" chip), F9
//! picking diagnostics fed straight by `pick_org_with_diag` (hit radius,
//! candidates, ray + ground hit, selection, queen), GPU selection/queen
//! marker rings fed per-frame via `scene::markers::set_selection`, and a
//! queen status line in the corner HUD.

use crate::app::AlifeApp;
use crate::theme;
use alife_core::SimApi;
use alife_view3d::camera::picking::{
    ground_height_near, hover_org, pick_org, pick_org_with_diag, MAX_RAY_DIST,
};
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
    // state itself when the view changes). Not offered while the queen POV
    // sub-mode pilots the view.
    if matches!(app.mode3d, CamMode::Orbit { .. })
        && !app.cam3d.pov_active()
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

    // queen POV alpha channel: while the POV sub-mode is engaged the camera
    // state machine steers eye/look from the queen's live position + heading
    // (heading comes straight from the sim — `Organism.dir`); if she dies or
    // uncrowns mid-ride the un-override would freeze the gaze, so dismount
    // honestly into the lost-queen orbit.
    let queen_now = app
        .sim
        .queen_id()
        .and_then(|q| app.sim.organism(q as u64))
        .map(|o| ((o.x, o.y), o.dir));
    if app.cam3d.pov_active() {
        match queen_now {
            Some((at, dir)) => {
                ci.pov_target = Some((at.0, at.1, dir, alife_view3d::camera::QUEEN_BODY_LEN));
            }
            None => {
                let at = app
                    .cam3d
                    .pov_at()
                    .unwrap_or((world.width * 0.5, world.height * 0.5));
                app.cam3d.set_queen_pov(None);
                app.mode3d = CamMode::Orbit { focus: at };
                app.toasts.push(
                    crate::widgets::ToastKind::Info,
                    "Queen is gone — POV dismounted, orbiting her last position",
                );
            }
        }
    }

    // the bottom-left action button block steals pointer interaction from the
    // world: no orbit drag / wheel while pointing at it (click-pick is gated
    // with the same flag below)
    let over_buttons = ui
        .input(|i| i.pointer.latest_pos())
        .map(|p| cam_button_block(rect).contains(p))
        .unwrap_or(false);
    if over_buttons {
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
                app.sim.is_queen(o.id),
            )
        })
        .collect();

    // ---- shared per-frame pick context: click, hover, diagnostics and the
    // screen-space markers all read these (one source of truth per frame).
    let focus = app.cam3d.look_ground();
    let gh = ground_height_near(world.as_ref(), focus.0, focus.1);
    let (view_m, proj_m) = app.cam3d.matrices();
    let vp_m = proj_m * view_m; // world -> clip
    let view_proj = vp_m.to_cols_array_2d();
    let eye = app.cam3d.eye();
    let pointer_pos = if ui.rect_contains_pointer(rect) && !over_buttons {
        ui.input(|i| i.pointer.latest_pos().filter(|p| rect.contains(*p)))
    } else {
        None
    };

    // F9 toggles the picking diagnostics overlay while the pointer is over
    // the 3D canvas (the app-side toggle button lands the same field).
    if ui.rect_contains_pointer(rect) && ui.input(|i| i.key_pressed(Key::F9)) {
        app.picking_diag = !app.picking_diag;
        app.toasts.push(
            crate::widgets::ToastKind::Info,
            if app.picking_diag { "Pick diagnostics ON (F9)" } else { "Pick diagnostics OFF (F9)" },
        );
    }

    // click-to-select, routed through the SAME selection state the 2D canvas
    // and inspector use (app.selected). Picking only while the pointer is
    // over the viewport and this is a clean click (not the end of a drag).
    let clicked: Option<Pos2> = if ui.rect_contains_pointer(rect) && !over_buttons {
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

    // ---- hover pick (Agent D): mirrors the 2D forgiving-hover contract via
    // the shared hover_pick + hover_chip state (the 3D canvas is the sole
    // owner of that state while view3d_on). Every other frame — every frame
    // while diagnostics are on — through VIEW3D's allocation-free
    // `hover_org`, the same ray math the click uses, so hover and click
    // cannot disagree. No clicks are intercepted.
    if pointer_pos.is_none() || ui.input(|i| i.pointer.primary_down()) {
        app.hover_pick = None; // off-canvas or drag-look: no hover concept
    } else if app.picking_diag || app.frame_no % 2 == 0 {
        app.hover_pick = pointer_pos.and_then(|px| {
            hover_org(&view_proj, [eye.x, eye.y, eye.z], rect, px, focus, gh, &orgs)
        }).and_then(|cand| {
            let id = *org_ids.get(cand.org_index)?;
            let o = app.sim.organism(id)?;
            Some((id, o.x, o.y))
        });
    }
    if let Some((id, _, _)) = app.hover_pick {
        if app.hover_chip.0 != id {
            app.hover_chip = (id, format!("id #{id}"));
        }
    } else {
        app.hover_chip = (0, String::new());
    }

    // ---- diagnostics payload: honest click-test telemetry straight from
    // VIEW3D's `pick_org_with_diag` — the exact entry point a click resolves,
    // so the overlay cannot drift from click behavior. Allocated by marker
    // contract, so it runs only while the overlay is open.
    let f_y = proj_m.to_cols_array_2d()[1][1]; // 1/tan(fov_y/2)
    let diag = if app.picking_diag {
        let mut d = UiPickDiag {
            cursor: pointer_pos,
            ray: None,
            hit_t: None,
            ground_h: gh,
            hit_radius: 0.0,
            hit_radius_px: 0.0,
            hit_id: None,
            selected: app.selected,
            queen: app.sim.queen_id(),
            cands: Vec::new(),
        };
        if let Some(px) = pointer_pos {
            let (picked, pd) = pick_org_with_diag(
                &view_proj, [eye.x, eye.y, eye.z], rect, px, focus, gh, &orgs,
            );
            d.hit_id = picked.and_then(|i| org_ids.get(i).copied());
            d.ground_h = pd.ground_h;
            d.hit_radius = pd.hit_radius;
            d.cands = pd
                .candidates
                .iter()
                .map(|c| UiPickCand {
                    id: org_ids.get(c.org_index).copied().unwrap_or(0),
                    dist: c.hit_dist,
                    fwd: c.dist,
                })
                .collect();
            // ray dir is all-zero when no ray formed; grazing horizons come
            // back with `hit_radius = tol_at(MAX_RAY_DIST)` — shown honestly
            if pd.ray_dir.iter().any(|v| v.is_finite() && *v != 0.0) {
                d.ray = Some(pd.ray_dir);
                // ray travel to the ground-plane hit (same plane pick lays)
                let t = if pd.ray_dir[1].abs() > 1e-4 {
                    Some((pd.ground_h - eye.y) / pd.ray_dir[1])
                        .filter(|t| t.is_finite() && *t > 0.0)
                } else {
                    None
                }
                .map(|t| t.min(MAX_RAY_DIST));
                d.hit_t = t;
                if let Some(t) = t {
                    // base tolerance projected to pixels at that ray travel
                    d.hit_radius_px = (d.hit_radius * f_y * (rect.height() * 0.5) / t.max(1.0))
                        .max(2.0)
                        .min(rect.width().max(rect.height()));
                }
            }
        }
        Some(d)
    } else {
        None
    };

    // ---- GPU selection markers (VIEW3D `scene::markers`): indices are
    // positions in this frame's orgs slice (what pick/hover return), fed via
    // the global set_selection state and consumed by the renderer on the
    // SAME frame's callback -> NEXT update_gpu. Queen None falls back to the
    // renderer's is_queen flag scan; selection outside the render radius
    // simply draws no GPU marker (the HUD still shows it).
    let inst_of_id = |id: u64| org_ids.iter().position(|&i| i == id).map(|i| i as u32);
    let hover_idx = app.hover_pick.and_then(|(id, _, _)| inst_of_id(id));
    let queen_idx = app.sim.queen_id().and_then(|q| inst_of_id(q as u64));
    let sel_idx = app.selected.and_then(inst_of_id);
    alife_view3d::scene::markers::set_selection(sel_idx, hover_idx, queen_idx);

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

    // pass-2 overlays: drawn after the 3D callback, so they sit on top; pure
    // painter ops ⇒ nothing allocates input and clicks pass through. The
    // selected/hovered/queen body rings draw in the 3D pass itself (GPU
    // markers via `scene::markers::set_selection`); the hover chip and pick
    // diagnostics stay egui painters.
    if let Some(d) = &diag {
        paint_pick_diag(ui, rect, d);
    }
    paint_hover_ui(app, ui, pointer_pos);

    paint_hud(app, ui, rect);
    paint_view_diag(app, ui, rect);
    paint_cam_buttons(app, ui, rect);
}

/// Camera mode label shared by the corner HUD and the §7 diag block:
/// ("QUEEN FOCUS #id" / "FOLLOW #id" / "HABITAT [FREE FLY]" / "QUEEN POV" /
/// "PLANET", color accent).
fn cam_label(app: &AlifeApp) -> (String, egui::Color32) {
    let pov = app.cam3d.pov_active();
    let free = app.cam3d.free_cam();
    let queen_id = app.sim.queen_id().map(|q| q as u64);
    match app.mode3d {
        CamMode::Follow { id } => {
            if Some(id) == queen_id {
                (format!("QUEEN FOCUS  #{id}"), theme::OK)
            } else {
                (format!("FOLLOW  #{id}"), theme::WARN)
            }
        }
        CamMode::Orbit { .. } => {
            if pov {
                ("QUEEN POV".to_string(), theme::OK)
            } else if free {
                ("HABITAT  FREE FLY".to_string(), theme::OK)
            } else {
                ("HABITAT".to_string(), theme::ACCENT)
            }
        }
        CamMode::Planet => ("PLANET".to_string(), theme::ACCENT),
    }
}

/// Corner HUD: camera mode, ground coords under the crosshair and the active
/// key hints. Bounded to <= 220 x 140 px, pass-through (no interaction). The
/// queen POV / focus sub-modes relabel the mode row and report her coords.
fn paint_hud(app: &AlifeApp, ui: &egui::Ui, rect: Rect) {
    let pad = 8.0;
    let hud_w = 220.0_f32.min(rect.width() - 2.0 * pad).max(0.0);
    let free = app.cam3d.free_cam();
    let pov = app.cam3d.pov_active();
    let (mode_name, accent) = cam_label(app);    // while riding her POV the coords row reports the queen anchor itself
    let (gx, gy) = match (pov, app.cam3d.pov_at()) {
        (true, Some((x, y))) => (x, y),
        _ => app.cam3d.look_ground(),
    };
    let coords = format!("at ({:.0}, {:.0})", gx, gy);
    let queen_id = app.sim.queen_id().map(|q| q as u64);
    let hints: [&str; 4] = match app.mode3d {
        CamMode::Planet => ["Drag / Wheel orbit", "Enter: descend to select", "Esc: back to 2D", "F9: pick diag"],
        CamMode::Orbit { .. } => if pov {
            ["POV gaze from the queen head", "Drag is ignored · click selects", "Esc: exit to world orbit", "F9: pick diag"]
        } else if free {
            ["WASD move · Q/E alt", "Drag look · C exit fly", "Click select · Esc back", "F9: pick diag"]
        } else {
            ["Drag / Wheel orbit", "WASD pan · C free fly", "Click select · Esc back", "F9: pick diag"]
        },
        CamMode::Follow { id } => if Some(id) == queen_id {
            ["Drag / Wheel zoom", "Esc: back to world orbit", "Enter: refocus her", "F9: pick diag"]
        } else {
            ["Drag / Wheel zoom", "Enter: back to habitat", "Esc: pop one level", "F9: pick diag"]
        },
    };

    let line_h = 15.0;
    // mode row + 4 hint rows (+ queen status row when the sim has one)
    let queen_line = queen_status(app);
    let rows = 34.0 + hints.len() as f32 * line_h + queen_line.is_some() as u32 as f32 * line_h;
    let hud_h = (rows + pad).min(140.0);
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
    if let Some((txt, color, follow_txt)) = queen_line {
        let y = 34.0 + hints.len() as f32 * line_h;
        p.text(min + egui::vec2(8.0, y), Align2::LEFT_TOP, &txt, mono.clone(), color);
        if let Some(ft) = follow_txt {
            p.text(min + egui::vec2(8.0 + hud_w * 0.66, y), Align2::LEFT_TOP, ft, mono, theme::WEAK);
        }
    }
}

/// Queen UX surface: "QUEEN #id alive/dead". Alive comes straight from
/// `sim.queen_id()`; a dead (or formerly crowned) queen degrades honestly to
/// the last queen event in the sim's bounded ring. None when no queen was
/// ever crowned this session.
fn queen_status(app: &AlifeApp) -> Option<(String, egui::Color32, Option<&'static str>)> {
    if let Some(q) = app.sim.queen_id() {
        if app.cam3d.pov_active() {
            Some((format!("QUEEN #{q} alive"), theme::OK, Some("POV riding")))
        } else if app.selected == Some(q as u64) {
            Some((format!("QUEEN #{q} alive"), theme::OK, Some("following")))
        } else {
            Some((format!("QUEEN #{q} alive"), theme::OK, Some("click her")))
        }
    } else {
        app.sim
            .queen_events()
            .last()
            .map(|ev| (format!("QUEEN #{} dead", ev.org_id), theme::DANGER, None))
    }
}

// ---------------------------------------------------------------- §7 diag

/// §7 diagnostics mini-panel (top-right, always visible while the 3D layer
/// is on): three mono rows, pass-through painter only.
///
/// * `GRASS clumps n chunks c LOD a/b/c trample r` — live renderer state:
///   `Grass::diag_counts` (resident clump cache + drawn chunks) and
///   `Grass::diag_lod` (near/mid/far chunk histogram of the last draw); the
///   trample radius is the morph planner's real queen footprint radius.
/// * `QUEEN len | r | trample | pos | speed | brain` — length and radius are
///   the documented queen constants (`QUEEN_BODY_LEN`, `2*QUEEN_UNIT_RADIUS`);
///   position rides the planned trample center when available (render-side
///   truth) and falls back to the sim organism; speed is her live `o.speed`
///   (u/s, cheap single-arena lookup); brain is OK iff the plugged brain
///   reports synapses and no factory failure, "err" otherwise, "-" crownless.
/// * `CAM ... | QUEEN ...` — camera label + queen alive/id. Everything is
///   real (no approximated values) because the diag accessors sit on the
///   persistent View3D in egui's callback resources, read UI-side through
///   the stored `RenderState`; all of them are Relaxed-atomic / short-mutex
///   and lock-free for the render thread's lane.
fn paint_view_diag(app: &AlifeApp, ui: &egui::Ui, rect: Rect) {
    if !app.view3d_on {
        return;
    }
    let Some(renderer_lock) = app.view3d_renderer.as_ref() else { return };
    let renderer = renderer_lock.read();
    let Some(view) = renderer.callback_resources.get::<alife_view3d::View3D>() else {
        return;
    };
    let (clumps, chunks) = view.grass.diag_counts();
    let (lod_n, lod_m, lod_f) = view.grass.diag_lod();
    let fp = view.morph.queen_footprint();
    let (tr_r, tr_at) = fp
        .map(|(x, z, r, _)| (Some(r), Some((x, z))))
        .unwrap_or((None, None));

    let queen = app
        .sim
        .queen_id()
        .and_then(|q| app.sim.organism(q as u64))
        .map(|o| ((o.x, o.y), o.speed));
    let q_pos = tr_at.or(queen.map(|(p, _)| p));
    let q_speed = queen.map(|(_, s)| s).filter(|v| v.is_finite());
    let brain = match app.sim.queen_id() {
        Some(_) => match app.sim.queen_brain_stats() {
            Some(s) if s.synapse_count > 0 && app.queen_factory_err.is_none() => "OK",
            _ => "err",
        },
        None => "-",
    };

    let grass_line = format!(
        "GRASS clumps {clumps} chunks {chunks} LOD {lod_n}/{lod_m}/{lod_f} trample {}",
        tr_r.map(|r| format!("r{r:.1}")).unwrap_or_else(|| "r-".into()),
    );
    let queen_line = format!(
        "QUEEN len {:.1}u | r {:.1}u | trample r{} | pos ({}) | {} | brain {}",
        alife_view3d::camera::QUEEN_BODY_LEN,
        2.0 * alife_view3d::morph::decomp::QUEEN_UNIT_RADIUS,
        tr_r.map(|r| format!("{r:.1}")).unwrap_or_else(|| "-".into()),
        q_pos
            .map(|(x, z)| format!("{x:.0},{z:.0}"))
            .unwrap_or_else(|| "-,-".into()),
        q_speed
            .map(|s| format!("speed {s:.1}u/s"))
            .unwrap_or_else(|| "speed -".into()),
        brain,
    );
    let q_state = match app.sim.queen_id() {
        Some(q) => format!("QUEEN #{q} alive"),
        None => app
            .sim
            .queen_events()
            .last()
            .map(|ev| format!("QUEEN #{} dead", ev.org_id))
            .unwrap_or_else(|| "QUEEN none".into()),
    };
    let (cam_txt, cam_col) = cam_label(app);
    let cam_line = format!("CAM {cam_txt} | {q_state}");

    let pad = 8.0;
    let line_h = 15.0;
    let rows: [(&str, egui::Color32); 3] = [
        (grass_line.as_str(), theme::WEAK),
        (queen_line.as_str(), theme::WEAK),
        (cam_line.as_str(), cam_col),
    ];
    // measure once per frame (3 cheap galleys) so the backdrop fits exactly
    let mono = FontId::monospace(10.0);
    let p = ui.painter();
    let widest = rows
        .iter()
        .map(|(t, _)| p.layout_no_wrap((*t).into(), mono.clone(), theme::WEAK).rect.width())
        .fold(0.0_f32, f32::max)
        + 10.0;
    let top_right = Pos2::new(rect.max.x - pad, rect.min.y + pad);
    let bg = Rect::from_min_size(
        Pos2::new(top_right.x - widest, top_right.y),
        egui::vec2(widest, 3.0 * line_h + 6.0),
    );
    p.rect_filled(bg, 4.0, theme::BG_PANEL2.gamma_multiply(0.82));
    p.rect(
        bg,
        4.0,
        theme::STROKE,
        egui::Stroke::new(0.75_f32, theme::STROKE),
    );
    for (n, (txt, col)) in rows.iter().enumerate() {
        p.text(
            Pos2::new(top_right.x, top_right.y + 3.0 + n as f32 * line_h),
            Align2::RIGHT_TOP,
            *txt,
            mono.clone(),
            *col,
        );
    }
}

// ---------------------------------------------------------------- queen cam

/// Bottom-left quick-cam action row: [FOCUS QUEEN] [QUEEN POV] [WORLD VIEW].
/// Sizing constants are shared with [`cam_button_block`] which mirrors the
/// Area rect for the canvas hover/pick gating (`over_buttons` in `canvas3d`).
const CAM_BTN_W: f32 = 92.0;
const CAM_BTN_H: f32 = 20.0;
const CAM_BTN_GAP: f32 = 6.0;
const CAM_BTN_ID: &str = "cam3d_quick_actions";

fn cam_button_block(rect: Rect) -> Rect {
    let size = egui::vec2(CAM_BTN_W * 3.0 + CAM_BTN_GAP * 2.0, CAM_BTN_H);
    Rect::from_min_size(
        Pos2::new(rect.min.x + 10.0, rect.max.y - 10.0 - size.y),
        size,
    )
}

fn paint_cam_buttons(app: &mut AlifeApp, ui: &egui::Ui, rect: Rect) {
    let block = cam_button_block(rect);
    egui::Area::new(egui::Id::new(CAM_BTN_ID))
        .fixed_pos(block.min)
        .show(ui.ctx(), |ui| {
            let gap = CAM_BTN_GAP;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                let queen = app
                    .sim
                    .queen_id()
                    .and_then(|qid| app.sim.organism(qid as u64))
                    .map(|o| ((o.x, o.y), o.dir, o.id));
                let btn = |label: &str, dim: bool| {
                    let mut rt = egui::RichText::new(label).font(FontId::monospace(10.0));
                    if dim {
                        rt = rt.color(theme::WEAK);
                    }
                    egui::Button::new(rt).small()
                };
                let sz = egui::vec2(CAM_BTN_W, CAM_BTN_H);

                // --- FOCUS QUEEN: third-person follow cam on her
                let r = ui.add_sized(sz, btn("FOCUS QUEEN", queen.is_none()));
                if r.clicked() {
                    match queen {
                        Some((at, _, q)) if ensure_view3d(app) => {
                            let th = terrain_height_at(&app.sim.world, at.0, at.1);
                            app.arc_world();
                            app.selected = Some(q as u64); // same path as click-select
                            app.show_inspector = true;
                            app.cam3d.request_queen_focus(q as u64, at, th);
                            app.mode3d = CamMode::Follow { id: q as u64 };
                            app.toasts.push(
                                crate::widgets::ToastKind::Ok,
                                format!("Camera locked on the queen (#{q})"),
                            );
                        }
                        _ => app.toasts.push(
                            crate::widgets::ToastKind::Info,
                            "No queen crowned yet — nothing to focus",
                        ),
                    }
                }

                // --- QUEEN POV: first person from her crown (toggle; WORLD
                // VIEW also exits)
                let pov = app.cam3d.pov_active();
                let r = ui.add_sized(sz, btn(if pov { "EXIT POV" } else { "QUEEN POV" }, queen.is_none() && !pov));
                if r.clicked() {
                    if pov {
                        let center = (app.sim.world.width * 0.5, app.sim.world.height * 0.5);
                        let at = app.cam3d.pov_at().unwrap_or(center);
                        app.cam3d.set_queen_pov(None);
                        app.mode3d = CamMode::Orbit { focus: at };
                        app.toasts.push(crate::widgets::ToastKind::Info, "POV off — orbiting the queen");
                    } else if let Some((at, dir, q)) = queen {
                        if ensure_view3d(app) {
                            app.arc_world();
                            app.selected = Some(q as u64);
                            app.cam3d.set_queen_pov(Some((
                                q as u64,
                                at,
                                dir,
                                alife_view3d::camera::QUEEN_BODY_LEN,
                            )));
                            app.mode3d = CamMode::Orbit { focus: at };
                            app.toasts.push(
                                crate::widgets::ToastKind::Ok,
                                "POV: riding the queen's gaze",
                            );
                        }
                    } else {
                        app.toasts.push(
                            crate::widgets::ToastKind::Info,
                            "No queen crowned yet — nothing to ride",
                        );
                    }
                }

                // --- WORLD VIEW: always available; pops POV/follow/fly back
                // to the wide habitat anchor (queen pos when she exists)
                let r = ui.add_sized(sz, btn("WORLD VIEW", false));
                if r.clicked() {
                    let center = (app.sim.world.width * 0.5, app.sim.world.height * 0.5);
                    let anchor = queen
                        .map(|(at, _, _)| at)
                        .unwrap_or(center);
                    let th = terrain_height_at(&app.sim.world, anchor.0, anchor.1);
                    if ensure_view3d(app) {
                        app.arc_world();
                        app.cam3d.return_to_world_orbit(Some(anchor), th);
                        app.mode3d = CamMode::Orbit { focus: anchor };
                        app.toasts.push(crate::widgets::ToastKind::Info, "Back to the world view");
                    }
                }
            });
        });
}

/// 3D-on guarantee for button actions (same honest toast path as the `3`
/// toggle): returns whether the renderer is on and ready.
fn ensure_view3d(app: &mut AlifeApp) -> bool {
    if !app.view3d_ready {
        toggle(app);
        if !app.view3d_on {
            return false;
        }
    }
    app.view3d_on = true;
    true
}

// ---------------------------------------------------------------- overlays

/// UI view of the diagnostics bundle for the overlay (F9). Sourced from
/// VIEW3D's `pick_org_with_diag` (candidate ids resolved against the
/// frame's org_id slice) plus the app's selection/queen state.
struct UiPickDiag {
    cursor: Option<Pos2>,
    /// normalized view ray through the cursor (x, y, z)
    ray: Option<[f32; 3]>,
    /// ray travel to the ground-plane hit (world units), None when grazing
    hit_t: Option<f32>,
    ground_h: f32,
    /// world-unit base pick tolerance at that ray travel (`PickDiag::hit_radius`)
    hit_radius: f32,
    /// the same tolerance projected to pixels at the cursor
    hit_radius_px: f32,
    hit_id: Option<u64>,
    selected: Option<u64>,
    queen: Option<u32>,
    /// candidates inside tolerance, nearest first
    cands: Vec<UiPickCand>,
}

/// One in-tolerance organism under the cursor ray.
struct UiPickCand {
    id: u64,
    /// distance from the ground hit to the organism center (world units)
    dist: f32,
    /// ray travel to the organism (world units)
    fwd: f32,
}

/// Hover UX: light cursor ring + "id #N" chip (same text contract as 2D).
/// Pure painter at the cursor — click routing is untouched.
fn paint_hover_ui(app: &AlifeApp, ui: &mut egui::Ui, pointer: Option<Pos2>) {
    let Some(px) = pointer else { return };
    if app.hover_pick.is_none() {
        return;
    }
    let p = ui.painter();
    p.circle_stroke(px, 9.0, egui::Stroke::new(1.5_f32, theme::ACCENT.gamma_multiply(0.55)));
    p.text(
        px + egui::vec2(12.0, -10.0),
        Align2::LEFT_BOTTOM,
        &app.hover_chip.1,
        FontId::monospace(10.5),
        theme::TEXT,
    );
}

/// Diagnostics overlay (top-right of the canvas): cursor, ray, ground hit,
/// hit radius (world + on-screen ring), top candidates, current click pick,
/// selection and queen. Painter-only; no input is consumed.
fn paint_pick_diag(ui: &mut egui::Ui, rect: Rect, d: &UiPickDiag) {
    let p = ui.painter();
    // the click test's world tolerance drawn as a ring at the cursor
    if let (Some(px), true) = (d.cursor, d.hit_radius_px > 0.0) {
        p.circle_stroke(px, d.hit_radius_px, egui::Stroke::new(1.0_f32, theme::WARN.gamma_multiply(0.45)));
    }

    let mono = FontId::monospace(10.5);
    let (mut lines, mut colors): (Vec<String>, Vec<egui::Color32>) = (
        vec![
            "PICK DIAG  (F9 off)".to_string(),
            match d.cursor {
                Some(px) => format!("cursor {:.0},{:.0}", px.x, px.y),
                None => "cursor — (no pointer)".to_string(),
            },
        ],
        vec![theme::WARN, theme::TEXT],
    );
    let (ray_s, ray_c) = match d.ray {
        Some(r) => (
            format!("ray ({:.2},{:.2},{:.2})", r[0], r[1], r[2]),
            theme::TEXT,
        ),
        None => ("ray — (degenerate pixel)".to_string(), theme::WEAK),
    };
    lines.push(ray_s);
    colors.push(ray_c);
    let (t_s, t_c) = match d.hit_t {
        Some(t) => (format!("gnd h {:.1} · t {:.0}", d.ground_h, t), theme::TEXT),
        None => (format!("gnd h {:.1} · t — (grazing)", d.ground_h), theme::WEAK),
    };
    lines.push(t_s);
    colors.push(t_c);
    lines.push(format!("tol {:.1} u ≈ {:.0} px", d.hit_radius, d.hit_radius_px));
    colors.push(theme::TEXT);
    let pick_s = match d.hit_id {
        Some(id) => format!("click picks id #{id}"),
        None => "click picks ground (deselect)".to_string(),
    };
    lines.push(pick_s);
    colors.push(if d.hit_id.is_some() { theme::OK } else { theme::WEAK });
    lines.push(format!(
        "cands {} · sel {} · queen {}",
        d.cands.len(),
        d.selected.map(|s| s.to_string()).unwrap_or_else(|| "—".into()),
        d.queen.map(|q| format!("#{q}")).unwrap_or_else(|| "—".into())
    ));
    colors.push(theme::TEXT);
    for c in d.cands.iter().take(8) {
        lines.push(format!("id #{} d {:.1}u fwd {:.0}", c.id, c.dist, c.fwd));
        colors.push(theme::WEAK);
    }

    let line_h = 13.0;
    let pad = 8.0;
    let w = 216.0_f32.min(rect.width() - 2.0 * pad).max(0.0);
    let h = (lines.len() as f32) * line_h + 2.0 * pad;
    let min = (rect.max - egui::vec2(w + pad, h + pad)).max(rect.min);
    p.rect_filled(Rect::from_min_size(min, egui::vec2(w, h)), 4.0, theme::BG_PANEL2.gamma_multiply(0.88));
    p.rect(
        Rect::from_min_size(min, egui::vec2(w, h)),
        4.0,
        theme::STROKE,
        egui::Stroke::new(1.0_f32, theme::STROKE),
    );
    for (n, l) in lines.iter().enumerate() {
        p.text(min + egui::vec2(7.0, pad + 1.0 + n as f32 * line_h), Align2::LEFT_TOP, l, mono.clone(), colors[n]);
    }
}
