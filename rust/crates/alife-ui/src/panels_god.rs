//! God Mode view + weather HUD + armed-target plumbing for the world canvas.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets::{self, ToastKind};
use crate::{god_api, TimeScale};
use alife_common::{CatastropheKind, GodEvent};
use alife_core::SimApi;
use egui::{pos2, Align2, Color32, Context, FontId, Id, Order, Stroke, Ui, Vec2};

/// Cheap throttled display window (weather values refresh ~10 Hz).
const HUD_REFRESH_FRAMES: u64 = 6;

// ------------------------------------------------------------------ firing

/// Current default target: last hovered world coords, else world center.
pub fn target(app: &AlifeApp) -> (f32, f32) {
    app.last_hover
        .unwrap_or((app.sim.world.width * 0.5, app.sim.world.height * 0.5))
}

/// Fire a god event: engine result becomes a toast + honest UI log entry.
pub fn fire_god_event(app: &mut AlifeApp, ev: GodEvent) {
    let result = god_api::trigger(&mut app.sim, &ev);
    let step = app.sim.step_count();
    let (note, succeeded) = match result {
        Ok(n) => (n, true),
        Err(e) => (e, false),
    };
    app.god_log.push(god_api::GodLogEntry {
        step,
        label: ev.label().to_string(),
        note: note.clone(),
        succeeded,
    });
    if app.god_log.len() > 60 {
        let over = app.god_log.len() - 60;
        app.god_log.drain(0..over);
    }
    let label = ev.label().to_string();
    if succeeded {
        app.toasts.push(ToastKind::Ok, format!("{label} — {note}"));
    } else {
        app.toasts.push(ToastKind::Warn, format!("{label} — {note}"));
    }
}

/// Fire a tool at the panel target with the shared radius.
pub fn fire_tool_at_target(app: &mut AlifeApp, tool: god_api::GodTool) {
    let (x, y) = target(app);
    let ev = god_api::build_event(tool, x, y, app.god_radius);
    fire_god_event(app, ev);
}

// ------------------------------------------------------------------- view

pub fn god_view(app: &mut AlifeApp, ctx: &Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("God Mode"));
                ui.label(widgets::weak(
                    "Weather, catastrophes and lineage tools. Acts apply through the engine; \
                     if the engine capability is not wired yet the toast says so — nothing \
                     is faked.",
                ));
                ui.add_space(6.0);

                timescale_mirror(app, ui);
                target_section(app, ui);
                weather_acts(app, ui);
                global_acts(app, ui);
                lineage_tools(app, ui);
                intervention_list(app, ui);
                ui.add_space(20.0);
            });
        });
}

/// Tiny mirror of the toolbar time-scale segmented control.
fn timescale_mirror(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Time scale");
    ui.horizontal_wrapped(|ui| {
        for ts in TimeScale::ALL {
            let sel = app.timescale == ts;
            let tip = format!(
                "{}x speed{}",
                ts.mult(),
                if ts.note().is_some() { " (per-frame budget; may be throttled)" } else { "" }
            );
            if widgets::select_btn(ui, ts.label(), sel, &tip).clicked() {
                app.timescale = ts;
            }
        }
        ui.label(widgets::mono(
            format!(
                "x{:.0} | achieved x{:.2}",
                app.timescale.mult(),
                app.achieved_multiplier()
            ),
            theme::ACCENT,
        ))
        .on_hover_text(
            "Achieved sim time per wall second, measured — may be lower than requested while \
             the per-frame step budget throttles the engine",
        );
    });
    if let Some(note) = app.timescale.note() {
        ui.label(widgets::mono(format!("({note})"), theme::WARN));
    }
}

/// Shared aiming: radius slider + live target confirm chip + armed hint.
fn target_section(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Target");
    ui.horizontal(|ui| {
        ui.add_sized(
            Vec2::new(110.0, 14.0),
            egui::Label::new(widgets::mono("radius r", theme::WEAK)),
        )
        .on_hover_text("Radius in world units used by every spatial god act");
        ui.add(egui::Slider::new(&mut app.god_radius, 8.0..=300.0))
            .on_hover_text("Radius for rain, drought, storm and nutrient dumps");
        ui.label(widgets::mono(format!("{:.0}", app.god_radius), theme::ACCENT));
    });

    ui.horizontal(|ui| {
        let (x, y) = target(app);
        ui.label(widgets::mono(
            format!("at ({:.0}, {:.0}) r={:.0} units", x, y, app.god_radius),
            theme::ACCENT,
        ))
        .on_hover_text("Act target = last hovered world position (or arm click mode below)");
        match app.god_tool {
            Some(tool) => {
                ui.colored_label(
                    theme::DANGER,
                    format!("armed: {} — next click on the world fires there", tool.label()),
                );
            }
            None => {
                ui.label(widgets::mono("not armed", theme::WEAK));
            }
        }
        if widgets::mini_btn(ui, "Clear armed", "Disarm the crosshair mode (Esc)").clicked() {
            app.god_tool = None;
        }
    });
}

/// Local (targeted) acts. Buttons fire at the current target; arm chips
/// switch the world canvas into armed click-target mode (crosshair cursor).
fn weather_acts(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Weather and matter acts");
    ui.label(widgets::weak(
        "Aim buttons fire at the current target now; arm chips prepare the next world click.",
    ));

    ui.horizontal_wrapped(|ui| {
        const DIRECT_TOOLS: [god_api::GodTool; 5] = [
            god_api::GodTool::Rain,
            god_api::GodTool::Drought,
            god_api::GodTool::Storm,
            god_api::GodTool::Lightning,
            god_api::GodTool::NutrientDump,
        ];
        for t in DIRECT_TOOLS {
            let tip = t.effect(app.god_radius);
            if widgets::mini_btn(ui, t.label(), &tip).clicked() {
                fire_tool_at_target(app, t);
            }
        }
    });

    ui.add_space(4.0);
    ui.label(widgets::weak("Click-target mode (armed crosshair on the map):"));
    ui.horizontal_wrapped(|ui| {
        const ARM_TOOLS: [god_api::GodTool; 6] = [
            god_api::GodTool::Rain,
            god_api::GodTool::Drought,
            god_api::GodTool::Storm,
            god_api::GodTool::Lightning,
            god_api::GodTool::NutrientDump,
            god_api::GodTool::Summon,
        ];
        for t in ARM_TOOLS {
            let sel = app.god_tool == Some(t);
            let tip = format!(
                "{} — armed next world click. {}",
                t.label(),
                t.effect(app.god_radius)
            );
            if widgets::select_btn(ui, t.label(), sel, &tip).clicked() {
                app.god_tool = if sel { None } else { Some(t) };
            }
        }
    });
    if app.god_tool.is_some() {
        ui.label(widgets::mono(
            "Go to the World view: a crosshair follows the cursor, the Armed chip shows the \
             tool; Esc cancels.",
            theme::DANGER,
        ));
    }
}

/// Planet-scale acts all route through a confirm dialog.
fn global_acts(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Catastrophes (planet scale)");
    ui.horizontal_wrapped(|ui| {
        for k in [
            CatastropheKind::Meteor,
            CatastropheKind::GlobalStorm,
            CatastropheKind::GlobalDrought,
        ] {
            let (label, tip) = catastrophe_copy(k);
            if widgets::mini_btn(ui, label, tip).clicked() {
                app.dialog = Some(crate::dialogs::Dialog::ConfirmCatastrophe { kind: k });
            }
        }
    });
}

/// Scale + expected-effect copy shared with the confirm dialog.
pub fn catastrophe_copy(kind: CatastropheKind) -> (&'static str, &'static str) {
    match kind {
        CatastropheKind::Meteor => (
            "Meteor",
            "planet scale | one impact zone: burns, dust cloud and shockwave hazards; local \
             extinction spike, crater fertility refreshed",
        ),
        CatastropheKind::GlobalStorm => (
            "Global storm",
            "planet scale | whole map: world-wide rain front plus a lightning barrage; heavy \
             wetting, exposed organisms die",
        ),
        CatastropheKind::GlobalDrought => (
            "Global drought",
            "planet scale | whole map: moisture and fertility collapse; food regen sputters, \
             heavy selection pressure",
        ),
    }
}

/// Species picker + summon / purge / protect tools.
fn lineage_tools(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Lineage tools");
    let selected: Option<u32> = app.god_species.or_else(|| {
        app.sim
            .stats()
            .species_counts
            .first()
            .map(|s| s.species_id)
    });

    ui.horizontal(|ui| {
        ui.label(widgets::mono("species", theme::WEAK));
        let top: Vec<(u32, u32)> = {
            let counts = &app.sim.stats().species_counts;
            counts.iter().take(8).map(|s| (s.species_id, s.count)).collect()
        };
        for (sp, count) in top {
            let sel = selected == Some(sp);
            widgets::chip(ui, alife_render::paint::species_color(sp));
            let tip = format!("Select species {sp} ({count} alive) as the lineage-tool target");
            if widgets::select_btn(ui, format!("#{sp} ({count})").as_str(), sel, &tip).clicked() {
                app.god_species = Some(sp);
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label(widgets::mono("by id", theme::WEAK));
        ui.add(
            egui::TextEdit::singleline(&mut app.god_species_text)
                .hint_text("e.g. 14")
                .desired_width(120.0),
        )
        .on_hover_text("Pick a species id manually (overrides the chips)");
        if let Ok(v) = app.god_species_text.trim().parse::<u32>() {
            if !app.god_species_text.trim().is_empty() {
                app.god_species = Some(v);
                ui.label(widgets::mono("valid", theme::OK));
            }
        } else if !app.god_species_text.trim().is_empty() {
            ui.label(widgets::mono("not a number", theme::DANGER));
        }
    });

    let Some(sp) = selected else {
        widgets::weak_label(ui, "no species exists yet");
        return;
    };

    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        let prot = god_api::protected(&app.sim, sp);
        let (shield_col, shield_txt) = if prot {
            (theme::OK, format!("species #{sp} — shield on"))
        } else {
            (theme::WEAK, format!("species #{sp} — unprotected"))
        };
        ui.label(egui::RichText::new("⛨").size(14.0).color(shield_col))
            .on_hover_text("Protected lineages are immune to weather damage and purge acts");
        ui.label(widgets::mono(shield_txt, theme::TEXT));

        if widgets::mini_btn(
            ui,
            "Summon at target",
            god_api::GodTool::Summon.effect(app.god_radius).as_str(),
        )
        .clicked()
        {
            fire_tool_at_target(app, god_api::GodTool::Summon);
        }

        let armed = app.god_purge_arm == sp;
        let purge_label = if armed {
            format!("Confirm purge #{sp}")
        } else {
            format!("Purge lineage #{sp}")
        };
        let purge_tip = format!("Remove all organisms of species {sp} — press twice to confirm");
        if widgets::mini_btn(ui, &purge_label, &purge_tip).clicked() {
            if armed {
                fire_god_event(app, GodEvent::PurgeLineage { species_id: sp });
                app.god_purge_arm = 0;
            } else {
                app.god_purge_arm = sp;
                app.toasts
                    .push(ToastKind::Warn, format!("Arm: purge species #{sp} — press again to confirm"));
            }
        }

        let protect_label = if prot { "Unprotect" } else { "Protect" };
        let protect_tip = format!(
            "Toggle the shield on species {sp} (engine keeps protection across weather and \
             purge acts)"
        );
        if widgets::mini_btn(ui, protect_label, &protect_tip).clicked() {
            match god_api::protect_lineage(&mut app.sim, sp, !prot) {
                Ok(()) => {
                    app.toasts.push(
                        ToastKind::Ok,
                        format!("species #{sp} {}", if prot { "unprotected" } else { "protected" }),
                    );
                }
                Err(e) => {
                    app.god_log.push(god_api::GodLogEntry {
                        step: app.sim.step_count(),
                        label: "Protect lineage".into(),
                        note: e.clone(),
                        succeeded: false,
                    });
                    app.toasts.push(ToastKind::Warn, format!("Protect lineage — {e}"));
                }
            }
        }
    });
}

/// Last 10 interventions: engine log when available, otherwise the honest
/// UI-side attempted log. Engine records live engine-side and intentionally
/// have no UI clear button.
fn intervention_list(app: &mut AlifeApp, ui: &mut Ui) {
    widgets::section_header(ui, "Intervention log (last 10)");

    let stale = app
        .god_cache
        .as_ref()
        .map_or(true, |c| c.built.is_none() || c.built.unwrap().elapsed().as_secs_f32() > 0.5);
    if stale {
        let mut c = app.god_cache.take().unwrap_or_default();
        c.interventions = god_api::interventions(&app.sim);
        c.built = Some(std::time::Instant::now());
        app.god_cache = Some(c);
    }

    let engine: Vec<alife_common::InterventionRecord> = app
        .god_cache
        .as_ref()
        .map(|c| c.interventions.clone())
        .unwrap_or_default();
    let mut shown = 0usize;
    if !engine.is_empty() {
        ui.label(widgets::mono(
            "engine record (authoritative, cannot be cleared)",
            theme::WEAK,
        ));
        egui::Grid::new("alife.god.interventions")
            .num_columns(3)
            .spacing([12.0, 2.0])
            .show(ui, |ui| {
                ui.label(widgets::mono("step", theme::WEAK));
                ui.label(widgets::mono("event", theme::WEAK));
                ui.label(widgets::mono("note", theme::WEAK));
                ui.end_row();
                for rec in engine.iter().rev().take(10) {
                    ui.label(widgets::mono(rec.step.to_string(), theme::TEXT));
                    ui.label(widgets::mono(rec.event.label(), theme::TEXT));
                    ui.label(widgets::weak(&rec.note));
                    ui.end_row();
                    shown += 1;
                }
            });
    }
    if shown == 0 {
        ui.label(widgets::mono(
            "attempted this session (engine record not available yet)",
            theme::WEAK,
        ));
        if app.god_log.is_empty() {
            widgets::weak_label(ui, "no god attempts yet — fire something from above");
        } else {
            egui::Grid::new("alife.god.attempts")
                .num_columns(3)
                .spacing([12.0, 2.0])
                .show(ui, |ui| {
                    ui.label(widgets::mono("step", theme::WEAK));
                    ui.label(widgets::mono("event", theme::WEAK));
                    ui.label(widgets::mono("result", theme::WEAK));
                    ui.end_row();
                    for g in app.god_log.iter().rev().take(10) {
                        ui.label(widgets::mono(g.step.to_string(), theme::TEXT));
                        ui.label(widgets::mono(&g.label, theme::TEXT));
                        let col = if g.succeeded { theme::OK } else { theme::WEAK };
                        ui.label(widgets::mono(&g.note, col));
                        ui.end_row();
                    }
                });
        }
    }
}

// ------------------------------------------------------------- overlay HUD

/// Bottom-left viewport corner weather readout: cloudiness, rain, flash ring,
/// thunder icon while a storm banner is up. Cheap, throttled, hides when idle.
pub fn weather_hud(app: &mut AlifeApp, ctx: &Context) {
    if app.frame_no % HUD_REFRESH_FRAMES == 0 {
        app.hud_weather = god_api::weather(&app.sim);
    }
    let Some(w) = app.hud_weather else { return };
    if w.idle() {
        return;
    }
    let Some(rect) = app.canvas_rect else { return };
    let sr = ctx.screen_rect();
    let offset = egui::vec2(
        rect.min.x - sr.min.x + 12.0,
        -((sr.max.y - rect.max.y) + 12.0),
    );
    egui::Area::new(Id::new("alife.weather_hud"))
        .order(Order::Middle)
        .anchor(Align2::LEFT_BOTTOM, offset)
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(6.0))
                .stroke(Stroke::new(1.0_f32, theme::STROKE_SOFT))
                .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let thunder_col: Color32 = if w.storm { theme::WARN } else { theme::WEAK };
                        ui.label(egui::RichText::new("⚡").size(14.0).color(thunder_col))
                            .on_hover_text(if w.storm {
                                "storm banner active — lightning possible"
                            } else {
                                "no storm"
                            });
                        ui.vertical(|ui| {
                            ui.label(widgets::mono(format!("cloud {:.2}", w.cloud), theme::TEXT));
                            ui.label(widgets::mono(format!("rain {:.2}", w.rain), theme::TEXT));
                        });
                        if let Some(age) = w.flash_age {
                            // flash ring decays over the 2 s after a bolt
                            let t = (1.0 - age / 2.0).clamp(0.0, 1.0);
                            let (rect, _) =
                                ui.allocate_exact_size(Vec2::splat(18.0), egui::Sense::hover());
                            ui.painter().circle_stroke(
                                rect.center(),
                                7.0,
                                Stroke::new(2.0_f32, theme::WARN.gamma_multiply(t)),
                            );
                            ui.label(widgets::mono(format!("flash {:.1} s ago", age), theme::WEAK));
                        }
                    });
                });
        });
}

/// Armed crosshair drawn over the world canvas in god click-target mode.
#[allow(dead_code)]
/// Replaced by the render crate's draw_brush_preview (R2); kept for tests
/// and as the reference rendering of the armed cursor.
pub fn armed_crosshair(app: &AlifeApp, ui: &mut Ui, hover: (f32, f32), scale: f32) {
    let Some(tool) = app.god_tool else { return };
    let c = app.view.world_to_screen(hover.0, hover.1);
    let r = (app.god_radius * scale).max(6.0);
    let col = theme::DANGER.gamma_multiply(0.7);
    let p = ui.painter();
    p.circle_stroke(c, r, Stroke::new(1.2_f32, col));
    p.line_segment(
        [pos2(c.x - 8.0, c.y), pos2(c.x + 8.0, c.y)],
        Stroke::new(1.2_f32, col),
    );
    p.line_segment(
        [pos2(c.x, c.y - 8.0), pos2(c.x, c.y + 8.0)],
        Stroke::new(1.2_f32, col),
    );
    // wet showing under the cursor when the engine samples the weather layer
    let wet_line = match god_api::wet_at(&app.sim, hover.0, hover.1) {
        Some(w) => format!(" — wet here {:.2}", w),
        None => String::new(),
    };
    p.text(
        pos2(c.x, c.y - r - 12.0),
        Align2::CENTER_BOTTOM,
        format!("armed: {}{}", tool.label(), wet_line),
        FontId::proportional(11.0),
        theme::DANGER,
    );
}

/// Small top-center chip inside world view while a god tool is armed.
pub fn armed_strip(app: &AlifeApp, ctx: &Context) {
    let Some(tool) = app.god_tool else { return };
    let Some(rect) = app.canvas_rect else { return };
    let sr = ctx.screen_rect();
    let offset = egui::vec2(rect.center().x - sr.center().x, rect.min.y - sr.min.y + 30.0);
    egui::Area::new(Id::new("alife.god.armed"))
        .order(Order::Middle)
        .anchor(Align2::CENTER_TOP, offset)
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(6.0))
                .stroke(Stroke::new(1.0_f32, theme::DANGER.gamma_multiply(0.8)))
                .inner_margin(egui::Margin::symmetric(10.0, 5.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "armed: {} — next click fires here at the crosshair; Esc cancels",
                                tool.label()
                            ))
                            .size(11.5)
                            .color(theme::DANGER),
                        );
                    });
                });
        });
}

