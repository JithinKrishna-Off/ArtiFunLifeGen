//! Modal dialogs: New World, Save, Load (with delete confirm) and Reset confirm.
//!
//! Every interaction is non-blocking: file IO is a handful of fast local
//! writes; failures become toasts (and are logged by the DataStore). The
//! Esc key (handled in app.rs) closes whichever dialog is open.

use crate::app::{random_seed, AlifeApp};
use crate::panels_god;
use crate::theme;
use crate::widgets::{self, ToastKind};
use alife_common::persist::WorldSave;
use alife_common::{CatastropheKind, LineageRecord};
use alife_core::SimApi;

#[derive(Clone, Debug)]
pub struct NewWorldData {
    pub preset_idx: usize,
    pub seed: String,
    pub width: u32,
    pub height: u32,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct SaveData {
    pub name: String,
    pub overwrite: bool,
}

#[derive(Clone, Debug)]
pub enum Dialog {
    NewWorld(NewWorldData),
    Save(SaveData),
    /// `confirm_delete`: a save name awaiting deletion confirmation.
    Load { confirm_delete: Option<String> },
    ConfirmReset,
    /// God Mode catastrophe confirm (Meteor / Global storm / Global drought).
    ConfirmCatastrophe { kind: CatastropheKind },
}

impl Dialog {
    /// New World dialog prefilled from the editor regenerate scratch fields.
    pub fn new_world_scratch(app: &AlifeApp) -> Dialog {
        Dialog::NewWorld(NewWorldData {
            preset_idx: app.gen_preset_idx.min(8),
            seed: app.gen_seed.to_string(),
            width: app.gen_width,
            height: app.gen_height,
            name: String::new(),
        })
    }

    /// Save dialog with a sensible default name.
    pub fn save_scratch(app: &AlifeApp) -> Dialog {
        let base = if app.world_name.trim().is_empty() {
            format!("world-{}", app.sim.step_count())
        } else {
            app.world_name.clone()
        };
        Dialog::Save(SaveData { name: base, overwrite: false })
    }
}

// ------------------------------------------------------------------ actions

/// Returns true when the dialog should stay open.
fn apply_new_world(app: &mut AlifeApp, d: &NewWorldData) -> bool {
    let cat = crate::save_io::preset_catalog();
    let Some(p) = cat.get(d.preset_idx) else {
        app.toasts.push(ToastKind::Error, "Preset selection invalid");
        return true;
    };
    let trimmed = d.seed.trim();
    let seed = if trimmed.is_empty() {
        random_seed()
    } else {
        match trimmed.parse::<u32>() {
            Ok(v) => v,
            Err(_) => {
                app.toasts.push(
                    ToastKind::Error,
                    "Seed must be a whole number (0..4294967295)",
                );
                return true;
            }
        }
    };
    let name = if d.name.trim().is_empty() {
        format!("world-{}", app.sim.step_count())
    } else {
        d.name.trim().to_string()
    };
    let mut cfg = p.world;
    cfg.seed = seed;
    cfg.width = d.width.max(600);
    cfg.height = d.height.max(600);
    let params = p.sim.clone();
    app.install_new_world(cfg, &params, Some(p.id.to_string()), &name);
    app.gen_preset_idx = d.preset_idx;
    false
}

/// Returns true when the dialog should stay open.
fn apply_save(app: &mut AlifeApp, d: &mut SaveData) -> bool {
    let name = d.name.trim().to_string();
    if name.is_empty() {
        app.toasts.push(ToastKind::Warn, "Give the save a name first");
        return true;
    }

    let records: Vec<LineageRecord> = app.records_seen.values().cloned().collect();
    let preset = if app.preset_name.trim().is_empty() {
        None
    } else {
        Some(app.preset_name.trim().to_string())
    };
    let params = app.sim.params.clone();
    let cfg = app.world_cfg;
    let history = app.history.clone();
    let ws: WorldSave = crate::save_io::assemble_save(
        &name,
        preset,
        &params,
        &cfg,
        &app.sim,
        &records,
        &history,
    );

    let target = format!("{}.json", name);
    let exists = app.store.list().into_iter().any(|n| n == target);
    if exists && !d.overwrite {
        d.overwrite = true; // warn inline; second Save press overwrites
        return true;
    }

    let result = if exists {
        app.store.overwrite_world(&ws)
    } else {
        app.store.save_world(&ws)
    };
    match result {
        Ok(()) => {
            app.world_name = name.clone();
            app.toasts.push(ToastKind::Ok, format!("Saved '{}' to saves/", name));
            false
        }
        Err(e) => {
            if !d.overwrite && e.contains("already exists") {
                d.overwrite = true;
                true
            } else {
                app.toasts.push(ToastKind::Error, e.clone());
                alife_data::log_error(&app.store.root, &e);
                false
            }
        }
    }
}

/// Returns true when the dialog should stay open.
fn apply_load(app: &mut AlifeApp, name: &str) -> bool {
    match app.store.load_world(name) {
        Ok(ws) => {
            app.apply_loaded(&ws);
            false
        }
        Err(e) => {
            app.toasts.push(ToastKind::Error, e.clone());
            true
        }
    }
}

// ------------------------------------------------------------------ display

pub fn show(app: &mut AlifeApp, ctx: &egui::Context) {
    if app.dialog.is_none() {
        return;
    }

    // Dim + click-blocking backdrop so the dialogs are truly modal.
    egui::Area::new(egui::Id::new("alife.backdrop"))
        .order(egui::Order::Middle)
        .show(ctx, |ui| {
            ui.painter().rect_filled(
                ctx.screen_rect(),
                0.0,
                egui::Color32::from_rgba_unmultiplied(4, 6, 9, 150),
            );
            ui.allocate_rect(ctx.screen_rect(), egui::Sense::click());
        });

    let mut dialog = app.dialog.take();
    let mut keep_open = true;
    if let Some(d) = dialog.as_mut() {
        match d {
            Dialog::NewWorld(data) => {
                keep_open &= dialog_window(ctx, "New World", |ui| {
                    body_new_world(app, ui, data, keep_open)
                });
            }
            Dialog::Save(data) => {
                keep_open &= dialog_window(ctx, "Save world", |ui| {
                    body_save(app, ui, data, keep_open)
                });
            }
            Dialog::Load { confirm_delete } => {
                keep_open &= dialog_window(ctx, "Load world", |ui| {
                    body_load(app, ui, confirm_delete, keep_open)
                });
            }
            Dialog::ConfirmReset => {
                keep_open &= dialog_window(ctx, "Reset world", |ui| {
                    body_reset(app, ui, keep_open)
                });
            }
            Dialog::ConfirmCatastrophe { kind } => {
                let kind = *kind;
                keep_open &= dialog_window(ctx, catastrophe_title(kind), |ui| {
                    body_catastrophe(app, ui, kind, keep_open)
                });
            }
        }
    }
    if !keep_open {
        dialog = None;
    }
    app.dialog = dialog;
}

fn dialog_window(
    ctx: &egui::Context,
    title: &str,
    body: impl FnOnce(&mut egui::Ui) -> bool,
) -> bool {
    let mut result = true;
    egui::Window::new(title)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .order(egui::Order::Middle)
        .collapsible(false)
        .resizable(false)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(6.0))
                .stroke(egui::Stroke::new(1.0_f32, theme::STROKE_SOFT))
                .inner_margin(14.0),
        )
        .show(ctx, |ui| result = body(ui));
    result
}

fn body_new_world(app: &mut AlifeApp, ui: &mut egui::Ui, d: &mut NewWorldData, keep: bool) -> bool {
    let cat = crate::save_io::preset_catalog();
    let idx = d.preset_idx.min(cat.len().saturating_sub(1));
    let selected_name = cat.get(idx).map(|p| p.name).unwrap_or("Genesis");

    widgets::section_header(ui, "Preset");
    let combo_r = egui::ComboBox::from_id_salt("alife.newworld.preset")
        .width(280.0)
        .selected_text(selected_name)
        .show_ui(ui, |ui| {
            for (i, p) in cat.iter().enumerate() {
                ui.selectable_value(&mut d.preset_idx, i, p.name);
            }
        });
    combo_r.response.on_hover_text("Choose a starting world template");
    if let Some(p) = cat.get(idx) {
        ui.label(widgets::weak(p.description));
    }

    widgets::section_header(ui, "Seed");
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut d.seed)
                .hint_text("random")
                .desired_width(140.0),
        )
        .on_hover_text("Empty = random seed; otherwise a number 0..4294967295");
        if widgets::mini_btn(ui, "Random", "Pick a random seed").clicked() {
            d.seed = random_seed().to_string();
        }
        if trimmed_seed_blank(&d.seed) {
            ui.label(widgets::mono("random", theme::WEAK));
        } else if d.seed.trim().parse::<u32>().is_ok() {
            ui.label(widgets::mono("valid", theme::OK));
        } else {
            ui.label(widgets::mono("invalid number", theme::DANGER));
        }
    });

    widgets::section_header(ui, "Size (world units)");
    ui.add(
        egui::Slider::new(&mut d.width, 600..=6000)
            .text("width")
            .integer(),
    );
    ui.add(
        egui::Slider::new(&mut d.height, 600..=6000)
            .text("height")
            .integer(),
    );

    widgets::section_header(ui, "World name (optional)");
    ui.add(
        egui::TextEdit::singleline(&mut d.name)
            .hint_text("name shown in the toolbar, also used for saves")
            .desired_width(280.0),
    );

    ui.add_space(10.0);
    let mut keep = keep;
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "Cancel", "Close without changes (Esc)").clicked() {
            keep = false;
        }
        if ui
            .add(egui::Button::new(egui::RichText::new("Generate && switch").strong()))
            .clicked()
        {
            keep = apply_new_world(app, d);
        }
    });
    keep
}

fn trimmed_seed_blank(s: &str) -> bool {
    s.trim().is_empty()
}

fn body_save(app: &mut AlifeApp, ui: &mut egui::Ui, d: &mut SaveData, keep: bool) -> bool {
    widgets::section_header(ui, "Save name");
    ui.add(
        egui::TextEdit::singleline(&mut d.name)
            .hint_text("world-1")
            .desired_width(280.0),
    );

    let names = app.store.list();
    widgets::section_header(ui, "Existing saves");

    let trimmed = d.name.trim().to_string();
    let target = format!("{}.json", trimmed);
    let conflicts = !trimmed.is_empty() && names.iter().any(|n| *n == target);
    if conflicts {
        ui.colored_label(
            theme::WARN,
            if d.overwrite {
                format!("'{}' exists — Overwrite && save replaces it", trimmed)
            } else {
                format!("'{}' exists — press Save again to confirm overwrite", trimmed)
            },
        );
    }

    let mut keep = keep;
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "Cancel", "Close without saving (Esc)").clicked() {
            keep = false;
        }
        let label = if d.overwrite {
            "Overwrite && save"
        } else {
            "Save"
        };
        if ui
            .add(egui::Button::new(egui::RichText::new(label).strong()))
            .clicked()
        {
            keep = apply_save(app, d);
        }
    });
    keep
}

fn body_load(
    app: &mut AlifeApp,
    ui: &mut egui::Ui,
    confirm_delete: &mut Option<String>,
    keep: bool,
) -> bool {
    let names = app.store.list();
    let mut keep = keep;

    if names.is_empty() {
        widgets::empty_state(
            ui,
            "No saves found",
            "Run a world and use Save to create one",
            None,
        );
    } else {
        egui::ScrollArea::vertical()
            .max_height(300.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for name in names {
                    let display = name.trim_end_matches(".json").to_string();
                    ui.horizontal(|ui| {
                        ui.set_min_width(ui.available_width() - 4.0);
                        ui.label(widgets::strong(&display));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if widgets::mini_btn(ui, "Load", "Restore this world").clicked() {
                                keep = apply_load(app, &name);
                            }
                            if widgets::mini_btn(ui, "Delete", "Remove this save file").clicked() {
                                *confirm_delete = Some(name.clone());
                            }
                        });
                    });
                }
            });
    }

    if let Some(name) = confirm_delete.clone() {
        ui.add_space(6.0);
        ui.colored_label(
            theme::WARN,
            format!("Delete '{}' from disk?", name.trim_end_matches(".json")),
        );
        ui.horizontal(|ui| {
            if widgets::mini_btn(ui, "Yes, delete", "This cannot be undone").clicked() {
                match app.store.delete(&name) {
                    Ok(()) => {
                        app.toasts
                            .push(ToastKind::Ok, format!("Deleted '{}'", name.trim_end_matches(".json")));
                        *confirm_delete = None;
                    }
                    Err(e) => {
                        app.toasts.push(ToastKind::Error, e.clone());
                        *confirm_delete = None;
                    }
                }
            }
            if widgets::mini_btn(ui, "No", "Keep the save").clicked() {
                *confirm_delete = None;
            }
        });
    }

    ui.add_space(10.0);
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "Close", "Close (Esc)").clicked() {
            keep = false;
        }
    });
    keep
}

fn body_reset(app: &mut AlifeApp, ui: &mut egui::Ui, keep: bool) -> bool {
    ui.label(widgets::strong("Restart this world from the beginning?"));
    ui.add_space(4.0);
    ui.label(widgets::weak(
        "Same terrain and parameters, but population, statistics and history start over.",
    ));
    ui.add_space(10.0);
    let mut keep = keep;
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "Cancel", "Keep this world (Esc)").clicked() {
            keep = false;
        }
        if ui
            .add(egui::Button::new(
                egui::RichText::new("Reset world").color(theme::DANGER),
            ))
            .clicked()
        {
            app.reset_world();
            keep = false;
        }
    });
    keep
}

fn catastrophe_title(kind: CatastropheKind) -> &'static str {
    match kind {
        CatastropheKind::Meteor => "Catastrophe: meteor",
        CatastropheKind::GlobalStorm => "Catastrophe: global storm",
        CatastropheKind::GlobalDrought => "Catastrophe: global drought",
    }
}

fn body_catastrophe(
    app: &mut AlifeApp,
    ui: &mut egui::Ui,
    kind: CatastropheKind,
    keep: bool,
) -> bool {
    let (scale, effect) = match kind {
        CatastropheKind::Meteor => (
            "planet scale | one impact zone",
            "An impact zone burns: dust cloud and shockwave hazards follow. Expect a local \
             extinction spike; the crater is refreshed with ash fertility.",
        ),
        CatastropheKind::GlobalStorm => (
            "planet scale | whole map",
            "A world-wide rain front with a lightning barrage: heavy wetting raises fertility, \
             bolts re-energize patches and exposed organisms die.",
        ),
        CatastropheKind::GlobalDrought => (
            "planet scale | whole map",
            "Moisture and fertility collapse world-wide: food regen sputters, weak genotypes \
             starve — strong selection pressure follows.",
        ),
    };
    let flavor = crate::save_io::preset_catalog()
        .iter()
        .find(|p| p.id == app.preset_name)
        .map(|p| format!("'{}' — {}", p.name, p.description))
        .unwrap_or_else(|| "custom world (no preset flavor)".to_string());

    ui.label(widgets::strong("This is a planet-scale intervention"));
    ui.add_space(4.0);
    ui.label(widgets::mono(scale, theme::WARN));
    ui.add_space(6.0);
    ui.label(widgets::weak(effect));
    ui.add_space(6.0);
    ui.label(widgets::mono(format!("world: {flavor}"), theme::WEAK))
        .on_hover_text("Preset flavor: rough habitat context for the expected effect");
    ui.add_space(10.0);
    let mut keep = keep;
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "Cancel", "Leave the planet intact (Esc)").clicked() {
            keep = false;
        }
        let go = ui
            .add(egui::Button::new(egui::RichText::new("Trigger").color(theme::DANGER)))
            .on_hover_text("Apply the catastrophe via the engine");
        if go.clicked() {
            panels_god::fire_god_event(app, alife_common::GodEvent::Catastrophe { kind });
            keep = false;
        }
    });
    keep
}
