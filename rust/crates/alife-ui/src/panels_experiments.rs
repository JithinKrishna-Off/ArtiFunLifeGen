//! Experiments view: live parameter tuning, static (new-world-only) params,
//! preset launch cards and export buttons.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets::{self, ToastKind};
use alife_common::LiveParams;
use alife_core::SimApi;
use egui::Context;

pub fn experiments_view(app: &mut AlifeApp, ctx: &Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("Experiments"));
                ui.label(widgets::weak(
                    "Tune the running world live; some parameters only apply to a new world.",
                ));
                ui.add_space(6.0);

                live_params(app, ui);
                ui.add_space(10.0);
                static_params(app, ui);
                ui.add_space(10.0);
                preset_cards(app, ui);
                ui.add_space(10.0);
                exports(app, ui);
                ui.add_space(20.0);
            });
        });
}

fn set_fn(ui: &mut egui::Ui, label: &str, tip: &str) {
    ui.add_sized(
        egui::Vec2::new(150.0, 14.0),
        egui::Label::new(widgets::mono(label.to_string(), theme::WEAK)),
    )
    .on_hover_text(tip);
}

fn live_params(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Live parameters (apply instantly)");

    // max_population
    ui.horizontal(|ui| {
        set_fn(ui, "max_population", "Population the ecosystem may reach");
        let mut v = app.sim.params.max_population;
        if ui
            .add(egui::Slider::new(&mut v, 10..=20000).integer())
            .changed()
        {
            app.sim.update_live_params(&LiveParams {
                max_population: Some(v),
                ..Default::default()
            });
        }
    });
    // max_food
    ui.horizontal(|ui| {
        set_fn(ui, "max_food", "Food items the world may carry");
        let mut v = app.sim.params.max_food;
        if ui
            .add(egui::Slider::new(&mut v, 10..=20000).integer())
            .changed()
        {
            app.sim.update_live_params(&LiveParams {
                max_food: Some(v),
                ..Default::default()
            });
        }
    });
    // food_regen_per_sec
    ui.horizontal(|ui| {
        set_fn(ui, "food_regen_per_sec", "New food items per second");
        let mut v = app.sim.params.food_regen_per_sec;
        if ui.add(egui::Slider::new(&mut v, 0.0..=200.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                food_regen_per_sec: Some(v),
                ..Default::default()
            });
        }
    });
    // food_energy
    ui.horizontal(|ui| {
        set_fn(ui, "food_energy", "Mean energy per new food item");
        let mut v = app.sim.params.food_energy;
        if ui.add(egui::Slider::new(&mut v, 1.0..=120.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                food_energy: Some(v),
                ..Default::default()
            });
        }
    });
    // metabolism_base
    ui.horizontal(|ui| {
        set_fn(ui, "metabolism_base", "Basal energy drain per second at metabolism 1");
        let mut v = app.sim.params.metabolism_base;
        if ui.add(egui::Slider::new(&mut v, 0.01..=4.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                metabolism_base: Some(v),
                ..Default::default()
            });
        }
    });
    // movement_cost
    ui.horizontal(|ui| {
        set_fn(ui, "movement_cost", "Energy drain per speed unit per second");
        let mut v = app.sim.params.movement_cost;
        if ui.add(egui::Slider::new(&mut v, 0.0..=0.2)).changed() {
            app.sim.update_live_params(&LiveParams {
                movement_cost: Some(v),
                ..Default::default()
            });
        }
    });
    // sensing_cost
    ui.horizontal(|ui| {
        set_fn(ui, "sensing_cost", "Cost of large sensor ranges");
        let mut v = app.sim.params.sensing_cost;
        if ui.add(egui::Slider::new(&mut v, 0.0..=0.08)).changed() {
            app.sim.update_live_params(&LiveParams {
                sensing_cost: Some(v),
                ..Default::default()
            });
        }
    });
    // mutation_rate
    ui.horizontal(|ui| {
        set_fn(ui, "mutation_rate", "Per-gene mutation probability");
        let mut v = app.sim.params.mutation_rate;
        if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                mutation_rate: Some(v),
                ..Default::default()
            });
        }
    });
    // mutation_magnitude
    ui.horizontal(|ui| {
        set_fn(ui, "mutation_magnitude", "Mutation step size (gauss sigma multiplier)");
        let mut v = app.sim.params.mutation_magnitude;
        if ui.add(egui::Slider::new(&mut v, 0.05..=8.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                mutation_magnitude: Some(v),
                ..Default::default()
            });
        }
    });
    // big_jump_chance
    ui.horizontal(|ui| {
        set_fn(ui, "big_jump_chance", "Chance a mutation leaps 10x farther");
        let mut v = app.sim.params.big_jump_chance;
        if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)).changed() {
            app.sim.update_live_params(&LiveParams {
                big_jump_chance: Some(v),
                ..Default::default()
            });
        }
    });
}

fn static_params(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Locked — needs new world");
    let p = &app.sim.params;
    let rows: [(&str, String); 7] = [
        ("seed", p.seed.to_string()),
        ("initial_population", p.initial_population.to_string()),
        ("initial_food", p.initial_food.to_string()),
        ("food_energy_variance", format!("{:.2}", p.food_energy_variance)),
        (
            "reproduction_cooldown_steps",
            p.reproduction_cooldown_steps.to_string(),
        ),
        ("min_repro_age_steps", p.min_repro_age_steps.to_string()),
        ("wrapping", p.wrapping.to_string()),
    ];
    for (k, v) in rows {
        ui.horizontal(|ui| {
            ui.add_sized(
                egui::Vec2::new(150.0, 14.0),
                egui::Label::new(widgets::mono(k, theme::WEAK)),
            );
            ui.add_sized(
                egui::Vec2::new(120.0, 14.0),
                egui::Label::new(widgets::mono(v, theme::TEXT)),
            );
            ui.label(widgets::mono("needs new world", theme::WARN));
        });
    }
}

fn preset_cards(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Start from preset");
    for p in crate::save_io::preset_catalog() {
        ui.horizontal(|ui| {
            ui.add_sized(
                egui::Vec2::new(130.0, 16.0),
                egui::Label::new(widgets::strong(p.name)),
            );
            widgets::weak_label(ui, p.description);
            let tip = format!("Generate a new '{}' world (replaces the current sim state)", p.name);
            if widgets::mini_btn(ui, "Launch", tip.as_str()).clicked() {
                let name = format!("{}-{}", p.id, app.sim.step_count());
                app.install_new_world(p.world, &p.sim, Some(p.id.to_string()), &name);
            }
        });
    }
}

fn exports(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Export");
    ui.label(widgets::weak("Exports land under the app data directory"));
    ui.horizontal(|ui| {
        if widgets::mini_btn(ui, "History CSV", "Write the statistics history to exports/ as CSV").clicked()
        {
            let base = export_base_name(app, "history");
            let records: Vec<alife_common::LineageRecord> =
                app.records_seen.values().cloned().collect();
            let preset = if app.preset_name.trim().is_empty() {
                None
            } else {
                Some(app.preset_name.clone())
            };
            let ws = crate::save_io::assemble_save(
                &base,
                preset,
                &app.sim.params,
                &app.world_cfg,
                &app.sim,
                &records,
                &app.history,
            );
            match app.store.export_history_csv(&ws) {
                Ok(csv) => {
                    let dir = app.store.export_path();
                    let file = dir.join(format!("{}.csv", base));
                    match std::fs::write(&file, csv.as_bytes()) {
                        Ok(()) => {
                            app.toasts.push(ToastKind::Ok, format!("CSV exported: {}", file.display()));
                        }
                        Err(e) => {
                            let msg = format!("CSV write failed: {}", e);
                            app.toasts.push(ToastKind::Error, msg.clone());
                            alife_data::log_error(&app.store.root, &msg);
                        }
                    }
                }
                Err(e) => {
                    app.toasts.push(ToastKind::Error, e.clone());
                }
            }
        }
        if widgets::mini_btn(
            ui,
            "Config JSON",
            "Write current preset + parameters to experiments/ as JSON",
        )
        .clicked()
        {
            let base = export_base_name(app, "config");
            let preset = if app.preset_name.trim().is_empty() {
                String::new()
            } else {
                app.preset_name.clone()
            };
            match crate::save_io::export_config_json(
                &app.store.root,
                &base,
                &preset,
                &app.sim.params,
                &app.world_cfg,
            ) {
                Ok(path) => {
                    app.toasts.push(ToastKind::Ok, format!("Config exported: {}", path));
                }
                Err(e) => {
                    app.toasts.push(ToastKind::Error, e.clone());
                }
            }
        }
    });
}

fn export_base_name(app: &AlifeApp, tag: &str) -> String {
    let raw = if app.world_name.trim().is_empty() {
        format!("{}-{}", tag, app.sim.step_count())
    } else {
        format!("{}-{}", app.world_name.trim(), tag)
    };
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
