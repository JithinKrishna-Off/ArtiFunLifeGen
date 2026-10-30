//! World Editor view: interactive canvas + tool chips, brush radius, undo and
//! world regeneration controls (seed / preset / size) with confirm dialog.

use crate::app::AlifeApp;
use crate::panels_world::canvas;
use crate::theme;
use crate::widgets::{self, ToastKind};
use alife_common::InteractionMode;
use alife_core::SimApi;
use egui::Context;

const TOOLS: [(InteractionMode, &str, &str); 6] = [
    (InteractionMode::Inspect, "Inspect", "Click organisms to select and inspect them (Esc exits the tool)"),
    (InteractionMode::Food, "Food", "Click to drop a single food item"),
    (InteractionMode::Obstacle, "Obstacle", "Click to place a circular wall sized by the brush"),
    (InteractionMode::Patch, "Patch", "Click to place a food patch sized by the brush"),
    (InteractionMode::Hazard, "Hazard", "Click to place a hazard zone sized by the brush"),
    (InteractionMode::Erase, "Erase", "Click to erase obstacles, patches, hazards and nearby food"),
];

pub fn editor_view(app: &mut AlifeApp, ctx: &Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(widgets::strong("World Editor"));
                ui.label(widgets::weak(format!(
                    "tool: {}  |  Esc cancels, Ctrl+Z undoes",
                    tool_name(app.editor_mode)
                )));
            });

            // interactive canvas; keep a fixed height so controls stay visible
            let canvas_h = ((ui.available_height() - 300.0).clamp(240.0, 470.0)).max(200.0);
            let mut rect = ui.available_rect_before_wrap();
            rect.set_height(canvas_h);
            canvas(app, ui, rect);
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .id_salt("alife.editor.controls")
                .show(ui, |ui| {
                    tools(app, ui);
                    ui.add_space(8.0);
                    regenerate(app, ui);
                    ui.add_space(16.0);
                });
        });
}

fn tool_name(m: InteractionMode) -> &'static str {
    TOOLS
        .iter()
        .find(|(mode, _, _)| *mode == m)
        .map(|(_, label, _)| *label)
        .unwrap_or("Inspect")
}

fn tools(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Canvas tool");
    ui.horizontal_wrapped(|ui| {
        for (mode, label, tip) in TOOLS {
            let sel = app.editor_mode == mode;
            if widgets::select_btn(ui, label, sel, tip).clicked() {
                app.editor_mode = mode;
            }
        }
    });

    let brush_relevant = matches!(
        app.editor_mode,
        InteractionMode::Obstacle
            | InteractionMode::Patch
            | InteractionMode::Hazard
            | InteractionMode::Erase
    );
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_sized(
            egui::Vec2::new(150.0, 14.0),
            egui::Label::new(widgets::mono("brush radius", theme::WEAK)),
        );
        let r = ui.add_enabled(
            brush_relevant,
            egui::Slider::new(&mut app.editor_brush, 5.0..=300.0),
        );
        ui.add_sized(
            egui::Vec2::new(70.0, 14.0),
            egui::Label::new(widgets::mono(
                format!("{:.0}", app.editor_brush),
                theme::TEXT,
            )),
        );
        if brush_relevant {
            r.on_hover_text("Radius for obstacles, patches, hazards and erase");
        } else {
            r.on_disabled_hover_text(
                "Brush only applies to Obstacle / Patch / Hazard / Erase tools",
            );
        }
    });

    if widgets::mini_btn(ui, "Undo last edit", "Revert the most recent environment edit (Ctrl+Z)")
        .clicked()
    {
        app.editor_undo();
    }
    if widgets::mini_btn(ui, "Redo", "Re-apply the most recently undone edit (Ctrl+Y)")
        .clicked()
    {
        app.editor_redo();
    }
    ui.horizontal(|ui| {
        widgets::weak_label(ui, format!(
            "environment edits: {} obstacles, {} patches, {} hazards",
            app.sim.edits().obstacles.len(),
            app.sim.edits().patches.len(),
            app.sim.edits().hazards.len()
        ));
    });
}

fn regenerate(app: &mut AlifeApp, ui: &mut egui::Ui) {
    widgets::section_header(ui, "Regenerate world");

    ui.horizontal(|ui| {
        ui.add_sized(
            egui::Vec2::new(150.0, 16.0),
            egui::Label::new(widgets::mono("seed", theme::WEAK)),
        );
        ui.add(
            egui::DragValue::new(&mut app.gen_seed)
                .speed(1.0)
                .range(0..=u32::MAX),
        )
        .on_hover_text("Seed for the next generated world");
        if widgets::mini_btn(ui, "Random", "Pick a random seed").clicked() {
            app.gen_seed = crate::app::random_seed();
        }
    });

    let cat = crate::save_io::preset_catalog();
    let idx = app.gen_preset_idx.min(cat.len().saturating_sub(1));
    let label = cat.get(idx).map(|p| p.name).unwrap_or("Genesis");
    ui.horizontal(|ui| {
        ui.add_sized(
            egui::Vec2::new(150.0, 16.0),
            egui::Label::new(widgets::mono("preset", theme::WEAK)),
        );
        let combo = egui::ComboBox::from_id_salt("alife.editor.preset")
            .width(220.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (i, p) in cat.iter().enumerate() {
                    ui.selectable_value(&mut app.gen_preset_idx, i, p.name);
                }
            });
        combo.response.clone().on_hover_text("Template applied to the regenerated world");
        if let Some(p) = cat.get(app.gen_preset_idx.min(cat.len().saturating_sub(1))) {
            if combo.response.clone().changed() {
                app.gen_seed = p.world.seed;
                app.gen_width = p.world.width;
                app.gen_height = p.world.height;
            }
        }
    });
    if let Some(p) = cat.get(app.gen_preset_idx.min(cat.len().saturating_sub(1))) {
        ui.label(widgets::weak(p.description));
    }

    ui.add(
        egui::Slider::new(&mut app.gen_width, 600..=6000)
            .text("width")
            .integer(),
    );
    ui.add(
        egui::Slider::new(&mut app.gen_height, 600..=6000)
            .text("height")
            .integer(),
    );

    ui.add_space(6.0);
    let g = ui.add(egui::Button::new(egui::RichText::new("Generate new world").strong()))
        .on_hover_text("Confirm and generate a fresh world with these settings");
    if g.clicked() {
        app.dialog = Some(crate::dialogs::Dialog::new_world_scratch(app));
        app.toasts
            .push(ToastKind::Info, "Confirm the generation in the dialog");
    }
}
