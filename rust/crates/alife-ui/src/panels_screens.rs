//! Settings view + the shortcut cheatsheet floating sheet.

use crate::app::{AlifeApp, Quality};
use crate::theme;
use crate::widgets;
use egui::Context;

pub fn settings_view(app: &mut AlifeApp, ctx: &Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("Settings"));
                ui.add_space(6.0);

                widgets::section_header(ui, "Rendering quality");
                ui.horizontal_wrapped(|ui| {
                    for q in Quality::ALL {
                        let sel = app.quality == q;
                        let tip = match q {
                            Quality::Efficient => {
                                "No trails, most throttling — best for integrated graphics"
                            }
                            Quality::Balanced => "Trails every 4th frame (default)",
                            Quality::High => "Trails every frame, full-rate charts",
                        };
                        if widgets::select_btn(ui, q.label(), sel, tip).clicked() {
                            app.quality = q;
                        }
                    }
                });
                widgets::weak_label(ui, match app.quality {
                    Quality::Efficient => "Trails disabled; statistics refresh still applies.",
                    Quality::Balanced => "Trails sampled 1 in 4 frames while population < 900.",
                    Quality::High => "Every frame is fully sampled — heavier on the GPU/CPU.",
                });

                widgets::section_header(ui, "World view");
                let mut trails_on = app.view.trails.enabled();
                if ui.checkbox(&mut trails_on, "Movement trails").changed() {
                    app.view.trails.set_enabled(trails_on);
                }
                ui.checkbox(&mut app.follow_selected, "Follow selected organism")
                    .on_hover_text("Keep the camera locked on the current selection");
                ui.checkbox(&mut app.view.show_grid, "Coordinate grid overlay")
                    .on_hover_text("Toggle with G");
                ui.horizontal(|ui| {
                    ui.label(widgets::mono("pick radius (click selection)", theme::WEAK));
                    ui.add_sized(
                        egui::Vec2::new(140.0, 18.0),
                        egui::Slider::new(&mut app.pick_radius_px, 6.0..=26.0),
                    )
                    .on_hover_text(
                        "Forgiving click radius in screen pixels around an organism \
                         (hover and click pick share this)",
                    );
                    ui.label(widgets::mono(
                        format!("{:.0} px", app.pick_radius_px),
                        theme::ACCENT,
                    ));
                });
                ui.horizontal(|ui| {
                    ui.label(widgets::mono("stats refresh rate", theme::WEAK));
                    let hz_slider = egui::Slider::new(&mut app.stats_hz, 2.0..=15.0)
                        .show_value(false);
                    ui.add_sized(egui::Vec2::new(140.0, 18.0), hz_slider)
                        .on_hover_text("How often the statistics charts rebuild the series");
                    ui.label(widgets::mono(format!("{:.0} Hz", app.stats_hz), theme::ACCENT));
                });

                widgets::section_header(ui, "Keyboard shortcuts");
                for (key, desc) in theme::SHORTCUTS {
                    ui.horizontal(|ui| {
                        widgets::keycap(ui, key);
                        widgets::weak_label(ui, *desc);
                    });
                }

                ui.add_space(10.0);
                widgets::section_header(ui, "About");
                widgets::weak_label(ui, "Artificial Life — a Rust ecosystem sandbox.");
                widgets::weak_label(ui, "egui/eframe 0.29 renderer; deterministic core simulation at 60 steps per sim second.");
                widgets::weak_label(ui, "World persistence lives in the saves/ subdirectory of the app data folder; exports under exports/ and experiments/.");
                ui.add_space(20.0);
            });
        });
}

pub fn shortcuts_sheet(app: &mut AlifeApp, ctx: &Context) {
    if !app.show_shortcuts {
        return;
    }
    egui::Window::new("Shortcuts")
        .id(egui::Id::new("alife.shortcuts"))
        .anchor(egui::Align2::RIGHT_CENTER, [-24.0, 0.0])
        .order(egui::Order::Foreground)
        .collapsible(false)
        .resizable(false)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(6.0))
                .stroke(egui::Stroke::new(1.0_f32, theme::STROKE_SOFT))
                .inner_margin(14.0),
        )
        .show(ctx, |ui| {
            for (key, desc) in theme::SHORTCUTS {
                ui.horizontal(|ui| {
                    widgets::keycap(ui, key);
                    widgets::weak_label(ui, *desc);
                });
            }
            ui.add_space(8.0);
            if widgets::mini_btn(ui, "Close", "Hide the cheatsheet (?)").clicked() {
                app.show_shortcuts = false;
            }
        });
}
