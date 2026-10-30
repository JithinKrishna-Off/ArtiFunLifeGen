//! First-run tutorial: 4 short steps, one at a time, persisted via a done
//! marker file (`tutorial_done.txt`) under the app data root so it only ever
//! runs once per user.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets;
use egui::{Align2, Context, Id, Order};
use std::path::Path;

/// Marker file name under the data root.
const DONE_FILE: &str = "tutorial_done.txt";

const STEPS: [(&str, &str, &str); 4] = [
    (
        "Run the world",
        "This is a fixed-step engine: 60 simulation steps happen per sim second. Nothing moves until it is running.",
        "Try it: press Space to start — the toolbar button turns into Pause.",
    ),
    (
        "Inspect an organism",
        "Every organism runs its own neural network. Clicking one selects it and the inspector panel shows body, traits, lineage and the live network.",
        "Try it: click a glowing organism in the world view.",
    ),
    (
        "Change the time scale",
        "The toolbar segmented control switches speed — real-time, fast, generational, century and Millennium+ (budgeted). Fast-forward and watch species drift and die out.",
        "Try it: set the time scale to Fast and watch the population churn.",
    ),
    (
        "God Mode",
        "You command the weather and catastrophes: rain, drought, storms, lightning, nutrient dumps, meteors, plus summon, purge and protect lineage tools.",
        "Try it: press K to open God Mode, pick Rain, then click the map.",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TutorialState {
    pub active: bool,
    pub step: usize,
}

/// Fresh app state: active only when the marker file is absent.
pub fn initial_state(root: &Path) -> TutorialState {
    TutorialState {
        active: !root.join(DONE_FILE).exists(),
        step: 0,
    }
}

fn mark_done(app: &mut AlifeApp) {
    if let Err(e) = std::fs::write(app.store.root.join(DONE_FILE), "done\n") {
        app.toasts
            .push(crate::widgets::ToastKind::Warn, format!("tutorial marker not written: {e}"));
    }
    app.tutorial.active = false;
}

pub fn show(app: &mut AlifeApp, ctx: &Context) {
    if !app.tutorial.active {
        return;
    }
    let step = app.tutorial.step.min(STEPS.len() - 1);
    let (title, body, try_hint) = STEPS[step];

    egui::Window::new("Getting started")
        .id(Id::new("alife.tutorial"))
        .anchor(Align2::CENTER_CENTER, [0.0, -30.0])
        .order(Order::Foreground)
        .collapsible(false)
        .resizable(false)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_PANEL2)
                .rounding(egui::Rounding::same(8.0))
                .stroke(egui::Stroke::new(1.0_f32, theme::ACCENT.gamma_multiply(0.7)))
                .inner_margin(18.0),
        )
        .show(ctx, |ui| {
            ui.set_max_width(430.0);
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(format!("Step {} of {}", step + 1, STEPS.len()))
                        .monospace()
                        .size(10.5)
                        .color(theme::WEAK),
                );
                ui.add_space(6.0);
                ui.label(widgets::strong(title));
                ui.add_space(8.0);
                ui.label(widgets::weak(body));
                ui.add_space(10.0);
                ui.label(egui::RichText::new(try_hint).size(12.0).color(theme::ACCENT));
                ui.add_space(14.0);
                let last = step + 1 == STEPS.len();
                ui.horizontal(|ui| {
                    if widgets::mini_btn(ui, "Skip", "Dismiss the tutorial for good").clicked() {
                        mark_done(app);
                    }
                    let next_label = if last { "Finish" } else { "Next" };
                    let tip = if last { "Complete the tutorial" } else { "Go to the next step" };
                    if ui
                        .add(egui::Button::new(egui::RichText::new(next_label).strong()))
                        .on_hover_text(tip)
                        .clicked()
                    {
                        if last {
                            mark_done(app);
                        } else {
                            app.tutorial.step = step + 1;
                        }
                    }
                });
            });
        });
}
