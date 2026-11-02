//! Shared UI widgets: toasts, bars, mono readouts, section headers,
//! icon buttons, sparklines, empty states.

use crate::theme;
use egui::{
    pos2, Align2, Button, Color32, Context, Id, Label, Rect, Response, RichText, Sense, Stroke, Ui,
    Vec2,
};

// ---------------------------------------------------------------- toasts

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ToastKind {
    Info,
    Ok,
    Warn,
    Error,
}

struct Toast {
    kind: ToastKind,
    msg: String,
    age: f32,
}

const TOAST_TTL: f32 = 4.2;
const TOAST_WIDTH: f32 = 310.0;

#[derive(Default)]
pub struct Toasts {
    items: Vec<Toast>,
}

impl Toasts {
    pub fn push(&mut self, kind: ToastKind, msg: impl Into<String>) {
        self.items.push(Toast { kind, msg: msg.into(), age: 0.0 });
        if self.items.len() > 5 {
            self.items.remove(0);
        }
    }

    pub fn update(&mut self, dt: f32) {
        for t in &mut self.items {
            t.age += dt;
        }
        self.items.retain(|t| t.age < TOAST_TTL);
    }

    pub fn show(&mut self, ctx: &Context) {
        egui::Area::new(Id::new("alife.toasts"))
            .anchor(Align2::RIGHT_BOTTOM, [-14.0, -30.0])
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                for t in self.items.iter().rev() {
                    let fade = ((TOAST_TTL - t.age) / 0.6).clamp(0.0, 1.0);
                    let a = (fade * 255.0) as u8;
                    let col = match t.kind {
                        ToastKind::Info => theme::ACCENT,
                        ToastKind::Ok => theme::OK,
                        ToastKind::Warn => theme::WARN,
                        ToastKind::Error => theme::DANGER,
                    };
                    let bg = Color32::from_rgba_unmultiplied(23, 30, 38, a);
                    let txt = Color32::from_rgba_unmultiplied(221, 228, 234, a);
                    egui::Frame::none()
                        .fill(bg)
                        .rounding(egui::Rounding::same(6.0))
                        .stroke(Stroke::new(1.0_f32, col.gamma_multiply(0.6)))
                        .inner_margin(egui::Margin::symmetric(10.0, 7.0))
                        .show(ui, |ui| {
                            ui.set_width(TOAST_WIDTH);
                            ui.horizontal(|ui| {
                                let glyph = match t.kind {
                                    ToastKind::Info => "[i]",
                                    ToastKind::Ok => "[+]",
                                    ToastKind::Warn => "[!]",
                                    ToastKind::Error => "[x]",
                                };
                                ui.label(RichText::new(glyph).monospace().size(11.0).color(col));
                                ui.label(RichText::new(&t.msg).size(12.0).color(txt));
                            });
                        });
                    ui.add_space(6.0);
                }
            });
    }
}

// -------------------------------------------------------------- primitives

/// Monospace readout (tabular numerals) sized 11 by default.
pub fn mono(text: impl Into<String>, color: Color32) -> RichText {
    RichText::new(text.into()).monospace().size(11.0).color(color)
}

pub fn strong(text: impl Into<String>) -> RichText {
    RichText::new(text.into()).strong().color(theme::TEXT)
}

pub fn weak(text: impl Into<String>) -> RichText {
    RichText::new(text.into()).weak()
}

pub fn section_header(ui: &mut Ui, name: &str) {
    ui.add_space(3.0);
    ui.label(RichText::new(name.to_uppercase()).small().strong().color(theme::WEAK));
    ui.add_space(3.0);
}

pub fn weak_label(ui: &mut Ui, s: impl Into<String>) {
    ui.label(RichText::new(s.into()).weak());
}

/// Horizontal bar with label, proportional fill and a mono value on the right.
#[allow(clippy::too_many_arguments)]
pub fn bar_row(ui: &mut Ui, label: &str, value: f32, max: f32, color: Color32, value_text: &str) -> Response {
    let row = ui.horizontal(|ui| {
        ui.add_sized(Vec2::new(104.0, 12.0), Label::new(RichText::new(label).weak().small()));
        let bar_w = (ui.available_width() - 64.0).max(40.0);
        let h = 10.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(bar_w, h), Sense::hover());
        ui.painter().rect_filled(rect, 3.0, Color32::from_rgb(12, 16, 21));
        let f = if max > 0.0 { (value / max).clamp(0.0, 1.0) } else { 0.0 };
        if f > 0.004 {
            ui.painter().rect_filled(
                Rect::from_min_size(
                    pos2(rect.min.x, rect.min.y),
                    Vec2::new(rect.width() * f, h),
                ),
                3.0,
                color,
            );
        }
        ui.add_sized(Vec2::new(60.0, 12.0), Label::new(mono(value_text.to_string(), theme::TEXT)));
        resp
    });
    row.inner
}

/// Icon/compact button with tooltip. Returns true when clicked.
pub fn icon_btn(ui: &mut Ui, glyph: &str, tooltip: &str) -> Response {
    ui.add_sized(Vec2::new(30.0, 24.0), Button::new(glyph))
        .on_hover_text(tooltip.to_string())
}

/// Compact label button with tooltip.
pub fn mini_btn(ui: &mut Ui, label: &str, tooltip: &str) -> Response {
    ui.add(Button::new(RichText::new(label).size(12.0)))
        .on_hover_text(tooltip.to_string())
}

/// Toggle-style selectable chip button (explicit selected state).
pub fn select_btn(ui: &mut Ui, label: &str, selected: bool, tooltip: &str) -> Response {
    let txt = if selected {
        RichText::new(label).size(12.0).strong().color(theme::TEXT)
    } else {
        RichText::new(label).size(12.0).weak()
    };
    let mut btn = Button::new(txt);
    if selected {
        btn = btn.stroke(Stroke::new(1.2_f32, theme::ACCENT)).fill(theme::BG_ACTIVE);
    }
    ui.add(btn).on_hover_text(tooltip.to_string())
}

/// Inline sparkline (normalized painter polyline; no plot overhead).
#[allow(dead_code)]
pub fn sparkline(ui: &mut Ui, values: &[f32], color: Color32) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width().max(40.0), 26.0),
        Sense::hover(),
    );
    let p = ui.painter();
    p.rect_filled(rect, 2.0, Color32::from_rgb(12, 16, 21));
    if values.len() < 2 {
        p.text(rect.center(), Align2::CENTER_CENTER, "no series yet", egui::FontId::proportional(10.0), theme::WEAK);
        return;
    }
    let min = values.iter().copied().fold(f32::MAX, f32::min);
    let max = values.iter().copied().fold(f32::MIN, f32::max);
    let span = (max - min).max(1e-6);
    let pts: Vec<egui::Pos2> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = i as f32 / (values.len() - 1) as f32;
            let y = rect.max.y - 2.0 - (v - min) / span * (rect.height() - 4.0);
            let x = rect.min.x + t * rect.width();
            pos2(x, y)
        })
        .collect();
    p.line_segment(
        [pos2(rect.min.x, rect.max.y - 0.5), pos2(rect.max.x, rect.max.y - 0.5)],
        Stroke::new(1.0_f32, theme::STROKE),
    );
    p.add(egui::Shape::line(pts, Stroke::new(1.3_f32, color)));
}

/// Centered empty-state; returns true when the optional action was clicked.
pub fn empty_state(ui: &mut Ui, title: &str, hint: &str, action: Option<&str>) -> bool {
    ui.vertical_centered_justified(|ui| {
        ui.add_space(48.0);
        ui.label(RichText::new(title).heading().color(theme::TEXT));
        ui.add_space(4.0);
        ui.label(RichText::new(hint).weak().small());
        if let Some(label) = action {
            ui.add_space(16.0);
            ui.add(Button::new(RichText::new(label).size(12.0).strong())).clicked()
        } else {
            false
        }
    })
    .inner
}

/// Colored species chip.
pub fn chip(ui: &mut Ui, color: Color32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
    ui.painter().rect_filled(rect, 3.0, color);
    resp.on_hover_ui(|ui| {
        ui.style_mut().visuals.override_text_color = theme::TEXT.into();
        let _ = ui;
    });
}

/// Keycap display for the cheatsheet.
pub fn keycap(ui: &mut Ui, key: &str) {
    let badge = Button::new(
        RichText::new(key).monospace().size(10.5).color(theme::ACCENT),
    )
    .fill(Color32::from_rgb(14, 18, 23))
    .stroke(Stroke::new(1.0_f32, theme::STROKE))
    .rounding(egui::Rounding::same(4.0));
    ui.add_sized(Vec2::new(96.0, 20.0), badge);
}
