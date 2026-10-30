//! Visual design tokens + egui style override (dark scientific theme).

use egui::{
    style::Selection, Color32, Context, FontId, Rounding, Stroke, TextStyle, Vec2, Visuals,
};

pub const BG_WINDOW: Color32 = Color32::from_rgb(10, 13, 17); // #0a0d11
pub const BG_PANEL: Color32 = Color32::from_rgb(17, 22, 28); // #11161c
pub const BG_PANEL2: Color32 = Color32::from_rgb(23, 30, 38); // #171e26
pub const BG_HOVER: Color32 = Color32::from_rgb(30, 39, 49);
pub const BG_ACTIVE: Color32 = Color32::from_rgb(38, 49, 60);
pub const STROKE: Color32 = Color32::from_rgb(35, 44, 53); // #232c35
pub const STROKE_SOFT: Color32 = Color32::from_rgb(43, 54, 65);
pub const TEXT: Color32 = Color32::from_rgb(221, 228, 234); // #dde4ea
pub const WEAK: Color32 = Color32::from_rgb(139, 150, 162); // #8b96a2
pub const ACCENT: Color32 = Color32::from_rgb(45, 212, 191); // cyan #2dd4bf
pub const TEAL: Color32 = Color32::from_rgb(20, 184, 166); // #14b8a6
pub const OK: Color32 = Color32::from_rgb(52, 211, 153); // #34d399
pub const WARN: Color32 = Color32::from_rgb(245, 158, 11); // #f59e0b
pub const DANGER: Color32 = Color32::from_rgb(248, 113, 113); // #f87171

/// Apply the theme once at startup.
pub fn apply(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    tokens(&mut style);
    ctx.set_style(style);
}

pub fn tokens(style: &mut egui::Style) {
    let mut v = Visuals::dark();
    v.dark_mode = true;
    v.panel_fill = BG_PANEL;
    v.window_fill = BG_WINDOW;
    v.extreme_bg_color = Color32::from_rgb(8, 10, 14);
    v.faint_bg_color = BG_PANEL2;
    v.override_text_color = Some(TEXT);
    v.selection = Selection {
        bg_fill: ACCENT.gamma_multiply(0.30),
        stroke: Stroke::new(1.0_f32, ACCENT),
    };
    v.widgets.noninteractive.bg_fill = BG_PANEL2;
    v.widgets.noninteractive.weak_bg_fill = BG_PANEL2;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, WEAK);
    v.widgets.noninteractive.bg_stroke = Stroke::new(0.5_f32, STROKE);
    v.widgets.noninteractive.rounding = Rounding::same(4.0);

    v.widgets.inactive.weak_bg_fill = BG_PANEL2;
    v.widgets.inactive.bg_fill = BG_PANEL2;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, STROKE);
    v.widgets.inactive.rounding = Rounding::same(4.0);

    v.widgets.hovered.weak_bg_fill = BG_HOVER;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, STROKE_SOFT);
    v.widgets.hovered.rounding = Rounding::same(4.0);

    v.widgets.active.weak_bg_fill = BG_ACTIVE;
    v.widgets.active.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    v.widgets.active.bg_stroke = Stroke::new(1.2_f32, ACCENT);
    v.widgets.active.rounding = Rounding::same(4.0);

    v.widgets.open.weak_bg_fill = BG_PANEL2;
    v.widgets.open.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.open.bg_stroke = Stroke::new(1.0_f32, STROKE);
    v.widgets.open.rounding = Rounding::same(4.0);

    // egui 0.29 has no separate "focused" bucket: text-entry style lives on `hovered`
    v.widgets.hovered.weak_bg_fill = BG_HOVER;
    v.widgets.hovered.bg_fill = Color32::from_rgb(26, 34, 42);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);

    v.menu_rounding = Rounding::same(6.0);
    v.window_rounding = Rounding::same(6.0);
    v.window_stroke = Stroke::new(1.0_f32, STROKE);
    style.visuals = v;

    // typography: body 13, monospace tabular numerals for readouts
    style.text_styles = [
        (TextStyle::Body, FontId::proportional(13.0)),
        (TextStyle::Button, FontId::proportional(13.0)),
        (TextStyle::Heading, FontId::proportional(15.0)),
        (TextStyle::Monospace, FontId::monospace(11.5)),
        (TextStyle::Small, FontId::proportional(10.5)),
    ]
    .into_iter()
    .collect();

    style.spacing.button_padding = Vec2::new(10.0, 4.0);
    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.interact_size.y = 24.0;
    style.spacing.slider_width = 140.0;
    style.spacing.indent = 16.0;
}

/// Keyboard key list for the cheatsheet.
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("Space", "play / pause"),
    ("Right Arrow", "single step (when paused)"),
    ("Ctrl+N", "new world dialog"),
    ("Ctrl+Z", "undo last editor edit (Ctrl+Shift+Z / Ctrl+Y: redo)"),
    ("S", "save dialog"),
    ("R", "reset world (confirm)"),
    ("K", "God Mode view (weather, catastrophes, lineage tools)"),
    ("T", "evolutionary tree view"),
    ("I", "toggle inspector panel"),
    ("F", "fit world to view"),
    ("+ / -", "zoom in / out"),
    ("Esc", "close dialog / exit tool / clear selection"),
    ("G", "toggle coordinate grid overlay"),
    ("?", "shortcut cheatsheet (this panel)"),
    ("Wheel", "zoom at cursor"),
    ("Drag", "pan camera"),
    ("Click", "select organism / apply editor tool"),
];
