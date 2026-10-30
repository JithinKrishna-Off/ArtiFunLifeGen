//! God brush preview on the world canvas: translucent radius circle,
//! crosshair and a small `r=NN` readout at the cursor. Stateless per frame.

use crate::camera::WorldView;
use egui::{Align2, Color32, FontId, Painter, Stroke};

/// Which intervention a previewed brush corresponds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushKind {
    Neutral,
    Rain,
    Drought,
    Lightning,
    Dump,
    Summon,
}

/// Distinct color per kind (drought/rain/storm colors match the effect rings;
/// dump/summon reuse the accent family so the preview reads as "spawn", not
/// as weather).
pub fn brush_color(kind: BrushKind) -> Color32 {
    match kind {
        BrushKind::Neutral => Color32::from_rgb(0x55, 0x60, 0x6c),
        BrushKind::Rain => Color32::from_rgb(0x4a, 0x7b, 0xd0),
        BrushKind::Drought => Color32::from_rgb(0xc0, 0x7a, 0x30),
        BrushKind::Lightning => Color32::from_rgb(0x6a, 0x4a, 0xd0),
        BrushKind::Dump => Color32::from_rgb(0x34, 0xd3, 0x99),
        BrushKind::Summon => Color32::from_rgb(0x45, 0xd2, 0xbf),
    }
}

/// True when two brush kinds render with distinguishable colors.
pub fn colors_distinguishable(a: BrushKind, b: BrushKind) -> bool {
    (a == b) || (brush_color(a) != brush_color(b))
}

/// Screen-space crosshair half-length in px.
pub const CROSSHAIR_HALF: f32 = 8.0;

fn crosshair_points(c: egui::Pos2) -> [([f32; 2], [f32; 2]); 2] {
    [
        ([c.x - CROSSHAIR_HALF, c.y], [c.x + CROSSHAIR_HALF, c.y]),
        ([c.x, c.y - CROSSHAIR_HALF], [c.x, c.y + CROSSHAIR_HALF]),
    ]
}

/// Draw the armed brush preview at the world position `(x, y)` with radius `r`
/// (world units) for `kind`. Cheap: 2 strokes + 2 lines + 1 text label, only
/// while a god tool is armed.
pub fn draw_brush_preview(
    painter: &Painter,
    view: &WorldView,
    x: f32,
    y: f32,
    r: f32,
    kind: BrushKind,
) {
    let c = view.world_to_screen(x, y);
    let rs = (r * view.scale()).max(5.0);
    let base = brush_color(kind);
    let edge = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 185);
    let fill = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 26);
    painter.circle_filled(c, rs, fill);
    painter.circle_stroke(c, rs, Stroke::new(1.5_f32, edge));
    for (a, b) in crosshair_points(c) {
        painter.line_segment(
            [egui::pos2(a[0], a[1]), egui::pos2(b[0], b[1])],
            Stroke::new(1.2_f32, edge),
        );
    }
    painter.text(
        egui::pos2(c.x + rs + 6.0, c.y - rs - 4.0),
        Align2::LEFT_BOTTOM,
        format!("r={:.0}", r.max(0.0)),
        FontId::proportional(11.0),
        edge,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_have_distinguishable_colors() {
        let kinds = [
            BrushKind::Neutral,
            BrushKind::Rain,
            BrushKind::Drought,
            BrushKind::Lightning,
            BrushKind::Dump,
            BrushKind::Summon,
        ];
        for a in kinds {
            for b in kinds {
                assert!(colors_distinguishable(a, b));
            }
        }
    }

    #[test]
    fn crosshair_is_symmetric() {
        let pts = crosshair_points(egui::Pos2::new(10.0, 20.0));
        assert_eq!(pts.len(), 2);
        for (a, b) in pts {
            let midx = (a[0] + b[0]) / 2.0;
            let midy = (a[1] + b[1]) / 2.0;
            assert!((midx - 10.0).abs() < 1e-3);
            assert!((midy - 20.0).abs() < 1e-3);
            assert!(((a[0] - b[0]).abs() == 2.0 * CROSSHAIR_HALF)
                || ((a[1] - b[1]).abs() == 2.0 * CROSSHAIR_HALF));
        }
    }
}
