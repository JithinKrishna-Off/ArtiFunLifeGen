//! Inspector side panel: organism details, trait bars, lineage and the
//! painted neural diagram; deceased organisms fall back to a record card.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets; //
use alife_common::{
    max_energy_for, organism_radius, DeathCause, Genotype, LineageRecord, Network, Organism,
    OUTPUT_LABELS, INPUT_LABELS, NET_HIDDEN, NET_INPUTS, NET_OUTPUTS, TRAIT_MAX, TRAIT_MIN,
};
use alife_core::SimApi;
use alife_render::paint::species_color;
use egui::{pos2, Align2, Color32, FontId, Pos2, Sense, Stroke, Ui, Vec2};

pub fn inspector(app: &mut AlifeApp, ui: &mut Ui) {
    ui.label(widgets::strong("Inspector"));
    ui.add_space(2.0);

    let Some(sel) = app.selected else {
        widgets::empty_state(
            ui,
            "No organism selected",
            "Click an organism in the world view; Esc clears the selection",
            None,
        );
        return;
    };

    let organism = app.sim.organism(sel).cloned();
    let record = app.sim.get_record(sel);

    if let Some(o) = organism {
        living_card(app, ui, &o);
    } else if let Some(rec) = record {
        deceased_card(app, ui, &rec);
    } else {
        widgets::empty_state(
            ui,
            "Organism vanished",
            "The selected id is no longer in the world or the record log",
            None,
        );
    }
}

// ------------------------------------------------------------ living card

#[allow(clippy::too_many_lines)]
fn living_card(app: &mut AlifeApp, ui: &mut Ui, o: &Organism) {
    ui.horizontal(|ui| {
        ui.label(widgets::strong(format!("#{}", o.id)));
        if widgets::mini_btn(ui, "copy id", "Copy the organism id to the clipboard (egui clipboard)")
            .clicked()
        {
            ui.ctx().copy_text(format!("{}", o.id));
            app.toasts
                .push(crate::widgets::ToastKind::Ok, format!("id #{} copied to clipboard", o.id));
        }
    });
    ui.horizontal(|ui| {
        widgets::chip(ui, species_color(o.lineage));
        widgets::weak_label(ui, format!("spec #{} (gen {})", o.lineage, o.generation));
    });
    ui.horizontal(|ui| {
        widgets::weak_label(ui, format!(
            "position {:.0}, {:.0}",
            o.x, o.y
        ));
    });

    // follow camera toggle
    let following = app.follow_selected && app.selected == Some(o.id);
    let follow_btn = ui.selectable_label(following, "Follow");
    if follow_btn.clicked() {
        app.follow_selected = !app.follow_selected;
    }
    follow_btn.on_hover_text("Keep the camera centered on this organism");

    ui.add_space(4.0);
    widgets::section_header(ui, "Vitals");
    widgets::bar_row(
        ui,
        "energy",
        o.energy,
        o.max_energy.max(1.0),
        theme::ACCENT,
        format!("{:.1} / {:.1}", o.energy, o.max_energy).as_str(),
    );
    let lifespan = o.genotype.traits.lifespan.max(1) as f32;
    widgets::bar_row(
        ui,
        "age",
        o.age as f32,
        lifespan,
        theme::WARN,
        format!("{:.0} / {:.0}", o.age, o.genotype.traits.lifespan).as_str(),
    );
    widgets::bar_row(
        ui,
        "offspring",
        o.offspring_count as f32,
        o.offspring_count.max(8) as f32,
        theme::OK,
        format!("{}", o.offspring_count).as_str(),
    );
    widgets::bar_row(
        ui,
        "speed",
        o.speed,
        o.genotype.traits.max_speed.max(1.0),
        theme::TEAL,
        format!("{:.1} u/s", o.speed).as_str(),
    );
    widgets::weak_label(ui, format!(
        "body radius {:.1} units, max energy {:.0}",
        organism_radius(&o.genotype.traits),
        max_energy_for(&o.genotype.traits)
    ));

    ui.add_space(4.0);
    widgets::section_header(ui, "Traits");
    trait_rows(ui, &o.genotype);

    ui.add_space(4.0);
    widgets::section_header(ui, "Lineage");
    ui.horizontal(|ui| {
        match o.mom {
            Some(m) => parent_button(app, ui, "mom", m),
            None => widgets::weak_label(ui, "no parent (founder)"),
        }
        match o.dad {
            Some(d) => parent_button(app, ui, "dad", d),
            None => {}
        }
    });
    if let Some(r) = app.sim.get_record(o.id) {
        widgets::weak_label(ui, format!(
            "born at step {} ({:.0} sim s)",
            r.born_step,
            r.born_step as f32 / 60.0
        ));
    }

    if o.genotype.net.w1.len() == NET_HIDDEN * NET_INPUTS
        && o.genotype.net.w2.len() == NET_OUTPUTS * NET_HIDDEN
        && o.last_inputs.len() == NET_INPUTS
        && o.last_outputs.len() == NET_OUTPUTS
    {
        ui.add_space(4.0);
        widgets::section_header(ui, "Neural activity (live)");
        draw_network(ui, &o.genotype.net, &o.last_inputs, &o.last_outputs);
    }
}

fn parent_button(app: &mut AlifeApp, ui: &mut Ui, rel: &str, id: u64) {
    let alive = app.sim.organism(id).is_some();
    let label = format!("{} #{}", rel, id);
    if alive {
        let btn = ui.selectable_label(false, &label);
        if btn.clicked() {
            app.selected = Some(id);
            app.show_inspector = true;
        }
        btn.on_hover_text("Select this parent (currently alive)");
    } else {
        ui.label(widgets::mono(label, theme::WEAK));
    }
}

fn trait_rows(ui: &mut Ui, g: &Genotype) {
    let f = |v: f32, min: f32, max: f32| -> f32 {
        if max > min {
            (v - min) / (max - min)
        } else {
            0.0
        }
    };
    let t = g.traits;
    widgets::bar_row(ui, "max_speed", f(t.max_speed, TRAIT_MIN.max_speed, TRAIT_MAX.max_speed), 1.0, theme::ACCENT, format!("{:.0} u/s", t.max_speed).as_str());
    widgets::bar_row(ui, "efficiency", f(t.efficiency, TRAIT_MIN.efficiency, TRAIT_MAX.efficiency), 1.0, theme::OK, format!("{:.2}", t.efficiency).as_str());
    widgets::bar_row(ui, "sensor_range", f(t.sensor_range, TRAIT_MIN.sensor_range, TRAIT_MAX.sensor_range), 1.0, theme::TEAL, format!("{:.0}", t.sensor_range).as_str());
    widgets::bar_row(ui, "metabolism", f(t.metabolism, TRAIT_MIN.metabolism, TRAIT_MAX.metabolism), 1.0, theme::WARN, format!("{:.2}", t.metabolism).as_str());
    widgets::bar_row(ui, "size", f(t.size, TRAIT_MIN.size, TRAIT_MAX.size), 1.0, Color32::from_rgb(126, 226, 148), format!("{:.2}", t.size).as_str());
    widgets::bar_row(ui, "repr_threshold", f(t.repr_threshold, TRAIT_MIN.repr_threshold, TRAIT_MAX.repr_threshold), 1.0, theme::DANGER, format!("{:.2}", t.repr_threshold).as_str());
    widgets::bar_row(ui, "repr_cost", f(t.repr_cost, TRAIT_MIN.repr_cost, TRAIT_MAX.repr_cost), 1.0, theme::DANGER, format!("{:.2}", t.repr_cost).as_str());
    let life = f(t.lifespan as f32, TRAIT_MIN.lifespan as f32, TRAIT_MAX.lifespan as f32);
    widgets::bar_row(ui, "lifespan", life, 1.0, theme::WEAK, format!("{} steps", t.lifespan).as_str());
}

// ---------------------------------------------------------- deceased card

fn deceased_card(app: &mut AlifeApp, ui: &mut Ui, rec: &LineageRecord) {
    ui.label(widgets::strong(format!("Deceased #{}", rec.id)));
    if widgets::mini_btn(ui, "copy id", "Copy the organism id to the clipboard (egui clipboard)")
        .clicked()
    {
        ui.ctx().copy_text(format!("{}", rec.id));
        app.toasts
            .push(crate::widgets::ToastKind::Ok, format!("id #{} copied to clipboard", rec.id));
    }
    ui.horizontal(|ui| {
        widgets::chip(ui, species_color(rec.species_id));
        widgets::weak_label(ui, format!("spec #{} (gen {})", rec.species_id, rec.generation));
    });

    ui.add_space(4.0);
    widgets::section_header(ui, "Death");
    let cause = rec.cause.map(cause_label).unwrap_or("unknown");
    ui.label(egui::RichText::new(cause).color(theme::DANGER));
    match rec.death_step {
        Some(d) => widgets::weak_label(ui, format!("died at step {}", d)),
        None => widgets::weak_label(ui, "death step not yet available"),
    }
    widgets::weak_label(ui, "position not recorded in the engine's lineage record");

    ui.add_space(4.0);
    widgets::section_header(ui, "Life summary");
    let life_len = rec
        .death_step
        .map(|d| d.saturating_sub(rec.born_step))
        .unwrap_or(0);
    widgets::weak_label(ui, format!(
        "born at step {}",
        rec.born_step
    ));
    if rec.death_step.is_some() {
        widgets::weak_label(ui, format!("lifetime {} steps = {:.0} sim s", life_len, life_len as f32 / 60.0));
    }
}

fn cause_label(c: DeathCause) -> &'static str {
    match c {
        DeathCause::Starvation => "starvation",
        DeathCause::OldAge => "old age",
        DeathCause::Hazard => "hazard damage",
    }
}

// -------------------------------------------------------- neural diagram

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| -> u8 { (x as f32 + (y as f32 - x as f32) * t) as u8 };
    Color32::from_rgba_unmultiplied(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
        mix(a.a(), b.a()),
    )
}

const MAX_LINES: usize = 300;

fn draw_network(ui: &mut Ui, net: &Network, inputs: &[f32], outputs: &[f32]) {
    let width = ui.available_width().max(220.0);
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(width, 210.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 4.0, Color32::from_rgb(12, 16, 21));

    let x_in = rect.min.x + 62.0;
    let x_hi = rect.center().x;
    let x_out = rect.max.x - 74.0;
    let top = rect.min.y + 14.0;
    let bottom = rect.max.y - 24.0;
    let spread = |i: usize, n: usize| top + (i as f32 + 0.5) * (bottom - top) / n as f32;

    let in_pos: Vec<Pos2> = (0..NET_INPUTS).map(|i| pos2(x_in, spread(i, NET_INPUTS))).collect();
    let hi_pos: Vec<Pos2> =
        (0..NET_HIDDEN).map(|h| pos2(x_hi, spread(h, NET_HIDDEN))).collect();
    let out_pos: Vec<Pos2> =
        (0..NET_OUTPUTS).map(|o| pos2(x_out, spread(o, NET_OUTPUTS))).collect();

    // collect candidate edges (type: (abs_w, from, to, positive)), w1 then w2
    let mut edges: Vec<(f32, Pos2, Pos2, bool)> = Vec::with_capacity(NET_HIDDEN * NET_INPUTS + NET_OUTPUTS * NET_HIDDEN);
    for h in 0..NET_HIDDEN {
        for i in 0..NET_INPUTS {
            let w = net.w1[h * NET_INPUTS + i];
            if w.to_bits() == 0 {
                continue;
            }
            edges.push((w.abs(), in_pos[i], hi_pos[h], w > 0.0));
        }
    }
    for o in 0..NET_OUTPUTS {
        for h in 0..NET_HIDDEN {
            let w = net.w2[o * NET_HIDDEN + h];
            if w.to_bits() == 0 {
                continue;
            }
            edges.push((w.abs(), hi_pos[h], out_pos[o], w > 0.0));
        }
    }
    edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    edges.truncate(MAX_LINES);
    let max_e = edges.iter().map(|e| e.0).fold(0.0f32, f32::max).max(1e-6);
    for (aw, from, to, positive) in &edges {
        let t = *aw / max_e;
        let color = if *positive { theme::ACCENT } else { theme::DANGER };
        p.line_segment(
            [*from, *to],
            Stroke::new(0.4 + 1.8 * t, color.gamma_multiply(0.25 + 0.6 * t)),
        );
    }

    // nodes
    for (i, ipos) in in_pos.iter().enumerate() {
        let v = inputs.get(i).copied().unwrap_or(0.0);
        draw_node(p, *ipos, v);
        p.text(
            pos2(x_in - 8.0, ipos.y),
            Align2::RIGHT_CENTER,
            INPUT_LABELS[i],
            FontId::proportional(8.5),
            theme::WEAK,
        );
    }
    for hpos in &hi_pos {
        draw_node(p, *hpos, 0.0);
    }
    for (o, opos) in out_pos.iter().enumerate() {
        let v = outputs.get(o).copied().unwrap_or(0.0);
        draw_node(p, *opos, v);
        p.text(
            pos2(x_out + 8.0, opos.y),
            Align2::LEFT_CENTER,
            OUTPUT_LABELS[o],
            FontId::proportional(8.5),
            theme::WEAK,
        );
    }

    p.text(
        pos2(rect.left(), rect.bottom() - 10.0),
        Align2::LEFT_BOTTOM,
        "+ positive weight (cyan)   - negative weight (coral)",
        FontId::monospace(8.5),
        theme::WEAK,
    );
}

fn draw_node(p: &egui::Painter, center: Pos2, v: f32) {
    let base = Color32::from_rgb(24, 30, 38);
    let fill = if v > 0.01 {
        lerp_color(base, theme::ACCENT, v.min(1.0) * 0.8)
    } else if v < -0.01 {
        lerp_color(base, theme::DANGER, (-v).min(1.0) * 0.8)
    } else {
        base
    };
    p.circle_filled(center, 4.5, fill);
    p.circle_stroke(center, 4.5, Stroke::new(1.0_f32, theme::STROKE_SOFT));
}
