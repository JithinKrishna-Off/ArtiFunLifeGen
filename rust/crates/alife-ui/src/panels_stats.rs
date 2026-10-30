//! Statistics + Evolution views: history line charts (egui_plot), species
//! table and trait histograms, with time-throttled data caches.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets;
use alife_common::persist::HistoryBuffer;
use alife_common::{TRAIT_MAX, TRAIT_MIN};
use alife_core::SimApi;
use alife_render::paint::species_color;
use egui::{Color32, Context};
use egui_plot::{Legend, Line, Plot};
use std::time::Instant;

// -------------------------------------------------------------------- cache

/// Throttled snapshot of the history buffer used by the Statistics view.
#[derive(Default)]
pub struct StatsCache {
    pub built: Option<Instant>,
    pub population: Vec<[f64; 2]>,
    pub births: Vec<[f64; 2]>,
    pub deaths: Vec<[f64; 2]>,
    pub avg_energy: Vec<[f64; 2]>,
    pub avg_age: Vec<[f64; 2]>,
    pub avg_generation: Vec<[f64; 2]>,
    pub diversity: Vec<[f64; 2]>,
    pub food: Vec<[f64; 2]>,
    /// wet coverage in [0, 1] per sample (weather layer)
    pub wet: Vec<[f64; 2]>,
    /// species id -> series (top species first)
    pub species: Vec<(u32, Vec<[f64; 2]>)>,
}

impl StatsCache {
    fn stale(&self, hz: f32) -> bool {
        match self.built {
            Some(t) => t.elapsed().as_secs_f32() >= 1.0 / hz.max(1.0),
            None => true,
        }
    }

    fn rebuild(&mut self, h: &HistoryBuffer) {
        self.population.clear();
        self.births.clear();
        self.deaths.clear();
        self.avg_energy.clear();
        self.avg_age.clear();
        self.avg_generation.clear();
        self.diversity.clear();
        self.food.clear();
        self.wet.clear();
        let mut per_species: std::collections::HashMap<u32, Vec<(u64, u32)>> =
            std::collections::HashMap::new();
        for s in &h.samples {
            let x = s.step as f64;
            self.population.push([x, s.population as f64]);
            self.births.push([x, s.births as f64]);
            self.deaths.push([x, s.deaths as f64]);
            self.avg_energy.push([x, s.avg_energy as f64]);
            self.avg_age.push([x, s.avg_age as f64]);
            self.avg_generation.push([x, s.avg_generation as f64]);
            self.diversity.push([x, s.diversity as f64]);
            self.food.push([x, s.food_count as f64]);
            self.wet.push([x, s.wet_cover as f64]);
            for (id, count) in &s.species_top {
                let series = per_species.entry(*id).or_default();
                series.push((s.step, *count));
            }
        }
        // order species by their most recent count, keep top 8
        let mut order: Vec<(u32, u32)> = per_species
            .iter()
            .map(|(id, pts)| (*id, pts.last().map(|p| p.1).unwrap_or(0)))
            .collect();
        order.sort_by(|a, b| b.1.cmp(&a.1));
        order.truncate(8);
        self.species = order
            .into_iter()
            .map(|(id, _)| {
                let series = per_species
                    .remove(&id)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(step, c)| [step as f64, c as f64])
                    .collect();
                (id, series)
            })
            .collect();
        self.built = Some(Instant::now());
    }
}

/// One row of the species table.
pub struct SpeciesRow {
    pub id: u32,
    pub count: u32,
    pub avg_generation: f32,
}

/// Throttled snapshot used by the Evolution view (~2 Hz).
#[derive(Default)]
pub struct HistCache {
    pub built: Option<Instant>,
    pub bins_speed: [u32; 16],
    pub bins_efficiency: [u32; 16],
    pub bins_sensor: [u32; 16],
    pub species: Vec<SpeciesRow>,
}

impl HistCache {
    fn stale(&self) -> bool {
        match self.built {
            Some(t) => t.elapsed().as_secs_f32() >= 0.5,
            None => true,
        }
    }

    fn rebuild(&mut self, organisms: &[alife_common::Organism]) {
        self.bins_speed = [0; 16];
        self.bins_efficiency = [0; 16];
        self.bins_sensor = [0; 16];
        let mut per: std::collections::HashMap<u32, (u32, u64)> = std::collections::HashMap::new();
        for o in organisms {
            bin_of(&mut self.bins_speed, o.genotype.traits.max_speed, TRAIT_MIN.max_speed, TRAIT_MAX.max_speed);
            bin_of(&mut self.bins_efficiency, o.genotype.traits.efficiency, TRAIT_MIN.efficiency, TRAIT_MAX.efficiency);
            bin_of(&mut self.bins_sensor, o.genotype.traits.sensor_range, TRAIT_MIN.sensor_range, TRAIT_MAX.sensor_range);
            let e = per.entry(o.lineage).or_insert((0, 0));
            e.0 += 1;
            e.1 += o.generation as u64;
        }
        let mut rows: Vec<SpeciesRow> = per
            .into_iter()
            .map(|(id, (count, gens))| SpeciesRow {
                id,
                count,
                avg_generation: if count > 0 { gens as f32 / count as f32 } else { 0.0 },
            })
            .collect();
        rows.sort_by(|a, b| b.count.cmp(&a.count).then(a.id.cmp(&b.id)));
        self.species = rows;
        self.built = Some(Instant::now());
    }
}

fn bin_of(bins: &mut [u32; 16], v: f32, min: f32, max: f32) {
    let t = if max > min { (v - min) / (max - min) } else { 0.0 };
    let idx = (t.clamp(0.0, 0.999) * 16.0) as usize;
    bins[idx] += 1;
}

// ------------------------------------------------------------------ plots

fn plot_single(ui: &mut egui::Ui, id: &str, series: &[[f64; 2]], name: &str, color: Color32) {
    Plot::new(egui::Id::new(id))
        .height(140.0)
        .x_axis_label("sim step")
        .show(ui, |p| {
            p.line(Line::new(series.to_vec()).name(name).color(color));
        });
}

fn plot_legend(ui: &mut egui::Ui, id: &str, series: &[(&str, &[[f64; 2]], Color32)]) {
    Plot::new(egui::Id::new(id))
        .height(140.0)
        .x_axis_label("sim step")
        .legend(Legend::default())
        .show(ui, |p| {
            for (name, pts, color) in series {
                p.line(Line::new(pts.to_vec()).name(*name).color(*color));
            }
        });
}

// ------------------------------------------------------------ Statistics

pub fn statistics_view(app: &mut AlifeApp, ctx: &Context) {
    let stale = app
        .stats_cache
        .as_ref()
        .map_or(true, |c| c.stale(app.stats_hz));
    if stale {
        let mut c = app.stats_cache.take().unwrap_or_default();
        c.rebuild(&app.history);
        app.stats_cache = Some(c);
    }

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            if app.history.samples.is_empty() {
                let run = widgets::empty_state(
                    ui,
                    "No statistics yet",
                    "Charts appear once the simulation collects history samples",
                    Some("Run simulation"),
                );
                if run {
                    app.running = true;
                }
                return;
            }
            let Some(cache) = &app.stats_cache else { return };
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("World statistics"));
                ui.label(widgets::weak(format!(
                    "{} samples at {} step intervals — refresh {:.0} Hz",
                    app.history.samples.len(),
                    app.history.interval,
                    app.stats_hz
                )));
                ui.add_space(8.0);

                widgets::section_header(ui, "Population");
                plot_single(ui, "alife.ch.pop", &cache.population, "population", theme::ACCENT);

                widgets::section_header(ui, "Births and deaths (per sample interval)");
                plot_legend(
                    ui,
                    "alife.ch.bd",
                    &[
                        ("births", &cache.births, theme::OK),
                        ("deaths", &cache.deaths, theme::DANGER),
                    ],
                );

                widgets::section_header(ui, "Average energy");
                plot_single(ui, "alife.ch.energy", &cache.avg_energy, "avg energy", theme::WARN);

                widgets::section_header(ui, "Average age (steps)");
                plot_single(ui, "alife.ch.age", &cache.avg_age, "avg age", theme::TEAL);

                widgets::section_header(ui, "Average generation");
                plot_single(ui, "alife.ch.gen", &cache.avg_generation, "avg generation", theme::OK);

                widgets::section_header(ui, "Diversity index");
                plot_single(ui, "alife.ch.div", &cache.diversity, "diversity", theme::ACCENT);

                widgets::section_header(ui, "Food count");
                plot_single(ui, "alife.ch.food", &cache.food, "food", Color32::from_rgb(126, 226, 148));

                widgets::section_header(ui, "Wet coverage (weather layer)");
                let any_wet = cache.wet.iter().any(|p| p[1] > 0.0);
                if any_wet {
                    plot_single(
                        ui,
                        "alife.ch.wet",
                        &cache.wet,
                        "wet cover",
                        Color32::from_rgb(96, 165, 250),
                    );
                } else {
                    ui.label(widgets::weak(
                        "not yet available — wet coverage appears once a weather event runs",
                    ))
                    .on_hover_text("HistorySample.wet_cover stays 0 while the world has no active weather");
                }

                widgets::section_header(ui, "Top species population share");
                if cache.species.is_empty() {
                    widgets::weak_label(ui, "No species tracked yet");
                } else {
                    Plot::new(egui::Id::new("alife.ch.species"))
                        .height(160.0)
                        .x_axis_label("sim step")
                        .y_axis_label("count")
                        .legend(Legend::default())
                        .show(ui, |p| {
                            for (id, pts) in &cache.species {
                                p.line(
                                    Line::new(pts.clone())
                                        .name(mode_sp_label(*id))
                                        .color(species_color(*id)),
                                );
                            }
                        });
                    ui.horizontal(|ui| {
                        for (id, _) in &cache.species {
                            widgets::chip(ui, species_color(*id));
                            ui.label(widgets::mono(format!("species {}", id), theme::WEAK));
                        }
                    });
                }

                widgets::section_header(ui, "Deaths by cause");
                let tox = crate::god_api::diagnosis(&app.sim);
                match tox {
                    Some(d) => {
                        ui.label(widgets::mono("engine diagnosis (authoritative)", theme::WEAK));
                        death_cause_card(ui, [d.starvation, d.old_age, d.hazard]);
                    }
                    None if app.death_seen => {
                        ui.label(widgets::mono(
                            "engine diagnosis: not yet available — showing deaths observed this session",
                            theme::WARN,
                        ));
                        death_cause_card(ui, app.death_causes);
                    }
                    None => {
                        ui.label(widgets::weak(
                            "no deaths observed this session and the engine diagnosis yields \
                             no breakdown yet",
                        ));
                    }
                }
                ui.add_space(20.0);
            });
        });
}

fn mode_sp_label(id: u32) -> String {
    format!("species {}", id)
}

/// Session-local death-cause breakdown (starvation / old age / hazard) drawn
/// as proportional bars with mono counts.
fn death_cause_card(ui: &mut egui::Ui, causes: [u64; 3]) {
    let names = [("starvation", theme::DANGER), ("old age", theme::WEAK), ("hazard", theme::WARN)];
    let max = causes.iter().copied().max().unwrap_or(0) as f32;
    let total: u64 = causes.iter().sum();
    for (i, (name, color)) in names.into_iter().enumerate() {
        let v = causes[i];
        widgets::bar_row(
            ui,
            name,
            v as f32,
            max.max(1.0),
            color,
            &format!("{} ({:.1}%)", v, if total > 0 { v as f32 * 100.0 / total as f32 } else { 0.0 }),
        );
    }
}


// -------------------------------------------------------------- Evolution

pub fn evolution_view(app: &mut AlifeApp, ctx: &Context) {
    let stale = app
        .hist_cache
        .as_ref()
        .map_or(true, HistCache::stale);
    if stale {
        let mut c = app.hist_cache.take().unwrap_or_default();
        let organisms = app.sim.organisms().to_vec();
        c.rebuild(&organisms);
        app.hist_cache = Some(c);
    }

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            let Some(cache) = &app.hist_cache else {
                widgets::weak_label(ui, "Building snapshot…");
                return;
            };
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("Evolution"));
                ui.label(widgets::weak("Species census and trait distributions"));
                ui.add_space(8.0);

                // ---------------- species table
                widgets::section_header(ui, "Species");
                if cache.species.is_empty() {
                    widgets::empty_state(
                        ui,
                        "No organisms",
                        "The ecosystem is empty — spawn founders from the world view",
                        None,
                    );
                } else {
                    egui::Grid::new("alife.species.grid")
                        .num_columns(5)
                        .spacing([16.0, 3.0])
                        .show(ui, |ui| {
                            ui.label(widgets::weak("color"));
                            ui.label(widgets::weak("species id"));
                            ui.label(widgets::weak("count"));
                            ui.label(widgets::weak("share"));
                            ui.label(widgets::weak("avg generation"));
                            ui.end_row();
                            let total = cache.species.iter().map(|r| r.count).sum::<u32>().max(1);
                            for r in &cache.species {
                                widgets::chip(ui, species_color(r.id));
                                ui.label(widgets::mono(format!("{}", r.id), theme::TEXT));
                                ui.label(widgets::mono(format!("{}", r.count), theme::TEXT));
                                ui.label(widgets::mono(
                                    format!("{:.1}%", r.count as f32 / total as f32 * 100.0),
                                    theme::TEXT,
                                ));
                                ui.label(widgets::mono(
                                    format!("{:.1}", r.avg_generation),
                                    theme::TEXT,
                                ));
                                ui.end_row();
                            }
                        });
                }
                ui.add_space(12.0);

                // ---------------- trait histograms
                widgets::section_header(ui, "Trait distribution — 16 bins");
                histogram(
                    ui,
                    &cache.bins_speed,
                    TRAIT_MIN.max_speed,
                    TRAIT_MAX.max_speed,
                    "max_speed",
                    theme::ACCENT,
                );
                histogram(
                    ui,
                    &cache.bins_efficiency,
                    TRAIT_MIN.efficiency,
                    TRAIT_MAX.efficiency,
                    "efficiency",
                    theme::WARN,
                );
                histogram(
                    ui,
                    &cache.bins_sensor,
                    TRAIT_MIN.sensor_range,
                    TRAIT_MAX.sensor_range,
                    "sensor_range",
                    theme::TEAL,
                );
                ui.add_space(20.0);
            });
        });
}

/// Painted bar chart of one trait's distribution.
fn histogram(ui: &mut egui::Ui, bins: &[u32; 16], min: f32, max: f32, label: &str, color: Color32) {
    ui.horizontal(|ui| {
        ui.add_sized(
            egui::Vec2::new(96.0, 12.0),
            egui::Label::new(widgets::weak(label)),
        );
        ui.add_sized(
            egui::Vec2::new(160.0, 12.0),
            egui::Label::new(widgets::mono(
                format!("{:.1} .. {:.1}", min, max),
                theme::WEAK,
            )),
        );
    });
    let height = 74.0;
    let (rect, resp) = ui.allocate_exact_size(
        egui::Vec2::new(ui.available_width().max(60.0), height),
        egui::Sense::hover(),
    );
    let p = ui.painter();
    p.rect_filled(rect, 3.0, Color32::from_rgb(12, 16, 21));

    let n = bins.len();
    let slot = rect.width() / n as f32;
    let max_v = bins.iter().copied().max().unwrap_or(0).max(1);
    for (i, v) in bins.iter().enumerate() {
        let w = (slot - 2.0).max(2.0);
        let x = rect.min.x + i as f32 * slot + 1.0;
        let h = (*v as f32 / max_v as f32) * (height - 4.0);
        if h > 0.5 {
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(x, rect.max.y - h),
                    egui::Vec2::new(w, h),
                ),
                1.5,
                color,
            );
        }
    }
    p.text(
        rect.left_bottom() + egui::vec2(2.0, -12.0),
        egui::Align2::LEFT_BOTTOM,
        min.to_string(),
        egui::FontId::monospace(9.0),
        theme::WEAK,
    );
    p.text(
        rect.right_bottom() + egui::vec2(-2.0, -12.0),
        egui::Align2::RIGHT_BOTTOM,
        max.to_string(),
        egui::FontId::monospace(9.0),
        theme::WEAK,
    );
    p.text(
        rect.center_top() + egui::vec2(0.0, 2.0),
        egui::Align2::CENTER_TOP,
        format!("peak {}/bin", max_v),
        egui::FontId::monospace(9.0),
        theme::WEAK,
    );

    // hover readout for a single bin
    if resp.hovered() {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            let idx = (((pos.x - rect.min.x) / slot).clamp(0.0, 15.99)) as usize;
            resp.on_hover_text(format!("bin {}: {} organism(s)", idx, bins[idx]));
        }
    }
}
