//! Evolutionary tree view: species ancestor chains reconstructed from
//! `LineageRecord` mom/dad walks for the current top-10 species.
//!
//! Everything is derived from real engine records; anything the engine no
//! longer retains is reported honestly as "record expired".

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets;
use alife_core::{SimApi, Simulation};
use alife_render::paint::species_color;
use egui::Context;

/// Chain walk bound per sampled lineage (spec asks: up to 12 nodes/species).
const WALK_DEPTH: usize = 12;
/// Organisms sampled per species to reconstruct its ancestry.
const SAMPLES_PER_SPECIES: usize = 3;

// -------------------------------------------------------------------- cache

/// One node of an ancestor chain.
pub struct ChainNode {
    pub rec: alife_common::LineageRecord,
    pub alive: bool,
    /// this organism was born in a different species than its parent
    pub speciation: bool,
}

/// One sampled ancestor chain, tip (youngest) first.
pub struct Chain {
    pub tip: u64,
    pub nodes: Vec<ChainNode>,
    /// ancestors beyond the engine's retained record window
    pub expired: u32,
}

/// The tree for one species.
pub struct SpeciesTree {
    pub species_id: u32,
    pub count: u32,
    pub share: f32,
    pub chains: Vec<Chain>,
}

/// Throttled tree snapshot: rebuilt only when its change key moved — the
/// sim step bucket (60-step bins ≈ 1 sim-second) plus the population count.
/// A paused sim therefore never triggers a rebuild.
#[derive(Default)]
pub struct TreeCache {
    /// last rebuild identity: (step bucket, population)
    pub key: Option<(u64, u32)>,
    pub alive_total: u32,
    pub trees: Vec<SpeciesTree>,
    pub expired_total: u32,
}

impl TreeCache {
    /// Change-detection key for the current engine state.
    fn identity(sim: &Simulation) -> (u64, u32) {
        (
            sim.step_count() / 60,
            sim.stats().population,
        )
    }

    fn stale(&self, key: (u64, u32)) -> bool {
        self.key != Some(key)
    }

    fn rebuild(&mut self, sim: &Simulation) {
        self.key = Some(Self::identity(sim));
        let stats = sim.stats();
        let total = stats.population.max(1);
        self.alive_total = stats.population;
        self.expired_total = 0;
        self.trees.clear();

        for sc in stats.species_counts.iter().take(10) {
            let tips: Vec<u64> = {
                let mut v = Vec::new();
                for o in sim.organisms() {
                    if o.alive && o.lineage == sc.species_id {
                        v.push(o.id);
                        if v.len() >= SAMPLES_PER_SPECIES {
                            break;
                        }
                    }
                }
                v
            };
            let mut chains = Vec::new();
            for tip in tips {
                let (chain, expired) = walk_chain(sim, tip);
                self.expired_total += expired;
                chains.push(chain);
            }
            self.trees.push(SpeciesTree {
                species_id: sc.species_id,
                count: sc.count,
                share: sc.count as f32 / total as f32,
                chains,
            });
        }
    }
}

/// Walk `mom` (fall back to `dad`) records from a tip toward its founder.
fn walk_chain(sim: &Simulation, tip: u64) -> (Chain, u32) {
    let mut nodes: Vec<ChainNode> = Vec::new();
    let mut expired = 0u32;
    let mut cur = tip;
    let mut prev_species: Option<u32> = None;
    for _ in 0..WALK_DEPTH {
        match sim.get_record(cur) {
            Some(rec) => {
                let next = if rec.mom.is_some() { rec.mom } else { rec.dad };
                let speciation = match prev_species {
                    Some(p) => rec.species_id != p,
                    None => false,
                };
                prev_species = Some(rec.species_id);
                nodes.push(ChainNode {
                    rec: rec.clone(),
                    alive: rec.death_step.is_none(),
                    speciation,
                });
                match next {
                    Some(n) if n != cur => cur = n,
                    _ => break, // founder reached (no parent in the record)
                }
            }
            None => {
                expired += 1;
                break;
            }
        }
    }
    (Chain { tip, nodes, expired }, expired)
}

// --------------------------------------------------------------------- view

pub fn tree_view(app: &mut AlifeApp, ctx: &Context) {
    // change-detection rebuild: only when (60-step bucket, population) moved
    let key = TreeCache::identity(&app.sim);
    let stale = app.tree_cache.as_ref().map_or(true, |c| c.stale(key));
    if stale {
        let mut c = app.tree_cache.take().unwrap_or_default();
        c.rebuild(&app.sim);
        app.tree_cache = Some(c);
    }

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(widgets::strong("Evolutionary tree"));
                ui.label(widgets::weak(
                    "Ancestor chains for the current top-10 species, reconstructed from the \
                     engine's lineage records (mom/dad walk, bounded to 12 nodes per species; 3 \
                     sampled lineages each).",
                ));
                let Some(cache) = &app.tree_cache else { return };

                if cache.expired_total > 0 {
                    ui.label(widgets::mono(
                        format!(
                            "{} ancestor record(s) expired — beyond the engine's retained window",
                            cache.expired_total
                        ),
                        theme::WARN,
                    ))
                    .on_hover_text(
                        "The engine keeps a bounded lineage record map; only already-dead old \
                         records are evicted. Expired ancestors cannot be walked further.",
                    );
                }

                if cache.trees.is_empty() {
                    widgets::empty_state(
                        ui,
                        "No species live yet",
                        "The tree fills as soon as organisms exist; run or reseed the world",
                        None,
                    );
                    return;
                }

                for t in &cache.trees {
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        widgets::chip(ui, species_color(t.species_id));
                        ui.label(widgets::strong(format!("species #{}", t.species_id)));
                        ui.label(widgets::mono(
                            format!("{} alive ({:.1}% share)", t.count, t.share * 100.0),
                            theme::TEXT,
                        ));
                    });
                    if t.chains.is_empty() {
                        widgets::weak_label(ui, "no ancestor records for this species (all lineages faded)");
                        continue;
                    }
                    for chain in &t.chains {
                        ui.add_space(2.0);
                        egui::CollapsingHeader::new(
                            egui::RichText::new(format!("lineage from #{}", chain.tip))
                                .size(11.0)
                                .color(theme::WEAK),
                        )
                        .id_salt(format!("alife.tree.chain.{}.{}", t.species_id, chain.tip))
                        .default_open(t.chains.len() == 1)
                        .show(ui, |ui| chain_rows(ui, chain));
                    }
                }
                ui.add_space(20.0);
            });
        });
}

fn chain_rows(ui: &mut egui::Ui, chain: &Chain) {
    egui::Grid::new(format!("alife.tree.nodes.{}", chain.tip))
        .num_columns(4)
        .spacing([10.0, 2.0])
        .show(ui, |ui| {
            for (depth, node) in chain.nodes.iter().enumerate() {
                // indent chip: role (speciation / founder / ancestor)
                let role = if node.speciation {
                    "new species"
                } else if node.rec.mom.is_none() && node.rec.dad.is_none() {
                    "founder"
                } else {
                    "ancestor"
                };
                widgets::chip(ui, species_color(node.rec.species_id));
                ui.label(widgets::mono(
                    format!("#{}", node.rec.id),
                    theme::TEXT,
                ))
                .on_hover_text(format!("generation {}", node.rec.generation));
                ui.label(widgets::mono(
                    format!("d{} {}", depth, role),
                    if node.speciation { theme::ACCENT } else { theme::WEAK },
                ))
                .on_hover_text(format!("dump/speciation: born in species {}", node.rec.species_id));
                ui.horizontal(|ui| {
                    ui.label(widgets::mono(
                        format!("born step {}", node.rec.born_step),
                        theme::TEXT,
                    ));
                    if node.alive {
                        ui.label(
                            egui::RichText::new("alive").small().strong().color(theme::OK),
                        );
                    } else {
                        ui.label(widgets::mono(
                            format!(
                                "died step {}",
                                node.rec.death_step.unwrap_or(0)
                            ),
                            theme::WEAK,
                        ));
                    }
                });
                ui.end_row();
            }
        });
    if chain.expired > 0 {
        ui.label(widgets::mono(
            format!("… {} ancestor record(s) not yet available (record expired)", chain.expired),
            theme::WARN,
        ));
    }
}
