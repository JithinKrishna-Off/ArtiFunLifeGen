//! God Mode glue between the UI and the engine's intervention / weather /
//! diagnosis capability.
//!
//! Everything routes through the local [`GodActions`] trait implemented for
//! `Simulation`. Before the T1 engine contract (`Simulation::trigger`,
//! `interventions()`, `weather()` sampling, `protect_lineage()`, `diagnosis()`)
//! lands, the impl below returns honest placeholder results:
//! triggers are `Err("god actions: pending engine integration ...")` and
//! queries are empty / `None`, which the UI renders as "not yet available".
//! AT INTEGRATION: the coordinator replaces ONLY the `impl GodActions for
//! Simulation` bodies below — no other call site in the UI changes.
//!
//! Tool + log data types consumed by `AlifeApp` also live here so the app
//! state stays independent of panel implementation details.

use alife_common::{DeathCause, GodEvent, InterventionRecord};
use alife_core::{weather::EffectKind, SimApi, Simulation};

use std::time::Instant;

/// Throttled snapshot of the engine-side god log (see panels_god).
#[derive(Default)]
pub struct GodCache {
    pub built: Option<Instant>,
    pub interventions: Vec<InterventionRecord>,
}

/// Weather layer snapshot for the HUD and previews.
#[derive(Clone, Copy, Debug, Default)]
pub struct WeatherSnapshot {
    /// cloud coverage fraction in [0, 1]
    pub cloud: f32,
    /// rainfall intensity fraction in [0, 1]
    pub rain: f32,
    /// storm banner active (drives the thunder icon)
    pub storm: bool,
    /// lightning bolts fired in the recent window
    pub recent_bolts: u32,
    /// seconds since the most recent flash (None: none observed)
    pub flash_age: Option<f32>,
}

impl WeatherSnapshot {
    /// True when weather contributes nothing visible; the HUD hides then.
    pub fn idle(&self) -> bool {
        !self.storm && self.cloud < 0.02 && self.rain < 0.02 && self.recent_bolts == 0
    }
}

/// Death-cause breakdown from the engine diagnosis layer.
/// Constructed by the engine impl once the diagnosis contract lands; the UI
/// reads it through `Option<Diagnosis>` either way.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Diagnosis {
    pub starvation: u64,
    pub old_age: u64,
    pub hazard: u64,
}

/// Read/write access to the god-mode engine capability.
pub trait GodActions {
    // ---- mutating ----
    /// Apply one intervention. Returns a short engine note on success
    /// ("+26 food", "wetting for 45 s") or the failure reason.
    fn god_trigger(&mut self, ev: &GodEvent) -> Result<String, String>;
    /// Toggle lineage protection (immune to weather / purge effects).
    fn god_protect_lineage(&mut self, species: u32, on: bool) -> Result<(), String>;

    // ---- queries ----
    /// Recorded interventions in application order (engine-side log).
    fn god_interventions(&self) -> Vec<InterventionRecord>;
    /// Current weather layer snapshot ("not yet available" = None pre-T1).
    fn god_weather(&self) -> Option<WeatherSnapshot>;
    /// Wetness sampling at a world position (brush/preview support).
    fn god_wet_at(&self, x: f32, y: f32) -> Option<f32>;
    /// Whether a lineage is protected.
    fn god_protected(&self, species: u32) -> bool;
    /// Death-cause diagnosis ("not yet available" = None pre-B2).
    fn god_diagnosis(&self) -> Option<Diagnosis>;
}

// Free-standing dispatchers so UI code never needs `GodActions` imported.

pub fn trigger(sim: &mut Simulation, ev: &GodEvent) -> Result<String, String> {
    sim.god_trigger(ev)
}

pub fn protect_lineage(sim: &mut Simulation, species: u32, on: bool) -> Result<(), String> {
    sim.god_protect_lineage(species, on)
}

pub fn interventions(sim: &Simulation) -> Vec<InterventionRecord> {
    sim.god_interventions()
}

pub fn weather(sim: &Simulation) -> Option<WeatherSnapshot> {
    sim.god_weather()
}

pub fn wet_at(sim: &Simulation, x: f32, y: f32) -> Option<f32> {
    sim.god_wet_at(x, y)
}

pub fn protected(sim: &Simulation, species: u32) -> bool {
    sim.god_protected(species)
}

pub fn diagnosis(sim: &Simulation) -> Option<Diagnosis> {
    sim.god_diagnosis()
}

// Template matching for the armed crosshair tool (works pre-integration).

/// Which god act is armed for the next click on the world canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GodTool {
    Rain,
    Drought,
    Storm,
    Lightning,
    NutrientDump,
    Summon,
}

impl GodTool {
    pub fn label(self) -> &'static str {
        match self {
            GodTool::Rain => "Rain",
            GodTool::Drought => "Drought",
            GodTool::Storm => "Storm",
            GodTool::Lightning => "Lightning",
            GodTool::NutrientDump => "Nutrient dump",
            GodTool::Summon => "Summon nucleator",
        }
    }

    /// Effect sentence for chips and confirm dialogs.
    pub fn effect(self, r: f32) -> String {
        let secs = RAIN_DURATION_STEPS as f32 / 60.0;
        match self {
            GodTool::Rain => format!("rain r={r:.0} at target — soil wetting for {secs:.0} s"),
            GodTool::Drought => format!("drought r={r:.0} — moisture/fertility penalty for {secs:.0} s"),
            GodTool::Storm => format!(
                "storm r={r:.0} — cloud + rain for {secs:.0} s with {} bolt strikes",
                STORM_BOLTS
            ),
            GodTool::Lightning => "single bolt at the point — scar + fertility flash".into(),
            GodTool::NutrientDump => format!(
                "scatter {DUMP_ENERGY:.0} energy of nutrients inside r={r:.0}"
            ),
            GodTool::Summon => "spawn one fresh founder organism at the point".into(),
        }
    }
}

/// Shared god event parameters (one slider warms the whole panel).
pub const RAIN_DURATION_STEPS: u32 = 2700; // 45 sim-seconds
pub const STORM_BOLTS: u32 = 12;
pub const DUMP_ENERGY: f32 = 3000.0;

/// Build the concrete `GodEvent` for an armed tool at a target.
pub fn build_event(tool: GodTool, x: f32, y: f32, radius: f32) -> GodEvent {
    match tool {
        GodTool::Rain => GodEvent::Rain { x, y, r: radius, duration_steps: RAIN_DURATION_STEPS },
        GodTool::Drought => GodEvent::Drought { x, y, r: radius, duration_steps: RAIN_DURATION_STEPS },
        GodTool::Storm => GodEvent::Storm {
            x,
            y,
            r: radius,
            duration_steps: RAIN_DURATION_STEPS,
            lightning_bolts: STORM_BOLTS,
        },
        GodTool::Lightning => GodEvent::Lightning { x, y },
        GodTool::NutrientDump => GodEvent::NutrientDump { x, y, r: radius, energy: DUMP_ENERGY },
        GodTool::Summon => GodEvent::SummonOrganism { x, y },
    }
}

/// One locally logged god attempt (UI-side mirror; the engine keeps its own
/// authoritative [`InterventionRecord`] log once the T1 contract lands).
#[derive(Clone, Debug)]
pub struct GodLogEntry {
    pub step: u64,
    /// event label ("Rain", "Purge lineage", "Catastrophe")
    pub label: String,
    /// engine note, or the pending-integration explanation
    pub note: String,
    pub succeeded: bool,
}

// --------------------------------------------------------------------------
// The single integration seam. Replace the stub bodies when the engine
// contract lands; signatures and callers stay untouched.
// --------------------------------------------------------------------------

impl GodActions for Simulation {
    fn god_trigger(&mut self, ev: &GodEvent) -> Result<String, String> {
        self.trigger(ev).map(|_| ev.label().to_string())
    }

    fn god_protect_lineage(&mut self, species: u32, on: bool) -> Result<(), String> {
        if self.protect_lineage(species, on) { Ok(()) } else {
            Err(format!("no lineage with species id {species}"))
        }
    }

    fn god_interventions(&self) -> Vec<InterventionRecord> {
        self.interventions().to_vec()
    }

    fn god_weather(&self) -> Option<WeatherSnapshot> {
        let w = SimApi::weather(self);
        // an effect is removed from the list by update() when it expires, so
        // presence == active
        let storm = w.effects.iter().any(|e| matches!(e.kind, EffectKind::Storm | EffectKind::Fire));
        let rain = {
            let n = w.rain.len();
            if n == 0 { 0.0 } else {
                let mut wet_bins = 0usize;
                for &r in &w.rain {
                    if r > 0.01 { wet_bins += 1; }
                }
                wet_bins as f32 / n as f32
            }
        };
        let flash_age = if w.flash > 0.005 {
            // FLASH_DECAY is 2.5/s in the engine: age = ln(1/intensity)/2.5
            Some(((-w.flash.max(1e-3).ln()) / 2.5).max(0.0))
        } else {
            None
        };
        Some(WeatherSnapshot {
            cloud: w.cover,
            rain,
            storm,
            recent_bolts: w.recent_bolts().len() as u32,
            flash_age,
        })
    }

    fn god_wet_at(&self, x: f32, y: f32) -> Option<f32> {
        Some(SimApi::weather(self).wet(x, y))
    }

    fn god_protected(&self, species: u32) -> bool {
        self.protected_lines().contains(&species)
    }

    fn god_diagnosis(&self) -> Option<Diagnosis> {
        let mut starvation = 0u64;
        let mut old_age = 0u64;
        let mut hazard = 0u64;
        for (cause, n) in self.recent_deaths_breakdown() {
            match cause {
                DeathCause::Starvation => starvation = n as u64,
                DeathCause::OldAge => old_age = n as u64,
                DeathCause::Hazard => hazard = n as u64,
            }
        }
        Some(Diagnosis { starvation, old_age, hazard })
    }
}
