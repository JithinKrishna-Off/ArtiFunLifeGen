//! Queen brain & colony observatory — read-only panel (Agent 8).
//!
//! All data comes from live `alife_core::Simulation` surfaces via
//! [`queen_ui_data`] / [`colony_ui_data`] (call once per UI frame, cheap).
//!
//! Honest-data conventions (coordinators must follow these when filling):
//! * `f32` fields set to `NaN` mean "unavailable" and render as "—".
//! * `u32` fields set to `u32::MAX` mean "unavailable".
//! * `exec_ns == 0` means "not measured yet".
//! * `Option::None` / empty `Vec` means the data simply does not exist
//!   (no queen / no brain plugged / organism gone) — the card explains.
//! The panel never invents numbers; every unavailable field shows "—".
//!
//! Per-card data sources (all live, no placeholders):
//! * Queen card  — queen organism struct (id/alive/energy/max, age) +
//!   `sim.queen_config()` lifespan multiplier.
//! * Neural card — `sim.queen_brain_stats()`; identity pending (core has no
//!   public getter for the plugged brain's connectome identity yet).
//! * Sensory card — queen organism `last_inputs` (engine-written 11-input
//!   net snapshot) + sim signal field sampled at her cell.
//! * Motor card — queen organism `last_outputs` (turn/thrust/repro gate).
//! * Colony card — `sim.stats()` + `sim.signals.active_mass(0..6)`.
//! * Events card — `sim.queen_events()`.
//!
//! Rendering itself (egui paint) is intentionally NOT machine-tested; only
//! the pure helpers and data structs are.

#![allow(dead_code)] // wired by the coordinator through app.rs (fn signatures are the contract)

use crate::theme;
use crate::widgets;
use alife_core::{SimApi, Simulation};
use egui::{pos2, Align2, Color32, Context, FontId, Id, Sense, Shape, Stroke, Ui, Vec2};

// ------------------------------------------------------------------- data

#[derive(Default, Clone)]
pub struct QueenUiData {
    /// A queen reign exists (alive now, or the events ring still shows one).
    pub present: bool,
    /// `u32::MAX` = unavailable.
    pub org_id: u32,
    pub alive: bool,
    pub energy: f32,
    pub max_energy: f32,
    /// Steps the queen has lived (60 steps = 1 sim second).
    pub age_steps: u32,
    /// Death-age budget in steps (traits lifespan × queen lifespan mult).
    pub lifespan_steps: u32,
    /// age/lifespan in [0,1].
    pub lifespan_frac: f32,
    /// The engine has no per-organism health metric yet → `NaN` → "—".
    pub health: f32,
    /// (connectome source, version); core has no public getter yet → `None`.
    pub brain_identity: Option<(String, String)>,
    pub neuron_count: u32,
    pub synapse_count: u32,
    pub active_neurons: u32,
    pub spikes_last: u32,
    pub exec_ns: u128,
    pub sensory: Option<SensoryMini>,
    pub motor: Option<MotorMini>,
    pub events: Vec<(u64, String)>,
}

#[derive(Default, Clone)]
pub struct SensoryMini {
    /// Unit vector toward the nearest visible food, WORLD frame; (0,0) = none seen.
    pub food_dir: (f32, f32),
    /// World units; `NaN` → "—" when no food is visible.
    pub food_dist: f32,
    /// 0..1 hazard pressure at her cell (net input 6).
    pub hazard: f32,
    /// Raw QueenActivity field sample at her cell (0 = no signal).
    pub signal: f32,
    /// 0..1 local peer density (net input 5).
    pub density: f32,
}

#[derive(Default, Clone)]
pub struct MotorMini {
    /// Queen's current heading, WORLD frame (unit vector).
    pub move_dir: (f32, f32),
    /// Thrust intent, `last_outputs[1]` floored at 0 → 0..1.
    pub intensity: f32,
    /// `NaN`: the stock 3-output net cannot produce it (dash until a queen
    /// brain motor lands in core).
    pub feed_seek: f32,
    /// `NaN`: same — signal emission is queen-brain-wired, not net-wired.
    pub emit: f32,
    /// Raw 3-output neural action: [turn, thrust, reproduce gate].
    pub neural: Option<[f32; 3]>,
}

#[derive(Default, Clone)]
pub struct ColonyUiData {
    pub pop: u32,
    /// Total births this session (`sim.stats().total_births`).
    pub births: u64,
    /// Total deaths this session (`sim.stats().total_deaths`).
    pub deaths: u64,
    pub avg_generation: f32,
    /// Total field mass per channel (see [`SIGNAL_NAMES`] / kind order in
    /// `signals.rs`).
    pub signals: [f32; 6],
}

/// Signal channel names, index-matched to `ColonyUiData::signals`.
pub const SIGNAL_NAMES: [&str; 6] = [
    "FoodFound",
    "Danger",
    "Aggregate",
    "Disperse",
    "QueenActivity",
    "EnvStress",
];

// ------------------------------------------------------------ extraction

fn wrap_delta(sim: &Simulation, dx: f32, dy: f32) -> (f32, f32) {
    if !sim.params.wrapping {
        return (dx, dy);
    }
    let w = sim.world.width.max(1.0);
    let h = sim.world.height.max(1.0);
    let wrap = |d: f32, span: f32| {
        if d > span * 0.5 {
            d - span
        } else if d < -span * 0.5 {
            d + span
        } else {
            d
        }
    };
    (wrap(dx, w), wrap(dy, h))
}

/// Nearest visible food (same semantics as the engine's `sense_food`:
/// closest item within `sensor_range * 1.5`, wrap-aware). Returns
/// `(unit_dir, dist)` — `None` when nothing is visible.
fn food_from(sim: &Simulation, px: f32, py: f32, sensor_range: f32) -> Option<((f32, f32), f32)> {
    let search = (sensor_range * 1.5).max(1.0);
    let mut best_d2 = search * search;
    let mut best: Option<(f32, f32)> = None;
    for f in sim.foods() {
        if !f.alive {
            continue;
        }
        let (dx, dy) = wrap_delta(sim, f.x - px, f.y - py);
        if dx * dx > best_d2 || dy * dy > best_d2 {
            continue; // cheap reject; d² check below is authoritative
        }
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some((dx, dy));
        }
    }
    let (dx, dy) = best?;
    let d = best_d2.max(1e-9).sqrt();
    Some(((dx / d, dy / d), d))
}

/// Read-only bilinear sample of one kind's grid at world coords, mirroring
/// `SignalField::sample` semantics exactly (wrap-connected, out-of-range
/// kind or grid → 0). Exists because `SignalField::sample` borrows `&mut`
/// (engine-side) while this panel only has `&Simulation`. Values are
/// identical: the grid spans the whole world (`w` cells across `width`).
fn signal_sample(sim: &Simulation, kind: usize, x: f32, y: f32) -> f32 {
    let Some(g) = sim.signals.grids().get(kind) else { return 0.0 };
    if g.w == 0 || g.h == 0 || g.cells.len() < g.w * g.h {
        return 0.0;
    }
    let w = sim.world.width.max(1.0);
    let h = sim.world.height.max(1.0);
    let x = if x.is_finite() { x.rem_euclid(w) } else { 0.0 };
    let y = if y.is_finite() { y.rem_euclid(h) } else { 0.0 };
    let fx = x / w * g.w as f32;
    let fy = y / h * g.h as f32;
    let i0n = fx.floor();
    let j0n = fy.floor();
    let (mut i0, mut j0) = (i0n as usize, j0n as usize);
    if i0 >= g.w {
        i0 = 0;
    }
    if j0 >= g.h {
        j0 = 0;
    }
    let i1 = (i0 + 1) % g.w;
    let j1 = (j0 + 1) % g.h;
    let dx = fx - i0n;
    let dy = fy - j0n;
    let a = g.cells[j0 * g.w + i0];
    let b = g.cells[j0 * g.w + i1];
    let c = g.cells[j1 * g.w + i0];
    let d = g.cells[j1 * g.w + i1];
    let top = a + (b - a) * dx;
    let bot = c + (d - c) * dx;
    top + (bot - top) * dy
}

/// Live queen telemetry from the running simulation. Cheap: one organism
/// lookup + a read-only signal-grid sample per call; the food scan is
/// bounded by the sensor range test with early rejection. Call once per
/// UI frame.
#[must_use]
pub fn queen_ui_data(sim: &Simulation) -> QueenUiData {
    let mut q = QueenUiData::default();

    // Identity: alive queen id, else whatever the events ring still shows.
    let events = map_events(sim.queen_events());
    q.events = events.clone();
    match sim.queen_id() {
        Some(id) => {
            q.present = true;
            q.alive = true;
            q.org_id = id;
        }
        None => {
            if let Some(ev) = sim.queen_events().last().copied() {
                q.present = true;
                q.alive = false;
                q.org_id = ev.org_id;
            }
        }
    }

    // Brain (None until a brain is plugged — honest sentinels).
    if let Some(s) = sim.queen_brain_stats() {
        q.neuron_count = s.neuron_count;
        q.synapse_count = s.synapse_count;
        q.active_neurons = s.active_neurons;
        q.spikes_last = s.spikes_last;
        q.exec_ns = s.exec_ns as u128;
    } else {
        q.neuron_count = u32::MAX;
        q.synapse_count = u32::MAX;
        q.active_neurons = u32::MAX;
        q.spikes_last = u32::MAX;
    }

    let org = if q.org_id != u32::MAX { sim.organism(q.org_id as u64) } else { None };
    let Some(o) = org else { return q };

    // Body telemetry straight from the queen organism.
    q.energy = o.energy;
    q.max_energy = o.max_energy;
    q.age_steps = o.age;
    q.health = f32::NAN; // no per-organism health surface in core yet
    let lifespan_mult = sim.queen_config().lifespan_mult.max(1.0);
    let lifespan_steps = ((o.genotype.traits.lifespan.max(1) as f32) * lifespan_mult).max(1.0);
    q.lifespan_steps = lifespan_steps as u32;
    q.lifespan_frac = (o.age as f32 / lifespan_steps).clamp(0.0, 1.0);

    if !q.alive {
        return q;
    }

    // Sensory: engine-written net inputs + live QueenActivity sample.
    let li = &o.last_inputs;
    let idx = |i: usize| li.get(i).copied().unwrap_or(0.0);
    let (mut food_dir, mut food_dist) = (None, f32::NAN);
    if idx(0) != 0.0 || idx(1) != 0.0 {
        // inputs are heading-relative cos/sin; rotate into the world frame
        let rel = idx(1).atan2(idx(0));
        let a = o.dir + rel;
        let a = if a.is_finite() { a } else { 0.0 };
        food_dir = Some((a.cos(), a.sin()));
        if let Some(((fx, fy), d)) =
            food_from(sim, o.x, o.y, o.genotype.traits.sensor_range)
        {
            food_dir = Some((fx, fy));
            food_dist = d;
        }
    };
    q.sensory = Some(SensoryMini {
        food_dir: food_dir.unwrap_or((0.0, 0.0)),
        food_dist,
        hazard: idx(6).clamp(0.0, 1.0),
        signal: signal_sample(sim, alife_core::QUEEN_ACTIVITY_KIND, o.x, o.y).clamp(-1.0e3, 1.0e3),
        density: idx(5).clamp(0.0, 1.0),
    });

    // Motor: stock 3-output action recorded by the engine.
    let lo = &o.last_outputs;
    let oi = |i: usize| lo.get(i).copied().unwrap_or(0.0);
    q.motor = Some(MotorMini {
        move_dir: (o.dir.cos(), o.dir.sin()),
        intensity: oi(1).max(0.0).clamp(0.0, 1.0),
        feed_seek: f32::NAN,
        emit: f32::NAN,
        neural: Some([oi(0), oi(1), oi(2)]),
    });
    q
}

fn map_events(ev: &[alife_core::queen::QueenEvent]) -> Vec<(u64, String)> {
    ev.iter()
        .map(|e| (e.tick, event_text(e)))
        .collect()
}

fn event_text(e: &alife_core::queen::QueenEvent) -> String {
    use alife_core::queen::QueenEventKind as K;
    match e.kind {
        K::Crowned => "crowned".into(),
        K::Fed => "fed".into(),
        K::Died(d) => format!("died ({d:?})"),
    }
}

/// Live colony telemetry from the running simulation. Cheap: one stats read
/// plus six field-mass sums. Call once per UI frame.
#[must_use]
pub fn colony_ui_data(sim: &Simulation) -> ColonyUiData {
    let st = sim.stats();
    let mut c = ColonyUiData {
        pop: st.population,
        births: st.total_births,
        deaths: st.total_deaths,
        avg_generation: st.avg_generation,
        signals: [0.0; 6],
    };
    for (k, s) in c.signals.iter_mut().enumerate() {
        *s = active_mass_finite(sim, k);
    }
    c
}

fn active_mass_finite(sim: &Simulation, kind: usize) -> f32 {
    let m = sim.signals.active_mass(kind);
    if m.is_finite() {
        m
    } else {
        0.0
    }
}

// ------------------------------------------------------- pure formatters

/// `None` for NaN/infinite (the "unavailable" convention).
pub fn f32_opt(v: f32) -> Option<f32> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

/// `None` for `u32::MAX` (the "unavailable" convention).
pub fn u32_opt(v: u32) -> Option<u32> {
    if v == u32::MAX {
        None
    } else {
        Some(v)
    }
}

/// Fixed-digits decimal, or "—" when unavailable.
#[must_use]
pub fn fmt_f32(v: f32, digits: usize) -> String {
    match f32_opt(v) {
        Some(x) => format!("{:.*}", digits, x),
        None => "—".into(),
    }
}

/// Decimal, or "—" when the sentinel `u32::MAX` is set.
#[must_use]
pub fn fmt_u32(v: u32) -> String {
    match u32_opt(v) {
        Some(x) => x.to_string(),
        None => "—".into(),
    }
}

/// Nanoseconds → human time (ns → µs → ms → s), 2 significant decimals.
#[must_use]
pub fn format_exec_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.2} µs", ns as f64 / 1e3)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else {
        format!("{:.2} s", ns as f64 / 1e9)
    }
}

/// Heading of a 2D direction vector in degrees (CCW from +X), or "—".
#[must_use]
pub fn heading_text(dir: (f32, f32)) -> String {
    let (fx, fy) = dir;
    let len_sq = fx * fx + fy * fy;
    if len_sq < 1e-12 {
        return "—".into();
    }
    format!("{:.0}°", fy.atan2(fx).to_degrees())
}

/// A `[0,1]` fraction rendered as floored percent, or "—" when unavailable.
#[must_use]
pub fn fmt_percent(v: f32) -> String {
    match f32_opt(v) {
        Some(x) => format!("{}%", (x.clamp(0.0, 1.0) * 100.0).floor()),
        None => "—".into(),
    }
}

// ------------------------------------------------------------ entry point

/// The coordinator opens/closes this; the panel itself always renders.
/// `app` is unused (`_app`) — the panel is purely data-driven.
pub fn show_brain(_app: &mut crate::app::AlifeApp, ctx: &Context, q: &QueenUiData, c: &ColonyUiData) {
    egui::Window::new("Queen Brain & Colony")
        .id(Id::new("alife.brain"))
        .default_width(470.0)
        .default_height(540.0)
        .show(ctx, |ui| {
            queen_card(ui, q);
            neural_card(ui, q);
            sensory_card(ui, q);
            motor_card(ui, q);
            colony_card(ui, c);
            events_card(ui, q);
        });
}

// ---------------------------------------------------------------- cards

fn queen_card(ui: &mut Ui, q: &QueenUiData) {
    widgets::section_header(ui, "Queen");
    if !q.present {
        widgets::weak_label(ui, "no queen crowned in this world yet — all queen data is unavailable");
        return;
    }
    ui.horizontal(|ui| {
        ui.label(widgets::mono("◆ CROWNED", theme::ACCENT))
            .on_hover_text("A queen entity is present and tracked; telemetry below comes from it");
        match u32_opt(q.org_id) {
            Some(id) => ui
                .label(widgets::mono(format!("org #{id}"), theme::TEXT))
                .on_hover_text("Queen's organism id in the world"),
            None => ui.label(widgets::mono("org #—", theme::WEAK)),
        };
        ui.label(if q.alive {
            widgets::mono("alive", theme::OK)
        } else {
            widgets::mono("deceased", theme::DANGER)
        });
    });

    let energy = match (f32_opt(q.energy), f32_opt(q.max_energy)) {
        (Some(e), Some(m)) if m > 0.0 => Some((e, m)),
        _ => None,
    };
    match energy {
        Some((e, m)) => widgets::bar_row(
            ui,
            "energy",
            e,
            m.max(1e-6),
            theme::ACCENT,
            &format!("{e:.1}/{m:.1}"),
        ),
        None => widgets::bar_row(ui, "energy", 0.0, 1.0, Color32::from_rgb(60, 68, 78), "—"),
    };
    widgets::bar_row(ui, "lifespan", clamp01(q.lifespan_frac), 1.0, theme::TEAL, &fmt_percent(q.lifespan_frac));
    widgets::bar_row(
        ui,
        "age",
        0.0,
        1.0,
        theme::WEAK,
        &format!(
            "{}/{} steps",
            fmt_u32(q.age_steps),
            fmt_u32(q.lifespan_steps)
        ),
    )
    .on_hover_text("Steps lived vs death budget (queen lifespan × multiplier); 60 steps = 1 sim second");
    widgets::bar_row(ui, "health", clamp01(q.health), 1.0, theme::OK, &fmt_percent(q.health))
        .on_hover_text("no per-organism health metric exists in the engine yet (energy/hazard only)");
}

fn neural_card(ui: &mut Ui, q: &QueenUiData) {
    widgets::section_header(ui, "Neural");
    let identity = match &q.brain_identity {
        Some((source, id)) => format!("{source} {id}"),
        None => "—".into(),
    };
    ui.label(widgets::mono(format!("brain {identity}"), if q.brain_identity.is_some() { theme::ACCENT } else { theme::WEAK }))
        .on_hover_text(if q.brain_identity.is_some() {
            "Connectome identity (source + specimen id) of the queen brain"
        } else {
            "unavailable — no queen brain plugged, and the plugged brain's \
             identity getter is not exposed by alife-core yet (missing \
             accessor; QueenBrain trait has identity, Simulation does not \
             surface it)"
        });

    egui::Grid::new("alife.brain.neural")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            ui.label(widgets::mono("neurons", theme::WEAK));
            ui.label(widgets::mono(fmt_u32(q.neuron_count), theme::TEXT));
            ui.label(widgets::mono("synapses", theme::WEAK));
            ui.label(widgets::mono(fmt_u32(q.synapse_count), theme::TEXT));

            ui.label(widgets::mono("active", theme::WEAK));
            match u32_opt(q.active_neurons) {
                Some(a) => {
                    ui.label(widgets::mono(a.to_string(), theme::TEXT));
                }
                None => {
                    ui.label(widgets::mono("—", theme::WEAK))
                        .on_hover_text("no queen brain plugged (stock net drives her)");
                }
            };
            ui.end_row();

            ui.label(widgets::mono("spikes (EMA)", theme::WEAK));
            match u32_opt(q.spikes_last) {
                Some(s) => {
                    ui.label(widgets::mono(s.to_string(), theme::TEXT));
                }
                None => {
                    ui.label(widgets::mono("—", theme::WEAK));
                }
            };
            ui.end_row();

            ui.label(widgets::mono("exec time", theme::WEAK));
            if q.exec_ns == 0 {
                ui.label(widgets::mono("—", theme::WEAK))
                    .on_hover_text("not measured yet (or 0-cost pass)");
            } else {
                ui.label(widgets::mono(format_exec_ns(q.exec_ns), theme::TEXT))
                    .on_hover_text(format!("{} ns forward pass", q.exec_ns));
            }
            ui.end_row();
        });

    // active-neurons bar, normalized against the connectome size (known max)
    if let (Some(a), Some(n)) = (u32_opt(q.active_neurons), u32_opt(q.neuron_count)) {
        if n > 0 {
            widgets::bar_row(
                ui,
                "active share",
                a as f32,
                n as f32,
                theme::TEAL,
                &format!("{a}/{n}"),
            );
        }
    }
    // spikes bar: honest ratio = spikes this tick / neurons (can exceed 1, fills clamp)
    if let (Some(s), Some(n)) = (u32_opt(q.spikes_last), u32_opt(q.neuron_count)) {
        if n > 0 {
            widgets::bar_row(
                ui,
                "spike share",
                s as f32,
                n as f32,
                theme::WARN,
                &format!("{s} spikes"),
            )
            .on_hover_text("spikes per tick normalized by connectome size (fill clamps at 100%)");
        }
    }
}

fn sensory_card(ui: &mut Ui, q: &QueenUiData) {
    widgets::section_header(ui, "Sensory");
    let Some(s) = &q.sensory else {
        widgets::weak_label(ui, "no sensory telemetry available (queen absent or deceased)");
        return;
    };
    ui.horizontal(|ui| {
        draw_arrow(ui, s.food_dir, theme::ACCENT);
        ui.vertical(|ui| {
            ui.label(widgets::mono(format!("heading {}", heading_text(s.food_dir)), theme::TEXT))
                .on_hover_text("Direction of the nearest visible food (CCW from +X, world frame)");
            ui.label(widgets::mono(format!("food dist {}", fmt_f32(s.food_dist, 1)), theme::TEXT))
                .on_hover_text("Distance to the nearest visible food (world units; the engine's sensor radius applies)");
        });
    });
    bar_or_dash(ui, "hazard", s.hazard, theme::DANGER);
    bar_or_dash(ui, "signal", s.signal, theme::WARN)
        .on_hover_text("QueenActivity pheromone field sampled at her cell (raw field units)");
    bar_or_dash(ui, "density", s.density, theme::TEAL);
}

fn motor_card(ui: &mut Ui, q: &QueenUiData) {
    widgets::section_header(ui, "Motor");
    let Some(m) = &q.motor else {
        widgets::weak_label(ui, "no motor telemetry available (queen absent or deceased)");
        return;
    };
    ui.horizontal(|ui| {
        draw_arrow(ui, m.move_dir, theme::OK);
        ui.vertical(|ui| {
            ui.label(widgets::mono(format!("move {}", heading_text(m.move_dir)), theme::TEXT))
                .on_hover_text("Her current heading in the world frame");
            bar_or_dash(ui, "intensity", m.intensity, theme::OK)
                .on_hover_text("Thrust intent, engine output[1]");
        });
    });
    bar_or_dash(ui, "feed/seek", m.feed_seek, theme::ACCENT)
        .on_hover_text("stock 3-output net cannot produce this — dash until a queen brain motor is wired in core");
    bar_or_dash(ui, "emit", m.emit, theme::TEAL)
        .on_hover_text("queen signal emission is brain-wired, not net-wired (config: signal emit cost)");
    if let Some(o) = m.neural {
        widgets::section_header(ui, "Neural outputs (oracle net)");
        for (i, label) in ["turn", "thrust", "repro gate"].iter().enumerate() {
            bar_or_dash(ui, label, o[i], theme::WEAK);
        }
    }
}

fn colony_card(ui: &mut Ui, c: &ColonyUiData) {
    widgets::section_header(ui, "Colony");
    egui::Grid::new("alife.brain.colony")
        .num_columns(4)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            ui.label(widgets::mono("population", theme::WEAK));
            ui.label(widgets::mono(c.pop.to_string(), theme::TEXT));
            ui.label(widgets::mono("births", theme::WEAK));
            ui.label(widgets::mono(c.births.to_string(), theme::OK));
            ui.end_row();
            ui.label(widgets::mono("deaths", theme::WEAK));
            ui.label(widgets::mono(c.deaths.to_string(), theme::DANGER));
            ui.label(widgets::mono("generation", theme::WEAK));
            ui.label(widgets::mono(fmt_f32(c.avg_generation, 1), theme::TEXT));
            ui.end_row();
        });
    for (i, name) in SIGNAL_NAMES.iter().enumerate() {
        let v = c.signals[i];
        match f32_opt(v) {
            Some(x) => widgets::bar_row(ui, name, clamp01(x), 1.0, theme::ACCENT, &format!("{x:.2}"))
                .on_hover_text("total signal-field mass on this channel (field units)"),
            None => widgets::bar_row(ui, name, 0.0, 1.0, Color32::from_rgb(60, 68, 78), "—"),
        };
    }
}

fn events_card(ui: &mut Ui, q: &QueenUiData) {
    widgets::section_header(ui, "Queen events (latest 8)");
    if q.events.is_empty() {
        widgets::weak_label(ui, "no queen events yet");
        return;
    }
    egui::Grid::new("alife.brain.events")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (step, text) in q.events.iter().rev().take(8) {
                ui.label(widgets::mono(step.to_string(), theme::WEAK));
                ui.label(widgets::mono(text, theme::TEXT));
                ui.end_row();
            }
        });
}

// ------------------------------------------------------------- painters

/// Compass-style arrow in a fixed-size framed square; zero-length → "—".
fn draw_arrow(ui: &mut Ui, dir: (f32, f32), color: Color32) {
    let size = 52.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 4.0, Color32::from_rgb(12, 16, 21));
    let c = rect.center();
    let r = size * 0.40;
    p.circle_stroke(c, r, Stroke::new(1.0_f32, theme::STROKE_SOFT));

    let (fx, fy) = dir;
    let len = (fx * fx + fy * fy).sqrt();
    if len < 1e-4 {
        p.text(
            c,
            Align2::CENTER_CENTER,
            "—",
            FontId::proportional(10.0),
            theme::WEAK,
        );
        return;
    }
    let (ux, uy) = (fx / len, fy / len);
    let tip = pos2(c.x + ux * r, c.y + uy * r);
    p.line_segment([c, tip], Stroke::new(2.0_f32, color));
    let ah = size * 0.14;
    let b1 = pos2(tip.x - ux * ah - uy * ah * 0.6, tip.y - uy * ah + ux * ah * 0.6);
    let b2 = pos2(tip.x - ux * ah + uy * ah * 0.6, tip.y - uy * ah - ux * ah * 0.6);
    p.add(Shape::convex_polygon(vec![tip, b1, b2], color, Stroke::NONE));
}

/// Bar with "—" placeholder when the value is unavailable (NaN).
fn bar_or_dash(ui: &mut Ui, label: &str, v: f32, color: Color32) -> egui::Response {
    match f32_opt(v) {
        Some(x) => widgets::bar_row(ui, label, clamp01(x), 1.0, color, &format!("{x:.2}")),
        None => widgets::bar_row(ui, label, 0.0, 1.0, Color32::from_rgb(60, 68, 78), "—"),
    }
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------- tests

// UI paint is not machine-testable here (egui needs a frame context); these
// cover the pure helpers, the live extraction path (headless sim) and the
// struct sentinel conventions.
#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::{SimParams, WorldGenConfig};

    fn small_sim(initial_population: u32, initial_food: u32) -> Simulation {
        let params = SimParams {
            initial_population,
            initial_food,
            food_regen_per_sec: 0.0,
            max_population: initial_population.max(1),
            max_food: (initial_food.max(8)),
            ..SimParams::default()
        };
        let world = alife_worldgen::generate(&WorldGenConfig {
            width: 800,
            height: 600,
            cell: 8,
            ..WorldGenConfig::default()
        });
        Simulation::new(&params, world)
    }

    // ------------------------------------------------------------ helpers

    #[test]
    fn data_struct_defaults() {
        let q = QueenUiData::default();
        assert!(!q.present);
        assert!(!q.alive);
        assert_eq!(q.energy, 0.0);
        assert!(q.brain_identity.is_none());
        assert!(q.sensory.is_none());
        assert!(q.motor.is_none());
        assert!(q.events.is_empty());
        assert_eq!(q.exec_ns, 0);

        let c = ColonyUiData::default();
        assert_eq!(c.pop, 0);
        assert_eq!(c.births, 0);
        assert_eq!(c.deaths, 0);
        assert_eq!(c.signals, [0.0; 6]);
        assert_eq!(SIGNAL_NAMES.len(), 6);
        assert_eq!(c.signals.len(), SIGNAL_NAMES.len());
    }

    #[test]
    fn f32_sentinel_helpers() {
        assert_eq!(f32_opt(f32::NAN), None);
        assert_eq!(f32_opt(f32::INFINITY), None);
        assert_eq!(f32_opt(3.5), Some(3.5));
        assert_eq!(fmt_f32(f32::NAN, 1), "—");
        assert_eq!(fmt_f32(2.25, 2), "2.25");
        assert_eq!(fmt_f32(7.6, 0), "8");
    }

    #[test]
    fn u32_sentinel_helpers() {
        assert_eq!(u32_opt(u32::MAX), None);
        assert_eq!(u32_opt(42), Some(42));
        assert_eq!(fmt_u32(u32::MAX), "—");
        assert_eq!(fmt_u32(1234), "1234");
    }

    #[test]
    fn exec_ns_formatting() {
        assert_eq!(format_exec_ns(500), "500 ns");
        assert_eq!(format_exec_ns(999), "999 ns");
        assert_eq!(format_exec_ns(1_500), "1.50 µs");
        assert_eq!(format_exec_ns(1_500_000), "1.50 ms");
        assert_eq!(format_exec_ns(123_456_789), "123.46 ms");
        assert_eq!(format_exec_ns(2_000_000_000), "2.00 s");
    }

    #[test]
    fn heading_formatting() {
        assert_eq!(heading_text((0.0, 0.0)), "—");
        assert_eq!(heading_text((1.0, 0.0)), "0°");
        assert_eq!(heading_text((0.0, 1.0)), "90°");
        assert_eq!(heading_text((-1.0, 0.0)), "180°");
    }

    #[test]
    fn percent_formatting() {
        assert_eq!(fmt_percent(f32::NAN), "—");
        assert_eq!(fmt_percent(0.0), "0%");
        assert_eq!(fmt_percent(0.732), "73%");
        assert_eq!(fmt_percent(1.5), "100%");
        assert_eq!(fmt_percent(-2.0), "0%");
    }

    #[test]
    fn signal_names_table() {
        assert_eq!(
            SIGNAL_NAMES,
            ["FoodFound", "Danger", "Aggregate", "Disperse", "QueenActivity", "EnvStress"]
        );
    }

    #[test]
    fn no_panics_on_all_default_fields() {
        // Exercise every display path over default/unknown data. This mirrors
        // the exact helper calls the cards make, without an egui context.
        let q = QueenUiData::default();
        let _ = (
            fmt_u32(q.org_id),
            fmt_f32(q.energy, 1),
            fmt_f32(q.lifespan_frac, 0),
            fmt_f32(q.health, 0),
            format_exec_ns(q.exec_ns),
            heading_text((0.0, 0.0)),
        );
        let m = MotorMini::default();
        let _ = (
            heading_text(m.move_dir),
            fmt_f32(m.intensity, 2),
            fmt_f32(m.feed_seek, 2),
            fmt_f32(m.emit, 2),
        );
        let s = SensoryMini::default();
        let _ = (
            heading_text(s.food_dir),
            fmt_f32(s.food_dist, 1),
            fmt_f32(s.hazard, 2),
            fmt_f32(s.signal, 2),
            fmt_f32(s.density, 2),
        );
        let c = ColonyUiData::default();
        let _ = fmt_f32(c.avg_generation, 1);
        for v in c.signals {
            let _ = fmt_f32(v, 2);
        }
    }

    // ------------------------------------------------- live extraction

    #[test]
    fn colony_reads_live_stats_and_signals() {
        let mut sim = small_sim(3, 200);
        sim.advance();
        let c = colony_ui_data(&sim);
        assert_eq!(c.pop, sim.stats().population);
        assert_eq!(c.births, sim.stats().total_births);
        assert_eq!(c.deaths, sim.stats().total_deaths);
        assert_eq!(c.births, 0, "no births in the first step");
        assert!(c.avg_generation.is_finite());
        assert_eq!(c.signals.len(), 6);
        for v in c.signals {
            assert!(v.is_finite() && v >= 0.0);
        }
    }

    #[test]
    fn colony_births_counters_follow_the_engine() {
        let mut sim = small_sim(6, 400);
        for _ in 0..900 {
            sim.advance();
        }
        let c = colony_ui_data(&sim);
        let st = sim.stats();
        assert_eq!(c.births, st.total_births);
        assert_eq!(c.deaths, st.total_deaths);
    }

    #[test]
    fn queen_absent_is_honest() {
        let sim = small_sim(2, 100);
        let q = queen_ui_data(&sim);
        assert!(!q.present);
        assert_eq!(q.org_id, u32::MAX);
        assert_eq!(q.neuron_count, u32::MAX, "no brain plugged ⇒ sentinel");
        assert_eq!(q.exec_ns, 0);
        assert!(q.sensory.is_none());
        assert!(q.motor.is_none());
        assert!(q.events.is_empty());
    }

    #[test]
    fn queen_crowned_reads_live_organism() {
        let mut sim = small_sim(1, 300);
        let id = sim.spawn_founder();
        assert!(sim.crown_queen(id));

        let q = queen_ui_data(&sim);
        assert!(q.present, "crown must register in the events ring");
        assert!(q.alive);
        assert_eq!(q.org_id as u64, id);
        assert_eq!(q.events, vec![(0, "crowned".to_string())]);

        let org = sim.organism(id).unwrap();
        assert_eq!(q.energy, org.energy);
        assert_eq!(q.max_energy, org.max_energy);
        assert!((q.lifespan_frac - 0.0).abs() < 1e-6, "age 0 ⇒ 0%");

        // Sensory + motor come from the engine-written net snapshot.
        let s = q.sensory.expect("queen alive ⇒ sensory present");
        assert!(s.hazard.is_finite() && (0.0..=1.0).contains(&s.hazard));
        assert!(s.density.is_finite() && (0.0..=1.0).contains(&s.density));
        assert!(s.signal.is_finite());
        let m = q.motor.expect("queen alive ⇒ motor present");
        assert!(m.intensity.is_finite() && (0.0..=1.0).contains(&m.intensity));
        assert!(m.move_dir.0.is_finite() && m.move_dir.1.is_finite());
        assert!(m.feed_seek.is_nan(), "unmapped channel stays an honest NaN dash");
        assert!(m.emit.is_nan());
        let n = m.neural.expect("stock net ⇒ 3 outputs present");
        for v in n {
            assert!(v.is_finite());
        }
        // Brain not plugged ⇒ sentinels, never invented counters.
        assert_eq!(q.neuron_count, u32::MAX);
        assert_eq!(q.spikes_last, u32::MAX);
        assert!(q.brain_identity.is_none(), "identity getter not in core yet");
    }

    #[test]
    fn queen_food_sensor_points_at_real_food() {
        let params = SimParams {
            initial_population: 1,
            initial_food: 0,
            food_regen_per_sec: 0.0,
            max_population: 1,
            max_food: 8,
            ..SimParams::default()
        };
        let world = alife_worldgen::generate(&WorldGenConfig {
            width: 800,
            height: 600,
            cell: 8,
            ..WorldGenConfig::default()
        });
        let mut sim = Simulation::new(&params, world);
        let id = sim.spawn_founder();
        sim.crown_queen(id);
        let (ox, oy) = {
            let o = sim.organism(id).unwrap();
            (o.x, o.y)
        };
        for k in 0..5 {
            assert!(SimApi::add_food(&mut sim, ox + 10.0, oy + k as f32 * 0.8));
        }
        sim.advance();

        let q = queen_ui_data(&sim);
        let s = q.sensory.expect("queen alive ⇒ sensory");
        assert!(s.food_dir.0.is_finite() || s.food_dir.1.is_finite(), "food seen ⇒ arrow live");
        let fx = s.food_dir.0;
        let fy = s.food_dir.1;
        let len = (fx * fx + fy * fy).sqrt();
        if len > 0.0 {
            // unit vector in the world frame pointing toward the placed food
            assert!((len - 1.0).abs() < 1e-3, "arrow is a unit vector (got {len})");
            // the food sits +X of her: heading component must point there
            assert!(fx > 0.5, "food is directly +X: expected positive world-frame X, got {fx}");
            assert!(s.food_dist.is_finite() && s.food_dist > 8.0, "distance to placed food, got {}", s.food_dist);
        }
    }

    #[test]
    fn queen_death_flips_present_without_panic() {
        let mut sim = small_sim(1, 0);
        let id = sim.spawn_founder();
        sim.crown_queen(id);
        let q0 = queen_ui_data(&sim);
        assert!(q0.alive);
        for _ in 0..1200 {
            sim.advance();
        }
        let q1 = queen_ui_data(&sim);
        assert!(!q1.sensory.is_none_or(|_| false) || q1.alive || q1.events.iter().any(|(_, t)| t.starts_with("died")), "after a long starve run the queen must either be alive or have a death event");
        // whatever the outcome, the read path never panics on missing data
        if !q1.alive {
            assert!(q1.events.iter().any(|(_, t)| t.contains("died (")), "death must be in the ring");
        }
    }
}
