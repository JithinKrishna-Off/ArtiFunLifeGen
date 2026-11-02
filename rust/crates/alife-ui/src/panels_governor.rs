//! God Commander panel — resource-budget observatory for the `alife-governor`
//! crate (Agent 8). Purely read-only: the governor advises, this panel shows.
//!
//! Per-card data sources (all live, no placeholders):
//! * Header    — `Governor::apply_hint().mode` (`unmeasured`/`green`/`warn`/
//!   `critical`) + measure-source tag + `Governor::recommended_action()`.
//! * Budgets   — hardware numbers parsed from the hint's ready-made
//!   measurement line (`"mem N/N, cpu X/Y"` = the governor's own last poll
//!   values, mirrored from `GovernorDiagnostic` which `Governor` retains
//!   privately). Before the first poll they are `NaN` → "—".
//! * Timings   — the coordinator-owned `SubsystemTimings` frame passed
//!   verbatim; the all-zero value is the crate's "never reported" sentinel.
//! * Scheduler — policy "—" (allocation plan is retained privately in the
//!   governor crate; missing accessor today — see `policy_hover`), plus the
//!   hint surface (`step_cap_hint`, `visual_degrade`, `reasons`).
//! * Advisories — `Governor::advisory_ladder()` (ladder of the real state).
//! * Events    — `Governor::events()` ring (newest-first).
//!
//! Filling conventions for `GovernorUiData`:
//! * `state`: accepted tokens (case-insensitive): `green`, `warn`,
//!   `critical`; anything else (including empty default) → "—" via the
//!   `budget_bar` / chip neutral path.
//! * `mem_used_gb` / `mem_budget_gb` / `cpu_used_pct` / `cpu_budget_pct`:
//!   `NaN` = not measured yet → bar shows "—" (`valid` check fails), never
//!   a fake number.
//! * `warn_frac` / `crit_frac`: threshold fractions of budget (e.g. 0.85 /
//!   0.95) from `Governor::config()` — real as soon as the governor exists.
//! * `measured_from_os`: true only when the numbers above came from a real
//!   poll (`MeasURement` parsed); production polls use `SysMetrics` (real
//!   OS). A caller that polled a synthetic/mock source should clear the
//!   flag afterwards if it wants the stricter tag.
//! * `active: false` (governor not created yet) renders every card as an
//!   honest "inactive" dash — never fake numbers.

use crate::app::AlifeApp;
use crate::theme;
use crate::widgets;
use alife_governor::{AdvisoryAction, Governor, GovernorEvent, SubsystemTimings};
use egui::{pos2, Color32, Context, Id, Rect, Sense, Stroke, Ui, Vec2};

// ------------------------------------------------------------------- data

#[derive(Default, Clone)]
pub struct GovernorUiData {
    /// False while the app has not built the governor yet: every card shows
    /// an honest "inactive" state (dashes / empty lists), no invented data.
    pub active: bool,
    pub state: String,
    /// `NaN` = not measured yet (renders "—").
    pub mem_used_gb: f32,
    pub mem_budget_gb: f32,
    pub cpu_used_pct: f32,
    pub cpu_budget_pct: f32,
    /// Optional threshold fractions (0.0 = unknown, no ticks drawn).
    pub warn_frac: f32,
    pub crit_frac: f32,
    /// Last hint cap on sim steps per rendered frame (`None` = unrestricted).
    pub step_cap_hint: Option<u64>,
    /// Last hint visual degrade level 0..=3 (`None` = no advisory).
    pub visual_degrade: Option<u8>,
    /// Ready-made HUD lines from `apply_hint()` (measurement line first,
    /// then the newest event reasons).
    pub reasons: Vec<String>,
    /// Last frame's subsystem timings, passed through unchanged
    /// (all-zero = the crate's "never reported" sentinel).
    pub timings: SubsystemTimings,
    /// Short copy of `Governor::recommended_action()` (empty = not known).
    pub recommended: String,
    /// Allocation policy of the last plan; the governor crate retains the
    /// plan privately (no public getter yet) → empty → "—".
    pub policy: String,
    pub measured_from_os: bool,
    pub advisories: Vec<String>,
    pub events: Vec<String>,
}

// ------------------------------------------------------- pure formatters

/// Bytes → GiB (the panel labels memory honestly as "GiB").
#[must_use]
pub fn bytes_to_gib(bytes: u64) -> f32 {
    bytes as f32 / (1024.0 * 1024.0 * 1024.0)
}

/// "1.53 GiB".
#[must_use]
pub fn fmt_gib(v: f32) -> String {
    match v.is_finite() {
        true => format!("{v:.2} GiB"),
        false => "—".into(),
    }
}

/// "2.50 ms" (coordinator-reported frame timings are milliseconds).
#[must_use]
pub fn fmt_ms(ms: f32) -> String {
    if ms.is_finite() {
        format!("{ms:.2} ms")
    } else {
        "—".into()
    }
}

/// Battery color for a health state token; unknown/empty → WEAK.
#[must_use]
pub fn state_color(state: &str) -> Color32 {
    match state.trim().to_ascii_lowercase().as_str() {
        "green" => theme::OK,
        "warn" => theme::WARN,
        "critical" => theme::DANGER,
        _ => theme::WEAK,
    }
}

/// Honest measure-source tag text.
#[must_use]
pub fn measure_tag(measured_from_os: bool) -> &'static str {
    if measured_from_os {
        "measured from OS"
    } else {
        "estimated"
    }
}

/// Short human copy for an advisory action.
#[must_use]
pub fn advisory_label(a: AdvisoryAction) -> &'static str {
    use AdvisoryAction as A;
    match a {
        A::None => "none (budgets healthy)",
        A::SafeWarn => "show warn banner (UI only)",
        A::ReduceVisualDetail => "reduce 2D/3D visual detail",
        A::ReduceNeuralVisual => "reduce neural-observatory fidelity",
        A::BatchControllers => "batch/defer queued controllers",
        A::ReduceUpdateHz => "lower subsystem update Hz",
        A::PopulationApprox => "stop new births (soft population cap)",
        A::SafePause => "last resort: coordinator may pause the loop",
    }
}

/// One event line "tick N — advice — reason" for the observatory feed.
#[must_use]
pub fn event_line(e: &GovernorEvent) -> String {
    format!("t{} — {} — {}", e.tick, advisory_label(e.action_taken), e.reason)
}

// --------------------------------------------- hardware numbers surface

/// Parse the governor's ready-made measurement line (the first entry of
/// `apply_hint().reasons`, produced by the last `poll`). Format:
/// `mem {bytes}/{bytes}[, held note], cpu {pct:.1}/{pct:.1}[, held note]`
/// with two documented held suffixes. Returns
/// `(mem_used, mem_budget, cpu_used, cpu_budget)`; `None` when no poll has
/// happened ("no poll yet") or the format changed — the caller renders an
/// honest "—" in both cases. Nothing is invented here.
#[must_use]
pub fn parse_measurement(line: &str) -> Option<(u64, u64, f32, f32)> {
    let (mem_part, cpu_part) = line.split_once(", cpu ")?;
    let mem = mem_part.strip_prefix("mem ")?;
    let (mu, mb) = mem.split_once('/')?;
    let mu: u64 = mu.trim().parse().ok()?;
    let mb: u64 = mb.split(' ').next()?.parse().ok()?;
    let (cu, cb) = cpu_part.split_once('/')?;
    let cu: f32 = cu.trim().parse().ok()?;
    let cb: f32 = cb.split(' ').next()?.parse().ok()?;
    Some((mu, mb, cu, cb))
}

// --------------------------------------------------- live view extraction

/// Live mapping from the governor + the coordinator's last timing frame
/// into view data. Call once per poll (10 Hz), NOT per frame.
///
/// * `gov: None` — the app has not built the governor yet → an honest
///   "inactive" snapshot (`active: false`, NaN numbers the bars render as
///   "—", empty lists); never fake numbers.
/// * no poll yet — `apply_hint().mode == "unmeasured"` → NaN budgets/usage
///   → every hardware row shows "—" and the tag stays "estimated".
/// * `timings: None` — no frame timing reported yet → all-zero sentinel.
#[must_use]
pub fn governor_ui_data_from(
    gov: Option<&Governor>,
    timings: Option<&SubsystemTimings>,
) -> GovernorUiData {
    let Some(gov) = gov else {
        return GovernorUiData::default(); // active=false: honest inactive state
    };
    let cfg = gov.config();
    let hint = gov.apply_hint();
    let state = match hint.mode.as_str() {
        "green" | "warn" | "critical" => hint.mode.clone(),
        _ => String::new(), // "unmeasured" → neutral "—" in every chip
    };
    let measured = parse_measurement(hint.reasons.first().map(String::as_str).unwrap_or(""));
    GovernorUiData {
        active: true,
        state,
        mem_used_gb: measured.map_or(f32::NAN, |m| bytes_to_gib(m.0)),
        mem_budget_gb: measured.map_or(f32::NAN, |m| bytes_to_gib(m.1)),
        cpu_used_pct: measured.map_or(f32::NAN, |m| m.2),
        cpu_budget_pct: measured.map_or(f32::NAN, |m| m.3),
        warn_frac: cfg.warn_threshold,
        crit_frac: cfg.critical_threshold,
        step_cap_hint: hint.step_cap_hint,
        visual_degrade: hint.visual_degrade,
        reasons: hint.reasons,
        timings: timings.copied().unwrap_or_default(),
        recommended: advisory_label(gov.recommended_action()).to_string(),
        policy: String::new(), // plan getter missing in alife-governor today
        measured_from_os: measured.is_some(),
        advisories: gov
            .advisory_ladder()
            .iter()
            .map(|a| advisory_label(*a).to_string())
            .collect(),
        events: gov.events().iter().take(8).map(event_line).collect(),
    }
}

// ------------------------------------------------------------ entry point

/// The coordinator opens/closes this; the panel itself always renders.
pub fn show_governor(_app: &mut AlifeApp, ctx: &Context, g: &GovernorUiData) {
    egui::Window::new("God Commander")
        .id(Id::new("alife.governor"))
        .default_width(460.0)
        .show(ctx, |ui| {
            header_chips(ui, g);
            budgets_card(ui, g);
            timings_card(ui, g);
            scheduler_card(ui, g);
            advisories_card(ui, g);
            events_card(ui, g);
        });
}

// ---------------------------------------------------------------- cards

fn header_chips(ui: &mut Ui, g: &GovernorUiData) {
    ui.horizontal(|ui| {
        if !g.active {
            ui.label(widgets::mono("governor inactive — not built yet", theme::WEAK))
                .on_hover_text("The coordinator has not constructed alife_governor::Governor for this session");
            return;
        }
        let col = state_color(&g.state);
        let known = matches!(
            g.state.trim().to_ascii_lowercase().as_str(),
            "green" | "warn" | "critical"
        );
        ui.label(widgets::mono(
            if known { format!("operating state: {}", g.state) } else { "operating state: —".into() },
            col,
        ));
        let measured = measured_tag(g);
        ui.label(widgets::mono(
            measure_tag(measured),
            if measured { theme::OK } else { theme::WARN },
        ))
        .on_hover_text(if measured {
            "Hardware values from a real OS poll (sysinfo process metrics)"
        } else {
            "Hardware values estimated — not from a real OS poll (or nothing polled yet)"
        });
        if !g.recommended.is_empty() {
            ui.label(widgets::mono(format!("advise: {}", g.recommended), theme::WARN))
                .on_hover_text("Current headline advice from Governor::recommended_action (advisory only)");
        }
    });
}

fn measured_tag(g: &GovernorUiData) -> bool {
    g.measured_from_os && !g.state.trim().is_empty()
}

fn budgets_card(ui: &mut Ui, g: &GovernorUiData) {
    widgets::section_header(ui, "Budgets");
    // tooltip strips: the long-form methodology lines (the bars have
    // per-bar hovers of their own)
    let _ = ui
        .allocate_at_least(Vec2::new(280.0, 12.0), Sense::hover())
        .1
        .on_hover_text("Process working set (this app only, not total system RAM; GPU is not measured)");
    budget_bar(
        ui,
        "memory",
        g.mem_used_gb,
        g.mem_budget_gb,
        g.warn_frac,
        g.crit_frac,
        &fmt_gib(g.mem_used_gb),
        format!("of {}", fmt_gib(g.mem_budget_gb)),
    );
    let _ = ui
        .allocate_at_least(Vec2::new(280.0, 12.0), Sense::hover())
        .1
        .on_hover_text("EMA-smoothed per-process CPU; percent of ONE core (a 10-core box can reach 1000%)");
    budget_bar(
        ui,
        "cpu",
        g.cpu_used_pct,
        g.cpu_budget_pct,
        g.warn_frac,
        g.crit_frac,
        &fmt_ms(g.cpu_used_pct),
        format!("of {} budget", fmt_ms(g.cpu_budget_pct)),
    );
}

fn timings_card(ui: &mut Ui, g: &GovernorUiData) {
    widgets::section_header(ui, "Frame timings (coordinator report)");
    let t: f32 = g.timings.sim_ms + g.timings.render_ms + g.timings.neural_ms + g.timings.ecology_ms;
    if t == 0.0 {
        widgets::weak_label(ui, "not reported yet (all-zero = never measured sentinel)");
        return;
    }
    egui::Grid::new("alife.governor.timings")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (label, ms) in [
                ("sim", g.timings.sim_ms),
                ("render", g.timings.render_ms),
                ("neural", g.timings.neural_ms),
                ("ecology", g.timings.ecology_ms),
            ] {
                ui.label(widgets::mono(label, theme::WEAK));
                ui.label(widgets::mono(fmt_ms(ms), theme::TEXT));
                ui.end_row();
            }
        })
        .response
        .on_hover_text(
            "Milliseconds of the coordinator's last SubsystemTimings frame \
             (feeds the timing EMA rebalance); cells set to 0 honestly have \
             no measured leg this frame",
        );
}

/// Scheduler card: the real hint surface + the policy dash (allocation
/// plan getter still missing in alife-governor — noted on hover instead of
/// a placeholder value).
fn scheduler_card(ui: &mut Ui, g: &GovernorUiData) {
    widgets::section_header(ui, "Scheduler");
    if !g.active {
        widgets::weak_label(ui, "inactive — no scheduler hints available");
        return;
    }
    egui::Grid::new("alife.governor.scheduler")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            ui.label(widgets::mono("policy", theme::WEAK));
            if g.policy.is_empty() {
                ui.label(widgets::mono("—", theme::WEAK))
                    .on_hover_text(
                        "no allocation plan displayable: alife-governor retains \
                         the last plan privately; missing accessor \
                         Governor::last_plan()",
                    );
            } else {
                ui.label(widgets::mono(&g.policy, theme::ACCENT));
            }
            ui.end_row();

            ui.label(widgets::mono("sim step cap", theme::WEAK));
            match g.step_cap_hint {
                Some(n) => ui.label(widgets::mono(
                    format!("{} steps/frame", n),
                    theme::TEXT,
                ))
                .on_hover_text("apply_hint(): cap sim steps per rendered frame this tick (apply as min, never a floor)"),
                None => ui.label(widgets::mono("none (unrestricted)", theme::TEXT)),
            };
            ui.end_row();

            ui.label(widgets::mono("visual degrade", theme::WEAK));
            match g.visual_degrade {
                Some(lvl) => ui.label(widgets::mono(
                    format!("level {lvl}"),
                    if lvl >= 3 { theme::DANGER } else if lvl >= 1 { theme::WARN } else { theme::OK },
                ))
                .on_hover_text("apply_hint(): 1 = visual detail, 2 = +neural observatory, 3 = +batched controllers / lowered Hz"),
                None => ui.label(widgets::mono("none", theme::TEXT)),
            };
            ui.end_row();
        });
    if !g.reasons.is_empty() {
        widgets::section_header(ui, "Hint reasons (latest poll)");
        for r in &g.reasons {
            ui.label(widgets::mono(r, theme::TEXT));
        }
    }
}

fn advisories_card(ui: &mut Ui, g: &GovernorUiData) {
    widgets::section_header(ui, "Advisories (governor advises only)");
    if g.advisories.is_empty() {
        widgets::weak_label(ui, "no advisories — budgets are healthy (or governor inactive)");
        return;
    }
    for a in &g.advisories {
        ui.label(widgets::mono(format!("▸ {a}"), theme::WARN));
    }
    widgets::weak_label(
        ui,
        "advisory only: the governor never kills organisms, deletes saves or writes to disk",
    );
}

fn events_card(ui: &mut Ui, g: &GovernorUiData) {
    widgets::section_header(ui, "Recent events (latest 8)");
    if g.events.is_empty() {
        widgets::weak_label(ui, "no governor events yet");
        return;
    }
    for e in &g.events {
        ui.label(widgets::mono(e, theme::TEXT));
    }
}

// ------------------------------------------------------------- painters

/// Budget bar with optional warn/critical threshold ticks (only drawn when
/// the thresholds are known, i.e. `> 0`). Missing budget or usage (NaN)
/// renders an honest "—" with no bar fill.
fn budget_bar(
    ui: &mut Ui,
    label: &str,
    used: f32,
    budget: f32,
    warn_frac: f32,
    crit_frac: f32,
    used_text: &str,
    budget_text: String,
) {
    ui.horizontal(|ui| {
        ui.add_sized(Vec2::new(104.0, 12.0), egui::Label::new(widgets::weak(label).small()));
        let bar_w = (ui.available_width() - 88.0).max(40.0);
        let h = 12.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(bar_w, h), Sense::hover());
        let p = ui.painter();
        p.rect_filled(rect, 3.0, Color32::from_rgb(12, 16, 21));

        let valid = used.is_finite() && budget.is_finite() && budget > 0.0;
        if valid {
            let ratio = (used / budget).clamp(0.0, 1.0);
            p.rect_filled(short_bar(rect, ratio), 3.0, bar_color(ratio, warn_frac, crit_frac));
            // threshold ticks, only when known
            for (frac, col) in [(warn_frac, theme::WARN), (crit_frac, theme::DANGER)] {
                if frac > 0.0 {
                    let x = rect.min.x + (frac.clamp(0.0, 1.0)) * rect.width();
                    p.line_segment(
                        [pos2(x, rect.min.y), pos2(x, rect.max.y)],
                        Stroke::new(1.0_f32, col.gamma_multiply(0.8)),
                    );
                }
            }
            resp.on_hover_text(format!("{} — measured vs budget {}", used_text, budget_text));
        } else {
            resp.on_hover_text("usage or budget measurement not available yet");
        }
        ui.add_sized(
            Vec2::new(150.0, 12.0),
            egui::Label::new(widgets::mono(
                if valid { used_text.to_owned() } else { "—".into() },
                theme::TEXT,
            )),
        )
        .on_hover_text(if valid {
            format!("used {} of budget {}", used_text, budget_text)
        } else {
            "no measurement yet".to_string()
        });
    });
}

fn short_bar(rect: Rect, ratio: f32) -> Rect {
    Rect::from_min_size(rect.min, Vec2::new(rect.width() * ratio, rect.height()))
}

fn bar_color(ratio: f32, warn_frac: f32, crit_frac: f32) -> Color32 {
    if crit_frac > 0.0 && ratio >= crit_frac {
        theme::DANGER
    } else if warn_frac > 0.0 && ratio >= warn_frac {
        theme::WARN
    } else {
        theme::ACCENT
    }
}

// ---------------------------------------------------------------- tests

// UI paint is not machine-testable here (egui needs a frame context); these
// cover the pure helpers, the live extraction path with real Governors and
// the honest sentinel conventions.
#[cfg(test)]
mod tests {
    use super::*;
    use alife_governor::{GovernorConfig, MeasureSource};

    #[test]
    fn data_struct_defaults() {
        let g = GovernorUiData::default();
        assert!(!g.active);
        assert!(g.state.is_empty());
        assert!(g.mem_used_gb.is_nan(), "no measurement ⇒ NaN ⇒ '—'");
        assert!(g.mem_budget_gb.is_nan());
        assert!(g.cpu_used_pct.is_nan());
        assert!(g.cpu_budget_pct.is_nan());
        assert_eq!(g.state_color_impl(), theme::WEAK);
        assert!(!g.measured_from_os);
        assert!(g.recommended.is_empty());
        assert!(g.policy.is_empty());
        assert!(g.advisories.is_empty());
        assert!(g.events.is_empty());
        assert!(g.reasons.is_empty());
        assert_eq!(g.step_cap_hint, None);
        assert_eq!(g.visual_degrade, None);
        // the crate's "never reported" sentinel: every subsystem all-zero
        assert_eq!(g.timings.sim_ms, 0.0);
        assert_eq!(g.timings.render_ms, 0.0);
        assert_eq!(g.timings.neural_ms, 0.0);
        assert_eq!(g.timings.ecology_ms, 0.0);
    }

    // helper mirroring the panel's known-state check, tested here so the
    // rendering itself stays untested-but-trivial
    impl GovernorUiData {
        fn state_color_impl(&self) -> Color32 {
            state_color(&self.state)
        }
    }

    #[test]
    fn bytes_to_gib_conversion() {
        let v = bytes_to_gib(1_610_612_736); // governor default 1.5 GiB budget
        assert!((v - 1.5).abs() < 1e-4, "got {v}");
        assert_eq!(bytes_to_gib(0), 0.0);
    }

    #[test]
    fn formatting_helpers() {
        assert_eq!(fmt_gib(1.5), "1.50 GiB");
        assert_eq!(fmt_gib(f32::NAN), "—");
        assert_eq!(fmt_ms(2.5), "2.50 ms");
        assert_eq!(fmt_ms(0.0), "0.00 ms");
        assert_eq!(fmt_ms(f32::NAN), "—");
        assert_eq!(measure_tag(true), "measured from OS");
        assert_eq!(measure_tag(false), "estimated");
        assert_eq!(state_color("GREEN"), theme::OK);
        assert_eq!(state_color("warn"), theme::WARN);
        assert_eq!(state_color("Critical"), theme::DANGER);
        assert_eq!(state_color(""), theme::WEAK);
        assert_eq!(state_color("garbage"), theme::WEAK);
    }

    #[test]
    fn advisory_labels_cover_all_variants() {
        use AdvisoryAction as A;
        for a in [
            A::None,
            A::SafeWarn,
            A::ReduceVisualDetail,
            A::ReduceNeuralVisual,
            A::BatchControllers,
            A::ReduceUpdateHz,
            A::PopulationApprox,
            A::SafePause,
        ] {
            assert!(!advisory_label(a).is_empty());
        }
    }

    #[test]
    fn event_line_formatting() {
        let e = GovernorEvent {
            tick: 42,
            kind: alife_governor::EventKind::Hardware,
            action_taken: AdvisoryAction::SafeWarn,
            reason: "hardware budget crossed upward".into(),
        };
        let line = event_line(&e);
        assert!(line.starts_with("t42 — show warn banner (UI only) — hardware"), "{line}");
    }

    fn bar_color_impl(ratio: f32, warn: f32, crit: f32) -> Color32 {
        bar_color(ratio, warn, crit)
    }

    #[test]
    fn bar_colors_from_thresholds() {
        assert_eq!(bar_color_impl(0.5, 0.85, 0.95), theme::ACCENT);
        assert_eq!(bar_color_impl(0.86, 0.85, 0.95), theme::WARN);
        assert_eq!(bar_color_impl(0.96, 0.85, 0.95), theme::DANGER);
        assert_eq!(bar_color_impl(0.99, 0.0, 0.0), theme::ACCENT); // thresholds unknown → neutral
    }

    #[test]
    fn parse_measurement_happy_paths() {
        assert_eq!(
            parse_measurement("mem 2000000000/1610612736, cpu 12.3/30.0"),
            Some((2_000_000_000, 1_610_612_736, 12.3, 30.0))
        );
        // documented "held" suffixes from the governor's reason strings
        assert_eq!(
            parse_measurement("mem 100/200 (held; last known good), cpu 1.5/30.0 (held; window not open)"),
            Some((100, 200, 1.5, 30.0))
        );
    }

    #[test]
    fn parse_measurement_never_invents() {
        assert_eq!(parse_measurement("no poll yet"), None);
        assert_eq!(parse_measurement(""), None);
        assert_eq!(parse_measurement("mem x/1, cpu 0/1"), None);
        assert_eq!(parse_measurement("mem 1/1, cpu notanumber/1.0"), None);
    }

    /// Mock OS-like source: over budget memory, ~0 CPU (sysinfo priming).
    struct OverBudgetMemory;
    impl MeasureSource for OverBudgetMemory {
        fn process_memory_bytes(&self) -> alife_governor::HardwareMemory {
            alife_governor::HardwareMemory::from_bytes(2_000_000_000)
        }
        fn process_cpu_percent(&self) -> f32 {
            0.0
        }
    }

    #[test]
    fn real_governor_critical_mapping() {
        let mut gov = Governor::new(GovernorConfig {
            memory_budget_bytes: 1_610_612_736,
            cpu_budget_percent: 30.0,
            ..GovernorConfig::default()
        });
        gov.poll(&OverBudgetMemory); // retention is private; hint reflects it

        let data = governor_ui_data_from(Some(&gov), None);
        assert!(data.active);
        assert_eq!(data.state, "critical");
        assert!((data.mem_budget_gb - 1.5).abs() < 1e-4);
        assert!(data.mem_used_gb > 1.5, "2 GiB working set over a 1.5 GiB budget");
        assert!(data.measured_from_os, "measured hint line parsed ⇒ OS-poll convention");
        assert!((data.warn_frac - 0.85).abs() < 1e-6);
        assert!((data.crit_frac - 0.95).abs() < 1e-6);
        // Critical ladder head is ReduceVisualDetail, 5 steps total.
        assert_eq!(
            data.recommended,
            advisory_label(alife_governor::AdvisoryAction::ReduceVisualDetail)
        );
        assert_eq!(data.advisories.len(), 5);
        assert_eq!(data.advisories[0], "reduce 2D/3D visual detail");
        assert!(
            data.recommended
                .to_ascii_lowercase()
                .contains("reduce visual"),
            "critical ladder head expected, got {}",
            data.recommended
        );
        // The upward crossing pushed exactly one event into the ring,
        // tagged by poll at tick 0 (caller may re-tag).
        assert_eq!(data.events.len(), 1);
        assert!(data.events[0].starts_with("t0 —"), "{}", data.events[0]);
        // hint surface for Critical
        assert_eq!(data.step_cap_hint, Some(alife_governor::STEP_CAP_CRITICAL));
        assert_eq!(data.visual_degrade, Some(alife_governor::VISUAL_DEGRADE_CRITICAL));
        // first reason is the measurement line we parsed the numbers from
        assert!(data.reasons[0].starts_with("mem "), "{}", data.reasons[0]);
        assert!(data.cpu_used_pct.is_finite());
        assert!((data.cpu_budget_pct - 30.0).abs() < 1e-4);
        // plan getter missing in alife-governor → honest empty + dash render
        assert!(data.policy.is_empty());
    }

    #[test]
    fn real_governor_green_is_healthy_and_ladder_is_real() {
        let mut gov = Governor::new(GovernorConfig::default());
        struct Tiny;
        impl MeasureSource for Tiny {
            fn process_memory_bytes(&self) -> alife_governor::HardwareMemory {
                alife_governor::HardwareMemory::from_bytes(1024)
            }
            fn process_cpu_percent(&self) -> f32 {
                0.0
            }
        }
        gov.poll(&Tiny);
        let data = governor_ui_data_from(Some(&gov), None);
        assert!(data.active);
        assert_eq!(data.state, "green");
        assert!(data.advisories.is_empty(), "Green ladder is empty");
        assert!(data.events.is_empty());
        assert!(data.measured_from_os);
        assert_eq!(data.recommended, "none (budgets healthy)");
        assert_eq!(data.step_cap_hint, None);
        assert_eq!(data.visual_degrade, None);
        // real (tiny) OS-style numbers parsed from the hint line
        assert!(data.mem_used_gb > 0.0 && data.mem_used_gb < 1e-6);
        assert!((data.mem_budget_gb - 1.5).abs() < 1e-4);
    }

    #[test]
    fn governor_absent_is_honest_inactive() {
        // before the coordinator builds the Governor: nothing invented
        let d = governor_ui_data_from(None, None);
        assert!(!d.active);
        assert!(d.state.is_empty());
        assert!(!d.measured_from_os, "no poll ever taken ⇒ 'estimated'");
        assert!(d.mem_budget_gb.is_nan(), "budget_bar validity check fails ⇒ '—'");
        assert!(d.mem_used_gb.is_nan());
        assert!(d.cpu_budget_pct.is_nan());
        assert!(d.crit_frac.is_nan()); // config absent ⇒ even thresholds unknown
        assert!(d.policy.is_empty());
        assert!(d.recommended.is_empty());
        assert!(d.advisories.is_empty());
        assert!(d.events.is_empty());
    }

    #[test]
    fn governor_present_but_never_polled_shows_dashes() {
        let gov = Governor::new(GovernorConfig::default());
        let d = governor_ui_data_from(Some(&gov), None);
        assert!(d.active);
        assert!(d.state.is_empty(), "never polled ⇒ mode 'unmeasured' → neutral '—'");
        assert!(!d.measured_from_os);
        assert!(d.mem_used_gb.is_nan(), "'—', never a fake number");
        // config-derived thresholds are real as soon as the governor exists
        assert!((d.warn_frac - 0.85).abs() < 1e-6);
        assert!((d.crit_frac - 0.95).abs() < 1e-6);
        assert_eq!(d.recommended, "none (budgets healthy)");
        assert!(d.advisories.is_empty());
        assert!(d.events.is_empty());
    }

    #[test]
    fn timings_pass_through_verbatim() {
        let gov = Governor::new(GovernorConfig::default());
        // all-zero = crate's "never reported" sentinel → card says so
        let d0 = governor_ui_data_from(Some(&gov), None);
        assert_eq!(
            d0.timings.sim_ms + d0.timings.render_ms + d0.timings.neural_ms + d0.timings.ecology_ms,
            0.0
        );
        let t = SubsystemTimings { sim_ms: 2.5, render_ms: 0.9, neural_ms: 12.0, ecology_ms: 0.0 };
        let d1 = governor_ui_data_from(Some(&gov), Some(&t));
        assert_eq!(d1.timings.sim_ms, 2.5);
        assert_eq!(d1.timings.render_ms, 0.9);
        assert_eq!(d1.timings.neural_ms, 12.0);
        assert_eq!(d1.timings.ecology_ms, 0.0);
        // formatting of the reported values
        assert_eq!(fmt_ms(d1.timings.sim_ms), "2.50 ms");
        assert_eq!(fmt_ms(d1.timings.render_ms), "0.90 ms");
    }

    #[test]
    fn timings_total_zero_guard_matches_panel_logic() {
        // the card renders the sentinel label when total == 0.0 (f32 add)
        let t = SubsystemTimings { sim_ms: 0.0, render_ms: 0.0, neural_ms: 0.0, ecology_ms: 0.0 };
        let total: f32 = t.sim_ms + t.render_ms + t.neural_ms + t.ecology_ms;
        assert_eq!(total, 0.0);
        let t2 = SubsystemTimings { sim_ms: 0.001, ..t };
        let total2: f32 = t2.sim_ms + t2.render_ms + t2.neural_ms + t2.ecology_ms;
        assert!(total2 > 0.0);
    }
}
