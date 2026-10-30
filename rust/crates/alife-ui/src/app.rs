//! Application state, native window bootstrap, shortcut handling and the
//! fixed-step simulation accumulator.

use crate::dialogs::{self, Dialog};
use crate::panels_editor;
use crate::panels_experiments;
use crate::panels_inspector;
use crate::panels_screens;
use crate::panels_stats::{HistCache, StatsCache};
use crate::panels_tree::TreeCache;
use crate::panels_tutorial;
use crate::panels_world;
use crate::save_io;
use crate::theme;
use crate::god_api::{GodCache, GodLogEntry, GodTool, WeatherSnapshot};
use crate::widgets::{ToastKind, Toasts};
use alife_common::{
    persist::{HistoryBuffer, WorldSave},
    InteractionMode, LineageRecord, SimParams, WorldGenConfig,
};
use alife_core::{SimApi, Simulation, DT};
use alife_data::DataStore;
use alife_render::{WeatherLayer, WorldView};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Render/perf setting chosen in Settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// No trails, most throttling.
    Efficient,
    /// Trails sampled every 4th frame (default).
    Balanced,
    /// Trails every frame, charts at full rate.
    High,
}

impl Quality {
    pub const ALL: [Quality; 3] = [Quality::Efficient, Quality::Balanced, Quality::High];

    pub fn label(self) -> &'static str {
        match self {
            Quality::Efficient => "Efficient",
            Quality::Balanced => "Balanced",
            Quality::High => "High",
        }
    }

    /// Number of frames between trail captures (0 = trails off).
    pub fn trail_every(self) -> u64 {
        match self {
            Quality::Efficient => 0,
            Quality::Balanced => 4,
            Quality::High => 1,
        }
    }
}

/// Segmented time-scale control. Speed multiplies steps per frame per second;
/// the fixed sim step stays 1/60 and a hard per-frame budget keeps the UI alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeScale {
    RealTime,
    Fast,
    Generational,
    Century,
    Millennium,
}

impl TimeScale {
    pub const ALL: [TimeScale; 5] = [
        TimeScale::RealTime,
        TimeScale::Fast,
        TimeScale::Generational,
        TimeScale::Century,
        TimeScale::Millennium,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TimeScale::RealTime => "Real-time",
            TimeScale::Fast => "Fast",
            TimeScale::Generational => "Generational",
            TimeScale::Century => "Century",
            TimeScale::Millennium => "Millennium+",
        }
    }

    /// Requested multiplier (sim seconds per wall second).
    pub fn mult(self) -> f32 {
        match self {
            TimeScale::RealTime => 1.0,
            TimeScale::Fast => 4.0,
            TimeScale::Generational => 16.0,
            TimeScale::Century => 64.0,
            TimeScale::Millennium => 256.0,
        }
    }

    /// Honest approximation note for the extreme end of the scale.
    pub fn note(self) -> Option<&'static str> {
        match self {
            TimeScale::Millennium => Some(
                "honest approx: only biology integrates; a cohort's micro-interactions fog over",
            ),
            _ => None,
        }
    }
}

/// Which main view is displayed in the center.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppView {
    World,
    Statistics,
    Evolution,
    Tree,
    Experiments,
    GodMode,
    WorldEditor,
    Settings,
}

impl AppView {
    pub const ALL: [AppView; 8] = [
        AppView::World,
        AppView::Statistics,
        AppView::Evolution,
        AppView::Tree,
        AppView::Experiments,
        AppView::GodMode,
        AppView::WorldEditor,
        AppView::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AppView::World => "World",
            AppView::Statistics => "Statistics",
            AppView::Evolution => "Evolution",
            AppView::Tree => "Tree",
            AppView::Experiments => "Experiments",
            AppView::GodMode => "God Mode",
            AppView::WorldEditor => "World Editor",
            AppView::Settings => "Settings",
        }
    }

    /// Two-letter compact label for the narrow nav rail (< 1200px window).
    pub fn short(self) -> &'static str {
        match self {
            AppView::World => "WO",
            AppView::Statistics => "ST",
            AppView::Evolution => "EV",
            AppView::Tree => "TR",
            AppView::Experiments => "EX",
            AppView::GodMode => "GO",
            AppView::WorldEditor => "ED",
            AppView::Settings => "SE",
        }
    }
}

/// Hard per-frame simulation step budget (budgeted speeds expect throttling).
pub const MAX_STEPS_PER_FRAME: u64 = 2048;
/// Smoothed achieved-rate samples (one per 0.5 s measurement window).
const RATE_RING_N: usize = 8;

/// One Instant-checked slow UI clock (10 Hz) for every throttled read:
/// weather snapshots, intervention log, cached toolbar/status strings.
#[derive(Default)]
struct Hz10Clock {
    last: Option<Instant>,
}

impl Hz10Clock {
    fn due(&mut self) -> bool {
        let now = Instant::now();
        let due = self
            .last
            .map_or(true, |l| now.duration_since(l).as_secs_f32() >= 0.1);
        if due {
            self.last = Some(now);
        }
        due
    }
}

/// Toolbar + status strip strings, refreshed only on the 10 Hz clock —
/// no `format!` per frame in the shell hot path.
#[derive(Default)]
pub struct StatusText {
    pub pop: String,
    pub step: String,
    pub short_status: String,
    pub achieved: String,
    pub sim_clock: String,
    pub species: String,
    pub food: String,
    pub cursor: Option<String>,
    pub world_dim: String,
}

/// Deterministic seed derived from wall clock (no unwrap on system time).
pub fn random_seed() -> u32 {
    const MIX: u32 = 0x9E37_79B9;
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ ((d.as_secs() as u32).wrapping_mul(MIX)))
        .unwrap_or(1337)
        | 1
}

/// The whole running application.
pub struct AlifeApp {
    pub store: DataStore,

    // world / sim
    pub sim: Simulation,
    pub history: HistoryBuffer,
    pub view: WorldView,
    pub world_cfg: WorldGenConfig,
    pub world_name: String,
    pub preset_name: String,
    pub world_tag: u32,

    // playback
    pub running: bool,
    pub timescale: TimeScale,
    /// true when the per-frame step budget was hit and the engine throttled
    pub budgeted: bool,
    acc: f32,

    // god mode / weather
    pub god_tool: Option<GodTool>,
    pub god_radius: f32,
    pub god_species: Option<u32>,
    pub god_species_text: String,
    pub god_log: Vec<GodLogEntry>,
    pub god_cache: Option<GodCache>,
    pub god_purge_arm: u32,
    /// 10 Hz weather sample for the viewport HUD
    pub hud_weather: Option<WeatherSnapshot>,
    /// 10 Hz render-side weather layer snapshot (Arc-shared, cheap to clone)
    pub weather_layer: Option<WeatherLayer>,

    // selection polish (throttled forgiving hover pick)
    pub hover_pick: Option<(u64, f32, f32)>,
    /// "id #N" chip text cache, rebuilt only when the hovered id changes
    pub hover_chip: (u64, String),

    // tutorial
    pub tutorial: panels_tutorial::TutorialState,

    // death-cause tracking from observed step events (session-local)
    pub death_causes: [u64; 3],
    pub death_seen: bool,

    /// 1 Hz tree cache (see panels_tree)
    pub tree_cache: Option<TreeCache>,

    // selection / camera / editor
    pub view_sel: AppView,
    pub selected: Option<u64>,
    pub follow_selected: bool,
    pub editor_mode: InteractionMode,
    pub editor_brush: f32,
    pub pick_radius_px: f32,

    // ui state
    pub show_inspector: bool,
    pub show_shortcuts: bool,
    pub show_hint: bool,
    pub hint_age: f32,
    pub quality: Quality,
    pub stats_hz: f32,
    pub toasts: Toasts,
    pub dialog: Option<Dialog>,

    // editor regenerate scratch
    pub gen_seed: u32,
    pub gen_preset_idx: usize,
    pub gen_width: u32,
    pub gen_height: u32,

    // canvas telemetry for overlays + status strip
    pub canvas_rect: Option<egui::Rect>,
    pub last_hover: Option<(f32, f32)>,

    // population tracking (extinction banner)
    pub was_populated: bool,
    pub warned_extinct: bool,

    // lineage record bookkeeping for saves
    pub records_seen: HashMap<u64, LineageRecord>,

    // throttled analytics caches
    pub stats_cache: Option<StatsCache>,
    pub hist_cache: Option<HistCache>,

    // memory-minded UI: 10 Hz clock + cached shell strings
    clock: Hz10Clock,
    pub text: StatusText,

    // timing
    last_frame: Option<Instant>,
    steps_win_start: Instant,
    steps_win_steps: u64,
    pub sps: f64,
    /// smoothed sim seconds per wall second (ring-average)
    pub achieved_rate: f64,
    pub rate_ring: [f64; RATE_RING_N],
    rate_idx: usize,
    rate_filled: usize,
    pub frame_no: u64,

    // ---- 3D living world (panels_3d) ----
    /// wgpu backend present && View3D successfully installed
    pub view3d_ready: bool,
    /// the 3D canvas replaces the 2D canvas while enabled
    pub view3d_on: bool,
    /// current multi-scale camera request (Follow / Orbit / Planet)
    pub mode3d: alife_view3d::camera::CameraMode,
    /// persistent camera (transitions live in alife-view3d)
    pub cam3d: alife_view3d::camera::CameraState,
    /// Arc-cached world model for the renderer (recreated only on regeneration)
    world3d_cache: Option<Arc<alife_worldgen::WorldModel>>,
    world3d_tag: Option<u32>,
    /// last known position of a followed organism (death fallback anchor)
    pub last_follow_pos: Option<(f32, f32)>,
    /// elapsed seconds in the 3D layer (shader time)
    pub cam_t: f32,
    /// smooth frame dt for camera damping (UI-measured)
    pub dt_last: f32,
}

impl AlifeApp {
    pub fn new(cc: &eframe::CreationContext<'_>, data_dir: std::path::PathBuf) -> Self {
        theme::apply(&cc.egui_ctx);

        // attach the 3D renderer when eframe runs the wgpu backend
        let view3d_ready = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| alife_view3d::render::install(rs))
            .unwrap_or(false);

        let (cfg, params) = save_io::apply_preset_with_fallback("genesis");
        let root = data_dir.clone();
        let store = DataStore::new(&data_dir);
        let world = alife_worldgen::generate(&cfg);
        let sim = Simulation::new(&params, world);

        let mut view = WorldView::new();
        view.trails.set_enabled(true);
        view.invalidate_world(1);

        let mut app = Self {
            store,
            sim,
            history: HistoryBuffer::new(20, 2048),
            view,
            world_cfg: cfg,
            world_name: "genesis-start".to_string(),
            preset_name: "genesis".to_string(),
            world_tag: 1,
            running: false,
            timescale: TimeScale::RealTime,
            budgeted: false,
            acc: 0.0,

            god_tool: None,
            god_radius: 90.0,
            god_species: None,
            god_species_text: String::new(),
            god_log: Vec::new(),
            god_cache: None,
            god_purge_arm: 0,
            hud_weather: None,
            weather_layer: None,

            hover_pick: None,
            hover_chip: (0, String::new()),

            tutorial: panels_tutorial::initial_state(&root),

            death_causes: [0; 3],
            death_seen: false,

            tree_cache: None,

            view_sel: AppView::World,
            selected: None,
            follow_selected: false,
            editor_mode: InteractionMode::Inspect,
            editor_brush: 60.0,
            pick_radius_px: 14.0,
            show_inspector: true,
            show_shortcuts: false,
            show_hint: true,
            hint_age: 0.0,
            quality: Quality::Balanced,
            stats_hz: 10.0,
            toasts: Toasts::default(),
            dialog: None,
            gen_seed: cfg.seed,
            gen_preset_idx: 0,
            gen_width: cfg.width,
            gen_height: cfg.height,
            canvas_rect: None,
            last_hover: None,
            was_populated: false,
            warned_extinct: false,
            records_seen: HashMap::new(),
            stats_cache: None,
            hist_cache: None,
            clock: Hz10Clock::default(),
            text: StatusText::default(),
            last_frame: None,
            steps_win_start: Instant::now(),
            steps_win_steps: 0,
            sps: 0.0,
            achieved_rate: 0.0,
            rate_ring: [0.0; RATE_RING_N],
            rate_idx: 0,
            rate_filled: 0,
            frame_no: 0,

            view3d_ready,
            view3d_on: false,
            mode3d: alife_view3d::camera::CameraMode::Planet,
            cam3d: alife_view3d::camera::CameraState::new(),
            world3d_cache: None,
            world3d_tag: None,
            last_follow_pos: None,
            cam_t: 0.0,
            dt_last: 1.0 / 60.0,
        };
        app.toasts
            .push(ToastKind::Info, "World ready — press Space to run");
        app
    }

    // ------------------------------------------------------------ world mgmt

    /// Install a freshly generated world as the running simulation.
    pub fn install_new_world(
        &mut self,
        cfg: WorldGenConfig,
        params: &SimParams,
        preset: Option<String>,
        name: &str,
    ) {
        let world = alife_worldgen::generate(&cfg);
        self.sim = Simulation::new(params, world);
        self.history = HistoryBuffer::new(20, 2048);
        self.world_cfg = cfg;
        self.world_name = name.to_string();
        self.preset_name = preset.unwrap_or_default();
        self.records_seen.clear();
        self.selected = None;
        self.follow_selected = false;
        self.was_populated = false;
        self.warned_extinct = false;
        self.stats_cache = None;
        self.hist_cache = None;
        self.tree_cache = None;
        self.god_cache = None;
        self.hud_weather = None;
        self.weather_layer = None;
        self.hover_pick = None;
        self.hover_chip = (0, String::new());
        self.death_causes = [0; 3];
        self.death_seen = false;
        self.god_purge_arm = 0;
        self.running = false;
        self.acc = 0.0;
        self.gen_seed = cfg.seed;
        self.gen_width = cfg.width;
        self.gen_height = cfg.height;
        self.reset_view_for_world();
    }
    /// Restart the current world: same terrain + params, fresh population.
    pub fn reset_world(&mut self) {
        let params = self.sim.params.clone();
        let world = self.sim.world.clone();
        self.sim = Simulation::new(&params, world);
        self.history = HistoryBuffer::new(20, 2048);
        self.records_seen.clear();
        self.selected = None;
        self.follow_selected = false;
        self.was_populated = false;
        self.warned_extinct = false;
        self.stats_cache = None;
        self.hist_cache = None;
        self.tree_cache = None;
        self.god_cache = None;
        self.hud_weather = None;
        self.weather_layer = None;
        self.hover_pick = None;
        self.hover_chip = (0, String::new());
        self.death_causes = [0; 3];
        self.death_seen = false;
        self.god_purge_arm = 0;
        self.running = false;
        self.acc = 0.0;
        self.view.trails.clear();
        self.toasts.push(ToastKind::Ok, "World reset");
    }

    /// Install a simulation restored from a loaded `WorldSave`.
    pub fn apply_loaded(&mut self, ws: &WorldSave) {
        match save_io::restore_from_save(ws) {
            Ok(sim) => {
                self.sim = sim;
                self.history = ws.history.clone();
                self.world_cfg = ws.world_config;
                self.preset_name = ws.preset.clone().unwrap_or_default();
                self.world_name = ws.name.clone();
                self.records_seen.clear();
                self.selected = None;
                self.follow_selected = false;
                self.was_populated = true;
                self.warned_extinct = false;
                self.stats_cache = None;
                self.hist_cache = None;
                self.tree_cache = None;
                self.god_cache = None;
                self.hud_weather = None;
        self.weather_layer = None;
        self.hover_pick = None;
        self.hover_chip = (0, String::new());
                self.death_causes = [0; 3];
                self.death_seen = false;
                self.god_purge_arm = 0;
                self.running = false;
                self.acc = 0.0;
                self.gen_seed = ws.world_config.seed;
                self.gen_width = ws.world_config.width;
                self.gen_height = ws.world_config.height;
                self.reset_view_for_world();
                self.view.trails.clear();
                self.view_sel = AppView::World;
                self.toasts.push(
                    ToastKind::Ok,
                    format!("Loaded '{}' at step {}", ws.name, self.sim.step_count()),
                );
            }
            Err(e) => {
                self.toasts.push(ToastKind::Error, e.clone());
                alife_data::log_error(&self.store.root, &e);
            }
        }
    }

    fn reset_view_for_world(&mut self) {
        let tag = if self.world_tag == u32::MAX {
            1
        } else {
            self.world_tag + 1
        };
        self.world_tag = tag;
        self.view.invalidate_world(tag);
        self.view_sel = AppView::World;
    }

    // ------------------------------------------------------------ timing

    fn compute_dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = match self.last_frame {
            Some(t) => (now - t).as_secs_f32().min(0.25),
            None => 1.0 / 60.0,
        };
        self.last_frame = Some(now);
        dt
    }

    /// Fixed-step simulation with accumulator and a hard per-frame budget.
    /// When the requested time scale needs more than `MAX_STEPS_PER_FRAME`
    /// steps in one frame the engine throttles and `budgeted` is raised so
    /// the HUD can honestly report achieved < requested.
    fn simulate(&mut self, dt: f32) {
        let was_running = self.running;
        self.budgeted = false;
        if self.running {
            self.acc += dt * self.timescale.mult();
            let mut steps = 0u64;
            while self.acc >= DT {
                if steps >= MAX_STEPS_PER_FRAME {
                    self.budgeted = true;
                    self.acc = 0.0; // drop backlog: the budget is the honest cap
                    break;
                }
                let events = self.sim.step();
                alife_data::step_history(&mut self.history, self.sim.stats(), &events);
                self.observe_deaths(&events);
                self.acc -= DT;
                steps += 1;
            }
            self.steps_win_steps += steps;

            let pop = self.sim.stats().population;
            if pop > 0 {
                self.was_populated = true;
                self.warned_extinct = false;
            } else {
                self.was_populated = true;
            }
            if pop == 0 && self.was_populated && !self.warned_extinct {
                self.warned_extinct = true;
                self.toasts
                    .push(ToastKind::Warn, "Population collapsed — ecosystem extinct");
            }

            if self.frame_no % 15 == 0 {
                save_io::collect_records(&mut self.records_seen, &self.sim);
            }
        }
        let elapsed = self.steps_win_start.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let secs = elapsed.as_secs_f64();
            let rate = if was_running {
                self.steps_win_steps as f64 / secs.max(1e-6)
            } else {
                0.0
            };
            self.sps = rate;
            // small ring buffer: smoothed achieved rate over the last windows
            if self.rate_filled == 0 {
                self.rate_idx = 0;
            }
            self.rate_ring[self.rate_idx] = rate;
            self.rate_idx = (self.rate_idx + 1) % RATE_RING_N;
            self.rate_filled = (self.rate_filled + 1).min(RATE_RING_N);
            self.achieved_rate = if was_running {
                self.rate_ring.iter().take(self.rate_filled).sum::<f64>()
                    / self.rate_filled.max(1) as f64
            } else {
                0.0
            };
            self.steps_win_steps = 0;
            self.steps_win_start = Instant::now();
        }
    }

    /// Track death causes from observed step events (honest session-local
    /// counts; the engine diagnosis layer replaces these once it lands).
    fn observe_deaths(&mut self, events: &alife_common::StepEvents) {
        if events.deaths.is_empty() {
            return;
        }
        self.death_seen = true;
        for d in &events.deaths {
            self.death_causes[d.cause as usize] += 1;
        }
    }

    /// Achieved sim seconds per wall second (ring-smoothed measurement).
    pub fn achieved_sim_seconds_per_wall_second(&self) -> f64 {
        self.achieved_rate / 60.0
    }

    /// Achieved effective multiplier (requested multiplier when not budgeted).
    pub fn achieved_multiplier(&self) -> f64 {
        self.achieved_sim_seconds_per_wall_second()
    }

    fn hint_tick(&mut self, dt: f32) {
        if self.show_hint {
            self.hint_age += dt;
            if self.hint_age > 30.0 {
                self.show_hint = false;
            }
        }
    }

    // ------------------------------------------------- 10 Hz throttled reads

    /// One slow Instant-checked clock drives every throttled UI identity read:
    /// the weather snapshot (+ render layer), the engine intervention log and
    /// the cached toolbar/status strings. Nothing samples them per frame.
    fn tick_slow_clock(&mut self) {
        if !self.clock.due() {
            return;
        }
        let snapshot = crate::god_api::weather(&self.sim);
        self.hud_weather = snapshot;
        self.weather_layer = if snapshot.map_or(false, |w| w.idle()) {
            // keep the last layer; the canvas gates off idle snapshots itself
            self.weather_layer.take()
        } else {
            Some(alife_render::weather_layer(&self.sim))
        };
        let mut c = self.god_cache.take().unwrap_or_default();
        c.interventions = crate::god_api::interventions(&self.sim);
        c.built = Some(Instant::now());
        self.god_cache = Some(c);

        let st = self.sim.stats();
        let step = self.sim.step_count();
        let text = StatusText {
            pop: format!("pop {}", st.population),
            step: format!("step {step}"),
            short_status: format!(
                "pop {}  step {}  {:.0} steps/s",
                st.population, step, self.sps
            ),
            achieved: format!(
                "achieved x{:.2} ({:.0} sim s/s)",
                self.achieved_multiplier(),
                self.achieved_rate
            ),
            sim_clock: format!("sim clock {:.1} s", step as f32 / 60.0),
            species: format!("species {}", st.species_counts.len()),
            food: format!("food {}", st.food_count),
            cursor: self.last_hover.map(|(hx, hy)| format!("cursor {:.0}, {:.0}", hx, hy)),
            world_dim: format!(
                "world {} x {}",
                self.sim.world.width as u32,
                self.sim.world.height as u32
            ),
        };
        self.text = text;
    }

    // --------------------------------------------------------- shortcuts

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // Escape works even while a text field has focus: editor tool -> dialog -> selection.
        let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        if esc {
            if self.view3d_on && !crate::panels_3d::pop_level(self) {
                return;
            }
            if self.god_tool.is_some() {
                self.god_tool = None;
                self.toasts.push(ToastKind::Info, "God target mode cancelled");
                return;
            }
            if self.editor_mode != InteractionMode::Inspect {
                self.editor_mode = InteractionMode::Inspect;
                self.toasts.push(ToastKind::Info, "Editor tool cancelled");
                return;
            }
            if self.dialog.is_some() {
                self.dialog = None;
                return;
            }
            if self.selected.is_some() {
                self.selected = None;
            }
        }

        if ctx.wants_keyboard_input() || self.dialog.is_some() {
            return;
        }

        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.running = !self.running;
            }
            if i.key_pressed(egui::Key::ArrowRight) && !self.running {
                self.single_step();
            }
            if i.key_pressed(egui::Key::K) {
                self.view_sel = if self.view_sel == AppView::GodMode {
                    AppView::World
                } else {
                    AppView::GodMode
                };
            }
            if i.key_pressed(egui::Key::T) {
                self.view_sel = if self.view_sel == AppView::Tree {
                    AppView::World
                } else {
                    AppView::Tree
                };
            }
            if i.key_pressed(egui::Key::F) {
                self.view.fit(&self.sim.world);
            }
            if i.key_pressed(egui::Key::Num3) {
                crate::panels_3d::toggle(self);
            }
            if i.key_pressed(egui::Key::Enter) {
                crate::panels_3d::descend_to_selected(self);
            }
            if i.key_pressed(egui::Key::Equals) {
                let center = self.view.viewport().center();
                self.view.zoom_at_screen(center, 1.25);
            }
            if i.key_pressed(egui::Key::Minus) {
                let center = self.view.viewport().center();
                self.view.zoom_at_screen(center, 0.8);
            }
            if i.key_pressed(egui::Key::G) {
                self.view.show_grid = !self.view.show_grid;
            }
            if i.key_pressed(egui::Key::I) {
                self.show_inspector = !self.show_inspector;
            }
            if i.key_pressed(egui::Key::Questionmark) {
                self.show_shortcuts = !self.show_shortcuts;
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::N) {
                self.dialog = Some(Dialog::new_world_scratch(self));
            }
            if i.key_pressed(egui::Key::S) {
                self.dialog = Some(Dialog::save_scratch(self));
            }
            if i.key_pressed(egui::Key::R) {
                self.dialog = Some(Dialog::ConfirmReset);
            }
            if i.key_pressed(egui::Key::Z) && i.modifiers.ctrl {
                if i.modifiers.shift {
                    self.editor_redo();
                } else {
                    self.editor_undo();
                }
            }
            if i.key_pressed(egui::Key::Y) && i.modifiers.ctrl {
                self.editor_redo();
            }
        });
    }

    pub fn single_step(&mut self) {
        if self.running {
            return;
        }
        let events = self.sim.step();
        alife_data::step_history(&mut self.history, self.sim.stats(), &events);
        self.steps_win_steps += 1;
    }

    pub fn editor_undo(&mut self) {
        if !self.sim.undo_edit() {
            self.toasts.push(ToastKind::Info, "Nothing to undo");
        }
    }

    pub fn editor_redo(&mut self) {
        if !self.sim.redo_edit() {
            self.toasts.push(ToastKind::Info, "Nothing to redo");
        }
    }

    /// Arc-shared world snapshot for the 3D renderer. Rebuilt only when the
    /// world was regenerated (renderer binds it by shareable ref every frame
    /// and must NOT clone 1.4 MB of fields per frame).
    pub fn arc_world(&mut self) -> Arc<alife_worldgen::WorldModel> {
        let tag = self.sim.world.generation_tag;
        if self.world3d_tag != Some(tag) || self.world3d_cache.is_none() {
            self.world3d_cache = Some(Arc::new(self.sim.world.clone()));
            self.world3d_tag = Some(tag);
        }
        self.world3d_cache
            .as_ref()
            .map(Arc::clone)
            .expect("world cache freshly built")
    }

    /// Read-only view for panels that need no refresh logic.
    pub fn sim_world_ref3d(&self) -> Arc<alife_worldgen::WorldModel> {
        self.world3d_cache
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| Arc::new(self.sim.world.clone()))
    }

    // ---------------------------------------------------------- shell

    fn shell(&mut self, ctx: &egui::Context) {
        self.top_toolbar(ctx);
        self.nav_rail(ctx);
        self.status_strip(ctx);
    }

    fn toolbar_frame() -> egui::Frame {
        egui::Frame::none()
            .fill(theme::BG_PANEL)
            .inner_margin(egui::Margin::symmetric(10.0, 5.0))
            .stroke(egui::Stroke::new(0.5_f32, theme::STROKE_SOFT))
    }
}

// ---------------------------------------------------------------- app loop

impl eframe::App for AlifeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_no = self.frame_no.wrapping_add(1);
        let dt = self.compute_dt();
        self.toasts.update(dt);

        self.handle_shortcuts(ctx);
        self.simulate(dt);
        self.hint_tick(dt);
        self.tick_slow_clock();
        self.shell(ctx);

        match self.view_sel {
            AppView::World => panels_world::world_view(self, ctx),
            AppView::Statistics => crate::panels_stats::statistics_view(self, ctx),
            AppView::Evolution => crate::panels_stats::evolution_view(self, ctx),
            AppView::Tree => crate::panels_tree::tree_view(self, ctx),
            AppView::Experiments => panels_experiments::experiments_view(self, ctx),
            AppView::GodMode => crate::panels_god::god_view(self, ctx),
            AppView::WorldEditor => panels_editor::editor_view(self, ctx),
            AppView::Settings => panels_screens::settings_view(self, ctx),
        }

        self.inspector_panel(ctx);
        dialogs::show(self, ctx);
        self.toasts.show(ctx);
        if !self.tutorial.active {
            panels_world::hint_overlay(self, ctx);
        }
        panels_tutorial::show(self, ctx);
        panels_screens::shortcuts_sheet(self, ctx);

        if self.running {
            ctx.request_repaint_after(Duration::from_millis(16));
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

// ---------------------------------------------------------------- widgets

impl AlifeApp {
    // --------------------------------------------------------- top toolbar

    fn top_toolbar(&mut self, ctx: &egui::Context) {
        let frame = Self::toolbar_frame();
        egui::TopBottomPanel::top("alife.toolbar")
            .exact_height(34.0)
            .frame(frame)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(crate::widgets::strong(&self.world_name));
                    if !self.preset_name.is_empty() {
                        ui.label(crate::widgets::mono(
                            format!("preset: {}", self.preset_name),
                            theme::WEAK,
                        ));
                    }
                    ui.separator();

                    let play_label = if self.running { "Pause" } else { "Run" };
                    let tip = if self.running {
                        "Pause the simulation (Space)"
                    } else {
                        "Run the simulation (Space)"
                    };
                    if crate::widgets::mini_btn(ui, play_label, tip).clicked() {
                        self.running = !self.running;
                    }
                    let step_resp =
                        ui.add_enabled(!self.running, egui::Button::new("Step").small());
                    if step_resp.clicked() {
                        self.single_step();
                    }
                    step_resp.on_hover_text("Advance one step (Right Arrow; paused only)");

                    ui.label(crate::widgets::weak("time scale"));
                    for ts in TimeScale::ALL {
                        let sel = self.timescale == ts;
                        let mut tip = format!(
                            "{}x — {} sim seconds per real second",
                            ts.mult(),
                            ts.mult()
                        );
                        if let Some(note) = ts.note() {
                            tip = format!("{tip}. {note}");
                        }
                        if crate::widgets::select_btn(ui, ts.label(), sel, &tip).clicked() {
                            self.timescale = ts;
                        }
                    }
                    if self.running {
                        ui.label(crate::widgets::mono(&self.text.achieved, theme::WEAK))
                            .on_hover_text("Measured sim seconds per wall second (ring-smoothed)");
                        if self.budgeted {
                            let b = ui.label(crate::widgets::mono("budgeted", theme::WARN));
                            b.on_hover_text(format!(
                                "Requested {}x exceeds the per-frame step budget \
                                 ({} steps/frame): achieved sim time per real second is \
                                 lower than requested — the engine throttles honestly",
                                self.timescale.mult(),
                                MAX_STEPS_PER_FRAME
                            ));
                        }
                    }

                    if crate::widgets::mini_btn(ui, "Reset", "Restart this world (R)").clicked() {
                        self.dialog = Some(Dialog::ConfirmReset);
                    }
                    if crate::widgets::mini_btn(ui, "New World", "Create a new world (Ctrl+N)")
                        .clicked()
                    {
                        self.dialog = Some(Dialog::new_world_scratch(self));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if crate::widgets::icon_btn(ui, "?", "Shortcut list (?)").clicked() {
                            self.show_shortcuts = !self.show_shortcuts;
                        }
                        if crate::widgets::mini_btn(ui, "Load", "Load a saved world").clicked() {
                            self.dialog = Some(Dialog::Load { confirm_delete: None });
                        }
                        if crate::widgets::mini_btn(ui, "Save", "Save this world (S)").clicked() {
                            self.dialog = Some(Dialog::save_scratch(self));
                        }
                        ui.separator();
                        // cached strings from the 10 Hz clock — no format per frame
                        ui.label(crate::widgets::mono(&self.text.pop, theme::TEXT));
                        ui.label(crate::widgets::mono(&self.text.step, theme::TEXT));
                    });
                });
            });
    }

    // ----------------------------------------------------------- nav rail

    fn nav_rail(&mut self, ctx: &egui::Context) {
        let narrow = ctx.screen_rect().width() < 1200.0;
        let width = if narrow { 44.0 } else { 120.0 };
        egui::SidePanel::left("alife.nav")
            .exact_width(width)
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(6.0)),
            )
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for v in AppView::ALL {
                    let (label, tip) = if narrow {
                        (v.short(), format!("Switch to the {} view", v.label()))
                    } else {
                        (v.label(), format!("Switch to the {} view", v.label()))
                    };
                    let sel = self.view_sel == v;
                    if crate::widgets::select_btn(ui, label, sel, &tip).clicked() {
                        self.view_sel = v;
                    }
                }
            });
    }

    // -------------------------------------------------------- status strip

    fn status_strip(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("alife.status")
            .exact_height(22.0)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::symmetric(10.0, 3.0)),
            )
            .show(ctx, |ui| {
                let w = ctx.screen_rect().width();
                // all strings come from the 10 Hz clock cache
                if w < 1000.0 {
                    ui.label(crate::widgets::mono(&self.text.short_status, theme::TEXT));
                } else {
                    ui.label(crate::widgets::mono(&self.text.achieved, if self.running { theme::OK } else { theme::WEAK }))
                    .on_hover_text("Measured sim time per wall second; a budgeted badge means the per-frame step cap throttled the engine");
                    if self.running && self.budgeted {
                        ui.label(crate::widgets::mono(
                            format!("budgeted (< {} steps/frame cap)", MAX_STEPS_PER_FRAME),
                            theme::WARN,
                        ));
                    }
                    ui.label(crate::widgets::mono(&self.text.sim_clock, theme::TEXT));
                    ui.label(crate::widgets::mono(&self.text.species, theme::TEXT));
                    ui.label(crate::widgets::mono(&self.text.food, theme::TEXT));
                    if let Some(cursor) = &self.text.cursor {
                        ui.label(crate::widgets::mono(cursor, theme::WEAK));
                    }
                    ui.label(crate::widgets::mono(&self.text.world_dim, theme::WEAK));
                }
            });
    }

    // ---------------------------------------------------- inspector panel

    fn inspector_panel(&mut self, ctx: &egui::Context) {
        if self.view_sel != AppView::World || !self.show_inspector {
            return;
        }
        if ctx.screen_rect().width() < 1200.0 {
            return; // auto-collapsed on narrow windows
        }
        egui::SidePanel::right("alife.inspector")
            .resizable(true)
            .default_width(320.0)
            .min_width(240.0)
            .max_width(460.0)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(8.0)),
            )
            .show(ctx, |ui| {
                panels_inspector::inspector(self, ui);
            });
    }
}

/// Native window bootstrap: 1500x940 (min 1000x640), dark theme applied to the
/// egui visuals (the eframe clear color follows `visuals.window_fill` ==
/// #0a0d11 from theme.rs).
pub fn run_app(data_dir: std::path::PathBuf) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 940.0])
            .with_min_inner_size([1000.0, 640.0]),
        // wgpu backend: required for the 3D living-world view; keeps the
        // whole app on one graphics stack
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 32,
        stencil_buffer: 0,
        ..Default::default()
    };
    eframe::run_native(
        "Artificial Life",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(AlifeApp::new(cc, data_dir)) as Box<dyn eframe::App>)
        }),
    )
}

