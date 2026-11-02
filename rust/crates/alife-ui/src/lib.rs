//! Artificial Life — native desktop application shell (eframe/egui 0.29).
//!
//! Modules:
//! - [`app`]              — application state, frame loop, shortcuts, shell panels
//! - [`panels_world`]     — world canvas, overlays, minimap, hint screen
//! - [`panels_stats`]     — Statistics + Evolution views (charts, histograms)
//! - [`panels_tree`]      — evolutionary tree view (lineage chains)
//! - [`panels_god`]       — God Mode view + weather HUD + armed-target plumbing
//! - [`god_api`]          — local glue trait between UI and the engine god API
//! - [`panels_experiments`]— live parameters, preset launcher, exports
//! - [`panels_editor`]    — world editor tools + regenerate controls
//! - [`panels_inspector`] — organism inspector with neural diagram
//! - [`panels_screens`]   — settings / about / shortcuts sheet
//! - [`panels_tutorial`]  — first-run walkthrough (4 steps, done marker file)
//! - [`dialogs`]          — modal dialogs (new world, save, load, confirm reset)

mod app;
mod dialogs;
mod god_api;
mod panels_3d;
mod panels_brain;
mod panels_editor;
mod panels_experiments;
mod panels_god;
mod panels_governor;
mod panels_inspector;
mod panels_screens;
mod panels_stats;
mod panels_tree;
mod panels_tutorial;
mod panels_world;

// completed subsystems from prior agents (API kept as-is)
mod save_io;
mod theme;
mod widgets;

pub use app::{run_app, AlifeApp, AppView, Quality, TimeScale};
