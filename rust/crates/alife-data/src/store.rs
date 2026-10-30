//! File-backed `DataStore`: directory layout `saves/`, `experiments/`,
//! `exports/`, `logs/` under `root`, created on demand. Saves are written
//! atomically (temp file + rename). Save/load failures are best-effort logged.

use crate::log::log_error;
use alife_common::persist::{self, WorldSave};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Save format versions accepted for reading. v1 is read forever (lab files
/// must not rot); v2 adds the optional engine `core` blob (SaveCore JSON).
/// Anything above `MAX_READ_VERSION` comes from a newer app and is rejected.
pub const MAX_READ_VERSION: u64 = 2;
/// Save format version written by this crate (see alife-data::SAVE_FORMAT_VERSION).
const WRITE_VERSION: u64 = crate::SAVE_FORMAT_VERSION as u64;
/// `created_recently_hint` in [`SaveSummary`]: modified within this window.
const RECENT_WINDOW: Duration = Duration::from_secs(10 * 60);

/// Cheap UI card for a save file, derived from header-level JSON fields only
/// (no organism details are copied into the summary).
#[derive(Clone, Debug)]
pub struct SaveSummary {
    /// Save name (file stem, without `.json`).
    pub name: String,
    /// Preset id the world was created with, if recorded.
    pub preset: Option<String>,
    /// Simulated steps at last save.
    pub step_count: u64,
    /// Organism count (array length; no organism details).
    pub population: u32,
    /// Average generation: taken from the newest history sample when
    /// available (header-ish read), else 0.0.
    pub avg_generation: f32,
    /// Number of history samples stored with the save.
    pub sample_count: usize,
    /// True when the file was modified within the last 10 minutes.
    pub created_recently_hint: bool,
}

pub struct DataStore {
    pub root: PathBuf,
}

impl DataStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    // ---------------------------------------------------------- directories

    fn ensure_dir(&self, dir: &Path, what: &str) -> Result<(), String> {
        if dir.is_dir() {
            return Ok(());
        }
        fs::create_dir_all(dir).map_err(|e| {
            format!("cannot create {what} directory '{}': {e} ({})", dir.display(), io_kind(&e))
        })
    }

    fn saves_dir(&self) -> Result<PathBuf, String> {
        let dir = self.root.join("saves");
        self.ensure_dir(&dir, "saves")?;
        Ok(dir)
    }

    /// Directory for experiment state snapshots (created on demand).
    pub fn experiment_dir(&self) -> PathBuf {
        let dir = self.root.join("experiments");
        let _ = fs::create_dir_all(&dir);
        dir
    }

    /// Directory for CSV/JSON exports (created on demand).
    pub fn export_path(&self) -> PathBuf {
        let dir = self.root.join("exports");
        let _ = fs::create_dir_all(&dir);
        dir
    }

    // --------------------------------------------------------- name guards

    fn check_name(name: &str) -> Result<&str, String> {
        let n = name.trim();
        if n.is_empty() {
            return Err("save name is empty".into());
        }
        if n == "." || n == ".." {
            return Err(format!("invalid save name '{n}'"));
        }
        if n.contains('/') || n.contains('\\') || n.contains(':') {
            return Err(format!("invalid save name '{n}' — path separators are not allowed"));
        }
        Ok(n)
    }

    fn save_path(dir: &Path, name: &str) -> PathBuf {
        dir.join(format!("{name}.json"))
    }

    // ----------------------------------------------------------------- save

    /// Serialize pretty once (deterministic field order via serde derive),
    /// write to a temp file in the same directory, then rename — atomic.
    /// The document is streamed straight into the file (`to_writer_pretty`,
    /// byte-identical to `to_string_pretty`): no full-save-sized String
    /// buffer is ever materialized in memory.
    fn write_atomic(dir: &Path, name: &str, ws: &WorldSave) -> Result<(), String> {
        let dest = Self::save_path(dir, name);
        let tmp = dir.join(format!("{name}.json.tmp-{}", std::process::id()));
        let write_isolated = |tmp: &Path| -> Result<(), String> {
            let mut f = std::io::BufWriter::new(fs::File::create(tmp).map_err(|e| {
                format!("cannot create temp file for '{name}': {e} ({})", io_kind(&e))
            })?);
            serde_json::to_writer_pretty(&mut f, ws)
                .map_err(|e| format!("cannot serialize save '{name}': {e}"))?;
            f.flush()
                .and_then(|_| f.get_ref().sync_all())
                .map_err(|e| format!("cannot write save '{name}': {e} ({})", io_kind(&e)))
        };
        if let Err(e) = write_isolated(&tmp) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        // Modern std rename replaces an existing destination on Windows
        // (MOVEFILE_REPLACE_EXISTING); fall back to remove+rename if it fails.
        if fs::rename(&tmp, &dest).is_err() {
            let _ = fs::remove_file(&dest);
            if fs::rename(&tmp, &dest).is_err() {
                let _ = fs::remove_file(&tmp);
                return Err(format!(
                    "cannot install save '{}': rename failed (disk full or read-only?)",
                    dest.display()
                ));
            }
        }
        Ok(())
    }

    /// Save atomically (tmp + rename). Never silently overwrites: caller
    /// checks `list` first (the UI does conflict-checking).
    pub fn save_world(&self, ws: &WorldSave) -> Result<(), String> {
        if let Err(e) = self.save_impl(ws, false) {
            let msg = format!("save '{}': {e}", ws.name);
            log_error(&self.root, &msg);
            return Err(e);
        }
        Ok(())
    }

    /// Overwrite-capable variant of `save_world` (explicitly requested by the
    /// UI agent); replaces any existing file with the same name.
    pub fn overwrite_world(&self, ws: &WorldSave) -> Result<(), String> {
        self.save_impl(ws, true).map_err(|e| {
            log_error(&self.root, &format!("overwrite save '{}': {e}", ws.name));
            e
        })
    }

    fn save_impl(&self, ws: &WorldSave, allow_overwrite: bool) -> Result<(), String> {
        let name = Self::check_name(&ws.name)?;
        let dir = self.saves_dir()?;
        let path = Self::save_path(&dir, name);
        if !allow_overwrite && path.exists() {
            return Err(format!(
                "save '{name}' already exists in '{}' — delete it first or call overwrite_world",
                dir.display()
            ));
        }
        DataStore::write_atomic(&dir, name, ws)
    }

    // ----------------------------------------------------------------- load

    /// Load a save; validates format id and version before full parsing.
    pub fn load_world(&self, name: &str) -> Result<WorldSave, String> {
        let fail = |msg: String| -> String {
            log_error(&self.root, &msg);
            msg
        };
        let name = Self::check_name(name).map_err(fail)?;
        let path = match self.saves_dir() {
            Ok(d) => Self::save_path(&d, name),
            Err(e) => return Err(fail(e)),
        };
        let raw = fs::read_to_string(&path).map_err(|e| {
            fail(format!("cannot read save '{}': {e} ({})", path.display(), io_kind(&e)))
        })?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| fail(format!("corrupt JSON in '{}': {e}", path.display())))?;

        let format_id = value.get("format_id").and_then(Value::as_str).unwrap_or("");
        if format_id != persist::FORMAT_ID {
            return Err(fail(format!(
                "'{}' is not an alife save (format id '{}' != '{}')",
                path.display(),
                if format_id.is_empty() { "<missing>" } else { format_id },
                persist::FORMAT_ID
            )));
        }
        let version = value.get("version").and_then(Value::as_u64).unwrap_or(0);
        match version {
            1 | MAX_READ_VERSION => {}
            0 => {
                return Err(fail(format!(
                    "unsupported format version 0 in '{}' — expected 1 or {} (missing version field?)",
                    path.display(),
                    WRITE_VERSION
                )))
            }
            v => {
                return Err(fail(format!(
                    "unknown format version {v} in '{}' — file created by a newer app (supported: 1..={MAX_READ_VERSION})",
                    path.display()
                )))
            }
        }
        let ws: WorldSave = serde_json::from_value(value).map_err(|e| {
            fail(format!("save '{}' is corrupt (version {version}): {e}", path.display()))
        })?;
        Ok(ws)
    }

    // ----------------------------------------------------------------- list

    /// Save file names (with extension) in `saves/`, sorted by modified time desc.
    pub fn list(&self) -> Vec<String> {
        let mut out: Vec<(String, SystemTime)> = Vec::new();
        if let Ok(entries) = fs::read_dir(self.root.join("saves")) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(os_name) = path.file_name().and_then(|n| n.to_str()) else { continue };
                if !os_name.ends_with(".json") || !path.is_file() {
                    continue;
                }
                let modified = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                out.push((os_name.to_string(), modified));
            }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
        out.into_iter().map(|(name, _)| name).collect()
    }

    // --------------------------------------------------------------- delete

    pub fn delete(&self, name: &str) -> Result<(), String> {
        self.delete_impl(name).map_err(|e| {
            log_error(&self.root, &format!("delete '{name}': {e}"));
            e
        })
    }

    fn delete_impl(&self, name: &str) -> Result<(), String> {
        let name = Self::check_name(name)?;
        let path = Self::save_path(&self.saves_dir()?, name);
        if !path.exists() {
            return Err(format!("save '{name}' not found in '{}'", self.root.display()));
        }
        fs::remove_file(&path)
            .map_err(|e| format!("cannot delete '{}': {e} ({})", path.display(), io_kind(&e)))
    }

    // ------------------------------------------------------- experiment branches

    /// Branch filename convention: `<parent>@<suffix>.json` inside `saves/`.
    /// The branch's "parent" is everything before the first `@`.
    fn split_branch(file_stem: &str) -> Option<(String, String)> {
        let at = file_stem.find('@')?;
        if at == 0 {
            return None; // '@suffix' has no parent name
        }
        let parent = file_stem[..at].to_string();
        Some((file_stem.to_string(), parent))
    }

    /// Copy a save to a new experiment branch: load, rename, save. Never
    /// overwrites silently — errors when `new_name` exists. Branch names
    /// following the `<parent>@<suffix>.json` convention are recognized by
    /// `branch_list`, but any valid save name is accepted.
    pub fn branch_world(&self, source_name: &str, new_name: &str) -> Result<(), String> {
        self.branch_impl(source_name, new_name).map_err(|e| {
            log_error(&self.root, &format!("branch '{source_name}' -> '{new_name}': {e}"));
            e
        })
    }

    fn branch_impl(&self, source_name: &str, new_name: &str) -> Result<(), String> {
        let mut ws = self.load_world(source_name)?;
        let new_name = Self::check_name(new_name)?;
        if source_name.trim() == new_name {
            return Err(format!(
                "branch target '{new_name}' equals the source save — branching would be a no-op"
            ));
        }
        let dir = self.saves_dir()?;
        if Self::save_path(&dir, new_name).exists() {
            return Err(format!(
                "branch target '{new_name}' already exists in '{}' — pick a new name or delete the file first",
                dir.display()
            ));
        }
        ws.name = new_name.to_string();
        self.save_world(&ws).map_err(|e| {
            format!(
                "branching '{source_name}': the source loaded fine but writing '{new_name}' failed: {e}"
            )
        })
    }

    /// All saves that follow the `<parent>@<suffix>.json` branch convention:
    /// `(branch_stem, parent_stem)`, sorted by branch name. Plain saves with
    /// no `@` are not branches and are not listed here — `list` still returns
    /// every `.json` file unchanged (sorted by modified time, newest first).
    pub fn branch_list(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Ok(entries) = fs::read_dir(self.saves_dir().unwrap_or_else(|_| self.root.clone())) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(os_name) = path.file_name().and_then(|n| n.to_str()) else { continue };
                if !os_name.ends_with(".json") || !path.is_file() {
                    continue;
                }
                let stem = os_name.trim_end_matches(".json");
                if let Some((branch, parent)) = Self::split_branch(stem) {
                    out.push((branch, parent));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Read a save's header fields only (JSON pointer-style reads; organism
    /// details are never copied into the summary) to build a cheap UI card.
    pub fn describe_save(&self, name: &str) -> Result<SaveSummary, String> {
        self.describe_impl(name).map_err(|e| {
            log_error(&self.root, &format!("describe '{name}': {e}"));
            e
        })
    }

    fn describe_impl(&self, name: &str) -> Result<SaveSummary, String> {
        let name = Self::check_name(name)?;
        let path = Self::save_path(&self.saves_dir()?, name);
        let raw = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read save '{}': {e} ({})", path.display(), io_kind(&e)))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("corrupt JSON in '{}': {e}", path.display()))?;

        let step_count = value.get("step_count").and_then(Value::as_u64).unwrap_or(0);
        let preset = value
            .get("preset")
            .and_then(Value::as_str)
            .map(|p| p.to_string());
        let population = value
            .get("organisms")
            .and_then(Value::as_array)
            .map(|a| a.len() as u32)
            .unwrap_or(0);

        // Header-ish generation figure: newest history sample if any.
        let avg_generation = value
            .get("history")
            .and_then(|h| h.get("samples"))
            .and_then(Value::as_array)
            .and_then(|samples| samples.last())
            .and_then(|last| last.get("avg_generation"))
            .and_then(Value::as_f64)
            .map(|g| g as f32)
            .unwrap_or(0.0);
        let sample_count = value
            .get("history")
            .and_then(|h| h.get("samples"))
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);

        let created_recently_hint = fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .map(|elapsed| elapsed < RECENT_WINDOW)
            .unwrap_or(false);

        Ok(SaveSummary {
            name: name.to_string(),
            preset,
            step_count,
            population,
            avg_generation: alife_common::sanitize(avg_generation),
            sample_count,
            created_recently_hint,
        })
    }

    // ----------------------------------------------------------------- export

    /// CSV with one row per history sample. `wet_cover` is an extra trailing
    /// column so older files/imports with the previous header still load.
    /// step,population,births,deaths,avg_energy,avg_age,avg_generation,food_count,diversity,wet_cover
    pub fn export_history_csv(&self, ws: &WorldSave) -> Result<String, String> {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(
            96 * (ws.history.samples.len().saturating_add(1)),
        );
        out.push_str(
            "step,population,births,deaths,avg_energy,avg_age,avg_generation,food_count,diversity,wet_cover\n",
        );
        for s in &ws.history.samples {
            let _ = write!(
                out,
                "{},{},{},{},{:.2},{:.2},{:.2},{},{:.3},{:.3}\n",
                s.step,
                s.population,
                s.births,
                s.deaths,
                s.avg_energy,
                s.avg_age,
                s.avg_generation,
                s.food_count,
                s.diversity,
                s.wet_cover,
            );
        }
        Ok(out)
    }
}

fn io_kind(e: &std::io::Error) -> &'static str {
    match e.kind() {
        std::io::ErrorKind::NotFound => "not found",
        std::io::ErrorKind::PermissionDenied => "permission denied",
        _ => "io error",
    }
}
