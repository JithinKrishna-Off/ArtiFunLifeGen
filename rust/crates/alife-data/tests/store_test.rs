//! DataStore + WorldSave round-trip, save-format v1/v2, experiment branching,
//! save summaries and CSV export tests (tmpdir-backed).

use alife_common::{
    persist::{HistoryBuffer, HistorySample, WorldSave},
    EnvironmentEdits, SimParams, WorldGenConfig,
};
use alife_data::{DataStore, SaveSummary, build_save, build_save_v2, core_blob};

fn sample_save(name: &str) -> WorldSave {
    let mut history = HistoryBuffer::new(20, 2048);
    history.samples.push(HistorySample {
        step: 20,
        population: 100,
        births: 7,
        deaths: 3,
        avg_energy: 51.5,
        avg_age: 12.0,
        avg_generation: 1.5,
        food_count: 200,
        species_top: vec![(5, 40), (2, 35)],
        diversity: 0.75,
        wet_cover: 0.25,
        merged: false,
    });
    history.interval = 20;
    build_save(
        name,
        Some("genesis".to_string()),
        &SimParams::default(),
        &WorldGenConfig::default(),
        42,
        &[],
        &[],
        &EnvironmentEdits::default(),
        9001,
        Vec::new(),
        &history,
    )
}

#[test]
fn save_load_delete_list_conflict() {
    let dir = std::env::temp_dir().join(format!("alife-data-test-{}", std::process::id()));
    let store = DataStore::new(&dir);
    let ws = sample_save("alpha");

    // missing root dirs are created on demand
    store.save_world(&ws).unwrap();
    // second save with the same name is a conflict; overwrite flag passes
    assert!(store.save_world(&ws).is_err());
    store.overwrite_world(&ws).unwrap();

    // clean round-trip
    let loaded = store.load_world("alpha").unwrap();
    assert_eq!(loaded.name, "alpha");
    assert_eq!(loaded.step_count, 42);
    assert_eq!(loaded.rng_state, 9001);
    assert_eq!(loaded.params.seed, 1337);
    assert_eq!(loaded.history.samples.len(), 1);
    assert_eq!(loaded.history.samples[0].species_top, vec![(5, 40), (2, 35)]);

    // corrupt JSON is descriptive, never a panic
    std::fs::write(store.root.join("saves").join("junk.json"), "{not valid json").unwrap();
    assert!(store.load_world("junk").unwrap_err().contains("corrupt JSON"));

    // listing only .json, delete misses then hits
    let names = store.list();
    assert_eq!(names.len(), 2);
    // junk.json was written after alpha.json, so it is newest (sorted desc)
    assert_eq!(names[0], "junk.json");
    assert!(store.delete("beta").is_err());
    store.delete("alpha").unwrap();
    assert!(store.load_world("alpha").is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn csv_export() {
    let store = DataStore::new("unused-root");
    let csv = store.export_history_csv(&sample_save("x")).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next().unwrap(),
        // wet_cover is an extra trailing column; old files/imports still load
        "step,population,births,deaths,avg_energy,avg_age,avg_generation,food_count,diversity,wet_cover"
    );
    assert_eq!(lines.next().unwrap(), "20,100,7,3,51.50,12.00,1.50,200,0.750,0.250");
    assert_eq!(lines.next(), None);
}

#[test]
fn name_guards() {
    let store = DataStore::new(std::env::temp_dir());
    assert!(store.load_world("").is_err());
    assert!(store.load_world("a/b").is_err());
    assert!(store.load_world("no-such-save-ever").is_err());
}

// ---------------------------------------------------------------- v1 / v2

#[test]
fn v1_save_file_still_loads() {
    let dir = std::env::temp_dir().join(format!("alife-data-v1-{}", std::process::id()));
    let store = DataStore::new(&dir);

    // hand-build a genuine v1 file: no `core`, version 1
    let mut ws = sample_save("ancient");
    ws.version = 1;
    ws.core = None;
    let mut raw = serde_json::to_string_pretty(&ws).unwrap();
    // v1 files predate the field entirely: surgically remove the `"core"`
    // line (it is the last struct field) and the trailing comma before it
    let without_core: Vec<&str> = raw.lines().filter(|l| !l.contains("\"core\"")).collect();
    assert_eq!(raw.lines().count() - without_core.len(), 1, "core field location changed");
    let mut restored: Vec<String> = without_core.into_iter().map(String::from).collect();
    // after dropping the core line the previous field becomes the last
    // field: strip its trailing comma (skip the closing `}` brace lines)
    for line in restored.iter_mut().rev() {
        let t = line.as_str();
        if t.trim_end() == "}" || t.trim_end() == "}," {
            continue;
        }
        let trimmed = t.trim_end();
        *line = trimmed
            .strip_suffix(',')
            .expect("a trailing comma precedes the core field")
            .to_string();
        break;
    }
    raw = restored.join("\n");
    std::fs::create_dir_all(store.root.join("saves")).unwrap();
    std::fs::write(store.root.join("saves").join("ancient.json"), &raw).unwrap();
    assert!(!raw.contains("\"core\""), "v1 fixture must not carry a core blob");
    assert!(raw.contains("\"version\": 1"));

    let loaded = store.load_world("ancient").unwrap();
    assert_eq!(loaded.version, 1);
    assert!(core_blob(&loaded).is_none());
    assert_eq!(loaded.step_count, 42);

    // unknown version (3 > supported 2) is a descriptive error
    let mut ws3 = sample_save("from-future");
    ws3.version = 3;
    store.overwrite_world(&ws3).unwrap();
    let err = store.load_world("from-future").unwrap_err();
    assert!(err.contains("unknown format version 3"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_roundtrip_preserves_core_blob() {
    let dir = std::env::temp_dir().join(format!("alife-data-v2-{}", std::process::id()));
    let store = DataStore::new(&dir);

    let core = serde_json::json!({
        "engine": "alife-core",
        "weather": { "wet": [[1.0f32, 2.5]] },
        "interventions": [
            { "step": 42, "event": { "Rain": { "x": 10.0, "y": 20.0, "r": 80.0, "duration_steps": 300 } }, "note": "+rain" }
        ]
    });
    let ws = build_save_v2(
        "cored",
        Some("genesis".to_string()),
        &SimParams::default(),
        &WorldGenConfig::default(),
        99,
        &[],
        &[],
        &EnvironmentEdits::default(),
        7,
        Vec::new(),
        &HistoryBuffer::new(20, 2048),
        &core,
    );
    assert_eq!(ws.version, 2);
    assert!(ws.core.is_some());
    store.save_world(&ws).unwrap();

    let loaded = store.load_world("cored").unwrap();
    assert_eq!(loaded.version, 2);
    let blob = core_blob(&loaded).unwrap();
    // byte-ish-equal: canonical JSON of the blob matches the original value
    let a = serde_json::to_string(blob).unwrap();
    let b = serde_json::to_string(&core).unwrap();
    assert_eq!(a, b);
    assert_eq!(blob["weather"]["wet"][0][0], 1.0);
    assert_eq!(blob["interventions"][0]["note"], "+rain");

    // non-serializable core degrades gracefully: still a save, still loads
    struct Evil;
    impl serde::Serialize for Evil {
        fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("nope"))
        }
    }
    let bad = build_save_v2(
        "degraded",
        None,
        &SimParams::default(),
        &WorldGenConfig::default(),
        0,
        &[],
        &[],
        &EnvironmentEdits::default(),
        0,
        Vec::new(),
        &HistoryBuffer::new(20, 2048),
        &Evil,
    );
    assert!(bad.core.is_none());
    assert_eq!(bad.version, 2);
    store.save_world(&bad).unwrap();
    assert!(store.load_world("degraded").is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------- branches

#[test]
fn branch_world_loads_renames_and_rejects_overwrite() {
    let dir = std::env::temp_dir().join(format!("alife-data-branch-{}", std::process::id()));
    let store = DataStore::new(&dir);
    store.save_world(&sample_save("genesis")).unwrap();

    // happy path: copy + rename
    store.branch_world("genesis", "genesis@ab").unwrap();
    let ws = store.load_world("genesis@ab").unwrap();
    assert_eq!(ws.name, "genesis@ab");
    assert_eq!(ws.step_count, 42);
    // original untouched
    assert_eq!(store.load_world("genesis").unwrap().name, "genesis");

    // never overwrites silently
    let err = store.branch_world("genesis", "genesis@ab").unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    // the branch file was not clobbered
    assert_eq!(store.load_world("genesis@ab").unwrap().step_count, 42);

    // branching onto the source's own name is a no-op trap
    assert!(store.branch_world("genesis", "genesis").is_err());

    // missing source
    let err = store.branch_world("ghost", "ghost@x").unwrap_err();
    assert!(err.contains("cannot read save"), "{err}");

    // branch files that follow the <parent>@<suffix>.json convention group
    store.branch_world("genesis", "genesis@b2").unwrap();
    store.save_world(&sample_save("no-parent")).unwrap();
    assert!(!store.root.join("saves").join("no-parent.json").is_empty());
    let mut branches = store.branch_list();
    branches.sort();
    assert_eq!(
        branches,
        vec![
            ("genesis@ab".to_string(), "genesis".to_string()),
            ("genesis@b2".to_string(), "genesis".to_string()),
        ]
    );
    // list() is unchanged: still every .json file (sorted by mtime desc)
    assert_eq!(store.list().len(), 4);

    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------- summaries

#[test]
fn describe_save_builds_a_card_from_header_fields() {
    let dir = std::env::temp_dir().join(format!("alife-data-desc-{}", std::process::id()));
    let store = DataStore::new(&dir);

    let ws = sample_save("card");
    store.save_world(&ws).unwrap();
    let s: SaveSummary = store.describe_save("card").unwrap();
    assert_eq!(s.name, "card");
    assert_eq!(s.preset.as_deref(), Some("genesis"));
    assert_eq!(s.step_count, 42);
    assert_eq!(s.population, 0);
    assert!((s.avg_generation - 1.5).abs() < 1e-6);
    assert_eq!(s.sample_count, 1);
    assert!(s.created_recently_hint, "file written seconds ago");

    // missing save is descriptive (and never panics)
    assert!(store.describe_save("ghost").is_err());

    let _ = std::fs::remove_dir_all(&dir);
}
