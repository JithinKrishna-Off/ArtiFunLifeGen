# Artificial Life

A Rust-native desktop artificial-biosphere simulator: procedurally generated worlds,
energy-based ecology, evolvable neural controllers, weather and climate, God Mode
interventions, and a full scientific exploration interface.

![crate overview]
- Engine: fixed-step CPU simulation (60 steps/second at 1x speed, budgeted up to higher speeds)
- Rendering: egui 0.29 native UI + instant-mode world renderer (terrain field textures, batched organism painting)
- Genetics: 11→10→3 tanh feed-forward controller per organism, mutated on inheritance
- Weather: drifting cloud layer with rain, droughts, storms, lightning and wetness-driven fertility (updates 1x/second)
- Persistence: versioned JSON saves under `rust/appdata/` (v2 embeds the full engine state incl. weather and intervention history; v1 saves still load)

## Run

Build once (requires Rust 1.98+):

```
cd rust
cargo build --release -p alife-app
```

Launch:

- double-click `run-alife.bat` (project root, starts the built exe), or
- `rust\target\release\alife-app.exe`, or
- `cd rust; cargo run --release -p alife-app`

## Controls

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `->` (Right) | Single step (while paused) |
| `+` / `-` | Zoom out / in |
| `F` | Fit world to view |
| `I` | Toggle inspector |
| `G` | Grid/stat overlay toggle |
| `K` | God Mode view |
| `T` | Evolutionary tree view |
| `S` | Save dialog |
| `Ctrl+N` | New world |
| `Ctrl+Z` | Undo last environment edit (`Ctrl+Y` / `Ctrl+Shift+Z` to redo) |
| `R` | Reset world (confirm) |
| `?` | Shortcut cheatsheet |
| Mouse wheel | Zoom at cursor |
| Drag | Pan |
| Click | Select organism / apply editor tool |

Toolbar time-scale selector: Real-time 1x, Fast 4x, Generational 16x, Century
64x, Millennium+ 256x. Higher speeds are budget-capped: if the frame budget
cannot keep up, the HUD honestly labels the speed as "budgeted" (wall-clock
sim-seconds per real second is shown). Rendering lag can never slow or speed
life itself.

## Weather, climate and God Mode

- A coarse cloud layer drifts with the wind; rain wets cells, boosting local
  fertility and food spawn; droughts dry and penalize them. Wetness coverage
  is tracked as `wet_cover` in history samples and saved.
- Storms schedule lightning strikes: each bolt damaged nearby organisms and
  fertilizes a small ash zone. Meteor / Global Storm / Global Drought represent
  catastrophes at world scale.
- God Mode (press `K`): arm Rain / Drought / Storm / Lightning / Nutrient dump
  then click the world; or trigger Summon organism, Purge lineage, Protect
  lineage and the global catastrophes with a confirm dialog. Every intervention
  is recorded (last 64) with step, event and an engine note, and travels in
  the save slide.
- Lineage protection makes a species immune to the weather / purges.
- Speed note: engine sim time is per-intervention wall-scaled; high speeds
  integrate biology only (approximation, honestly labelled).
- The evolutionary tree view (press `T`) draws founder/parent chains from the
  lineage records with alive/dead, population share and speciation steps.

## What is simulated

- **World**: seeded procedural fields (elevation, moisture, temperature,
  fertility, hazard) composed from fBm + domain warping + ridged noise.
  Fertility drives food regeneration and where organisms can meaningfully live.
- **Energy economy**: organisms eat food items (exclusive consumption), and
  drain `metabolism_base*metabolism + movement_cost*speed*efficiency +
  sensing_cost*sqrt(range/100)` per second, plus hazard damage.
  No energy is created from nothing.
- **Sensors** (all local, no privileged world knowledge): bearing to nearest
  food, food intensity, bearing+density of nearest peers, hazard intensity and
  direction (gradient sampling), own energy, own age.
- **Behavior**: tanh neural controller (11 inputs, 10 hidden, 3 outputs:
  turn, thrust, reproduce-gate). Outputs drive real steering/motion and gate
  reproduction only when energy + cooldown + age conditions hold.
- **Reproduction**: costs `repr_cost * max_energy` taken from the parent,
  offspring inherit and mutate the genotype (rate/magnitude/big-jump odds all
  configurable live).
- **Species**: a child is a new species when genotype distance to its parent
  exceeds 0.62 (`alife-evo::SPECIES_DISTANCE_THRESHOLD`). Species colors are a
  golden-angle hue walk on the species id.
- **Death**: starvation, old age (per-trait lifespan), hazard. Population 0 is a
  legitimate outcome; the UI shows a collapse banner with reseed/new-world.
- **Editing**: place food patches, obstacles, hazards, or erase them at any
  time; undo supported.

## Statistical views

Statistics view plots population, births/deaths, mean energy/age/generation,
diversity, food count, and top-8 species over time from bounded history
samples (buffer downsamples by pair-merge when it exceeds 2048 samples).
Evolution view shows species shares and live trait histograms.

## Reproducibility

- The world generator and simulation use independent seeded generators
  (seeded noise + mulberry32). The same seed, preset, and parameter values
  produce the same world and the same trajectory.
- RNG state is stored in saves, so a loaded save continues the same random
  sequence.
- Limits: results are defined for one platform/version; floating-point
  behavior can differ across platforms/compilers and across versions of the
  noise crate. Treat cross-machine bit-for-bit equality as approximate.

## Saving / data

- Save format: versioned JSON (`format_id: artificial-life/save`);
  **version 2** (current) embeds the engine `core` blob - weather state,
  intervention records and lineage protection. **Version 1** saves (header
  only) still load; weather then restarts from the idle climate.
- Experiment branching: a save can be copied via
  `DataStore::branch_world(source, new)`; branch files are named
  `<source>@<suffix>.json` and listed by `branch_list()`.
- Data lives under `rust/appdata/`: `saves/`, `experiments/`, `exports/`,
  `logs/`.
- Saves store full world config, all organisms + genomes, foods, environment
  edits, RNG state, lineage records, and bounded history samples.
- Saves are atomic (tmp + rename). Loading validates format id/version and
  reports descriptive errors instead of crashing.

## Presets

Genesis (default), Primordial Ocean (single ancestor in a fertile young sea),
Eden, Arid Expanse, Fractured World, Toxic Frontier
(pulsing hazard), Archipelago, Living Labyrinth, Chaotic Bloom, Blank Canvas.

## Project layout

```
rust/crates/alife-common    shared model, contracts, RNG, presets type
rust/crates/alife-worldgen  seeded field generator
rust/crates/alife-core      simulation engine, spatial hash, edits, records
rust/crates/alife-evo       controllers, mutation, species book
rust/crates/alife-render    world view, camera, terrain baking, painting
rust/crates/alife-ui        eframe application shell, panels, dialogs
rust/crates/alife-data      DataStore, history, presets, logging
rust/crates/alife-app       executable entry point
```

## Legacy note

The TypeScript/Electron prototype that seeded the parameter values remains in
this directory (`src/`, `electron/`, vite config) but is NOT part of the
shipping application, which is pure Rust and requires no Node runtime.
