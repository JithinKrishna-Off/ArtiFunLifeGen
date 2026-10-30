# FLYBRAIN_IMPLEMENTATION — `alife-neural`

Status: released crate-side as of 2026-10-30. This document states FACTS on
disk (module sources, tests in `src/`, measured numbers from `cargo test`) and
explicitly labels everything else as assumption / unknown. The crate NEVER
presents invented structure as biology.

Task/session: Agent B (FlyBrain neural runtime, resumed session
`agentb-flybrain-resume-20261030`; baseline `b7a3f3b`).

---

## 1. Architecture (module map)

```
crates/alife-neural/src/
  connectome.rs          CSR sparse graph (row_offsets/col_idx/weights) +
                         optional per-synapse delays & polarity, compact
                         per-neuron meta with region/cell-type side tables,
                         FNV-1a structural identity, validation, stats,
                         memory_bytes(). NO dense matrices anywhere.
  state.rs               Per-organism BrainState (v, refractory, inj, spiked,
                         delayed-spike ring). Snapshot-serializable.
  runtime.rs             FlyBrainRuntime: deterministic 1 ms stepping,
                         event-driven propagation, tick budget truncation,
                         Queen-facing info()/snapshot()/provenance().
  sensory.rs             SensoryInput (world -> brain contract), 11 scalar
                         channels, documented default neuron binding, rate /
                         linear encoders (deterministic phase-accumulator).
  motor.rs               MotorOutput (brain -> world contract): 6 output
                         cells, bounded decode from spike-rate EMAs.
  fixtures.rs            SYNTHETIC 68-neuron test circuit (never real data;
                         all provenance Assumption, IDs >= 900_000_000).
  connectome_format.rs   SOURCE-INDEPENDENT normalized TSV/sidecar schema
                         ("flybrain-normalized-connectome", v1). A future FAFB
                         converter can emit the same files without touching
                         this crate. Datasets are never merged (one directory
                         = one dataset/import).
  connectome_import.rs   Streaming TSV -> sorted-unique bodyId->compact index
                         -> CSR (3 passes over edges; stable mapping).
  connectome_binary.rs   Versioned "FLBC" artifact: identity hash (FNV-1a-64
                         over all bytes) + CRC-32 footer, corruption detection,
                         memmap-friendly load.
  subnetwork.rs          Region/cell-type filters; induced-subgraph extract
                         that keeps ORIGINAL body IDs and provenance tags;
                         default_mapping helper (documented heuristic).
  provenance.rs          ProvenanceTag field tags + ProvenanceRecord
                         (verified dataset facts; download-dependent fields
                         blank until human-supplied).
tools/convert_malecns.py Official Feather -> normalized TSV converter
                         (streams Arrow record batches; schema-inspect mode).
testdata/malecns-sample/ SYNTHETIC miniature sample for importer unit tests.
```

## 2. LIF runtime (`state.rs` + `runtime.rs`)

### 2.1 Cell equations (1 ms fixed step, no RNG in stepping)

```
leak   = exp(-dt/tau)                             // precomputed once
if refractory[i] > 0:
    refractory[i] -= 1; v[i] = v_reset; no spike
else:
    v' = leak*(v - v_rest) + v_rest + I           // I = ring arrivals + sensory
    if v' >= v_thresh: spike (v := v_reset, refractory := refractory_steps)
    else:              v := v'
```

### 2.2 Parameter table (`LifParams::default()`)

| Parameter             | Value   | Unit / meaning                      |
|-----------------------|---------|-------------------------------------|
| `dt_ms`               | 1.0     | integration step (deterministic)    |
| `tau_ms`              | 20.0    | membrane time constant              |
| `v_rest`              | -70.0   | (state init = v_rest, not 0)        |
| `v_reset`             | -75.0   | post-spike reset                    |
| `v_thresh`            | -57.0   | spike threshold                     |
| `refractory_steps`    | 2       | absolute refractory (ticks)         |
| `default_delay_steps` | 1       | delay when no explicit synapse delay; always >= 1 |

Ring depth: `MAX_RING_DELAY = 16` (explicit delays > 16 clamp to it); ring
length = max(default 1, connectome max explicit delay) + 1, min 2.

### 2.3 Per-tick step order (fixed, deterministic)

1. Sensory encoders inject current into channel-bound input cells
   (`sensory::encode_into`).
2. Delayed arrivals scheduled for this tick drain from the ring into `inj`.
3. Every non-refractory cell integrates in ascending index order; spikes
   collected in index order.
4. Event-driven propagation: ONLY spiked neurons' CSR rows are touched,
   ascending pre-index then in-row position; contributions enter the ring at
   `tick + effective_delay`.
5. Budget (`TickBudget`): max `max_spikes` spiked rows processed, max
   `max_synops` synapses relaxed; the remainder is dropped deterministically
   (lowest indices first) and counted in `Stats.dropped_spikes /
   dropped_synops`. Default budget 4096 spikes / 262_144 synops; unbounded
   available via `TickBudget::unbounded()`.
6. Motor decode from the last-6-neurons spike counts (rate EMAs).

### 2.4 Unknown-polarity policy (default, documented hedge)

Default `UnknownPolarityPolicy::Scale(0.25)`: a synapse whose polarity is
Unknown (no neurotransmitter data imported) propagates as excitatory at 25 %
weight. Configurable: `Ignore` / `AsExcitatory` / `AsInhibitory` /
`Scale(f in (0,1])`. This is a modeling hedge, NOT biology; per-synapse
polarity flags dominate when present, then the pre-neuron polarity tag.

### 2.5 Measured numbers (this machine, `cargo test -p alife-neural`, debug build)

- Test suite: **36 tests, 36 passed** (35 pre-existing + 1 new timing guard).
- SYNTHETIC fixture: **68 neurons, 154 synapses** (asserted in `fixtures.rs`
  tests / runtime stats).
- Average tick cost (fixture, 1000 unbounded ticks, `Stats.exec_ns`):
  **~5.1 microseconds/tick** (measured 5101 ns, debug build; last tick:
  3 spikes, 25 synops relaxed). Guard test
  `runtime::tests::measured_tick_cost_fixture_well_under_millisecond` asserts
  avg < 250 microseconds (generous CI/debug tolerance; it is a gross-regression
  guard, not precision timing — rebuild `--release` and re-measure via
  `cargo test -p alife-neural -- --nocapture measured_tick_cost` for quoted
  numbers on optimization changes).
- Relevance: at ~5 us/tick (68 neurons debug), a 60 Hz frame affords ~27,000
  fixture ticks/millisecond of CPU budget within one frame; full-brain cost is
  NOT extrapolated here (see Limitations).

## 3. Sensory encoding / motor decoding

### 3.1 Channels (`sensory.rs`)

11 scalar channels, fixed order (`CH_*` constants): optic speed, optic cos,
optic sin, light cos, light sin (temperature-gradient PROXY — never described
as biological phototaxis), obstacle, ground contact, food proximity, food cos,
food sin, queen signal. Bounds contract: magnitude channels [0,1], direction
vectors normalized unit/zero via `safe_unit`, all clamped defensively.

Mapping: signed channels map `0.5 + 0.5*v` into the encoder domain; each
channel owns `channel_cells` neurons (default 2): channel c, cell k -> neuron
`c*channel_cells + k` = first 22 neurons of the default layout. This default
binding is a STRUCTURAL ASSUMPTION (`Assumption`), configurable via
`SensoryConfig`, validated by `validate_mapping`; for imported anatomical data
a region-based mapping comes from `subnetwork::default_mapping` (also a
documented heuristic; no biological claim).

Encoders (both deterministic):
- `Rate` (default): phase accumulator at `rate = baseline_hz + gain_hz*v`
  (defaults 4 / 26 Hz); each accrued whole pulse injects
  `pulse_current` (15.0) into the channel cell. No RNG.
- `Linear`: direct analogue current `pulse_current * v * gain_hz/30`.

### 3.2 Output cells (`motor.rs`)

6 output cells, default binding = LAST 6 neurons (order: DirX, DirY,
Intensity, Turn, FeedSeek, Emit). Per-cell spike-rate EMA (`ema_tau_ms` 40)
with rate mapping `rate_mid_hz = 12`, `rate_max_hz = 30`. Signed cells carry a
tonic baseline (silence = zero deviation); gate cells are silence-null. All
outputs hard-clamped: move_dir components [-1,1], turn [-1,1], others [0,1].
The pool->scalar mapping is a MODEL DECISION for the synthetic circuit.

## 4. SYNTHETIC fixture (`fixtures.rs`) — test-only, never the real brain

68 neurons, 154 synapses, deterministic taxi circuit, name
`synthetic-taxi-fixture`, template `fixture-v1`, all neurons
`ProvenanceTag::Assumption`, body IDs `900_000_000..900_000_067`.

| Index range | Role                                                  |
|-------------|-------------------------------------------------------|
| 0..22       | 11 channels x 2 sensory input cells                   |
| 22..44      | relays (one per channel cell, strong excitation      |
| 44..50      | inhibitory interneurons (avoid-driven cross-inhibition) |
| 50..54      | attract-taxis pool                                   |
| 54..58      | avoid/escape pool                                    |
| 58..62      | delayed feedback loop (back to attract; explicit delays 1/2) |
| 62..68      | outputs DirX, DirY, Intensity, Turn, FeedSeek, Emit   |

Weights: `W_EXC = 14.0` (one spike suprathreshold at defaults), `W_EXC_WEAK =
8.0`, `W_INH = 18.0` (effective via Inhibitory polarity). Guaranteed
sensory->motor path: FoodDirCos input (16) -> relay (38) -> attract (50) ->
DirX output (62). The fixture is SYNTHETIC and must never be presented as real
fly brain data; this applies to `testdata/malecns-sample/` too
(see its README).

## 5. Import pipeline (official data path)

```
[MANUAL human download]
   connectome-weights-male-cns-v1.0-minconf-0.5.feather   (~1.1 GB)
   body-annotations-male-cns-v1.0-minconf-0.5.feather     (~13 MB)
        |
        v                     (never downloads by itself)
tools/convert_malecns.py  --schema-inspect (first, read-only)
        |                  then:  --edges <weights.feather> --nodes <annotations.feather>
        |                          --out <dir> [--sha256] (streamed hashing of the 1.1 GB)
        v
  <dir>/edges.tsv   body_pre <TAB> body_post <TAB> weight   (verbatim values)
  <dir>/nodes.tsv   bodyId <TAB> type <TAB> superclass <TAB> status <TAB> somaSide
                    ("" = not stated by source; nothing invented)
  <dir>/sidecar.json  format/format_version/dataset/"MaleCNS v1.0", source_files,
                    converter 1.1.0, source_sha256 ("" until supplied),
                    rows_in/rows_out_edges/rows_out_nodes, converted_at_utc,
                    notes w/ source column map
        |
        v
connectome_import::import_dir(path, &ImportOptions)
   PASS 1: streams nodes.tsv + edges.tsv -> body-ID set; ids sorted unique ->
           compact index (same ID set => same indices; order-independent)
   PASS 2/3: streaming out-degree count + CSR fill in file order (weights
             VERBATIM f32; unknown endpoints: CreateImplicit (default,
             data-faithful) or RejectEdge (strict, counted))
   -> ImportResult { Connectome (validated CSR, side tables from annotations
      only, "" fields = Unavailable, never invented), Sidecar, ImportStats }
        |
        v
connectome_binary::save(&connectome, "path.flbc")
   "FLBC" version-1 container, FNV-1a-64 identity hash + CRC-32 footer;
   load_memmap() is mapping-friendly and fails on any corruption
        |
        v
Arc<Connectome> -> FlyBrainRuntime::new(arc, seed)
```

Converter properties (verified by code review + smoke test on synthetic
Feather with alternative column names):
- Column names are CONFIGURABLE (`--edge-cols PRE,POST,WEIGHT`,
  `--node-cols ID,TYPE,SUPERCLASS,STATUS,SOMASIDE`); defaults documented, and
  a mismatch is a HARD error listing found columns ("refusing to guess a
  mapping"). The sidecar notes record the exact mapping used.
- `--schema-inspect` prints the full Arrow schema (columns/types/batch count)
  of both inputs read-only and exits — required first step because the exact
  upstream connectome-weights column names are UNVERIFIED on official pages.
- Streams `pa.ipc.open_file` record batches; the 1.1 GB file is NEVER fully
  resident. Body IDs require integer Arrow columns (float rejected: precision
  risk). Weights emit shortest round-trip decimal text (`repr(f)`).
- PyArrow requirement is stated on import failure ("pip install pyarrow>=14");
  that is the only third-party dependency.

## 6. Field provenance tags (`provenance.rs`)

Every neuron carries an origin tag on disk (serialized in the FLBC artifact):

| Tag               | Meaning                                                    | Used by                                     |
|-------------------|------------------------------------------------------------|---------------------------------------------|
| `SuppliedBySource`| value came directly from the source dataset                 | annotated nodes.tsv bodies                   |
| `Derived`         | computed by our code from source values (documented)        | (reserved; documented precedence)            |
| `Estimated`       | heuristic placeholder for missing data (documented)         | (reserved)                                   |
| `Assumption`      | wiring-scheme assumption, NOT in the source                 | entire synthetic fixture; default I/O binding |
| `Unavailable`     | no data; placeholder, must not be read as fact              | unannotated/implicit bodies from import      |

Region/cell-type side tables contain only annotation strings actually present
in nodes.tsv; an unannotated neuron keeps index `u32::MAX` (NO_IDX).

## 7. Dataset facts, license, provenance summary

(Agent E verified these on the official pages, read 2026-10-30.)

- Dataset: **MaleCNS v1.0**, released 2026-06-08 per the official release
  notes page (`https://male-cns.janelia.org/release/`).
- Files (official download page `https://male-cns.janelia.org/download/`):
  `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (~1.1 GB) and
  `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13 MB) under
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
- License: the official download page states the Male CNS dataset is licensed
  **CC-BY** with a link to `https://creativecommons.org/licenses/by/4.0/`
  (i.e. CC-BY 4.0). Status: **verified (Agent E, 2026-10-30)**.
- UNVERIFIED (deliberately): the exact `connectome-weights` COLUMN names.
  Mitigation in this crate: `--schema-inspect` first, configurable column
  mapping (`--edge-cols`/`--node-cols`), hard error on mismatch.
- Download-dependent provenance fields (`source_sha256`,
  `original_filename`, imported counts, timestamps) stay BLANK until the
  human who performed the manual download supplies them
  (`ProvenanceRecord::malecns_default()`). We never fabricate a hash or URL.

### Manual acquisition step (human, one-time)

1. Go to `https://male-cns.janelia.org/download`.
2. Download the two files listed above to a local directory.
3. (Optional, recommended) Run
   `python tools/convert_malecns.py --schema-inspect --edges <weights> --nodes <annotations>`
   and confirm/paste the real column names into `--edge-cols/--node-cols`.
4. Run the conversion (optionally with `--sha256`, streamed over the 1.1 GB
   file) and fill `source_sha256`/`original_filename` into the provenance
   record afterwards — the code keeps them blank on its own.

## 8. Queen / Coordinator integration hook (alife-core adapter)

Public API surface (all tested in-crate):

```rust
// Creation — STATIC (shared) or DYNAMIC (genotype) connectome; same API.
FlyBrainRuntime::new(Arc<Connectome>, seed: u32) -> Result<Self, RuntimeBuildError>

// Per world tick (60 Hz core; brain runs its own 1 ms grid; see wiring note)
runtime.tick(&sensory::SensoryInput, &TickBudget) -> motor::MotorOutput

// Introspection for the coordinator
runtime.info() -> RuntimeInfo           // name, template_version, neurons,
                                        // synapses, regions_available,
                                        // cell_types_available, LifParams,
                                        // channels/motor counts,
                                        // unknown_polarity_policy, provenance
runtime.connectome() -> &Arc<Connectome>// full side tables + identity()
runtime.last_stats() -> Stats           // spikes/drops/exec_ns/mem_est
runtime.set_provenance(ProvenanceRecord)

// Save/restore (unit test: runtime::tests::snapshot_roundtrip_restores_state)
runtime.snapshot() -> RuntimeSnapshot   // tick + BrainState + sensory + motor
runtime.restore_snapshot(RuntimeSnapshot)
```

Suggested wiring (code to live in alife-core's Queen adapter, NOT this crate):

```rust
pub struct FlyBrainOrganism {
    /// SHARED by default: one Arc<Connectome> for all organisms of the species
    /// (never a per-organism clone of a shared static connectome).
    pub rt: FlyBrainRuntime,
}

impl FlyBrainOrganism {
    /// Call every organism every core step with that organism's world view.
    /// The 60 Hz core can drive 0..N brain ticks per step (budget-dependent);
    /// accumulate/refract via snapshot() if pausing between bursts.
    pub fn step_from_world(&mut self, senses: fly_senses_to_channels(...)) -> apply_motor(
        self.rt.tick(&senses, &TickBudget::default())) { ... }
}
```

- Build `SensoryInput` from the Organism's world state (field lookups,
  normalized per the bounds contract in `sensory.rs` — e.g. `food_dir` zero
  vector when no food known, which encodes to 0.5/side).
- Apply `MotorOutput` via the existing `apply_motor`-style path (move_dir /
  intensity / turn / feed_seek / emit map onto Organism behavior fields).
- `RuntimeBuildError::TooSmall` decides STATIC-vs-default-mapping fallback;
  `default_mapping_ok()` / `subnetwork::default_mapping` mark the default
  binding as an assumption until a region-based mapping is installed.

## 9. Limitations (explicit)

1. **Full-brain performance unknown.** All measured numbers are from the
   68-neuron synthetic fixture. Until MaleCNS v1.0 (~10^5 neurons, millions
   of edges) is manually downloaded and imported, steady-state full-brain
   tick cost and artifact size are unmeasured. The import pipeline is
   streaming/CSR/memmap-designed for this scale, but that is a design claim
   only, not a measurement.
2. **Polarity Unknown by default.** No neurotransmitter table is imported;
   every imported neuron starts `Polarity::Unknown` and the default
   `Scale(0.25)` policy is a modeling hedge (can be exchanged for Ignore /
   full-w / inhibitory by configuration).
3. **`f32` weight boundary.** The TSV carries shortest round-trip decimal
   text of the source float, but the importer parses `f32`; the crate
   documents that source weights exceeding f32 precision are narrowed.
4. **Default I/O binding is structural assumption**, not anatomy; region
   mapping helpers are heuristic (`Assumption`) until a real mapping is
   specified.
5. **Signed-channel zero is mid-domain**: zero sensory input still yields
   tonic baseline firing (documented design; no "quiescent at zero" claim).
6. **Identity hash is FNV-1a-64**: fast checksum/identity, NOT a
   cryptographic integrity guarantee (`source_sha256` covers authenticity,
   once human-supplied).
7. **Column names upstream unverified**; the converter refuses to guess and
   requires schema inspection on the real download.
8. **`Stats.exec_ns` is wall-clock** and excluded from determinism/snapshots
   by design; timing numbers above are debug-build and machine-dependent.
