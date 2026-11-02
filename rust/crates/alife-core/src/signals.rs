//! Colony communication signal fields (Agent 7).
//!
//! Six scalar "pheromone" fields over a wrap-connected world. Signals are
//! emitted at world positions, decay exponentially with per-kind half-lives
//! and diffuse with a one-pass separable 3-tap kernel (horizontal then
//! vertical sweep, wrap edges). Single-threaded and fully deterministic:
//! identical emission + tick sequences produce bit-identical grids.
//!
//! Kind indices: 0 FoodFound, 1 Danger, 2 Aggregate, 3 Disperse, 4
//! QueenActivity, 5 EnvStress (see [`kind_name`]).

use serde::{Deserialize, Serialize};

/// Number of distinguishable signal kinds.
pub const SIGNAL_KINDS: usize = 6;

/// Grid of one signal kind (`w`/`h` = cells per axis, row-major `cells`).
#[derive(Clone, Serialize, Deserialize)]
pub struct SignalGrid {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<f32>,
}

/// Six-channel signal field over a wrap-connected world.
#[derive(Clone, Serialize, Deserialize)]
pub struct SignalField {
    pub(crate) grids: [SignalGrid; SIGNAL_KINDS],
    /// Mirrors `config.decay` (spec-fixed field; both are written together
    /// at construction and never diverge).
    pub(crate) decay: [f32; SIGNAL_KINDS],
    pub(crate) world_w: f32,
    pub(crate) world_h: f32,
    /// World-space cell size: (cell_w, cell_h).
    pub(crate) cell: (f32, f32),
    pub(crate) config: SignalConfig,
    /// Reused scratch row/column buffer, allocated once in `new`
    /// (never re-allocated per tick).
    #[serde(skip)]
    scratch: Vec<f32>,
    #[serde(skip)]
    emit_calls: u32,
    #[serde(skip)]
    emit_units: f64,
}

/// Tunables for [`SignalField`].
///
/// Note: the task sketch listed both a `Default` derive and a hand-tuned
/// `default()`; those cannot coexist, so the hand-tuned impl below is the
/// authority (derive would only yield zeros).
#[derive(Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Per-kind exponential decay rate (per sim second). Half-life = `ln2/decay[k]`.
    pub decay: [f32; 6],
    /// Per-kind diffusion weight (3-tap separable). Keep <= 0.4 for stability.
    pub diffusion: [f32; 6],
    /// Maximum signal mass added per `emit` call to one cell.
    pub emission_cap: f32,
}

impl Default for SignalConfig {
    /// Tuned defaults. Decay half-lives: FoodFound 60 s (slow, persistent
    /// trail), Danger 20 s, Aggregate 30 s (medium), Disperse 25 s
    /// (medium-strong fade), QueenActivity 8 s (fast, heavy turnover),
    /// EnvStress 45 s (slow background). Diffusion small so gradients stay
    /// local.
    fn default() -> Self {
        let ln2 = std::f32::consts::LN_2;
        Self {
            decay: [ln2 / 60.0, ln2 / 20.0, ln2 / 30.0, ln2 / 25.0, ln2 / 8.0, ln2 / 45.0],
            diffusion: [0.30, 0.40, 0.30, 0.40, 0.25, 0.20],
            emission_cap: 0.5,
        }
    }
}

/// Stable name for a kind index (UI / observatory helper).
pub fn kind_name(kind: usize) -> &'static str {
    match kind {
        0 => "FoodFound",
        1 => "Danger",
        2 => "Aggregate",
        3 => "Disperse",
        4 => "QueenActivity",
        5 => "EnvStress",
        _ => "unknown",
    }
}

/// Queen biological/abstract cost report: emission call count and the total
/// signal mass that actually landed (clipping-aware).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmissionReport {
    pub calls: u32,
    pub signal_units: f32,
}

/// Budgeted tick passes: decay rows and diffusion rows/columns move in
/// chunks of this size (deterministic, power-of-two granularity).
const CHUNK_ROWS: usize = 8;
/// Blob header: magic + version + world dims + grid dims.
const BLOB_HEADER_BYTES: usize = 24;
const BLOB_MAGIC: &[u8; 6] = b"ALFSGG";
const BLOB_VERSION: u16 = 1;

impl SignalField {
    /// `res` cells per axis (both axes), default config.
    pub fn new(world_w: f32, world_h: f32, res: u32) -> Self {
        Self::with_config(world_w, world_h, res, SignalConfig::default())
    }

    /// `res` cells per axis (both axes) with a custom config.
    pub fn with_config(world_w: f32, world_h: f32, res: u32, cfg: SignalConfig) -> Self {
        Self::build(world_w, world_h, res, res, cfg)
    }

    fn build(world_w: f32, world_h: f32, res_w: u32, res_h: u32, mut cfg: SignalConfig) -> Self {
        let ww = if world_w.is_finite() && world_w > 0.0 { world_w } else { 1.0 };
        let wh = if world_h.is_finite() && world_h > 0.0 { world_h } else { 1.0 };
        let w = res_w.max(1) as usize;
        let h = res_h.max(1) as usize;
        // Sanitize config: NaNs/negatives become neutral values.
        for d in cfg.decay.iter_mut() {
            *d = if d.is_finite() { (*d).max(0.0) } else { 0.0 };
        }
        for d in cfg.diffusion.iter_mut() {
            *d = if d.is_finite() { d.clamp(0.0, 1.0) } else { 0.0 };
        }
        cfg.emission_cap =
            if cfg.emission_cap.is_finite() { cfg.emission_cap.max(0.0) } else { 0.0 };
        let decay = cfg.decay;
        let grids = std::array::from_fn(|_| SignalGrid { w, h, cells: vec![0.0; w * h] });
        Self {
            decay,
            grids,
            world_w: ww,
            world_h: wh,
            cell: (ww / w as f32, wh / h as f32),
            config: cfg,
            scratch: vec![0.0; w.max(h)],
            emit_calls: 0,
            emit_units: 0.0,
        }
    }

    // -------------------------------------------------------------- emission

    /// Add `strength` signal of `kind` at world position (wrapped). The added
    /// mass is clamped to `[0, emission_cap]` per call, and a cell never
    /// exceeds 1.0. Non-finite strengths / invalid kinds are no-ops.
    pub fn emit(&mut self, kind: usize, x: f32, y: f32, strength: f32) {
        if kind >= SIGNAL_KINDS || !strength.is_finite() {
            return;
        }
        let add = strength.clamp(0.0, self.config.emission_cap);
        if add <= 0.0 {
            return;
        }
        let (ix, iy) = match self.cell_index(x, y) {
            Some(v) => v,
            None => return,
        };
        let g = &mut self.grids[kind];
        let idx = iy * g.w + ix;
        let cur = g.cells[idx];
        let next = (cur + add).min(1.0);
        g.cells[idx] = next;
        self.emit_calls = self.emit_calls.saturating_add(1);
        self.emit_units += (next - cur) as f64;
    }

    /// Continuous (bilinear-ready) cell coordinates of a world point.
    fn cell_coords(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        if !x.is_finite()
            || !y.is_finite()
            || self.cell.0 <= 0.0
            || self.cell.1 <= 0.0
        {
            return None;
        }
        // World is wrap-connected everywhere: modulo into [0, world).
        let wx = x.rem_euclid(self.world_w);
        let wy = y.rem_euclid(self.world_h);
        // Tiny pull-in so that wx == world_w cannot index one row past the edge.
        let gx = (wx / self.cell.0).clamp(0.0, self.grids[0].w as f32 - 1.0e-4);
        let gy = (wy / self.cell.1).clamp(0.0, self.grids[0].h as f32 - 1.0e-4);
        Some((gx, gy))
    }

    fn cell_index(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let (gx, gy) = self.cell_coords(x, y)?;
        Some((gx as usize, gy as usize))
    }

    // ------------------------------------------------------------------ tick

    /// Exponential decay + one-pass separable 3-tap diffusion (horizontal
    /// then vertical sweep), wrap edges, in-kind mass conserving.
    /// Deterministic fixed order: kinds 0..6, rows top-to-bottom.
    pub fn tick(&mut self, dt: f32) {
        self.tick_internal(dt, u32::MAX as u64);
    }

    /// Budgeted tick: identical to `tick` while the op budget lasts; when it
    /// runs out, the remaining work is skipped deterministically. Ops are
    /// counted per cell update (a full kind costs `w*h` decay ops plus
    /// `2*w*h` diffusion ops); rows/columns are processed top-to-bottom and
    /// left-to-right in fixed-size chunks, so the truncation point depends
    /// only on `(state, dt, max_ops)` — never on wall time. Skipped cells
    /// just miss this tick's decay/diffusion (mass lags, is not lost).
    pub fn tick_budgeted(&mut self, dt: f32, max_ops: u32) {
        self.tick_internal(dt, max_ops as u64);
    }

    fn tick_internal(&mut self, dt: f32, mut budget: u64) {
        if !dt.is_finite() || dt <= 0.0 || budget == 0 {
            return;
        }
        for kind in 0..SIGNAL_KINDS {
            // --- decay pass (row chunks) ---
            let rate = self.decay[kind].clamp(0.0, 1.0e9);
            let factor = (1.0 - rate * dt).clamp(0.0, 1.0);
            let (w, h) = (self.grids[kind].w, self.grids[kind].h);
            if factor < 1.0 && budget >= w as u64 {
                let mut y = 0usize;
                while y < h && budget >= w as u64 {
                    let rows = ((budget / w as u64) as usize).min(h - y).min(CHUNK_ROWS);
                    let base = y * w;
                    for v in &mut self.grids[kind].cells[base..base + rows * w] {
                        *v *= factor;
                    }
                    budget -= (rows * w) as u64;
                    y += rows;
                }
            }
            // --- diffusion pass (columns first horizontally, then vertically) ---
            let d = self.config.diffusion[kind].clamp(0.0, 1.0);
            if d <= 0.0 || budget < w as u64 {
                continue;
            }
            // Horizontal sweep: each row is independent; scratch holds the
            // row before it is overwritten.
            let mut y = 0usize;
            while y < h && budget >= w as u64 {
                let rows = ((budget / w as u64) as usize).min(h - y).min(CHUNK_ROWS);
                for r in y..y + rows {
                    let base = r * w;
                    self.scratch[..w].copy_from_slice(&self.grids[kind].cells[base..base + w]);
                    for x in 0..w {
                        let left = self.scratch[(x + w - 1) % w];
                        let right = self.scratch[(x + 1) % w];
                        let c = self.scratch[x];
                        self.grids[kind].cells[base + x] =
                            (1.0 - d) * c + d * 0.5 * (left + right);
                    }
                }
                budget -= (rows * w) as u64;
                y += rows;
            }
            // Vertical sweep: each column is independent.
            if budget < h as u64 {
                continue;
            }
            let mut x = 0usize;
            while x < w && budget >= h as u64 {
                let cols = ((budget / h as u64) as usize).min(w - x).min(CHUNK_ROWS);
                for c in x..x + cols {
                    for yy in 0..h {
                        self.scratch[yy] = self.grids[kind].cells[yy * w + c];
                    }
                    for yy in 0..h {
                        let up = self.scratch[(yy + h - 1) % h];
                        let down = self.scratch[(yy + 1) % h];
                        let m = self.scratch[yy];
                        self.grids[kind].cells[yy * w + c] =
                            (1.0 - d) * m + d * 0.5 * (up + down);
                    }
                }
                budget -= (cols * h) as u64;
                x += cols;
            }
        }
    }

    // -------------------------------------------------------------- sampling

    /// Bilinear sample of `kind` at world coords (wrapped); sampling is
    /// seamless across world edges because both the wrap and the kernel are
    /// modular. Out-of-range kinds and non-finite coords read as 0.
    pub fn sample(&mut self, kind: usize, x: f32, y: f32) -> f32 {
        if kind >= SIGNAL_KINDS {
            return 0.0;
        }
        let (gx, gy) = match self.cell_coords(x, y) {
            Some(v) => v,
            None => return 0.0,
        };
        let (w, h) = (self.grids[kind].w, self.grids[kind].h);
        let i0 = gx.floor() as usize;
        let fx = (gx - i0 as f32).clamp(0.0, 1.0);
        let j0 = gy.floor() as usize;
        let fy = (gy - j0 as f32).clamp(0.0, 1.0);
        let i1 = (i0 + 1) % w;
        let j1 = (j0 + 1) % h;
        let a = self.grids[kind].cells[j0 * w + i0];
        let b = self.grids[kind].cells[j0 * w + i1];
        let c = self.grids[kind].cells[j1 * w + i0];
        let d2 = self.grids[kind].cells[j1 * w + i1];
        let top = a + (b - a) * fx;
        let bot = c + (d2 - c) * fx;
        top + (bot - top) * fy
    }

    /// Direct read handle over all kind grids (renderers / observatory).
    pub fn grids(&self) -> &[SignalGrid; SIGNAL_KINDS] {
        &self.grids
    }

    /// Total signal mass of one kind (f64 accumulation in fixed order).
    pub fn active_mass(&self, kind: usize) -> f32 {
        if kind >= SIGNAL_KINDS {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for v in &self.grids[kind].cells {
            if v.is_finite() {
                acc += *v as f64;
            }
        }
        acc as f32
    }

    /// Estimated memory footprint (struct + grid cell/scratch allocations).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self
                .grids
                .iter()
                .map(|g| g.cells.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self.scratch.capacity() * std::mem::size_of::<f32>()
    }

    pub fn config(&self) -> &SignalConfig {
        &self.config
    }

    /// Reset-on-read emission ledger (Queen cost reporting).
    pub fn emission_report(&mut self) -> EmissionReport {
        let calls = std::mem::replace(&mut self.emit_calls, 0);
        let units = std::mem::replace(&mut self.emit_units, 0.0);
        EmissionReport { calls, signal_units: units as f32 }
    }

    // ------------------------------------------------------------------ blob

    /// Compact little-endian save:
    /// `[magic "ALFSGG"][version u16 LE][world_w f32 bits][world_h f32 bits]
    /// [w u32 LE][h u32 LE][6 * w * h f32 bits, kind-major, row-major]`.
    /// Only field *content* is persisted; config/decay are taken from the
    /// `cfg` argument of `from_blob` (keeps blobs small, lets callers retune
    /// decay on load).
    pub fn to_blob(&self) -> Vec<u8> {
        let g0 = &self.grids[0];
        let mut out =
            Vec::with_capacity(BLOB_HEADER_BYTES + 4 * g0.cells.len() * SIGNAL_KINDS);
        out.extend_from_slice(BLOB_MAGIC);
        out.extend_from_slice(&BLOB_VERSION.to_le_bytes());
        out.extend_from_slice(&self.world_w.to_bits().to_le_bytes());
        out.extend_from_slice(&self.world_h.to_bits().to_le_bytes());
        out.extend_from_slice(&(g0.w as u32).to_le_bytes());
        out.extend_from_slice(&(g0.h as u32).to_le_bytes());
        for kind in 0..SIGNAL_KINDS {
            for v in &self.grids[kind].cells {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        out
    }

    /// Inverse of `to_blob` (identity-checked). Returns `None` on bad magic,
    /// unknown version, wrong length, zero grid dims or cell-count mismatch.
    /// Non-finite cells are sanitized to 0 rather than rejected.
    pub fn from_blob(bytes: &[u8], cfg: SignalConfig) -> Option<Self> {
        if bytes.len() < BLOB_HEADER_BYTES || &bytes[0..6] != BLOB_MAGIC {
            return None;
        }
        let ver = u16::from_le_bytes([bytes[6], bytes[7]]);
        if ver != BLOB_VERSION {
            return None;
        }
        let ww =
            f32::from_bits(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]));
        let wh =
            f32::from_bits(u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]));
        let w = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let h = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
        if w == 0 || h == 0 {
            return None;
        }
        let needed = w * h * SIGNAL_KINDS * 4;
        if bytes.len() < BLOB_HEADER_BYTES + needed {
            return None;
        }
        let mut field = Self::build(ww, wh, w as u32, h as u32, cfg);
        let mut pos = BLOB_HEADER_BYTES;
        for kind in 0..SIGNAL_KINDS {
            for v in field.grids[kind].cells.iter_mut() {
                let bits = u32::from_le_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                ]);
                pos += 4;
                let val = f32::from_bits(bits);
                *v = if val.is_finite() { val } else { 0.0 };
            }
        }
        Some(field)
    }
}

/// Companion wrapper: QueenActivity is kind 4.
pub fn emit_queen_activity(f: &mut SignalField, x: f32, y: f32, strength: f32) {
    f.emit(4, x, y, strength);
}
