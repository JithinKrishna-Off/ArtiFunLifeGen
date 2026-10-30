//! Genome->phenotype decomposition + per-frame instance planning (pure CPU,
//! deterministic, testable without a device).
//!
//! Every organism snapshot (`OrgInst`) is decomposed into a handful of
//! template instances (shell layers / tail / billboard), bounded by hard
//! caps; the planner never reads GPU state and never allocates unboundedly.
//! The warm path ([`Planner`]) reuses its scratch buffers every frame.

use super::{BRAND_CELL, BRAND_COLONY, BRAND_FILAMENT, BRAND_FLAGELLATE};
use crate::scene::OrgInst;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

/// scan bound mirrors `Frame3D.build`'s 1600-organism snapshot cap
pub const MAX_ORGS_SCAN: usize = 1600;
/// full-morph model budget per frame
pub const MODEL_CAP: u32 = 60;
/// mid-range billboard budget per frame
pub const QUAD_CAP: u32 = 240;
/// instance caps (60 morphs x at most 5 sub-shells: filaments/colonies)
pub const SHELL_CAP: u32 = 300;
pub const TAIL_CAP: u32 = 60;
pub const INST_CAP: usize = SHELL_CAP as usize + TAIL_CAP as usize + QUAD_CAP as usize;

/// instance record: 13 x f32 = 52 bytes, bytemuck Pod
pub const INST_FLOATS: usize = 13;
pub const INST_STRIDE: usize = INST_FLOATS * 4;

pub const MODE_ORBIT: u8 = 0;
pub const MODE_FOLLOW: u8 = 1;
pub const MODE_PLANET: u8 = 2;

/// full-morph ring radius by camera mode; billboards extend to
/// [`quad_radius`] (approximately the 450-world-unit contract in orbit).
pub fn morph_radius(mode: u8) -> f32 {
    match mode {
        MODE_FOLLOW => 110.0,
        MODE_PLANET => 265.0,
        _ => 420.0,
    }
}

pub fn quad_radius(mode: u8) -> f32 {
    match mode {
        MODE_FOLLOW => 250.0,
        MODE_PLANET => 430.0,
        _ => 460.0,
    }
}

/// Per-instance render record (= WGSL instance attributes, in order).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct MorphInst {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub heading: f32,
    /// template scale in world units (body radius)
    pub radius: f32,
    /// membrane breathing amplitude (0 disables)
    pub breathe: f32,
    pub hue: f32,
    pub age: f32,
    /// alive fade (also the billboard distance fade)
    pub alpha: f32,
    /// packed anatomy gate: frac(flags) > 0.5 => nucleus visible;
    /// floor(flags / 10) = tail count for the tail template
    pub flags: f32,
    /// stable animation phase (species hue derived)
    pub phase: f32,
    /// turgor 0.35..1 (water-pressure fullness of the cell)
    pub turgor: f32,
    /// animation rate multiplier (flagellar beat / membrane wobble)
    pub wave: f32,
    pub pad: f32,
}

/// A sub-cell of an organism expressed at (brand, hue, size): local offset +
/// world radius + anatomy gate flags. Offsets are LOCAL (template space),
/// rotated by the heading when written.
#[derive(Clone, Debug)]
pub struct ShellSpec {
    pub off: [f32; 3],
    pub radius: f32,
    pub flags: f32,
}

/// One organism's decomposition: shell instances + optional tail instance.
#[derive(Clone, Debug)]
pub struct OrgBuild {
    pub shells: Vec<ShellSpec>,
    /// Some((radius, tail_count)) when the brand grows flagella
    pub tail: Option<(f32, u32)>,
}

const TAU: f32 = std::f32::consts::TAU;

/// Deterministic signature of an instance: brand (2 bits) + hue bucket
/// (8 bits) + size bucket (7 bits). Anatomy depends only on these; per-frame
/// states (energy/age/heading/position) are applied at write time, so the
/// decomposition itself is cacheable by this key.
pub fn signature(o: &OrgInst) -> u64 {
    let brand = (o.brand & 0x3) as u64;
    let hue_b = (o.hue.rem_euclid(1.0).max(0.0) * 256.0).floor() as u64 & 0xff;
    let size_b = (o.size.clamp(0.0, 4.0).max(0.0) * 32.0).floor() as u64 & 0x7f;
    brand | hue_b << 2 | size_b << 10
}

/// Species-variant anatomy knob (deterministic in the species hue, so it is
/// stable for a lineage; no anatomy is invented beyond 1..2 flagella etc.).
pub fn hue_variant(hue: f32, modulo: f32) -> bool {
    (hue.rem_euclid(1.0) * modulo).fract() > 0.5
}

/// Decomposes an organism into template instances from snapshot fields only.
/// Anatomy thresholds: nucleus only for large trait size > 1.1; flagella
/// 1..2; colony sub-cells 2..3; filament segments 3..5.
pub fn org_build(o: &OrgInst) -> OrgBuild {
    let e = o.energy.clamp(0.0, 1.0);
    let tf = 0.62 + 0.38 * e; // turgor scale: drained cells deflate to ~0.62
    let size = o.size.clamp(0.3, 2.0);
    let nuc = if (o.brand == BRAND_CELL || o.brand == BRAND_FLAGELLATE) && size > 1.1 {
        1.0
    } else {
        0.0
    };
    let hue = o.hue.rem_euclid(1.0);
    let mut shells: Vec<ShellSpec> = Vec::new();
    let mut tail = None;
    match o.brand {
        BRAND_FLAGELLATE => {
            let r = 4.6 * size * tf;
            shells.push(ShellSpec { off: [0.0; 3], radius: r, flags: nuc });
            let tails = 1 + (hue_variant(hue, 7.0) as u32);
            tail = Some((r * 0.94, tails));
        }
        BRAND_COLONY => {
            // fused cluster of 2-3 offset scaled cells (golden-angle ring)
            let n = 2 + (hue_variant(hue, 5.0) as u32);
            let r = 3.4 * size * tf;
            for i in 0..n {
                let a = TAU * i as f32 / n as f32 + hue * TAU;
                let d = 0.58 * r;
                shells.push(ShellSpec {
                    off: [a.cos() * d, 0.12 * r * (2.3 * i as f32 + hue * 9.0).sin(), a.sin() * d],
                    radius: r * (1.0 - 0.12 * i as f32),
                    flags: 0.0,
                });
            }
        }
        BRAND_FILAMENT => {
            // chain of 3-5 linked rounded segments reaching forward (+X)
            let n = 3 + (hue_variant(hue, 3.0) as u32 + hue_variant(hue, 9.0) as u32).min(2);
            for i in 0..n {
                let k = i as f32;
                shells.push(ShellSpec {
                    off: [
                        (1.4 + 4.9 * k) * size * tf,
                        (0.9 * (1.25 * k + hue * TAU).sin()) * size * tf,
                        (0.5 * (2.05 * k + hue * 12.3).sin()) * size * tf,
                    ],
                    radius: 3.0 * size * tf * (1.0 - 0.06 * k),
                    flags: 0.0,
                });
            }
        }
        _ => {
            // plain cell: one soft membrane sphere (radius 1 template)
            shells.push(ShellSpec { off: [0.0; 3], radius: 5.5 * size * tf, flags: nuc });
        }
    }
    OrgBuild { shells, tail }
}

/// Bounded decomposition cache keyed by [`signature`].
pub struct DecompCache {
    map: HashMap<u64, OrgBuild>,
}

impl DecompCache {
    pub fn new() -> DecompCache {
        DecompCache { map: HashMap::new() }
    }

    pub fn build(&mut self, o: &OrgInst) -> OrgBuild {
        let sig = signature(o);
        if !self.map.contains_key(&sig) {
            if self.map.len() >= 4096 {
                self.map.clear();
            }
            self.map.insert(sig, org_build(o));
        }
        self.map.get(&sig).cloned().unwrap_or_else(|| org_build(o))
    }
}

pub struct FramePlan {
    /// contiguous instances ordered [TAIL bucket][SHELL bucket][QUAD bucket]
    pub insts: Vec<MorphInst>,
    pub tail_n: u32,
    pub shell_n: u32,
    pub quad_n: u32,
}

#[inline]
fn rot_heading(heading: f32, off: [f32; 3]) -> [f32; 3] {
    // matches the WGSL rot: template +X -> world (cos h, 0, sin h) --
    // the sim advances organisms along (dir.cos(), dir.sin()) in (x, y).
    let (c, s) = (heading.cos(), heading.sin());
    [off[0] * c - off[2] * s, off[1], off[0] * s + off[2] * c]
}

#[inline]
fn inst_phase(hue: f32) -> f32 {
    hue.rem_euclid(1.0) * TAU * 7.0
}

/// animation rate: flagella beat faster with turgor (energy is the speed
/// proxy on the render snapshot), filaments wobble modestly, colonies idle.
fn wave_rate(brand: u32, turgor01: f32) -> f32 {
    match brand {
        BRAND_FLAGELLATE => 5.2 + 4.8 * turgor01,
        BRAND_FILAMENT => 0.9 + 0.7 * turgor01,
        BRAND_COLONY => 1.3 + 0.5 * turgor01,
        _ => 1.6 + 0.6 * turgor01,
    }
}

/// Bounded per-frame planner. Every scratch buffer lives for the whole
/// program lifetime: after the warm-up frame, `plan` runs with no fresh heap
/// traffic (Vec::clear keeps capacity, sorts are in-place, buckets are
/// copied into `out.insts` in one pass).
pub struct Planner {
    decomp: DecompCache,
    out: FramePlan,
    morph_scr: Vec<(f32, u32)>,
    bill_scr: Vec<(f32, u32)>,
    ovf_scr: Vec<(f32, u32)>,
    cand_scr: Vec<(f32, u32)>,
    tails: Vec<MorphInst>,
    shells: Vec<MorphInst>,
    quads: Vec<MorphInst>,
}

impl Planner {
    pub fn new() -> Planner {
        Planner {
            decomp: DecompCache::new(),
            out: FramePlan {
                insts: Vec::with_capacity(INST_CAP),
                tail_n: 0,
                shell_n: 0,
                quad_n: 0,
            },
            morph_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            bill_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            ovf_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            cand_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            tails: Vec::with_capacity(TAIL_CAP as usize),
            shells: Vec::with_capacity(SHELL_CAP as usize),
            quads: Vec::with_capacity(QUAD_CAP as usize),
        }
    }

    /// Last planned frame (borrowed) — the paint path reads this between
    /// `plan` calls.
    pub fn current(&self) -> &FramePlan {
        &self.out
    }

    /// Builds the bounded per-frame instance plan and keeps it (borrowed)
    /// until the next call. `height_at(x, y)` returns the terrain height
    /// (world units) for grounding; `visible(center, radius)` is the camera
    /// frustum test.
    pub fn plan(
        &mut self,
        orgs: &[OrgInst],
        view: [f32; 3],
        mode: u8,
        height_at: &mut dyn FnMut(f32, f32) -> f32,
        visible: &mut dyn FnMut([f32; 3], f32) -> bool,
    ) -> &FramePlan {
        let (mr, qr) = (morph_radius(mode), quad_radius(mode));
        let (mr2, qr2) = (mr * mr, qr * qr);
        self.morph_scr.clear();
        self.bill_scr.clear();
        for (i, o) in orgs.iter().take(MAX_ORGS_SCAN).enumerate() {
            if o.alive <= 0.0 {
                continue; // corpses fade via `alive`; nothing visible to draw
            }
            let dx = o.x - view[0];
            let dz = o.y - view[2];
            let d2 = dx * dx + dz * dz;
            if !d2.is_finite() {
                continue;
            }
            if d2 < mr2 {
                self.morph_scr.push((d2, i as u32));
            } else if d2 < qr2 {
                self.bill_scr.push((d2, i as u32));
            }
        }
        let cmp = |a: &(f32, u32), b: &(f32, u32)| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        };
        self.morph_scr.sort_by(cmp);
        self.bill_scr.sort_by(cmp);

        self.tails.clear();
        self.shells.clear();
        self.quads.clear();

        // nearest-first full-morph organisms; model cap + per-template caps
        let mut kept = 0u32;
        self.ovf_scr.clear();
        for k in 0..self.morph_scr.len() {
            let (d2, i) = self.morph_scr[k];
            let o = &orgs[i as usize];
            let (shells, tail) = {
                let b = self.decomp.build(o);
                (b.shells.clone(), b.tail)
            };
            let need_tail = if tail.is_some() { 1u32 } else { 0 };
            if kept < MODEL_CAP
                && self.shells.len() as u32 + shells.len() as u32 <= SHELL_CAP
                && self.tails.len() as u32 + need_tail <= TAIL_CAP
            {
                let body_r = shells[0].radius.max(0.05);
                let gy = height_at(o.x, o.y);
                let base = [o.x, gy + body_r * 0.62, o.y];
                if !visible(base, body_r * 2.0) {
                    continue;
                }
                kept += 1;
                let alive = o.alive.clamp(0.0, 1.0);
                let e = o.energy.clamp(0.0, 1.0);
                let phase = inst_phase(o.hue);
                let hue = o.hue.rem_euclid(1.0);
                let turgor = 0.35 + 0.65 * e;
                let rate = wave_rate(o.brand, e);
                if let Some((tr, tcount)) = tail {
                    self.tails.push(MorphInst {
                        pos_x: base[0], pos_y: base[1], pos_z: base[2],
                        heading: o.heading,
                        radius: tr,
                        breathe: 0.0,
                        hue,
                        age: o.age.clamp(0.0, 1.0),
                        alpha: alive,
                        flags: tcount as f32 * 10.0,
                        phase,
                        turgor,
                        wave: rate,
                        pad: 0.0,
                    });
                }
                for s in &shells {
                    let off = rot_heading(o.heading, s.off);
                    self.shells.push(MorphInst {
                        pos_x: base[0] + off[0],
                        pos_y: base[1] + off[1],
                        pos_z: base[2] + off[2],
                        heading: o.heading,
                        radius: s.radius,
                        breathe: 0.05 + 0.08 * (1.0 - e),
                        hue,
                        age: o.age.clamp(0.0, 1.0),
                        alpha: alive,
                        flags: s.flags,
                        phase,
                        turgor,
                        wave: rate,
                        pad: 0.0,
                    });
                }
            } else {
                self.ovf_scr.push((d2, i));
            }
        }

        // billboards: beyond-ring orgs + model-cap overflow, far-first draw
        // order (back to front for the alpha blend).
        self.cand_scr.clear();
        self.cand_scr.extend(self.ovf_scr.iter().copied());
        self.cand_scr.extend(self.bill_scr.iter().copied());
        self.cand_scr.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(d2, i) in &self.cand_scr {
            if self.quads.len() >= QUAD_CAP as usize {
                break;
            }
            let o = &orgs[i as usize];
            let e = o.energy.clamp(0.0, 1.0);
            let radius = 5.5 * o.size.clamp(0.3, 2.0) * (0.62 + 0.38 * e);
            let d = d2.sqrt();
            // fade begins ~92% of the morph ring, gone by the outer ring
            let fade = (1.0 - (d - mr * 0.92) / (qr - mr * 0.92)).clamp(0.0, 1.0);
            let gy = height_at(o.x, o.y);
            let center = [o.x, gy + radius * 0.62, o.y];
            if !visible(center, radius) {
                continue;
            }
            self.quads.push(MorphInst {
                pos_x: center[0], pos_y: center[1], pos_z: center[2],
                heading: o.heading,
                radius,
                breathe: 0.0,
                hue: o.hue.rem_euclid(1.0),
                age: o.age.clamp(0.0, 1.0),
                alpha: o.alive.clamp(0.0, 1.0) * fade,
                flags: 0.0,
                phase: inst_phase(o.hue),
                turgor: 0.35 + 0.65 * e,
                wave: 0.0,
                pad: 0.0,
            });
        }

        // assemble the contiguous draw ranges: [tails][shells][billboards]
        let tail_n = self.tails.len() as u32;
        let shell_n = self.shells.len() as u32;
        self.out.insts.clear();
        self.out.insts.extend_from_slice(&self.tails);
        self.out.insts.extend_from_slice(&self.shells);
        self.out.insts.extend_from_slice(&self.quads);
        self.out.tail_n = tail_n;
        self.out.shell_n = shell_n;
        self.out.quad_n = self.quads.len() as u32;
        debug_assert_eq!(
            (tail_n + shell_n + self.out.quad_n) as usize,
            self.out.insts.len()
        );
        debug_assert!(self.out.insts.len() <= INST_CAP);
        &self.out
    }
}

/// Owned-plan convenience wrapper (test path / one-shot callers); the live
/// renderer uses [`Planner`] for zero per-frame allocation.
pub fn plan_frame(
    orgs: &[OrgInst],
    view: [f32; 3],
    mode: u8,
    height_at: &mut dyn FnMut(f32, f32) -> f32,
    visible: &mut dyn FnMut([f32; 3], f32) -> bool,
) -> FramePlan {
    let mut p = Planner::new();
    let n = p.plan(orgs, view, mode, height_at, visible);
    FramePlan {
        insts: n.insts.clone(),
        tail_n: n.tail_n,
        shell_n: n.shell_n,
        quad_n: n.quad_n,
    }
}
