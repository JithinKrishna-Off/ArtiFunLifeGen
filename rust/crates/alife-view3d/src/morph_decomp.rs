//! Genome->phenotype decomposition + per-frame instance planning (pure CPU,
//! deterministic, testable without a device).
//!
//! Every organism snapshot (`OrgInst`) is decomposed into a handful of
//! template instances (shell layers / tail / billboard), bounded by hard
//! caps; the planner never reads GPU state and never allocates unboundedly.
//! The warm path ([`Planner`]) reuses its scratch buffers every frame.

use super::{BRAND_CELL, BRAND_COLONY, BRAND_FILAMENT, BRAND_FLAGELLATE, BRAND_QUEEN};
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
/// reserved queen slots (always full-morph, never billboards); the queen is
/// additionally counted into MODEL_CAP when she sits inside the morph ring
pub const QUEEN_CAP: usize = 2;
/// queens render as a full morph at ANY range inside this hard far bound
/// (beyond it she is simply off-camera by frustum/scan logic). Inside the
/// ring she costs her MODEL_CAP slot; outside it she is "reduced detail"
/// (flagged in `MorphInst.pad`), but keeps the identical queen silhouette and
/// never merges with ordinary billboards.
pub const QUEEN_FAR: f32 = 1600.0;

/// Queen template body length in RADIUS units (morph_queen: x spans
/// [-1.67, +1.61]); crown tips reach +2.05, belly ground line ~-0.87.
pub const QUEEN_BODY_LEN: f32 = 3.28;
/// target visible queen body length (crown .. tail tip, world units):
/// a ~16-unit giant against ordinary organisms re-scaled to ~1.2-1.5 units.
pub const QUEEN_TARGET_LEN: f32 = 16.0;
/// Ordinary (non-queen) template base body radius in world units at size 1,
/// full turgor: the tiny re-scale (was the shared `5.5` base). The queen is
/// DECOUPLED via [`QUEEN_UNIT_RADIUS`], so changing this never shifts her.
/// Visible cell body length = 2 * THIS * size * turgor ~= 1.39 at size 1 /
/// render-default turgor 0.924 (~1/12 of the queen's 16-unit body).
pub const ORGANISM_UNIT_RADIUS: f32 = 0.75;
/// Brand silhouette factors as fractions of [`ORGANISM_UNIT_RADIUS`] (the
/// old brand constants were 4.6 / 3.4 / 3.0 over a 5.5 base). The flagellate
/// factor is trimmed below the legacy 4.6/5.5 = 0.8364 ratio so its tail
/// strand keeps the whole silhouette inside the [0.8, 2.0] length band.
const FLAGELLATE_F: f32 = 0.62;
const COLONY_F: f32 = 3.4 / 5.5;
const FILAMENT_F: f32 = 3.0 / 5.5;
/// Source-of-truth queen body radius: explicit world units, DECOUPLED from
/// the ordinary template base so any ordinary re-scale never shifts her.
/// Visible body length = QUEEN_BODY_LEN * THIS * size * turgor
/// ~= [`QUEEN_TARGET_LEN`] at size 2.0 / turgor 1.0 (~14.8 at the render
/// default turgor 0.924). BEFORE the giant rebalance the chain was
/// `2.7 * (5.5 * size * turgor)` ~= 97 units at size 2.0 (~6x oversized for
/// the ordinary re-scale). size/turgor dependence kept so she breathes
/// visually. Physics/collision fields untouched.
pub const QUEEN_UNIT_RADIUS: f32 = QUEEN_TARGET_LEN / (QUEEN_BODY_LEN * 2.0);
/// QUEEN-only pre-pass body radius (single point shared by the planner).
#[inline]
pub fn queen_body_radius(size: f32, energy: f32) -> f32 {
    QUEEN_UNIT_RADIUS * size.clamp(0.3, 2.0) * (0.62 + 0.38 * energy.clamp(0.0, 1.0))
}
/// Legacy mirror multiplier so consumers that mirror the shared ordinary
/// template base (`5.5 * size * turgor * mul`) read the queen's true radius:
/// `5.5 * QUEEN_SCALE == QUEEN_UNIT_RADIUS` exactly (= 2.44). Sent non-1.0
/// via `OrgInst::visual_radius_mul`; pick/markers must treat any finite
/// positive mul != 1.0 as "queen". Rebased from the pre-rebalance value
/// 2.7 in step with the giant length retarget above.
pub const QUEEN_SCALE: f32 = QUEEN_UNIT_RADIUS / 5.5;
/// queen instance-origin height above terrain, in body-radius units: the
/// template ground line (abdomen belly incl. ripple + breathe headroom)
/// sits at local y = -0.87; lifting to 0.88*body_r seats her belly on the
/// ground while limb stubs (-0.64) clear the terrain. Heading rotation is
/// yaw-only (around Y), so no pitch bias is needed to keep her level.
pub const QUEEN_GROUND_LIFT: f32 = 0.88;
/// per-frame shortest-angle heading tracking rate (visual smoothing only)
const QUEEN_HEADING_TURN: f32 = 0.35;
/// queen world speed (units/s) at which the gait animation is fully engaged
const QUEEN_GAIT_REF: f32 = 45.0;
/// gait wave ceiling (leg/abdomen/antennae animation rate multiplier)
const QUEEN_GAIT_MAX_WAVE: f32 = 3.6;
/// beyond this squared per-frame jump the queen is considered teleported
/// (heading/gait state re-initializes instead of reporting a speed spike)
const QUEEN_TELEPORT2: f32 = 40.0 * 40.0;

/// Cached per-queen visual state for heading smoothing + gait-speed
/// estimation (visual only; never feeds sim state).
#[derive(Clone, Copy)]
struct QueenVis {
    x: f32,
    y: f32,
    heading: f32,
}

#[inline]
fn shortest_angle(from: f32, to: f32) -> f32 {
    (to - from + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI
}

pub const INST_CAP: usize =
    SHELL_CAP as usize + TAIL_CAP as usize + QUAD_CAP as usize + QUEEN_CAP;

/// instance record: 22 x f32 = 88 bytes, bytemuck Pod. Decode order =
/// WGSL instance attributes (morph.rs VS): locations 2..=13 are the legacy
/// 14 floats (ipos vec3 @2, scalars @3..=13), the A3 locomotion extension
/// appends `motion` (vec4 @14..=17) and `drive` (vec4 @18..=21). Row lanes
/// that never animate keep every extension field at 0.0:
/// - queen lane (reserved template; settled queen motion, pad 0/1)
/// - legacy tail/shell/billboard planner lanes (CPU diagnostics only) and
///   the organism blob lane (pad 0) — the VS creature gate
///   `pad in (0.0, 0.5)` excludes them, ordinary archetype rows carry
///   sat_tint there and animate.
pub const INST_FLOATS: usize = 22;
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
    /// queen LOD inside her own template family: 1.0 = full detail; 0.0 =
    /// reduced detail beyond the 450-unit ring (limb stubs + antennae
    /// collapsed, abdominal ridge pulse damped) while the body+crown
    /// silhouette stays identical. Always 0.0 for non-queen instances.
    pub pad: f32,
    // ------------------------------------------------ A3 locomotion (append-only)
    /// (bob_y, pitch, roll, stride_phase): vertical body bob in WORLD units
    /// (added +Y by the VS after grounding so the body lifts above the
    /// terrain-seat), small-angle pitch/roll in radians, and the accrued
    /// stride phase in radians (per-organism position-delta acceleration +
    /// per-instance desync seed; the VS derives per-leg offsets from the
    /// limb's template x/y slot). Zero for idle rows / never-animated lanes.
    pub motion: [f32; 4],
    /// (leg_swing, leg_lift, walk_squash, breathe): limb stride swing and
    /// vertical lift amplitudes in WORLD units (Consumed only by CODE_LIMB
    /// verts below the trunk midline — antennae are excluded by template
    /// y sign), the phase-locked squash-stretch amplitude for walk cycles
    /// (strongest on Primitive bodies, damped by armour) and the slow
    /// ambient breathing amplitude (idle `LOCO_IDLE_BREATHE`). Zero except
    /// `breathe` on idle rows.
    pub drive: [f32; 4],
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

/// Deterministic signature of an instance: brand (3 bits, incl. BRAND_QUEEN)
/// + hue bucket (8 bits) + size bucket (7 bits). Anatomy depends only on
/// these; per-frame states (energy/age/heading/position) are applied at write
/// time, so the decomposition itself is cacheable by this key.
pub fn signature(o: &OrgInst) -> u64 {
    let brand = (o.brand & 0x7) as u64;
    let hue_b = (o.hue.rem_euclid(1.0).max(0.0) * 256.0).floor() as u64 & 0xff;
    let size_b = (o.size.clamp(0.0, 4.0).max(0.0) * 32.0).floor() as u64 & 0x7f;
    brand | hue_b << 3 | size_b << 11
}

/// Species-variant anatomy knob (deterministic in the species hue, so it is
/// stable for a lineage; no anatomy is invented beyond 1..2 flagella etc.).
pub fn hue_variant(hue: f32, modulo: f32) -> bool {
    (hue.rem_euclid(1.0) * modulo).fract() > 0.5
}

/// Decomposes an organism into template instances from snapshot fields only.
/// Anatomy thresholds: nucleus only for large trait size > 1.1; flagella
/// 1..2; colony lobes 3; filament segments 3..4. The flagellate body shell
/// carries `flags = 10 + nucleus_gate`: frac() keeps the nucleus gate while
/// `floor(flags / 10) == 1` codes the teardrop silhouette (WGSL, morph.rs).
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
            let r = ORGANISM_UNIT_RADIUS * FLAGELLATE_F * size * tf;
            shells.push(ShellSpec { off: [0.0; 3], radius: r, flags: nuc + 10.0 });
            let tails = 1 + (hue_variant(hue, 7.0) as u32);
            tail = Some((r * 0.94, tails));
        }
        BRAND_COLONY => {
            // sessile 3-lobed fused cluster (golden-phase ring; lobes shrink
            // slightly with rank; z rides a per-lobe organic sine)
            let r = ORGANISM_UNIT_RADIUS * COLONY_F * size * tf;
            for i in 0..3 {
                let a = TAU * i as f32 / 3.0 + hue * TAU;
                let d = 0.58 * r;
                shells.push(ShellSpec {
                    off: [a.cos() * d, 0.12 * r * (2.3 * i as f32 + hue * 9.0).sin(), a.sin() * d],
                    radius: r * (1.0 - 0.12 * i as f32),
                    flags: 0.0,
                });
            }
        }
        BRAND_FILAMENT => {
            // short linked chain of 3-4 rounded segments reaching forward
            // (+X); offsets are expressed in segment-radius units so the
            // beads touch/overlap like a real filament
            let r = ORGANISM_UNIT_RADIUS * FILAMENT_F * size * tf;
            let n = 3 + (hue_variant(hue, 3.0) as u32 + hue_variant(hue, 9.0) as u32).min(1);
            for i in 0..n {
                let k = i as f32;
                shells.push(ShellSpec {
                    off: [
                        r * (0.5 + 0.95 * k),
                        r * 0.3 * (1.25 * k + hue * TAU).sin(),
                        r * 0.1667 * (2.05 * k + hue * 12.3).sin(),
                    ],
                    radius: r * (1.0 - 0.06 * k),
                    flags: 0.0,
                });
            }
        }
        _ => {
            // plain cell: one soft membrane sphere (radius 1 template)
            shells.push(ShellSpec {
                off: [0.0; 3],
                radius: ORGANISM_UNIT_RADIUS * size * tf,
                flags: nuc,
            });
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
    /// contiguous instances ordered [TAIL][SHELL][QUEEN][QUAD]
    pub insts: Vec<MorphInst>,
    pub tail_n: u32,
    pub shell_n: u32,
    /// reserved full-morph queen templates (always rendered; never billboards)
    pub queen_n: u32,
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

// ================================================================ A3 locomotion
// Procedural body motion ON TOP of the authoritative planner row (view-side
// only: sim positions/headings are never mutated; sway rides the row's own
// heading float). Pure functions of (body plan, body length, speed, phase)
// plus per-organism phase/gait state accumulated from actual position
// deltas (same estimator pattern as the queen lane below).
//
// Speed scale note: the ordinary archetype silhouette is ~1/12 of the
// queen's 16-unit giant, so the walk bands are ABSOLUTE world units tuned
// to the queen's QUEEN_GAIT_REF divisor rather than trait-speed units.

/// below this speed (world u/s) an organism is idle: bob/pitch/roll/legs
/// zero out, only the ambient breathe remains
pub const LOCO_IDLE_EPS: f32 = 1.5;
/// speed (world u/s) where the gait is fully engaged (amplitude/taper 1.0)
pub const LOCO_FULL: f32 = 22.0;
/// stride frequency = clamp(LOCO_FREQ_K / body_len, LOCO_F_MIN, LOCO_F_MAX)
/// Hz — small creatures wiggle faster (small-bl bodies push the 9 Hz cap)
pub const LOCO_FREQ_K: f32 = 6.0;
pub const LOCO_F_MIN: f32 = 1.2;
pub const LOCO_F_MAX: f32 = 9.0;
/// body bob at full speed, in body-length fraction (world units = * body_len)
pub const LOCO_BOB: f32 = 0.055;
/// per-plan bob scale: Primitive floats/squashes, arthropods bound,
/// multi-legged amble, quadruped full diagonal-pair bounce
const BOB_PLAN_SCALE: [f32; 4] = [0.30, 1.00, 0.85, 1.00];
/// jointedness (floppy limbs) smooths the gait: bob reduced by this fraction
pub const LOCO_JOINT_DAMP: f32 = 0.6;
/// pitch couples the bob derivative; clamped to LOCO_PITCH_MAX radians
pub const LOCO_PITCH_K: f32 = 0.5;
pub const LOCO_PITCH_MAX: f32 = 0.12;
/// yaw sway at full speed (radians) - applied to the row's heading only
pub const LOCO_SWAY: f32 = 0.06;
/// per-plan sway boost (multi-legged amble sways hardest)
const SWAY_PLAN_BOOST: [f32; 4] = [1.0, 1.0, 1.5, 1.0];
/// roll couples the sway derivative, clamped to LOCO_ROLL_MAX radians
pub const LOCO_ROLL_K: f32 = 1.1;
pub const LOCO_ROLL_MAX: f32 = 0.16;
/// limb stride swing at full speed, body-length fraction (* body_len for
/// world units); lift is LOCO_LEG_LIFT of it
pub const LOCO_LEG_AMP: f32 = 0.16;
pub const LOCO_LEG_LIFT: f32 = 0.55;
/// per-plan leg scale: Primitive is limbless, arthropod tripod tread at
/// full swing, multi-legged short strides, quadruped diagonal pairs
const LEG_PLAN_SCALE: [f32; 4] = [0.0, 1.00, 0.85, 1.00];
/// jointed limbs swing a touch less (knees absorb)
pub const LOCO_LEG_JOINT_DAMP: f32 = 0.3;
/// walk squash-stretch amplitude (Primitive bodies squash hardest;
/// armour plating stiffens the shell by LOCO_ARMOUR_DAMP)
pub const LOCO_SQUASH_WALK: f32 = 0.05;
pub const LOCO_SQUASH_BODY: f32 = 0.012;
pub const LOCO_ARMOUR_DAMP: f32 = 0.5;
/// idle ambient breathing scale amplitude (1 +- THIS in the VS)
pub const LOCO_IDLE_BREATHE: f32 = 0.02;
/// ambient breathe while walking is shallower than the full idle swell
pub const LOCO_WALK_BREATHE: f32 = 0.008;
/// idle yaw-twitch amplitude (antennae/eyestalk approximation; view-side
/// heading jitter, sim heading untouched)
pub const LOCO_IDLE_TWITCH: f32 = 0.012;
/// speed smoothing per frame (lerp factor toward the instantaneous estimate)
pub const LOCO_SPEED_LERP: f32 = 0.35;
/// beyond this squared per-frame position jump (40 u at the classic 1/60 s)
/// the organism is treated as teleported/spawned: speed resets, phase lives
pub const LOCO_TELEPORT2: f32 = 1600.0;
/// idle stride-phase drift (fraction of the walking rate)
pub const LOCO_IDLE_RATE: f32 = 0.25;
/// hard track cap (LRU-ish: the per-pass prune makes the steady-state map
/// hold exactly the live organism count)
pub const LOCO_TRACK_CAP: usize = 4096;

/// stride frequency in Hz for a visible body length (world units).
pub fn gait_freq(body_len: f32) -> f32 {
    (LOCO_FREQ_K / body_len.max(0.05)).clamp(LOCO_F_MIN, LOCO_F_MAX)
}

/// Per-leg stride phase offsets the VS derives from the limb's raw template
/// slot (mirrored in WGSL; keep in lockstep):
/// - x slot bucket: x >= 0 => front (+0), -0.24 <= x < 0 => mid (+PI/2),
///   x < -0.24 => rear (+PI)
/// - z side parity: z > 0 => +0, z <= 0 => +PI (left/right anti-phase rows;
///   with front/rear buckets this yields diagonal pairs = trot, and with
///   mid rows a tripod-ish tread using appendage parity)
pub fn leg_bucket_offset(template_x: f32) -> f32 {
    if template_x < -0.24 {
        std::f32::consts::PI
    } else if template_x < 0.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        0.0
    }
}

/// VS mirror: limb template z side -> parity offset (radians).
pub fn leg_side_parity(template_z: f32) -> f32 {
    if template_z > 0.0 { 0.0 } else { std::f32::consts::PI }
}

/// Pure per-frame body-motion output for one organism row. `phase` is the
/// ACCRUED stride phase (radians, includes the per-instance desync seed);
/// `body_len` is the visible body extension in world units (the instance
/// radius for archetype rows); `plan` is the BodyPlan discriminant
/// (0..3): 0 Primitive, 1 SegmentedArthropod, 2 MultiLegged, 3 Quadruped.
/// Deterministic: same inputs -> identical fields (testable without GPU).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    /// world-unit +Y lift (added by the VS after grounding)
    pub bob: f32,
    /// small-angle pitch (rad, ~ bob derivative)
    pub pitch: f32,
    /// small-angle roll (rad, ~ sway derivative)
    pub roll: f32,
    /// stride phase in radians (write-through, drives VS leg offsets)
    pub phase: f32,
    /// limb swing amplitude in world units (CODE_LIMB verts only)
    pub leg_amp: f32,
    /// limb vertical lift amplitude in world units
    pub leg_lift: f32,
    /// phase-locked walk squash amplitude (scale_y 1 +- s)
    pub squash: f32,
    /// slow ambient breathing amplitude (scale_y 1 +- s)
    pub breathe: f32,
    /// yaw sway in radians (view-side heading offset)
    pub sway: f32,
}

/// See [`Motion`]/the LOCO_* constant docs. `speed` is the smoothed
/// per-organism speed estimate (world u/s, position delta per frame / dt).
pub fn motion_for(plan: u8, body_len: f32, speed: f32, phase: f32, jointedness: f32, armour: f32) -> Motion {
    let p = (plan as usize).min(3);
    let bl = body_len.max(0.05);
    let f = gait_freq(bl);
    let walking = speed > LOCO_IDLE_EPS;
    let sp01 = if walking { (speed / LOCO_FULL).clamp(0.0, 1.0) } else { 0.0 };
    let joint = jointedness.clamp(0.0, 1.0);
    let arm = armour.clamp(0.0, 1.0);
    let two_pi_f = std::f32::consts::TAU * f;

    // yaw sway (view-side heading offset) + roll coupling of its derivative
    let sway_amp = LOCO_SWAY * sp01 * SWAY_PLAN_BOOST[p];
    let sway = sway_amp * (0.5 * phase).sin();
    let roll = (LOCO_ROLL_K * sway_amp * (0.5 * phase).cos()).clamp(-LOCO_ROLL_MAX, LOCO_ROLL_MAX);

    // body bob ~ speed; jointed bodies absorb; pitch ~ bob velocity
    let bob_amp = LOCO_BOB * sp01 * (1.0 - LOCO_JOINT_DAMP * joint) * BOB_PLAN_SCALE[p] * bl;
    let bob = bob_amp * phase.sin();
    let pitch = (LOCO_PITCH_K * two_pi_f * bob_amp / bl.max(0.05) * phase.cos())
        .clamp(-LOCO_PITCH_MAX, LOCO_PITCH_MAX);

    // limb stride: plan-scaled swing + lift (quadruped diagonal pairs and
    // the arthropod mid-row offset imply the tripod tread via the VS bucket
    // + side-parity tables; the head nod anti-phase rides the CODE_HEAD id)
    let leg_scale = LEG_PLAN_SCALE[p] * (1.0 - LOCO_LEG_JOINT_DAMP * joint);
    let leg_amp = if walking { LOCO_LEG_AMP * sp01 * leg_scale * bl } else { 0.0 };
    let leg_lift = leg_amp * LOCO_LEG_LIFT;

    // squash-stretch: Primitive bodies breathe hardest while walking
    // (shell-plating stiffens), tighter body squash for the others
    let squash_w = if walking {
        let base = if p == 0 { LOCO_SQUASH_WALK } else { LOCO_SQUASH_BODY };
        base * sp01 * (1.0 - LOCO_ARMOUR_DAMP * arm)
    } else {
        0.0
    };
    // ambient breathe: full idle swell, shallow while moving
    let breathe = if walking { LOCO_WALK_BREATHE } else { LOCO_IDLE_BREATHE };

    Motion { bob, pitch, roll, phase, leg_amp, leg_lift, squash: squash_w, breathe, sway }
}

/// Per-organism view-side locomotion state keyed by the sim id
/// (`OrgInst.id`). Advancing an id consumes the planner's position delta
/// against the cached previous position (speed = delta / frame_dt, same
/// estimator pattern as the queen lane), smooths the estimate and accrues
/// the stride phase at the body-planned frequency. Dead ids are pruned by a
/// generation stamp on every pass so the map holds exactly the live set.
#[derive(Clone, Copy)]
struct LocoTrack {
    px: f32,
    py: f32,
    /// smoothed speed estimate (world u/s)
    speed: f32,
    /// accrued stride phase (radians)
    phase: f32,
    gen: u64,
}

pub struct LocoTracks {
    map: HashMap<u64, LocoTrack>,
    /// engine frame dt (seconds); gait speed = distance/dt
    dt: f32,
    /// per-pass stamp (see [`LocoTracks::begin_pass`])
    pass: u64,
}

impl LocoTracks {
    pub fn new() -> LocoTracks {
        LocoTracks { map: HashMap::with_capacity(1024), dt: 1.0 / 60.0, pass: 0 }
    }

    /// Engine frame dt (seconds); NaN/negative dt is ignored (previous value
    /// stays, default 1/60); capped at 0.25 s like the queen lane. Visual-only.
    pub fn set_frame_dt(&mut self, dt: f32) {
        if dt.is_finite() && dt > 0.0 {
            self.dt = dt.min(0.25);
        }
    }

    /// Bumps the pass stamp. Call once at the start of every plan:
    /// every id advanced in this pass is stamped with the new value, so
    /// [`LocoTracks::prune`] (at the pass end) drops exactly the dead set.
    pub fn begin_pass(&mut self) {
        self.pass = self.pass.wrapping_add(1);
    }

    /// Advances one organism's state from its new position. Returns
    /// `(accrued_phase, smoothed_speed)`; callers add the per-instance
    /// desync seed to the phase. ids that never advance in a pass keep
    /// their state across LOD hops; the first frame of a new id starts at
    /// rest so there is no speed spike.
    pub fn advance(&mut self, id: u64, x: f32, y: f32) -> (f32, f32) {
        let p = self.pass;
        let dt = self.dt;
        let st = self.map.entry(id).or_insert(LocoTrack {
            px: x,
            py: y,
            speed: 0.0,
            phase: 0.0,
            gen: p,
        });
        st.gen = p;
        let dx = x - st.px;
        let dy = y - st.py;
        let d2 = dx * dx + dy * dy;
        st.px = x;
        st.py = y;
        if d2.is_finite() && d2 < LOCO_TELEPORT2 {
            let inst = (d2.sqrt() / dt.max(1e-4)).max(0.0);
            st.speed += (inst - st.speed) * LOCO_SPEED_LERP;
        } else {
            st.speed = 0.0; // teleport/spawn: no speed spike
        }
        // stride phase advances only while moving (idle limbs settle frozen;
        // parked ids keep the bytes stable across replans). Engagement cap:
        // rate tapers by gait engagement during spin-up.
        if st.speed > LOCO_IDLE_EPS {
            let gait01 = (st.speed / LOCO_FULL).clamp(0.0, 1.0);
            let rate = LOCO_IDLE_RATE + (1.0 - LOCO_IDLE_RATE) * gait01;
            st.phase += std::f32::consts::TAU * rate * dt;
        }
        if !st.phase.is_finite() {
            st.phase = 0.0;
            st.speed = 0.0;
        }
        if st.phase > std::f32::consts::TAU * 1024.0 {
            st.phase = st.phase.rem_euclid(std::f32::consts::TAU * 1024.0);
        }
        (st.phase, st.speed)
    }

    /// Per-pass prune: every advanced id was stamped with the current pass
    /// stamp (see [`LocoTracks::begin_pass`]), so a single retain drops
    /// exactly the dead set. Beyond [`LOCO_TRACK_CAP`] the whole map resets
    /// (bounded state over continuity).
    pub fn prune(&mut self) {
        if self.map.len() >= LOCO_TRACK_CAP {
            self.map.clear();
            return;
        }
        let p = self.pass;
        self.map.retain(|_, t| t.gen >= p);
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
    queen_scr: Vec<(f32, u32)>,
    tails: Vec<MorphInst>,
    shells: Vec<MorphInst>,
    queens: Vec<MorphInst>,
    quads: Vec<MorphInst>,
    /// per-queen visual smoothing state (index = queen pre-pass slot)
    queen_state: [Option<QueenVis>; QUEEN_CAP],
    /// (x, z, radius, strength) trample footprint of the nearest planned
    /// queen this frame; Some only when at least one queen rendered
    queen_foot: Option<(f32, f32, f32, f32)>,
    /// engine frame dt (set by the coordinator); gait speed = distance/dt
    frame_dt: f32,
}

impl Planner {
    pub fn new() -> Planner {
        Planner {
            decomp: DecompCache::new(),
            out: FramePlan {
                insts: Vec::with_capacity(INST_CAP),
                tail_n: 0,
                shell_n: 0,
                queen_n: 0,
                quad_n: 0,
            },
            morph_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            bill_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            ovf_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            cand_scr: Vec::with_capacity(MAX_ORGS_SCAN),
            queen_scr: Vec::with_capacity(QUEEN_CAP * 4),
            tails: Vec::with_capacity(TAIL_CAP as usize),
            shells: Vec::with_capacity(SHELL_CAP as usize),
            queens: Vec::with_capacity(QUEEN_CAP),
            quads: Vec::with_capacity(QUAD_CAP as usize),
            queen_state: [None, None],
            queen_foot: None,
            frame_dt: 1.0 / 60.0,
        }
    }

    /// Engine frame dt (seconds) used to estimate queen gait speed from her
    /// cached position delta. Coordinator calls this once per frame before
    /// `plan`; NaN/negative dt is ignored, the previous value (default
    /// 1/60) stays. Visual-only.
    pub fn set_frame_dt(&mut self, dt: f32) {
        if dt.is_finite() && dt > 0.0 {
            self.frame_dt = dt.min(0.25);
        }
    }

    /// Grass-trample hook: footprint of the nearest queen planned THIS frame
    /// as `(queen_x, queen_z, radius, strength)`. Radius ~
    /// 0.9 * queen visual body length (~14 units) — grass inside that ring
    /// bends away (WGSL `GrassUniform.trample`). `None` while no queen is
    /// rendered, or between plans. The lead wires this into
    /// `grass.rs:401` (`GrassUniform { trample: .. }` build site).
    pub fn queen_footprint(&self) -> Option<(f32, f32, f32, f32)> {
        self.queen_foot
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
        let qfar2 = QUEEN_FAR * QUEEN_FAR;
        let lod_ring2 = 450.0 * 450.0;
        self.morph_scr.clear();
        self.bill_scr.clear();
        self.queen_scr.clear();
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
            if o.is_queen != 0 || o.brand == BRAND_QUEEN {
                // priority route: the queen ALWAYS renders as a full morph,
                // never as a billboard; her slot is reserved and she never
                // competes with the ordinary full-morph / billboard scan
                if d2 < qfar2 {
                    self.queen_scr.push((d2, i as u32));
                }
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
        self.queen_scr.sort_by(cmp);

        self.tails.clear();
        self.shells.clear();
        self.queens.clear();
        self.quads.clear();
        self.queen_foot = None;

        // ---- queen pre-pass (priority before the ordinary ring budget) ----
        let mut ring_queens: u32 = 0;
        // nearest-first; every admitted queen renders her full template. She
        // is "reduced detail" beyond the 450-unit ring, silhouette identical.
        for k in 0..self.queen_scr.len().min(QUEEN_CAP) {
            let (d2, i) = self.queen_scr[k];
            let o = &orgs[i as usize];
            let e = o.energy.clamp(0.0, 1.0);
            let size = o.size.clamp(0.3, 2.0);
            let body_r = queen_body_radius(size, e);
            let gy = height_at(o.x, o.y);
            // seat the template ground line (QUEEN_GROUND_LIFT below the
            // origin) on the terrain: queen world Y always touches the
            // ground here — ordinates of the lobes cancel out along yaw
            let base = [o.x, gy + body_r * QUEEN_GROUND_LIFT, o.y];
            if !visible(base, body_r * 2.0) {
                continue;
            }
            // visual heading smoothing + gait speed (cached position delta;
            // position jumps beyond QUEEN_TELEPORT2 reset the state)
            let mut heading = o.heading;
            let mut gait01 = 0.0f32;
            {
                let mut st = self.queen_state[k].take();
                let continuous = st
                    .filter(|s| {
                        let dx = o.x - s.x;
                        let dz = o.y - s.y;
                        dx * dx + dz * dz < QUEEN_TELEPORT2
                    })
                    .is_some();
                if let Some(s) = st.as_mut() {
                    if continuous {
                        // shortest-angle chase toward the engine heading
                        s.heading += shortest_angle(s.heading, o.heading) * QUEEN_HEADING_TURN;
                        heading = s.heading;
                        let dx = o.x - s.x;
                        let dz = o.y - s.y;
                        let speed = (dx * dx + dz * dz).sqrt() / self.frame_dt;
                        gait01 = (speed / QUEEN_GAIT_REF).clamp(0.0, 1.0);
                        s.x = o.x;
                        s.y = o.y;
                    }
                }
                if !continuous {
                    st = Some(QueenVis { x: o.x, y: o.y, heading: o.heading });
                }
                self.queen_state[k] = st;
            }
            let own_phase = inst_phase(o.hue) + (o.x * 0.031 + o.y * 0.017); // per-id desync
            let wave = (1.1 + 0.5 * e + 2.5 * gait01).min(QUEEN_GAIT_MAX_WAVE);
            self.queens.push(MorphInst {
                pos_x: base[0], pos_y: base[1], pos_z: base[2],
                heading,
                radius: body_r,
                breathe: 0.05 + 0.03 * (1.0 - e),
                hue: o.hue.rem_euclid(1.0),
                age: o.age.clamp(0.0, 1.0),
                alpha: o.alive.clamp(0.0, 1.0),
                flags: 0.0,
                phase: own_phase,
                turgor: 0.35 + 0.65 * e,
                wave, // idle pulse scaled up with movement (gait speed)
                pad: if d2 < lod_ring2 { 1.0 } else { 0.0 },
                motion: [0.0; 4],
                drive: [0.0; 4],
            });
            // grass-trample footprint of the nearest on-screen queen
            if self.queen_foot.is_none() {
                let vlen = QUEEN_BODY_LEN * body_r;
                self.queen_foot = Some((o.x, o.y, 0.9 * vlen, 1.0));
            }
            // honesty about the shared budget: inside the morph ring she
            // consumes one of the MODEL_CAP full-morph organism slots
            if d2 < mr2 {
                ring_queens += 1;
            }
        }
        let mut kept = ring_queens;
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
                        motion: [0.0; 4],
                        drive: [0.0; 4],
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
                        motion: [0.0; 4],
                        drive: [0.0; 4],
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
            let radius = ORGANISM_UNIT_RADIUS * o.size.clamp(0.3, 2.0) * (0.62 + 0.38 * e);
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
                motion: [0.0; 4],
                drive: [0.0; 4],
            });
        }

        // assemble the contiguous draw ranges: [tails][shells][queens][billboards]
        let tail_n = self.tails.len() as u32;
        let shell_n = self.shells.len() as u32;
        let queen_n = self.queens.len() as u32;
        self.out.insts.clear();
        self.out.insts.extend_from_slice(&self.tails);
        self.out.insts.extend_from_slice(&self.shells);
        self.out.insts.extend_from_slice(&self.queens);
        self.out.insts.extend_from_slice(&self.quads);
        self.out.tail_n = tail_n;
        self.out.shell_n = shell_n;
        self.out.queen_n = queen_n;
        self.out.quad_n = self.quads.len() as u32;
        debug_assert_eq!(
            (tail_n + shell_n + queen_n + self.out.quad_n) as usize,
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
        queen_n: n.queen_n,
        quad_n: n.quad_n,
    }
}
