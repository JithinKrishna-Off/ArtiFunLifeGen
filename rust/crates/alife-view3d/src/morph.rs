//! Procedural organism morphology. Maps expressed phenotype (traits) to
//! morphology brands and renders them as translucent instanced biology:
//! displaced icosphere membranes, bezier flagella whose wave is animated in
//! the vertex shader, chained filament segments and camera-facing billboards
//! for distant organisms.
//!
//! GPU rules (mirrors grass.rs):
//! - `Morph::new(&device)` builds the prebaked template meshes, buffers and
//!   bind group layouts (nothing here depends on the target color format).
//! - The coordinates coordinator must call `ensure_pipeline(&device,
//!   color_format, depth_format)` once in `View3D::new`, and
//!   `update_gpu(&device, &queue, &frame)` each frame (queue is not
//!   reachable from `draw_with`). Until `ensure_pipeline`, `draw_with`
//!   draws nothing.
//! - Depth is standard forward-Z (`LessEqual`) to match terrain/grass; the blend is plain alpha (translucent membranes); no SSAO / read-depth tricks.
//! - Group 0 binds a morph-owned copy of the same `CamUniform` block used
//!   by terrain/grass; animation time rides in `.world_wh_time_flash.z` and
//!   lightning flash in `.w`, so there is no extra uniform.
//!
//! DRAW STRUCTURE (A2 organism archetype pass):
//! - The `decomp::Planner` still runs every frame UNCHANGED (locomotion
//!   agent owns it): its QUEEN rows are the only planner instances drawn
//!   (queen template + reserved states are byte-identical to before); the
//!   planner's legacy organism rows (tail/shell/billboard) remain CPU-side
//!   planner diagnostics and are NOT drawn anymore.
//! - Ordinary organisms are planned in [`OrganismPass`]: rows are GROUPED by
//!   archetype key (mirroring `creature_mesh::archetype_key` quantization);
//!   each group = one static shared mesh (`&'static [TVert]`, LRU-cached)
//!   + one contiguous instance-buffer slice = one draw call. Groups ≤
//!   `DETAILED_CAP` rows, blob group (QUAD billboard template) ≤ 240.
//! - LOD: d < [`NEAR_RING`] and d < [`MID_RING`] draw the full archetype
//!   mesh (tier already solved at build, ≤ 240 verts — A1 keeps ONE mesh
//!   per archetype); beyond `MID_RING` to `quad_radius` organisms render as
//!   the cheap shared billboard blob (hue via instance channel); beyond
//!   that they are culled. Crowd overflow spills to the blob route.
//! - Queen + selection-marker passes are structurally untouched: queen rows
//!   still come from the planner buffer; `draw_markers` unchanged.

#[path = "morph_templates.rs"]
pub mod templates;
#[path = "morph_decomp.rs"]
pub mod decomp;
#[path = "morph_queen.rs"]
pub mod queen;
#[path = "morph_tests.rs"]
#[cfg(test)]
mod tests;

use alife_common::{Organism, Traits};
use crate::creature_mesh as cmet;
use crate::scene::{CameraMode, Frame3D, OrgInst};
use decomp::{INST_CAP, INST_STRIDE, MODE_FOLLOW, MODE_ORBIT, MODE_PLANET};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use templates::TVert;
use wgpu::util::DeviceExt;

/// Morphology signature classes (uint brand):
pub const BRAND_CELL: u32 = 0;
pub const BRAND_FLAGELLATE: u32 = 1;
pub const BRAND_COLONY: u32 = 2;
pub const BRAND_FILAMENT: u32 = 3;
/// Queen brand (morphological emphasis; is_queen selector).
pub const BRAND_QUEEN: u32 = 4;

/// Deterministic expressed-trait plumbing -> morphology brand. Pure function
/// of the organism's contemporary traits (no hashing, no randomness; the
/// renderer must never invent anatomy a trait table cannot support).
/// Fallback: filaments/colonies when lineages grow large/life stages long.
/// Colony rule: slow, cheap-moving, long-range-sensing bodies express a
/// sessile 3-lobed multi-cell cluster life form (decomp colony).
pub fn brand_of(o: &Organism) -> u32 {
    let t = &o.genotype.traits;
    if t.size < 0.85 {
        BRAND_CELL
    } else if t.max_speed > 65.0 {
        BRAND_FLAGELLATE
    } else if t.size > 1.25 {
        BRAND_FILAMENT
    } else if t.max_speed <= 45.0 && t.efficiency < 1.0 && t.sensor_range >= 120.0 {
        BRAND_COLONY
    } else {
        BRAND_CELL
    }
}

/// Golden-angle species hue (matches the 2D renderer palette walk).
pub fn hue_of(species: u32) -> f32 {
    ((species as f64 * 0.618_033_988_749_894_9) % 1.0) as f32
}

/// Trait -> visual shape knobs the agents use to build meshes.
pub struct MorphoParam {
    pub body_radius: f32,
    pub limb_count: u32,
    pub length_factor: f32,
}

/// Render-style deltas for a distinguished individual (the Queen). Same
/// templates, same anatomy: hue nudge + body emphasis + livelier membrane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphStyle {
    /// radians of hue rotation to add (positive = warmer); keeps hue in [0,1)
    pub hue_shift: f32,
    /// body-scale emphasis multiplier
    pub size_scale: f32,
    /// membrane breathing amplitude multiplier
    pub breathe_scale: f32,
}

/// Returns the style for an organism; queens get the distinctive boost, all
/// others neutral (no visual data path changes for the rest of the field).
/// `size_scale` = 2.0: with the giant rebalance (`decomp::QUEEN_SCALE`) the
/// queen clamp target is trait size >= 1.0 -> visual size 2.0, which yields
/// her ~16-unit crown-to-tail body length (giant vs the tiny ordinary
/// re-scale via `decomp::ORGANISM_UNIT_RADIUS` = 0.75, ~1.2-1.5-unit bodies;
/// 10-20x apparent ratio).
pub fn style_for(org_is_queen: bool) -> MorphStyle {
    if org_is_queen {
        MorphStyle { hue_shift: 0.12, size_scale: 2.0, breathe_scale: 1.8 }
    } else {
        MorphStyle { hue_shift: 0.0, size_scale: 1.0, breathe_scale: 1.0 }
    }
}

/// Applies `style_for` onto a render instance in place. The coordinator calls
/// this when the queen flag is known (identity kept in the sim, not here).
/// Anatomy is unaffected: `org_Build` derives shape from brand + trait-size
/// bucket only, and bits are baked rather than scaled function is a pure
/// rescale, so the queen shares the same cached template soup.
pub fn apply_style(inst: &mut crate::scene::OrgInst, org_is_queen: bool) {
    let st = style_for(org_is_queen);
    inst.hue = (inst.hue.rem_euclid(1.0) + st.hue_shift).rem_euclid(1.0);
    inst.size = (inst.size * st.size_scale).clamp(0.3, 2.0);
    // breathing amplitude derives from energy in the planner; scale by the
    // queen factor by nudging energy upward within its normalized band
    if st.breathe_scale > 1.0 {
        inst.energy = (inst.energy * st.breathe_scale / 1.3).clamp(0.35, 1.0);
    }
}

pub fn morpho_param(t: &Traits) -> MorphoParam {
    MorphoParam {
        body_radius: 0.5 + 0.5 * t.size.clamp(0.3, 1.8) / 1.6,
        limb_count: (2.0 * t.sensor_range / 100.0).clamp(0.0, 6.0) as u32,
        length_factor: 0.6 + 0.8 * (t.metabolism / 1.9).clamp(0.2, 1.2),
    }
}

// ============================================================ A2: organism
// Organism archetype-rendering pass (pure CPU planning here; GPU buffer
// management lives in `Morph`). The decomp planner is untouched; queen rows
// keep flowing through it. Everything below draws ORDINARY organisms from
// A1's archetype meshes (creature_mesh.rs).

/// LOD: nearer than this counts as "near" (diag lane 1).
pub const NEAR_RING: f32 = 60.0;
/// LOD: detail ring — full archetype meshes only inside this radius (world
/// units); beyond it creatures render as the shared billboard blob.
pub const MID_RING: f32 = 180.0;
/// Hard per-frame slots for full archetype-mesh instances (nearest admitted
/// first; crowd overflow spills to the blob route). One instance = one whole
/// creature mesh.
pub const DETAILED_CAP: usize = 128;
/// Far blob slots: mirrors the legacy `decomp::QUAD_CAP`.
pub const BLOB_CAP: usize = decomp::QUAD_CAP as usize;

/// Planner-side fallback archetype key: bit 62 so it can never collide with
/// creature_mesh's real-archetype space (bit 63-free) or its brand-fallback
/// keys (bit 63 set). Used when the lead has not wired per-row descriptors.
#[inline]
pub fn organism_fallback_key(brand: u32) -> u64 {
    0x4000_0000_0000_0000 | (brand as u64 & 0x3)
}

/// Mesh-space ground footprint (local BL units, as authored by creature_mesh;
/// verts scale by the instance radius).
#[derive(Clone, Copy)]
pub(crate) struct Footprint {
    pub min_y: f32,
    pub max_y: f32,
}

/// Deterministic scan of a baked mesh: lowest + highest vertex Y.
pub(crate) fn mesh_footprint(mesh: &[TVert]) -> Footprint {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for v in mesh {
        lo = lo.min(v.pos[1]);
        hi = hi.max(v.pos[1]);
    }
    if !lo.is_finite() {
        lo = -1.0;
    }
    if !hi.is_finite() {
        hi = 1.0;
    }
    Footprint { min_y: lo, max_y: hi }
}

/// Mesh y-hull as world units at the given instance radius.
#[inline]
pub(crate) fn footprint_radius(fp: &Footprint, r: f32) -> f32 {
    ((fp.max_y - fp.min_y) * r * 0.5).max(0.02)
}

/// `hue`-derived animation phase (same walk as the decomp convention).
#[inline]
fn inst_phase(hue: f32) -> f32 {
    hue.rem_euclid(1.0) * std::f32::consts::TAU * 7.0
}

/// Animation-rate multiplier per brand (decomp convention), clamped to the
/// queen-safe gait band so archetype body breathing stays calm: the shared
/// 6.x VS branch multiplies breathing by `wave`, so flagellate-class tails
/// ride the same instance and must not quiver the trunk.
#[inline]
fn wave_rate_clamped(brand: u32, turgor01: f32) -> f32 {
    let raw = match brand {
        1 => 5.2 + 4.8 * turgor01,  // flagellate beat
        3 => 0.9 + 0.7 * turgor01,  // filament wobble
        2 => 1.3 + 0.5 * turgor01,  // colony idle
        _ => 1.6 + 0.6 * turgor01,  // plain cell
    };
    raw.clamp(1.0, 3.6)
}

/// Morphology cache: signature -> (archetype key, leaked-static mesh).
///
/// Strategy per the integration contract: geometry is shared per
/// `archetype_key` (creature_mesh LRU, ≤ 72 leaks), while the per-organism
/// `CreatureMorph` re-derivation is what the cache avoids on warm frames.
/// Without descriptors the plan derives nothing at all (brand fallbacks only),
/// and rows still resolve through the cache; with the lead's row-archetype
/// hook (`Morph::set_row_archetypes`) the cache refreshes an entry whenever a
/// hooked key differs (e.g. a genome-hash dress field drifted -> new
/// archetype), so an identical organism never rebuilds on the warm path.
/// Slices are leaked-static (creature_mesh graveyard guarantees validity),
/// so eviction can never dangle; entries are discarded wholesale at the LRU
/// cap (4096) like the decomp decomposition cache.
pub struct MorphCache {
    map: HashMap<u64, (u64, &'static [TVert])>,
    lru: VecDeque<u64>,
    /// cache misses (rebuilds) during the last `OrganismPass::plan`
    pub last_pass_rebuilds: usize,
}

const MORPH_CACHE_CAP: usize = 4096;

impl MorphCache {
    fn new() -> MorphCache {
        MorphCache {
            map: HashMap::new(),
            lru: VecDeque::new(),
            last_pass_rebuilds: 0,
        }
    }

    /// Warm path: refresh LRU and return the cached entry; rebuild only on a
    /// miss (or when a per-row hook key disagrees with the cached one).
    fn resolve(
        &mut self,
        sig: u64,
        want: Option<u64>,
        make: impl FnOnce() -> (u64, &'static [TVert]),
    ) -> (u64, &'static [TVert]) {
        if let Some(&(k, m)) = self.map.get(&sig) {
            if want.map_or(true, |w| w == k) {
                if let Some(pos) = self.lru.iter().position(|&s| s == sig) {
                    self.lru.remove(pos);
                    self.lru.push_back(sig);
                }
                return (k, m);
            }
        }
        if self.map.len() >= MORPH_CACHE_CAP {
            self.map.clear();
            self.lru.clear();
        }
        let entry = make();
        self.map.insert(sig, entry);
        self.lru.push_back(sig);
        self.last_pass_rebuilds += 1;
        entry
    }
}

/// One pass' organism render plan: archetype-grouped detailed rows + far
/// blob rows, all in the exact `decomp::MorphInst` layout (the WGSL instance
/// attributes are unchanged in shape — 22 x f32, locations 2..=21; the A3
/// locomotion fields ride locations 14..=21 and stay zero on non-creature
/// lanes so the queen template and blob paths decode identically).
pub struct OrgPlan {
    /// detailed rows, uniform within each `groups` range
    pub arch: Vec<decomp::MorphInst>,
    /// (archetype key, first row index, row count); keys ascending
    pub groups: Vec<(u64, u32, u32)>,
    /// far blob rows (QUAD billboard template), farthest-first
    pub blobs: Vec<decomp::MorphInst>,
}

impl OrgPlan {
    fn new() -> OrgPlan {
        OrgPlan {
            arch: Vec::with_capacity(DETAILED_CAP),
            groups: Vec::new(),
            blobs: Vec::with_capacity(BLOB_CAP),
        }
    }

    fn clear(&mut self) {
        self.arch.clear();
        self.groups.clear();
        self.blobs.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.arch.is_empty() && self.blobs.is_empty()
    }
}

/// Pure-CPU organism scheduler (no device state; tests drive it directly).
/// Scratch buffers are pre-allocated in `new` and reused every frame.
pub struct OrganismPass {
    det_scr: Vec<(f32, u32)>,
    far_scr: Vec<(f32, u32)>,
    rows_scr: Vec<(u64, decomp::MorphInst)>,
    spill_scr: Vec<(f32, u32)>,
    cache: MorphCache,
    /// A3: per-organism (sim id) locomotion state — stride phase + gait
    /// speed from cached position deltas (visual only). Prunes dead ids
    /// (by generation stamp) at the end of every `plan`.
    loco: decomp::LocoTracks,
    /// archetype key -> mesh (distinct keys this session; diag lane 2)
    key_mesh: HashMap<u64, &'static [TVert]>,
    /// (near, mid, far) counts of the last pass
    lod: (u32, u32, u32),
    /// skipped rows (dead/queen/culled/over-cap) last pass
    skipped: u32,
}

impl OrganismPass {
    pub fn new() -> OrganismPass {
        OrganismPass {
            det_scr: Vec::with_capacity(1600),
            far_scr: Vec::with_capacity(1600),
            rows_scr: Vec::with_capacity(DETAILED_CAP),
            spill_scr: Vec::with_capacity(1600),
            cache: MorphCache::new(),
            loco: decomp::LocoTracks::new(),
            key_mesh: HashMap::new(),
            lod: (0, 0, 0),
            skipped: 0,
        }
    }

    /// A3: engine frame dt (seconds) for the locomotion gait-speed
    /// estimator (position delta / dt). Visual only, mirrors the planner's
    /// queen estimator. NaN/negative dt is ignored.
    pub fn set_frame_dt(&mut self, dt: f32) {
        self.loco.set_frame_dt(dt);
    }

    /// (live cache entries, distinct archetype meshes, rebuilds last pass)
    pub fn diag(&self) -> (usize, usize, usize) {
        (self.cache.map.len(), self.key_mesh.len(), self.cache.last_pass_rebuilds)
    }

    /// (near, mid, far) instance counts of the last pass.
    pub fn lod_counts(&self) -> (u32, u32, u32) {
        self.lod
    }

    /// Skipped-row count of the last pass (diag).
    pub fn skipped_count(&self) -> u32 {
        self.skipped
    }

    /// Mesh slice for an archetype key this session (GPU-buffer build path).
    pub(crate) fn mesh_of(&self, key: u64) -> Option<&'static [TVert]> {
        self.key_mesh.get(&key).copied()
    }

    /// Resolve the (key, mesh, canonical descriptor) for one row. `hook`
    /// delivers the lead's per-row real `archetype_key` when wired;
    /// otherwise the brand fallback route applies (A1
    /// `organism_brand_fallback` + `fallback_morph(brand)` shape). The
    /// canonical descriptor is the shared-mesh-canonical anatomy: the A3
    /// locomotion model reads body_plan / jointedness / armour from it so
    /// gait style follows exactly the mesh that is drawn.
    fn resolve(
        &mut self,
        o: &OrgInst,
        hook: Option<u64>,
    ) -> (u64, &'static [TVert], alife_common::morphology::CreatureMorph) {
        let sig = decomp::signature(o);
        let brand = o.brand & 0x3;
        match hook {
            Some(hkey) => {
                // Real descriptor route: the hook key IS the archetype key; the
                // mesh registry is LRU-cached + leak-static, `ensure_mesh`
                // builds on the first miss only. Cache entry refresh keeps the
                // warm path free of registry traffic when the key is stable.
                let (key, mesh) = self.cache.resolve(sig, Some(hkey), || {
                    (hkey, cmet::ensure_mesh(hkey))
                });
                let canon = cmet::canonical_from_key(key);
                self.key_mesh.insert(key, mesh);
                (key, mesh, canon)
            }
            None => {
                // Brand fallback route: generic archetype mesh per plan
                // (BodyPlan discriminant == brand), used when no sim-derived
                // CreatureMorph is available (the frame snapshot carries no
                // genotype). Shape constants come from the documented
                // generic descriptors (~1.39 world units at nominal turgor).
                let fkey = organism_fallback_key(brand);
                let (key, mesh) = self.cache.resolve(sig, None, || {
                    (fkey, crate::morph::templates::organism_brand_fallback(brand))
                });
                let fm = cmet::fallback_morph(brand);
                self.key_mesh.insert(key, mesh);
                (key, mesh, fm)
            }
        }
    }

    /// Build one detailed row: organism pose + A1 scale recipe + A3 body
    /// motion (locomotion is view-side only; the sim row is authoritative).
    fn build_row(
        &mut self,
        o: &OrgInst,
        hook: Option<u64>,
        height_at: &mut dyn FnMut(f32, f32) -> f32,
        visible: &mut dyn FnMut([f32; 3], f32) -> bool,
    ) -> bool {
        let (key, mesh, m) = self.resolve(o, hook);
        let e = o.energy.clamp(0.0, 1.0);
        let size = o.size.clamp(0.3, 2.0);
        let tf = 0.62 + 0.38 * e; // existing energy/size modulation chain
        // A1 recipe: instance radius = RENDER_SCALE * body_length, with the
        // legacy size/turgor modulation kept on top of the nominal scale.
        let r = (cmet::RENDER_SCALE * m.body_length * size * tf).max(0.05);
        let fp = mesh_footprint(mesh);
        // seat the mesh's lowest vertex on the terrain (heading-rotation
        // invariants: yaw only, so the hull lift is heading-independent)
        let gy = height_at(o.x, o.y);
        let base_y = gy - fp.min_y * r;
        if !visible([o.x, base_y, o.y], footprint_radius(&fp, r)) {
            return false;
        }
        // A3: per-organism (sim id) stride phase + gait speed from the
        // cached position delta, then the pure body-motion model. The
        // visible body length is the instance radius (BL world extent).
        let (trk_phase, speed) = self.loco.advance(o.id, o.x, o.y);
        let phase = trk_phase + inst_phase(o.hue) + o.x * 0.031 + o.y * 0.017;
        let mo = decomp::motion_for(
            m.body_plan as u8,
            r,
            speed,
            phase,
            m.jointedness,
            m.armour,
        );
        self.rows_scr.push((
            key,
            decomp::MorphInst {
                pos_x: o.x,
                pos_y: base_y,
                pos_z: o.y,
                // yaw sway rides the row's own heading float (view-only)
                heading: o.heading + mo.sway,
                radius: r,
                breathe: 0.05 + 0.08 * (1.0 - e),
                hue: o.hue.rem_euclid(1.0),
                age: o.age.clamp(0.0, 1.0),
                alpha: o.alive.clamp(0.0, 1.0),
                flags: 0.0,
                phase: inst_phase(o.hue),
                turgor: 0.35 + 0.65 * e,
                wave: wave_rate_clamped(o.brand, e),
                // sat_tint rides the pad channel (0.15..0.45, strictly inside
                // (0, 0.5) so queen rows pad∈{0,1} can never be tinted)
                pad: m.sat_tint.clamp(0.0, 0.49),
                // A3 locomotion block
                motion: [mo.bob, mo.pitch, mo.roll, mo.phase],
                drive: [mo.leg_amp, mo.leg_lift, mo.squash, mo.breathe],
            },
        ));
        true
    }

    /// One legacy blob (billboard) row: byte-identical conventions to the
    /// planner's billboard pass (radius chain, distance fade), only the fade
    /// window starts at [`MID_RING`].
    fn build_blob(
        &mut self,
        o: &OrgInst,
        d: f32,
        mode: u8,
        out: &mut OrgPlan,
        height_at: &mut dyn FnMut(f32, f32) -> f32,
        visible: &mut dyn FnMut([f32; 3], f32) -> bool,
    ) -> bool {
        let e = o.energy.clamp(0.0, 1.0);
        let size = o.size.clamp(0.3, 2.0);
        let radius = decomp::ORGANISM_UNIT_RADIUS * size * (0.62 + 0.38 * e);
        let qr = decomp::quad_radius(mode);
        let fade = (1.0 - (d - MID_RING) / (qr - MID_RING)).clamp(0.0, 1.0);
        let gy = height_at(o.x, o.y);
        let center = [o.x, gy + radius * 0.62, o.y];
        if !visible(center, radius) {
            return false;
        }
        out.blobs.push(decomp::MorphInst {
            pos_x: center[0],
            pos_y: center[1],
            pos_z: center[2],
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
            // far blobs never animate (pad 0 -> the VS motion gate is off)
            motion: [0.0; 4],
            drive: [0.0; 4],
        });
        true
    }

    /// Pure-CPU organism scheduling. `row_keys`: optional lead hook, parallel
    /// to `orgs` (real `archetype_key(derive(org))` per row; `None` = brand
    /// fallbacks). `height_at`/`visible` mirror the decomp planner contract.
    /// Rows land in `out` (scratch-only borrow; the coordinator passes the
    /// mutex-guarded plan slot, tests pass a local).
    pub fn plan(
        &mut self,
        orgs: &[OrgInst],
        row_keys: Option<&[u64]>,
        view: [f32; 3],
        mode: u8,
        out: &mut OrgPlan,
        height_at: &mut dyn FnMut(f32, f32) -> f32,
        visible: &mut dyn FnMut([f32; 3], f32) -> bool,
    ) {
        self.cache.last_pass_rebuilds = 0;
        self.skipped = 0;
        self.det_scr.clear();
        self.far_scr.clear();
        self.rows_scr.clear();
        self.loco.begin_pass(); // pass stamp for the end-of-plan dead-id prune
        out.clear();
        let qr = decomp::quad_radius(mode);
        let mid2 = MID_RING * MID_RING;
        let qr2 = qr * qr;
        for (i, o) in orgs.iter().take(decomp::MAX_ORGS_SCAN).enumerate() {
            if o.alive <= 0.0 || o.is_queen != 0 || o.brand == crate::morph::BRAND_QUEEN {
                self.skipped += (o.alive > 0.0 && (o.is_queen != 0 || o.brand == crate::morph::BRAND_QUEEN)) as u32;
                continue; // queens render via the planner's reserved pass
            }
            let dx = o.x - view[0];
            let dz = o.y - view[2];
            let d2 = dx * dx + dz * dz;
            if !d2.is_finite() {
                self.skipped += 1;
                continue;
            }
            if d2 < mid2 {
                self.det_scr.push((d2, i as u32));
            } else if d2 < qr2 {
                self.far_scr.push((d2, i as u32));
            } else {
                self.skipped += 1;
            }
        }
        let cmp = |a: &(f32, u32), b: &(f32, u32)| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        };
        self.det_scr.sort_by(cmp);
        // farthest-first blob candidates (back-to-front alpha blend)
        self.far_scr.sort_by(|a, b| cmp(b, a));

        // ---- detailed rows, nearest-first, capped; crowd overflow spills to
        // the blob route (shared with the far pool, sorted desc below)
        let (mut near, mut mid) = (0u32, 0u32);
        self.spill_scr.clear();
        for k in 0..self.det_scr.len() {
            let (d2, i) = self.det_scr[k];
            if self.rows_scr.len() >= DETAILED_CAP {
                self.spill_scr.push((d2, i));
                continue;
            }
            let hook = row_keys.and_then(|k| k.get(i as usize).copied());
            if self.build_row(&orgs[i as usize], hook, height_at, visible) {
                if d2.sqrt() < NEAR_RING {
                    near += 1;
                } else {
                    mid += 1;
                }
            } else {
                self.skipped += 1;
            }
        }

        // ---- far blobs + detail-crowd spill, farthest-first
        for k in 0..self.far_scr.len() {
            let (d2, i) = self.far_scr[k];
            if out.blobs.len() >= BLOB_CAP {
                self.skipped += 1;
                continue;
            }
            if !self.build_blob(&orgs[i as usize], d2.sqrt(), mode, out, height_at, visible) {
                self.skipped += 1;
            }
        }
        self.spill_scr.sort_by(|a, b| cmp(b, a));
        for k in 0..self.spill_scr.len() {
            let (d2, i) = self.spill_scr[k];
            if out.blobs.len() >= BLOB_CAP {
                self.skipped += 1;
                continue;
            }
            if !self.build_blob(&orgs[i as usize], d2.sqrt(), mode, out, height_at, visible) {
                self.skipped += 1;
            }
        }

        // ---- group by archetype key (contiguous, key-ascending for a
        // deterministic, cache-friendly draw order)
        self.rows_scr.sort_by_key(|(k, _)| *k);
        let mut gi = 0usize;
        while gi < self.rows_scr.len() {
            let key = self.rows_scr[gi].0;
            let mut gj = gi;
            while gj < self.rows_scr.len() && self.rows_scr[gj].0 == key {
                gj += 1;
            }
            let first = out.arch.len() as u32;
            for (_, row) in &self.rows_scr[gi..gj] {
                out.arch.push(*row);
            }
            out.groups.push((key, first, (gj - gi) as u32));
            gi = gj;
        }
        self.lod = (near, mid, out.blobs.len() as u32);
        debug_assert!((out.arch.len() + out.blobs.len()) <= DETAILED_CAP + BLOB_CAP);
        self.loco.prune(); // drop tracks of ids that never advanced this pass
    }
}

fn mode_of(m: &CameraMode) -> u8 {
    match m {
        CameraMode::Follow { .. } => MODE_FOLLOW,
        CameraMode::Planet => MODE_PLANET,
        CameraMode::Orbit { .. } => MODE_ORBIT,
    }
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
}

/// Lazily created GPU vertex buffer for one archetype mesh (stable `&'static`
/// mesh slice -> one buffer, built once on the group's first draw).
struct ArchBuf {
    buf: wgpu::Buffer,
    vcount: u32,
}

/// Everything the organism pass needs; buffers are allocated exactly once
pub struct Morph {
    gpu: Option<Gpu>,
    /// prebaked template geometry (vertices uploaded once at new())
    verts_buf: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    /// per-frame planner; on the happy path this never touches the heap
    planner: Option<Mutex<decomp::Planner>>,
    /// A2: archetype vertex buffers, keyed by archetype key (created in
    /// update_gpu where the device is reachable; draw only reads).
    arch_bufs: HashMap<u64, ArchBuf>,
    /// A2: instance buffer for archetype-group rows + blob rows
    org_inst_buf: wgpu::Buffer,
    /// A2: last organism plan (locked hand-off from update_gpu to draw_with)
    org_plan: Mutex<OrgPlan>,
    /// A2 lead-integration hook: optional per-row archetype keys parallel to
    /// `Frame3D.orgs`, consumed by the next `update_gpu` (then cleared).
    row_keys: Mutex<Option<Vec<u64>>>,
    /// A2: pure-CPU organism scheduler (scratch + MorphCache + LOD counts)
    organisms: OrganismPass,
    /// selection/hover marker overlay (one extra instanced draw, <= 3
    /// instances); lazily pipelined with `ensure_pipeline`
    pub markers: crate::scene::markers::MarkerRig,
}

impl Morph {
    /// Builds CPU + most GPU state; the render pipeline is created by
    /// [`ensure_pipeline`] with the real target formats. `new` never fails.
    pub fn new(device: &wgpu::Device) -> Option<Morph> {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.morph.bgl.cam"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::scene::CamUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        // template soup: [shell 1440][tail 1152][billboard 6][queen ~3930] verts
        let t = templates::build();
        let mut verts: Vec<TVert> = Vec::with_capacity((t.vcount() + templates::QUEEN_VCOUNT) as usize);
        verts.extend_from_slice(&t.shell);
        verts.extend_from_slice(&t.tail);
        verts.extend_from_slice(&t.quad);
        verts.extend_from_slice(&queen::build());
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.morph.verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.morph.insts"),
            size: (INST_CAP * INST_STRIDE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A2: archetype-group rows [+ blob rows] instance ring
        let org_inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.morph.org.insts"),
            size: ((DETAILED_CAP + BLOB_CAP) * INST_STRIDE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.morph.cam"),
            size: std::mem::size_of::<crate::scene::CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.morph.cam.bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        Some(Morph {
            gpu: None,
            verts_buf,
            inst_buf,
            cam_buf,
            cam_bg,
            planner: Some(Mutex::new(decomp::Planner::new())),
            arch_bufs: HashMap::new(),
            org_inst_buf,
            org_plan: Mutex::new(OrgPlan::new()),
            row_keys: Mutex::new(None),
            organisms: OrganismPass::new(),
            markers: crate::scene::markers::MarkerRig::new(),
        })
    }

    /// One-time pipeline construction; the coordinator should call this in
    /// `View3D::new` with the same formats passed to `Terrain::new`
    /// (`rs.target_format` which is `Bgra8UnormSrgb` on Windows desktops,
    /// `Depth32Float` depth). Until then `draw_with` draws nothing.
    pub fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.morph.bgl.cam"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::scene::CamUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view3d.morph.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.morph.vs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MORPH_VS)),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.morph.fs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MORPH_FS)),
        });
        let t_attrs: Vec<wgpu::VertexAttribute> = (0u32..2)
            .map(|i| wgpu::VertexAttribute {
                offset: if i == 0 { 0 } else { 16 },
                shader_location: i,
                format: if i == 0 { wgpu::VertexFormat::Float32x4 } else { wgpu::VertexFormat::Float32x2 },
            })
            .collect();
        // Exactly 16 instance attributes (wgpu's default max_vertex_attributes
        // limit): pos(3) @2, 11 scalars @3..=13, then the A3 locomotion
        // pair packed as two vec4s: `motion` @14 (offset 56) and `drive`
        // @15 (offset 72). Never-animated lanes keep those bytes at zero so
        // the queen path is untouched.
        let mut i_attrs = vec![wgpu::VertexAttribute {
            offset: 0,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x3,
        }];
        for i in 3..14u32 {
            i_attrs.push(wgpu::VertexAttribute {
                offset: i as u64 * 4,
                shader_location: i,
                format: wgpu::VertexFormat::Float32,
            });
        }
        i_attrs.push(wgpu::VertexAttribute {
            offset: 56,
            shader_location: 14,
            format: wgpu::VertexFormat::Float32x4,
        });
        i_attrs.push(wgpu::VertexAttribute {
            offset: 72,
            shader_location: 15,
            format: wgpu::VertexFormat::Float32x4,
        });
        let pipeline = device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("view3d.morph.pline"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &vs_mod,
                    entry_point: "vs",
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 24,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &t_attrs,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: INST_STRIDE as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &i_attrs,
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_mod,
                    entry_point: "fs",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // blobs render from both sides when the camera zooms in
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    // standard forward-Z (matches terrain/grass): translucent
                    // pass must not write depth so membranes layer by draw order
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        self.gpu = Some(Gpu { pipeline });
        self.markers.ensure(device, color_format, depth_format);
    }

    /// TEMP diag: current plan bucket counts.
    ///
    /// NOTE (A2): since the organism archetype pass took over ordinary
    /// creature rendering, the tail/shell/billboard lanes are PLANNER-INTERNAL
    /// diagnostics only (the live draw consumes the queen lane from this
    /// buffer). Use [`diag_lod`] + [`diag_morph_cache`] for the live path.
    pub fn diag_plan(&self) -> (u32, u32, u32, u32) {
        match self.planner.as_ref() {
            Some(p) => {
                let g = p.lock().unwrap_or_else(|err| err.into_inner());
                let c = g.current();
                (c.tail_n, c.shell_n, c.queen_n, c.quad_n)
            }
            None => (0, 0, 0, 0),
        }
    }

    /// A2 diag: (live MorphCache entries, distinct archetype meshes in use,
    /// cache rebuilds during the last pass).
    pub fn diag_morph_cache(&self) -> (usize, usize, usize) {
        self.organisms.diag()
    }

    /// A2 diag: (near, mid, far) instance counts of the last organism pass.
    pub fn diag_lod(&self) -> (u32, u32, u32) {
        self.organisms.lod_counts()
    }

    /// A2 diag: rows skipped by the last organism pass
    /// (dead rows do not count; queens routing to the planner do).
    pub fn diag_skipped(&self) -> u32 {
        self.organisms.skipped_count()
    }

    /// A2 lead-integration hook: optional per-row archetype keys parallel to
    /// `Frame3D.orgs` (`creature_mesh::archetype_key(&morphology::derive(org))`
    /// computed at the UI build site, where the real organisms are in scope).
    /// The vector is consumed by the NEXT `update_gpu` (then cleared), so it
    /// must be re-armed every frame it applies. With the hook armed, rows
    /// render true per-genome archetype meshes; without it they render the
    /// brand fallback archetypes (same BodyPlan mapping, generic shape).
    pub fn set_row_archetypes(&mut self, keys: Option<Vec<u64>>) {
        *self.row_keys.lock().unwrap_or_else(|e| e.into_inner()) = keys;
    }

    /// QUEEN-only: this frame's queen trample footprint from the planner as
    /// `(queen_x, queen_z, radius, strength)`; `None` when no queen was
    /// planned (or before the first `update_gpu`). The lead feeds this into
    /// the grass `GrassUniform.trample` build site (grass.rs:401).
    pub fn queen_footprint(&self) -> Option<(f32, f32, f32, f32)> {
        match self.planner.as_ref() {
            Some(p) => p.lock().unwrap_or_else(|err| err.into_inner()).queen_footprint(),
            None => None,
        }
    }

    /// Per-frame planning + uploads; the coordinator calls this in
    /// `View3D::update` right before/after grass's `update_gpu`.
    ///
    /// Sequence: (1) the untouched decomp planner still plans queen rows
    /// (grass trample + queen states come from it); (2) only the QUEEN rows
    /// of its output are uploaded to the legacy instance buffer; (3) the pure
    /// [`OrganismPass`] plans ordinary creatures into archetype-grouped rows
    /// and far blobs; (4) group meshes get lazily-uploaded vertex buffers and
    /// the row bytes land in `org_inst_buf` ([arch rows | blob rows]).
    pub fn update_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame3D) {
        queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&frame.cam));
        self.markers.update(queue, frame);
        let Some(planner) = self.planner.as_ref() else { return };
        let mode = mode_of(&frame.cam_mode);
        let view = [frame.cam.view_pos[0], frame.cam.view_pos[1], frame.cam.view_pos[2]];
        // height + frustum live with terrain/grass-owned modules (read-only)
        let frustum = crate::grass::frustum::Frustum::from_cam(&frame.cam.view, &frame.cam.proj);
        let mut guard = planner.lock().unwrap_or_else(|p| p.into_inner());
        // QUEEN-only: engine dt for gait-speed estimation from position delta
        guard.set_frame_dt(frame.dt);
        // A3: ordinary creatures estimate gait speed the same way (OrganismPass)
        self.organisms.set_frame_dt(frame.dt);
        let okay = guard.plan(
            &frame.orgs,
            view,
            mode,
            &mut |x: f32, y: f32| crate::terrain::terrain_height_at(&frame.world, x, y) as f32,
            &mut |c: [f32; 3], r: f32| frustum.sphere_visible(c, r),
        );
        // ---- queen-only upload: slice [tail_n + shell_n .. + queen_n) so the
        // draw path keeps reading queen rows from this buffer at offset 0
        let (tn, sn, qun) = (okay.tail_n as usize, okay.shell_n as usize, okay.queen_n as usize);
        if qun > 0 {
            let all = bytemuck::cast_slice::<decomp::MorphInst, u8>(&okay.insts);
            queue.write_buffer(&self.inst_buf, 0, &all[(tn + sn) * INST_STRIDE..(tn + sn + qun) * INST_STRIDE]);
        }
        // ---- A2 organism archetype pass (planner untouched; separate plan).
        // Frame-armed keys (`Frame3D.org_arch`, preferred) or the one-shot
        // `set_row_archetypes` hook; empty/unarmed -> brand fallback meshes.
        let hook = if !frame.org_arch.is_empty() {
            Some(frame.org_arch.clone())
        } else {
            self.row_keys.lock().unwrap_or_else(|e| e.into_inner()).take()
        };
        let mut plan_guard = self.org_plan.lock().unwrap_or_else(|e| e.into_inner());
        self.organisms.plan(
            &frame.orgs,
            hook.as_deref(),
            view,
            mode,
            &mut plan_guard,
            &mut |x: f32, y: f32| crate::terrain::terrain_height_at(&frame.world, x, y) as f32,
            &mut |c: [f32; 3], r: f32| frustum.sphere_visible(c, r),
        );
        // lazily upload one vertex buffer per drawn archetype mesh
        for &(key, _, count) in &plan_guard.groups {
            if count == 0 || self.arch_bufs.contains_key(&key) {
                continue;
            }
            if let Some(mesh) = self.organisms.mesh_of(key) {
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("view3d.morph.org.arch"),
                    contents: bytemuck::cast_slice(mesh),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                self.arch_bufs.insert(key, ArchBuf { buf, vcount: mesh.len() as u32 });
            }
        }
        // rows: [arch | blobs]
        if !plan_guard.is_empty() {
            let arch_bytes = bytemuck::cast_slice::<decomp::MorphInst, u8>(&plan_guard.arch);
            if !arch_bytes.is_empty() {
                queue.write_buffer(&self.org_inst_buf, 0, arch_bytes);
            }
            let blob_bytes = bytemuck::cast_slice::<decomp::MorphInst, u8>(&plan_guard.blobs);
            if !blob_bytes.is_empty() {
                queue.write_buffer(&self.org_inst_buf, (plan_guard.arch.len() * INST_STRIDE) as u64, blob_bytes);
            }
        }
        drop(plan_guard);
    }

    /// Draw: QUEEN rows from the planner buffer (template soup), then one
    /// instanced draw per archetype group (shared static mesh + contiguous
    /// instance slice), then the far blob group (QUAD billboard template).
    /// alpha order: detailed creatures -> queen -> far blobs (farthest first).
    /// No-op until ensure_pipeline + first successful update_gpu.
    pub fn draw_with<'a>(&self, pass: &mut wgpu::RenderPass<'a>, frame: Option<&crate::scene::Frame3D>) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(frame) = frame else { return };
        let _ = frame;
        let Some(planner) = self.planner.as_ref() else { return };
        // NOTE: frame's visuals were already planned in update_gpu; the
        // planner mutex hands the queen rows to the paint path.
        let plan_guard = planner.lock().unwrap_or_else(|p| p.into_inner());
        let plan = plan_guard.current();
        let org_guard = self.org_plan.lock().unwrap_or_else(|p| p.into_inner());
        if plan.queen_n == 0 && org_guard.is_empty() {
            return;
        }
        let stride = INST_STRIDE as u64;
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);
        // ---- QUEEN: reserved template (unchanged path, planner-owned data)
        pass.set_vertex_buffer(0, self.verts_buf.slice(..));
        if plan.queen_n > 0 {
            pass.set_vertex_buffer(1, self.inst_buf.slice(0..(plan.queen_n as u64) * stride));
            pass.draw(
                templates::QUEEN_VBASE..templates::QUEEN_VBASE + templates::QUEEN_VCOUNT,
                0..plan.queen_n,
            );
        }
        // ---- archetype groups: one shared mesh buffer + one instance slice
        for &(key, first, count) in &org_guard.groups {
            if count == 0 {
                continue;
            }
            if let Some(ab) = self.arch_bufs.get(&key) {
                pass.set_vertex_buffer(0, ab.buf.slice(..));
                pass.set_vertex_buffer(
                    1,
                    self.org_inst_buf
                        .slice((first as u64) * stride..(first as u64 + count as u64) * stride),
                );
                pass.draw(0..ab.vcount, 0..count);
            }
        }
        // ---- far blobs: camera-facing billboard template, farthest-first
        if !org_guard.blobs.is_empty() {
            pass.set_vertex_buffer(0, self.verts_buf.slice(..));
            let a = org_guard.arch.len() as u64;
            let b = (org_guard.arch.len() + org_guard.blobs.len()) as u64;
            pass.set_vertex_buffer(1, self.org_inst_buf.slice(a * stride..b * stride));
            pass.draw(
                templates::QUAD_VBASE..templates::QUAD_VBASE + templates::QUAD_VCOUNT,
                0..org_guard.blobs.len() as u32,
            );
        }
    }

    /// Selection/hover marker overlay: ONE instanced draw call (<= 3
    /// camera-facing ring quads), drawn right after the organism body pass.
    /// No-op until `ensure_pipeline` armed it and no marker is set.
    pub fn draw_markers(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.markers.draw(pass);
    }
}

const MORPH_VS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct VIn {
    @location(0) tpos: vec4<f32>,      // template local pos (radius units)
    @location(1) taux: vec2<f32>,      // (param, code id)
    @location(2) ipos: vec3<f32>,      // instance origin (world)
    @location(3) iheading: f32,
    @location(4) iradius: f32,
    @location(5) ibreathe: f32,
    @location(6) ihue: f32,
    @location(7) iage: f32,
    @location(8) ialpha: f32,
    @location(9) iflags: f32,
    @location(10) iphase: f32,
    @location(11) iturgor: f32,
    @location(12) iwave: f32,
    @location(13) ipad: f32,
    // A3 locomotion extension (append-only; never-animated lanes keep 0.0):
    // imotion = (bob_y world units, pitch rad, roll rad, stride phase rad)
    // idrive  = (leg swing world, leg lift world, walk squash, breathe amp)
    @location(14) imotion: vec4<f32>,
    @location(15) idrive: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) info: vec4<f32>,      // (id, param, alpha, hue)
    @location(1) nuv: vec4<f32>,       // sphere normal .xyz | tail t in .w | billboard corner .xy
    @location(2) misc: vec4<f32>,      // (turgor, age, wave, phase)
};

// A2: queen rows carry pad = 0.0/1.0 (LOD flag; queen stubs/antennae are
// authored in world units). Ordinary archetype rows carry pad = sat_tint in
// (0, 0.5) — the gate below switches template scaling + tinting so the
// queen path stays visually byte-identical.
fn creature_gate(pad: f32) -> bool {
    return (pad > 0.001) && (pad < 0.5);
}

@vertex
fn vs(i: VIn) -> VOut {
    let id = i.taux.y;
    let t = cam.world_wh_time_flash.z;
    let r = max(i.iradius, 0.02);
    var local = i.tpos.xyz;
    // ORDINARY body silhouettes (ids 0..2.5 - queen ids start at 6.0) from
    // the unscaled template position:
    // - gentle prolate ellipsoid squash along heading (+X);
    // - teardrop taper when the flagellate shell carries the silhouette code
    //   floor(flags / 10) == 1 (frac(flags) keeps the nucleus gate).
    let ell = vec3<f32>(1.14, 0.86, 0.94);
    let body_s = clamp(0.5 + 0.5 * local.x, 0.0, 1.0);
    let taper = mix(1.0, 0.5 + 0.55 * body_s * body_s, floor(i.iflags / 10.0));
    var a_out = i.ialpha;
    var nuv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (id < 0.5) {
        // membrane: breathing sine (+ spatial delay so chains/crowds travel)
        let delay = (i.ipos.x + i.ipos.z) * 0.13;
        let breath = 0.55 * sin(t * i.iwave + i.iphase + delay)
                   + 0.30 * sin(t * i.iwave * 1.7 + i.iphase * 2.1 + delay * 1.3);
        local = local * taper * ell * (1.0 + i.ibreathe * breath) * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), nuv.w);
    } else if (id < 1.5) {
        // cytoplasm core: soft inner counter-pulse
        local = local * taper * ell * (1.0 + i.ibreathe * 0.35 * sin(t * i.iwave + i.iphase + 0.7)) * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), nuv.w);
    } else if (id < 2.5) {
        // nucleus-like organelle: alpha-gated by the trait table
        let gate = step(0.5, fract(i.iflags));
        a_out = a_out * gate;
        local = local * taper * ell * (1.0 + i.ibreathe * 0.2 * sin(t * i.iwave * 0.8 + i.iphase)) * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), nuv.w);
    } else if (id < 4.5) {
        // flagella/tail: vertex-shader wave along the chord (param in x).
        // A2: archetype-mesh tails are authored in body-length units and
        // scale by the instance radius; queen/legacy rows keep the unscaled
        // world-unit convention (pad gate).
        let ss = select(1.0, r, creature_gate(i.ipad));
        local = local * ss;
        let tt = i.taux.x;
        let amp = 0.42 * ss * (0.25 + 0.75 * tt);
        let ph = tt * 8.0 - t * i.iwave + i.iphase;
        local.y = local.y + amp * sin(ph);
        local.z = local.z + amp * 0.7 * sin(ph * 0.83 + 1.3);
        // second tail collapsed when this species grows one flagellum
        if (id > 3.5 && floor(i.iflags / 10.0) < 2.0) {
            local = local * 0.0;
            a_out = 0.0;
        }
        nuv = vec4<f32>(0.0, 0.0, 0.0, tt);
    } else if (id < 5.5) {
        // billboard disc: camera-facing expansion around the world center
        let va = vec3<f32>(cam.view[0][0], cam.view[1][0], cam.view[2][0]);
        let vu = vec3<f32>(cam.view[0][1], cam.view[1][1], cam.view[2][1]);
        local = va * (i.tpos.x * r) + vu * (i.tpos.y * r);
        nuv = vec4<f32>(i.tpos.x, i.tpos.y, 0.0, 0.0);
    } else if (id < 7.0) {
        // queen sphere lobes (head 6.0 / thorax 6.25 / abdomen 6.5):
        // regular breathing + pulse; abdomen carries ridge bands along +X
        var breathe = 1.0 + i.ibreathe * 0.14 * sin(t * i.iwave * 0.8 + i.iphase);
        if (id > 6.6) {
            breathe = breathe * (1.0 + 0.045 * sin(local.x * 7.0));
        }
        local = local * breathe * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), i.ipad);
    } else if (id < 7.5) {
        // antennae + limb chains: gentle wave (amp tapers to tip). A2:
        // archetype limbs are authored in body-length units and scale by the
        // instance radius; queen rows keep the prebaked world-unit stubs
        // (pad gate).
        let ss = select(1.0, r, creature_gate(i.ipad));
        local = local * ss;
        let tt = i.taux.x;
        let amp = 0.05 * ss * (0.3 + 0.7 * (1.0 - tt));
        local.y = local.y + amp * sin(tt * 7.0 - t * i.iwave * 0.7 + i.iphase);
        nuv = vec4<f32>(normalize(i.tpos.xyz), tt);
    } else {
        // crown spikes: dignified slow shimmer
        local = local * (1.0 + 0.035 * sin(t * 1.7 + i.iphase * 0.5)) * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), 0.0);
    }
    // A3 body-motion block: creature rows only (pad gate keeps the queen
    // template rows pad 0/1 and blob rows pad 0 byte-identical to before;
    // membrane/core ids < 2.5 never exist on archetype meshes). The drive
    // fields are CPU-computed per frame from the organism's actual gait
    // speed (position delta / dt) — pure view-side motion on top of the
    // authoritative sim row. Ovum-stage safety: any future egg-stage row
    // convention (pad == 0) automatically fails `creature_gate` and does
    // not animate.
    if (creature_gate(i.ipad) && id >= 2.5) {
        // small-angle pitch/roll about the local origin (equivalent to
        // rotating after the heading yaw about the heading-local axes)
        let cp = cos(i.imotion.y);
        let sp = sin(i.imotion.y);
        let cr = cos(i.imotion.z);
        let sr = sin(i.imotion.z);
        let p0 = vec2<f32>(local.x, local.y);
        local.x = p0.x * cp - p0.y * sp;
        local.y = p0.x * sp + p0.y * cp;
        let r0 = vec2<f32>(local.z, local.y);
        local.z = r0.x * cr - r0.y * sr;
        local.y = r0.x * sr + r0.y * cr;
        // vertical bob lifts ABOVE the grounded origin. Per-part animation
        // is unavailable per-instance, but the static material ids are:
        // CODE_LIMB verts below the trunk midline are leg verts (keep 3/4
        // of their bob so the feet stay near the ground), CODE_HEAD verts
        // (6.25) nod anti-phase to the rear (quadruped head counter-bob).
        let leg_v = (id > 6.9 && id < 7.1 && i.tpos.y < 0.0);
        var gain = 1.0;
        if (id > 6.2 && id < 6.3) {
            gain = -0.6; // head nod anti-phase
        } else if (leg_v) {
            gain = 0.25; // feet stay near the ground
        }
        local.y = local.y + i.imotion.x * gain;
        // limb stride swing/lift: per-leg phase = accrued phase + template-x
        // bucket offset {front:+0, mid:+PI/2, rear:+PI} + z side parity
        // (left/right anti-phase rows => diagonal pairs / tripod tread)
        if (leg_v && i.idrive.x > 0.0) {
            let bucket = select(1.5707963, 3.1415926, i.tpos.x < -0.24);
            let b2 = select(0.0, bucket, i.tpos.x < 0.0);
            let th = i.imotion.w + b2 + select(3.1415926, 0.0, i.tpos.z > 0.0);
            local.x = local.x + i.idrive.x * sin(th);
            local.y = local.y + i.idrive.y * max(0.0, sin(th + 1.0));
        }
        // squash-stretch along body Y: walk-coupled quarter-lag term (drive.z,
        // phase-locked to the stride via imotion.w) + slow ambient breathing
        // (drive.w, rides the instance wave/phase channels). Legs skip it so
        // feet are not pumped.
        let squash = i.idrive.z * sin(i.imotion.w - 1.5707963)
                   + i.idrive.w * sin(t * i.iwave * 0.55 + i.iphase);
        if (!leg_v) {
            local.y = local.y * (1.0 + squash);
        }
    }
    var world = vec3<f32>(0.0);
    if (id != 5.0) {
        // heading rotation: template +X -> world (cos h, 0, sin h)
        let c = cos(i.iheading);
        let s = sin(i.iheading);
        let rot = vec3<f32>(local.x * c - local.z * s, local.y, local.x * s + local.z * c);
        world = i.ipos + rot;
    } else {
        world = i.ipos + local;
    }
    var o: VOut;
    o.clip = cam.proj * cam.view * vec4(world, 1.0);
    o.info = vec4<f32>(id, i.taux.x, a_out, i.ihue);
    o.nuv = nuv;
    o.misc = vec4<f32>(i.iturgor, i.iage, i.iwave, i.iphase);
    return o;
}
"#;

const MORPH_FS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct FsIn {
    @location(0) info: vec4<f32>,    // (id, param, alpha, hue)
    @location(1) nuv: vec4<f32>,
    @location(2) misc: vec4<f32>,    // (turgor, age, wave, phase)
};

// A2 sat_tint channel: body-branch verts ride sat_tint in nuv.w (instance
// pad -> VS nuv.w). Exactly neutral (1.0) for queen rows (pad = 0.0/1.0).
fn creature_tint(w: f32) -> f32 {
    return select(1.0, 0.85 + 0.5 * w, (w > 0.001) && (w < 0.5));
}

fn hsv(h: f32, s: f32, val: f32) -> vec3<f32> {
    let i = i32(floor(h * 6.0));
    let f = h * 6.0 - floor(h * 6.0);
    let p = val * (1.0 - s);
    let q = val * (1.0 - s * f);
    let tt = val * (1.0 - s * (1.0 - f));
    var rgb = vec3(val);
    if (i % 6 == 0) {
        rgb = vec3(val, tt, p);
    } else if (i % 6 == 1) {
        rgb = vec3(q, val, p);
    } else if (i % 6 == 2) {
        rgb = vec3(p, val, tt);
    } else if (i % 6 == 3) {
        rgb = vec3(p, q, val);
    } else if (i % 6 == 4) {
        rgb = vec3(tt, p, val);
    } else {
        rgb = vec3(val, p, q);
    }
    return rgb;
}

@fragment
fn fs(fi: FsIn) -> @location(0) vec4<f32> {
    let id = fi.info.x;
    let a_in = clamp(fi.info.z, 0.0, 1.0);
    let hue = fract(fi.info.w);
    let turg = clamp(fi.misc.x, 0.0, 1.0);
    let age = clamp(fi.misc.y, 0.0, 1.0);
    let t = cam.world_wh_time_flash.z;
    let flash = clamp(cam.world_wh_time_flash.w, 0.0, 1.0);
    let sun = normalize(vec3<f32>(0.35, 0.55, 0.85));
    // aging desaturates and pales the pigment; drained cells go paler too
    let sat = clamp(0.62 - 0.34 * age - 0.12 * (1.0 - turg), 0.10, 1.0);
    var col = vec3<f32>(0.0);
    var a_mat = 1.0;
    if (id < 0.5) {
        // translucent membrane: rim-lit shell
        let n = normalize(fi.nuv.xyz);
        let lam = clamp(abs(dot(n, sun)), 0.0, 1.0);
        let rim = clamp(1.0 - lam, 0.0, 1.0);
        col = hsv(hue, sat * 0.9, mix(0.30, 0.52, lam)) + rim * 0.16;
        a_mat = 0.36 + 0.12 * turg;
    } else if (id < 1.5) {
        // cytoplasm core
        let n = normalize(fi.nuv.xyz);
        let lam = clamp(abs(dot(n, sun)), 0.0, 1.0);
        col = hsv(hue, sat, mix(0.26, 0.60, lam)) * 0.9;
        a_mat = 0.85;
    } else if (id < 2.5) {
        // nucleus-like organelle, brighter with turgor
        col = hsv(hue, sat * 0.6, 0.62 + 0.30 * turg);
        a_mat = 0.50;
    } else if (id < 4.5) {
        // flagella: darker tip + a subtle beat shimmer
        let tt = fi.info.y;
        let shimmer = 0.5 + 0.5 * sin(tt * 6.0 - t * fi.misc.z * 2.0);
        col = hsv(hue, sat * 0.85, mix(0.42, 0.60, tt)) * mix(0.85, 1.0, shimmer);
        a_mat = 0.72;
    } else if (id < 5.5) {
        // billboard: soft radial edge
        let rr = clamp(length(fi.nuv.xy), 0.0, 1.0);
        let edge = 1.0 - smoothstep(0.55, 1.0, rr);
        col = hsv(hue, sat * 0.9, 0.55);
        a_mat = edge * 0.85;
    } else if (id < 7.0) {
        // queen body lobes / creature body-head-eye-wing: gum-like body
        // color, rim-lit like the membrane but more opaque; brighter with
        // turgor + a subtle energy pulse. A2: per-genome sat_tint nudges the
        // chroma (neutral for queens + default fallback descriptors).
        let n = normalize(fi.nuv.xyz);
        let lam = clamp(abs(dot(n, sun)), 0.0, 1.0);
        let rim = clamp(1.0 - lam, 0.0, 1.0);
        let pulse = 0.03 * sin(t * 1.4);
        col = hsv(hue, sat * 0.88 * creature_tint(fi.nuv.w), mix(0.34, 0.56, lam) + pulse) + rim * 0.14;
        a_mat = 0.78;
    } else if (id < 7.5) {
        // antennae + limbs: darker chitin
        let tt = fi.info.y;
        col = hsv(hue, sat * 0.7, mix(0.32, 0.40, tt));
        a_mat = 0.82;
    } else {
        // crown: gold spikes with a slow dignity shimmer
        let shimmer = 0.5 + 0.5 * sin(t * 1.7);
        col = hsv(0.13, 0.55, 0.78 + 0.07 * shimmer);
        a_mat = 0.95;
    }
    var c = col;
    c += flash * 0.25;
    return vec4(c, a_in * a_mat);
}
"#;

