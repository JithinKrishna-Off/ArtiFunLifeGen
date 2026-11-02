//! Ray-based organism picking (Agent D). Pure math, zero GPU; allocates only
//! on the click-quality diagnostics path. Derives the view ray through a
//! mouse pixel from the camera matrices, intersects it with the ground plane
//! at the average terrain height near the camera focus, and returns the
//! nearest rendered organism inside a distance-scaled tolerance radius
//! (`max(2.0, dist * 0.05)` world units, capped at 400) widened by the
//! candidate's pick radius ([`pick_radius`]). Rays that never reach the
//! ground (grazing horizons) are clamped to a bounded search distance;
//! candidates behind the camera or farther than 400 world units from the
//! ground hit are rejected outright.
//!
//! Returns the index into `orgs` (slice order == the UI's id list order);
//! the UI maps that index to a real organism id.
//!
//! # Public API (stable names for the UI agent)
//!
//! - [`pick_org`] — unchanged click entry: `Option<u32>` index.
//! - [`hover_org`] — per-frame, allocation-free hover: same click-quality
//!   math, returns `Option<PickCandidate>` (the ray is computed once and
//!   reused inside; call it every frame while the pointer is over the
//!   viewport, it is the cheap path).
//! - [`pick_org_with_diag`] — one-shot, builds a bounded candidate list for
//!   a diagnostics overlay (see [`PickDiag`]); only call it on click or when
//!   the overlay is open (it allocates a `Vec`, capped at
//!   [`MAX_DIAG_CANDIDATES`]).
//! - [`ray_dir`] / [`tol_at`] / [`ground_height_near`] — unchanged helpers.
//!
//! # Queen-aware tolerance
//!
//! `OrgInst::visual_radius_mul` overrides the per-candidate pick radius:
//! ordinary organisms (mul `= 1.0`, `<= 0.0` or non-finite, the
//! `OrgInst::zeroed` default) keep the historical `size`-clamp proxy
//! bit-identical, while the queen (mul =
//! [`crate::morph::decomp::QUEEN_SCALE`] ~= 0.4435 after the giant
//! rebalance, previously 2.7, set by `scene::org_instance`) widens the hit
//! ring to her true visible body radius
//! `QUEEN_SCALE * 5.5 * size * (0.62 + 0.38 * energy)` — the exact
//! body-radius formula the queen planner uses (morph_decomp.rs, queen
//! pre-pass), so crown clicks land on what is actually rendered.

#[cfg(test)]
#[path = "picking_tests.rs"]
mod tests;

use crate::scene::OrgInst;
use glam::{Mat4, Vec3};

/// Farthest ray distance we still sample (bounds precision loss + runtime
/// for rays that graze the horizon instead of hitting the ground plane).
pub const MAX_RAY_DIST: f32 = 3000.0;
/// Absolute cap on the pick tolerance (world units): beyond ~400 units of
/// aim error a pick is rejected no matter the ray distance.
pub const MAX_TOL: f32 = 400.0;
/// Hard rejection radius around the ground hit (world units).
pub const MAX_PICK_DIST: f32 = 400.0;
/// Hard clamp on queen-style widened pick radius (world units): the largest
/// possible queen body is `QUEEN_SCALE * 5.5 * size(2.0) * turgor(1.0) =
/// QUEEN_UNIT_RADIUS * 2.0 ~= 4.9` after the giant rebalance; the 30 cap
/// (kept from the pre-rebalance scale) stays a harmless generous ceiling.
pub const MAX_PICK_BODY_R: f32 = 30.0;
/// Candidate-list cap for [`PickDiag::candidates`].
pub const MAX_DIAG_CANDIDATES: usize = 8;

/// Pick tolerance at a given ray distance, world units: grows linearly with
/// distance but never below 2.0 nor above [`MAX_TOL`].
#[inline]
pub fn tol_at(dist: f32) -> f32 {
    (2.0f32).max(dist.max(0.0) * 0.05).min(MAX_TOL)
}

/// One candidate hit of the pick ray. `dist` is the 3D travel distance of
/// the ray to the candidate (front-of-camera check already applied);
/// `hit_dist` is the planar (ground-plane) distance between the ray's ground
/// hit and the organism — the value compared against the tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickCandidate {
    /// index into the `orgs` slice handed to the pick call
    pub org_index: usize,
    /// 3D distance along the view ray where the candidate sits
    pub dist: f32,
    /// planar distance from the ray's ground hit to the organism (what the
    /// tolerance test compares)
    pub hit_dist: f32,
}

/// Diagnostics for a pick: ordered nearest-first candidate list (capped),
/// the base hit ring radius, the ray and the local ground plane it used.
/// Built only by [`pick_org_with_diag`] (click-quality, allocates).
#[derive(Clone, Debug)]
pub struct PickDiag {
    /// nearest-first hits inside tolerance, capped at
    /// [`MAX_DIAG_CANDIDATES`]; empty when the ray was invalid or missed.
    pub candidates: Vec<PickCandidate>,
    /// base pick tolerance at the ground hit (`tol_at` of the ray travel to
    /// the plane) — the number every candidate's widened tolerance derives
    /// from; 0.0 when the ray never formed.
    pub hit_radius: f32,
    /// normalized view ray direction; all zeros when no ray could be built.
    pub ray_dir: [f32; 3],
    /// the (sanitized) ground height the plane was laid at
    pub ground_h: f32,
    /// mirrors the returned pick (`Option<usize>` of the winner index)
    pub selected: Option<usize>,
}

/// Per-candidate pick radius multiplier source (world units of the tolerance
/// ring scale). Ordinary candidates (`visual_radius_mul <= 0.0` or `= 1.0` —
/// including the `OrgInst::zeroed` default 0.0) keep the historical
/// `size`-clamp proxy bit-identical. Overridden candidates (the queen,
/// `visual_radius_mul = QUEEN_SCALE` ~= 0.4435 after the giant rebalance;
/// previously 2.7) widen to their true visible body radius, mirroring the
/// queen planner's body radius formula
/// (`QUEEN_SCALE * 5.5 * size * turgor`). Purely visual: the sim never
/// reads this.
#[inline]
pub fn pick_radius(o: &OrgInst) -> f32 {
    let mul = o.visual_radius_mul;
    let widened = mul.is_finite() && mul > 0.0 && mul != 1.0;
    if !widened {
        return o.size.clamp(0.6, 2.2);
    }
    // mirrors morph_decomp::Planner's queen pre-pass:
    // body_r = QUEEN_SCALE * 5.5 * size * tf, tf = 0.62 + 0.38 * energy
    let e = o.energy.clamp(0.0, 1.0);
    let tf = 0.62 + 0.38 * e;
    let size = o.size.clamp(0.3, 2.0);
    (5.5 * size * tf * mul).min(MAX_PICK_BODY_R)
}

/// View ray direction through a mouse pixel (`mouse_px` relative to the
/// viewport top-left, y down). None for a degenerate viewport / pointer /
/// non-finite matrix.
pub fn ray_dir(
    view_proj: &Mat4,
    eye: Vec3,
    viewport: (f32, f32),
    mouse_px: (f32, f32),
) -> Option<Vec3> {
    let (vw, vh) = viewport;
    if !(vw.is_finite() && vh.is_finite() && vw > 0.0 && vh > 0.0) {
        return None;
    }
    if !(mouse_px.0.is_finite() && mouse_px.1.is_finite() && eye.is_finite()) {
        return None;
    }
    let u = (mouse_px.0 / vw).clamp(0.0, 1.0) * 2.0 - 1.0;
    // pixel y grows downward, NDC v grows upward
    let v = (mouse_px.1 / vh).clamp(0.0, 1.0) * 2.0 - 1.0;
    let inv = view_proj.inverse();
    if !inv.is_finite() {
        return None;
    }
    // wgpu depth range is 0 near / 1 far
    let near = inv.project_point3(Vec3::new(u, -v, 0.0));
    let far = inv.project_point3(Vec3::new(u, -v, 1.0));
    let d = far - near;
    (d.is_finite() && d.length_squared() > 1e-6).then(|| d.normalize())
}

/// The pick ray laid to the local mean terrain plane: direction, along-ray
/// distance of the ground hit (`t`, clamped to [`MAX_RAY_DIST`] for grazed
/// horizons) and the hit point itself. Guard sequence is shared verbatim by
/// every entry point so all paths stay behaviorally identical.
struct GroundRay {
    dir: Vec3,
    t: f32,
    hit: Vec3,
    gh: f32,
}

/// Builds the pick ray laid to the local mean terrain plane.
fn lay_ray(
    view_proj: &[[f32; 4]; 4],
    view_pos: [f32; 3],
    viewport: egui::Rect,
    mouse_px: egui::Pos2,
    ground_h: f32,
) -> Option<GroundRay> {
    let vp = Mat4::from_cols_array_2d(view_proj);
    let eye = Vec3::new(view_pos[0], view_pos[1], view_pos[2]);
    if !vp.is_finite() || !eye.is_finite() {
        return None;
    }
    // non-finite ground collapses to the plane at the world floor
    let gh = if ground_h.is_finite() { ground_h } else { 0.0 };
    if !(viewport.width() > 0.0 && viewport.height() > 0.0) {
        return None;
    }
    if !viewport.contains(mouse_px) || !mouse_px.is_finite() {
        return None;
    }
    let dir = ray_dir(
        &vp,
        eye,
        (viewport.width(), viewport.height()),
        (mouse_px.x - viewport.left(), mouse_px.y - viewport.top()),
    )?;
    // ground-plane hit at the local mean terrain height; rays that graze or
    // fly away from the plane clamp to a bounded horizon cap instead of
    // unbounded precision loss
    let along = if dir.y.abs() > 1e-4 {
        Some((gh - eye.y) / dir.y).filter(|t| t.is_finite() && *t > 0.0)
    } else {
        None
    };
    let t = along.unwrap_or(MAX_RAY_DIST).min(MAX_RAY_DIST);
    Some(GroundRay { dir, t, hit: eye + dir * t, gh })
}

/// Single-candidate evaluation shared verbatim by hover(diag-free) and the
/// diagnostics scan. Returns `(along_ray_dist, planar_hit_dist)`.
#[inline]
fn eval_org(o: &OrgInst, eye: Vec3, dir: Vec3, hit: Vec3, gh: f32) -> Option<(f32, f32)> {
    // corpses fade out of the brush and are not pickable
    if o.alive < 0.5 {
        return None;
    }
    let pos = Vec3::new(o.x, gh, o.y);
    let rel = pos - eye;
    // in front of the camera (and therefore between eye and horizon)
    let fwd = rel.dot(dir);
    if fwd <= 0.0 {
        return None;
    }
    let dx = hit.x - o.x;
    let dz = hit.z - o.y;
    let d = (dx * dx + dz * dz).sqrt();
    if d > MAX_PICK_DIST {
        return None;
    }
    Some((fwd, d))
}

/// Cheap, allocation-free hover: identical math to [`pick_org`] (click
/// quality on purpose) but returns the full [`PickCandidate`] so the UI can
/// feed both the candidate index and its ground offsets to overlays. Reuses
/// the ray computation internally — safe to call every frame.
pub fn hover_org(
    view_proj: &[[f32; 4]; 4],
    view_pos: [f32; 3],
    viewport: egui::Rect,
    mouse_px: egui::Pos2,
    focus: (f32, f32),
    ground_h: f32,
    orgs: &[OrgInst],
) -> Option<PickCandidate> {
    let _ = focus; // the plane height is already local to the focus region
    if orgs.is_empty() {
        return None;
    }
    let ray = lay_ray(view_proj, view_pos, viewport, mouse_px, ground_h)?;
    let eye = Vec3::from_array(view_pos);
    let mut best: Option<(usize, f32, f32)> = None;
    for (i, o) in orgs.iter().enumerate() {
        let Some((fwd, d)) = eval_org(o, eye, ray.dir, ray.hit, ray.gh) else {
            continue;
        };
        // distance tolerance scales with ray travel; pick radius (queen
        // widened) multiplies the base ring
        let tol = tol_at(fwd) * pick_radius(o);
        if d <= tol && best.map_or(true, |(_, _, bd)| d < bd) {
            best = Some((i, fwd, d));
        }
    }
    best.map(|(i, fwd, d)| PickCandidate { org_index: i, dist: fwd, hit_dist: d })
}

/// Pick the organism under `mouse_px` (egui screen pixel) from the rendered
/// instance slice. `ground_h` is the average terrain height near the camera
/// `focus` (world x, y); organisms are assumed to stand at that plane.
/// Returns the index of the *nearest* hit (3D distance along the view ray),
/// or None when nothing inside tolerance is under the pointer.
///
/// Signature is unchanged from the original click API; the winner is exactly
/// `hover_org(..)?.org_index`.
pub fn pick_org(
    view_proj: &[[f32; 4]; 4],
    view_pos: [f32; 3],
    viewport: egui::Rect,
    mouse_px: egui::Pos2,
    focus: (f32, f32),
    ground_h: f32,
    orgs: &[OrgInst],
) -> Option<u32> {
    hover_org(view_proj, view_pos, viewport, mouse_px, focus, ground_h, orgs)
        .map(|c| c.org_index as u32)
}

/// Click-quality pick plus a bounded diagnostics bundle for the overlay UI:
/// nearest-first candidate list (capped at [`MAX_DIAG_CANDIDATES`], sorted by
/// [`PickCandidate::hit_dist`]), the base [`PickDiag::hit_radius`], the
/// normalized ray and the sanitized ground height. Allocates a small `Vec` —
/// click-path / overlay-open only, never per-idle-frame.
pub fn pick_org_with_diag(
    view_proj: &[[f32; 4]; 4],
    view_pos: [f32; 3],
    viewport: egui::Rect,
    mouse_px: egui::Pos2,
    focus: (f32, f32),
    ground_h: f32,
    orgs: &[OrgInst],
) -> (Option<usize>, PickDiag) {
    let _ = focus;
    let empty = PickDiag {
        candidates: Vec::new(),
        hit_radius: 0.0,
        ray_dir: [0.0, 0.0, 0.0],
        ground_h: if ground_h.is_finite() { ground_h } else { 0.0 },
        selected: None,
    };
    if orgs.is_empty() {
        return (None, empty);
    }
    let Some(ray) = lay_ray(view_proj, view_pos, viewport, mouse_px, ground_h) else {
        return (None, empty);
    };
    let eye = Vec3::from_array(view_pos);
    // (planar_dist, along_ray_dist, index), nearest-first, then capped
    let mut hits: Vec<(f32, f32, usize)> = Vec::new();
    for (i, o) in orgs.iter().enumerate() {
        if let Some((fwd, d)) = eval_org(o, eye, ray.dir, ray.hit, ray.gh) {
            let tol = tol_at(fwd) * pick_radius(o);
            if d <= tol {
                hits.push((d, fwd, i));
            }
        }
    }
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(MAX_DIAG_CANDIDATES);
    let selected = hits.first().map(|&(_, _, i)| i);
    let diag = PickDiag {
        candidates: hits
            .into_iter()
            .map(|(d, fwd, i)| PickCandidate { org_index: i, dist: fwd, hit_dist: d })
            .collect(),
        hit_radius: tol_at(ray.t),
        ray_dir: [ray.dir.x, ray.dir.y, ray.dir.z],
        ground_h: ray.gh,
        selected,
    };
    (selected, diag)
}

/// Average terrain height in a small cross around the focus point. The 3D
/// ground mesh eases between cells, so the flat ground-plane used for
/// picking takes the local mean (clamped at world edges by the sampler).
pub fn ground_height_near(world: &alife_worldgen::WorldModel, x: f32, y: f32) -> f32 {
    use crate::terrain::terrain_height_at;
    let h0 = terrain_height_at(world, x, y);
    let (mut acc, mut n) = (h0, 1.0_f32);
    for r in [24.0, 60.0] {
        for (dx, dy) in [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)] {
            let h = terrain_height_at(world, x + dx, y + dy);
            if h.is_finite() {
                acc += h;
                n += 1.0;
            }
        }
    }
    if acc.is_finite() { acc / n } else { 0.0 }
}
