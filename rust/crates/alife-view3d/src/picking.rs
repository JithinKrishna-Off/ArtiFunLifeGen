//! Ray-based organism picking (Agent D). Pure math, zero GPU and zero
//! allocation on the hot path: derives the view ray through a mouse pixel
//! from the camera matrices, intersects it with the ground plane at the
//! average terrain height near the camera focus, and returns the nearest
//! rendered organism inside a distance-scaled tolerance radius
//! (`max(2.0, dist * 0.05)` world units, capped at 400). Rays that never
//! reach the ground (grazing horizons) are clamped to a bounded search
//! distance; candidates behind the camera or farther than 400 world units
//! from the ground hit are rejected outright.
//!
//! Returns the index into `orgs` (slice order == the UI's id list order);
//! the UI maps that index to a real organism id.

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

/// Pick tolerance at a given ray distance, world units: grows linearly with
/// distance but never below 2.0 nor above [`MAX_TOL`].
#[inline]
pub fn tol_at(dist: f32) -> f32 {
    (2.0f32).max(dist.max(0.0) * 0.05).min(MAX_TOL)
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

/// Pick the organism under `mouse_px` (egui screen pixel) from the rendered
/// instance slice. `ground_h` is the average terrain height near the camera
/// `focus` (world x, y); organisms are assumed to stand at that plane.
/// Returns the index of the *nearest* hit (3D distance along the view ray),
/// or None when nothing inside tolerance is under the pointer.
pub fn pick_org(
    view_proj: &[[f32; 4]; 4],
    view_pos: [f32; 3],
    viewport: egui::Rect,
    mouse_px: egui::Pos2,
    focus: (f32, f32),
    ground_h: f32,
    orgs: &[OrgInst],
) -> Option<u32> {
    let _ = focus; // the plane height is already local to the focus region
    let vp = Mat4::from_cols_array_2d(view_proj);
    let eye = Vec3::new(view_pos[0], view_pos[1], view_pos[2]);
    if !vp.is_finite() || !eye.is_finite() || orgs.is_empty() {
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
    let hit = eye + dir * t;
    let mut best: Option<u32> = None;
    let mut best_d = f32::INFINITY;
    for (i, o) in orgs.iter().enumerate() {
        // corpses fade out of the brush and are not pickable
        if o.alive < 0.5 {
            continue;
        }
        let pos = Vec3::new(o.x, gh, o.y);
        let rel = pos - eye;
        // in front of the camera (and therefore between eye and horizon)
        let fwd = rel.dot(dir);
        if fwd <= 0.0 {
            continue;
        }
        let dx = hit.x - o.x;
        let dz = hit.z - o.y;
        let d = (dx * dx + dz * dz).sqrt();
        if d > MAX_PICK_DIST {
            continue;
        }
        // distance tolerance scales with ray travel; morphology size widens it
        let tol = tol_at(fwd) * o.size.clamp(0.6, 2.2);
        if d <= tol && d < best_d {
            best_d = d;
            best = Some(i as u32);
        }
    }
    best
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
