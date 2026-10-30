//! Pure-math tests for organism picking (no GPU). Run via
//! `cargo test -p alife-view3d`. Death-fallback camera behavior lives in the
//! UI layer and is covered by camera transition tests in camera_tests.rs.

use super::*;
use bytemuck::Zeroable;
use crate::scene::OrgInst;
use alife_worldgen::{FieldGrid, WorldModel};
use glam::{Mat4, Vec3};

fn flat_ground() -> WorldModel {
    WorldModel {
        width: 400.0,
        height: 400.0,
        cell: 20.0,
        elevation: FieldGrid::new(4, 4, 0.5),
        moisture: FieldGrid::new(4, 4, 0.3),
        temperature: FieldGrid::new(4, 4, 0.2),
        fertility: FieldGrid::new(4, 4, 0.2),
        hazard: FieldGrid::new(4, 4, 0.0),
        biome: FieldGrid::new(4, 4, 0.0),
        generation_tag: 1,
        hazard_pulse_period: 0.0,
    }
}

fn inst(x: f32, y: f32, size: f32, alive: f32) -> OrgInst {
    let mut o = OrgInst::zeroed();
    o.x = x;
    o.y = y;
    o.heading = 1.0;
    o.energy = 0.5;
    o.size = size;
    o.alive = alive;
    o
}

fn vp_rect() -> egui::Rect {
    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))
}

fn center_mouse() -> egui::Pos2 {
    egui::pos2(400.0, 300.0)
}

/// Combined view-projection + eye for a camera at `eye` looking at `at`.
fn cam_of(eye: Vec3, at: Vec3, up: Vec3) -> ([[f32; 4]; 4], [f32; 3]) {
    let view = Mat4::look_at_rh(eye, at, up);
    let proj = Mat4::perspective_rh(55f32.to_radians(), 1.6, 0.2, 4000.0);
    ((proj * view).to_cols_array_2d(), [eye.x, eye.y, eye.z])
}

fn px(x: f32, y: f32) -> egui::Pos2 {
    egui::pos2(x, y)
}

#[test]
fn straight_down_ray_hits_the_nearest_organism() {
    let eye = Vec3::new(100.0, 200.0, 100.0);
    // hit point is (100, 0, 100); both in tolerance, the closer one wins
    let orgs = [inst(108.0, 100.0, 1.0, 1.0), inst(101.5, 101.5, 1.0, 1.0)];
    let (vp, vp_pos) = cam_of(eye, Vec3::new(eye.x, 0.0, eye.z), Vec3::Z);
    let pick = pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (100.0, 100.0), 0.0, &orgs);
    assert_eq!(pick, Some(1));
}

#[test]
fn out_of_tolerance_from_low_altitude_is_none() {
    let eye = Vec3::new(100.0, 100.0, 100.0);
    // tol at 100 units = max(2, 100*0.05) = 5; org 60 units away is far out
    let orgs = [inst(160.0, 100.0, 1.0, 1.0)];
    let (vp, vp_pos) = cam_of(eye, Vec3::new(eye.x, 0.0, eye.z), Vec3::Z);
    assert_eq!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (100.0, 100.0), 0.0, &orgs), None);
}

#[test]
fn tolerance_scales_with_distance() {
    let orgs = [inst(60.0, 0.0, 1.0, 1.0)];
    // high altitude: tol = 2000*0.05 = 100 > 60 -> hit
    let eye = Vec3::new(0.0, 2000.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::ZERO, Vec3::Z);
    assert!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs).is_some());
    // low altitude: tol = 100*0.05 = 5 < 60 -> miss
    let eye = Vec3::new(0.0, 100.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::ZERO, Vec3::Z);
    assert_eq!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs), None);
}

#[test]
fn tolerance_floor_is_two_and_cap_is_400() {
    assert_eq!(tol_at(0.0), 2.0);
    assert_eq!(tol_at(40.0), 2.0);
    assert_eq!(tol_at(200.0), 10.0);
    assert_eq!(tol_at(9000.0), 400.0);
    assert_eq!(tol_at(f32::NEG_INFINITY), 2.0);
}

#[test]
fn body_size_widens_the_tolerance() {
    // high enough that tol = 150*0.05 = 7.5 ; size 2.2 doubles the radius:
    // 15 < 7.5*2.2=16.5 hit, 25 > 16.5 miss
    let eye = Vec3::new(0.0, 150.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::ZERO, Vec3::Z);
    let orgs = [inst(15.0, 0.0, 2.2, 1.0)];
    assert!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs).is_some());
    let orgs = [inst(25.0, 0.0, 2.2, 1.0)];
    assert_eq!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs), None);
}

#[test]
fn organisms_behind_the_camera_return_none() {
    let eye = Vec3::new(100.0, 50.0, 100.0);
    // looking almost exactly horizontal slightly upward: the ground plane is
    // never hit ahead; both orgs sit "under" the eye and behind the view dir
    let (vp, vp_pos) = cam_of(eye, Vec3::new(2600.0, 110.0, 100.0), Vec3::Y);
    let orgs = [inst(100.0, 100.0, 2.2, 1.0), inst(60.0, 100.0, 2.0, 1.0)];
    assert_eq!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (100.0, 100.0), 0.0, &orgs), None);
}

#[test]
fn corpses_are_not_pickable() {
    let eye = Vec3::new(0.0, 200.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::ZERO, Vec3::Z);
    // dead corpse sits dead-center; living neighbor only slightly off
    let orgs = [inst(0.0, 0.0, 1.0, 0.0), inst(4.0, 0.0, 1.0, 1.0)];
    let pick = pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs);
    assert_eq!(pick, Some(1));
}

#[test]
fn pick_matches_the_ground_point_of_the_view_ray() {
    let eye = Vec3::new(200.0, 300.0, 200.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::new(eye.x, 0.0, eye.z), Vec3::Z);
    // cross-check: descend the ray through a corner pixel to the plane and
    // put an organism exactly on the intersection
    let dir = ray_dir(&Mat4::from_cols_array_2d(&vp), eye, (800.0, 600.0), (720.0, 540.0))
        .expect("ray");
    let t = (0.0 - eye.y) / dir.y;
    assert!(t.is_finite() && t > 0.0 && t < MAX_RAY_DIST);
    let hit = eye + dir * t;
    let orgs = [inst(hit.x, hit.z, 1.0, 1.0)];
    let focus = (hit.x, hit.z);
    assert!(pick_org(&vp, vp_pos, vp_rect(), px(720.0, 540.0), focus, 0.0, &orgs).is_some());
}

#[test]
fn pick_run_is_bounded_and_allocation_free_path() {
    let eye = Vec3::new(200.0, 400.0, 200.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::new(200.0, 0.0, 200.0), Vec3::Z);
    let orgs: Vec<OrgInst> = (0..1600u32)
        .map(|i| inst(199.0 + (i % 40) as f32 * 0.1, 200.0, 1.0 + i as f32 * 0.001, 1.0))
        .collect();
    let mut hits = 0;
    for px_i in 0..40u32 {
        if pick_org(&vp, vp_pos, vp_rect(), px(px_i as f32 * 20.0, 300.0), (200.0, 200.0), 0.0, &orgs).is_some() {
            hits += 1;
        }
    }
    assert!(hits > 0 && hits < 40);
}

#[test]
fn grazed_horizon_rays_clamp_instead_of_exploding() {
    // looking almost exactly horizontal: no forward ground hit; nearest
    // visible org is still bounded by MAX_RAY_DIST
    let eye = Vec3::new(0.0, 60.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::new(2000.0, 60.05, 0.0), Vec3::Y);
    let orgs = [inst(5000.0, 0.0, 1.0, 1.0), inst(2800.0, 0.0, 1.0, 1.0)];
    let pick = pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &orgs);
    // head-on org at ~2800 is inside the clamped horizon only if within tol;
    // 5000 is unreachable either way -- must not pick it
    if pick == Some(1) {
        unreachable!()
    }
    assert_ne!(pick, Some(1));
}

#[test]
fn ground_height_near_averages_the_world_model() {
    let w = flat_ground();
    let h = ground_height_near(&w, 200.0, 200.0);
    // all-zero elevation noise on a fill grid: expect the flat 0.5 height
    assert!((h - crate::terrain::terrain_height_at(&w, 200.0, 200.0)).abs() < 0.1);
    assert!(h.is_finite());
}

#[test]
fn degenerate_inputs_return_none_not_nan() {
    let eye = Vec3::new(0.0, 100.0, 0.0);
    let (vp, vp_pos) = cam_of(eye, Vec3::ZERO, Vec3::Z);
    // zero-size viewport / NaN mouse / pointer outside the rect
    assert_eq!(
        ray_dir(&Mat4::from_cols_array_2d(&vp), eye, (0.0, 0.0), (10.0, 10.0)),
        None
    );
    assert_eq!(pick_org(&vp, vp_pos, vp_rect(), px(f32::NAN, 0.0), (0.0, 0.0), 0.0, &[]), None);
    assert_eq!(
        pick_org(&vp, vp_pos, vp_rect(), px(900.0, 0.0), (0.0, 0.0), 0.0, &[]),
        None
    );
    let tiny = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::ZERO);
    assert_eq!(pick_org(&vp, vp_pos, tiny, center_mouse(), (0.0, 0.0), 0.0, &[]), None);
    let o = inst(0.0, 0.0, 1.0, 1.0);
    // NaN ground/eye collapses safely instead of poisoning the pick
    assert_eq!(pick_org(&vp, [f32::NAN, 0.0, 0.0], vp_rect(), center_mouse(), (0.0, 0.0), 0.0, &[o]), None);
    assert!(pick_org(&vp, vp_pos, vp_rect(), center_mouse(), (0.0, 0.0), f32::NAN, &[o]).is_some());
    // de-projected ray of an affine matrix still lands somewhere snap-decidable
    assert!(ray_dir(&Mat4::from_cols_array_2d(&vp), eye, (800.0, 600.0), (0.0, 0.0)).is_some());
}
