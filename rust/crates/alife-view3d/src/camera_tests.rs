//! Camera transition / free-fly regression tests (no GPU). Pure math and
//! state-machine only; a death-fallback smoke check piggybacks here via the
//! follow -> orbit path (the UI decides the fallback request itself).

use super::*;
use crate::scene::CamInput;

fn dt() -> f32 {
    1.0 / 60.0
}

fn settle(c: &mut CameraState, mode: CameraMode, world: (f32, f32), frames: usize, th: f32, follow: Option<(f32, f32)>) {
    for _ in 0..frames {
        c.update(mode, &CamInput::default(), world, follow, th, dt());
    }
}

#[test]
fn planet_to_habitat_reaches_full_orbit_state() {
    let mut c = CameraState::new();
    c.aspect = 1.6;
    settle(&mut c, CameraMode::Planet, (3000.0, 3000.0), 120, 0.0, None);
    assert_eq!(c.mode(), Mode::Planet);

    let focus = (1400.0, 1200.0);
    c.request_habitat(focus, 4.0);
    settle(&mut c, CameraMode::Orbit { focus }, (3000.0, 3000.0), 300, 4.0, None);

    assert_eq!(c.mode(), Mode::Habitat);
    assert!(!c.transition_active());
    // fully settled onto the orbit anchor position (not just nearby)
    let d = 260.0_f32;
    let want = Vec3::new(
        1400.0 + 0.6_f32.cos() * d,
        4.0 + 0.5 * d * 0.9 + 8.0,
        1200.0 + 0.6_f32.sin() * d,
    );
    assert!((c.eye() - want).length() < 1.0, "eye spot {:?} vs want {:?}", c.eye(), want);
    assert!((c.look() - Vec3::new(1400.0, 4.0, 1200.0)).length() < 0.5);
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn habitat_to_organism_settles_without_nan() {
    let mut c = CameraState::new();
    let focus = (500.0, 600.0);
    settle(&mut c, CameraMode::Orbit { focus }, (2000.0, 2000.0), 120, 3.0, None);
    c.request_follow(7, (520.0, 640.0), 3.0);
    settle(&mut c, CameraMode::Follow { id: 7 }, (2000.0, 2000.0), 300, 3.0, Some((520.0, 640.0)));
    assert_eq!(c.mode(), Mode::Organism);
    assert!(!c.transition_active());
    assert!(c.follow_platform() == Some(7));
    let want = Vec3::new(
        520.0 + c.follow.yaw.cos() * FOLLOW_START_DIST,
        3.0 + 0.35 * FOLLOW_START_DIST * 0.5 + 2.0,
        640.0 + c.follow.yaw.sin() * FOLLOW_START_DIST,
    );
    assert!((c.eye() - want).length() < 1.0);
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn free_fly_clamps_to_world_bounds_and_altitudes() {
    let mut c = CameraState::new();
    let world = (2000.0, 1600.0);
    let focus = (1000.0, 900.0);
    settle(&mut c, CameraMode::Orbit { focus }, world, 60, 3.0, None);
    assert!(!c.free_cam());

    c.set_free_cam(true);
    assert!(c.free_cam());
    // fly hard toward the corner + | +max wheel for the whole run
    let mut ci = CamInput { forward: true, right: true, wheel: 5.0, ..CamInput::default() };
    ci.drag = (0.6, 1.2);
    for _ in 0..900 {
        c.update(CameraMode::Orbit { focus }, &ci, world, None, 3.0, dt());
    }
    let e = c.eye();
    assert!(e.x >= -40.0 && e.x <= 2040.0, "x out of bounds: {}", e.x);
    assert!(e.z >= -40.0 && e.z <= 1640.0, "z out of bounds: {}", e.z);
    assert!(e.y <= 1900.0 && e.y >= 2.0, "altitude out of bounds: {}", e.y);
    assert!(e.is_finite());
    assert!(c.fly_pitch_is_bounded());

    // leaving the fly re-enters normal orbit damped and settled, anchored
    // under the fly position with the preserved orbit bearing
    c.set_free_cam(false);
    assert!(!c.free_cam());
    settle(&mut c, CameraMode::Orbit { focus }, world, 400, 3.0, None);
    assert!(!c.transition_active());
    let o = &c.orbit;
    let terrain_h = 3.0;
    let want = Vec3::new(
        o.focus.x + o.yaw.cos() * o.dist,
        terrain_h + o.pitch * o.dist * 0.9 + 8.0,
        o.focus.z + o.yaw.sin() * o.dist,
    );
    let settled = c.eye().distance(want);
    assert!(settled < 1.0, "return-to-habitat drift {}", settled);
    assert!(c.eye().is_finite() && c.look().is_finite());
}

impl CameraState {
    fn fly_pitch_is_bounded(&self) -> bool {
        matches!(self.free_fly, Some(f) if f.pitch.abs() <= 1.3501)
    }
}

#[test]
fn free_fly_only_alive_in_habitat() {
    let mut c = CameraState::new();
    let focus = (500.0, 500.0);
    settle(&mut c, CameraMode::Orbit { focus }, (2000.0, 2000.0), 60, 2.0, None);
    c.set_free_cam(true);
    // a follow request parks the fly seamlessly
    c.set_free_cam(false);
    c.request_follow(3, (520.0, 520.0), 2.0);
    settle(&mut c, CameraMode::Follow { id: 3 }, (2000.0, 2000.0), 240, 2.0, Some((520.0, 520.0)));
    assert_eq!(c.mode(), Mode::Organism);
    assert!(!c.free_cam());
    // planet too
    c.set_free_cam(true);
    c.update(CameraMode::Planet, &CamInput::default(), (2000.0, 2000.0), None, 2.0, dt());
    assert!(!c.free_cam());
}

#[test]
fn follow_auto_orbit_circles_the_target() {
    let mut c = CameraState::new();
    let focus = (500.0, 500.0);
    settle(&mut c, CameraMode::Orbit { focus }, (2000.0, 2000.0), 60, 2.0, None);
    c.set_follow_auto_orbit(true);
    c.request_follow(9, (510.0, 500.0), 2.0);
    let yaw0 = c.follow.yaw;
    settle(&mut c, CameraMode::Follow { id: 9 }, (2000.0, 2000.0), 120, 2.0, Some((510.0, 500.0)));
    assert!((c.follow.yaw - yaw0) < -0.3, "auto orbit did not advance: {} -> {}", yaw0, c.follow.yaw);
    // eye stays ringed around the target
    let d = (c.eye() - Vec3::new(510.0, 0.0, 500.0)).length();
    assert!(d > FOLLOW_START_DIST * 0.5 && d < FOLLOW_START_DIST * 2.0);
}

#[test]
fn no_nan_with_bad_dt_or_bad_input() {
    let mut c = CameraState::new();
    let focus = (500.0, 500.0);
    c.update(CameraMode::Orbit { focus }, &CamInput::default(), (2000.0, 2000.0), None, 2.0, dt());
    let eye0 = c.eye();
    // NaN dt keeps the camera mathematically alive
    c.update(CameraMode::Orbit { focus }, &CamInput::default(), (2000.0, 2000.0), None, 2.0, f32::NAN);
    assert!(c.eye().is_finite() && c.look().is_finite());
    // NaN drag/wheel is rejected instead of poisoning the orbit state
    let mut bad = CamInput::default();
    bad.drag = (f32::NAN, f32::INFINITY);
    bad.wheel = f32::NAN;
    c.update(CameraMode::Orbit { focus }, &bad, (2000.0, 2000.0), None, 2.0, dt());
    assert!((c.eye() - eye0).length() < 12.0);
    assert!(c.eye().is_finite() && c.look().is_finite());
    assert!(c.orbit.yaw.is_finite() && c.orbit.pitch.is_finite() && c.orbit.dist.is_finite());
}

#[test]
fn death_fallback_orbit_target_is_reachable_and_stable() {
    // follow semantics when the target position is unavailable lock the eye
    // to the stored orbit anchor: preserved-focus fallback behavior
    let mut c = CameraState::new();
    let focus = (800.0, 700.0);
    settle(&mut c, CameraMode::Orbit { focus }, (2000.0, 2000.0), 120, 5.0, None);
    c.request_follow(11, (820.0, 690.0), 5.0);
    c.update(CameraMode::Follow { id: 11 }, &CamInput::default(), (2000.0, 2000.0), Some((820.0, 690.0)), 5.0, dt());
    // target vanishes (death): update still runs, camera keeps orbit focus
    for _ in 0..240 {
        c.update(CameraMode::Follow { id: 12 }, &CamInput::default(), (2000.0, 2000.0), None, 5.0, dt());
    }
    assert!(c.eye().is_finite() && c.look().is_finite());
    assert!(c.eye().y > 5.0); // stays above terrain
    let ground = c.look_ground();
    assert!(ground.0.is_finite() && ground.1.is_finite());
}

#[test]
fn transition_sweep_stays_cinematic_not_wild() {
    let mut c = CameraState::new();
    settle(&mut c, CameraMode::Planet, (3000.0, 3000.0), 120, 0.0, None);
    let focus = (1500.0, 1500.0);
    c.request_habitat(focus, 10.0);
    let max_len_change = {
        let mut m = 0.0_f32;
        for _ in 0..300 {
            let before = (c.eye() - c.look()).length();
            c.update(CameraMode::Orbit { focus }, &CamInput::default(), (3000.0, 3000.0), None, 10.0, dt());
            let after = (c.eye() - c.look()).length();
            m = m.max((before - after).abs());
        }
        m
    };
    // the sweep never flap-teleports: segment length evolves smoothly
    assert!(max_len_change < 15.0, "transition jitter {}", max_len_change);
    assert!(!c.transition_active());
}

#[test]
fn free_fly_q_descends_and_never_sinks_below_terrain() {
    let mut c = CameraState::new();
    let focus = (500.0, 500.0);
    settle(&mut c, CameraMode::Orbit { focus }, (2000.0, 2000.0), 60, 2.0, None);
    c.set_free_cam(true);
    let start_y = c.eye().y;
    // hold Q (-> CamInput::down): descend to the terrain floor clamp
    let ci = CamInput { down: true, ..CamInput::default() };
    for _ in 0..300 {
        c.update(CameraMode::Orbit { focus }, &ci, (2000.0, 2000.0), None, 2.0, dt());
    }
    let e = c.eye();
    // 2.0 is the fly-terrain floor; the eased eye may sit an epsilon below it
    assert!(e.y >= 2.0 - 1e-2, "below terrain floor: {}", e.y);
    assert!(e.y < start_y, "Q did not descend: {} -> {}", start_y, e.y);
    assert!(e.is_finite());
}

fn nxt(s: &mut u32, m: u32) -> u32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    (*s >> 9) % m
}

#[test]
fn camera_stays_finite_over_10000_updates() {
    let mut c = CameraState::new();
    let world = (3000.0_f32, 3000.0_f32);
    let mut ci = CamInput::default();
    let mut s = 7u32;
    for i in 0..10000u32 {
        if i % 97 == 0 {
            ci = CamInput::default();
            ci.drag = (nxt(&mut s, 40) as f32 - 20.0, nxt(&mut s, 40) as f32 - 20.0);
            ci.wheel = if i % 3 == 0 { nxt(&mut s, 7) as f32 - 3.0 } else { 0.0 };
            ci.forward = nxt(&mut s, 2) == 0;
            ci.left = nxt(&mut s, 2) == 0;
            ci.down = nxt(&mut s, 2) == 0;
        }
        let mode = match nxt(&mut s, 3) {
            0 => CameraMode::Planet,
            1 => CameraMode::Orbit { focus: (1400.0 + nxt(&mut s, 200) as f32, 1400.0) },
            _ => CameraMode::Follow { id: (nxt(&mut s, 50) + 1) as u64 },
        };
        let follow = match mode {
            CameraMode::Follow { .. } => Some((1500.0_f32, 1500.0_f32)),
            _ => None,
        };
        // periodic broken dt must never poison the eased camera
        let dt = if i % 4001 == 0 { f32::NAN } else { dt() };
        c.update(mode, &ci, world, follow, 4.0, dt);
        if i % 1000 == 0 {
            assert!(c.eye().is_finite() && c.look().is_finite(), "frame {i}: non-finite camera");
        }
        if i % 2500 == 0 {
            c.set_free_cam(nxt(&mut s, 2) == 0);
            c.set_follow_auto_orbit(nxt(&mut s, 2) == 0);
        }
        if i % 3000 == 0 {
            c.force_place(Vec3::new(500.0, 300.0, 500.0), Vec3::new(501.0, 0.0, 501.0));
        }
    }
    assert!(c.eye().is_finite() && c.look().is_finite());
    assert!(c.orbit.yaw.is_finite() && c.orbit.pitch.is_finite() && c.follow.dist.is_finite());
}
