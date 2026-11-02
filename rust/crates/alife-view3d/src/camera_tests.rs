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
    let d = HABITAT_START_DIST;
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

// ------------------------------------------------------- queen cam / POV

#[test]
fn queen_focus_request_uses_focus_dist_and_recenters_without_jump() {
    let mut c = CameraState::new();
    let world = (2000.0, 2000.0);
    settle(&mut c, CameraMode::Orbit { focus: (500.0, 500.0) }, world, 120, 2.0, None);

    // fresh focus: follow-cam on the queen at 2.6 body lengths (42)
    c.request_queen_focus(21, (600.0, 700.0), 2.0);
    settle(&mut c, CameraMode::Follow { id: 21 }, world, 300, 2.0, Some((600.0, 700.0)));
    assert_eq!(c.mode(), Mode::Organism);
    assert_eq!(c.follow.id, 21);
    assert_eq!(c.follow.dist, QUEEN_FOCUS_DIST);
    assert!(!c.transition_active());
    let want = Vec3::new(
        600.0 + c.follow.yaw.cos() * QUEEN_FOCUS_DIST,
        2.0 + 0.4 * QUEEN_FOCUS_DIST * 0.5 + 2.0,
        700.0 + c.follow.yaw.sin() * QUEEN_FOCUS_DIST,
    );
    assert!((c.eye() - want).length() < 1.0, "eye spot {:?}", c.eye());
    assert!(c.eye().is_finite() && c.look().is_finite());

    // already focusing her: recenter pitch only — id, yaw and the zoomed
    // user dist stay (no jump), the camera just eases back to 0.4 pitch
    let mut ci = CamInput::default();
    ci.drag = (0.0, 60.0); // shove pitch away from 0.4
    ci.wheel = 40.0; // user zooms out too
    for _ in 0..40 {
        c.update(CameraMode::Follow { id: 21 }, &ci, world, Some((600.0, 700.0)), 2.0, dt());
    }
    assert!(c.follow.pitch > 0.5 && c.follow.dist > QUEEN_FOCUS_DIST + 20.0);
    let (old_yaw, old_dist) = (c.follow.yaw, c.follow.dist);
    c.request_queen_focus(21, (600.5, 700.3), 2.0);
    assert_eq!(c.follow.id, 21);
    assert_eq!(c.follow.dist, old_dist, "recenter keeps the user zoom");
    assert_eq!(c.follow.yaw, old_yaw);
    assert_eq!(c.follow.pitch, 0.4);
    for _ in 0..200 {
        c.update(CameraMode::Follow { id: 21 }, &CamInput::default(), world, Some((600.0, 700.0)), 2.0, dt());
    }
    assert!((c.follow.pitch - 0.4).abs() < 0.01);
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn queen_pov_eye_look_math_drag_ignored_and_exit_lands_orbit() {
    let mut c = CameraState::new();
    let world = (2000.0, 2000.0);
    let at = (1000.0, 1000.0);
    let th = 3.0;
    settle(&mut c, CameraMode::Orbit { focus: at }, world, 60, th, None);

    // engage from the follow-cam style call; POV math then runs off the
    // per-frame alpha channel while cam_mode stays Orbit { focus = queen }
    c.set_queen_pov(Some((30, at, 0.0, QUEEN_BODY_LEN)));
    assert!(c.pov_active());
    let mode = CameraMode::Orbit { focus: at };
    let mut ci = CamInput { pov_target: Some((at.0, at.1, 0.0, QUEEN_BODY_LEN)), ..CamInput::default() };
    ci.drag = (500.0, 500.0); // POV pilot seat: drag must be ignored
    ci.wheel = 100.0;
    for _ in 0..400 {
        c.update(mode, &ci, world, None, th, dt());
    }
    assert!(c.pov_active());
    assert_eq!(c.mode(), Mode::Habitat); // grass window math keeps habitat

    // documented POV math at heading 0, len 16:
    //   fwd = 0.55 * 16 = 8.8, head = 3.0 + 0.35 * 16 = 8.6
    //   gaze = 2.2 * 16 = 35.2; look drops by gaze * sin(0.28), horizon by
    //   gaze * cos(0.28); the eye never flinches from the ignored drag
    let head = th + 0.35 * QUEEN_BODY_LEN;
    let fwd = 0.55 * QUEEN_BODY_LEN;
    let gaze = 2.2 * QUEEN_BODY_LEN;
    let eye_want = Vec3::new(at.0 + fwd, head, at.1);
    let look_want = eye_want
        + Vec3::new(
            gaze * 0.28_f32.cos(),
            -gaze * 0.28_f32.sin(),
            0.0,
        );
    let eye = c.eye();
    let look = c.look();
    assert!(eye.distance(eye_want) < 1.2, "pov eye {:?} vs {eye_want:?}", eye);
    assert!(look.distance(look_want) < 1.2, "pov look {:?} vs {look_want:?}", look);
    assert!(eye.is_finite() && look.is_finite());

    // non-finite alpha channel is rejected per-frame: state holds
    let bad = CamInput { pov_target: Some((f32::NAN, f32::NAN, f32::NAN, -1.0)), ..CamInput::default() };
    for _ in 0..30 {
        c.update(mode, &bad, world, None, th, dt());
    }
    assert!(c.pov_at() == Some(at));
    assert!(c.pov_active());

    // heading smoothing: a new heading converges through shortest angles
    let good = CamInput { pov_target: Some((at.0, at.1, std::f32::consts::PI, QUEEN_BODY_LEN)), ..CamInput::default() };
    for _ in 0..200 {
        c.update(mode, &good, world, None, th, dt());
    }

    // dismount: habitat orbit behind the queen at QUEEN_FOCUS_DIST; the
    // dist warp is consumed by the habitat-entry snap
    c.set_queen_pov(None);
    assert!(!c.pov_active());
    for _ in 0..400 {
        c.update(mode, &CamInput::default(), world, None, th, dt());
    }
    assert!(!c.transition_active());
    assert_eq!(c.mode(), Mode::Habitat);
    assert_eq!(c.orbit.dist, QUEEN_FOCUS_DIST, "exit dist after warp");
    assert!((c.orbit.focus.x - at.0).abs() < 0.5 && (c.orbit.focus.z - at.1).abs() < 0.5);
    let want = Vec3::new(
        c.orbit.focus.x + c.orbit.yaw.cos() * c.orbit.dist,
        th + c.orbit.pitch * c.orbit.dist * 0.9 + 8.0,
        c.orbit.focus.z + c.orbit.yaw.sin() * c.orbit.dist,
    );
    let settled = c.eye().distance(want);
    assert!(settled < 1.0, "pov-exit drift {settled}");
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn set_queen_pov_rejects_non_finite_and_recalling_keeps_look() {
    let mut c = CameraState::new();
    // garbage engage is a no-op
    c.set_queen_pov(Some((1, (f32::NAN, 100.0), 0.0, QUEEN_BODY_LEN)));
    assert!(!c.pov_active());
    c.set_queen_pov(Some((1, (100.0, 100.0), f32::INFINITY, QUEEN_BODY_LEN)));
    assert!(!c.pov_active());

    // real engage, then a re-call with a moved target: look yaw continuity
    c.set_queen_pov(Some((2, (500.0, 500.0), 0.0, QUEEN_BODY_LEN)));
    let world = (2000.0, 2000.0);
    let mode = CameraMode::Orbit { focus: (500.0, 500.0) };
    let ci = CamInput { pov_target: Some((500.0, 500.0, 0.3, QUEEN_BODY_LEN)), ..CamInput::default() };
    for _ in 0..60 {
        c.update(mode, &ci, world, None, 2.0, dt());
    }
    let yaw_before = c.pov_active();
    // re-engaging refreshes the target but never snaps the smoothed yaw
    c.set_queen_pov(Some((2, (520.0, 510.0), 2.0, QUEEN_BODY_LEN)));
    assert_eq!(yaw_before, true);
    let ci2 = CamInput { pov_target: Some((520.0, 510.0, 2.0, QUEEN_BODY_LEN)), ..CamInput::default() };
    for _ in 0..30 {
        c.update(mode, &ci2, world, None, 2.0, dt());
    }
    // no NaN poisoning anywhere
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn return_to_world_orbit_anchors_pov_then_fallback() {
    let mut c = CameraState::new();
    let world = (2000.0, 2000.0);
    settle(&mut c, CameraMode::Planet, world, 60, 0.0, None);
    let at = (900.0, 1100.0);
    c.set_queen_pov(Some((9, at, 1.0, QUEEN_BODY_LEN)));
    let mode = CameraMode::Orbit { focus: at };
    let ci = CamInput { pov_target: Some((at.0, at.1, 1.0, QUEEN_BODY_LEN)), ..CamInput::default() };
    for _ in 0..200 {
        c.update(mode, &ci, world, None, 4.0, dt());
    }

    // World View from POV: back to the queen as the wide anchor
    c.return_to_world_orbit(Some((1000.0, 1000.0)), 4.0);
    assert!(!c.pov_active());
    let mode = CameraMode::Orbit { focus: at };
    for _ in 0..400 {
        c.update(mode, &CamInput::default(), world, None, 4.0, dt());
    }
    assert_eq!(c.mode(), Mode::Habitat);
    assert!(!c.transition_active());
    assert_eq!(c.orbit.dist, HABITAT_START_DIST);
    assert!((c.orbit.focus.x - at.0).abs() < 0.5 && (c.orbit.focus.z - at.1).abs() < 0.5);
    assert!(c.eye().is_finite() && c.look().is_finite());

    // from follow-cam: the fallback anchor wins (no POV to note)
    c.request_follow(14, at, 4.0);
    let mode = CameraMode::Follow { id: 14 };
    for _ in 0..200 {
        c.update(mode, &CamInput::default(), world, Some(at), 4.0, dt());
    }
    let fallback_src = (1234.0, 1234.0);
    c.return_to_world_orbit(Some(fallback_src), 6.0);
    assert_eq!(c.mode(), Mode::Planet, "return forces the habitat-entry snap");
    let mode = CameraMode::Orbit { focus: fallback_src };
    for _ in 0..400 {
        c.update(mode, &CamInput::default(), world, None, 6.0, dt());
    }
    assert!((c.orbit.focus.x - fallback_src.0).abs() < 0.5 && (c.orbit.focus.z - fallback_src.1).abs() < 0.5);
    assert_eq!(c.orbit.dist, HABITAT_START_DIST);
    assert!(c.eye().is_finite() && c.look().is_finite());
}

#[test]
fn free_fly_q_descends_and_never_sinks_below_terrain() {    let mut c = CameraState::new();
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
