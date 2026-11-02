//! Multi-scale camera for the living-world 3D experience (M1+). Smoothly
//! interpolates between Planet / Habitat / Organism modes; no abrupt jumps.
//! The UI owns this struct per frame and passes the final matrices inside a
//! `Frame3D`; all damping/animation lives here. Owned by Agent D.
//!
//! Cruising: transitions use a smoothstep along a straight eye->target line
//! plus a small circular sweep around the look point mid-flight so long
//! descents (planet -> habitat, habitat -> organism) read as a cinematic
//! orbit rather than a straight dive. Habitat additionally supports a free
//! FPS-like fly sub-set(`C` toggle from the UI: WASD move, mouse look, Q/E
//! or wheel altitude) and follow mode can auto-orbit its target.

use glam::{Mat4, Vec3};
pub use crate::scene::{CameraMode, CamInput};

#[path = "camera_tests.rs"]
#[cfg(test)]
mod camera_tests;

#[path = "picking.rs"]
pub mod picking;

/// Pick-radius based descent distance in world units for habitat entering.
/// Composition for the ~16-unit giant queen: apparent view-height fraction
/// ~= queen_len / (2 * dist * tan(fov/2)); at fov 55 deg the denominator is
/// 2 * dist * 0.5206, so dist 120 -> 16 / 124.9 ~= 13% of the view height
/// (target 15-20% with the grassland window still around her; dist 300
/// rendered her at ~5% — far too small for the giant-queen rebalance).
pub const HABITAT_START_DIST: f32 = 120.0;
/// Follow-camera starting distance (organism local scale).
/// giant queen follow framing (~2.9 body lengths): 56/(16*1.25) ~= 2.9 ->
/// full insect plus grassland headroom in a 55-degree frame.
pub const FOLLOW_START_DIST: f32 = 56.0;
/// Queen body length in world units (visual rebalance reference: the giant
/// queen renders ~16 units crown-to-body). All queen-camera composition is
/// expressed in multiples of this.
pub const QUEEN_BODY_LEN: f32 = 16.0;
/// Queen focus/POV-exit orbit distance: 2.6 body lengths (2.6 * 16 ~= 42) —
/// full anatomy view with a margin of grassland around her, tighter than the
/// generic 2.9-body-length [`FOLLOW_START_DIST`].
pub const QUEEN_FOCUS_DIST: f32 = 42.0;
/// Nominal POV gaze pitch (radians): the look vector is pitched down by this
/// angle from the level horizon so the sward floor reads under the gaze.
pub const POV_PITCH: f32 = 0.28;
/// POV gaze smoothing stiffness (1/s): shortest-angle smoothing of the look
/// yaw toward the live queen heading, matching the eased turn read of her
/// body. Idle queen -> heading delta ~ 0 -> look simply holds.
const POV_TURN_RATE: f32 = 6.0;
/// POV eye height as a body-length fraction (~5.6 units at len 16: above the
/// ~5-unit grass tops, at her crown).
const POV_HEAD_FRAC: f32 = 0.35;
/// POV eye forward offset as a body-length fraction (~8.8 units at len 16:
/// the eye sits at the front of the crown, inside her silhouette).
const POV_FWD_FRAC: f32 = 0.55;
/// POV look horizon distance in body lengths (gaze samples the world a
/// little over two lengths ahead along the heading).
const POV_GAZE_FACTOR: f32 = 2.2;

/// First-person (queen POV) state, engaged by `set_queen_pov`. While it is
/// `Some`, the UI parks `cam_mode` on `Orbit { focus == queen pos }` (so the
/// grass/morph/terrain consumers keep their existing window math) and feeds
/// the live `(x, z, heading, body_len)` alpha channel through
/// `CamInput::pov_target`; `update` then swaps the final eye/look for the
/// POV gaze and ignores orbit drag.
#[derive(Clone, Copy, Debug)]
pub struct PovState {
    pub id: u64,
    /// smoothed look bearing (radians, shortest-angle toward live heading)
    pub yaw: f32,
    /// nominal gaze pitch (radians, [`POV_PITCH`])
    pub pitch: f32,
    /// last queen position seen through the alpha channel
    pub at: (f32, f32),
    pub body_len: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Planet,
    Habitat,
    Organism,
}

/// Free-fly (FPS-like) sub-state, only meaningful while in the Habitat view:
/// WASD moves, drag aims, wheel raises/lowers the flight altitude.
#[derive(Clone, Copy, Debug)]
pub struct FreeFly {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct CameraState {
    mode: Mode,
    /// current/eased camera state
    eye: Vec3,
    look: Vec3,
    /// per-mode parameter store
    orbit: OrbitState,
    planet: PlanetState,
    follow: FollowState,
    /// free-fly sub-mode (habitat only)
    free_fly: Option<FreeFly>,
    /// follow mode: slowly circle the followed organism
    follow_orbit: bool,
    /// transition easing 0..1 when active
    transition: Option<Transition>,
    /// queen POV override (first person): documented on [`PovState`]
    pov: Option<PovState>,
    /// pending dist override for the habitat-entry orbit snap right after a
    /// POV dismount (`set_queen_pov(None)`); consumed by the next `Orbit`
    /// entry so the exit lands at [`QUEEN_FOCUS_DIST`] instead of
    /// [`HABITAT_START_DIST`]. Cleared by any request that skips the snap.
    pov_exit_warp: Option<f32>,
    pub aspect: f32,
    pub fov_y_deg: f32,
}

pub struct OrbitState {
    pub focus: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

pub struct PlanetState {
    pub center: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

pub struct FollowState {
    pub id: u64,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

struct Transition {
    from_eye: Vec3,
    from_look: Vec3,
    t: f32,
    /// radians of circular sweep around the look point at mid-transition
    swing: f32,
    /// flight time in seconds, scaled by jump length so long descents
    /// (planet -> habitat, habitat -> queen close-up) ease slower and short
    /// hops land quickly
    dur: f32,
}

impl CameraState {
    pub fn new() -> Self {
        Self {
            mode: Mode::Planet,
            eye: Vec3::ZERO,
            look: Vec3::ZERO,
            orbit: OrbitState { focus: Vec3::ZERO, yaw: 0.6, pitch: 0.55, dist: HABITAT_START_DIST },
            planet: PlanetState { center: Vec3::ZERO, yaw: 0.6, pitch: 0.9, dist: 900.0 },
            follow: FollowState { id: 0, yaw: 0.0, pitch: 0.6, dist: FOLLOW_START_DIST },
            free_fly: None,
            follow_orbit: false,
            transition: None,
            pov: None,
            pov_exit_warp: None,
            aspect: 1.6,
            fov_y_deg: 55.0,
        }
    }

    pub fn mode(&self) -> Mode { self.mode }

    pub fn follow_platform(&self) -> Option<u64> {
        (self.mode == Mode::Organism).then_some(self.follow.id)
    }

    /// Whether the free-fly sub-mode is currently piloting the habitat view.
    pub fn free_cam(&self) -> bool { self.free_fly.is_some() }

    /// Toggle the free-fly sub-mode. Entering snapshots the eased camera so
    /// there is no jump; leaving returns to a damped habitat orbit anchored
    /// under the fly position with the same bearing ("return-to-habitat").
    pub fn set_free_cam(&mut self, on: bool) {
        if on {
            if self.free_fly.is_none() {
                self.free_fly = Some(Self::fly_from(self.eye, self.look, self.orbit.yaw));
            }
        } else if let Some(f) = self.free_fly.take() {
            self.land_free_fly(f, None);
        }
    }

    /// Derive a fly state from an arbitrary eye/look pair (no jump).
    fn fly_from(eye: Vec3, look: Vec3, fallback_yaw: f32) -> FreeFly {
        let d = (look - eye).normalize_or_zero();
        let horiz = Vec3::new(d.x, 0.0, d.z);
        let horiz = (horiz.length_squared() > 1e-8)
            .then(|| horiz.normalize())
            .unwrap_or_else(|| Vec3::new(-fallback_yaw.cos(), 0.0, -fallback_yaw.sin()));
        let pitch = (-d.y.clamp(-1.0, 1.0).asin()).clamp(-1.35, 1.35);
        let yaw = (-horiz.z).atan2(-horiz.x);
        FreeFly { eye, yaw, pitch }
    }

    /// Switch back to orbit params anchored under the fly position.
    fn land_free_fly(&mut self, f: FreeFly, terrain_height: Option<f32>) {
        self.orbit.yaw = f.yaw;
        self.orbit.pitch = self.orbit.pitch.clamp(0.15, 1.2);
        self.orbit.focus = Vec3::new(
            f.eye.x,
            terrain_height.unwrap_or(self.orbit.focus.y),
            f.eye.z,
        );
    }

    /// Follow mode slowly circles its target while enabled.
    pub fn set_follow_auto_orbit(&mut self, on: bool) { self.follow_orbit = on; }
    pub fn follow_auto_orbit(&self) -> bool { self.follow_orbit }

    /// Whether the first-person queen POV is currently piloting the camera.
    pub fn pov_active(&self) -> bool { self.pov.is_some() }

    /// Last queen position seen through the POV alpha channel (dismount/exit
    /// anchor when the UI needs one).
    pub fn pov_at(&self) -> Option<(f32, f32)> { self.pov.as_ref().map(|p| p.at) }

    /// Engage the first-person queen POV (`Some((id, at, heading, body_len))`)
    /// or dismount (`None`).
    ///
    /// Engaged: the UI keeps `cam_mode` on `Orbit { focus == queen pos }`
    /// (grass/terrain window math untouched) and streams the live values via
    /// `CamInput::pov_target`; `update` swaps the final eye/look for the POV
    /// gaze and ignores orbit drag. Entering snapshots the look yaw from the
    /// heading (no jump); re-calling while engaged only refreshes the target
    /// (look continuity kept). Free-fly is parked.
    ///
    /// Dismount (`None`): return-to-habitat (the free-fly leaving pattern) —
    /// orbit anchored at the last POV position at [`QUEEN_FOCUS_DIST`],
    /// bearing behind the queen's last heading so the exit looks out along
    /// her gaze. The anchor flight is a normal transition, no cut.
    ///
    /// Non-finite input is rejected (state preserved).
    pub fn set_queen_pov(&mut self, target: Option<(u64, (f32, f32), f32, f32)>) {
        match target {
            Some((id, at, heading, body_len)) => {
                let ok = at.0.is_finite()
                    && at.1.is_finite()
                    && heading.is_finite()
                    && body_len.is_finite()
                    && body_len > 0.0;
                if !ok {
                    return;
                }
                self.free_fly = None;
                // re-engage keeps the smoothed yaw: the gaze never snaps
                let yaw = self.pov.as_ref().map_or(heading, |p| p.yaw);
                self.pov = Some(PovState { id, yaw, pitch: POV_PITCH, at, body_len });
            }
            None => {
                if let Some(p) = self.pov.take() {
                    // orbit yaw is focus->eye bearing: heading + PI parks the
                    // eye behind her, looking out along her last heading
                    let yaw = p.yaw + std::f32::consts::PI;
                    self.mode = Mode::Planet; // force the habitat-entry snap
                    self.free_fly = None;
                    // bearing continuity rides the same planet.yaw channel
                    // request_habitat already uses for the snap
                    self.planet.yaw = yaw;
                    self.pov_exit_warp = Some(QUEEN_FOCUS_DIST);
                    self.orbit = OrbitState {
                        focus: Vec3::new(p.at.0, self.orbit.focus.y, p.at.1),
                        yaw,
                        pitch: 0.5,
                        dist: QUEEN_FOCUS_DIST,
                    };
                }
            }
        }
    }

    /// Third-person "focus queen": follow-cam anchored on her with the
    /// 2.6-body-length [`QUEEN_FOCUS_DIST`] anatomy framing. If she is
    /// already the follow target only pitch re-centers (no dist/yaw jump —
    /// the camera eases). Any spawned fly state / POV is parked.
    pub fn request_queen_focus(&mut self, queen_id: u64, _queen_pos: (f32, f32), _th: f32) {
        self.pov = None;
        self.pov_exit_warp = None;
        self.free_fly = None;
        if self.mode == Mode::Organism && self.follow.id == queen_id {
            self.follow.pitch = 0.4;
            return;
        }
        self.follow = FollowState {
            id: queen_id,
            yaw: self.orbit.yaw,
            pitch: 0.4,
            dist: QUEEN_FOCUS_DIST,
        };
    }

    /// Wide world view again: habitat orbit at [`HABITAT_START_DIST`] anchored
    /// at the last noted queen/POV position (or the given fallback anchor),
    /// popping out of follow / POV / free-fly in one hop.
    pub fn return_to_world_orbit(&mut self, fallback: Option<(f32, f32)>, th: f32) {
        let at = self
            .pov
            .as_ref()
            .map(|p| p.at)
            .or(fallback)
            .unwrap_or((self.orbit.focus.x, self.orbit.focus.z));
        self.pov = None;
        self.pov_exit_warp = None;
        self.request_habitat(at, th);
    }

    pub fn transition_active(&self) -> bool { self.transition.is_some() }

    /// Called by the UI once per frame BEFORE rendering.
    /// `frame`: world size, selected org pos (when following), input.
    pub fn update(
        &mut self,
        mode: CameraMode,
        input: &crate::scene::CamInput,
        world: (f32, f32),
        follow_pos: Option<(f32, f32)>,
        terrain_height: f32,
        dt_raw: f32,
    ) {
        let (ww, wh) = world;
        // robustness: clamp/predictable dt + non-finite input rejection
        let dt = if dt_raw.is_finite() && dt_raw > 0.0 { dt_raw.min(0.25) } else { 1.0 / 60.0 };
        let crash_invalid = !(input.drag.0.is_finite() && input.drag.1.is_finite());
        let drag = if crash_invalid { (0.0, 0.0) } else { input.drag };
        let wheel = if input.wheel.is_finite() { input.wheel } else { 0.0 };

        // free fly only lives inside the habitat view
        if self.free_fly.is_some() && !matches!(mode, CameraMode::Orbit { .. }) {
            let f = self.free_fly.take().expect("checked");
            self.land_free_fly(f, Some(terrain_height));
        }

        // orphaned POV (the UI drove the camera mode out from under it, e.g.
        // Esc past the habitat): take the standard dismount path so the
        // follow/planet framings are never silently steered by a stale gaze
        if self.pov.is_some() && !matches!(mode, CameraMode::Orbit { .. }) {
            self.set_queen_pov(None);
        }

        // 1) apply per-mode input to parameter state (target params)
        match mode {
            CameraMode::Planet => {
                self.pov_exit_warp = None;
                if self.mode == Mode::Habitat {
                    // return-to-planet keeps the compass bearing continuous
                    self.planet.yaw = self.orbit.yaw;
                }
                self.planet.dist = (self.planet.dist + wheel * 60.0).clamp(200.0, 1600.0);
            }
            CameraMode::Orbit { focus } => {
                if self.mode != Mode::Habitat {
                    self.orbit = OrbitState {
                        focus: Vec3::new(focus.0, terrain_height, focus.1),
                        yaw: self.planet.yaw,
                        pitch: 0.5,
                        dist: HABITAT_START_DIST,
                    };
                    // a dismount re-enters at the queen focus distance, not
                    // the default habitat start
                    if let Some(dist) = self.pov_exit_warp.take() {
                        self.orbit.dist = dist;
                    }
                }
                if let Some(f) = self.free_fly.as_mut() {
                    f.yaw -= drag.0 * 0.006;
                    f.pitch = (f.pitch + drag.1 * 0.006).clamp(-1.35, 1.35);
                    let speed = 170.0 * dt * (self.orbit.dist / 200.0).clamp(0.6, 3.0);
                    let fx = f.yaw.cos();
                    let fy = f.yaw.sin();
                    if input.forward { f.eye.x -= speed * fx; f.eye.z -= speed * fy; }
                    if input.back { f.eye.x += speed * fx; f.eye.z += speed * fy; }
                    if input.right { f.eye.x += speed * fy; f.eye.z -= speed * fx; }
                    if input.left { f.eye.x -= speed * fy; f.eye.z += speed * fx; }
                    // wheel (or E) = altitude up while flying; Q descends
                    f.eye.y = (f.eye.y + wheel * 50.0).clamp(2.0, 1900.0);
                    if input.down {
                        let vspd = 150.0 * dt * (self.orbit.dist / 200.0).clamp(0.6, 3.0);
                        f.eye.y = (f.eye.y - vspd).max(2.0);
                    }
                    // hard bounds around the world box (never fly off-plan)
                    f.eye.x = f.eye.x.clamp(-40.0, ww + 40.0);
                    f.eye.z = f.eye.z.clamp(-40.0, wh + 40.0);
                } else if self.pov.is_some() {
                    // POV pilot seat: orbit drag / pan / wheel are ignored
                    // (the gaze override below owns eye/look), so the resting
                    // orbit params stay sane for the dismount landing
                } else {
                    self.orbit.yaw -= drag.0 * 0.006;
                    self.orbit.pitch = (self.orbit.pitch + drag.1 * 0.005).clamp(0.12, 1.35);
                    self.orbit.dist = (self.orbit.dist + wheel * 25.0).clamp(30.0, 700.0);
                    // free WASD pan on the ground plane
                    let speed = 140.0 * dt * (self.orbit.dist / 200.0).clamp(0.4, 3.0);
                    let fx = self.orbit.yaw.cos();
                    let fy = self.orbit.yaw.sin();
                    if input.forward { self.orbit.focus.z -= speed * fx; self.orbit.focus.x += speed * fy; }
                    if input.back { self.orbit.focus.z += speed * fx; self.orbit.focus.x -= speed * fy; }
                    if input.left {
                        self.orbit.focus.x -= speed * fx;
                        self.orbit.focus.z -= speed * fy;
                    }
                    if input.right {
                        self.orbit.focus.x += speed * fx;
                        self.orbit.focus.z += speed * fy;
                    }
                    self.orbit.focus.x = self.orbit.focus.x.clamp(0.0, ww);
                    self.orbit.focus.z = self.orbit.focus.z.clamp(0.0, wh);
                }
            }
            CameraMode::Follow { id } => {
                // entering follow-colonia: stage a fresh state only for a
                // new target. For the staged target (request_queen_focus /
                // request_follow already filled yaw/pitch/dist — the queen
                // focus requests her 2.6-body-length framing) the staged
                // values must survive the habitat->organism entry or the
                // override would snap back to FOLLOW_START_DIST.
                if self.mode != Mode::Organism && self.follow.id != id {
                    self.follow = FollowState {
                        id,
                        yaw: self.orbit.yaw,
                        pitch: 0.35,
                        dist: FOLLOW_START_DIST,
                    };
                }
                if self.follow_orbit {
                    self.follow.yaw -= dt * 0.25;
                }
                self.follow.yaw -= drag.0 * 0.006;
                self.follow.pitch = (self.follow.pitch + drag.1 * 0.005).clamp(0.08, 1.25);
                // giant-queen follow framing: zoom out enough to hold the
                // ~16-unit queen fully in view (10 = tight, 380 = wide)
                self.follow.dist = (self.follow.dist + wheel * 2.5).clamp(10.0, 380.0);
            }
        }

        let (mut target_eye, mut target_look) = match mode {
            CameraMode::Planet => {
                let p = &self.planet;
                let cx = ww * 0.5 + (p.dist * 0.15);
                let cy = wh * 0.5 + (p.dist * 0.1);
                let eye = Vec3::new(cx, p.dist * 1.1, cy);
                (eye, Vec3::new(ww * 0.5, 0.0, wh * 0.5))
            }
            CameraMode::Orbit { .. } => {
                if let Some(f) = self.free_fly.as_ref() {
                    // aim along the fly heading; pitch > 0 looks down
                    let cp = f.pitch.cos();
                    let dir = Vec3::new(-f.yaw.cos() * cp, -f.pitch.sin(), -f.yaw.sin() * cp);
                    (f.eye, f.eye + dir)
                } else {
                    let o = &self.orbit;
                    // note: orbit focus is snapshotted into orbit state at entry,
                    // so the pattern binding `focus` of the request is unused
                    let mut eye = Vec3::new(o.focus.x + o.yaw.cos() * o.dist, o.focus.y + o.pitch * o.dist * 0.9 + 8.0, o.focus.z + o.yaw.sin() * o.dist);
                    // vegetation-aware floor: grass tops reach ~5 units, so
                    // never let the eye sink into the sward (min 1.4 above
                    // ground + 16% of orbit distance keeps the near grass
                    // below the frame at grazing pitch settings)
                    eye.y = eye.y.max(terrain_height + (0.16 * o.dist).max(1.4));
                    (eye, o.focus)
                }
            }
            CameraMode::Follow { id } => {
                if let Some((fx, fy)) = follow_pos.filter(|_| id == self.follow.id) {
                    let f = &self.follow;
                    let mut eye = Vec3::new(fx + f.yaw.cos() * f.dist, terrain_height + f.pitch * f.dist * 0.5 + 2.0, fy + f.yaw.sin() * f.dist);
                    // same sward floor as the habitat orbit (grass tops ~5)
                    eye.y = eye.y.max(terrain_height + (0.16 * f.dist).max(1.4));
                    (eye, Vec3::new(fx, terrain_height + 1.5, fy))
                } else {
                    // death fallback lives in the UI (orbit the last position)
                    let o = &self.orbit;
                    let mut eye = Vec3::new(o.focus.x + o.yaw.cos() * o.dist, o.focus.y + o.pitch * o.dist * 0.9, o.focus.z + o.yaw.sin() * o.dist);
                    eye.y = eye.y.max(terrain_height + (0.16 * o.dist).max(1.4));
                    (eye, o.focus)
                }
            }
        };

        // 1b) queen POV gaze override: cam_mode stays `Orbit { focus = queen
        // pos }` (grass/morph/terrain window math untouched); only the final
        // eye/look are swapped. Eye rides at her crown front
        // (0.35*len high, 0.55*len along the heading), look is level out
        // ahead along the smoothed heading (nominal POV_PITCH grazing tilt).
        // Drag is already ignored above; `pov_target` is the per-frame alpha
        // channel (reject non-finite, hold the last look when absent so an
        // idle queen simply keeps her gaze).
        if let Some(pov) = self.pov.as_mut() {
            if let Some((x, z, heading, len)) = input.pov_target {
                let ok = x.is_finite() && z.is_finite() && heading.is_finite() && len.is_finite() && len > 0.0;
                if ok {
                    // shortest-angle smoothing toward the live heading
                    let mut d = heading - pov.yaw;
                    while d > std::f32::consts::PI { d -= std::f32::consts::TAU; }
                    while d < -std::f32::consts::PI { d += std::f32::consts::TAU; }
                    pov.yaw += d * (1.0 - (-POV_TURN_RATE * dt).exp());
                    pov.at = (x, z);
                    pov.body_len = len;
                }
            }
            if let Some(pov) = self.pov.as_ref() {
                let dir = Vec3::new(pov.yaw.cos(), 0.0, pov.yaw.sin());
                let len = pov.body_len;
                // terrain_height is sampled at the anchor (the queen here)
                let head_y = terrain_height + (POV_HEAD_FRAC * len).max(1.4);
                let eye = Vec3::new(
                    pov.at.0 + dir.x * POV_FWD_FRAC * len,
                    head_y,
                    pov.at.1 + dir.z * POV_FWD_FRAC * len,
                );
                // gaze slightly pitched down from the level horizon
                let gaze = POV_GAZE_FACTOR * len;
                let gaze_h = gaze * POV_PITCH.cos();
                let look = eye
                    + Vec3::new(dir.x * gaze_h, -gaze * POV_PITCH.sin(), dir.z * gaze_h);
                (target_eye, target_look) = (eye, look);
            }
        }

        // 2) start/step transition animation
        if self.mode != mode_to_mode(mode) || self.transition.is_some() {
            if self.transition.is_none()
                && (self.mode != mode_to_mode(mode) || (self.eye - target_eye).length() > 4.0)
            {
                // cinematic sweep proportional to the jump length
                let swing = ((self.eye - target_eye).length() * 0.0016).clamp(0.0, 0.5);
                // flight time in seconds, scaled by jump length so long
                // descents ease slower and short hops land quickly
                let dur = (((self.eye - target_eye).length() * 0.02).max(0.6)).min(3.0);
                self.transition = Some(Transition {
                    from_eye: self.eye,
                    from_look: self.look,
                    t: 0.0,
                    swing,
                    dur,
                });
            }
        }
        self.mode = mode_to_mode(mode);

        // 3) EASE toward the target; damping factor (never a jump)
        let ease = 1.0 - (-6.0 * dt).exp().min(1.0);
        if let Some(tr) = self.transition.as_mut() {
            tr.t += dt / tr.dur.max(0.1);
            if tr.t >= 1.0 {
                self.transition = None;
            }
        }
        if let Some(tr) = self.transition.as_ref() {
            let k = smoothstep(tr.t).max(0.0);
            // coarse scripted path: lerp between from and live target ...
            self.eye = tr.from_eye.lerp(target_eye, k);
            self.look = tr.from_look.lerp(target_look, k);
            // ... plus a small circular sweep around the look mid-flight
            if tr.swing > 1e-3 {
                let a = tr.swing * (std::f32::consts::PI * k).sin();
                let off = self.eye - self.look;
                let (ca, sa) = (a.cos(), a.sin());
                self.eye = self.look
                    + Vec3::new(off.x * ca - off.z * sa, off.y, off.x * sa + off.z * ca);
            }
        } else {
            self.eye = self.eye.lerp(target_eye, ease);
            self.look = self.look.lerp(target_look, ease);
        }

        // safety net: never hand NaN matrices to the renderer
        if !(self.eye.is_finite() && self.look.is_finite()) {
            self.eye = target_eye;
            self.look = target_look;
            self.transition = None;
        }
    }

    /// Final matrices for this frame (rendered from the eased state, so any
    /// per-frame movement of the followed organism is naturally tracked).
    pub fn matrices(&self) -> (Mat4, Mat4) {
        let view = Mat4::look_at_rh(self.eye, self.look, Vec3::Y);
        let proj = Mat4::perspective_rh(
            self.fov_y_deg.to_radians(),
            self.aspect.max(0.05),
            0.2,
            4000.0,
        );
        (view, proj)
    }

    /// Shader uniform derived from the eased camera + frame data.
    pub fn uniform(&self, world: (f32, f32), _time: f32, _flash: f32) -> crate::scene::CamUniform {
        let (view, proj) = self.matrices();
        crate::scene::CamUniform {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view_pos: [self.eye.x, self.eye.y, self.eye.z, 1.0],
            world_wh_time_flash: [world.0, world.1, _time, _flash],
        }
    }

    pub fn eye(&self) -> Vec3 { self.eye }
    pub fn look(&self) -> Vec3 { self.look }

    pub fn force_place(&mut self, eye: Vec3, look: Vec3) {
        self.eye = eye;
        self.look = look;
        self.transition = None;
    }

    /// Ground point the camera is currently aimed near (world coords).
    pub fn look_ground(&self) -> (f32, f32) {
        (self.look.x, self.look.z)
    }

    /// Descend from planet straight to a habitat anchor: computes the start
    /// position from the planet view so the transition begins where we were.
    pub fn request_habitat(&mut self, at: (f32, f32), terrain_height: f32) {
        let (x, z) = at;
        self.mode = Mode::Planet;
        // leaving fly mode behind (optically seamless, below the follow cam)
        self.free_fly = None;
        // POV / pending are sub-modes of the habitat view too
        self.pov = None;
        self.pov_exit_warp = None;
        self.orbit = OrbitState {
            focus: Vec3::new(x, terrain_height, z),
            yaw: self.planet.yaw,
            pitch: 0.5,
            dist: HABITAT_START_DIST,
        };
    }

    pub fn request_follow(&mut self, id: u64, _at: (f32, f32), _terrain_height: f32) {
        self.pov = None;
        self.pov_exit_warp = None;
        self.follow = FollowState {
            id,
            yaw: self.orbit.yaw,
            // generic follow-entry snap pitch (the update() entry no longer
            // overwrites a staged target, so the staged values ARE the snap)
            pitch: 0.35,
            dist: FOLLOW_START_DIST,
        };
    }

    pub fn pitch(&self) -> f32 {
        match self.mode {
            Mode::Planet => self.planet.pitch,
            Mode::Habitat => self.orbit.pitch,
            Mode::Organism => self.follow.pitch,
        }
    }
    pub fn dist(&self) -> f32 {
        match self.mode {
            Mode::Planet => self.planet.dist,
            Mode::Habitat => self.orbit.dist,
            Mode::Organism => self.follow.dist,
        }
    }
}

fn mode_to_mode(m: CameraMode) -> Mode {
    match m {
        CameraMode::Planet => Mode::Planet,
        CameraMode::Orbit { .. } => Mode::Habitat,
        CameraMode::Follow { .. } => Mode::Organism,
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
