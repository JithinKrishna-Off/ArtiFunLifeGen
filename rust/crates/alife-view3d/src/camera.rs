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
pub const HABITAT_START_DIST: f32 = 260.0;
/// Follow-camera starting distance (organism local scale).
pub const FOLLOW_START_DIST: f32 = 28.0;

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

        // 1) apply per-mode input to parameter state (target params)
        match mode {
            CameraMode::Planet => {
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
                if self.mode != Mode::Organism {
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
                self.follow.dist = (self.follow.dist + wheel * 2.5).clamp(8.0, 160.0);
            }
        }

        let (target_eye, target_look) = match mode {
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
                    let eye =
                        Vec3::new(o.focus.x + o.yaw.cos() * o.dist, o.focus.y + o.pitch * o.dist * 0.9 + 8.0, o.focus.z + o.yaw.sin() * o.dist);
                    (eye, o.focus)
                }
            }
            CameraMode::Follow { id } => {
                if let Some((fx, fy)) = follow_pos.filter(|_| id == self.follow.id) {
                    let f = &self.follow;
                    let eye = Vec3::new(fx + f.yaw.cos() * f.dist, terrain_height + f.pitch * f.dist * 0.5 + 2.0, fy + f.yaw.sin() * f.dist);
                    (eye, Vec3::new(fx, terrain_height + 1.5, fy))
                } else {
                    // death fallback lives in the UI (orbit the last position)
                    let o = &self.orbit;
                    let eye = Vec3::new(o.focus.x + o.yaw.cos() * o.dist, o.focus.y + o.pitch * o.dist * 0.9, o.focus.z + o.yaw.sin() * o.dist);
                    (eye, o.focus)
                }
            }
        };

        // 2) start/step transition animation
        if self.mode != mode_to_mode(mode) || self.transition.is_some() {
            if self.transition.is_none()
                && (self.mode != mode_to_mode(mode) || (self.eye - target_eye).length() > 4.0)
            {
                // cinematic sweep proportional to the jump length
                let swing = ((self.eye - target_eye).length() * 0.0016).clamp(0.0, 0.5);
                self.transition = Some(Transition {
                    from_eye: self.eye,
                    from_look: self.look,
                    t: 0.0,
                    swing,
                });
            }
        }
        self.mode = mode_to_mode(mode);

        // 3) EASE toward the target; damping factor (never a jump)
        let ease = 1.0 - (-6.0 * dt).exp().min(1.0);
        if let Some(tr) = self.transition.as_mut() {
            tr.t += dt / if tr.swing > 0.2 { 1.8 } else { 1.4 };
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
        self.orbit = OrbitState {
            focus: Vec3::new(x, terrain_height, z),
            yaw: self.planet.yaw,
            pitch: 0.5,
            dist: HABITAT_START_DIST,
        };
    }

    pub fn request_follow(&mut self, id: u64, _at: (f32, f32), _terrain_height: f32) {
        self.follow = FollowState {
            id,
            yaw: self.orbit.yaw,
            pitch: 0.4,
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
