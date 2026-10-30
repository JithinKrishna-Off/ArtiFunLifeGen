//! Multi-scale camera for the living-world 3D experience (M1). Smoothly
//! interpolates between Planet / Habitat / Organism modes; no abrupt jumps.
//! The UI owns this struct per frame and passes the final matrices inside a
//! `Frame3D`; all damping/animation lives here. Owned by Agent D.

use glam::{Mat4, Vec3};
pub use crate::scene::{CameraMode, CamInput};

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

pub struct CameraState {
    mode: Mode,
    /// current/eased camera state
    eye: Vec3,
    look: Vec3,
    /// per-mode parameter store
    orbit: OrbitState,
    planet: PlanetState,
    follow: FollowState,
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
            transition: None,
            aspect: 1.6,
            fov_y_deg: 55.0,
        }
    }

    pub fn mode(&self) -> Mode { self.mode }

    pub fn follow_platform(&self) -> Option<u64> {
        (self.mode == Mode::Organism).then_some(self.follow.id)
    }

    /// Called by the UI once per frame BEFORE rendering.
    /// `frame`: world size, selected org pos (when following), input.
    pub fn update(
        &mut self,
        mode: CameraMode,
        input: &crate::scene::CamInput,
        world: (f32, f32),
        follow_pos: Option<(f32, f32)>,
        terrain_height: f32,
        dt: f32,
    ) {
        let (ww, wh) = world;
        // 1) apply per-mode input to parameter state (target params)
        match mode {
            CameraMode::Planet => {
                self.planet.dist = (self.planet.dist + input.wheel * 60.0).clamp(200.0, 1600.0);
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
                self.orbit.yaw -= input.drag.0 * 0.006;
                self.orbit.pitch = (self.orbit.pitch + input.drag.1 * 0.005).clamp(0.12, 1.35);
                self.orbit.dist = (self.orbit.dist + input.wheel * 25.0).clamp(30.0, 700.0);
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
            CameraMode::Follow { id } => {
                if self.mode != Mode::Organism {
                    self.follow = FollowState {
                        id,
                        yaw: self.orbit.yaw,
                        pitch: 0.35,
                        dist: FOLLOW_START_DIST,
                    };
                }
                self.follow.yaw -= input.drag.0 * 0.006;
                self.follow.pitch = (self.follow.pitch + input.drag.1 * 0.005).clamp(0.08, 1.25);
                self.follow.dist = (self.follow.dist + input.wheel * 2.5).clamp(8.0, 160.0);
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
                let o = &self.orbit;
                // note: orbit focus is snapshotted into orbit state at entry,
                // so the pattern binding `focus` of the request is unused
                let eye =
                    Vec3::new(o.focus.x + o.yaw.cos() * o.dist, o.focus.y + o.pitch * o.dist * 0.9 + 8.0, o.focus.z + o.yaw.sin() * o.dist);
                (eye, o.focus)
            }
            CameraMode::Follow { id } => {
                if let Some((fx, fy)) = follow_pos.filter(|_| id == self.follow.id) {
                    let f = &self.follow;
                    let eye = Vec3::new(fx + f.yaw.cos() * f.dist, terrain_height + f.pitch * f.dist * 0.5 + 2.0, fy + f.yaw.sin() * f.dist);
                    (eye, Vec3::new(fx, terrain_height + 1.5, fy))
                } else {
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
                self.transition = Some(Transition { from_eye: self.eye, from_look: self.look, t: 0.0 });
            }
        }
        self.mode = mode_to_mode(mode);

        // 3) EASE toward the target; damping factor (never a jump)
        let ease = 1.0 - (-6.0 * dt).exp().min(1.0);
        if let Some(tr) = self.transition.as_mut() {
            tr.t += dt / 1.4;
            if tr.t >= 1.0 {
                self.transition = None;
            }
        }
        if self.transition.is_some() {
            let tr = self.transition.as_ref().unwrap();
            let k = smoothstep(tr.t).max(0.0);
            // coarse scripted path: lerp between from and live target
            self.eye = tr.from_eye.lerp(target_eye, k);
            self.look = tr.from_look.lerp(target_look, k);
        } else {
            self.eye = self.eye.lerp(target_eye, ease);
            self.look = self.look.lerp(target_look, ease);
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
