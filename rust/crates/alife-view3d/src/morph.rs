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
use crate::scene::{CameraMode, Frame3D};
use decomp::{INST_CAP, INST_STRIDE, MODE_FOLLOW, MODE_ORBIT, MODE_PLANET};
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
        // 13 floats: pos(3) then 10 scalar instance attributes (locs 3..13)
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
    pub fn update_gpu(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame3D) {
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
        let okay = guard.plan(
            &frame.orgs,
            view,
            mode,
            &mut |x: f32, y: f32| crate::terrain::terrain_height_at(&frame.world, x, y) as f32,
            &mut |c: [f32; 3], r: f32| frustum.sphere_visible(c, r),
        );
        let bytes = bytemuck::cast_slice(&okay.insts);
        if !bytes.is_empty() {
            queue.write_buffer(&self.inst_buf, 0, bytes);
        }
    }

    /// Three instanced draws (tail / shell / billboard buckets). No-op until
    /// ensure_pipeline + first successful update_gpu.
    pub fn draw_with<'a>(&self, pass: &mut wgpu::RenderPass<'a>, frame: Option<&crate::scene::Frame3D>) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(frame) = frame else { return };
        let _ = frame;
        let Some(planner) = self.planner.as_ref() else { return };
        // NOTE: frame's visuals were already planned in update_gpu; the
        // planner mutex hands the current plan to the paint path.
        let plan_guard = planner.lock().unwrap_or_else(|p| p.into_inner());
        let plan = plan_guard.current();
        if (plan.tail_n + plan.shell_n + plan.queen_n + plan.quad_n) == 0 || plan.insts.is_empty() {
            return;
        }
        let (tn, sn, qun, qn) = (
            plan.tail_n as u64,
            plan.shell_n as u64,
            plan.queen_n as u64,
            plan.quad_n as u64,
        );
        let stride = INST_STRIDE as u64;
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);
        pass.set_vertex_buffer(0, self.verts_buf.slice(..));
        if tn > 0 {
            pass.set_vertex_buffer(1, self.inst_buf.slice(0..tn * stride));
            pass.draw(templates::TAIL_VBASE..templates::TAIL_VBASE + templates::TAIL_VCOUNT, 0..tn as u32);
        }
        if sn > 0 {
            pass.set_vertex_buffer(1, self.inst_buf.slice(tn * stride..(tn + sn) * stride));
            pass.draw(
                templates::SHELL_VBASE..templates::SHELL_VBASE + templates::SHELL_VCOUNT,
                0..sn as u32,
            );
        }
        if qun > 0 {
            pass.set_vertex_buffer(
                1,
                self.inst_buf
                    .slice((tn + sn) * stride..(tn + sn + qun) * stride),
            );
            pass.draw(
                templates::QUEEN_VBASE..templates::QUEEN_VBASE + templates::QUEEN_VCOUNT,
                0..qun as u32,
            );
        }
        if qn > 0 {
            pass.set_vertex_buffer(1, self.inst_buf.slice((tn + sn + qun) * stride..(tn + sn + qun + qn) * stride));
            pass.draw(
                templates::QUAD_VBASE..templates::QUAD_VBASE + templates::QUAD_VCOUNT,
                0..qn as u32,
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
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) info: vec4<f32>,      // (id, param, alpha, hue)
    @location(1) nuv: vec4<f32>,       // sphere normal .xyz | tail t in .w | billboard corner .xy
    @location(2) misc: vec4<f32>,      // (turgor, age, wave, phase)
};

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
        // flagella: vertex-shader wave along the bezier chord (param in x)
        let tt = i.taux.x;
        let amp = 0.42 * (0.25 + 0.75 * tt);
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
        nuv = vec4<f32>(normalize(i.tpos.xyz), 0.0);
    } else if (id < 7.5) {
        // antennae + limb stubs: gentle enzymless wave (amp tapers to tip)
        let tt = i.taux.x;
        let amp = 0.05 * (0.3 + 0.7 * (1.0 - tt));
        local.y = local.y + amp * sin(tt * 7.0 - t * i.iwave * 0.7 + i.iphase);
        nuv = vec4<f32>(normalize(i.tpos.xyz), tt);
    } else {
        // crown spikes: dignified slow shimmer
        local = local * (1.0 + 0.035 * sin(t * 1.7 + i.iphase * 0.5)) * r;
        nuv = vec4<f32>(normalize(i.tpos.xyz), 0.0);
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
        // queen body lobes: gum-like body color, rim-lit like the membrane
        // but more opaque; brighter with turgor + a subtle energy pulse
        let n = normalize(fi.nuv.xyz);
        let lam = clamp(abs(dot(n, sun)), 0.0, 1.0);
        let rim = clamp(1.0 - lam, 0.0, 1.0);
        let pulse = 0.03 * sin(t * 1.4);
        col = hsv(hue, sat * 0.88, mix(0.34, 0.56, lam) + pulse) + rim * 0.14;
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

