//! Procedural vegetation (grass / tufts / moss / shore algae) instanced on
//! the land. Agent 4's deliverable: per-CLUMP instancing over a shared
//! blade-major lattice (7 blades x 24 verts = 168 verts) with compact
//! per-instance attributes, wind animation in the vertex shader, organism
//! interaction (trample field), density LOD folded into the deterministic
//! drop rate, cheap per-chunk frustum culling AND per-chunk blade-count LOD
//! (near chunks draw all 7 blades, mid 4, far 1: vertex-range sub-draws).
//!
//! Tri budget: <= 9k clumps on screen x 56 tris/clump = <= 504k triangles
//! (draw loop clamps every 32x32 chunk to <= 150 clumps).
//!
//! Resource rules (see report):
//! - `Grass::new(&device)` only prepares CPU state (never fails; the target
//!   color/depth format is not known at that call site).
//! - The coordinator must call `ensure_pipeline(&device, color_format,
//!   depth_format)` once in `View3D::new`, and `update_gpu(&device, &queue,
//!   &frame)` every frame inside `View3D::update` (before `draw`). Until
//!   then `draw_with` is a no-op: the contract `draw_with(&self, pass,
//!   frame)` has no queue access, so ALL GPU uploads happen in `update_gpu`.
//! - Bind group 0 is a grass-owned copy of the camera uniform (same std140
//!   block layout as the terrain), bind group 1 is grass-local
//!   (wind gust + trample field).

#[path = "grass_place.rs"]
pub mod place;
#[path = "grass_frustum.rs"]
pub mod frustum;
#[path = "grass_tests.rs"]
#[cfg(test)]
mod tests;

use crate::grass_clump::{blades_per_clump, clump_vert_count, clump_verts as clump_lattice};
use crate::scene::{CameraMode, Frame3D};
use place::{mode_of, rings_for, window_for, GrassInst, MAX_BLADES};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// prevailing wind direction (unit-ish, fixed so worlds differ only by fields)
const WIND_DIR: [f32; 2] = [0.86, 0.51];

/// std140 (4 x vec4) grass-local uniform: wind gust state + organism field
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GrassUniform {
    /// (dir_x, dir_z, gust amplitude, unused)
    wind_gust: [f32; 4],
    /// (x, z, radius, strength) trample field around the followed organism
    trample: [f32; 4],
    _spare1: [f32; 4],
    _spare2: [f32; 4],
}

const CALM_UNIFORM: GrassUniform = GrassUniform {
    wind_gust: [WIND_DIR[0], WIND_DIR[1], 0.35, 0.0],
    trample: [0.0, 0.0, -1.0, 0.0],
    _spare1: [0.0; 4],
    _spare2: [0.0; 4],
};

struct CacheKey {
    gen: u32,
    cell: (i32, i32),
    mode: u8,
    bucket: u8,
}

struct CacheChunk {
    cx: i32,
    cy: i32,
    count: u32,
    /// source byte offset range [off, off + count * 48) in the instance buffer
    byte_off: u64,
}

struct Cache {
    key: Option<CacheKey>,
    chunks: Vec<CacheChunk>,
    total: u32,
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    gust_buf: wgpu::Buffer,
    gust_bg: wgpu::BindGroup,
    verts_buf: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    /// lattice vert count uploaded to `verts_buf` (168 clump / 24 fallback)
    vert_count: u32,
    /// blade-major lattice: blades per clump (7 clump / 1 single-blade fallback)
    blades: u32,
    /// verts per blade in the lattice (always 24: 4 stacked quads)
    per_blade: u32,
}

pub struct Grass {
    gpu: Option<Gpu>,
    cache: Mutex<Cache>,
    /// smoothed wind gust factor (atomic f32 bits: frame paths are &self)
    gust: AtomicU32,
    /// EMA of frame delta time, drives adaptive density
    dt_ema: AtomicU32,
    /// LOD histogram of the last draw: near/mid/far chunk counts
    lod_near: AtomicU32,
    lod_mid: AtomicU32,
    lod_far: AtomicU32,
    /// grass-local visual clock (f32 bits: frame paths are &self): wind
    /// keeps waving while the sim is paused; advanced by real frame dt
    vt: AtomicU32,
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

impl Grass {
    /// TEMP diag: (resident clumps, visible chunk count) without the queue.
    pub fn diag_counts(&self) -> (u32, u32) {
        let g = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        (g.total, g.chunks.len() as u32)
    }

    /// Diagnostics for the status overlay: (near, mid, far) chunk counts
    /// from the last draw pass â€” three cheap Relaxed atomics, no lock.
    pub fn diag_lod(&self) -> (u32, u32, u32) {
        (
            self.lod_near.load(Ordering::Relaxed),
            self.lod_mid.load(Ordering::Relaxed),
            self.lod_far.load(Ordering::Relaxed),
        )
    }

    /// CPU-side state only (contract signature unchanged). The render
    /// pipeline is built by [`ensure_pipeline`] once the coordinator passes
    /// the target formats; until then `draw_with` draws nothing.
    pub fn new(_device: &wgpu::Device) -> Option<Grass> {
        Some(Grass {
            gpu: None,
            cache: Mutex::new(Cache { key: None, chunks: Vec::new(), total: 0 }),
            gust: AtomicU32::new((0.35f32).to_bits()),
            dt_ema: AtomicU32::new((1.0f32 / 60.0).to_bits()),
            lod_near: AtomicU32::new(0),
            lod_mid: AtomicU32::new(0),
            lod_far: AtomicU32::new(0),
            vt: AtomicU32::new(0),
        })
    }

    /// One-time pipeline construction; the coordinator calls this in
    /// `View3D::new` with the very formats passed to `Terrain::new`.
    pub fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let bgl_cam = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.grass.bgl.cam"),
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
        let bgl_gust = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.grass.bgl.gust"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<GrassUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view3d.grass.pl"),
            bind_group_layouts: &[&bgl_cam, &bgl_gust],
            push_constant_ranges: &[],
        });
        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.grass.vs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(GRASS_VS)),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.grass.fs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(GRASS_FS)),
        });
        // shared static clump lattice (blade-major: 7 blades x 24 verts =
        // 168 verts / 56 tris per clump), or the single 24-vert blade
        // fallback while Agent 2's mesh module has not landed
        let (verts, lat_blades) = lattice();
        let vert_count = verts.len() as u32;
        let per_blade = (vert_count / lat_blades.max(1)).max(1);
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.grass.verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.grass.insts"),
            // hard instance budget: <= 9k clumps, 48B each => <= 504k tris
            size: (MAX_BLADES * std::mem::size_of::<GrassInst>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.grass.cam"),
            size: std::mem::size_of::<crate::scene::CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.grass.cam.bg"),
            layout: &bgl_cam,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        let gust_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.grass.gust"),
            contents: bytemuck::bytes_of(&CALM_UNIFORM),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let gust_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.grass.gust.bg"),
            layout: &bgl_gust,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gust_buf.as_entire_binding(),
            }],
        });
        let inst_stride = std::mem::size_of::<GrassInst>() as u64;
        let inst_attrs: Vec<wgpu::VertexAttribute> = (0u32..11)
            .map(|i| wgpu::VertexAttribute {
                offset: i as u64 * 4,
                shader_location: 1 + i,
                format: if i == 0 {
                    wgpu::VertexFormat::Float32x2
                } else {
                    wgpu::VertexFormat::Float32
                },
            })
            .collect();
        let pipeline = device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("view3d.grass.pline"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &vs_mod,
                    entry_point: "vs",
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: inst_stride,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &inst_attrs,
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_mod,
                    entry_point: "fs",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // blades must be visible from both sides
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    // standard forward-Z: egui_wgpu clears the shared depth
                    // attachment to 1.0, so LessEqual is the passing test
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        self.gpu = Some(Gpu {
            pipeline,
            cam_buf,
            cam_bg,
            gust_buf,
            gust_bg,
            verts_buf,
            inst_buf,
            vert_count,
            blades: lat_blades,
            per_blade,
        });
    }

    /// Per-frame CPU regeneration + GPU uploads. The coordinator calls this
    /// in `View3D::update` (needs the queue, which `draw_with` cannot get).
    /// `queen_trample`: visual trample field of the giant queen from the
    /// morph planner (x, z, radius, strength) â€” fallback keeps ordinary
    /// follow targets bending a small patch.
    pub fn update_gpu(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &Frame3D,
        queen_trample: Option<[f32; 4]>,
    ) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        // grass-local visual clock: real frame dt so wind keeps waving while
        // the biological simulation is paused
        let vt = load_f32(&self.vt) + frame.dt.clamp(0.0, 0.1);
        store_f32(&self.vt, vt);
        let world = &frame.world;

        // adaptive density: fps pressure + storms shrink the target count
        let dt = frame.dt.max(1.0 / 240.0);
        let ema = load_f32(&self.dt_ema) * 0.92 + dt * 0.08;
        store_f32(&self.dt_ema, ema);
        let mut scale = (0.024 / ema.max(0.004)).clamp(0.3, 1.0);
        if frame.rain_cover > 0.55 {
            scale *= 0.65;
        }
        let bucket = ((scale * 10.0).round() as i32).clamp(2, 10) as u8;
        let scale = bucket as f32 / 10.0;

        let mode = mode_of(&frame.cam_mode);
        let focus = match frame.cam_mode {
            CameraMode::Orbit { focus } => focus,
            CameraMode::Follow { .. } => {
                frame.follow_target.unwrap_or((world.width * 0.5, world.height * 0.5))
            }
            CameraMode::Planet => (world.width * 0.5, world.height * 0.5),
        };
        let cell = ((focus.0 / 32.0).floor() as i32, (focus.1 / 32.0).floor() as i32);

        let key = CacheKey { gen: world.generation_tag, cell, mode, bucket };
        let rebuild = {
            let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            !cache
                .key
                .as_ref()
                .map(|k| {
                    k.gen == key.gen && k.cell == key.cell && k.mode == key.mode && k.bucket == key.bucket
                })
                .unwrap_or(false)
        };
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if rebuild {
            let res = place::build(
                world,
                &place::Builder {
                    gen: world.generation_tag,
                    cell,
                    mode,
                    focus,
                    lod_scale: scale,
                },
            );
            let inst_size = std::mem::size_of::<GrassInst>();
            let mut flat: Vec<u8> = Vec::with_capacity(res.total * inst_size);
            let mut chunks = Vec::with_capacity(res.chunks.len());
            let mut off = 0u64;
            for c in res.chunks {
                let bytes = bytemuck::cast_slice(&c.insts);
                let beg = off;
                off += bytes.len() as u64;
                flat.extend_from_slice(bytes);
                chunks.push(CacheChunk { cx: c.cx, cy: c.cy, count: c.insts.len() as u32, byte_off: beg });
            }
            if !flat.is_empty() {
                queue.write_buffer(&gpu.inst_buf, 0, &flat);
            }
            *cache = Cache { key: Some(key), chunks, total: res.total as u32 };
        }
        // wind gust: EMA-damped amplitude (target rises with storm cover)
        let target = (0.30 + 0.55 * frame.cloud_cover + 0.45 * frame.rain_cover * frame.cloud_cover)
            .clamp(0.30, 1.6);
        let g = load_f32(&self.gust) + (target - load_f32(&self.gust)) * (dt * 2.5).clamp(0.05, 0.6);
        store_f32(&self.gust, g);
        let uniform = match frame.follow_target {
            Some((x, z)) => {
                // queen trample: visual footprint from the morph planner
                // (scaled to her giant body), fallback kept for ordinary
                // follow targets
                let t = queen_trample.unwrap_or([x, z, 7.0, 0.9]);
                GrassUniform {
                    wind_gust: [WIND_DIR[0], WIND_DIR[1], g, 0.0],
                    trample: t,
                    // grass-local visual clock: keeps waving while the sim
                    // is paused (shader falls back to the shared clock when
                    // this is <= 0)
                    _spare1: [vt, 0.0, 0.0, 0.0],
                    _spare2: [0.0; 4],
                }
            }
            None => GrassUniform {
                wind_gust: [WIND_DIR[0], WIND_DIR[1], g, 0.0],
                trample: [0.0, 0.0, -1.0, 0.0],
                _spare1: [vt, 0.0, 0.0, 0.0],
                _spare2: [0.0; 4],
            },
        };
        queue.write_buffer(&gpu.gust_buf, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&frame.cam));
    }

    /// Instanced draws, one per frustum-visible chunk, with per-chunk blade
    /// LOD on the blade-major lattice: the chunk centre distance to the
    /// camera focus picks the vertex-range sub-draw (near = all 7 blades,
    /// mid = 4, far = 1). No-op until ensure_pipeline and the first
    /// successful update_gpu.
    pub fn draw_with<'a>(
        &self,
        pass: &mut wgpu::RenderPass<'a>,
        frame: Option<&crate::scene::Frame3D>,
    ) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(frame) = frame else { return };
        // fresh LOD histogram per recording pass
        self.lod_near.store(0, Ordering::Relaxed);
        self.lod_mid.store(0, Ordering::Relaxed);
        self.lod_far.store(0, Ordering::Relaxed);
        let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if cache.total == 0 {
            return;
        }
        let f = frustum::Frustum::from_cam(&frame.cam.view, &frame.cam.proj);
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.cam_bg, &[]);
        pass.set_bind_group(1, &gpu.gust_bg, &[]);
        pass.set_vertex_buffer(0, gpu.verts_buf.slice(..));
        let inst_size = std::mem::size_of::<GrassInst>() as u64;
        // same focus + ring radii as the placement density pass
        let world = &frame.world;
        let focus = match frame.cam_mode {
            CameraMode::Orbit { focus } => focus,
            CameraMode::Follow { .. } => {
                frame.follow_target.unwrap_or((world.width * 0.5, world.height * 0.5))
            }
            CameraMode::Planet => (world.width * 0.5, world.height * 0.5),
        };
        let mode = mode_of(&frame.cam_mode);
        let (near_t, mid_t) = rings_for(mode, window_for(mode, world));
        // blade-count sub-draw spans into the blade-major lattice
        let full = gpu.vert_count;
        let mid_span = (4.min(gpu.blades) * gpu.per_blade).min(full);
        let far_span = (1.min(gpu.blades) * gpu.per_blade).min(full);
        for c in &cache.chunks {
            let ccx = c.cx as f32 * place::CHUNK + 16.0;
            let ccz = c.cy as f32 * place::CHUNK + 16.0;
            if !f.sphere_visible([ccx, 24.0, ccz], 20.0) {
                continue;
            }
            let d = ((ccx - focus.0).powi(2) + (ccz - focus.1).powi(2)).sqrt();
            let (span, bin) = if d < near_t {
                (full, &self.lod_near)
            } else if d < mid_t {
                (mid_span, &self.lod_mid)
            } else {
                (far_span, &self.lod_far)
            };
            bin.fetch_add(1, Ordering::Relaxed);
            // per-chunk clump budget (new placement peaks ~130/chunk), with
            // an absolute guard so a drifted chunk cannot wedge the GPU work
            let count = c.count.min(150).min(2000);
            if count == 0 {
                continue;
            }
            pass.set_vertex_buffer(
                1,
                gpu.inst_buf.slice(c.byte_off..c.byte_off + (count as u64) * inst_size),
            );
            pass.draw(0..span, 0..count);
        }
    }
}

/// single-blade fallback lattice vert count (one blade = 4 stacked quads /
/// 8 tris). Used while the clump mesh is missing or drifts to empty, so the
/// pipeline degrades to one blade per instance instead of drawing nothing.
const FALLBACK_BLADES: u32 = 24;

/// Upload lattice: the shared clump mesh (7 blades x 24 verts, blade-major)
/// from `grass_clump`, or the 24-vert single-blade fallback. Returns
/// `(verts, blades_in_lattice)`; every blade occupies a contiguous
/// `[k * 24, (k + 1) * 24)` vert span so sub-draws stay simple.
fn lattice() -> (Vec<[f32; 2]>, u32) {
    let clump = clump_lattice();
    if clump_vert_count() == 0 || clump.is_empty() {
        // Agent-2 drift protection mid-integration
        return (blade_verts(), 1);
    }
    debug_assert!(
        clump.len() == 168,
        "clump lattice drifted: {} verts (expected 168)",
        clump.len()
    );
    (clump, blades_per_clump().max(1) as u32)
}

/// fallback static blade lattice: 4 stacked quads (24 verts / 8 tris); verts
/// are (lateral within [-1,1] pre-profile, v height fraction within [0,1]).
/// The width profile + static curvature are applied per instance in the
/// vertex shader, so this lattice stays variant-agnostic.
fn blade_verts() -> Vec<[f32; 2]> {
    let mut v = Vec::with_capacity(FALLBACK_BLADES as usize);
    for q in 0..4u32 {
        let v0 = q as f32 / 4.0;
        let v1 = (q + 1) as f32 / 4.0;
        let a = [-1.0, v0];
        let b = [1.0, v0];
        let c = [-1.0, v1];
        let d = [1.0, v1];
        for vert in [a, c, b, b, c, d] {
            v.push(vert);
        }
    }
    v
}

const GRASS_VS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;
struct GrassGust {
    wind_gust: vec4<f32>,   // dir_x, dir_z, gust amp, -
    trample: vec4<f32>,     // x, z, radius, strength
    spare1: vec4<f32>,
    spare2: vec4<f32>,
};
@group(1) @binding(0) var<uniform> gwin: GrassGust;

struct VIn {
    @builtin(vertex_index) vid: u32,
    @location(0) blade: vec2<f32>,  // lateral (pre-tapered), v in [0,1]
    @location(1) xz: vec2<f32>,
    @location(2) height: f32,
    @location(3) width: f32,
    @location(4) hue: f32,
    @location(5) lean: f32,
    @location(6) biome: f32,
    @location(7) dim: f32,
    @location(8) seed: f32,
    @location(9) wet: f32,
    @location(10) sway: f32,
    @location(11) pad: f32,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) frag: vec4<f32>,   // (v, hue(+jitter+tip dry), dim, wet)
    @location(2) side: vec3<f32>,   // in-plane lateral direction of the blade
    @location(3) biome: f32,
    @location(4) shim: vec2<f32>,   // (wind-light shimmer flutter, radial AO)
}

const TAU: f32 = 6.2831853;

/// cheap integer hash: seed f32 -> u32 via bitcast, integer mix,
/// fract(k * 43758.5453). Deterministic per (clump seed, stream k).
fn hash_blend(seed: f32, k: u32) -> f32 {
    var x = bitcast<u32>(seed) ^ (k * 0x9E3779B9u);
    x = x ^ (x >> 16u);
    x = x * 0x45D9F3Bu;
    x = x ^ (x >> 16u);
    x = x * 0x45D9F3Bu;
    x = x ^ (x >> 16u);
    return fract(f32(x) * 43758.5453);
}

/// per-blade const tables: (yaw rad, radial_frac, hscale)
var<private> yaw_tbl: array<f32, 7> = array<f32, 7>(
    0.0000000, 2.0943951, 4.1887902,
    1.0471976, 3.1415927, 5.2359878,
    0.2617994   // blade 6: forward lean ~15deg
);
var<private> rad_tbl: array<f32, 7> = array<f32, 7>(
    0.00, 0.55, 0.55,
    0.85, 0.85, 0.85,
    0.50
);
var<private> hs_tbl: array<f32, 7> = array<f32, 7>(
    1.00, 0.92, 0.92,
    0.78, 0.78, 0.78,
    0.84
);

@vertex
fn vs(in: VIn) -> VOut {
    let v = in.blade.y;
    let lat = in.blade.x;
    let h = max(in.height, 0.05);
    // clump spine rotation: lean is the clump yaw phase (2*pi*lean)
    let clump_yaw = in.lean * TAU;
    // blade-major lattice: 24 verts per blade, sub-draws start at vertex 0
    let bid = min(in.vid / 24u, 6u);
    // per-clump/per-blade jitter: edge yaw +-14deg, radial +-0.08,
    // hscale +-7%, hue +-0.008
    let jyaw = (hash_blend(in.seed, bid * 4u + 0u) * 2.0 - 1.0) * 0.2443461;
    let jrad = (hash_blend(in.seed, bid * 4u + 1u) * 2.0 - 1.0) * 0.08;
    let jhs  = (hash_blend(in.seed, bid * 4u + 2u) * 2.0 - 1.0) * 0.07;
    let jhue = (hash_blend(in.seed, bid * 4u + 3u) * 2.0 - 1.0) * 0.008;
    let byaw = yaw_tbl[bid] + jyaw;
    let rk = clamp(rad_tbl[bid] + jrad, 0.0, 1.0);
    let hs = max(hs_tbl[bid] + jhs, 0.3);
    let h_eff = max(h * hs, 0.05);
    // clump spread radius from the shape pad; radial anchor placement
    let spread_r = h * (0.28 + 0.44 * (in.pad + 0.5));
    // radial anchor direction is spun by the clump spine (radial_dir
    // of blade_yaw + clump_yaw), so the rosette rotates with the instance
    let total_yaw = clump_yaw + byaw;
    let radir = vec2(cos(total_yaw), sin(total_yaw));
    let anchor = in.xz + radir * (rk * spread_r);
    // blade plane yaw: spine rotation + blade table yaw + edge jitter
    let plane_yaw = clump_yaw + byaw;
    let side = vec2(cos(plane_yaw), sin(plane_yaw));
    // blade width profile: slightly narrow base, widest at mid, thin tip.
    // per-instance variant (in.pad in [-0.5, 0.5]) shifts where the blade
    // peaks and how pointed the tip reads, for shape variation without
    // extra geometry streams.
    let shape = in.pad;
    let peak = 0.24 + 0.16 * (shape + 0.5);          // 0.24..0.40
    let tip  = 0.10 + 0.12 * (0.5 - abs(shape));    // 0.10..0.16
    let lp = clamp(v / max(peak, 1e-3), 0.0, 1.0);
    var prof: f32;
    if (v <= peak) {
        prof = mix(0.68, 1.0, lp);
    } else {
        let dp = clamp((v - peak) / max(1.0 - peak, 1e-3), 0.0, 1.0);
        prof = mix(1.0, tip, pow(dp, 1.35));
    }
    let base = anchor + side * (lat * prof * in.width * 0.5);
    // grass-local visual clock via the spare uniform slot, with a fallback
    // to the world clock until the lead wires spare1.x
    let vt = select(cam.world_wh_time_flash.z, gwin.spare1.x, gwin.spare1.x > 0.0);
    let gust_a = clamp(gwin.wind_gust.z, 0.0, 2.5) * in.sway;
    // coherent per-clump wave (Gems 7.4.4) + per-blade flutter shimmer
    let wave = sin(vt * 1.8 + (anchor.x + anchor.y) * 0.05 + clump_yaw * 2.0);
    let flut = sin(vt * 5.5 + f32(bid) * 4.7 + in.seed * 11.0);
    var bend = gwin.wind_gust.xy * (h_eff * v * v)
        * (wave * gust_a * 0.55 + flut * gust_a * 0.04);
    // static splay: blades lean outward from the clump spine
    bend += radir * (spread_r * 0.15 * v * (1.0 + 0.3 * abs(jrad)));
    // static droop + asymmetric curl variant (deterministic per blade)
    let curl = shape;
    let lean_d = vec2(sin(clump_yaw * 0.9 + in.seed * 5.0), cos(clump_yaw * 0.9 + in.seed * 3.0));
    bend += lean_d * ((0.09 + 0.05 * abs(curl)) * h_eff * v * v);
    bend += vec2(-side.y, side.x) * (curl * 0.08 * h_eff * v * v);
    // organism interaction: bend away from the trample field centre
    // (field distance taken from the CLUMP anchor)
    let dvec = in.xz - gwin.trample.xy;
    let rr = max(gwin.trample.z, 0.001);
    let fall = clamp(1.0 - dot(dvec, dvec) / (rr * rr), 0.0, 1.0);
    let td = normalize(dvec + vec2(1e-4, 1e-4));
    bend += td * (max(gwin.trample.w, 0.0) * fall * fall * h_eff * v * v);
    let blen = length(bend);
    if (blen > h_eff * 0.9) {
        bend = bend * (h_eff * 0.9 / blen);
    }
    let wp = vec3(base.x + bend.x, v * h_eff - 0.35 * blen * v, base.y + bend.y);
    var o: VOut;
    o.clip = cam.proj * cam.view * vec4(wp, 1.0);
    o.world = wp;
    o.frag = vec4(v, in.hue + jhue + 0.02 * v, in.dim, in.wet);
    o.side = vec3(side.x, 0.0, side.y);
    o.biome = in.biome;
    // shimmer = flutter * gust * 0.12; ao = radial fallback darkening
    o.shim = vec2(flut * gust_a * 0.12, 1.0 - 0.18 * (1.0 - rk));
    return o;
}
"#;

const GRASS_FS: &str = r#"
struct FsIn {
    @location(0) world: vec3<f32>,
    @location(1) frag: vec4<f32>,
    @location(2) side: vec3<f32>,
    @location(3) biome: f32,
    @location(4) shim: vec2<f32>,   // (wind-light shimmer flutter, radial AO)
}

@group(0) @binding(0) var<uniform> cam: CamBlock;
struct CamBlock {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
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
fn fs(in: FsIn) -> @location(0) vec4<f32> {
    let v = in.frag.x;
    let hue = in.frag.y;
    let dim = in.frag.z;
    let wet = in.frag.w;
    let ao_in = in.shim.y;
    // view-facing blade normal with an upward bias that grows toward the tip
    // (curved blades catch the sky at the top, go dark toward the ground)
    let n = normalize(vec3(-in.side.z, 0.30 + 0.55 * v, in.side.x));
    let sun = normalize(vec3(0.35, 0.55, 0.85));
    // view-aware double-sided flip (cull mode is none, so both blade faces
    // are rasterized with the same interpolated normal)
    let dir_v = normalize(cam.view_pos.xyz - in.world);
    let n_sel = n * select(1.0, -1.0, dot(n, dir_v) < 0.0);
    // half-Lambert (soft terminator) + back-light translucency
    let hl = 0.5 + 0.5 * dot(n_sel, sun);
    let lit = 0.45 + 0.55 * hl;
    let tl = max(0.0, -dot(n_sel, sun));
    let sat = 0.30 + 0.06 * (1.0 - in.biome / 6.0);
    // base-to-tip value ramp: dark roots, bright sunlit tips
    let val = mix(0.34, 0.88, pow(v, 0.75));
    var c = hsv(hue, sat, val) * lit * ao_in;
    c *= 0.55 + 0.45 * v;
    c += tl * 0.22 * v;
    // ground-contact band: darken the lowest sliver of the blade
    if (v < 0.08) {
        c *= 0.85;
    }
    // rain-wet sheen
    c = mix(c, vec3(0.16, 0.44, 0.40), wet * 0.45);
    let d = distance(in.world, cam.view_pos.xyz);
    let fade = smoothstep(70.0, 150.0, d);
    c *= mix(1.0, max(dim, 0.50), fade);
    let fog = clamp((d - 700.0) / 2400.0, 0.0, 0.45);
    c = mix(c, vec3(0.55, 0.60, 0.66), fog);
    c *= 1.0 + in.shim.x;
    c += clamp(cam.world_wh_time_flash.w, 0.0, 1.0) * 0.35;
    return vec4(c, 1.0);
}
"#;
