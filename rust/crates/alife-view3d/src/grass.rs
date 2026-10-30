//! Procedural vegetation (grass / tufts / moss / shore algae) instanced on
//! the land. Agent B's deliverable: per-blade instancing with compact
//! per-instance attributes, wind animation in the vertex shader, organism
//! interaction (trample field), density LOD folded into the deterministic
//! drop rate, and cheap per-chunk frustum culling.
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

use crate::scene::{CameraMode, Frame3D};
use place::{mode_of, GrassInst, MAX_BLADES};
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
}

pub struct Grass {
    gpu: Option<Gpu>,
    cache: Mutex<Cache>,
    /// smoothed wind gust factor (atomic f32 bits: frame paths are &self)
    gust: AtomicU32,
    /// EMA of frame delta time, drives adaptive density
    dt_ema: AtomicU32,
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

impl Grass {
    /// CPU-side state only (contract signature unchanged). The render
    /// pipeline is built by [`ensure_pipeline`] once the coordinator passes
    /// the target formats; until then `draw_with` draws nothing.
    pub fn new(_device: &wgpu::Device) -> Option<Grass> {
        Some(Grass {
            gpu: None,
            cache: Mutex::new(Cache { key: None, chunks: Vec::new(), total: 0 }),
            gust: AtomicU32::new((0.35f32).to_bits()),
            dt_ema: AtomicU32::new((1.0f32 / 60.0).to_bits()),
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
        // shared static blade mesh: 3 tapered quads = 18 verts / 6 tris
        let verts = blade_verts();
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.grass.verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.grass.insts"),
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
                    depth_compare: wgpu::CompareFunction::GreaterEqual,
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
        });
    }

    /// Per-frame CPU regeneration + GPU uploads. The coordinator calls this
    /// in `View3D::update` (needs the queue, which `draw_with` cannot get).
    pub fn update_gpu(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame3D) {
        let Some(gpu) = self.gpu.as_ref() else { return };
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
            Some((x, z)) => GrassUniform {
                wind_gust: [WIND_DIR[0], WIND_DIR[1], g, 0.0],
                trample: [x, z, 7.0, 0.9],
                _spare1: [0.0; 4],
                _spare2: [0.0; 4],
            },
            None => GrassUniform { wind_gust: [WIND_DIR[0], WIND_DIR[1], g, 0.0], ..CALM_UNIFORM },
        };
        queue.write_buffer(&gpu.gust_buf, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&frame.cam));
    }

    /// Instanced draws, one per frustum-visible chunk. No-op until
    /// ensure_pipeline and the first successful update_gpu.
    pub fn draw_with<'a>(
        &self,
        pass: &mut wgpu::RenderPass<'a>,
        frame: Option<&crate::scene::Frame3D>,
    ) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(frame) = frame else { return };
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
        for c in &cache.chunks {
            let ccx = c.cx as f32 * place::CHUNK + 16.0;
            let ccz = c.cy as f32 * place::CHUNK + 16.0;
            if !f.sphere_visible([ccx, 24.0, ccz], 20.0) {
                continue;
            }
            pass.set_vertex_buffer(
                1,
                gpu.inst_buf.slice(c.byte_off..c.byte_off + (c.count as u64) * inst_size),
            );
            pass.draw(0..18, 0..c.count);
        }
    }
}

/// shared static blade geometry: 3 tapered quads (18 verts / 6 tris); verts
/// are (lateral within [-1,1], v height fraction within [0,1])
fn blade_verts() -> Vec<[f32; 2]> {
    let taper = |vy: f32| 1.0 - 0.85 * vy;
    let mut v = Vec::with_capacity(18);
    for q in 0..3u32 {
        let v0 = q as f32 / 3.0;
        let v1 = (q + 1) as f32 / 3.0;
        let (t0, t1) = (taper(v0), taper(v1));
        let a = [-t0, v0];
        let b = [t0, v0];
        let c = [-t1, v1];
        let d = [t1, v1];
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
    @location(1) frag: vec4<f32>,   // (v, hue, dim, wet)
    @location(2) side: vec3<f32>,   // in-plane lateral direction of the blade
    @location(3) biome: f32,
}

const TAU: f32 = 6.2831853;

@vertex
fn vs(in: VIn) -> VOut {
    let v = in.blade.y;
    let lat = in.blade.x;
    let h = max(in.height, 0.05);
    let yaw = in.lean * TAU + in.seed * 17.0;
    let side = vec2(cos(yaw), sin(yaw));
    let base = in.xz + side * (lat * in.width * 0.5);
    let t = cam.world_wh_time_flash.z;
    let phase = in.lean * TAU;
    let sway_mul = clamp(in.sway, 0.2, 1.5);
    let gust_a = clamp(gwin.wind_gust.z, 0.0, 2.5) * sway_mul;
    let wave = sin(t * 1.8 + (in.xz.x + in.xz.y) * 0.05 + phase * 2.0);
    let flut = sin(t * 5.5 + phase * 7.0 + in.seed * 11.0);
    var bend = gwin.wind_gust.xy * (h * 0.30 * v * v) * (wave * gust_a * 0.9 + flut * 0.08 * gust_a);
    let lean_d = vec2(sin(phase * 0.9 + in.seed * 5.0), cos(phase * 0.9 + in.seed * 3.0));
    bend += lean_d * (0.16 * h * v * v);
    // organism interaction: bend away from the trample field centre
    let dvec = in.xz - gwin.trample.xy;
    let rr = max(gwin.trample.z, 0.001);
    let fall = clamp(1.0 - dot(dvec, dvec) / (rr * rr), 0.0, 1.0);
    let td = normalize(dvec + vec2(1e-4, 1e-4));
    bend += td * (max(gwin.trample.w, 0.0) * fall * fall * h * v * v);
    let blen = length(bend);
    if (blen > h * 0.9) {
        bend = bend * (h * 0.9 / blen);
    }
    let wp = vec3(base.x + bend.x, v * h - 0.5 * blen * v, base.y + bend.y);
    var o: VOut;
    o.clip = cam.proj * cam.view * vec4(wp, 1.0);
    o.world = wp;
    o.frag = vec4(v, in.hue, in.dim, in.wet);
    o.side = vec3(side.x, 0.0, side.y);
    o.biome = in.biome;
    return o;
}
"#;

const GRASS_FS: &str = r#"
struct FsIn {
    @location(0) world: vec3<f32>,
    @location(1) frag: vec4<f32>,
    @location(2) side: vec3<f32>,
    @location(3) biome: f32,
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
    let n = normalize(vec3(-in.side.z, 0.35 + 0.4 * v, in.side.x));
    let sun = normalize(vec3(0.35, 0.55, 0.85));
    let lambert = clamp(abs(dot(n, sun)), 0.0, 1.0);
    let sat = 0.40 + 0.06 * (1.0 - in.biome / 6.0);
    var c = hsv(hue, sat, mix(0.42, 0.74, v)) * (0.42 + 0.68 * lambert);
    c *= mix(0.45, 1.0, v);
    c = mix(c, vec3(0.16, 0.44, 0.40), wet * 0.45);
    let d = distance(in.world, cam.view_pos.xyz);
    let fade = smoothstep(70.0, 150.0, d);
    c *= mix(1.0, max(dim, 0.35), fade);
    let fog = clamp((d - 500.0) / 2600.0, 0.0, 0.62);
    c = mix(c, vec3(0.55, 0.60, 0.66), fog);
    c += clamp(cam.world_wh_time_flash.w, 0.0, 1.0) * 0.35;
    return vec4(c, 1.0);
}
"#;
