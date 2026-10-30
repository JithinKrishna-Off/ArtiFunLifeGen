//! 3D terrain from the real `WorldModel` fields (owned by Agent A).
//! M1 baseline: CPU-generated heightfield window mesh rebuilt only when the
//! camera window or world model changes. Chunked LOD/caching is Agent A's
//! deliverable layered onto this file; the draw API below must stay stable:
//! `new(device, color_format, depth_format) -> Option<Terrain>`,
//! `ensure_mesh(...)`, `cam_buf()`, `pipeline()`, `bind_group()`,
//! `vertex_buffer()`, `invalidate()`.

use crate::scene::{CameraMode, CamUniform};
use alife_worldgen::WorldModel;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const SEA_LEVEL: f32 = 0.42;
/// world-units of vertical scale per unit of the [0,1] elevation field
pub const HEIGHT_SCALE: f32 = 46.0;
/// habitat window in world units (square around the focus)
pub const HABITAT_WINDOW: f32 = 260.0;
/// planet view resolution square cap (vertices per axis)
const PLANET_N: u32 = 220;
const HABITAT_N: u32 = 170;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LandVert {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub struct Terrain {
    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    verts: Vec<LandVert>,
    vbuf: Option<wgpu::Buffer>,
    key: Option<CacheKey>,
}

/// Identity of the built mesh: generation tag + window cell + res class.
struct CacheKey {
    epoch: u64,
    cell: (i32, i32),
    res: u32,
}

pub fn terrain_height_at(world: &WorldModel, x: f32, y: f32) -> f32 {
    world.sample_elevation(x, y) * HEIGHT_SCALE
}

pub fn biome_color(b: f32, e: f32, moisture: f32) -> [f32; 3] {
    let hue = (b * 0.045).clamp(0.0, 0.95);
    let l = (e * 0.7 + 0.3).clamp(0.16, 1.0);
    let sat = (0.22 + 0.26 * moisture.clamp(0.0, 1.0) - 0.09 * (b * b).min(4.0)).clamp(0.05, 0.5);
    hsv_to_rgb(hue, sat, l)
}

fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let m = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
    let sign = if n[1] >= 0.0 { 1.0 } else { -1.0 };
    [n[0] / m * sign, n[1] / m * sign, n[2] / m * sign]
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b]
}

impl Terrain {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Option<Terrain> {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.terrain.bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CamUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view3d.terrain.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.terrain.vs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TERRAIN_VS)),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.terrain.fs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TERRAIN_FS)),
        });
        let pipeline = device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("view3d.terrain.pline"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &vs_mod,
                    entry_point: "vs",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<LandVert>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                        ],
                    }],
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
                    cull_mode: Some(wgpu::Face::Back),
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
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.terrain.cam"),
            size: std::mem::size_of::<CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.terrain.cam.bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }],
        });
        Some(Terrain {
            pipeline,
            cam_buf,
            cam_bg,
            verts: Vec::with_capacity(1 << 17),
            vbuf: None,
            key: None,
        })
    }

    pub fn cam_buf(&self) -> &wgpu::Buffer {
        &self.cam_buf
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.cam_bg
    }

    pub fn invalidate(&mut self) {
        self.key = None;
    }

    pub fn vertex_buffer(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.vbuf.as_ref().map(|b| (b, (self.verts.len()) as u32))
    }

    /// Rebuilds the heightfield window mesh only when the region of interest
    /// changed. Pure CPU generation from the world fields.
    pub fn ensure_mesh(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        world: &Arc<WorldModel>,
        mode: CameraMode,
        focus: (f32, f32),
    ) {
        let (window, n) = match mode {
            CameraMode::Planet => ((world.width.max(world.height) * 1.05).max(32.0), PLANET_N),
            CameraMode::Orbit { .. } => (HABITAT_WINDOW, HABITAT_N),
            CameraMode::Follow { .. } => (90.0, 120),
        };
        let cell_key = (
            (focus.0 / 32.0).floor() as i32,
            (focus.1 / 32.0).floor() as i32,
        );
        let new_key = CacheKey {
            epoch: world.generation_tag as u64,
            cell: cell_key,
            res: n,
        };
        let same = self
            .key
            .as_ref()
            .map(|k| k.epoch == new_key.epoch && k.cell == new_key.cell && k.res == new_key.res)
            .unwrap_or(false);
        if same && self.vbuf.is_some() {
            return;
        }
        self.verts.clear();
        let (nx, ny) = (n, (n as f32 * 0.72).max(24.0) as u32);
        let dx = window / nx as f32;
        let dy = (window * 0.72) / ny as f32;
        let x0 = (focus.0 - window * 0.5).clamp(0.0, world.width.max(0.0));
        let y0 = (focus.1 - window * 0.5).clamp(0.0, world.height.max(0.0));
        let bounce = |v: f32, max: f32| v.min((max - 0.01).max(0.0));
        // sample the world fields on the grid
        let mut h = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
        let mut bio = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
        let mut moist = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
        for iy in 0..=ny {
            let wy = bounce(y0 + dy * iy as f32, world.height);
            for ix in 0..=nx {
                let wx = bounce(x0 + dx * ix as f32, world.width);
                h.push(world.sample_elevation(wx, wy));
                bio.push(world.biome.sample(wx, wy, world.width, world.height));
                moist.push(world.sample_moisture(wx, wy));
            }
        }
        let stride = nx as usize + 1;
        let at = |ix: usize, iy: usize| -> LandVert {
            let i = iy * stride + ix;
            let e = h[i];
            let b = bio[i];
            let m = moist[i];
            let wx = bounce(x0 + dx * ix as f32, world.width);
            let wy = bounce(y0 + dy * iy as f32, world.height);
            LandVert {
                pos: [wx, e * HEIGHT_SCALE, wy],
                normal: [0.0, 1.0, 0.0],
                color: biome_color(b, e, m),
            }
        };
        for iy in 0..ny as usize {
            for ix in 0..nx as usize {
                let p00 = at(ix, iy);
                let p10 = at(ix + 1, iy);
                let p01 = at(ix, iy + 1);
                let p11 = at(ix + 1, iy + 1);
                // triangle winds depend on height so both get correct normals
                for tri in [(p00, p01, p10), (p10, p01, p11)] {
                    let (a, b, c) = tri;
                    let nn = tri_normal(a.pos, b.pos, c.pos);
                    if nn[1] < -0.9 {
                        // inverted winding: flip vertices instead
                        let _ = nn;
                    }
                    let nrev = tri_normal(a.pos, c.pos, b.pos);
                    self.verts.push(LandVert { pos: a.pos, normal: nrev, color: a.color });
                    self.verts.push(LandVert { pos: c.pos, normal: nrev, color: c.color });
                    self.verts.push(LandVert { pos: b.pos, normal: nrev, color: b.color });
                }
            }
        }
        let bytes = bytemuck::cast_slice(&self.verts).to_vec();
        self.vbuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.terrain.vbuf"),
            contents: &bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        self.key = Some(new_key);
    }
}

const TERRAIN_VS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@vertex
fn vs(in: VIn) -> VOut {
    var o: VOut;
    o.clip = cam.proj * cam.view * vec4<f32>(in.pos, 1.0);
    o.world = in.pos;
    o.normal = in.normal;
    o.color = in.color;
    return o;
}
"#;

const TERRAIN_FS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct FsIn {
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@fragment
fn fs(in: FsIn) -> @location(0) vec4<f32> {
    let sun_dir = normalize(vec3(0.35, 0.55, 0.85));
    let lambert = clamp(dot(normalize(in.normal), sun_dir), 0.0, 1.0);
    var c = in.color * (0.30 + 0.80 * lambert);
    // underwater / shoreline tint where the height is close to/below sea level
    let hn = in.world.y / 46.0;
    let under = clamp((0.42 - hn) * 6.5, 0.0, 1.0);
    c = mix(c, vec3(0.20, 0.48, 0.56), under * 0.85);
    let d = distance(in.world, cam.view_pos.xyz);
    let fog = clamp((d - 500.0) / 2600.0, 0.0, 0.62);
    let flash = clamp(cam.world_wh_time_flash.w, 0.0, 1.0);
    c = mix(c, vec3(0.55, 0.60, 0.66), fog);
    c += flash * 0.35;
    return vec4(c, 1.0);
}
"#;
