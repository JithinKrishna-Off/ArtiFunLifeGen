//! Trees + bushes GPU module: ONE static mesh upload (shared trunk LOD-B
//! first so sub-draws work like the grass blade lattice), per-TREE-CHUNK
//! instance buffer slices (one 48 B/instance buffer, rebuilt ONLY on cache
//! key change — no per-frame instance streams), chunk-centre frustum
//! culling via the shared grass frustum helper, and chunk-distance LOD:
//! near (<160) draws the full LOD-A span, mid/far draw the LOD-B span; the
//! far ring also got dim 0.55 + half density baked at placement time.
//!
//! Wind animation is deliberately NOT implemented (documented future work:
//! trees stay static to keep the per-vertex cost zero on Iris Xe).
//!
//! Resource rules (same contract as grass): `Tree::new(&device)` prepares
//! CPU state only; the coordinator calls `ensure_pipeline(&device,
//! color_format, depth_format)` once in `View3D::new`, and
//! `update_gpu(&device, &queue, &frame)` every frame in `View3D::update`.
//! Bind group 0 is a tree-owned copy of the camera uniform.

use crate::grass::frustum::Frustum;
use crate::scene::{CameraMode, Frame3D};
use crate::terrain::HEIGHT_SCALE;
use crate::tree_mesh::TreeMesh;
use crate::tree_place as place;
use place::TreeInst;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// hard instance budget: planet caps 400 trees + 800 bushes
const MAX_INSTS: usize = place::CAP_TREE_PLANET + place::CAP_BUSH_PLANET;

struct CacheKey {
    gen: u32,
    cell: (i32, i32),
    mode: u8,
    bucket: u8,
}

struct CacheChunk {
    cx: i32,
    cy: i32,
    /// trees + bushes packed trees-first inside the chunk's instance span
    trees: u32,
    bushes: u32,
    /// source byte offset of the tree span start in the instance buffer
    byte_off: u64,
}

struct Cache {
    key: Option<CacheKey>,
    chunks: Vec<CacheChunk>,
    trees: u32,
    bushes: u32,
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    verts_buf: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    /// LOD-B draw span (trunk + canopy L0)
    lod_b_len: u32,
    /// LOD-A full draw span
    full_len: u32,
    /// bush draw span [bush_first, bush_first + bush_len)
    bush_first: u32,
    bush_len: u32,
}

pub struct Tree {
    gpu: Option<Gpu>,
    cache: Mutex<Cache>,
    /// EMA of frame delta time, drives adaptive density (mirrors grass)
    dt_ema: AtomicU32,
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

impl Tree {
    /// CPU-side state only (contract signature unchanged).
    pub fn new(_device: &wgpu::Device) -> Option<Tree> {
        Some(Tree {
            gpu: None,
            cache: Mutex::new(Cache { key: None, chunks: Vec::new(), trees: 0, bushes: 0 }),
            dt_ema: AtomicU32::new((1.0f32 / 60.0).to_bits()),
        })
    }

    /// diagnostics: (resident trees, bushes, chunks) without the queue
    pub fn diag_counts(&self) -> (u32, u32, u32) {
        let c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        (c.trees, c.bushes, c.chunks.len() as u32)
    }

    /// One-time pipeline + static mesh construction; same target formats
    /// as Terrain::new.
    pub fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let mesh: TreeMesh = crate::tree_mesh::build();
        let bgl_cam = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.tree.bgl.cam"),
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
            label: Some("view3d.tree.pl"),
            bind_group_layouts: &[&bgl_cam],
            push_constant_ranges: &[],
        });
        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.tree.vs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TREE_VS)),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.tree.fs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TREE_FS)),
        });
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.tree.verts"),
            contents: bytemuck::cast_slice(&mesh.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.tree.insts"),
            size: (MAX_INSTS * std::mem::size_of::<TreeInst>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.tree.cam"),
            size: std::mem::size_of::<crate::scene::CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.tree.cam.bg"),
            layout: &bgl_cam,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        // mesh vertex layout: 32 B, pos/nrm/part/yfrac
        let mesh_attrs = &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 24, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 28, shader_location: 3 },
        ];
        // instance layout: 48 B = 12 x f32 (documented stride in tree_place)
        let inst_attrs: Vec<wgpu::VertexAttribute> = (0u32..10)
            .map(|i| wgpu::VertexAttribute {
                offset: i as u64 * 4,
                shader_location: 4 + i,
                format: if i == 0 {
                    wgpu::VertexFormat::Float32x2
                } else {
                    wgpu::VertexFormat::Float32
                },
            })
            .collect();
        let pipeline = device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("view3d.tree.pline"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &vs_mod,
                    entry_point: "vs",
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<crate::tree_mesh::MeshVert>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: mesh_attrs,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<TreeInst>() as u64,
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
                    // deterministic mesh winding is not cull-verified; keep
                    // both faces so a bad hash jitter cannot punch holes
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
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
            verts_buf,
            inst_buf,
            lod_b_len: mesh.lod_b_len,
            full_len: mesh.full_len,
            bush_first: mesh.bush_first,
            bush_len: mesh.bush_len,
        });
    }

    /// Per-frame CPU regeneration + GPU uploads (mirror of grass, sans
    /// wind). Coordinator calls this in `View3D::update`.
    pub fn update_gpu(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame3D) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let world = &frame.world;
        // adaptive density bucket: same pressure model as grass, but
        // trees never go below 0.4 density
        let dt = frame.dt.max(1.0 / 240.0);
        let ema = load_f32(&self.dt_ema) * 0.92 + dt * 0.08;
        store_f32(&self.dt_ema, ema);
        let scale = (0.024 / ema.max(0.004)).clamp(0.3, 1.0);
        let bucket = ((scale * 10.0).round() as i32).clamp(4, 10) as u8;
        let scale = bucket as f32 / 10.0;

        let mode = place::mode_of(&frame.cam_mode);
        let focus = match frame.cam_mode {
            CameraMode::Orbit { focus } => focus,
            CameraMode::Follow { .. } => {
                frame.follow_target.unwrap_or((world.width * 0.5, world.height * 0.5))
            }
            CameraMode::Planet => (world.width * 0.5, world.height * 0.5),
        };
        // tree chunks are 64x64 == terrain chunks (alignment by design)
        let cell = ((focus.0 / place::CHUNK).floor() as i32, (focus.1 / place::CHUNK).floor() as i32);

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
                &place::TreeBuilder {
                    gen: world.generation_tag,
                    cell,
                    mode,
                    focus,
                    lod_scale: scale,
                },
            );
            let inst_size = std::mem::size_of::<TreeInst>();
            let mut flat: Vec<u8> =
                Vec::with_capacity((res.trees + res.bushes) * inst_size);
            let mut chunks = Vec::with_capacity(res.chunks.len());
            let mut off = 0u64;
            for c in res.chunks {
                let beg = off;
                for t in c.trees.iter().chain(c.bushes.iter()) {
                    flat.extend_from_slice(bytemuck::bytes_of(t));
                }
                off += (c.trees.len() + c.bushes.len()) as u64 * inst_size as u64;
                chunks.push(CacheChunk {
                    cx: c.cx,
                    cy: c.cy,
                    trees: c.trees.len() as u32,
                    bushes: c.bushes.len() as u32,
                    byte_off: beg,
                });
            }
            if !flat.is_empty() {
                queue.write_buffer(&gpu.inst_buf, 0, &flat);
            }
            *cache = Cache {
                key: Some(key),
                chunks,
                trees: res.trees as u32,
                bushes: res.bushes as u32,
            };
        }
        queue.write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&frame.cam));
    }

    /// Instanced draws, one to two per frustum-visible tree chunk: trees
    /// draw the LOD span picked by chunk distance, bushes draw the bush
    /// span. No-op until ensure_pipeline + first successful update_gpu.
    pub fn draw_with<'a>(
        &self,
        pass: &mut wgpu::RenderPass<'a>,
        frame: Option<&crate::scene::Frame3D>,
    ) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(frame) = frame else { return };
        let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if cache.trees + cache.bushes == 0 {
            return;
        }
        let f = Frustum::from_cam(&frame.cam.view, &frame.cam.proj);
        let world = &frame.world;
        let focus = match frame.cam_mode {
            CameraMode::Orbit { focus } => focus,
            CameraMode::Follow { .. } => {
                frame.follow_target.unwrap_or((world.width * 0.5, world.height * 0.5))
            }
            CameraMode::Planet => (world.width * 0.5, world.height * 0.5),
        };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.cam_bg, &[]);
        pass.set_vertex_buffer(0, gpu.verts_buf.slice(..));
        let inst_size = std::mem::size_of::<TreeInst>() as u64;
        for c in &cache.chunks {
            let ccx = c.cx as f32 * place::CHUNK + 32.0;
            let ccz = c.cy as f32 * place::CHUNK + 32.0;
            // ground-contact sphere at the chunk centre (one field sample)
            let gy = world.sample_elevation(ccx, ccz) * HEIGHT_SCALE;
            if !f.sphere_visible([ccx, gy + 7.0, ccz], 30.0) {
                continue;
            }
            let d = ((ccx - focus.0).powi(2) + (ccz - focus.1).powi(2)).sqrt();
            // near ring -> full LOD-A span, mid/far -> LOD-B span
            let span = if d < place::RING_NEAR { gpu.full_len } else { gpu.lod_b_len };
            if c.trees > 0 {
                pass.set_vertex_buffer(
                    1,
                    gpu.inst_buf.slice(c.byte_off..c.byte_off + (c.trees as u64) * inst_size),
                );
                pass.draw(0..span, 0..c.trees);
            }
            if c.bushes > 0 {
                let b0 = c.byte_off + (c.trees as u64) * inst_size;
                pass.set_vertex_buffer(1, gpu.inst_buf.slice(b0..b0 + (c.bushes as u64) * inst_size));
                pass.draw(gpu.bush_first..gpu.bush_first + gpu.bush_len, 0..c.bushes);
            }
        }
    }
}

const TREE_VS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct VIn {
    // mesh (32 B stride): pos, normal, part (0 trunk / 1 canopy), yfrac
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) part: f32,
    @location(3) yfrac: f32,
    // instance (48 B stride = 12 x f32, order mirrors TreeInst)
    @location(4) ixz: vec2<f32>,
    @location(5) ibase_y: f32,
    @location(6) iyaw: f32,
    @location(7) iscale: f32,
    @location(8) ihue: f32,
    @location(9) idim: f32,
    @location(10) ikind: f32,
    @location(11) irad: f32,
    @location(12) iseed: f32,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) nrmw: vec3<f32>,
    @location(2) frag: vec4<f32>,  // (part, hue, dim, yfrac)
};

/// static trees wind anchor: NO per-frame animation (future work, see
/// module docs); yaw rotation + uniform scaling only
@vertex
fn vs(in: VIn) -> VOut {
    let a = cos(in.iyaw);
    let s = sin(in.iyaw);
    let lx = in.pos.x * in.iscale * in.irad;
    let lz = in.pos.z * in.iscale * in.irad;
    let ly = in.pos.y * in.iscale;
    let wx = in.ixz.x + lx * a - lz * s;
    let wz = in.ixz.y + lx * s + lz * a;
    let wy = in.ibase_y + ly;
    let nx = in.nrm.x * a - in.nrm.z * s;
    let nz = in.nrm.x * s + in.nrm.z * a;
    var o: VOut;
    o.world = vec3(wx, wy, wz);
    o.nrmw = normalize(vec3(nx, in.nrm.y, nz));
    o.frag = vec4(in.part, in.ihue, in.idim, in.yfrac);
    o.clip = cam.proj * cam.view * vec4(o.world, 1.0);
    return o;
}
"#;

const TREE_FS: &str = r#"
struct FsIn {
    @location(0) world: vec3<f32>,
    @location(1) nrmw: vec3<f32>,
    @location(2) frag: vec4<f32>,
}

struct CamBlock {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: CamBlock;

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
    let is_canopy = in.frag.x > 0.5;
    let n = normalize(in.nrmw);
    let sun = normalize(vec3(0.35, 0.55, 0.85));
    let dir_v = normalize(cam.view_pos.xyz - in.world);
    let n_sel = n * select(1.0, -1.0, dot(n, dir_v) < 0.0);
    let hl = 0.5 + 0.5 * dot(n_sel, sun);
    let lit = 0.42 + 0.58 * hl;
    var c: vec3<f32>;
    if (is_canopy) {
        // vertical gradient: dark under-layers, bright sunlit tops
        let t = in.frag.w;
        let val = mix(0.30, 0.86, pow(t, 0.7));
        c = hsv(in.frag.y, 0.42, val) * lit;
    } else {
        // low-poly bark: brown, darker roots via yfrac
        let val = mix(0.30, 0.46, in.frag.w);
        c = hsv(0.075, 0.38, val) * lit;
    }
    let d = distance(in.world, cam.view_pos.xyz);
    // LOD dim ramp anchored at the ring radii (dim is baked per instance
    // by the placement pass: 1.0 near/mid, 0.55 far)
    let dramp = smoothstep(140.0, 430.0, d);
    c *= mix(1.0, max(in.frag.z, 0.5), dramp);
    let fog = clamp((d - 700.0) / 2400.0, 0.0, 0.55);
    c = mix(c, vec3(0.55, 0.60, 0.66), fog);
    c += clamp(cam.world_wh_time_flash.w, 0.0, 1.0) * 0.35;
    return vec4(c, 1.0);
}
"#;
