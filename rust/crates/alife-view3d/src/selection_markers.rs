//! Selection / hover markers in the 3D viewport (Agent: view3d).
//!
//! # What draws
//! Camera-facing billboard quads (ONE extra InstancedTriangleList draw call,
//! max 3 instances: selected, hovered, queen) rendered by the morph pass via
//! `Morph::draw_markers` right after the ordinary morph draws. The ring is a
//! GPU fragment band (no extra geometry beyond 6 quad vertices); styles:
//! selected = solid hue-white ring, hovered = thin dashed ring (dash rotates
//! slowly with scene time), queen = gold double ring. Depth matches the
//! morph pass (forward-Z, LessEqual, no depth write, alpha blend).
//!
//! # Zoom behavior (no transition jumps)
//! Marker radius is a pure, continuous function of the organism's body
//! radius and the camera distance: `r = body * 1.35 * (1 + 0.22 *
//! ln(1 + d/180)) * queen_offset`, clamped to [1, 80]. There is no camera-
//! mode branch, so orbit->follow->planet transitions can never produce a
//! step; the distance term only grows smoothly with zoom-out so the ring
//! stays legible. A `close_fade` instance factor fades the ring out as the
//! camera dives inside it (never a full-screen overlay).
//!
//! # How the UI feeds selection (stable API for the UI agent)
//! - [`set_selection`] / [`View3D::set_selection`] (see `render.rs`) /
//!   `alife_view3d::render::set_selection` — call any time (per frame or on
//!   change): `set_selection(selected: Option<u32>, hovered: Option<u32>,
//!   queen_id: Option<u32>)`. Arguments are *indices into the Frame3D orgs
//!   slice* (exactly what `pick_org` / `hover_org` return; `queen_id = None`
//!   falls back to the snapshot's `is_queen` flag via `scene::find_queen`).
//! - State is consumed on the NEXT frame's `Morph::update_gpu` (cheap
//!   snapshot copy of a 3-Option struct; no per-frame allocations beyond the
//!   fixed `[MarkerInst; 3]` upload — no Vec, no per-frame heap traffic).
//!
//! `plan_markers` (pure, device-free) computes the instances and is fully
//! unit-tested: sanitizes NaN/degenerate fields, never emits more than
//! [`MAX_MARKERS`] instances, and always clamps radii into bounds.

use bytemuck::Zeroable;
use crate::scene::OrgInst;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// instance styles (must match the WGSL branches below)
pub const MARKER_SELECTED: u32 = 0;
pub const MARKER_HOVERED: u32 = 1;
pub const MARKER_QUEEN: u32 = 2;
/// hard cap: at most selected + hovered + queen instanced in one draw
pub const MAX_MARKERS: usize = 3;

/// UI-side selection state (indices into the frame's orgs slice).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelectionMarkers {
    pub selected: Option<u32>,
    pub hovered: Option<u32>,
    /// Some(i) forces the queen marker onto orgs slice index `i`; None
    /// falls back to scanning `OrgInst.is_queen` in the current frame.
    pub queen_id: Option<u32>,
}

static PENDING: Mutex<SelectionMarkers> = Mutex::new(SelectionMarkers {
    selected: None,
    hovered: None,
    queen_id: None,
});

/// UI entry: set this frame's selection/hover/queen indices (orgs slice
/// indices). Wire through `View3D::set_selection` or the free
/// `render::set_selection` alias.
pub fn set_selection(selected: Option<u32>, hovered: Option<u32>, queen_id: Option<u32>) {
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    *g = SelectionMarkers { selected, hovered, queen_id };
}

/// Copies the pending selection state (poison-safe).
pub fn current() -> SelectionMarkers {
    PENDING.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// One marker instance uploaded to the GPU (8 x f32 = 32 bytes stride).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MarkerInst {
    /// world center of the marker ring
    pub pos: [f32; 3],
    /// ring radius in world units (zoom-compensated, clamped)
    pub radius: f32,
    /// [`MARKER_SELECTED`] / [`MARKER_HOVERED`] / [`MARKER_QUEEN`]
    pub style: f32,
    /// species hue in [0,1) (shader refracts only for selected/hovered)
    pub hue: f32,
    /// near-camera fade factor in [0,1] (computed on the CPU, deterministic)
    pub close_fade: f32,
    pub pad: f32,
}

/// The marker's base body radius. Ordinary organisms (mul = 1.0, or invalid
/// `OrgInst::zeroed` 0.0 / NaN) read the tiny re-based unit radius
/// ([`crate::morph::decomp::ORGANISM_UNIT_RADIUS`]) so rings hug the small
/// silhouette the planner now renders. Any other finite positive mul (the
/// queen = [`crate::morph::decomp::QUEEN_SCALE`], ~= 0.4435, per the
/// `visual_radius_mul` mirror contract) cancels its own still-shared mirror
/// base (`QUEEN_UNIT_RADIUS / QUEEN_SCALE` = the legacy 5.5), so her ring
/// radius stays exactly `QUEEN_UNIT_RADIUS * size * turgor` — identical to
/// the agent-7/8 pick/marked ring for the giant rebalance and independent of
/// the ordinary rebase.
pub fn visual_body_radius(o: &OrgInst) -> f32 {
    let mul = if o.visual_radius_mul.is_finite() && o.visual_radius_mul > 0.0 {
        o.visual_radius_mul
    } else {
        1.0
    };
    let e = o.energy.clamp(0.0, 1.0);
    let tf = 0.62 + 0.38 * e;
    let size = o.size.clamp(0.3, 2.0);
    let base = if mul == 1.0 {
        crate::morph::decomp::ORGANISM_UNIT_RADIUS
    } else {
        crate::morph::decomp::QUEEN_UNIT_RADIUS / crate::morph::decomp::QUEEN_SCALE
    };
    (base * size * tf * mul).clamp(0.2, 30.0)
}

/// Marker ring radius for one style at camera distance `dist` (world units):
/// continuous zoom compensation, style offsets so coincident markers stay
/// distinct, clamped. The floor is 1.0 (tiny organisms still need a legible
/// highlight ring without swallowing them like the old 2.0 floor did after
/// the `ORGANISM_UNIT_RADIUS` rebase). Pure + deterministic (no camera-mode
/// branch => no jumps across orbit/follow/planet transitions).
pub fn marker_radius(body: f32, style: u32, dist: f32) -> f32 {
    let d = if dist.is_finite() { dist.max(0.0) } else { 0.0 };
    let comp = 1.0 + 0.22 * (1.0 + d / 180.0).ln();
    let style_off = match style {
        MARKER_QUEEN => 1.18,
        MARKER_HOVERED => 1.06,
        _ => 1.0,
    };
    (body * 1.35 * comp * style_off).clamp(1.0, 80.0)
}

/// Near-camera fade: fully faded when the camera sits within ~1.5 ring
/// radii, opaque past ~3 radii. ContinuousInOut.
pub fn marker_close_fade(radius: f32, dist: f32) -> f32 {
    let r = radius.max(0.02);
    let d = if dist.is_finite() { dist.max(0.0) } else { 0.0 };
    (d / (r * 2.5)).clamp(0.0, 1.0)
}

/// Ground-plane position of an organism's marker center (same grounding the
/// planner uses: terrain height + 0.62 * body radius). None for
/// non-finite/NaN plan positions (marker skipped, never NaN).
fn marker_center(o: &OrgInst, body: f32, height: &mut dyn FnMut(f32, f32) -> f32) -> Option<[f32; 3]> {
    if !(o.x.is_finite() && o.y.is_finite()) {
        return None;
    }
    let gy = height(o.x, o.y);
    let gy = if gy.is_finite() { gy } else { 0.0 };
    Some([o.x, gy + body * 0.62, o.y])
}

/// Pure planner: builds at most [`MAX_MARKERS`] instances in the fixed order
/// [selected, hovered, queen]; returns `(instances, count)` — only the first
/// `count` entries are meaningful (the rest are zeroed). Device-free and
/// allocation-free (fixed-size array).
pub fn plan_markers(
    orgs: &[OrgInst],
    view_pos: [f32; 3],
    height: &mut dyn FnMut(f32, f32) -> f32,
    sel: &SelectionMarkers,
) -> ([MarkerInst; MAX_MARKERS], usize) {
    let mut out = [MarkerInst::zeroed(); MAX_MARKERS];
    let mut n = 0usize;
    // queen fallback: trust the UI's explicit queen index, else scan the flag
    let queen_i: Option<usize> = sel
        .queen_id
        .filter(|i| (*i as usize) < orgs.len())
        .map(|i| i as usize)
        .or_else(|| orgs.iter().position(|o| o.is_queen != 0));
    let wanted: [(Option<u32>, u32); 3] = [
        (sel.selected, MARKER_SELECTED),
        (sel.hovered, MARKER_HOVERED),
        (queen_i.map(|i| i as u32), MARKER_QUEEN),
    ];
    for (idx, style) in wanted {
        let Some(i) = idx else { continue };
        let i = i as usize;
        if i >= orgs.len() {
            continue;
        }
        let o = &orgs[i];
        if o.alive < 0.5 {
            continue;
        }
        let body = visual_body_radius(o);
        if !(body.is_finite() && body > 0.0) {
            continue;
        }
        let Some(center) = marker_center(o, body, height) else { continue };
        let dx = center[0] - view_pos[0];
        let dy = center[1] - view_pos[1];
        let dz = center[2] - view_pos[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt().max(0.0);
        if !d.is_finite() {
            continue;
        }
        let radius = marker_radius(body, style, d);
        let hue = o.hue.rem_euclid(1.0);
        if !(radius.is_finite() && radius > 0.0 && hue.is_finite()) {
            continue;
        }
        out[n] = MarkerInst {
            pos: center,
            radius,
            style: style as f32,
            hue,
            close_fade: marker_close_fade(radius, d),
            pad: 0.0,
        };
        n += 1;
    }
    for inst in out[..n].iter_mut() {
        debug_assert!(inst.pos.iter().all(|v| v.is_finite()));
    }
    (out, n)
}

// ------------------------------------------------------------- GPU rig

const QUAD_CORNERS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

pub const MARKER_INST_FLOATS: usize = 8;
pub const MARKER_INST_STRIDE: usize = MARKER_INST_FLOATS * 4;
pub const MARKER_INST_CAP: usize = MAX_MARKERS;

struct MarkerGpu {
    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    verts_buf: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
}

/// Owns the marker pass's GPU state; built lazily inside
/// `Morph::ensure_pipeline` (one-time), updated + drawn by the morph module.
pub struct MarkerRig {
    gpu: Option<MarkerGpu>,
    /// last planned instance count (drives the single draw call)
    planned_n: std::sync::atomic::AtomicU32,
}

impl MarkerRig {
    pub const fn new() -> MarkerRig {
        MarkerRig { gpu: None, planned_n: std::sync::atomic::AtomicU32::new(0) }
    }

    pub fn is_ready(&self) -> bool {
        self.gpu.is_some()
    }

    /// Instances the current plan drew (0 when idle / not yet updated).
    pub fn planned_n(&self) -> u32 {
        self.planned_n.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// One-time pipeline + buffers (idempotent). Same target formats as the
    /// morph pass.
    pub fn ensure(&mut self, device: &wgpu::Device, color_format: wgpu::TextureFormat, depth_format: wgpu::TextureFormat) {
        if self.gpu.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view3d.markers.bgl.cam"),
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
            label: Some("view3d.markers.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.markers.vs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MARKER_VS)),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view3d.markers.fs"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MARKER_FS)),
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.markers.cam"),
            size: std::mem::size_of::<crate::scene::CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view3d.markers.cam.bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view3d.markers.quad"),
            contents: bytemuck::cast_slice(&QUAD_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.markers.insts"),
            size: (MARKER_INST_CAP * MARKER_INST_STRIDE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("view3d.markers.pline"),
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
                            array_stride: MARKER_INST_STRIDE as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    offset: 0,
                                    shader_location: 1,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    offset: 12,
                                    shader_location: 2,
                                    format: wgpu::VertexFormat::Float32,
                                },
                                wgpu::VertexAttribute {
                                    offset: 16,
                                    shader_location: 3,
                                    format: wgpu::VertexFormat::Float32,
                                },
                                wgpu::VertexAttribute {
                                    offset: 20,
                                    shader_location: 4,
                                    format: wgpu::VertexFormat::Float32,
                                },
                                wgpu::VertexAttribute {
                                    offset: 24,
                                    shader_location: 5,
                                    format: wgpu::VertexFormat::Float32,
                                },
                                wgpu::VertexAttribute {
                                    offset: 28,
                                    shader_location: 6,
                                    format: wgpu::VertexFormat::Float32,
                                },
                            ],
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
                    cull_mode: None, // billboard quads face the camera always
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    // matching the morph pass: forward-Z, translucent, never
                    // occludes geometry
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        self.gpu = Some(MarkerGpu { pipeline, cam_buf, cam_bg, verts_buf, inst_buf });
    }

    /// Per-frame: consume the pending selection + current orgs snapshot and
    /// upload the (<= 3) marker instances + a fresh camera block. No heap
    /// allocation (fixed-size instance array).
    pub fn update(&mut self, queue: &wgpu::Queue, frame: &crate::scene::Frame3D) {
        use std::sync::atomic::Ordering;
        let Some(gpu) = self.gpu.as_ref() else { return };
        let order = Ordering::Relaxed;
        queue.write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&frame.cam));
        if frame.orgs.is_empty() {
            self.planned_n.store(0, order);
            return;
        }
        let sel = current();
        let world = frame.world.as_ref();
        let mut height = |x: f32, y: f32| crate::terrain::terrain_height_at(world, x, y);
        let (insts, n) = plan_markers(
            &frame.orgs,
            [frame.cam.view_pos[0], frame.cam.view_pos[1], frame.cam.view_pos[2]],
            &mut height,
            &sel,
        );
        self.planned_n.store(n as u32, order);
        if n > 0 {
            queue.write_buffer(&gpu.inst_buf, 0, bytemuck::bytes_of(&insts));
        }
    }

    /// ONE draw call (0..6 quad verts, 0..n instanced) — skipped entirely
    /// when no marker is armed, keeping idle frames draw-free.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let n = self.planned_n().min(MARKER_INST_CAP as u32);
        if n == 0 {
            return;
        }
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.cam_bg, &[]);
        pass.set_vertex_buffer(0, gpu.verts_buf.slice(..));
        pass.set_vertex_buffer(
            1,
            gpu.inst_buf.slice(0..(n as u64) * (MARKER_INST_STRIDE as u64)),
        );
        pass.draw(0..6, 0..n);
    }
}

const MARKER_VS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct VIn {
    @location(0) corner: vec2<f32>,
    @location(1) ipos: vec3<f32>,
    @location(2) iradius: f32,
    @location(3) istyle: f32,
    @location(4) ihue: f32,
    @location(5) iclose_fade: f32,
    @location(6) ipad: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) misc: vec4<f32>, // (style, hue, close_fade, pad)
};

@vertex
fn vs(i: VIn) -> VOut {
    let r = max(i.iradius, 0.02);
    // camera-facing expansion (same basis trick as the morph billboards)
    let va = vec3<f32>(cam.view[0][0], cam.view[1][0], cam.view[2][0]);
    let vu = vec3<f32>(cam.view[0][1], cam.view[1][1], cam.view[2][1]);
    let local = va * (i.corner.x * r) + vu * (i.corner.y * r);
    var o: VOut;
    o.clip = cam.proj * cam.view * vec4(i.ipos + local, 1.0);
    o.uv = i.corner;
    o.misc = vec4<f32>(i.istyle, i.ihue, i.iclose_fade, 0.0);
    return o;
}
"#;

const MARKER_FS: &str = r#"
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    world_wh_time_flash: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;

struct FsIn {
    @location(0) uv: vec2<f32>,
    @location(1) misc: vec4<f32>, // (style, hue, close_fade, pad)
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

const TAU: f32 = 6.28318530718;

@fragment
fn fs(fi: FsIn) -> @location(0) vec4<f32> {
    let r = clamp(length(fi.uv), 0.0, 1.0);
    let style = fi.misc.x;
    let hue = fract(fi.misc.y);
    let close_fade = clamp(fi.misc.z, 0.0, 1.0);
    let t = cam.world_wh_time_flash.z;
    var col = vec3(1.0);
    var a = 0.0;
    var c = 0.8;
    var wid = 0.055;
    if (style < 0.5) {
        // selected: solid species-tint ring (bright, unmistakable)
        col = mix(hsv(hue, 0.65, 0.9), vec3(1.0), 0.3);
        a = 0.95;
    } else if (style < 1.5) {
        // hovered: thinner, dashed, slowly rotating ring
        wid = 0.032;
        col = hsv(hue, 0.3, 1.0);
        let ang = atan2(fi.uv.y, fi.uv.x) / TAU * 12.0 + t * 0.4;
        a = 0.6 * step(0.3, fract(ang));
    } else {
        // queen: gold double ring (inner band + thin outer halo)
        col = vec3(1.0, 0.8, 0.3);
        a = 0.95;
        let g = 1.0 - smoothstep(0.02, 0.05, abs(r - 0.92));
        a = max(a, g * 0.75);
    }
    let band = 1.0 - smoothstep(wid * 0.55, wid, abs(r - c));
    let alpha = band * a * close_fade;
    return vec4(col, alpha);
}
"#;

// ------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    fn org(x: f32, y: f32, size: f32, alive: f32, queen: bool) -> OrgInst {
        let mut o = OrgInst::zeroed();
        o.x = x;
        o.y = y;
        o.size = size;
        o.energy = 0.8;
        o.alive = alive;
        o.is_queen = if queen { 1 } else { 0 };
        o.visual_radius_mul = if queen { crate::morph::decomp::QUEEN_SCALE } else { 1.0 };
        o.hue = 0.3;
        o
    }

    fn flat_h(_x: f32, _y: f32) -> f32 {
        0.5
    }

    fn constraints(sel: Option<u32>, hov: Option<u32>, queen: Option<u32>) -> SelectionMarkers {
        SelectionMarkers { selected: sel, hovered: hov, queen_id: queen }
    }

    fn all_finite(insts: &[MarkerInst; MAX_MARKERS], n: usize) -> bool {
        insts[..n].iter().all(|m| {
            m.pos.iter().all(|v| v.is_finite())
                && m.radius.is_finite()
                && m.style.is_finite()
                && m.hue.is_finite()
                && m.close_fade.is_finite()
        })
    }

    fn view_of(x: f32, y: f32, z: f32) -> [f32; 3] {
        [x, y, z]
    }

    #[test]
    fn markers_never_nan_on_degenerate_fields() {
        // NaN positions, NaN sizes, out-of-range indices, empty orgs, NaN cam
        let mut bad = org(f32::NAN, 0.0, 1.0, 1.0, true);
        bad.size = f32::NAN;
        let ok = org(10.0, 10.0, 1.0, 1.0, false);
        let orgs = [bad, ok];
        let (insts, n) = plan_markers(
            &orgs,
            [f32::NAN, 20.0, 0.0],
            &mut flat_h,
            &constraints(Some(0), Some(1), None),
        );
        assert!(all_finite(&insts, n));
        // the NaN selected org was skipped entirely, so only the healthy
        // hovered org survives as the first (and only) drawn marker
        assert_eq!(n, 1, "NaN org skipped, healthy org marker still drawn");
        assert_eq!(insts[0].style, MARKER_HOVERED as f32);
        // out-of-range + empty inputs always degenerate to nothing
        let (_, n2) = plan_markers(&[], view_of(0.0, 40.0, 0.0), &mut flat_h, &constraints(Some(0), None, Some(1)));
        assert_eq!(n2, 0);
        let (insts3, n3) = plan_markers(
            &[ok],
            view_of(0.0, 40.0, 0.0),
            &mut flat_h,
            &constraints(Some(7), Some(9), Some(3)),
        );
        assert!(all_finite(&insts3, n3), "OOD indices skipped sans NaN");
    }

    #[test]
    fn marker_respects_zoom_without_transition_jumps() {
        let orgs = [org(0.0, 0.0, 1.4, 1.0, false)];
        // walk the camera along a straight, continuous path (as a smooth
        // orbit->follow->planet transition is): radius must stay finite,
        // grow monotonically with distance and never step for any camera
        // MODE — plan_markers takes no camera mode at all by construction.
        let mut prev_r: Option<f32> = None;
        // fine 10-unit steps keep the Lipschitz bound meaningful: a mode
        // branch would be a step orders of magnitude larger than the smooth
        // 0.22*ln/180 compensation can produce per 10 units
        for k in 0..=88u32 {
            let dstep = 20.0 + 10.0 * k as f32;
            if dstep > 900.0 {
                break;
            }
            // step the camera along +Y at the org's ground band height
            let cam = view_of(0.0, 0.5 + 1.4 * 0.62 + dstep, 0.0);
            let (insts, n) = plan_markers(
                &orgs,
                cam,
                &mut flat_h,
                &constraints(Some(0), None, None),
            );
            assert_eq!(n, 1);
            assert!(all_finite(&insts, n));
            let r = insts[0].radius;
            if let Some(p) = prev_r {
                assert!(r >= p - 1e-4, "radius monotone non-decreasing: {p} -> {r}");
                // Lipschitz-bound the change over each continuous camera step
                assert!(
                    (r - p) <= 0.02 * (p + r) * 0.5 + 1e-3,
                    "no zoom jump between consecutive camera positions"
                );
            }
            prev_r = Some(r);
        }
        let far = prev_r.unwrap();
        assert!(
            far < 80.0 && far > 1.0,
            "radius walks between the hard bounds, got {far}"
        );
        // legibility after the ORGANISM_UNIT_RADIUS rebase: the walk floor is
        // the marker_radius clamp of 1.0 (never below), not the old 2.0 that
        // the pre-rebase `5.5`-based bodies produced.
    }

    #[test]
    fn queen_marker_gold_style_and_widened_circle() {
        let orgs = [org(0.0, 0.0, 1.4, 1.0, true), org(10.0, 10.0, 1.4, 1.0, false)];
        let (insts, n) = plan_markers(
            &orgs,
            view_of(30.0, 40.0, 30.0),
            &mut flat_h,
            &constraints(None, None, None), // queen found via the flag
        );
        assert_eq!(n, 1, "unflagged plan must mark the flag-scanned queen only");
        assert!(all_finite(&insts, n));
        assert_eq!(insts[0].style, MARKER_QUEEN as f32);
        // queen ring mirrors the giant queen radius chain exactly: her
        // visible body radius is QUEEN_UNIT_RADIUS * size * turgor (decoupled
        // from the ordinary ORGANISM_UNIT_RADIUS rebase). The queen:ordinary
        // RADIUS ratio is therefore QUEEN_UNIT_RADIUS / ORGANISM_UNIT_RADIUS
        // (~= 3.25, ~16-unit queen vs ~1.4-unit ordinary bodies ~1/12 over
        // length), not the pre-rebase shared-base QUEEN_SCALE factor.
        let qb = visual_body_radius(&orgs[0]);
        let ob = visual_body_radius(&orgs[1]);
        let ratio = crate::morph::decomp::QUEEN_UNIT_RADIUS / crate::morph::decomp::ORGANISM_UNIT_RADIUS;
        assert!(
            (qb - ob * ratio).abs() < 1e-4,
            "queen visual radius mirrors the giant/ordinary split"
        );
        // and re-derives straight from the queen chain
        assert!(
            (qb - crate::morph::decomp::QUEEN_UNIT_RADIUS * 1.4 * (0.62 + 0.38 * 0.8)).abs() < 1e-4,
            "queen ring = QUEEN_UNIT_RADIUS * size * turgor"
        );
        // radius matches marker_radius at the planner's actual camera distance
        let D: f32 = 30.0;
        let cy = flat_h(0.0, 0.0) + qb * 0.62;
        let d = (D * D + (40.0 - cy).powi(2) + D * D).sqrt();
        assert_eq!(insts[0].radius, marker_radius(qb, MARKER_QUEEN, d));
        // explicit queen_id wins over the flag scan
        let (insts2, n2) = plan_markers(
            &orgs,
            view_of(30.0, 40.0, 30.0),
            &mut flat_h,
            &constraints(None, None, Some(0)),
        );
        assert_eq!(n2, 1);
        assert_eq!(insts2[0].pos, insts[0].pos);
    }

    #[test]
    fn selected_and_hovered_styles_are_distinct() {
        let orgs = [org(0.0, 0.0, 1.0, 1.0, false)];
        let base = view_of(0.0, 60.0, 0.0);
        let (sel_i, sn) = plan_markers(&orgs, base, &mut flat_h, &constraints(Some(0), None, None));
        let (hov_i, hn) = plan_markers(&orgs, base, &mut flat_h, &constraints(None, Some(0), None));
        assert!(all_finite(&sel_i, sn) && all_finite(&hov_i, hn));
        assert_eq!(sn, 1);
        assert_eq!(hn, 1);
        assert_eq!(sel_i[0].style, MARKER_SELECTED as f32);
        assert_eq!(hov_i[0].style, MARKER_HOVERED as f32);
        assert!(
            sel_i[0].radius < hov_i[0].radius,
            "hovered sits just outside the selected band so both can coexist"
        );
        // both together + queen fallback still capped at MAX_MARKERS
        let (both, bn) = plan_markers(
            &[orgs[0], org(600.0, 0.0, 2.0, 1.0, true)],
            base,
            &mut flat_h,
            &constraints(Some(0), Some(0), None),
        );
        assert!(bn <= MAX_MARKERS);
        assert!(all_finite(&both, bn));
    }

    #[test]
    fn corpses_are_never_marked() {
        let orgs = [org(0.0, 0.0, 1.0, 0.4, false), org(5.0, 0.0, 1.0, 1.0, false)];
        let (insts, n) = plan_markers(
            &orgs,
            view_of(0.0, 60.0, 0.0),
            &mut flat_h,
            &constraints(Some(0), Some(1), None),
        );
        assert!(all_finite(&insts, n));
        assert_eq!(n, 1, "dying org skipped, the alive hovered one marked");
        assert_eq!(insts[0].style, MARKER_HOVERED as f32);
    }

    #[test]
    fn selection_state_roundtrip_is_poison_safe() {
        set_selection(Some(2), Some(3), Some(4));
        let s = current();
        assert_eq!(s.selected, Some(2));
        assert_eq!(s.hovered, Some(3));
        assert_eq!(s.queen_id, Some(4));
        set_selection(None, None, None);
        assert_eq!(current(), SelectionMarkers::default());
    }
}
