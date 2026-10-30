//! 3D terrain from the real `WorldModel` fields (owned by Agent A).
//! Chunked ring-LOD engine:
//! - the world is tiled into 64-unit chunks (`CHUNK`), keyed by (chunk
//!   coords, generation tag, aggregation); the camera lives on an "anchor"
//!   chunk with hysteresis (re-anchor only when the focus drifts >= 2
//!   chunks away), so ordinary moves rebuild nothing;
//! - ring LOD by Chebyshev chunk distance: the near ring is full detail at
//!   ~`world.cell` spacing, mid rings 2x, then 4x/8x, far rings degrade to
//!   flat corner-quad plates; a 220k global vertex budget and a 128-chunk
//!   cap (evict-farthest-first, because the plan is ranked near-first) are
//!   enforced every `ensure_mesh`;
//! - vertices live in ONE persistent buffer with per-chunk slot
//!   sub-allocations (slot index * fixed vertex capacity = offset, real
//!   verts = len; slot slack is first-vertex-padded and freed slots are
//!   zeroed, so the legacy single-draw path `vertex_buffer()` + one draw
//!   never rasterises garbage);
//! - `set_cull` + `draw_ranges` add optional pure-math frustum culling
//!   (chunk AABB vs planes from view-proj) for the coordinator; until
//!   wired, `vertex_buffer()` keeps drawing everything;
//! - world edits rebuild ONLY touched chunks (`invalidate` /
//!   `invalidate_region`); generation_tag changes full-swap the cache.
//!
//! Draw API contract (unchanged): `new(device, color_format, depth_format)
//! -> Option<Terrain>`, `ensure_mesh(...)`, `cam_buf()`, `pipeline()`,
//! `bind_group()`, `vertex_buffer()`, `invalidate()`.

#[path = "terrain_plan.rs"]
pub mod plan;
#[path = "terrain_mesh.rs"]
pub mod mesh;
#[path = "terrain_cull.rs"]
pub mod cull;
#[path = "terrain_tests.rs"]
#[cfg(test)]
mod tests;

use crate::scene::{CameraMode, CamUniform};
use alife_worldgen::WorldModel;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::sync::Arc;

pub const SEA_LEVEL: f32 = 0.42;
/// world-units of vertical scale per unit of the [0,1] elevation field
pub const HEIGHT_SCALE: f32 = 46.0;
/// chunk side in world units
pub const CHUNK: f32 = 64.0;
/// hard cap of live chunks (ring cache); eviction is farthest-first
const MAX_CHUNKS: usize = 128;
/// global vertex budget (used vertices, real geometry)
const MAX_VERTS: u32 = 220_000;
/// rayon build pool cap
const BUILD_THREADS: usize = 4;

/// test-only: FNV-1a hash + len of the last real-vertex upload per slot,
/// appended by `upload_slot` so tests can compare rebuilds byte-for-byte
#[cfg(test)]
pub(super) static TEST_VERT_TAP: std::sync::Mutex<Vec<(u32, u32, u64)>> =
    std::sync::Mutex::new(Vec::new());

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LandVert {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// resident chunk record (slot sub-allocation inside the persistent buffer)
struct Resident {
    bx: i32,
    by: i32,
    agg: u32,
    tag: u32,
    lod: u32,
    slot: u32,
    used: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

/// persistent vertex buffer with per-chunk slot sub-allocations
struct Pool {
    buf: wgpu::Buffer,
    /// vertex capacity of one slot (full-detail block mesh size)
    cap: u32,
    /// allocated slot count; slot k occupies [k*cap, (k+1)*cap)
    slots: u32,
    /// bump high-water mark; freed slots are zero-padded before reuse
    hi: u32,
    free: Vec<u32>,
}

/// identity of the maintained plan (every input of the desired chunk set)
struct CacheKey {
    epoch: u64,
    anchor: (i32, i32),
    agg: u32,
    cell: f32,
    dims: (u32, u32),
    wrap: (bool, bool),
}

/// one geometry job (a resident rebuild or a fresh chunk entering)
struct BuildJob {
    slot: u32,
    bx: i32,
    by: i32,
    agg: u32,
    tag: u32,
    lod: u32,
    params: mesh::GeoParams,
}

pub struct Terrain {
    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    pool: Option<Pool>,
    res: Vec<Resident>,
    anchor: Option<(i32, i32)>,
    key: Option<CacheKey>,
    dirty_full: bool,
    dirty_rects: Vec<(f32, f32, f32, f32)>,
    last_regen: u32,
    total_rebuilds: u64,
    total_evictions: u64,
    last_frustum: Option<cull::Frustum>,
    ranges: Vec<std::ops::Range<u32>>,
    wrap: (bool, bool),
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

// lazy <=4-thread build pool (never spins up before the first slow path)
static BUILD_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

fn build_geos(world: &WorldModel, params: &[mesh::GeoParams]) -> Vec<mesh::ChunkGeo> {
    let pool = BUILD_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(BUILD_THREADS)
            .build()
            .expect("terrain build thread pool")
    });
    pool.install(|| params.par_iter().map(|p| mesh::build(world, p)).collect())
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
            pool: None,
            res: Vec::new(),
            anchor: None,
            key: None,
            dirty_full: false,
            dirty_rects: Vec::new(),
            last_regen: 0,
            total_rebuilds: 0,
            total_evictions: 0,
            last_frustum: None,
            ranges: Vec::new(),
            wrap: (false, false),
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

    /// Marks every chunk dirty: the next ensure_mesh rebuilds all resident
    /// chunks (world-wide edits).
    pub fn invalidate(&mut self) {
        self.dirty_full = true;
    }

    /// Scoped edit: only chunks whose AABB intersects the world rect
    /// rebuild on the next ensure_mesh.
    pub fn invalidate_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.dirty_rects.push((x0, y0, x1, y1));
    }

    /// Optional pure-math frustum culling: recompute per-chunk draw ranges
    /// from the camera block. The coordinator should call this once per
    /// frame with `frame.cam` (after ensure_mesh); until wired, everything
    /// stays drawn and `draw_ranges()` lists all resident spans.
    pub fn set_cull(&mut self, view: &[[f32; 4]; 4], proj: &[[f32; 4]; 4]) {
        self.last_frustum = Some(cull::Frustum::from_cam(view, proj));
        self.recompute_ranges();
    }

    /// Draw ranges (vertex index ranges within the persistent buffer) to
    /// draw this frame, merged where contiguous.
    pub fn draw_ranges(&self) -> &[std::ops::Range<u32>] {
        &self.ranges
    }

    // ------------------------------------------------- state getters

    pub fn resident_chunks(&self) -> usize {
        self.res.len()
    }

    /// number of chunks (re)generated during the most recent ensure_mesh
    /// that did any work; 0 on a cheap hit
    pub fn last_ensure_rebuilds(&self) -> u32 {
        self.last_regen
    }

    pub fn used_verts(&self) -> u32 {
        self.res.iter().map(|r| r.used).sum()
    }

    pub fn total_evictions(&self) -> u64 {
        self.total_evictions
    }

    pub fn total_rebuilds(&self) -> u64 {
        self.total_rebuilds
    }

    pub fn slot_cap(&self) -> u32 {
        self.pool.as_ref().map(|p| p.cap).unwrap_or(0)
    }

    /// legacy-safe draw count: every slot up to the bump high-water mark;
    /// dead slots are zeroed and slot slack is padded, so this range can
    /// never rasterise stale chunk geometry
    pub fn legacy_draw_count(&self) -> u32 {
        match &self.pool {
            Some(p) => p.hi * p.cap,
            None => 0,
        }
    }

    pub fn vertex_buffer(&self) -> Option<(&wgpu::Buffer, u32)> {
        let n = self.legacy_draw_count();
        self.pool.as_ref().filter(|_| n > 0).map(|p| (&p.buf, n))
    }

    // ------------------------------------------------------- internals

    fn aabb_wants_rebuild(&self, r: &Resident) -> bool {
        self.dirty_full
            || self.dirty_rects.iter().any(|&(x0, y0, x1, y1)| {
                r.aabb_max[0] >= x0
                    && r.aabb_min[0] <= x1
                    && r.aabb_max[2] >= y0
                    && r.aabb_min[2] <= y1
            })
    }

    fn plan_signature_changed(&self, k: &CacheKey) -> bool {
        match (&self.key, &self.pool) {
            (Some(old), Some(_)) => {
                old.epoch != k.epoch
                    || old.anchor != k.anchor
                    || old.agg != k.agg
                    || old.cell != k.cell
                    || old.dims != k.dims
                    || old.wrap != k.wrap
            }
            _ => true,
        }
    }

    fn recompute_ranges(&mut self) {
        self.ranges.clear();
        let cap = self.slot_cap();
        let mut spans: Vec<(u32, u32)> = self
            .res
            .iter()
            .filter(|r| match &self.last_frustum {
                None => true,
                Some(f) => f.aabb_visible(r.aabb_min, r.aabb_max),
            })
            .map(|r| (r.slot * cap, r.slot * cap + r.used))
            .collect();
        if spans.is_empty() {
            return;
        }
        spans.sort_unstable();
        let mut it = spans.into_iter().peekable();
        while let Some((s, e)) = it.next() {
            let mut cur = s..e;
            while let Some(&(s2, e2)) = it.peek() {
                if s2 == cur.end && e2 > cur.end {
                    cur.end = e2;
                    it.next();
                } else {
                    break;
                }
            }
            self.ranges.push(cur);
        }
    }

    fn create_pool(&mut self, device: &wgpu::Device, cap: u32) {
        let slots = MAX_CHUNKS as u32;
        let size = slots as u64 * cap as u64 * std::mem::size_of::<LandVert>() as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view3d.terrain.chunks"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.pool = Some(Pool { buf, cap, slots, hi: 0, free: Vec::new() });
    }

    fn alloc_slot(&mut self) -> Option<u32> {
        let pool = self.pool.as_mut()?;
        if let Some(s) = pool.free.pop() {
            return Some(s);
        }
        if pool.hi < pool.slots {
            let s = pool.hi;
            pool.hi += 1;
            return Some(s);
        }
        None
    }

    #[cfg(test)]
    fn tap_log(slot: u32, verts: &[LandVert]) {
        let bytes = bytemuck::cast_slice::<LandVert, u8>(verts);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        TEST_VERT_TAP
            .lock()
            .expect("terrain vert tap")
            .push((slot, verts.len() as u32, h));
    }

    /// one upload: real verts followed by first-vertex padding so a slot
    /// never retains bytes of a previous occupant
    fn upload_slot(&self, queue: &wgpu::Queue, slot: u32, verts: &[LandVert]) {
        #[cfg(test)]
        {
            Self::tap_log(slot, verts);
        }
        if let Some(pool) = &self.pool {
            let mut full = Vec::with_capacity(pool.cap as usize);
            full.extend_from_slice(verts);
            let first = verts[0];
            while full.len() < pool.cap as usize {
                full.push(first);
            }
            let byte_off = slot as u64 * pool.cap as u64 * std::mem::size_of::<LandVert>() as u64;
            queue.write_buffer(&pool.buf, byte_off, bytemuck::cast_slice(&full));
        }
    }

    /// zero a slot region (GPU mirror of an eviction; legacy draws never
    /// rasterise dead slots)
    fn zero_slot(&self, queue: &wgpu::Queue, slot: u32) {
        if let Some(pool) = &self.pool {
            let pad = LandVert {
                pos: [0.0; 3],
                normal: [0.0; 3],
                color: [0.0; 3],
            };
            let mut full = Vec::with_capacity(pool.cap as usize);
            while full.len() < pool.cap as usize {
                full.push(pad);
            }
            let byte_off = slot as u64 * pool.cap as u64 * std::mem::size_of::<LandVert>() as u64;
            queue.write_buffer(&pool.buf, byte_off, bytemuck::cast_slice(&full));
        }
    }

    /// Rebuilds / maintains the chunk set for THIS frame's camera. Needs
    /// device+queue for slot uploads; hits return before touching (or
    /// building) anything.
    pub fn ensure_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &Arc<WorldModel>,
        mode: CameraMode,
        focus: (f32, f32),
    ) {
        let agg = plan::agg_for_mode(&mode, world);
        let cell = world.cell;
        let nbx = plan::blocks_along(world.width, 1);
        let nby = plan::blocks_along(world.height, 1);
        let fcx = (((focus.0 / CHUNK).floor() as i32).clamp(0, nbx.max(1) as i32 - 1)).max(0);
        let fcy = (((focus.1 / CHUNK).floor() as i32).clamp(0, nby.max(1) as i32 - 1)).max(0);
        // hysteresis: re-anchor only when the focus drifts >= 2 chunks away
        let anchor = match self.anchor {
            Some(a) => {
                let d = (a.0 - fcx).abs().max(a.1 - fcy).abs();
                if d >= 2 {
                    (fcx, fcy)
                } else {
                    a
                }
            }
            None => (fcx, fcy),
        };
        self.anchor = Some(anchor);

        if !self.wrap.0 && !self.wrap.1 {
            self.wrap = mesh::edge_wrap(world);
        }

        let new_key = CacheKey {
            epoch: world.generation_tag as u64,
            anchor,
            agg,
            cell,
            dims: (nbx, nby),
            wrap: self.wrap,
        };
        let plan_changed = self.plan_signature_changed(&new_key);
        let dirty_any = self.dirty_full || !self.dirty_rects.is_empty();
        if !plan_changed && !dirty_any && self.pool.is_some() {
            self.last_regen = 0; // cheap hit: plan + cache stay valid
            return;
        }

        // ------------------------------- slow path: reconcile the cache
        let mut planned = Vec::with_capacity(256);
        plan::build_plan(world, agg, anchor, self.wrap, &mut planned);

        let cap = plan::verts_of(cell, agg, 1);
        let pool_mismatch = self.pool.as_ref().map(|p| p.cap != cap).unwrap_or(true);
        if pool_mismatch {
            self.res.clear();
            self.create_pool(device, cap);
        }

        let tag = world.generation_tag;
        let mut verts_used = self.res.iter().map(|r| r.used).sum::<u32>();

        // desired additions; slots are claimed after evictions (phase B)
        struct NewEntry {
            bx: i32,
            by: i32,
            lod: u32,
            params: mesh::GeoParams,
        }
        let mut new_entries: Vec<NewEntry> = Vec::new();
        let mut jobs: Vec<BuildJob> = Vec::new();

        // phase A: match planned chunks against residents
        let by_pos: HashMap<(u32, u32, i32, i32), usize> = self            .res
            .iter()
            .enumerate()
            .map(|(i, r)| ((r.tag, r.agg, r.bx, r.by), i))
            .collect();
        let mut visited = vec![false; self.res.len()];
        for c in planned.iter().take(MAX_CHUNKS) {
            match by_pos.get(&(tag, agg, c.bx, c.by)).copied() {
                Some(i) => {
                    visited[i] = true;
                    let r = &self.res[i];
                    if r.lod == c.lod && !self.aabb_wants_rebuild(r) {
                        continue;
                    }
                    let newv = plan::verts_of(cell, agg, c.lod);
                    verts_used = verts_used - r.used + newv;
                    let slot = self.res[i].slot;
                    jobs.push(BuildJob {
                        slot,
                        bx: c.bx,
                        by: c.by,
                        agg,
                        tag,
                        lod: c.lod,
                        params: mesh::GeoParams::from_plan(world, *c, self.wrap),
                    });
                }
                None => {
                    let v = plan::verts_of(cell, agg, c.lod);
                    let lod_eff = if verts_used + v <= MAX_VERTS {
                        c.lod
                    } else if verts_used + 6 <= MAX_VERTS {
                        0 // flat-plate fallback under vertex pressure
                    } else {
                        continue; // fog covers the horizon; skip entirely
                    };
                    verts_used += plan::verts_of(cell, agg, lod_eff);
                    new_entries.push(NewEntry {
                        bx: c.bx,
                        by: c.by,
                        lod: lod_eff,
                        params: mesh::GeoParams::from_plan(world, *c, self.wrap),
                    });
                }
            }
        }

        // phase B: evict residents the plan no longer wants (every
        // planned chunk is revisited early, so unvisited = farthest first)
        let mut freed: Vec<u32> = Vec::new();
        let mut kept: Vec<Resident> = Vec::with_capacity(self.res.len());
        for (i, r) in std::mem::take(&mut self.res).into_iter().enumerate() {
            if visited[i] {
                kept.push(r);
            } else {
                freed.push(r.slot);
                self.total_evictions += 1;
            }
        }
        self.res = kept;
        if let Some(pool) = self.pool.as_mut() {
            pool.free.extend(from_slot(freed.as_slice()));
        }

        // phase C: claim slots for the desired additions
        for e in new_entries {
            match self.alloc_slot() {
                Some(slot) => jobs.push(BuildJob {
                    slot,
                    bx: e.bx,
                    by: e.by,
                    agg,
                    tag,
                    lod: e.lod,
                    params: e.params,
                }),
                None => break, // cache full; the plan heals on the next move
            }
        }

        // phase D: build all geometry (rayon, <=4 threads) and upload
        let params: Vec<mesh::GeoParams> = jobs.iter().map(|j| j.params).collect();
        let geos = build_geos(world, &params);
        let regen = geos.len() as u32;
        for (job, geo) in jobs.into_iter().zip(geos) {
            self.upload_slot(queue, job.slot, &geo.verts);
            self.total_rebuilds += 1;
            let used = geo.verts.len() as u32;
            let find = self
                .res
                .iter()
                .position(|r| r.tag == job.tag && r.agg == job.agg && r.bx == job.bx && r.by == job.by);
            match find {
                Some(i) => {
                    let r = &mut self.res[i];
                    r.lod = job.lod;
                    r.used = used;
                    r.aabb_min = geo.aabb_min;
                    r.aabb_max = geo.aabb_max;
                }
                None => self.res.push(Resident {
                    bx: job.bx,
                    by: job.by,
                    agg: job.agg,
                    tag: job.tag,
                    lod: job.lod,
                    slot: job.slot,
                    used,
                    aabb_min: geo.aabb_min,
                    aabb_max: geo.aabb_max,
                }),
            }
        }

        // phase E: zero every slot that was freed but not re-used
        if let Some(pool) = self.pool.as_ref() {
            let stale: Vec<u32> = pool.free.clone();
            for slot in stale {
                self.zero_slot(queue, slot);
            }
        }

        // finalize
        self.res.sort_by_key(|r| r.slot); // deterministic resident order
        self.dirty_full = false;
        self.dirty_rects.clear();
        self.key = Some(new_key);
        self.last_regen = regen;
        self.recompute_ranges();
    }
}

fn from_slot(s: &[u32]) -> impl Iterator<Item = u32> + '_ {
    s.iter().copied()
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
    // underwater / shoreline tint where the height is close to/below sea
    // level; the tint is modulated by a slow wave over world position and
    // uniform time (world_wh_time_flash.z) so open water shimmers
    let hn = in.world.y / 46.0;
    let under = clamp((0.42 - hn) * 6.5, 0.0, 1.0);
    let wave = sin(in.world.x * 0.11 + cam.world_wh_time_flash.z * 1.9)
        * sin(in.world.z * 0.13 - cam.world_wh_time_flash.z * 1.4);
    c = mix(c, vec3(0.20, 0.48, 0.56), under * (0.70 + 0.15 * wave));
    let d = distance(in.world, cam.view_pos.xyz);
    let fog = clamp((d - 500.0) / 2600.0, 0.0, 0.62);
    let flash = clamp(cam.world_wh_time_flash.w, 0.0, 1.0);
    c = mix(c, vec3(0.55, 0.60, 0.66), fog);
    c += flash * 0.35;
    return vec4(c, 1.0);
}
"#;
