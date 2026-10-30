//! Tests for the chunked terrain engine (owned by Agent A).
//!
//! Pure CPU tests: plan ranking/LOD table, mesh determinism + byte identity,
//! wrap detection, bilinear height agreement, color helpers, frustum AABB
//! math. GPU tests: `ensure_mesh` determinism (identical vertex bytes across
//! two full rebuilds), rebuild budget on small focus moves, vertex budget,
//! LRU cap + far-move eviction, dirty-region scope, WGSL smoke. GPU tests
//! follow the established grass/morph pattern: silently skipped when no
//! adapter exists (or when `TERRAIN_SKIP_GPU` is set). Vertex bytes are
//! compared via the `#[cfg(test)]` FNV tap in terrain.rs (`TEST_VERT_TAP`).

use super::*;
use super::plan::PlanChunk;

// ------------------------------------------------------------ shared helpers

fn small_cfg() -> alife_worldgen::WorldGenConfig {
    alife_worldgen::WorldGenConfig {
        seed: 4242,
        width: 160,
        height: 120,
        cell: 2,
        terrain_scale: 1.0,
        detail: 4,
        warp: 0.3,
        moisture_scale: 0.8,
        resource_density: 1.0,
        hazard_amount: 0.3,
        fertility_contrast: 1.0,
        hazard_pulse_period: 0.0,
    }
}

/// Crafted world with fully controlled fields (grid 16x16 stretched to
/// whatever world rect the caller wants).
fn synthetic(width: f32, height: f32, cell: f32, elev: f32) -> WorldModel {
    let g = |v: f32| alife_worldgen::FieldGrid::new(16, 16, v);
    WorldModel {
        width,
        height,
        cell,
        elevation: g(elev),
        moisture: g(0.5),
        temperature: g(0.5),
        fertility: g(0.8),
        hazard: g(0.0),
        biome: g(0.0),
        generation_tag: 9,
        hazard_pulse_period: 0.0,
    }
}

/// Big sparse-grid world: 38x25 = 950 chunk blocks, so the 128-chunk cap
/// and the far-move eviction path actually bite.
fn big_synthetic() -> WorldModel {
    synthetic(2400.0, 1600.0, 4.0, 0.62)
}

fn plan_chunk(bx: i32, by: i32, lod: u32) -> PlanChunk {
    PlanChunk { agg: 1, bx, by, lod, dist: 0 }
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    if std::env::var("TERRAIN_SKIP_GPU").is_ok() {
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

fn orbit(f: (f32, f32)) -> CameraMode {
    CameraMode::Orbit { focus: f }
}

fn fmt_bytes(v: &[LandVert]) -> &[u8] {
    bytemuck::cast_slice::<LandVert, u8>(v)
}

// ======================================================== pure: plan module

#[test]
fn terrain_plan_lod_table_monotone_and_bounds() {
    assert_eq!(plan::lod_of(0), 1);
    assert_eq!(plan::lod_of(2), 1);
    assert_eq!(plan::lod_of(3), 2);
    assert_eq!(plan::lod_of(4), 2);
    assert_eq!(plan::lod_of(5), 4);
    assert_eq!(plan::lod_of(7), 4);
    assert_eq!(plan::lod_of(8), 8);
    assert_eq!(plan::lod_of(11), 8);
    assert_eq!(plan::lod_of(12), 16);
    assert_eq!(plan::lod_of(99), 16);
    // steps never grow with LOD and collapse to a 1-quad plate at lod 0
    for cell in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
        assert_eq!(plan::grid_steps(cell, 1, 0), 1, "lod 0 is a single plate");
        let mut prev = u32::MAX;
        for lod in 1..=16u32 {
            let n = plan::grid_steps(cell, 1, lod);
            assert!(n <= prev, "steps must not grow with LOD");
            assert!(n <= 24, "steps clamped to the near budget: {}", n);
            prev = n;
        }
        // verts round-trip the non-indexed quad math (6 verts per quad)
        for lod in 0..=16u32 {
            let s = plan::grid_steps(cell, 1, lod);
            assert_eq!(plan::verts_of(cell, 1, lod), 6 * s * s);
        }
        let fs = plan::full_steps(cell);
        assert!((4..=24).contains(&fs), "full_steps clamped to 4..24");
    }
    assert_eq!(plan::blocks_along(2400.0, 1), 38);
    assert_eq!(plan::blocks_along(1600.0, 1), 25);
    assert_eq!(plan::blocks_along(160.0, 1), 3);
    assert_eq!(plan::blocks_along(120.0, 1), 2);
}

#[test]
fn terrain_plan_near_first_order_deterministic() {
    let cfg = small_cfg();
    let world = alife_worldgen::generate(&cfg);
    let wrap = mesh::edge_wrap(&world);
    let mut a: Vec<PlanChunk> = Vec::new();
    let mut b: Vec<PlanChunk> = Vec::new();
    plan::build_plan(&world, 1, (1, 1), wrap, &mut a);
    plan::build_plan(&world, 1, (1, 1), wrap, &mut b);
    assert_eq!(a, b, "same inputs -> identical plans");
    // covers every block exactly once, ranked near-first, anchor first
    let n_blocks =
        (plan::blocks_along(world.width, 1) * plan::blocks_along(world.height, 1)) as usize;
    assert_eq!(a.len(), n_blocks);
    assert_eq!(a[0].dist, 0, "anchor block must rank first");
    for w in a.windows(2) {
        assert!(w[0].dist <= w[1].dist, "plan must be ranked near-first");
    }
    // torus metric kicks in only when the axis is continuous
    assert_eq!(plan::axis_dist(0, 36, 38, false), 36);
    assert_eq!(plan::axis_dist(0, 36, 38, true), 2);
    assert_eq!(plan::axis_dist(0, 0, 38, true), 0);
}

#[test]
fn terrain_plan_aggregation_and_block_rect_in_world() {
    let world = synthetic(160.0, 120.0, 2.0, 0.6);
    assert_eq!(plan::agg_for_mode(&orbit((40.0, 40.0)), &world), 1);
    assert_eq!(plan::agg_for_planet(160.0, 120.0), 1);
    assert_eq!(plan::agg_for_planet(2400.0, 1600.0), 4);
    let r = plan::block_rect(&world, 1, 1, 1);
    assert_eq!(r, (64.0, 64.0, 128.0, 120.0), "rect clamped to the world");
    assert_eq!(plan::block_rect(&world, 1, 0, 0), (0.0, 0.0, 64.0, 64.0));
    let (x0, y0, x1, y1) = plan::block_rect(&world, 1, 50, 50);
    assert_eq!((x1, y1), (x0, y0), "out-of-world rect degenerates, not negative");
    assert_eq!(plan::mode_tag(&CameraMode::Planet), 2);
    assert_eq!(plan::mode_tag(&CameraMode::Follow { id: 7 }), 1);
    assert_eq!(plan::mode_tag(&orbit((0.0, 0.0))), 0);

    // mesh on an out-of-world block is empty, never negative-sized
    let far = mesh::GeoParams::from_plan(&world, plan_chunk(50, 50, 16), (false, false));
    assert!(mesh::build(&world, &far).verts.is_empty(), "degenerate rect -> empty geo");
}

// ========================================================= pure: mesh module

#[test]
fn terrain_mesh_byte_determinism_and_lod_geometry() {
    // crafted flat world: heights/normals must be exact and byte-stable
    let flat = synthetic(128.0, 128.0, 1.0, 0.62);
    let wrap = mesh::edge_wrap(&flat);
    let p0 = mesh::GeoParams::from_plan(&flat, plan_chunk(0, 0, 1), wrap);
    let a = mesh::build(&flat, &p0);
    let b = mesh::build(&flat, &p0);
    assert_eq!(a.verts.len(), b.verts.len());
    assert_eq!(
        fmt_bytes(&a.verts),
        fmt_bytes(&b.verts),
        "equal inputs must produce identical vertex bytes"
    );
    let want_y = 0.62f32 * HEIGHT_SCALE;
    for v in &a.verts {
        assert_eq!(v.pos[1], want_y, "flat world -> exact constant height");
        assert_eq!(v.normal, [0.0, 1.0, 0.0], "flat world -> up normals");
    }
    assert!((a.aabb_min[0] - 0.0).abs() < 1e-4 && (a.aabb_max[0] - 64.0).abs() < 1e-4);
    assert!((a.aabb_min[2] - 0.0).abs() < 1e-4 && (a.aabb_max[2] - 64.0).abs() < 1e-4);

    // real generated small world: near chunk deterministic + matches the plan
    let world = alife_worldgen::generate(&small_cfg());
    let wrap = mesh::edge_wrap(&world);
    let mut pl = Vec::new();
    plan::build_plan(&world, 1, (1, 1), wrap, &mut pl);
    let near = *pl.iter().find(|c| c.lod == 1).expect("near ring lod1 exists");
    let gp1 = mesh::GeoParams::from_plan(&world, near, wrap);
    assert_eq!(gp1.steps, plan::grid_steps(world.cell, 1, near.lod));
    let c1a = mesh::build(&world, &gp1);
    let c1b = mesh::build(&world, &gp1);
    assert_eq!(fmt_bytes(&c1a.verts), fmt_bytes(&c1b.verts));
    assert_eq!(
        c1a.verts.len() as u32,
        plan::verts_of(world.cell, 1, near.lod),
        "built chunk vertex count matches plan::verts_of"
    );

    // big world: near vs far rings must use genuinely different detail
    let big = big_synthetic();
    let mut plb = Vec::new();
    plan::build_plan(&big, 1, (0, 0), (true, true), &mut plb);
    let near_b = *plb.iter().find(|c| c.lod == 1).expect("ring lod1");
    let far_b = *plb.iter().find(|c| c.lod == 16).expect("far ring plate");
    let bn = mesh::build(&big, &mesh::GeoParams::from_plan(&big, near_b, (true, true)));
    let bf = mesh::build(&big, &mesh::GeoParams::from_plan(&big, far_b, (true, true)));
    assert_eq!(bn.verts.len() as u32, plan::verts_of(big.cell, 1, 1));
    assert_eq!(bf.verts.len(), 6, "lod16 plate = single corner quad, 6 verts");
    assert!(bn.verts.len() > bf.verts.len());
}

#[test]
fn terrain_mesh_wrap_detected_only_on_periodic_fields() {
    // constant fields are periodic by the probe metric -> wrap on
    let flat = synthetic(128.0, 128.0, 1.0, 0.6);
    assert_eq!(mesh::edge_wrap(&flat), (true, true));
    // ramp along x: left/right edges differ -> no wrap on x, still on y
    let mut ramp = synthetic(128.0, 128.0, 1.0, 0.5);
    ramp.generation_tag = 3;
    for y in 0..16u32 {
        for x in 0..16u32 {
            ramp.elevation.data[(y * 16 + x) as usize] = x as f32 / 15.0;
        }
    }
    let wrap = mesh::edge_wrap(&ramp);
    assert!(!wrap.0, "x-ramp must not claim x-continuity");
    assert!(wrap.1, "y edges of the ramp still agree");
    // tiny worlds fall back to clamped behaviour on both axes
    ramp.width = 1.0;
    ramp.height = 1.0;
    assert_eq!(mesh::edge_wrap(&ramp), (false, false));
    // wrap_span is the identity/rem_euclid pair it claims
    assert_eq!(mesh::wrap_span(-1.0, 64.0, true), 63.0);
    assert_eq!(mesh::wrap_span(130.0, 64.0, true), 2.0);
    assert_eq!(mesh::wrap_span(-1.0, 64.0, false), 0.0, "clamped axis");
    assert_eq!(mesh::wrap_span(5.0, 0.0, true), 0.0, "degenerate span");
}

// ================================================= pure: height convention

#[test]
fn terrain_height_matches_grid_bilinear_exactly() {
    // mirror of FieldGrid::sample in raw grid math -> must agree bit-for-bit
    let mut w = synthetic(160.0, 120.0, 2.0, 0.5);
    for y in 0..16u32 {
        for x in 0..16u32 {
            w.elevation.data[(y * 16 + x) as usize] =
                0.15 + (x as f32 * 0.037 + y as f32 * 0.011).fract();
        }
    }
    let manual = |x: f32, y: f32| -> f32 {
        let g = &w.elevation;
        let gx = (x / w.width * g.w as f32 - 0.5).clamp(0.0, g.w as f32 - 0.001);
        let gy = (y / w.height * g.h as f32 - 0.5).clamp(0.0, g.h as f32 - 0.001);
        let x0 = gx as u32;
        let y0 = gy as u32;
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let at = |cx: u32, cy: u32| g.data[(cy * g.w + cx) as usize];
        let x1 = (x0 + 1).min(g.w - 1);
        let y1 = (y0 + 1).min(g.h - 1);
        let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        (a * (1.0 - fy) + b * fy) * HEIGHT_SCALE
    };
    for (x, y) in [
        (0.0, 0.0),
        (5.0, 3.0),
        (79.5, 61.25),
        (160.0, 120.0),
        (1.0, 119.0),
        (88.888, 44.444),
        (120.0, 60.0),
    ] {
        assert_eq!(
            terrain_height_at(&w, x, y),
            manual(x, y),
            "terrain_height_at must be HEIGHT_SCALE * bilinear grid sample at ({x},{y})"
        );
    }
    // and on a real generated world too (same bilinear contract)
    let world = alife_worldgen::generate(&small_cfg());
    let manual2 = |x: f32, y: f32| -> f32 {
        let g = &world.elevation;
        let gx = (x / world.width * g.w as f32 - 0.5).clamp(0.0, g.w as f32 - 0.001);
        let gy = (y / world.height * g.h as f32 - 0.5).clamp(0.0, g.h as f32 - 0.001);
        let x0 = gx as u32;
        let y0 = gy as u32;
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let at = |cx: u32, cy: u32| g.data[(cy * g.w + cx) as usize];
        let x1 = (x0 + 1).min(g.w - 1);
        let y1 = (y0 + 1).min(g.h - 1);
        let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        (a * (1.0 - fy) + b * fy) * HEIGHT_SCALE
    };
    for x in [0.0f32, 40.25, 91.5, 160.0] {
        for y in [0.0f32, 23.75, 89.5, 120.0] {
            assert_eq!(terrain_height_at(&world, x, y), manual2(x, y));
        }
    }
}

#[test]
fn terrain_height_monotone_in_elevation_field() {
    // ramp 0..1 along x -> heights non-decreasing along x for fixed y
    let mut w = synthetic(160.0, 120.0, 1.0, 0.5);
    w.generation_tag = 5;
    for y in 0..16u32 {
        for x in 0..16u32 {
            w.elevation.data[(y * 16 + x) as usize] = x as f32 / 15.0;
        }
    }
    for y in [10.0f32, 60.0, 110.0] {
        let mut prev_h = -1.0f32;
        for i in 0..=160 {
            let h = terrain_height_at(&w, i as f32, y);
            assert!(
                h >= prev_h,
                "height must not decrease along the elevation ramp: {} < {}",
                h,
                prev_h
            );
            assert_eq!(h, terrain_height_at(&w, i as f32, y), "repeat-call identity");
            prev_h = h;
        }
        assert!(prev_h > 0.7 * HEIGHT_SCALE, "ramp reaches its top band at the far edge");
    }
    // scale/sea helpers agree with the height convention
    assert_eq!(plan::sea_world_y(), SEA_LEVEL * HEIGHT_SCALE);
    assert_eq!(plan::norm_height(0.0), 0.0);
    assert_eq!(plan::norm_height(1.0), HEIGHT_SCALE);
    assert!((plan::sea_world_y() - 0.42 * 46.0).abs() < 1e-4);
}

#[test]
fn terrain_color_helpers_stay_in_range() {
    assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), [1.0, 0.0, 0.0]);
    assert_eq!(hsv_to_rgb(1.0 / 3.0, 1.0, 1.0), [0.0, 1.0, 0.0]);
    let c = biome_color(0.0, 0.75, 0.6);
    for v in c {
        assert!((0.0..=1.0).contains(&v), "colors in [0,1]: {}", v);
    }
    // hazard pulls red up and green down; zero hazard keeps base untouched
    let base = [0.11, 0.40, 0.24];
    let hz = mesh::hazard_color(base, 1.0);
    assert!(hz[0] > base[0] && hz[1] < base[1], "hazard tint shifts to red");
    assert_eq!(mesh::hazard_color(base, 0.0), base);
}

// ==================================================== pure: frustum culling

#[test]
fn terrain_frustum_rejects_boxes_behind_the_camera() {
    // camera at origin looking down -z (right-handed, same setup as grass)
    let view =
        glam::Mat4::look_at_rh(glam::Vec3::ZERO, glam::Vec3::new(0.0, 0.0, -1.0), glam::Vec3::Y);
    let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_1_SQRT_2, 1.0, 0.1, 300.0);
    let f = cull::Frustum::from_cam(&view.to_cols_array_2d(), &proj.to_cols_array_2d());
    // chunk-syntax AABB: ground plates span x/z, y is up
    assert!(
        f.aabb_visible([-2.0, 0.0, -200.0], [2.0, 60.0, -40.0]),
        "narrow front chunk visible"
    );
    assert!(
        f.aabb_visible([-60.0, 0.0, -260.0], [60.0, 60.0, -60.0]),
        "wide chunk intersecting the frustum at depth stays drawn (p-clip)"
    );
    assert!(
        !f.aabb_visible([0.0, 0.0, 40.0], [64.0, 60.0, 104.0]),
        "behind-camera chunk must be rejected"
    );
    assert!(
        !f.aabb_visible([-400.0, 0.0, 900.0], [-200.0, 60.0, 1000.0]),
        "far behind chunk rejected"
    );
    assert!(
        !f.aabb_visible([300.0, 0.0, -50.0], [360.0, 60.0, 50.0]),
        "chunk fully outside the left/right clip is rejected"
    );
    // degenerate point box well inside the cone -> visible (no crash path)
    assert!(f.aabb_visible([0.1, 0.1, -3.0], [0.1, 0.1, -3.0]));
    // the engine-level wiring (set_cull + draw_ranges) is exercised by the
    // GPU matrix below
}

// =================================================== GPU: engine behaviour

const GPU_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// GPU tests in this crate may run in parallel (cargo default), but they all
/// upload through the single process-wide TEST_VERT_TAP, so every GPU test
/// takes this lock before its first ensure_mesh.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn terrain_gpu_ensure_mesh_determinism_identical_vertex_bytes() {
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(alife_worldgen::generate(&small_cfg()));
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).expect("terrain pipeline");

    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert!(t.used_verts() > 0, "anchor build uploads real geometry");
    // sibling GPU tests may have left uploads in the shared process-wide tap
    // before this critical section; measure only what THIS ensure uploads
    super::TEST_VERT_TAP.lock().unwrap().clear();
    t.invalidate();
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    let pass1 = super::TEST_VERT_TAP.lock().unwrap().clone();
    assert!(pass1.len() == t.resident_chunks() as usize, "tap == one full rebuild");

    // second FULL rebuild: same world, same focus; invalidate() zaps the
    // cache so every resident chunk re-runs the whole build+upload path
    super::TEST_VERT_TAP.lock().unwrap().clear();
    let used1 = t.used_verts();
    let residents1 = t.resident_chunks();
    t.invalidate();
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    let pass2 = super::TEST_VERT_TAP.lock().unwrap().clone();

    assert_eq!(
        pass1, pass2,
        "two full ensure_mesh rebuilds -> identical per-slot vertex bytes"
    );
    assert_eq!(t.used_verts(), used1, "used vertex total stable across rebuilds");
    assert_eq!(t.resident_chunks(), residents1);

    // once resident and not dirty, a cheap hit does no work at all
    super::TEST_VERT_TAP.lock().unwrap().clear();
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert_eq!(t.last_ensure_rebuilds(), 0, "warm hit must be free");
    assert_eq!(super::TEST_VERT_TAP.lock().unwrap().len(), 0);
}

#[test]
fn terrain_gpu_small_focus_moves_stay_within_rebuild_budget() {
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(alife_worldgen::generate(&small_cfg()));
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).unwrap();

    t.ensure_mesh(&device, &queue, &world, orbit((30.0, 30.0)), (30.0, 30.0));
    let built = t.total_rebuilds();
    // 24-unit drift inside one 64-unit chunk and 16 units back: below the
    // 2-chunk hysteresis ring -> structurally 0 rebuilds (budget <= ~2)
    t.ensure_mesh(&device, &queue, &world, orbit((54.0, 30.0)), (54.0, 30.0));
    assert_eq!(t.last_ensure_rebuilds(), 0, "24-unit move inside the anchor chunk rebuilds nothing");
    t.ensure_mesh(&device, &queue, &world, orbit((14.0, 30.0)), (14.0, 30.0));
    assert_eq!(t.last_ensure_rebuilds(), 0, "16-unit move back is also free");
    // crossing into an adjacent chunk (Chebyshev chunk drift 1) still sits
    // inside the hysteresis ring -> no re-anchor, no rebuild
    t.ensure_mesh(&device, &queue, &world, orbit((70.0, 30.0)), (70.0, 30.0));
    assert_eq!(t.last_ensure_rebuilds(), 0, "adjacent chunk drift stays in the hysteresis ring");
    assert_eq!(t.total_rebuilds(), built, "zero slot work happened overall");
}

#[test]
fn terrain_gpu_vertex_budget_and_ranges_within_pool() {
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(alife_worldgen::generate(&small_cfg()));
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).unwrap();

    // planet sweep covers every block once (6 blocks on this world)
    t.ensure_mesh(&device, &queue, &world, CameraMode::Planet, (80.0, 60.0));
    assert!(t.used_verts() > 0);
    assert!(
        (t.used_verts() as u64) <= super::MAX_VERTS as u64,
        "vertex budget enforced: {}",
        t.used_verts()
    );
    assert!(t.slot_cap() > 0);
    assert!(
        t.legacy_draw_count() >= t.used_verts(),
        "legacy draw range must cover the real geometry ({} >= {})",
        t.legacy_draw_count(),
        t.used_verts()
    );
    // without a frustum the ranges list every resident span, sorted,
    // disjoint and inside the persistent buffer
    let ranges: Vec<std::ops::Range<u32>> = t.draw_ranges().to_vec();
    assert!(!ranges.is_empty());
    let mut prev_end: u32 = 0;
    for r in &ranges {
        assert!(r.start >= prev_end, "ranges must be sorted and disjoint");
        assert!(r.end <= t.legacy_draw_count(), "range inside the pool buffer");
        assert!(r.end > r.start);
        prev_end = r.end;
    }
    let (buf, n) = t.vertex_buffer().expect("pool buffer live after ensure");
    assert_eq!(n, t.legacy_draw_count());
    assert!(buf.size() >= n as u64 * std::mem::size_of::<LandVert>() as u64);
}

#[test]
fn terrain_gpu_lru_cap_and_far_move_evicts_farthest() {
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(big_synthetic());
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).unwrap();

    // corner anchor on a 38x25 block world: the chunk cap saturates
    t.ensure_mesh(&device, &queue, &world, orbit((30.0, 30.0)), (30.0, 30.0));
    assert!(
        t.resident_chunks() <= super::MAX_CHUNKS,
        "resident cap enforced: {}",
        t.resident_chunks()
    );
    assert!(t.resident_chunks() > 0);
    assert!((t.used_verts() as u64) <= super::MAX_VERTS as u64);
    let evictions0 = t.total_evictions();

    // jump to the opposite corner: the plan swap must evict stale far
    // residents and refill the freed slots (farthest-first cache)
    t.ensure_mesh(&device, &queue, &world, orbit((2370.0, 1570.0)), (2370.0, 1570.0));
    assert!(
        t.total_evictions() > evictions0,
        "far move must evict stale residents"
    );
    assert!(
        t.resident_chunks() <= super::MAX_CHUNKS,
        "resident cap holds after eviction"
    );
    assert!((t.used_verts() as u64) <= super::MAX_VERTS as u64);
    // ranges stay inside the buffer bounds after the reshuffle
    for r in t.draw_ranges() {
        assert!(r.end <= t.legacy_draw_count());
    }
}

#[test]
fn terrain_gpu_dirty_region_rebuilds_only_touched_chunks() {
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(alife_worldgen::generate(&small_cfg()));
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).unwrap();

    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert_eq!(t.resident_chunks(), 6, "160x120 world = 3x2 chunks, all resident");
    let rebuilt0 = t.total_rebuilds();

    // rect strictly inside chunk (1,1) -> exactly one chunk re-uploads
    super::TEST_VERT_TAP.lock().unwrap().clear();
    t.invalidate_region(72.0, 72.0, 110.0, 110.0);
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert_eq!(t.last_ensure_rebuilds(), 1, "scoped rect hits exactly one chunk");
    assert_eq!(super::TEST_VERT_TAP.lock().unwrap().len(), 1);
    assert_eq!(t.total_rebuilds(), rebuilt0 + 1);
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert_eq!(t.last_ensure_rebuilds(), 0, "dirty state is consumed by one ensure");

    // full invalidation: every resident rebuilds (and bytes stay stable,
    // checked in detail by the determinism test)
    t.invalidate();
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert_eq!(t.last_ensure_rebuilds(), 6, "full invalidation rebuilds all residents");
}

#[test]
fn terrain_gpu_pipeline_smoke() {
    // WGSL validity smoke test; skipped silently when no GPU adapter exists
    // (e.g. CI sandboxes). wgpu panics inside create_shader_module on invalid
    // WGSL, so a pass means both terrain modules are valid.
    let Some((device, queue)) = gpu() else { return; };
    let _gpu = GPU_LOCK.lock().expect("gpu lock");
    let world: Arc<WorldModel> = Arc::new(alife_worldgen::generate(&small_cfg()));
    let mut t = Terrain::new(&device, GPU_FMT, DEPTH_FMT).expect("terrain pipeline");
    t.ensure_mesh(&device, &queue, &world, orbit((40.0, 40.0)), (40.0, 40.0));
    assert!(t.cam_buf().size() >= std::mem::size_of::<crate::scene::CamUniform>() as u64);
    let _ = t.pipeline();
    let _ = t.bind_group();
    assert!(t.vertex_buffer().is_some(), "pool live after first ensure");
    assert!(!t.draw_ranges().is_empty());
}
