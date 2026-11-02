//! Chunk geometry generation for the chunked terrain (owned by Agent A).
//! Pure CPU: samples ONLY `WorldModel` fields (elevation / biome / moisture
//! / hazard) on the block grid and flattens to the shared non-indexed
//! `LandVert` stream. No GPU, no randomness; equal inputs -> identical
//! vertex bytes (determinism test).
//!
//! Circular-continuous coordinates: when the world's fields turn out to be
//! (nearly) periodic across a border, samples that fall beyond a wrapped
//! edge read the opposite side via `wrap_span` so heights/normals continue
//! smoothly instead of flattening on a clamped shelf. Edges that are NOT
//! periodic (probe mismatch) keep the clamped behaviour - wrap never
//! invents data the WorldModel does not imply.
//!
//! Vertex order per quad and shading inputs match the M1 single-window
//! baseline so the shared render pipeline stays byte-identical.

use super::plan;
use super::{biome_color, LandVert, HEIGHT_SCALE, SEA_LEVEL};
use alife_worldgen::{VirtualWorld, WorldModel, WorldSampling};

const EDGE_TOL: f32 = 0.06;
const EDGE_PROBES: usize = 5;

/// Reads a field coordinate with circular wrap where `wrap` says the axis
/// is periodic, plain clamp otherwise.
pub fn wrap_span(v: f32, span: f32, wrap: bool) -> f32 {
    if !span.is_finite() || span <= 0.0 {
        return 0.0;
    }
    if wrap {
        v.rem_euclid(span)
    } else {
        v.clamp(0.0, span.max(0.0))
    }
}

/// Detects per-axis border continuity by comparing elevation probes just
/// inside each edge. Returns (wrap_x, wrap_y).
pub fn edge_wrap(world: &WorldModel) -> (bool, bool) {
    let w = world.width.max(1.0);
    let h = world.height.max(1.0);
    let eps = 0.5f32.max(w * 0.004);
    let wrap_x = if w > 2.0 * eps {
        let mut gap = 0.0f32;
        for i in 0..EDGE_PROBES {
            let y = h * (0.1 * i as f32 + 0.05);
            let a = world.sample_elevation(eps, y);
            let b = world.sample_elevation(w - eps, y);
            gap = gap.max((a - b).abs());
        }
        gap <= EDGE_TOL
    } else {
        false
    };
    let wrap_y = if h > 2.0 * eps {
        let mut gap = 0.0f32;
        for i in 0..EDGE_PROBES {
            let x = w * (0.1 * i as f32 + 0.05);
            let a = world.sample_elevation(x, eps);
            let b = world.sample_elevation(x, h - eps);
            gap = gap.max((a - b).abs());
        }
        gap <= EDGE_TOL
    } else {
        false
    };
    (wrap_x, wrap_y)
}

/// Per-vertex colour: biome base + per-vertex hazard reddish tint.
pub fn hazard_color(base: [f32; 3], hazard: f32) -> [f32; 3] {
    let k = (hazard.clamp(0.0, 1.0) * 1.8 - 0.08).clamp(0.0, 1.0) * 0.8;
    [
        base[0] + (0.62 - base[0]) * k,
        base[1] + (0.16 - base[1]) * k,
        base[2] + (0.09 - base[2]) * k,
    ]
}

/// Geometry request resolved by the planner (pure function of plan + world).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoParams {
    pub cell: f32,
    pub steps: u32,
    pub agg: u32,
    pub bx: i32,
    pub by: i32,
    pub wrap: (bool, bool),
}

impl GeoParams {
    pub fn from_plan(world: &WorldModel, c: plan::PlanChunk, wrap: (bool, bool)) -> GeoParams {
        GeoParams {
            cell: world.cell,
            steps: plan::grid_steps(world.cell, c.agg, c.lod),
            agg: c.agg,
            bx: c.bx,
            by: c.by,
            wrap,
        }
    }
}

/// One block's flattened geometry + AABB (used for frustum culling and for
/// scoped dirty-region intersection).
#[derive(Clone, Debug, Default)]
pub struct ChunkGeo {
    pub verts: Vec<LandVert>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// Builds one block mesh. Wind + per-quad vertex order are identical to the
/// M1 baseline (quad (p00,p01,p10) then (p10,p01,p11)), normals are
/// heightfield-gradient based (central differences, one-sided at rect
/// edges, wrapped where the axis is periodic).
pub fn build(world: &WorldModel, p: &GeoParams) -> ChunkGeo {
    // Virtual-world extension: blocks beyond the simulated bounds (within
    // the streamed margin ring) still get geometry through the deterministic
    // `VirtualWorld` sampling; anything farther than EXTRA_CHUNKS_CAP stays
    // empty (mirrors the pre-streaming degenerate-rect behaviour).
    let nbx = plan::blocks_along(world.width, p.agg);
    let nby = plan::blocks_along(world.height, p.agg);
    let cap = plan::EXTRA_CHUNKS_CAP as i32;
    if p.bx < -cap || p.bx >= nbx as i32 + cap || p.by < -cap || p.by >= nby as i32 + cap {
        return ChunkGeo::default();
    }
    // virtual blocks outside the bounds use the unclamped rect; the clamped
    // rect only degenerates for such blocks (the world grid is aligned)
    let base = plan::block_rect(world, p.agg, p.bx, p.by);
    let (x0, y0, x1, y1) = if base.2 <= base.0 || base.3 <= base.1 {
        plan::block_rect_virtual(world, p.agg, p.bx, p.by)
    } else {
        base
    };
    let vw = VirtualWorld::wrap(world);
    let mut g = ChunkGeo::default();
    if p.steps == 0 || x1 - x0 <= 0.01 || y1 - y0 <= 0.01 {
        return g;
    }
    let steps = p.steps as usize;
    let stride = steps + 1;
    let dwx = (x1 - x0) / p.steps as f32;
    let dwy = (y1 - y0) / p.steps as f32;

    let mut wxv = Vec::with_capacity(stride);
    let mut wyv = Vec::with_capacity(stride);
    for ix in 0..stride {
        let raw = x0 + dwx * ix as f32;
        wxv.push(if p.wrap.0 { wrap_span(raw, world.width, true) } else { raw });
    }
    for iy in 0..stride {
        let raw = y0 + dwy * iy as f32;
        wyv.push(if p.wrap.1 { wrap_span(raw, world.height, true) } else { raw });
    }

    let n_samp = stride * stride;
    let mut h = Vec::with_capacity(n_samp);
    let mut col = Vec::with_capacity(n_samp);
    let mut moist = Vec::with_capacity(n_samp);
    let mut haz = Vec::with_capacity(n_samp);
    for iy in 0..stride {
        for ix in 0..stride {
            let (wx, wy) = (wxv[ix], wyv[iy]);
            // sampling goes through the VirtualWorld view: bit-identical to
            // the raw WorldModel samples inside the bounds; deterministic
            // generated + boundary-blended values outside
            h.push(vw.sample_elevation(wx, wy));
            col.push(vw.biome_raw(wx, wy));
            moist.push(vw.sample_moisture(wx, wy));
            haz.push(vw.sample_hazard(wx, wy));
        }
    }

    // slope normal at a lattice point; edges read one step beyond the rect
    // (wrapped through the border when the axis is periodic).
    let clampx = |v: usize| v.min(stride - 1);
    let clampy = |v: usize| v.min(stride - 1);
    let grid_normal = |ix: usize, iy: usize| -> [f32; 3] {
        let (ixm, ixp) = (clampx(ix.saturating_sub(1)), clampx(ix + 1));
        let (iym, iyp) = (clampy(iy.saturating_sub(1)), clampy(iy + 1));
        let edge_x = ix == 0 || ix == stride - 1;
        let edge_y = iy == 0 || iy == stride - 1;
        let hxm = if ix == 0 && p.wrap.0 {
            world.sample_elevation(wrap_span(x0 - dwx, world.width, true), wyv[iy])
        } else {
            h[iy * stride + ixm]
        };
        let hxp = if ix == stride - 1 && p.wrap.0 {
            world.sample_elevation(wrap_span(x1 + dwx, world.width, true), wyv[iy])
        } else {
            h[iy * stride + ixp]
        };
        let hym = if iy == 0 && p.wrap.1 {
            world.sample_elevation(wxv[ix], wrap_span(y0 - dwy, world.height, true))
        } else {
            h[iym * stride + ix]
        };
        let hyp = if iy == stride - 1 && p.wrap.1 {
            world.sample_elevation(wxv[ix], wrap_span(y1 + dwy, world.height, true))
        } else {
            h[iyp * stride + ix]
        };
        let sx = if edge_x { dwx * 0.5 } else { dwx };
        let sy = if edge_y { dwy * 0.5 } else { dwy };
        let dhx = (hxp - hxm) / (2.0 * sx).max(1e-5) * HEIGHT_SCALE;
        let dhy = (hyp - hym) / (2.0 * sy).max(1e-5) * HEIGHT_SCALE;
        let n = [dhx, 1.0, dhy];
        let m = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        [n[0] / m, n[1] / m, n[2] / m]
    };

    // vertical envelope for the AABB (plus sea surface, it is drawn by the
    // shader as a tint change over this same mesh)
    let mut emin = f32::INFINITY;
    let mut emax = f32::NEG_INFINITY;
    for &v in &h {
        let yv = v * HEIGHT_SCALE;
        emin = emin.min(yv);
        emax = emax.max(yv);
    }
    let sea_y = SEA_LEVEL * HEIGHT_SCALE;

    let mut verts = Vec::with_capacity(6 * steps * steps);
    for iy in 0..steps {
        for ix in 0..steps {
            let a = make_vert(&h, &col, &moist, &haz, iy * stride + ix, wxv[ix], wyv[iy], grid_normal(ix, iy));
            let b = make_vert(&h, &col, &moist, &haz, iy * stride + ix + 1, wxv[ix + 1], wyv[iy], grid_normal(ix + 1, iy));
            let c = make_vert(&h, &col, &moist, &haz, (iy + 1) * stride + ix, wxv[ix], wyv[iy + 1], grid_normal(ix, iy + 1));
            let d = make_vert(&h, &col, &moist, &haz, (iy + 1) * stride + ix + 1, wxv[ix + 1], wyv[iy + 1], grid_normal(ix + 1, iy + 1));
            let (p00, p10, p01, p11) = (a, b, c, d);
            verts.push(p00);
            verts.push(p01);
            verts.push(p10);
            verts.push(p10);
            verts.push(p01);
            verts.push(p11);
        }
    }
    g.verts = verts;
    g.aabb_min = [x0, emin.min(sea_y * 0.9) - 1.0, y0];
    g.aabb_max = [x1, emax.max(sea_y).abs().max(emax) + 1.0, y1];
    g
}

fn make_vert(
    h: &[f32],
    col: &[f32],
    moist: &[f32],
    haz: &[f32],
    i: usize,
    wx: f32,
    wy: f32,
    normal: [f32; 3],
) -> LandVert {
    let (e, b, m, hz) = (h[i], col[i], moist[i], haz[i]);
    LandVert {
        pos: [wx, e * HEIGHT_SCALE, wy],
        normal,
        color: hazard_color(biome_color(b, e, m), hz),
    }
}

#[cfg(test)]
mod vw_tests {
    use super::*;

    fn small_world() -> WorldModel {
        use alife_worldgen::WorldGenConfig;
        alife_worldgen::generate(&WorldGenConfig {
            seed: 777,
            width: 512,
            height: 512,
            cell: 8,
            terrain_scale: 1.0,
            detail: 3,
            warp: 0.35,
            moisture_scale: 0.8,
            resource_density: 1.0,
            hazard_amount: 0.4,
            fertility_contrast: 1.0,
            hazard_pulse_period: 0.0,
        })
    }

    #[test]
    fn virtual_blocks_extend_terrain_beyond_world_bounds() {
        let w = small_world();
        let nx = plan::blocks_along(w.width, 1);
        let ny = plan::blocks_along(w.height, 1);
        // a margin block directly beyond the +x bound: real streamed geometry
        let gp = GeoParams::from_plan(&w, plan::PlanChunk { agg: 1, bx: nx as i32, by: (ny / 2) as i32, lod: 1, dist: 1 }, (false, false));
        let a = build(&w, &gp);
        let b = build(&w, &gp);
        assert!(
            !a.verts.is_empty(),
            "margin block beyond the +x bound streams geometry"
        );
        assert_eq!(
            bytemuck::cast_slice::<LandVert, u8>(&a.verts),
            bytemuck::cast_slice::<LandVert, u8>(&b.verts),
            "virtual blocks are deterministic"
        );
        // and the heights agree with the virtual sampler at those points
        let vw = alife_worldgen::VirtualWorld::wrap(&w);
        for v in a.verts.iter().step_by(64) {
            let e = vw.sample_elevation(v.pos[0], v.pos[2]);
            assert!((v.pos[1] - e * HEIGHT_SCALE).abs() < 1e-3);
        }
        // a negative-side margin block streams too
        let gp = GeoParams::from_plan(&w, plan::PlanChunk { agg: 1, bx: -1, by: (ny / 2) as i32, lod: 1, dist: 1 }, (false, false));
        assert!(!build(&w, &gp).verts.is_empty(), "margin block beyond -x streams");
        // beyond the hard cap stays empty (legacy degenerate behaviour)
        let far = GeoParams::from_plan(&w, plan::PlanChunk { agg: 1, bx: 50, by: 50, lod: 16, dist: 50 }, (false, false));
        assert!(build(&w, &far).verts.is_empty(), "beyond the cap there is no geometry");
    }

    #[test]
    fn in_bounds_geometry_survives_the_virtual_sampling_path() {
        // byte-identity across rebuilds and non-negative sizes for real blocks
        let w = small_world();
        let gp = GeoParams::from_plan(&w, plan::PlanChunk { agg: 1, bx: 2, by: 3, lod: 2, dist: 1 }, (false, false));
        let a = build(&w, &gp);
        let b = build(&w, &gp);
        assert!(!a.verts.is_empty());
        assert_eq!(bytemuck::cast_slice::<LandVert, u8>(&a.verts), bytemuck::cast_slice::<LandVert, u8>(&b.verts));
        for v in &a.verts {
            assert!(v.pos[0] >= 0.0 && v.pos[0] <= w.width);
            assert!(v.pos[2] >= 0.0 && v.pos[2] <= w.height);
        }
    }

    /// grass placement consumes the VirtualWorld duck-type transparently:
    /// in-bounds results are identical to the plain WorldModel path, and
    /// building with a virtual world never panics nor exceeds the caps.
    #[test]
    fn grass_place_build_accepts_virtual_world() {
        use crate::grass::place::{self, MODE_ORBIT};
        let w = small_world();
        let vw = alife_worldgen::VirtualWorld::wrap(&w);
        let b = place::Builder {
            gen: w.generation_tag,
            cell: (4, 4),
            mode: MODE_ORBIT,
            focus: (300.0, 220.0),
            lod_scale: 1.0,
        };
        let direct = place::build(&w, &b);
        let virtual_ = place::build(&vw, &b);
        assert_eq!(direct.total, virtual_.total, "in-bounds placement identical");
        assert_eq!(direct.chunks.len(), virtual_.chunks.len());
        for (c1, c2) in direct.chunks.iter().zip(virtual_.chunks.iter()) {
            assert_eq!(c1.cx, c2.cx);
            assert_eq!(c1.insts.len(), c2.insts.len());
            for (i1, i2) in c1.insts.iter().zip(c2.insts.iter()) {
                assert_eq!(i1.x, i2.x);
                assert_eq!(i1.z, i2.z);
            }
        }
        // edge-adjacent focus using the virtual world: robust, within budget,
        // and vegetation now CONTINUES beyond the simulated bounds up to the
        // streamed margin (deterministically), mirroring the terrain ring
        let be = place::Builder {
            gen: w.generation_tag,
            cell: (7, 7),
            mode: MODE_ORBIT,
            focus: (w.width - 2.0, w.height - 2.0),
            lod_scale: 1.0,
        };
        let r = place::build(&vw, &be);
        assert!(r.total <= place::MAX_BLADES);
        let r2 = place::build(&vw, &be);
        for (c1, c2) in r.chunks.iter().zip(r2.chunks.iter()) {
            assert_eq!(c1.cx, c2.cx);
            for (i1, i2) in c1.insts.iter().zip(c2.insts.iter()) {
                assert_eq!((i1.x, i1.z, i1.height), (i2.x, i2.z, i2.height));
            }
        }
        // focus pinned at each of the four edges: clumps stream out to the
        // margin horizon (bounded by the placement GRASS_MARGIN cap)
        let margin = 512.0f32; // mirrors place::GRASS_MARGIN
        let edges = [
            (w.width - 1.0, w.height * 0.5),
            (0.0, w.height * 0.5),
            (w.width * 0.5, w.height - 1.0),
            (w.width * 0.5, 0.0),
        ];
        let mut streamed_any = false;
        for &(fx, fz) in &edges {
            let bf = place::Builder {
                gen: w.generation_tag,
                cell: (7, 7),
                mode: MODE_ORBIT,
                focus: (fx, fz),
                lod_scale: 1.0,
            };
            let res = place::build(&vw, &bf);
            for c in &res.chunks {
                for ins in &c.insts {
                    let out_amt = (ins.x - w.width).max(-(ins.x - 0.0)).max(ins.z - w.height).max(-(ins.z)).max(0.0);
                    assert!(out_amt <= margin, "placement beyond the margin cap: {out_amt}");
                    if out_amt > 0.5 {
                        streamed_any = true;
                    }
                }
            }
            assert!(res.total <= place::MAX_BLADES);
        }
        assert!(
            streamed_any,
            "vegetation must continue beyond the simulated bounds"
        );
    }
}
