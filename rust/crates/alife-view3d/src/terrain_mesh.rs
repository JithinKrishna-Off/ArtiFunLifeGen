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
use alife_worldgen::WorldModel;

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
    let (x0, y0, x1, y1) = plan::block_rect(world, p.agg, p.bx, p.by);
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
        wxv.push(wrap_span(x0 + dwx * ix as f32, world.width, p.wrap.0));
    }
    for iy in 0..stride {
        wyv.push(wrap_span(y0 + dwy * iy as f32, world.height, p.wrap.1));
    }

    let n_samp = stride * stride;
    let mut h = Vec::with_capacity(n_samp);
    let mut col = Vec::with_capacity(n_samp);
    let mut moist = Vec::with_capacity(n_samp);
    let mut haz = Vec::with_capacity(n_samp);
    for iy in 0..stride {
        for ix in 0..stride {
            let (wx, wy) = (wxv[ix], wyv[iy]);
            h.push(world.sample_elevation(wx, wy));
            col.push(world.biome.sample(wx, wy, world.width, world.height));
            moist.push(world.sample_moisture(wx, wy));
            haz.push(world.sample_hazard(wx, wy));
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
