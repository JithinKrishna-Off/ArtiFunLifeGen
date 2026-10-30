//! Pure, deterministic ring-LOD planning for the chunked terrain (owned by
//! Agent A). No GPU, no randomness: a plan is a pure function of (world dims,
//! focal anchor chunk, ring table, wrap flags). Build order is used to stage
//! scarce resources nearest-first in `terrain::ensure_mesh`, so it must be
//! sorted near-first and stable.
//!
//! Salved and cleaned from the aborted first run (the staging half-mutation
//! loop was wrong and moved into the engine, which knows the true resident
//! state). This module only classifies/ranks - it never owns resources.
//!
//! Circular-continuous world handling: when the world's fields are
//! (nearly) periodic across a border, chunk ring distance uses the torus
//! metric on that axis so LOD tiers do not get a hat-shaped discontinuity
//! at the seam. Detected once by `terrain::mesh::edge_wrap`.

use super::{CHUNK, HEIGHT_SCALE, SEA_LEVEL};
use crate::scene::CameraMode;
use alife_worldgen::WorldModel;

/// One planned block (= `agg` merged 64-unit chunks). `lod == 0` means the
/// flat-plate degradation (single quad built from the four corner samples).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanChunk {
    pub agg: u32,
    pub bx: i32,
    pub by: i32,
    pub lod: u32,
    pub dist: u32,
}

/// Ring -> LOD table (Chebyshev block distance from the anchor):
/// near full detail at ~`world.cell` spacing, mid rings 2x/4x/8x, far ring
/// 16x which at the default cell degenerates to a flat corner quad plate.
pub fn lod_of(d: u32) -> u32 {
    match d {
        0..=2 => 1,
        3..=4 => 2,
        5..=7 => 4,
        8..=11 => 8,
        _ => 16,
    }
}

/// Full-detail grid steps of ONE 64-unit chunk for the world cell size
/// (~world.cell spacing near the focus, clamped to keep the vertex budget).
pub fn full_steps(cell: f32) -> u32 {
    let c = cell.max(0.05);
    ((CHUNK / c).round() as u32).clamp(4, 24)
}

/// Grid steps for a block at a given LOD (0 = plate -> single quad).
pub fn grid_steps(cell: f32, _agg: u32, lod: u32) -> u32 {
    if lod == 0 {
        return 1;
    }
    let l1 = full_steps(cell) as f32;
    let n = (l1 / lod as f32).ceil() as u32;
    n.clamp(1, 24)
}

/// Non-indexed vertex count of the block mesh (6 per quad).
pub fn verts_of(cell: f32, agg: u32, lod: u32) -> u32 {
    6 * grid_steps(cell, agg, lod) * grid_steps(cell, agg, lod)
}

/// Mode classification used by the planner; also part of the engine's
/// plan signature so Orbit/Follow/Planet switches rebuild cleanly.
pub fn mode_tag(mode: &CameraMode) -> u8 {
    match mode {
        CameraMode::Orbit { .. } => 0,
        CameraMode::Follow { .. } => 1,
        CameraMode::Planet => 2,
    }
}

/// Planet view block aggregation: smallest power-of-two merge (1/2/4) so the
/// whole world stays inside ~93 blocks (keeps the 128-entry cache useful).
pub fn agg_for_planet(width: f32, height: f32) -> u32 {
    let mut agg = 1u32;
    loop {
        let block = CHUNK * agg as f32;
        let n = (width.div_euclid(block) + 1.0) * (height.div_euclid(block) + 1.0);
        if n <= 96.0 || agg >= 4 {
            return agg;
        }
        agg *= 2;
    }
}

/// Aggregation for a camera mode: orbit/follow use raw 64-unit chunks.
pub fn agg_for_mode(mode: &CameraMode, world: &WorldModel) -> u32 {
    match mode {
        CameraMode::Planet => agg_for_planet(world.width, world.height),
        _ => 1,
    }
}

/// Number of blocks per axis for an aggregation level.
pub fn blocks_along(span: f32, agg: u32) -> u32 {
    (span / (CHUNK * agg as f32)).ceil().clamp(0.0, 8192.0) as u32
}

/// World rectangle covered by a block (clamped to the world). Empty blocks
/// return a degenerate rect.
pub fn block_rect(world: &WorldModel, agg: u32, bx: i32, by: i32) -> (f32, f32, f32, f32) {
    let block = CHUNK * agg as f32;
    let x0 = bx as f32 * block;
    let y0 = by as f32 * block;
    let x1 = (x0 + block).min(world.width).max(x0);
    let y1 = (y0 + block).min(world.height).max(y0);
    (x0, y0, x1, y1)
}

/// Torus-metric Chebyshev distance on one axis when that axis is
/// continuous (wraps), plain metric otherwise.
pub fn axis_dist(a: i32, b: i32, n: u32, wrap: bool) -> i32 {
    let n = n as i32;
    if !wrap {
        return (a - b).abs();
    }
    let d = (a - b).rem_euclid(n).min((b - a).rem_euclid(n));
    d
}

/// Builds the near-first plan: every in-world block, ranked near-first by
/// torus-aware Chebyshev distance to the anchor, with its ring LOD.
/// Deterministic: stable sort of a fixed iteration order.
pub fn build_plan(
    world: &WorldModel,
    agg: u32,
    anchor_chunk: (i32, i32),
    wrap: (bool, bool),
    out: &mut Vec<PlanChunk>,
) {
    out.clear();
    let nbx = blocks_along(world.width, agg);
    let nby = blocks_along(world.height, agg);
    let abx = (anchor_chunk.0 as f32 / agg as f32).floor() as i32;
    let aby = (anchor_chunk.1 as f32 / agg as f32).floor() as i32;
    for by in 0..nby as i32 {
        for bx in 0..nbx as i32 {
            let dx = axis_dist(bx, abx, nbx, wrap.0);
            let dy = axis_dist(by, aby, nby, wrap.1);
            let d = dx.max(dy) as u32;
            out.push(PlanChunk { agg, bx, by, lod: lod_of(d), dist: d });
        }
    }
    // near-first order; stable keeps iteration order as tie-break
    out.sort_by_key(|c| c.dist);
}

/// Height convention helper shared by picking/fixtures (elevation field x
/// vertical scale). Kept in one place so chunk meshes and CPU picking agree.
pub fn norm_height(elevation: f32) -> f32 {
    elevation * HEIGHT_SCALE
}

/// Sea surface Y in world units (mirrors the WGSL constant).
pub fn sea_world_y() -> f32 {
    SEA_LEVEL * HEIGHT_SCALE
}
