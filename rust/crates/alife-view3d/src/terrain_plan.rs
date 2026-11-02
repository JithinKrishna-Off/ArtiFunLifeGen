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

/// World rectangle of a VIRTUAL block beyond the simulated bounds (no
/// clamping; used by the streamed margin ring whose sampling goes through
/// the wrapping `VirtualWorld`). Blocks inside the bounds must keep using
/// `block_rect`.
pub fn block_rect_virtual(world: &WorldModel, agg: u32, bx: i32, by: i32) -> (f32, f32, f32, f32) {
    let _ = world;
    let block = CHUNK * agg as f32;
    let x0 = bx as f32 * block;
    let y0 = by as f32 * block;
    (x0, y0, x0 + block, y0 + block)
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
    build_plan_ext(world, agg, anchor_chunk, wrap, (0, 0, 0, 0), out)
}

// -------------------------------------------------- virtual-world extension

/// How many chunk blocks the plan may stream BEYOND the simulated bounds per
/// side, once the focus approaches that boundary ([`EDGE_WINDOW`]). This is
/// the default margin (`+4`); the plan never uses more than
/// [`EXTRA_CHUNKS_CAP`].
pub const EXTRA_CHUNKS: u32 = 4;
/// absolute hard cap of the streamed margin ring per side
pub const EXTRA_CHUNKS_CAP: u32 = 8;
/// focus distance from a world bound below which that side's margin opens.
/// 1.25 chunks of lead-in: the streamed margin ring is built nearest-first
/// while the focus is still inside the rect, so the horizon is already
/// resident when the focus reaches the edge. The flip planes sit at least a
/// full chunk outboard of a small-drift focus, so the rebuild-budget
/// contract for small focus moves remains free; crossing a plane while
/// anchored is a genuine approach event (and its plan heal is deterministic,
/// nearest-first). A focus outside the world rect opens nothing (the queen
/// cannot leave).
pub const EDGE_WINDOW: f32 = 80.0;
/// minimum world span per axis for margin streaming at all: a sandbox world
/// smaller than the streamed ring itself has no interior (its whole playable
/// area fits the resident pool; a margin ring would only flood the LRU), so
/// the frame policy opens nothing there.
pub const RING_MIN_SPAN: f32 = 256.0;

/// Chebyshev-ring radius limit for VIRTUAL blocks around the anchor
/// (bounds the resident/vertex pressure of streamed cells)
pub fn extra_ring_cap() -> u32 {
    EXTRA_CHUNKS_CAP + 4
}

/// (left, right, bottom, top) chunk counts streamed beyond the bounds
pub type PlanExtra = (u32, u32, u32, u32);

/// clamps any request below the hard cap (total resident chunks stay
/// unchanged: the engine's LRU/vertex budget keeps taking the nearest)
pub fn cap_extra(e: PlanExtra) -> PlanExtra {
    (
        e.0.min(EXTRA_CHUNKS_CAP),
        e.1.min(EXTRA_CHUNKS_CAP),
        e.2.min(EXTRA_CHUNKS_CAP),
        e.3.min(EXTRA_CHUNKS_CAP),
    )
}

/// Margin opening from the frame focus: a side streams `EXTRA_CHUNKS`
/// blocks beyond a bound only when the focus stays inside the world rect
/// and walks within [`EDGE_WINDOW`] of it (and that axis is not periodic -
/// a wrapped axis needs no margin).
pub fn near_edge_extra(width: f32, height: f32, focus: (f32, f32), wrap: (bool, bool)) -> PlanExtra {
    if width < RING_MIN_SPAN || height < RING_MIN_SPAN {
        return (0, 0, 0, 0);
    }
    let in_x = focus.0 >= 0.0 && focus.0 <= width;
    let in_y = focus.1 >= 0.0 && focus.1 <= height;
    let l = u32::from(!wrap.0 && in_x && focus.0 < EDGE_WINDOW) * EXTRA_CHUNKS;
    let r = u32::from(!wrap.0 && in_x && focus.0 > width - EDGE_WINDOW) * EXTRA_CHUNKS;
    let b = u32::from(!wrap.1 && in_y && focus.1 < EDGE_WINDOW) * EXTRA_CHUNKS;
    let t = u32::from(!wrap.1 && in_y && focus.1 > height - EDGE_WINDOW) * EXTRA_CHUNKS;
    cap_extra((l, r, b, t))
}

/// Builds the near-first plan INCLUDING the virtual-world margin ring:
/// blocks beyond the simulated bounds are appended in the opened sides'
/// directions, ranked by the same Chebyshev ring distance (stable sort keeps
/// real blocks ahead of virtual blocks on ties), so the horizon extends as
/// the camera/queen approaches an edge and real geometry is never evicted
/// for virtual geometry.
#[allow(clippy::too_many_arguments)]
pub fn build_plan_ext(
    world: &WorldModel,
    agg: u32,
    anchor_chunk: (i32, i32),
    wrap: (bool, bool),
    extra: PlanExtra,
    out: &mut Vec<PlanChunk>,
) {
    out.clear();
    let nbx = blocks_along(world.width, agg);
    let nby = blocks_along(world.height, agg);
    let abx = (anchor_chunk.0 as f32 / agg as f32).floor() as i32;
    let aby = (anchor_chunk.1 as f32 / agg as f32).floor() as i32;
    let (exl, exr, exb, ext) = {
        let (l, r, b, t) = cap_extra(extra);
        let (l, r) = if wrap.0 { (0, 0) } else { (l, r) };
        let (b, t) = if wrap.1 { (0, 0) } else { (b, t) };
        (l as i32, r as i32, b as i32, t as i32)
    };
    let xlo = -exl;
    let xhi = nbx as i32 + exr;
    let ylo = -exb;
    let yhi = nby as i32 + ext;
    for by in ylo..yhi {
        for bx in xlo..xhi {
            let real_x = bx >= 0 && bx < nbx as i32;
            let real_y = by >= 0 && by < nby as i32;
            let dx = if real_x {
                axis_dist(bx, abx, nbx, wrap.0)
            } else {
                (bx - abx).abs()
            };
            let dy = if real_y {
                axis_dist(by, aby, nby, wrap.1)
            } else {
                (by - aby).abs()
            };
            let d = dx.max(dy) as u32;
            // virtual blocks stay a bounded ring around the focus even when
            // the opened margin spans a long bound (hard cap EXTRA_CHUNKS_CAP,
            // window floor keeps the streaming horizon a fixed-size fan)
            if (!real_x || !real_y) && d > extra_ring_cap() {
                continue;
            }
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

#[cfg(test)]
mod vw_tests {
    use super::*;
    use alife_worldgen::FieldGrid;

    fn world(width: f32, height: f32) -> WorldModel {
        let g = |v: f32| FieldGrid::new(16, 16, v);
        WorldModel {
            width,
            height,
            cell: 2.0,
            elevation: g(0.5),
            moisture: g(0.5),
            temperature: g(0.5),
            fertility: g(0.8),
            hazard: g(0.0),
            biome: g(0.0),
            generation_tag: 21,
            hazard_pulse_period: 0.0,
        }
    }

    #[test]
    fn zero_extra_matches_legacy_plan_exactly() {
        let w = world(320.0, 256.0);
        let mut legacy: Vec<PlanChunk> = Vec::new();
        build_plan(&w, 1, (2, 1), (false, false), &mut legacy);
        let n_block = (blocks_along(w.width, 1) * blocks_along(w.height, 1)) as usize;
        assert_eq!(legacy.len(), n_block);
        let mut ext: Vec<PlanChunk> = Vec::new();
        build_plan_ext(&w, 1, (2, 1), (false, false), (0, 0, 0, 0), &mut ext);
        assert_eq!(legacy, ext, "zero margin must be the legacy plan");
        // near-first order preserved with margins too
        build_plan_ext(&w, 1, (2, 1), (false, false), (4, 0, 0, 0), &mut ext);
        for win in ext.windows(2) {
            assert!(win[0].dist <= win[1].dist, "margin plan still near-first");
        }
        assert_eq!(ext[0], legacy[0], "same anchor first chunk");
    }

    #[test]
    fn margin_stream_blocks_beyond_bounds_near_the_focus_side() {
        let w = world(320.0, 256.0);
        // anchor on the right edge, right margin open
        let mut p: Vec<PlanChunk> = Vec::new();
        build_plan_ext(&w, 1, (4, 1), (false, false), (0, 4, 0, 0), &mut p);
        assert!(p.iter().any(|c| c.bx == blocks_along(w.width, 1) as i32));
        assert!(!p.iter().any(|c| c.bx < 0), "closed side stays closed");
        // open left margin, anchor at the left edge
        build_plan_ext(&w, 1, (0, 1), (false, false), (4, 0, 0, 0), &mut p);
        assert!(p.iter().any(|c| c.bx == -1));
        // margin folds in beyond the ring cap regardless of request size
        build_plan_ext(&world(320.0, 256.0), 1, (0, 1), (false, false), (999, 0, 0, 0), &mut p);
        assert!(
            p.iter().all(|c| c.bx >= -(EXTRA_CHUNKS_CAP as i32)),
            "the streamed ring never exceeds the hard cap: {:?}",
            p.iter().map(|c| c.bx).min()
        );
        // wrapped axes never stream
        build_plan_ext(&w, 1, (4, 1), (true, true), (4, 4, 4, 4), &mut p);
        assert!(p.iter().all(|c| c.bx >= 0 && c.bx < blocks_along(w.width, 1) as i32));
    }

    #[test]
    fn cap_extra_clamps_to_the_hard_cap() {
        assert_eq!(cap_extra((99, 0, 2, 9)), (EXTRA_CHUNKS_CAP, 0, 2, EXTRA_CHUNKS_CAP));
        assert_eq!(EXTRA_CHUNKS_CAP, 8);
        assert_eq!(EXTRA_CHUNKS, 4);
    }

    #[test]
    fn near_edge_extra_only_when_the_focus_walks_to_the_edge() {
        // sandbox worlds smaller than the ring span never stream
        assert_eq!(near_edge_extra(160.0, 120.0, (30.0, 30.0), (false, false)), (0, 0, 0, 0));
        let extra = near_edge_extra(256.0, 256.0, (30.0, 128.0), (false, false));
        assert_eq!(extra, (EXTRA_CHUNKS, 0, 0, 0), "focus < EDGE_WINDOW opens the near side");
        let far = near_edge_extra(256.0, 256.0, (100.0, 100.0), (false, false));
        assert_eq!(far, (0, 0, 0, 0));
        let at_edge_min = near_edge_extra(256.0, 256.0, (0.0, 300.0), (false, false));
        assert_eq!(at_edge_min, (EXTRA_CHUNKS, 0, 0, 0), "edge focus opens that side; out-of-rect axis opens nothing");
        let wreq = near_edge_extra(256.0, 256.0, (256.0, 0.0), (false, false));
        assert_eq!(wreq, (0, EXTRA_CHUNKS, EXTRA_CHUNKS, 0), "dual edges open dual sides");
        assert_eq!(near_edge_extra(256.0, 256.0, (0.0, 0.0), (true, false)).0, 0, "wrapped x never streams");
    }
}
