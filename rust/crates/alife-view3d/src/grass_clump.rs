//! Shared 7-blade grass clump mesh (Agent 2).
//!
//! Blade-major static geometry consumed by the instanced grass pipeline:
//! each blade is 24 verts (4 stacked quads x 6 verts, non-indexed), each
//! vertex stays TWO f32s: `[lateral, v]` with the shader deriving the width
//! profile from `v`. One clump = 7 blades = 168 verts / 56 tris, built
//! ONCE and shared across all instances; per-clump orientation/height
//! variation lives entirely in the instance uniforms (lean/height/width/
//! seed/pad), never in this mesh.
//!
//! Within the `(lateral, v)` attribute contract the only shape knobs here
//! are pre-warped lattices: every blade keeps the same v column (i/4 grid
//! of 4 stacked quads, roots anchored at v=0) but its lateral coordinate
//! is pre-multiplied and pre-skewed by a hard-coded blade spec, so the
//! fan of blades inside one clump differs in effective width, taper,
//! static lateral curl and mid-blade bend even though they all share one
//! mesh-space orientation (the per-instance yaw from `lean` rotates the
//! whole clump).

/// Static per-blade lattice warp parameters baked into the clump mesh.
/// `lateral` is pre-multiplied by `width`, then tapered toward the tip
/// (`tip` pinch, negative = flare, `taper` exponent on the ramp), then
/// given a static spine bend: `skew * v^2` (full-height curl, opposite
/// signs = curved-left / curved-right pair) plus `mid * 4v(1-v)` (bend
/// peaking at mid-blade, zero at root and tip).
struct BladeSpec {
    width: f32,
    tip: f32,
    taper: f32,
    skew: f32,
    mid: f32,
}

/// The 7 hard-coded blade variants of one clump: a straight reference, a
/// narrow blade, a curved-left/curved-right pair, a mid-bent blade, a
/// broad flaring blade and a gently leaning outlyer.
const BLADES: [BladeSpec; 7] = [
    // 0: straight reference (full nominal width)
    BladeSpec { width: 1.00, tip: 0.15, taper: 1.6, skew: 0.00, mid: 0.00 },
    // 1: narrow pinched
    BladeSpec { width: 0.70, tip: 0.45, taper: 1.3, skew: 0.00, mid: 0.00 },
    // 2: curved-left (static lateral curl, magnitude 0.35)
    BladeSpec { width: 1.05, tip: 0.10, taper: 2.0, skew: -0.35, mid: 0.00 },
    // 3: curved-right (mirror of blade 2)
    BladeSpec { width: 1.05, tip: 0.10, taper: 2.0, skew: 0.35, mid: 0.00 },
    // 4: bent-mid (bend peaks at v=0.5, root and tip stay on the spine)
    BladeSpec { width: 0.95, tip: 0.20, taper: 1.6, skew: 0.00, mid: 0.28 },
    // 5: broad, slightly flaring toward the tip
    BladeSpec { width: 1.25, tip: -0.08, taper: 2.0, skew: 0.00, mid: 0.00 },
    // 6: tall outlyer with a gentle rightward lean
    BladeSpec { width: 0.85, tip: 0.05, taper: 1.8, skew: 0.18, mid: 0.00 },
];

/// Applies one blade's static shaping to a raw lattice vertex.
fn warp(bl: &BladeSpec, lateral: f32, v: f32) -> [f32; 2] {
    let shaped = lateral * bl.width * (1.0 - bl.tip * v.powf(bl.taper));
    let bend = bl.skew * v * v + bl.mid * 4.0 * v * (1.0 - v);
    [shaped + bend, v]
}

/// Emitted verts per blade: 4 stacked quads x 6 non-indexed verts.
const VERTS_PER_BLADE: usize = 24;

/// Builds the shared clump mesh: 7 blades x 24 verts = 168 verts, blade
/// major (blade b occupies verts b*24 .. (b+1)*24). Same quad strip
/// winding as the single-blade lattice in grass.rs so the pipeline can
/// swap the vertex range without touching instance layout: triangles are
/// (a, c, b) and (b, c, d) per quad, a/c = left edge, b/d = right edge.
pub fn clump_verts() -> Vec<[f32; 2]> {
    let mut verts = Vec::with_capacity(BLADES.len() * VERTS_PER_BLADE);
    for bl in BLADES.iter() {
        for q in 0..4u32 {
            let v0 = q as f32 / 4.0;
            let v1 = (q + 1) as f32 / 4.0;
            let a = warp(bl, -1.0, v0);
            let b = warp(bl, 1.0, v0);
            let c = warp(bl, -1.0, v1);
            let d = warp(bl, 1.0, v1);
            for vert in [a, c, b, b, c, d] {
                verts.push(vert);
            }
        }
    }
    verts
}

/// Total vertex count of the clump mesh (7 * 24); consumed by the
/// pipeline's draw range and by LOD code sizing the shared vertex buffer.
pub fn clump_vert_count() -> usize {
    BLADES.len() * VERTS_PER_BLADE
}

/// Blades per clump; one draw call instance covers this many blades.
pub fn blades_per_clump() -> usize {
    BLADES.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_contract() {
        assert_eq!(clump_vert_count(), 168);
        assert_eq!(blades_per_clump(), 7);
        let v = clump_verts();
        assert_eq!(v.len(), 168);
        assert_eq!(168, 7 * 24);
    }

    #[test]
    fn lateral_stays_within_bound() {
        for [lat, _] in clump_verts() {
            assert!(
                lat.abs() <= 1.4,
                "lateral out of [-1.4, 1.4]: {lat}"
            );
        }
    }

    #[test]
    fn v_stays_in_unit_range() {
        for [_, v] in clump_verts() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&v),
                "v out of [0,1]: {v}"
            );
        }
    }

    #[test]
    fn deterministic_output() {
        assert_eq!(clump_verts(), clump_verts());
        assert_eq!(clump_verts(), clump_verts());
    }

    #[test]
    fn curved_blades_skew_against_reference() {
        let verts = clump_verts();
        let blade = |b: usize| &verts[b * 24..(b + 1) * 24];
        let mean_lat = |vs: &[[f32; 2]]| vs.iter().map(|p| p[0]).sum::<f32>() / 24.0;
        let m0 = mean_lat(blade(0));
        let m2 = mean_lat(blade(2));
        let m3 = mean_lat(blade(3));
        assert!(
            m2 < m0 - 0.05,
            "curved-left blade 2 must skew left vs blade 0: {m2} vs {m0}"
        );
        assert!(
            m3 > m0 + 0.05,
            "curved-right blade 3 must skew right vs blade 0: {m3} vs {m0}"
        );
    }
}
