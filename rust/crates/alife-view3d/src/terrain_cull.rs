//! Frustum culling for the chunked terrain (owned by Agent A). Pure math:
//! planes are extracted from the composed clip matrix M = proj * view
//! (column-major, same convention as the WGSL `proj * view * pos` path that
//! the terrain/grass pipelines use, i.e. glam `to_cols_array_2d()` inputs).
//! Chunks are tested by their PLANAR AABB (all 8 corners pushed to the
//! inside of every plane), which is cheaper than OBB and exact enough for
//! axis-aligned 64-unit blocks (the year-deliverable contract asks for
//! "AABB vs planes"; kept independent from the grass module's sphere cull
//! so ownership lines stay clean).

/// 6 outward planes as [a, b, c, d] with inside points satisfying
/// a*x + b*y + c*z + d >= 0.
#[derive(Clone, Debug)]
pub struct Frustum {
    pub planes: [[f32; 4]; 6],
}

/// Compose two matrices given in column-major array-of-columns form
/// (`[[f32;4];4]`, outer index = column).
fn mul(p: &[[f32; 4]; 4], v: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += p[k][i] * v[j][k];
            }
            out[j][i] = sum;
        }
    }
    out
}

fn norm(mut pl: [f32; 4]) -> [f32; 4] {
    let l = (pl[0] * pl[0] + pl[1] * pl[1] + pl[2] * pl[2]).sqrt();
    if l > 1e-9 {
        pl = [pl[0] / l, pl[1] / l, pl[2] / l, pl[3] / l];
    }
    pl
}

impl Frustum {
    /// Builds the frustum from the camera uniform block (`view`, `proj` are
    /// the bytemuck forms written into the shared cam uniform each frame).
    pub fn from_cam(view: &[[f32; 4]; 4], proj: &[[f32; 4]; 4]) -> Frustum {
        let m = mul(proj, view);
        // row i of column-major m: (m[0][i], m[1][i], m[2][i], m[3][i])
        let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        // GL clip space: -w <= x,y <= w, 0 <= z <= w
        let planes = [
            norm(add(r3, r0)), // left
            norm(sub(r3, r0)), // right
            norm(add(r3, r1)), // bottom
            norm(sub(r3, r1)), // top
            norm(r2),          // near (z >= 0)
            norm(sub(r3, r2)), // far
        ];
        Frustum { planes }
    }

    /// True when any part of the axis-aligned box is not entirely outside
    /// the frustum ("p-clip" test: pick the box corner most favourable to
    /// the plane; if even that is outside, the box is out).
    pub fn aabb_visible(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for p in &self.planes {
            let px = if p[0] >= 0.0 { max[0] } else { min[0] };
            let py = if p[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if p[2] >= 0.0 { max[2] } else { min[2] };
            if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
                return false;
            }
        }
        true
    }
}
