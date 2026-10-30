//! Dense uniform spatial hash, rebuilt once per step (cell side ~48 units).
//! Used for food pickup, peer scans and sensor queries; neighbor order is
//! fixed so results are deterministic.

pub struct SpatialGrid {
    cell: f32,
    cells_x: usize,
    cells_y: usize,
    cells: Vec<Vec<u32>>,
}

fn sane_span(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { 1.0 }
}

impl SpatialGrid {
    pub fn new(width: f32, height: f32, cell: f32) -> Self {
        let cell = if cell.is_finite() { cell.max(8.0) } else { 48.0 };
        let w = sane_span(width);
        let h = sane_span(height);
        let cells_x = ((w / cell).ceil() as usize).clamp(1, 8192);
        let cells_y = ((h / cell).ceil() as usize).clamp(1, 8192);
        let cells = (0..cells_x.saturating_mul(cells_y)).map(|_| Vec::new()).collect();
        Self { cell, cells_x, cells_y, cells }
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            c.clear();
        }
    }

    fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let cx = (x / self.cell).floor() as i64;
        let cy = (y / self.cell).floor() as i64;
        (
            cx.rem_euclid(self.cells_x as i64) as usize,
            cy.rem_euclid(self.cells_y as i64) as usize,
        )
    }

    pub fn insert(&mut self, x: f32, y: f32, idx: u32) {
        let (cx, cy) = self.cell_of(x, y);
        self.cells[cy * self.cells_x + cx].push(idx);
    }

    /// Candidate indices from all cells overlapping the query circle, in
    /// fixed cell order (row-major from the center, wrapping at world edges).
    /// Callers must distance-check candidates.
    pub fn query_into(&self, x: f32, y: f32, r: f32, out: &mut Vec<u32>) {
        out.clear();
        if !(x.is_finite() && y.is_finite()) || !r.is_finite() || r < 0.0 {
            return;
        }
        let (cx, cy) = self.cell_of(x, y);
        let span = (r / self.cell).floor() as i64 + 1;
        let sx = span.min(self.cells_x as i64);
        let sy = span.min(self.cells_y as i64);
        for oy in -sy..=sy {
            let gy = (cy as i64 + oy).rem_euclid(self.cells_y as i64) as usize;
            let row = gy * self.cells_x;
            for ox in -sx..=sx {
                let gx = (cx as i64 + ox).rem_euclid(self.cells_x as i64) as usize;
                for &v in &self.cells[row + gx] {
                    out.push(v);
                }
            }
        }
    }
}
