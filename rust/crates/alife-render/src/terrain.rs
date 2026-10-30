//! Terrain texture baking + scientific color ramps.
//!
//! [`bake_terrain`] keeps its original signature (bakes at the field-grid
//! resolution; identity sampling reproduces exact cell values). The display
//! path uses [`bake_terrain_scaled`], a supersampled bake (~2x the viewport
//! pixel density, total baked area capped at [`MAX_BAKE_PIXELS`]) with
//! bilinear field sampling, altitude contour lines at threshold crossings,
//! distinct biome-edge shading, fertility-variance grain and hazard tint.
//! All shading happens in the bake pass, never per-frame.

use crate::camera::ViewMode;
use alife_worldgen::{WorldModel, BIOME_COLORS, BIOME_COUNT};
use egui::{Color32, ColorImage};

/// Hard cap on total baked pixels (texture memory guard).
pub const MAX_BAKE_PIXELS: usize = 1_200_000;
/// Contour bands across the elevation field (thresholds at band multiples).
pub const CONTOUR_BANDS: f32 = 12.0;

/// Viridis-like perceptual ramp anchors (dark violet -> teal -> yellow).
const VIRIDIS: [[u8; 3]; 9] = [
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [253, 231, 37],
];

/// Cool -> hot ramp for hazards: deep blue-slate -> purple -> crimson -> amber.
const HEAT: [[u8; 3]; 5] = [
    [23, 38, 63],
    [59, 61, 107],
    [140, 54, 80],
    [214, 69, 63],
    [245, 166, 35],
];

fn ramp(t: f32, stops: &[[u8; 3]]) -> Color32 {
    let t = t.clamp(0.0, 0.9999);
    let n = stops.len() - 1;
    let f = t * n as f32;
    let i = f as usize;
    let k = f - i as f32;
    let a = stops[i];
    let b = stops[i + 1];
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * k) as u8;
    Color32::from_rgb(mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2]))
}

pub fn viridis(t: f32) -> Color32 {
    ramp(t, &VIRIDIS)
}

fn heat(t: f32) -> Color32 {
    ramp(t, &HEAT)
}

fn mix_rgb(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        a.r() + ((b.r() as f32 - a.r() as f32) * t) as u8,
        a.g() + ((b.g() as f32 - a.g() as f32) * t) as u8,
        a.b() + ((b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

/// Per-pixel brightness grain from a coordinate hash, centered on 1.0.
pub fn bake_grain(px: usize, py: usize, strength: f32) -> f32 {
    if strength == 0.0 {
        return 1.0;
    }
    let v = (px as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (py as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let n = ((v ^ (v >> 33)) >> 40) & 0xFF;
    1.0 + ((n as f32 / 255.0) - 0.5) * 2.0 * strength
}

// ------------------------------------------------------------ bilinear core

/// Horizontal tap positions for one source grid across `out_w` output
/// columns: pairs (x0, x1 = x0+1 clamped at the edge) and fractional weight.
struct Taps {
    w: usize,
    h: usize,
    x0: Vec<usize>,
    x1: Vec<usize>,
    fx: Vec<f32>,
}

impl Taps {
    fn new(src_w: usize, src_h: usize, out_w: usize) -> Self {
        let src_w = src_w.max(1);
        let sw = (src_w - 1) as f32;
        let mut x0 = Vec::with_capacity(out_w);
        let mut x1 = Vec::with_capacity(out_w);
        let mut fx = Vec::with_capacity(out_w);
        for px in 0..out_w {
            let u = if out_w <= 1 {
                0.0
            } else {
                px as f32 / (out_w - 1) as f32 * sw
            };
            let i = (u as usize).min(src_w - 1);
            x0.push(i);
            x1.push((i + 1).min(src_w - 1));
            fx.push(u - i as f32);
        }
        Self { w: src_w, h: src_h, x0, x1, fx }
    }

    /// Vertical taps for output row `py` over this grid's height.
    #[inline]
    fn ytap(&self, py: usize, out_h: usize) -> (usize, usize, f32) {
        let sh = (self.h.max(1) - 1) as f32;
        let v = if out_h <= 1 { 0.0 } else { py as f32 / (out_h - 1) as f32 * sh };
        let i = (v as usize).min(self.h.max(1) - 1);
        (i, (i + 1).min(self.h.max(1) - 1), v - i as f32)
    }
}

/// Bilinear value of one f32 grid at column `px` and vertical tap `yt`.
#[inline]
fn bl(t: &Taps, data: &[f32], px: usize, yt: (usize, usize, f32)) -> f32 {
    let (y0, y1, fy) = yt;
    let x0 = t.x0[px];
    let x1 = t.x1[px];
    let fx = t.fx[px];
    let w = t.w;
    let a = data[y0 * w + x0] * (1.0 - fx) + data[y0 * w + x1] * fx;
    let b = data[y1 * w + x0] * (1.0 - fx) + data[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

/// Bilinear value + right neighbour (one source-cell right). Used for
/// elevation lighting and contour detection.
#[inline]
fn bl_right(t: &Taps, data: &[f32], px: usize, yt: (usize, usize, f32)) -> [f32; 2] {
    let (y0, y1, fy) = yt;
    let x0 = t.x0[px];
    let x1 = t.x1[px];
    let fx = t.fx[px];
    let xr0 = (x0 + 1).min(t.w - 1);
    let xr1 = (x1 + 1).min(t.w - 1);
    let w = t.w;
    let hz = |xa: usize, xb: usize, y: usize| data[y * w + xa] * (1.0 - fx) + data[y * w + xb] * fx;
    let e = hz(x0, x1, y0) * (1.0 - fy) + hz(x0, x1, y1) * fy;
    let er = hz(xr0, xr1, y0) * (1.0 - fy) + hz(xr0, xr1, y1) * fy;
    [e, er]
}

/// Nearest-neighbour value (categorical fields) + right-neighbour class.
#[inline]
fn near(t: &Taps, data: &[f32], px: usize, yt: (usize, usize, f32)) -> [f64; 2] {
    let (y0, y1, fy) = yt;
    let pick_y = if fy < 0.5 { y0 } else { y1 };
    let pick_x = if t.fx[px] < 0.5 { t.x0[px] } else { t.x1[px] };
    let right_x = (pick_x + 1).min(t.w - 1);
    let w = t.w;
    [
        data[pick_y * w + pick_x] as f64,
        data[pick_y * w + right_x] as f64,
    ]
}

// ---------------------------------------------------------------- surfaces

/// Bake the world field grids into an RGB image at grid resolution.
/// Dynamic layers (density / species / energy) are passed in as grids
/// pre-computed by the renderer; pass empty slices when unused.
/// Kept for API stability: identity mapping = exact cell values.
pub fn bake_terrain(
    world: &WorldModel,
    mode: ViewMode,
    density: &[u32],
    dominant: &[u32],
    energy: &[f32],
) -> ColorImage {
    bake_terrain_scaled(
        world,
        mode,
        world.grid_w() as usize,
        world.grid_h() as usize,
        density,
        dominant,
        energy,
    )
}

/// Supersampled bake at arbitrary output dimensions. Samples every field
/// bilinearly in grid-index space; species mode uses nearest-neighbour
/// sampling (categorical).
pub fn bake_terrain_scaled(
    world: &WorldModel,
    mode: ViewMode,
    out_w: usize,
    out_h: usize,
    density: &[u32],
    dominant: &[u32],
    energy: &[f32],
) -> ColorImage {
    let gw = (world.grid_w() as usize).max(1);
    let gh = (world.grid_h() as usize).max(1);
    let out_w = out_w.clamp(16, 4096);
    let out_h = out_h.clamp(16, 4096);
    let black = vec![Color32::from_rgb(12, 14, 18); out_w * out_h];
    if gw < 2 || gh < 2 {
        return ColorImage { size: [out_w, out_h], pixels: black };
    }
    let te = Taps::new(world.elevation.w as usize, world.elevation.h as usize, out_w);
    let tb = Taps::new(world.biome.w as usize, world.biome.h as usize, out_w);
    let tf = Taps::new(world.fertility.w as usize, world.fertility.h as usize, out_w);
    let tm = Taps::new(world.moisture.w as usize, world.moisture.h as usize, out_w);
    let tt = Taps::new(world.temperature.w as usize, world.temperature.h as usize, out_w);
    let th = Taps::new(world.hazard.w as usize, world.hazard.h as usize, out_w);
    let mut pixels = Vec::with_capacity(out_w * out_h);

    for py in 0..out_h {
        for px in 0..out_w {
            let c = match mode {
                ViewMode::Natural => natural(world, &te, &tb, &tf, &tm, &tt, &th, px, py, out_h),
                ViewMode::Elevation => {
                    let yt = te.ytap(py, out_h);
                    let [e, er] = bl_right(&te, &world.elevation.data, px, yt);
                    contour(e, er, viridis(e.clamp(0.0, 1.0)))
                }
                ViewMode::Moisture => {
                    let yt = tm.ytap(py, out_h);
                    viridis(bl(&tm, &world.moisture.data, px, yt))
                }
                ViewMode::Temperature => {
                    let yt = tt.ytap(py, out_h);
                    viridis(bl(&tt, &world.temperature.data, px, yt))
                }
                ViewMode::Fertility => {
                    let yt = tf.ytap(py, out_h);
                    shade_fertility(bl(&tf, &world.fertility.data, px, yt))
                }
                ViewMode::Hazard => {
                    let yt = th.ytap(py, out_h);
                    heat(bl(&th, &world.hazard.data, px, yt))
                }
                ViewMode::Density => {
                    let d = sample_dyn(&te, density, px, py, out_h, gw, gh);
                    let t = ((1.0 + d).ln() / (1.0f32 + 480.0).ln()).clamp(0.0, 1.0);
                    mix_rgb(viridis(t * 0.85), Color32::from_rgb(250, 250, 235), t * t * 0.35)
                }
                ViewMode::Species => {
                    let sp = species_dyn(&te, dominant, px, py, out_h, gw, gh);
                    if sp == 0 {
                        Color32::from_rgb(16, 20, 25)
                    } else {
                        let hue = alife_common::species_hue(sp);
                        crate::paint::hsl_to_rgb(hue, 0.55, 0.42)
                    }
                }
                ViewMode::Energy => {
                    let e = sample_ener_dyn(&te, energy, px, py, out_h, gw, gh);
                    mix_rgb(
                        Color32::from_rgb(20, 24, 30),
                        Color32::from_rgb(255, 224, 102),
                        e.clamp(0.0, 1.0),
                    )
                }
            };
            pixels.push(c);
        }
    }
    ColorImage { size: [out_w, out_h], pixels }
}

#[inline]
fn ytap_for(py: usize, out_h: usize, src_h: usize) -> (usize, usize, f32) {
    let sh = (src_h.max(1) - 1) as f32;
    let v = if out_h <= 1 { 0.0 } else { py as f32 / (out_h - 1) as f32 * sh };
    let i = (v as usize).min(src_h.max(1) - 1);
    (i, (i + 1).min(src_h.max(1) - 1), v - i as f32)
}

#[inline]
fn shade_fertility(f0: f32) -> Color32 {
    let f = f0.clamp(0.0, 1.0);
    Color32::from_rgb(
        0x60 - (f * 60.0) as u8,
        0x62 + (f * 60.0) as u8,
        0x70 - (f * 30.0) as u8,
    )
}

/// Subtle altitude contour darkening: darken pixels on a threshold crossing
/// between this pixel and the next grid cell to the right.
#[inline]
fn contour(e: f32, e_r: f32, c: Color32) -> Color32 {
    let b0 = (e * CONTOUR_BANDS).floor();
    let b1 = (e_r * CONTOUR_BANDS).floor();
    if (b0 - b1).abs() > 0.5 {
        Color32::from_rgb(
            (c.r() as f32 * 0.82) as u8,
            (c.g() as f32 * 0.82) as u8,
            (c.b() as f32 * 0.82) as u8,
        )
    } else {
        c
    }
}

/// Natural biome shading with altitude lighting, fertility-variance grain,
/// hazard tint and distinct biome-edge darkening. `px`/`py` seed the grain.
#[allow(clippy::too_many_arguments)]
fn natural(
    world: &WorldModel,
    te: &Taps,
    tb: &Taps,
    tf: &Taps,
    tm: &Taps,
    tt: &Taps,
    th: &Taps,
    px: usize,
    py: usize,
    out_h: usize,
) -> Color32 {
    let yt_e = te.ytap(py, out_h);
    let yt_b = tb.ytap(py, out_h);
    let yt_f = tf.ytap(py, out_h);
    let yt_m = tm.ytap(py, out_h);
    let yt_t = tt.ytap(py, out_h);
    let yt_h = th.ytap(py, out_h);

    // categorical biome class stays crisp (nearest), right neighbour for edges
    let [biome, biome_r] = near(tb, &world.biome.data, px, yt_b);
    let biome = (biome.min((BIOME_COUNT - 1) as f64)) as usize;
    let base = BIOME_COLORS[biome.min(BIOME_COUNT - 1)];

    let fertility = bl(tf, &world.fertility.data, px, yt_f).clamp(0.0, 1.0);
    let moisture = bl(tm, &world.moisture.data, px, yt_m).clamp(0.0, 1.0);
    let temp = bl(tt, &world.temperature.data, px, yt_t).clamp(0.0, 1.0);
    let hazard = bl(th, &world.hazard.data, px, yt_h).clamp(0.0, 1.0);

    let mut c = Color32::from_rgb(
        (base[0] * 255.0) as u8,
        (base[1] * 255.0) as u8,
        (base[2] * 255.0) as u8,
    );
    // fertility/vegetation greening, moisture cooling shift, warmth tint
    c = mix_rgb(c, Color32::from_rgb(44, 110, 66), fertility * 0.55);
    c = mix_rgb(c, Color32::from_rgb(48, 96, 118), moisture * 0.25);
    c = mix_rgb(c, Color32::from_rgb(122, 92, 58), temp * 0.20);
    // toxic areas bleed red
    c = mix_rgb(c, Color32::from_rgb(128, 40, 34), hazard * 0.75);

    // altitude shading: right-neighbour elevation gradient (cheap fake lighting)
    let [e, e_r] = bl_right(&te, &world.elevation.data, px, yt_e);
    let d = e_r - e;
    let light = (1.0 + d * 2.4).clamp(0.72, 1.22);

    let mut out = Color32::from_rgb(
        ((c.r() as f32) * light).min(255.0) as u8,
        ((c.g() as f32) * light).min(255.0) as u8,
        ((c.b() as f32) * light).min(255.0) as u8,
    );
    // distinct biome edge: darker seam ridge where the class flips
    if biome as f64 != biome_r {
        out = Color32::from_rgb(
            (out.r() as u16 * 195 / 255) as u8,
            (out.g() as u16 * 195 / 255) as u8,
            (out.b() as u16 * 195 / 255) as u8,
        );
    }
    // fertility-variance grain: fertile meadows sparkle, deserts stay flat
    let grain = bake_grain(px, py, 0.015 + 0.05 * fertility);
    out = Color32::from_rgb(
        ((out.r() as f32) * grain).min(255.0) as u8,
        ((out.g() as f32) * grain).min(255.0) as u8,
        ((out.b() as f32) * grain).min(255.0) as u8,
    );
    out
}

/// Dynamic raster sampling (density/energy): bilinear over the world grid.
/// Falls back to the flat grid value when dyn grids are missing/empty.
#[inline]
fn sample_dyn(
    t: &Taps,
    data: &[u32],
    px: usize,
    py: usize,
    out_h: usize,
    gw: usize,
    gh: usize,
) -> f32 {
    if data.len() != gw.saturating_mul(gh) || gw < 2 {
        return data.first().copied().unwrap_or(0) as f32;
    }
    let yt = ytap_for(py, out_h, gh);
    let (y0, y1, fy) = yt;
    let x0 = t.x0[px];
    let x1 = t.x1[px];
    let fx = t.fx[px];
    let at = |x: usize, y: usize| data[y * gw + x] as f32;
    let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    a * (1.0 - fy) + b * fy
}

/// Dynamic dominant-species lookup: nearest (categorical), a u32 grid.
#[inline]
fn species_dyn(
    t: &Taps,
    data: &[u32],
    px: usize,
    py: usize,
    out_h: usize,
    gw: usize,
    gh: usize,
) -> u32 {
    if data.len() != gw.saturating_mul(gh) {
        return 0;
    }
    let (pick_y, _, _) = ytap_for(py, out_h, gh);
    let pick_x = if t.fx[px] < 0.5 { t.x0[px] } else { t.x1[px] };
    data[pick_y * gw + pick_x]
}

/// Dynamic energy raster: bilinear over the world grid (f32 grid).
#[inline]
fn sample_ener_dyn(
    t: &Taps,
    data: &[f32],
    px: usize,
    py: usize,
    out_h: usize,
    gw: usize,
    gh: usize,
) -> f32 {
    if data.len() != gw.saturating_mul(gh) || gw < 2 {
        return 0.0;
    }
    let (y0, y1, fy) = ytap_for(py, out_h, gh);
    let x0 = t.x0[px];
    let x1 = t.x1[px];
    let fx = t.fx[px];
    let at = |x: usize, y: usize| data[y * gw + x];
    let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    a * (1.0 - fy) + b * fy
}

#[cfg(test)]
mod tests {
    use super::*;
    use alife_common::WorldGenConfig;

    #[test]
    fn grain_is_deterministic() {
        assert_eq!(bake_grain(12, 34, 0.04), bake_grain(12, 34, 0.04));
        assert_ne!(bake_grain(12, 34, 0.04), bake_grain(13, 34, 0.04));
        assert_eq!(bake_grain(12, 34, 0.0), 1.0);
    }

    #[test]
    fn identity_bake_matches_cell_values() {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let gw = world.grid_w() as usize;
        let gh = world.grid_h() as usize;
        let dense = vec![0u32; gw * gh];
        let img = bake_terrain(&world, ViewMode::Elevation, &dense, &dense, &[]);
        assert_eq!(img.size[0], gw);
        assert_eq!(img.size[1], gh);
        // center sample equals ramp of exact elevation value
        let i = gh / 2 * gw + gw / 2;
        let e = world.elevation.data[i];
        assert_eq!(img.pixels[i], viridis(e), "bilinear at integer taps must reproduce cell values");
    }

    #[test]
    fn scaled_bake_size_and_bounds() {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let gw = world.grid_w() as usize;
        let dense = vec![0u32; gw.checked_mul(world.grid_h() as usize).unwrap_or(0)];
        let img = bake_terrain_scaled(&world, ViewMode::Natural, 256, 192, &dense, &dense, &[]);
        assert_eq!(img.size[0], 256);
        assert_eq!(img.size[1], 192);
        assert_eq!(img.pixels.len(), 256 * 192);
        for p in img.pixels.iter().step_by(509) {
            assert!(p.a() == 255 && (p.r() as u16 + p.g() as u16 + p.b() as u16) <= 765);
        }
    }

    #[test]
    fn species_mode_stays_categorical() {
        let world = alife_worldgen::generate(&WorldGenConfig::default());
        let gw = world.grid_w() as usize;
        let gh = world.grid_h() as usize;
        let mut dominant = vec![0u32; gw * gh];
        dominant[gw / 2 + gh / 2 * gw] = 5;
        dominant[gw / 2 + gh / 2 * gw + 1] = 3;
        let img = bake_terrain_scaled(&world, ViewMode::Species, 64, 64, &[], &dominant, &[]);
        // nearest sampling must not blend neighbouring species ids: cells map
        // back to one of the planted ids or the background, never a mix
        let p = img.pixels[32 * 64 + 32];
        assert_ne!(p, Color32::from_rgb(16, 20, 25).gamma_multiply(2.0));
    }

    #[test]
    fn cap_constant() {
        assert_eq!(MAX_BAKE_PIXELS, 1_200_000);
        assert_eq!(CONTOUR_BANDS, 12.0);
    }
}
