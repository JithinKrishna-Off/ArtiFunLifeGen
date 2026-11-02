//! grass_lint — OFFLINE math harness (cfg(test)).
//!
//! Re-implements the grass VS transform math in Rust (same constants, same
//! table order) and runs it over REAL `place::build` instances from the
//! worst-case habitat scenario, so pipeline-parameter bugs surface above
//! the waterline:
//!
//! * instance attribute table <-> `GrassInst` byte layout via `offset_of!`
//!   (the giant-pillar regression: scalars bound one field early when the
//!   vec2 consumed floats 0-1);
//! * world-space bounding box of blade 0 of every clump in the densest
//!   near chunk + whole-clump canopy bounds: blade height == inst.height
//!   (species law), blade width <= ~0.35 world units, canopy bounded by
//!   the spread law (pillars would read thousands of units);
//! * pad peak/tip profile sanity (tip fraction 0.10..0.16, never > 1);
//! * blade-major lattice/sub-draw alignment (0..24 far span must still
//!   index yaw/rad/hs tables consistently);
//! * root-row (v == 0) lateral symmetry — no off-by-one wall of blades.
//!
//! No GPU device needed: everything is pure f32 math.

use super::place::{self, GrassInst, MODE_ORBIT};
use super::{inst_attrs, GrassUniform};
use crate::grass_clump::{blades_per_clump, clump_vert_count, clump_verts};
use alife_worldgen::WorldModel;
use std::mem::{offset_of, size_of};

// ---------------------------------------------------------------- WGSL port

const TAU: f32 = 6.2831853;

/// calm-wind uniform constant mirrored from grass.rs `CALM_UNIFORM` for the
/// port: gust amplitude 0.35, trample radius -1.0 (fall == 0), vt clock 0.
const CALM_GUST: f32 = 0.35;

/// port of WGSL `hash_blend` (grass.rs VS) — same integer mix, same fract
fn hash_blend(seed: f32, k: u32) -> f32 {
    let mut x = seed.to_bits() ^ k.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B);
    x ^= x >> 16;
    (x as f32 * 43758.5453).fract()
}

/// per-blade const tables — MUST mirror the WGSL `yaw_tbl`/`rad_tbl`/`hs_tbl`
const YAW_TBL: [f32; 7] = [
    0.0000000, 2.0943951, 4.1887902, 1.0471976, 3.1415927, 5.2359878, 0.2617994,
];
const RAD_TBL: [f32; 7] = [0.00, 0.55, 0.55, 0.85, 0.85, 0.85, 0.50];
const HS_TBL: [f32; 7] = [1.00, 0.92, 0.92, 0.78, 0.78, 0.78, 0.84];

/// world position of one lattice vertex through the VS math (calm wind:
/// gust 0.35, no trample, deterministic clock 0.0); matches grass.rs `vs()`
/// line-for-line (trample bend == 0 at calm radius -1.0).
fn vs_world(inst: &GrassInst, bid: usize, lat: f32, v: f32) -> [f32; 3] {
    let h = inst.height.max(0.05);
    let clump_yaw = inst.lean_phase * TAU;
    // sub-draws always start at vertex 0, so the blade id is exactly the
    // blade-major slot — no drift vs the GPU path
    let k = bid as u32;
    let jyaw = (hash_blend(inst.seed, k * 4 + 0) * 2.0 - 1.0) * 0.2443461;
    let jrad = (hash_blend(inst.seed, k * 4 + 1) * 2.0 - 1.0) * 0.08;
    let jhs = (hash_blend(inst.seed, k * 4 + 2) * 2.0 - 1.0) * 0.07;
    let byaw = YAW_TBL[bid] + jyaw;
    let rk = (RAD_TBL[bid] + jrad).clamp(0.0, 1.0);
    let hs = (HS_TBL[bid] + jhs).max(0.3);
    let h_eff = (h * hs).max(0.05);
    let spread_r = h * (0.28 + 0.44 * (inst.pad + 0.5));
    let total_yaw = clump_yaw + byaw;
    let (rx, rz) = (total_yaw.cos(), total_yaw.sin());
    let ax = inst.x + rx * (rk * spread_r);
    let az = inst.z + rz * (rk * spread_r);
    let (sx, sz) = (total_yaw.cos(), total_yaw.sin());

    // width profile (pad in [-0.5, 0.5] keeps tip in [0.10, 0.16])
    let peak = 0.24 + 0.16 * (inst.pad + 0.5);
    let tip = 0.10 + 0.12 * (0.5 - inst.pad.abs());
    let prof = if v <= peak {
        (0.68 + (1.0 - 0.68) * (v / peak.max(1e-3))).clamp(0.0, 1.0)
    } else {
        let dp = ((v - peak) / (1.0 - peak).max(1e-3)).clamp(0.0, 1.0);
        (1.0 + (tip - 1.0) * dp.powf(1.35)).clamp(0.0, 1.0)
    };
    let bx = ax + sx * (lat * prof * inst.width * 0.5);
    let bz = az + sz * (lat * prof * inst.width * 0.5);

    // calm wind wave + static terms (gust_a <= 0.35 * sway)
    let wave = (0.0 * 1.8 + (ax + az) * 0.05 + clump_yaw * 2.0).sin();
    let flut = (0.0 * 5.5 + bid as f32 * 4.7 + inst.seed * 11.0).sin();
    let gust_a = CALM_GUST * inst.sway;
    let mut bend = [
        0.86 * (h_eff * v * v) * (wave * gust_a * 0.55 + flut * gust_a * 0.04),
        0.51 * (h_eff * v * v) * (wave * gust_a * 0.55 + flut * gust_a * 0.04),
    ];
    bend[0] += rx * (spread_r * 0.15 * v * (1.0 + 0.3 * jrad.abs()));
    bend[1] += rz * (spread_r * 0.15 * v * (1.0 + 0.3 * jrad.abs()));
    let curl = inst.pad;
    let lean_d = [
        (clump_yaw * 0.9 + inst.seed * 5.0).sin(),
        (clump_yaw * 0.9 + inst.seed * 3.0).cos(),
    ];
    bend[0] += lean_d[0] * ((0.09 + 0.05 * curl.abs()) * h_eff * v * v);
    bend[1] += lean_d[1] * ((0.09 + 0.05 * curl.abs()) * h_eff * v * v);
    bend[0] += (-sz) * (curl * 0.08 * h_eff * v * v);
    bend[1] += sx * (curl * 0.08 * h_eff * v * v);
    let blen = (bend[0] * bend[0] + bend[1] * bend[1]).sqrt();
    let (b0, b1) = if blen > h_eff * 0.9 {
        (bend[0] * (h_eff * 0.9 / blen), bend[1] * (h_eff * 0.9 / blen))
    } else {
        (bend[0], bend[1])
    };
    // world y = ground contact + blade rise (droop only lowers it)
    [bx + b0, inst.gy + v * h_eff - 0.35 * blen * v, bz + b1]
}

/// (world min, world max) over every blade + lattice vert of one clump
fn vs_bbox(inst: &GrassInst, verts: &[[f32; 2]], only_blade0: bool) -> ([f32; 3], [f32; 3]) {
    let blade_n = blades_per_clump();
    let per = clump_vert_count() / blade_n.max(1);
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for b in 0..blade_n {
        if only_blade0 && b != 0 {
            continue;
        }
        for vert in &verts[b * per..(b + 1) * per] {
            let w = vs_world(inst, b, vert[0], vert[1]);
            for i in 0..3 {
                mn[i] = mn[i].min(w[i]);
                mx[i] = mx[i].max(w[i]);
            }
        }
    }
    (mn, mx)
}

/// worst-case world-space canopy reach (xz distance from the clump anchor)
fn canopy_reach(inst: &GrassInst, verts: &[[f32; 2]]) -> f32 {
    let (mn, mx) = vs_bbox(inst, verts, false);
    [(mn[0], mn[2]), (mx[0], mx[2]), (mn[0], mx[2]), (mx[0], mn[2])]
        .iter()
        .map(|(px, pz)| (px - inst.x).hypot(pz - inst.z))
        .fold(0.0_f32, f32::max)
}

/// worst-case FULL lateral world width of blade 0 (lattice |lat| x profile
/// fraction, times inst.width — this isolates the width stream; droop bends
/// shift x/z and would pollute a naive bbox extent)
fn blade0_world_width(inst: &GrassInst, verts: &[[f32; 2]]) -> f32 {
    let per = clump_vert_count() / blades_per_clump().max(1);
    verts[..per]
        .iter()
        .map(|vert| {
            let peak = 0.24 + 0.16 * (inst.pad + 0.5);
            let tip = 0.10 + 0.12 * (0.5 - inst.pad.abs());
            let v = vert[1];
            let prof = if v <= peak {
                (0.68 + (1.0 - 0.68) * (v / peak.max(1e-3))).clamp(0.0, 1.0)
            } else {
                let dp = ((v - peak) / (1.0 - peak).max(1e-3)).clamp(0.0, 1.0);
                (1.0 + (tip - 1.0) * dp.powf(1.35)).clamp(0.0, 1.0)
            };
            vert[0].abs() * prof
        })
        .fold(0.0_f32, f32::max)
        * inst.width
}

fn default_world() -> WorldModel {
    alife_worldgen::generate(&alife_worldgen::WorldGenConfig::default())
}

/// worst-case near chunk: builds the habitat placement over a coarse focus
/// scan and picks the single densest FULL-density (near ring) chunk
fn densest_near_chunk(world: &WorldModel) -> Vec<GrassInst> {
    let mut best: Vec<GrassInst> = Vec::new();
    for (fx, fz) in [
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.6, 0.4),
        (0.9, 700.0 / 2000.0),
    ]
    .map(|(a, b)| (a * world.width, b * world.height))
    {
        let res = place::build(
            world,
            &place::Builder {
                gen: world.generation_tag,
                cell: ((fx / 32.0) as i32, (fz / 32.0) as i32),
                mode: MODE_ORBIT,
                focus: (fx, fz),
                lod_scale: 1.0,
            },
        );
        let (near, _mid) = place::rings_for(MODE_ORBIT, place::window_for(MODE_ORBIT, world));
        // concentrate on the densest FULL-density chunk of this focus
        let mut near_chunk: Option<Vec<GrassInst>> = None;
        for c in &res.chunks {
            let d = (c.cx as f32 * 32.0 - fx).hypot(c.cy as f32 * 32.0 - fz);
            if d >= near {
                continue;
            }
            if near_chunk.as_ref().map(|b| b.len() < c.insts.len()).unwrap_or(true) {
                near_chunk = Some(c.insts.clone());
            }
        }
        if let Some(chunk) = near_chunk {
            if chunk.len() > best.len() {
                best = chunk;
            }
        }
    }
    best
}

// ----------------------------------------------------------------- tests

#[test]
fn inst_attrs_pinned_to_struct_layout() {
    // scalar fields k = 2..=12: gy, height, width, hue, lean, biome, dim,
    // seed, wet, sway, pad — bound at their NATURAL byte offsets and
    // locations k (loc 1 is the (x, z) vec2 at byte 0).
    let want: [(u32, u64, &str); 11] = [
        (2, 8, "gy"),
        (3, 12, "height"),
        (4, 16, "width"),
        (5, 20, "tint_hue"),
        (6, 24, "lean_phase"),
        (7, 28, "biome"),
        (8, 32, "dim"),
        (9, 36, "seed"),
        (10, 40, "wet"),
        (11, 44, "sway"),
        (12, 48, "pad"),
    ];
    let attrs = inst_attrs();
    assert_eq!(attrs.len(), 12, "vec2 + 11 scalars must be declared");
    assert_eq!(attrs[0].offset, 0, "xz at byte 0");
    assert_eq!(attrs[0].shader_location, 1);
    for (loc, off, name) in want.iter() {
        let a = &attrs[(*loc - 1) as usize];
        assert_eq!(a.shader_location, *loc, "location drift on {name}");
        assert_eq!(a.offset, *off, "byte offset drift on {name}");
    }
    // pin each scalar to its struct field via offset_of! (the ROOT-CAUSE
    // regression guard: an off-by-one binding = giant/floating geometry)
    assert_eq!(offset_of!(GrassInst, gy), 8);
    assert_eq!(offset_of!(GrassInst, height), 12);
    assert_eq!(offset_of!(GrassInst, width), 16);
    assert_eq!(offset_of!(GrassInst, tint_hue), 20);
    assert_eq!(offset_of!(GrassInst, lean_phase), 24);
    assert_eq!(offset_of!(GrassInst, biome), 28);
    assert_eq!(offset_of!(GrassInst, dim), 32);
    assert_eq!(offset_of!(GrassInst, seed), 36);
    assert_eq!(offset_of!(GrassInst, wet), 40);
    assert_eq!(offset_of!(GrassInst, sway), 44);
    assert_eq!(offset_of!(GrassInst, pad), 48);
    assert_eq!(size_of::<GrassInst>(), 52, "13 x f32 instance record");
    // the shared instance/attribute stride must match the record size
    assert_eq!(size_of::<GrassUniform>(), 64, "4 x vec4 std140");
}

#[test]
fn profile_tip_and_peak_stay_in_bounds() {
    // mission suspect (d): pad in [-0.5, 0.5] must give tip 0.10..0.16 and
    // peak 0.24..0.40 — never a profile fraction > 1 (giant wide blades)
    for i in 0..=50 {
        let pad = i as f32 / 50.0 - 0.5;
        let peak = 0.24 + 0.16 * (pad + 0.5);
        let tip = 0.10 + 0.12 * (0.5 - pad.abs());
        assert!((0.24..=0.40).contains(&peak), "peak drift {peak} at pad {pad}");
        assert!((0.10..=0.16).contains(&tip), "tip drift {tip} at pad {pad}");
        prof_law(peak, tip, pad);
    }
    // real instances too
    let world = default_world();
    for inst in densest_near_chunk(&world) {
        let peak = 0.24 + 0.16 * (inst.pad + 0.5);
        let tip = 0.10 + 0.12 * (0.5 - inst.pad.abs());
        assert!((0.24..=0.40).contains(&peak));
        assert!((0.10..=0.16).contains(&tip));
        prof_law(peak, tip, inst.pad);
    }
}

fn prof_law(peak: f32, tip: f32, _pad: f32) {
    for i in 0..=40 {
        let v = i as f32 / 40.0;
        let prof = if v <= peak {
            0.68 + 0.32 * (v / peak.max(1e-3))
        } else {
            let dp = ((v - peak) / (1.0 - peak).max(1e-3)).clamp(0.0, 1.0);
            1.0 + (tip - 1.0) * dp.powf(1.35)
        };
        assert!(
            prof <= 1.0001 && prof >= 0.0,
            "profile fraction {prof} out of [0,1] (would widen blades)"
        );
    }
}

/// mission step-1 main bound: the worst-case near chunk's blade geometry
/// must equal the instance contract — NOT thousands-of-units pillars
#[test]
fn worst_near_chunk_world_bbox_bounded() {
    let world = default_world();
    let chunk = densest_near_chunk(&world);
    assert!(
        chunk.len() >= 40,
        "expected a probe window with a dense near chunk (got {} clumps)",
        chunk.len()
    );
    let verts = clump_verts();
    assert_eq!(verts.len(), clump_vert_count(), "lattice/vert_count drift");
    let mut worst_reach = 0.0_f32;
    let mut worst_width = 0.0_f32;
    for inst in &chunk {
        // species height law: base/moss/rocky/shore tufts 0.4..1.4, sward
        // 0.7..3.2, dry tufts 2.2..3.6, meadow outlyers 3.4..5.0, shrubs
        // 3.4..5.8
        assert!(
            (0.40..=5.85).contains(&inst.height),
            "inst.height off species law: {}",
            inst.height
        );
        // (0) blade width driven by inst.width, never by height/gy
        let full_w = blade0_world_width(inst, &verts);
        assert!(
            full_w <= 0.35,
            "blade-0 world width {full_w:.4} > 0.35 (width stream drift)"
        );
        assert!(
            full_w <= inst.width * 1.001,
            "blade-0 width {full_w:.4} exceeds its inst.width {}",
            inst.width
        );
        worst_width = worst_width.max(full_w);
        // (1) blade-0 geometry height == inst.height (rise law: hs >= 0.71,
        // droop -0.35 * |bend| keeps the top below 1.08x)
        let (bm0, mx0) = vs_bbox(inst, &verts, true);
        let rise1 = mx0[1] - inst.gy;
        assert!(
            rise1 <= inst.height * 1.08 + 1e-3,
            "blade towering above inst.height: {rise1:.2} vs {}",
            inst.height
        );
        assert!(
            rise1 >= inst.height * 0.60,
            "blade collapsed below inst.height: {rise1:.2} vs {}",
            inst.height
        );
        assert!(
            bm0[1] >= inst.gy - 1e-3,
            "root row dipped below ground contact: {:.3}",
            bm0[1]
        );
        // (2) clump canopy from the anchor: spread law + calm bend only
        let reach = canopy_reach(inst, &verts);
        assert!(
            reach <= inst.height * 1.3 + 0.75,
            "canopy reach {reach:.2} unbounded vs h {} (pillar signature)",
            inst.height
        );
        assert!(
            reach <= 12.0,
            "worst-case canopy reach {reach:.2} — pillar scale"
        );
        worst_reach = worst_reach.max(reach);
        for w in [bm0, mx0].iter().flat_map(|m| m.iter()) {
            assert!(w.is_finite(), "non-finite VS output {w}");
        }
        // (3) nothing sinks meaningfully below ground contact
        let (mn, _mx) = vs_bbox(inst, &verts, false);
        assert!(
            mn[1] >= inst.gy - 0.05,
            "blades sunk below ground: {:.2}",
            mn[1] - inst.gy
        );
    }
    // regression telescopes (measured laws, not arbitrary multipliers)
    assert!(
        worst_reach <= 4.2,
        "densest-chunk worst canopy reach {worst_reach:.2} regressed past the ~4 clamp law"
    );
    assert!(
        worst_width <= 0.34,
        "densest-chunk worst blade width {worst_width:.4} regressed past the 0.35 law"
    );
}

/// mission suspect (c): sub-draw spans must be blade-boundary multiples of
/// the per-blade lattice so the yaw/rad/hs tables place each drawn blade
/// consistently (far 0..24 = blade 0 slot exactly)
#[test]
fn sub_draw_spans_align_blade_lattice() {
    let verts = clump_verts();
    let blades = blades_per_clump() as u32;
    let per = (verts.len() as u32 / blades.max(1)).max(1);
    assert_eq!(verts.len() as u32, blades * per, "blade-major packing");
    assert_eq!(verts.len() as u32, 168, "7 x 24 contract");
    let full = blades * per;
    let mid = (4.min(blades) * per).min(full);
    let far = (1.min(blades) * per).min(full);
    assert_eq!((far, mid, full), (24, 96, 168));
    // drawn spans all start at blade 0 => table entries stay consistent
    for span in [far, mid, full] {
        assert_eq!(span % per, 0, "span {span} breaks the 24-vert blade grid");
        // each drawn blade occupies [k*per, (k+1)*per) with v in [0, 1]
        for b in 0..(span / per) {
            let slice = &verts[(b * per) as usize..((b + 1) * per) as usize];
            for vert in slice {
                assert!(
                    (0.0..=1.0).contains(&vert[1]),
                    "blade {b} lattice v out of unit range: {}",
                    vert[1]
                );
            }
        }
    }
}

/// mission suspect (e): at v == 0 the warp must stay symmetric (skew and
/// mid bend are 0 there) so root rows straddle the anchor instead of
/// building a one-sided wall of blades
#[test]
fn root_rows_symmetric_no_v0_wall() {
    let verts = clump_verts();
    let per = 24;
    for b in 0..7 {
        let slice = &verts[b * per..(b + 1) * per];
        // the ROOT row (v == 0, the first quad's a/b pair): every blade spec
        // has zero warp there (skew is v^2, mid is 4v(1-v)), so the root row
        // must straddle the anchor symmetrically — no one-sided wall
        let roots: Vec<f32> = slice.iter().filter(|s| s[1] < 1e-6).map(|s| s[0]).collect();
        assert!(roots.len() >= 2, "blade {b} missing its v==0 edge pair");
        let lo = roots.iter().copied().fold(f32::MAX, f32::min);
        let hi = roots.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            (lo + hi).abs() < 1e-4,
            "blade {b} root asymmetry {lo:.3}..{hi:.3}"
        );
        assert!(hi <= 1.45, "blade {b} lattice lateral {hi} unbounded");
        assert!(lo >= -1.45, "blade {b} lattice lateral {lo} unbounded");
    }
}

/// mission suspect (f): the gust uniform block stays std140-sized and the
/// winding fields it feeds are clamped CPU-side so the VS gust_a can never
/// wrap the bend clamp
#[test]
fn gust_uniform_fields_bounded() {
    assert_eq!(size_of::<GrassUniform>(), 64, "4 x vec4 std140");
    // wind dir is unit-ish: |d| <= 1.1 never grows with state
    let d = (0.86_f32 * 0.86 + 0.51_f32 * 0.51).sqrt();
    assert!(d <= 1.1, "wind dir magnitude {d}");
    // gust target clamp (update_gpu): [0.30, 1.6]; VS clamps to [0, 2.5]
    assert!(1.6 <= 2.5, "VS clamp does not dominate the CPU-side ceiling");
    // sway species law keeps gust_a <= 2.5 * 1.0
    for s in [1.0_f32, 0.9, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.25] {
        assert!(s <= 1.0 && s >= 0.0, "sway law drift {s}");
    }
}

/// harness self-check: the WGSL-port must agree with the shipped lattice in
/// its own binary range (lateral max 1.4) — catches grass_clump drift
#[test]
fn lattice_binary_contract_holds() {
    let verts = clump_verts();
    assert_eq!(verts.len(), 168);
    for (i, vert) in verts.iter().enumerate() {
        assert!(
            vert[0].abs() <= 1.4 && (0.0..=1.0).contains(&vert[1]),
            "lattice vertex {i} drifted: {:?}",
            vert
        );
    }
}

/// tuning probe (run explicitly): cargo test -p alife-view3d probe_grass_geometry -- --ignored --nocapture
/// prints the ACTUAL worst-case geometry laws over the densest near chunk
/// so regression bounds stay measurable facts, not guesses.
#[test]
#[ignore]
fn probe_grass_geometry() {
    let world = default_world();
    let chunk = densest_near_chunk(&world);
    println!("densest near chunk: {} clumps", chunk.len());
    let verts = clump_verts();
    let mut hmin = f32::INFINITY;
    let mut hmax = f32::NEG_INFINITY;
    let mut rise_ratio = (f32::INFINITY, f32::NEG_INFINITY);
    let mut reach_max = 0.0_f32;
    let mut width_max = 0.0_f32;
    for inst in &chunk {
        let (_bm0, mx0) = vs_bbox(inst, &verts, true);
        let rise = (mx0[1] - inst.gy) / inst.height;
        rise_ratio = (rise_ratio.0.min(rise), rise_ratio.1.max(rise));
        hmin = hmin.min(inst.height);
        hmax = hmax.max(inst.height);
        reach_max = reach_max.max(canopy_reach(inst, &verts));
        width_max = width_max.max(blade0_world_width(inst, &verts));
    }
    println!(
        "inst.height law [hmin {hmin:.3} .. hmax {hmax:.3}]\nblade rise/h ratio [{:.3} .. {:.3}]\nworst canopy reach {:.3}\nworst blade-0 width {:.4}",
        rise_ratio.0, rise_ratio.1, reach_max, width_max
    );
}
