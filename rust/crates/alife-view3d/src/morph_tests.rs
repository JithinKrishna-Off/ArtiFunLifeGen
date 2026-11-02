//! Pure-function tests for morphology decomposition, cull caps, template
//! generation and (GPU-permitting) shader compilation. Zero-alloc guarantees
//! of the live planner are structurally enforced (all scratch buffers are
//! pre-allocated in `Planner::new`), so tests focus on caps and determinism.

use super::decomp::{self, Planner, MODEL_CAP, QUAD_CAP};
use super::templates;
use super::*;
use crate::scene::OrgInst;
use bytemuck::Zeroable;

fn inst_at(x: f32, y: f32, brand: u32, hue: f32, size: f32) -> OrgInst {
    let mut o = OrgInst::zeroed();
    o.x = x;
    o.y = y;
    o.brand = brand;
    o.hue = hue;
    o.energy = 0.8;
    o.age = 0.2;
    o.size = size;
    o.alive = 1.0;
    o
}

// ----------------------------------------------------- signature determinism

#[test]
fn signature_is_deterministic_and_buckets_differ() {
    let a = inst_at(10.0, -5.0, BRAND_CELL, 0.37, 0.9);
    let b = inst_at(300.0, 800.0, BRAND_CELL, 0.37 + 1.0, 0.9); // hue wraps
    assert_eq!(
        decomp::signature(&a),
        decomp::signature(&b),
        "same brand/hue/size must give the same signature regardless of world pos/energy/age"
    );
    assert_eq!(decomp::signature(&a), decomp::signature(&a), "stable on repeat calls");

    let s0 = decomp::signature(&a);
    let mut d = a;
    d.brand = BRAND_FILAMENT;
    assert_ne!(s0, decomp::signature(&d), "brand bucket must act");
    d = a;
    d.hue = a.hue + 0.5;
    assert_ne!(s0, decomp::signature(&d), "hue bucket must act");
    d = a;
    d.size = a.size + 0.5;
    assert_ne!(s0, decomp::signature(&d), "size bucket must act");
}

// ----------------------------------------------------- brand mapping

fn organism_with(t: alife_common::Traits) -> alife_common::Organism {
    alife_common::Organism {
        id: 1,
        x: 0.0,
        y: 0.0,
        dir: 0.0,
        speed: 0.0,
        energy: 50.0,
        max_energy: 100.0,
        age: 10,
        generation: 0,
        genotype: alife_common::Genotype {
            net: alife_common::Network::zeros(),
            traits: t,
        },
        lineage: 1,
        mom: None,
        dad: None,
        offspring_count: 0,
        alive: true,
        repr_cooldown: 0,
        last_inputs: Vec::new(),
        last_outputs: Vec::new(),
    }
}

#[test]
fn brand_mapping_matches_trait_tables() {
    // small bodies -> plain cell
    let small = organism_with(alife_common::Traits {
        size: 0.65,
        max_speed: 40.0,
        efficiency: 0.6,
        sensor_range: 60.0,
        metabolism: 0.6,
        lifespan: 2400,
        repr_threshold: 0.5,
        repr_cost: 0.2,
    });
    assert_eq!(brand_of(&small), BRAND_CELL);
    // fast bodies -> flagellate
    let fast = organism_with(alife_common::Traits {
        max_speed: 150.0,
        size: 1.0,
        ..small.genotype.traits
    });
    assert_eq!(brand_of(&fast), BRAND_FLAGELLATE);
    // big slow bodies -> filament chain
    let big = organism_with(alife_common::Traits {
        size: 1.5,
        max_speed: 50.0,
        ..small.genotype.traits
    });
    assert_eq!(brand_of(&big), BRAND_FILAMENT);
    // sessile long-sensers -> colony cluster
    let sessile = organism_with(alife_common::Traits {
        size: 1.0,
        max_speed: 30.0,
        efficiency: 0.6,
        sensor_range: 200.0,
        ..small.genotype.traits
    });
    assert_eq!(brand_of(&sessile), BRAND_COLONY);
}

#[test]
fn anatomy_is_bounded_and_trait_gated() {
    // flagellate grows 1-2 tails
    let mut o = inst_at(0.0, 0.0, BRAND_FLAGELLATE, 0.31, 1.0);
    let b = decomp::org_build(&o);
    assert_eq!(b.shells.len(), 1);
    let tails = b.tail.unwrap().1;
    assert!((1..=2).contains(&tails), "flagella 1..2, got {}", tails);
    o.hue = (o.hue + 0.5) % 1.0; // variant hue may flip the count, never the bound
    let tails = decomp::org_build(&o).tail.unwrap().1;
    assert!((1..=2).contains(&tails));

    // colony fuses 2-3 sub-cells, no tail
    o = inst_at(0.0, 0.0, BRAND_COLONY, 0.13, 1.0);
    let b = decomp::org_build(&o);
    assert!(b.tail.is_none());
    assert!((2..=3).contains(&b.shells.len()), "colony 2..3 cells, got {}", b.shells.len());

    // filament chains 3-5 segments
    o = inst_at(0.0, 0.0, BRAND_FILAMENT, 0.62, 1.0);
    let b = decomp::org_build(&o);
    assert!(b.tail.is_none());
    assert!((3..=5).contains(&b.shells.len()), "filament 3..5 segments, got {}", b.shells.len());

    // nucleus appears only for large bodies (trait size > 1.1)
    o = inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.4);
    let b = decomp::org_build(&o);
    assert_eq!(b.shells[0].flags, 1.0, "large cell expresses nucleus");
    o.size = 0.9;
    let b = decomp::org_build(&o);
    assert_eq!(b.shells[0].flags, 0.0, "small cell has no nucleus");
}

// ----------------------------------------------------- turgor / instance bounds

#[test]
fn instance_fields_stay_in_biological_bounds() {
    let orgs: Vec<OrgInst> = (0..256)
        .map(|i| {
            inst_at(
                50.0 + (i % 16) as f32 * 8.0,
                50.0 + (i / 16) as f32 * 8.0,
                i as u32 % 4,
                (i % 32) as f32 / 32.0,
                0.4 + (i % 7) as f32 * 0.2,
            )
        })
        .collect();
    let plan = decomp::plan_frame(
        &orgs,
        [50.0, 20.0, 50.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    for i in &plan.insts {
        assert!((0.35..=1.0).contains(&i.turgor), "turgor in 0.35..1: {}", i.turgor);
        assert!((0.0..=1.0).contains(&i.hue.rem_euclid(1.0)));
        assert!((0.0..=1.0).contains(&i.age), "age normalized");
        assert!((0.0..=1.0).contains(&i.alpha), "alpha 0..1");
        assert!(i.radius > 0.0, "positive body radius");
    }
}

// ----------------------------------------------------- cull caps (60 / 240)

#[test]
fn cull_caps_full_morph_60_billboards_240() {
    // worst case: 1600 organisms clustered inside the full-morph ring
    let orgs: Vec<OrgInst> = (0..1600)
        .map(|i| {
            inst_at(
                400.0 + (i % 40) as f32 * 0.4,
                400.0 + (i / 40) as f32 * 0.4,
                BRAND_CELL,
                (i % 16) as f32 / 16.0,
                0.9,
            )
        })
        .collect();
    let mut p = Planner::new();
    let plan = p.plan(
        &orgs,
        [400.0, 20.0, 400.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    // plain cells are one shell each -> shell_n counts full-morph organisms
    assert!(plan.shell_n <= MODEL_CAP, "full-morph cap 60, got {}", plan.shell_n);
    assert_eq!(plan.shell_n, MODEL_CAP, "ring saturated with full morphs");
    // overflow outside the model cap falls back to billboards (max 240)
    assert_eq!(plan.quad_n, QUAD_CAP, "billboard cap saturated at 240, got {}", plan.quad_n);
    assert_eq!(plan.tail_n, 0);
    let total = (plan.tail_n + plan.shell_n + plan.quad_n) as usize;
    assert!(total <= decomp::INST_CAP, "instance cap breached: {}", total);
    assert_eq!(total, plan.insts.len());
}

#[test]
fn distance_lod_beyond_billboard_radius_draws_nothing() {
    let orgs: Vec<OrgInst> = (0..64)
        .map(|i| inst_at(20_000.0 + i as f32, 20_000.0, BRAND_COLONY, 0.2, 1.0))
        .collect();
    let plan = decomp::plan_frame(
        &orgs,
        [0.0, 20.0, 0.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    assert!(plan.insts.is_empty(), "beyond quad radius nothing is drawn");
}

#[test]
fn dead_organisms_are_skipped_by_the_planner() {
    let mut orgs: Vec<OrgInst> = (0..16)
        .map(|i| inst_at(10.0 + i as f32, 0.0, BRAND_CELL, 0.1, 1.0))
        .collect();
    orgs[0].alive = 0.0;
    let plan = decomp::plan_frame(
        &orgs,
        [10.0, 20.0, 0.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    let represented = plan.shell_n + plan.quad_n;
    assert_eq!(represented, 15, "corpse fades out entirely");
}

// ----------------------------------------------------- planner warm path

#[test]
fn planner_scratch_buffers_survive_frames() {
    // two consecutive plans must both run and agree (deterministic replan)
    let orgs: Vec<OrgInst> = (0..300)
        .map(|i| inst_at(60.0 + (i % 20) as f32, 60.0, i as u32 % 4, 0.3, 1.0))
        .collect();
    let mut p = Planner::new();
    let a = p.plan(&orgs, [60.0, 20.0, 60.0], decomp::MODE_ORBIT, &mut |_, _| 0.0, &mut |_, _| true);
    assert!(!a.insts.is_empty());
    let ab: Vec<u8> = bytemuck::cast_slice::<decomp::MorphInst, u8>(&a.insts).to_vec();
    let b = {
        let n = p.plan(&orgs, [60.0, 20.0, 60.0], decomp::MODE_ORBIT, &mut |_, _| 0.0, &mut |_, _| true);
        bytemuck::cast_slice::<decomp::MorphInst, u8>(&n.insts).to_vec()
    };
    assert_eq!(ab, b, "deterministic replan from identical input");
}

// ----------------------------------------------------- template generation

#[test]
fn template_generation_is_deterministic() {
    let a = templates::build();
    let b = templates::build();
    let ba: &[u8] = bytemuck::cast_slice(&a.shell);
    let bb: &[u8] = bytemuck::cast_slice(&b.shell);
    assert_eq!(ba, bb, "shell soup bit-identical across calls");
    let ta: &[u8] = bytemuck::cast_slice(&a.tail);
    let tb: &[u8] = bytemuck::cast_slice(&b.tail);
    assert_eq!(ta, tb, "flagella soup bit-identical across calls");
    // vertex budgets match the prebaked ranges used by draw_with
    assert_eq!(a.shell.len() as u32, templates::SHELL_VCOUNT);
    assert_eq!(a.tail.len() as u32, templates::TAIL_VCOUNT);
    assert_eq!(a.quad.len() as u32, templates::QUAD_VCOUNT);
}

// ----------------------------------------------------- queen style hook

#[test]
fn queen_style_boosts_same_templates() {
    let queen = style_for(true);
    let plain = style_for(false);
    assert!(queen.hue_shift > 0.0 && queen.size_scale > plain.size_scale);
    assert_eq!(plain, MorphStyle { hue_shift: 0.0, size_scale: 1.0, breathe_scale: 1.0 });
    let mut o = inst_at(0.0, 0.0, BRAND_CELL, 0.4, 1.0);
    let before = o.hue;
    apply_style(&mut o, true);
    assert_ne!(o.hue, before, "queen hue shifted");
    assert!(o.size > 1.0, "queen body emphasized");
    apply_style(&mut o, false);
}

// ----------------------------------------------------- ordinary tiny scale

/// Hull length in world units of a planned single-organism silhouette along
/// its heading (+X at heading 0): sub-shells contribute their template
/// sphere radius (the shader's <=1.27x ellipsoid/displacement stretch is the
/// breathing headroom), tails contribute their template reach (bezier chords
/// TAIL_A/TAIL_B end at local x = -2.60 body radii of the tail instance).
fn silhouette_len(plan: &decomp::FramePlan) -> f32 {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (k, i) in plan.insts.iter().enumerate() {
        if k < (plan.tail_n + plan.shell_n) as usize {
            if k < plan.tail_n as usize {
                lo = lo.min(i.pos_x - 2.6 * i.radius);
            } else {
                lo = lo.min(i.pos_x - i.radius);
                hi = hi.max(i.pos_x + i.radius);
            }
        }
    }
    (hi - lo).max(0.0)
}

#[test]
fn ordinary_organisms_read_tiny_nominal_lengths() {
    // typical snapshot: size 1.0, render-default energy 0.8 -> turgor 0.924;
    // every brand's nominal visible body length must sit in [0.8, 2.0]
    // (~1/12 of the 16-unit queen), before any style emphasis.
    for brand in [BRAND_CELL, BRAND_FLAGELLATE, BRAND_COLONY, BRAND_FILAMENT] {
        let o = inst_at(400.0, 400.0, brand, 0.31, 1.0);
        let plan = decomp::plan_frame(
            &[o],
            [400.0, 40.0, 400.0],
            decomp::MODE_ORBIT,
            &mut |_, _| 0.0,
            &mut |_, _| true,
        );
        assert_eq!(plan.queen_n, 0, "ordinary instance never plans a queen");
        let len = silhouette_len(&plan);
        assert!(
            (0.8..=2.0).contains(&len),
            "brand {} nominal length {} not in [0.8, 2.0]",
            brand,
            len
        );
    }
}

#[test]
fn ordinary_scale_stays_bounded_over_knobs_and_drain() {
    // full knob sweep (sizes x drained/full) keeps every brand's silhouette
    // inside the decoupled ordinary template family's working range
    for brand in [BRAND_CELL, BRAND_FLAGELLATE, BRAND_COLONY, BRAND_FILAMENT] {
        for size in [0.6_f32, 1.0, 1.5] {
            for e in [0.0_f32, 1.0] {
                let mut o = inst_at(400.0, 400.0, brand, 0.31, size);
                o.energy = e;
                let plan = decomp::plan_frame(
                    &[o],
                    [400.0, 40.0, 400.0],
                    decomp::MODE_ORBIT,
                    &mut |_, _| 0.0,
                    &mut |_, _| true,
                );
                let len = silhouette_len(&plan);
                assert!(
                    (0.4..=3.5).contains(&len),
                    "brand {} size {} energy {}: length {} outside [0.4, 3.5]",
                    brand,
                    size,
                    e,
                    len
                );
                // drained deflation is monotone: energy 0 never longer at
                // the same size
                if e == 0.0 {
                    let mut full = o;
                    full.energy = 1.0;
                    let p2 = decomp::plan_frame(
                        &[full],
                        [400.0, 40.0, 400.0],
                        decomp::MODE_ORBIT,
                        &mut |_, _| 0.0,
                        &mut |_, _| true,
                    );
                    assert!(len <= silhouette_len(&p2) + 1e-5, "drain deflates");
                }
            }
        }
    }
}

// ----------------------------------------------------- queen giant scale

fn queen_org() -> OrgInst {
    let mut q = inst_at(400.0, 400.0, BRAND_QUEEN, 0.21, 2.0);
    q.is_queen = 1;
    q
}

/// Giant rebalance: visible queen body length
/// (QUEEN_BODY_LEN * QUEEN_SCALE * 5.5 * size * turgor) must stay in the
/// 13..20 world-unit band while keeping its turgor (breathing) dependence.
#[test]
fn queen_visual_length_is_giant_but_bounded() {
    let plan = decomp::plan_frame(
        &[queen_org()],
        [400.0, 40.0, 400.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    assert_eq!(plan.queen_n, 1, "queen takes her reserved slot");
    let qi = &plan.insts[(plan.tail_n + plan.shell_n) as usize];
    let len = decomp::QUEEN_BODY_LEN * qi.radius;
    assert!((13.0..=20.0).contains(&len), "queen length {} in [13, 20]", len);
    // the planned radius still follows the scale chain exactly
    let tf = 0.62 + 0.38 * 0.8; // inst_at default energy
    let expect = decomp::QUEEN_SCALE * 5.5 * 2.0 * tf;
    assert!((qi.radius - expect).abs() < 1e-4, "scale chain intact");

    // drained queen breathes smaller but stays a giant silhouette
    let mut drained = queen_org();
    drained.energy = 0.0;
    let plan2 = decomp::plan_frame(
        &[drained],
        [400.0, 40.0, 400.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    let qi2 = &plan2.insts[(plan2.tail_n + plan2.shell_n) as usize];
    assert!(decomp::QUEEN_BODY_LEN * qi2.radius < len, "turgor dependence kept");
    // grounded: the belly sits at terrain height (lifter scale reads her Y)
    assert!(
        qi.pos_y > decomp::QUEEN_GROUND_LIFT * expect - 1e-4
            && qi.pos_y < decomp::QUEEN_GROUND_LIFT * expect + 1e-4,
        "queen origin sits at terrain + QUEEN_GROUND_LIFT * body_r"
    );
}

/// Grass-trample hook: the planner reports the nearest planned queen's
/// footprint (x, z, radius ~= 0.9 * visual length, strength 1.0), and None
/// with no queens.
#[test]
fn queen_footprint_reports_trample_field() {
    let mut q = queen_org();
    q.x = 120.0;
    q.y = 90.0;
    let mut p = Planner::new();
    let plan = p.plan(
        &[q],
        [120.0, 40.0, 90.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    assert_eq!(plan.queen_n, 1);
    let fp = p.queen_footprint().expect("footprint of the planned queen");
    let (x, z, radius, strength) = fp;
    assert!((x - 120.0).abs() < 1e-4 && (z - 90.0).abs() < 1e-4, "footprint at her x/z");
    let vlen = 14.786; // QUEEN_BODY_LEN * QUEEN_SCALE * 5.5 * 2.0 * tf(0.924)
    assert!((radius - 0.9 * vlen).abs() < 0.05, "radius ~= 0.9 * visual length, got {}", radius);
    assert!(radius > 10.0 && radius < 18.0, "trample circle ~= 14");
    assert!((strength - 1.0).abs() < 1e-6);

    let mut p0 = Planner::new();
    p0.plan(
        &[inst_at(0.0, 0.0, BRAND_CELL, 0.1, 1.0)],
        [0.0, 20.0, 0.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    assert!(p0.queen_footprint().is_none(), "no queen rendered -> no trample");
}

/// Heading smoothing converges along the shortest arc; gait wave rises
/// with movement speed (cached position delta / engine dt).
#[test]
fn queen_heading_smoothing_and_gait_speed() {
    let mut p = Planner::new();
    p.set_frame_dt(1.0 / 30.0);
    let mut q = queen_org();
    // two frames of engine heading +2.5 rad apart (shorter arc is negative)
    let seq = [0.0f32, 2.5];
    for (f, h) in seq.into_iter().enumerate() {
        q.heading = h;
        q.x = 400.0 + f as f32 * 12.0; // slow move -> some gait, no teleport
        let plan = p.plan(
            &[q],
            [400.0, 40.0, 400.0],
            decomp::MODE_ORBIT,
            &mut |_, _| 0.0,
            &mut |_, _| true,
        );
        assert_eq!(plan.queen_n, 1);
        let qi = &plan.insts[(plan.tail_n + plan.shell_n) as usize];
        match f {
            0 => assert_eq!(qi.heading, 0.0, "first frame snaps to engine heading"),
            1 => {
                // smoothed toward engine heading along the SHORTER arc accounts
                // for the wrap: here the shorter arc is +2.5 (>|pi/2| < pi)
                assert!(qi.heading > 0.0 && qi.heading < 2.5, "mid-smooth {}", qi.heading);
                // teleported-free move of 12 units over dt 1/30 => 360 u/s
                // -> gait fully engaged (wave at ceiling)
                assert_eq!(qi.wave, 3.6, "gait wave saturates for fast movement");
            }
            _ => {}
        }
    }
    // a standing queen keeps the idle wave (cached pos == current pos)
    q.x = 412.0;
    q.heading = 0.0;
    p.set_frame_dt(1.0 / 30.0);
    let plan = p.plan(&[q], [400.0, 40.0, 400.0], decomp::MODE_ORBIT, &mut |_, _| 0.0, &mut |_, _| true);
    let qi = &plan.insts[(plan.tail_n + plan.shell_n) as usize];
    let idle = qi.wave;
    assert!((idle - (1.1 + 0.5 * 0.8)).abs() < 1e-3, "idle wave only, got {}", idle);
}



/// WGSL validity smoke test; skipped silently when no GPU adapter exists
/// (e.g. CI sandboxes). wgpu panics inside create_shader_module on invalid
/// WGSL, so a pass means the morph vertex/fragment modules are valid.
/// Windows desktop target color is hardcoded-comment: Bgra8UnormSrgb
/// (readable swapchain format used by the coordinator).
#[test]
fn gpu_pipeline_smoke_wgsl_compiles() {
    if std::env::var("MORPH_SKIP_GPU").is_ok() {
        return;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) else {
        return;
    };
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .ok()
            .unwrap();
    let mut m = Morph::new(&device).unwrap();
    m.ensure_pipeline(
        &device,
        wgpu::TextureFormat::Bgra8UnormSrgb, // Windows swapchain format
        wgpu::TextureFormat::Depth32Float,
    );
    assert!(m.gpu.is_some());
}
