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

// ================================================== A2 organism archetype pass

use super::{BLOB_CAP, DETAILED_CAP, MID_RING, OrganismPass, organism_fallback_key,
            mesh_footprint, OrgPlan};
use crate::creature_mesh as cmet;

fn run_organisms(
    pass: &mut OrganismPass,
    orgs: &[OrgInst],
    keys: Option<&[u64]>,
    view: [f32; 3],
) -> OrgPlan {
    let mut plan = OrgPlan::new();
    pass.plan(
        orgs,
        keys,
        view,
        decomp::MODE_ORBIT,
        &mut plan,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    plan
}

/// Archetype grouping: rows are bucketed contiguously per key, keys ascend,
/// ranges cover every detailed row, and the fallback key mirrors the brand.
#[test]
fn organism_rows_group_by_archetype_contiguous() {
    let mut pass = OrganismPass::new();
    let orgs: Vec<OrgInst> = (0..6)
        .map(|i| {
            inst_at(
                10.0 + i as f32 * 6.0, // d ~ 12..21 units from view, all near
                -(10.0 + i as f32 * 5.0),
                [BRAND_CELL, BRAND_COLONY, BRAND_FLAGELLATE][i % 3],
                0.31,
                1.0,
            )
        })
        .collect();
    let plan = run_organisms(&mut pass, &orgs, None, [10.0, 20.0, 10.0]);
    assert_eq!(plan.arch.len(), 6, "every organism renders one instance");
    assert_eq!(plan.groups.len(), 3, "one group per distinct brand fallback");
    let mut prev_key = 0u64;
    let mut cursor = 0u32;
    for &(key, first, count) in &plan.groups {
        assert!(key > prev_key, "group keys ascend");
        prev_key = key;
        assert_eq!(first, cursor, "groups are contiguous");
        cursor += count;
        let brand = (key & 0x3) as u32;
        assert_eq!(key, organism_fallback_key(brand), "fallback key == brand");
        for row in &plan.arch[first as usize..(first + count) as usize] {
            // hue channel propagates (species hue == hue_of source)
            assert!((row.hue - 0.31).abs() < 1e-5, "hue propagated");
            assert!(row.radius > 0.0 && row.flags.abs() < 1e-5, "row sane");
            // sat_tint rides the pad channel for body-branch tinting
            assert!((row.pad - 0.3).abs() < 1e-6, "fallback sat_tint 0.3");
        }
    }
    assert_eq!(cursor as usize, plan.arch.len(), "ranges cover all rows");
    // keys were resolved to real (brand fallback) meshes
    let (_, meshes, _) = pass.diag();
    assert_eq!(meshes, 3, "one mesh per archetype in use");
}

/// A1 scale recipe honored exactly: radius = RENDER_SCALE * body_length
/// (fallback nominal) * size * turgor (0.62 + 0.38 * energy).
#[test]
fn organism_scale_recipe_matches_render_scale() {
    let mut pass = OrganismPass::new();
    let o = inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.0); // energy 0.8 -> tf 0.924
    let plan = run_organisms(&mut pass, &[o], None, [0.0, 20.0, 0.0]);
    assert_eq!(plan.arch.len(), 1);
    let row = &plan.arch[0];
    let bl = cmet::fallback_morph(0).body_length;
    let tf = 0.62 + 0.38 * 0.8;
    let want = cmet::RENDER_SCALE * bl * 1.0 * tf;
    assert!((row.radius - want).abs() < 1e-4, "radius {} != {}", row.radius, want);
    // drained cells deflate (existing energy modulation kept on top)
    let mut drained = inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.0);
    drained.energy = 0.0;
    let plan2 = run_organisms(&mut pass, &[drained], None, [0.0, 20.0, 0.0]);
    assert!(plan2.arch[0].radius < row.radius, "drain deflates");
    // size dependence kept
    let big = inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.6);
    let plan3 = run_organisms(&mut pass, &[big], None, [0.0, 20.0, 0.0]);
    assert!(plan3.arch[0].radius > row.radius, "size scales up");
    // ground seat: base y seats the mesh's lowest vertex at terrain height 0
    let key = organism_fallback_key(0);
    let mesh = pass.mesh_of(key).expect("cell mesh resolved");
    let fp = mesh_footprint(mesh);
    assert!(
        (plan.arch[0].pos_y - (-fp.min_y * row.radius)).abs() < 1e-4,
        "instance origin seats the lowest mesh vertex on the ground"
    );
    assert!(plan.arch[0].pos_y > 0.0, "hull lift positive for real meshes");
}

/// MorphCache invalidation: a trait change (here: brand change -> archetype
/// change) forces a rebuild; identical rows warm-reuse with zero rebuilds.
#[test]
fn morph_cache_rebuilds_on_key_change_and_diag_is_sane() {
    let mut pass = OrganismPass::new();
    let orgs: Vec<OrgInst> = (0..4)
        .map(|i| {
            let mut o = inst_at(10.0 + i as f32 * 6.0, 0.0, i as u32 % 2, 0.31, 1.0);
            o.id = i as u64; // unique sim ids: one locomotion track each
            o
        })
        .collect();
    let a = run_organisms(&mut pass, &orgs, None, [10.0, 20.0, 0.0]);
    let (entries, meshes, rebuilds) = pass.diag();
    assert!(entries > 0 && meshes > 0 && rebuilds > 0, "first pass builds cache");
    assert_eq!(entries, 2, "one entry per distinct signature");
    assert_eq!(meshes, entries, "one mesh per archetype in use");
    // warm identical pass: zero rebuilds, same bytes out
    let a_wrap = run_organisms(&mut pass, &orgs, None, [10.0, 20.0, 0.0]);
    let (_, _, warm) = pass.diag();
    assert_eq!(warm, 0, "warm pass must not rebuild");
    let ab: Vec<u8> = bytemuck::cast_slice(&a.arch).to_vec();
    let ab2: Vec<u8> = bytemuck::cast_slice(&a_wrap.arch).to_vec();
    assert_eq!(ab, ab2, "deterministic replan from identical input");
    // size (trait) change -> different signature -> new archetype rebuild
    let orgs2: Vec<OrgInst> = (0..4)
        .map(|i| inst_at(10.0 + i as f32, 0.0, (i + 1) % 2, 0.31, 1.4))
        .collect();
    run_organisms(&mut pass, &orgs2, None, [10.0, 20.0, 0.0]);
    let (entries2, meshes2, rebuilds2) = pass.diag();
    assert!(rebuilds2 > 0, "changed anatomy rebuilds");
    assert_eq!(entries2, 4, "cache tracks all live signatures");
    assert_eq!(meshes2, 2, "fallback archetypes are shared per brand");
}

/// Lead-integration hook: per-row REAL archetype keys route to true archetype
/// meshes (brand fallback bypassed), including distinct keys for same-brand
/// rows with different genomes.
#[test]
fn row_archetype_hook_wires_real_keys() {
    let mut pass = OrganismPass::new();
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
    let flag = organism_with(alife_common::Traits {
        max_speed: 150.0,
        size: 1.0,
        ..small.genotype.traits
    });
    let k_small = cmet::archetype_key(&alife_common::morphology::derive(&small));
    let k_flag = cmet::archetype_key(&alife_common::morphology::derive(&flag));
    assert_ne!(k_small, k_flag, "genome change must move the archetype");
    let mut rows = [
        inst_at(8.0, 0.0, BRAND_CELL, 0.31, 0.9),
        inst_at(16.0, 0.0, BRAND_CELL, 0.31, 0.9),
    ];
    // same signature, different hooked keys: the hook wins per row
    let want1 = run_organisms(&mut pass, &rows, Some(&[k_small, k_small]), [10.0, 20.0, 0.0]);
    assert_eq!(want1.groups.len(), 1, "same key shares one group even hooked");
    assert_eq!(want1.groups[0].0, k_small, "hooked key drives the group");
    rows[1].brand = BRAND_FLAGELLATE;
    let want2 = run_organisms(&mut pass, &rows, Some(&[k_small, k_flag]), [10.0, 20.0, 0.0]);
    assert_eq!(want2.groups.len(), 2, "distinct keys -> distinct groups");
    assert_eq!(want2.groups[0].0, k_small.min(k_flag));
    assert_eq!(want2.groups[1].0, k_small.max(k_flag));
    for g in &want2.groups {
        assert!(g.0 & 0x4000_0000_0000_0000 == 0, "real keys, not fallback space");
    }
}

/// LOD rings: < 60 near, < 180 mid (both full archetype mesh), between 180
/// and quad_radius blob (distance-faded), beyond culled.
#[test]
fn organism_lod_rings_and_far_blob_fade() {
    let mut pass = OrganismPass::new();
    let mk = |x: f32, y: f32| inst_at(x, y, BRAND_CELL, 0.31, 1.0);
    let orgs = [mk(0.0, 0.0), mk(130.0, 0.0), mk(320.0, 0.0), mk(20_000.0, 0.0)];
    // distances from view |p - (10, 0)|: 10 (near), 120 (mid), 310 (far), culled
    let plan = run_organisms(&mut pass, &orgs, None, [10.0, 20.0, 0.0]);
    let (near, mid, far) = pass.lod_counts();
    assert_eq!((near, mid, far), (1, 1, 1), "one organism per LOD lane");
    assert_eq!(plan.arch.len(), 2, "near + mid draw full archetype meshes");
    assert_eq!(plan.blobs.len(), 1, "far organism renders the cheap blob");
    // blob alpha carries the distance fade (alive 1.0 * fade < 1.0)
    let b = &plan.blobs[0];
    let qr = decomp::quad_radius(decomp::MODE_ORBIT);
    let fade = (1.0 - (310.0 - MID_RING) / (qr - MID_RING)).clamp(0.0, 1.0);
    assert!((b.alpha - fade).abs() < 1e-4, "fade window starts at MID_RING");
    assert!(b.alpha > 0.0 && b.alpha < 1.0);
    // legacy blob body radius chain preserved for the far silhouette
    assert!((b.radius - decomp::ORGANISM_UNIT_RADIUS * 1.0 * 0.924).abs() < 1e-4);
    // beyond quad radius: nothing planned, culled row accounted
    let mut pass2 = OrganismPass::new();
    let far_only = [mk(20_000.0, 0.0)];
    let plan2 = run_organisms(&mut pass2, &far_only, None, [10.0, 20.0, 0.0]);
    assert!(plan2.is_empty());
    assert_eq!(pass2.skipped_count(), 1);
}

/// Caps: at most DETAILED_CAP archetype instances + BLOB_CAP blob instances
/// per frame; crowd overflow inside the detail ring spills to the blob lane.
#[test]
fn organism_caps_detailed_blobs_and_skips_overflow() {
    let mut pass = OrganismPass::new();
    // 300 organisms clustered inside the detail ring
    let near_crowd: Vec<OrgInst> = (0..300)
        .map(|i| inst_at(10.0 + (i % 20) as f32 * 0.5, (i / 20) as f32 * 0.5, BRAND_CELL, 0.31, 1.0))
        .collect();
    let plan = run_organisms(&mut pass, &near_crowd, None, [10.0, 20.0, 0.0]);
    assert_eq!(plan.arch.len(), DETAILED_CAP, "detail cap holds");
    assert_eq!(plan.blobs.len(), 300 - DETAILED_CAP, "crowd spill fills blobs");
    assert_eq!(pass.skipped_count(), 0);
    // 1600 organisms in the far band blob-cap at 240
    let far_crowd: Vec<OrgInst> = (0..1600)
        .map(|i| inst_at(300.0 + (i % 40) as f32 * 3.0, (i / 40) as f32 * 3.0, BRAND_CELL, 0.31, 1.0))
        .collect();
    let plan2 = run_organisms(&mut pass, &far_crowd, None, [10.0, 20.0, 0.0]);
    assert!(plan2.arch.is_empty());
    assert_eq!(plan2.blobs.len(), BLOB_CAP, "blob cap holds");
    assert_eq!(pass.skipped_count(), (1600 - BLOB_CAP) as u32);
}

/// Queen rows never enter the organism pass (planner owns them): a field of
/// ordinary organisms + one queen plans exactly the ordinary ones, and the
/// queen's group counts are unchanged versus a queen-free field (queen lane
/// bytes come from the planner buffer, not this pass).
#[test]
fn queen_rows_are_routed_to_planner_pass_only() {
    let mut pass = OrganismPass::new();
    let plain: Vec<OrgInst> = (0..3)
        .map(|i| inst_at(10.0 + i as f32 * 5.0, 10.0, BRAND_CELL, 0.31, 1.0))
        .collect();
    let p_plain = run_organisms(&mut pass, &plain, None, [10.0, 20.0, 10.0]);
    let (e0, g0, _) = pass.diag();
    // now add a queen to the field: same view, same ordinary row set
    let mut queen = inst_at(60.0, -60.0, BRAND_QUEEN, 0.21, 2.0);
    queen.is_queen = 1;
    let mut field = plain.clone();
    field.push(queen);
    let p_field = run_organisms(&mut pass, &field, None, [10.0, 20.0, 10.0]);
    let (e1, g1, _) = pass.diag();
    assert_eq!(p_plain.arch.len(), 3);
    assert_eq!(p_field.arch.len(), 3, "queen never lands in the organism plan");
    assert!(p_field.blobs.is_empty() && p_plain.blobs.is_empty());
    assert_eq!(p_field.groups, p_plain.groups, "queen does not mutate groups");
    assert_eq!((e1, g1), (e0, g0), "queen rows burn no cache entries/meshes");
    // her skipped count is exactly 1 (routed to the planner reserved pass)
    assert_eq!(pass.skipped_count(), 1);
    // and the PLANNER still plans her reserved slot from the same scan
    let qp = decomp::plan_frame(
        &field,
        [10.0, 20.0, 10.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    assert_eq!(qp.queen_n, 1, "planner queen lane untouched");
    assert!(p_field.arch.iter().all(|r| r.pad < 0.5 && r.pad > 0.001), "ordinary rows keep sat_tint pad");
    assert!(qp.insts.iter().all(|r| r.pad == 0.0 || r.pad == 1.0), "planner rows keep the queen LOD flag");
}

/// Hooked rows carry the a1 hue recipe: instance hue stays the organism's
/// species hue (the FS chroma source) and sat_tint comes from the canonical
/// descriptor (pad channel, strictly inside (0, 0.5)).
#[test]
fn hooked_rows_propagate_instance_hue_and_canonical_sat_tint() {
    let mut pass = OrganismPass::new();
    let org = organism_with(alife_common::Traits {
        size: 1.0,
        max_speed: 40.0,
        efficiency: 0.7,
        sensor_range: 80.0,
        metabolism: 1.0,
        lifespan: 2400,
        repr_threshold: 0.5,
        repr_cost: 0.2,
    });
    let m = alife_common::morphology::derive(&org);
    let key = cmet::archetype_key(&m);
    // hue_channel is the fract()-domain chroma the FS blends with
    assert!((cmet::hue_channel(&m) - m.hue / 360.0).abs() < 1e-6);
    let o = inst_at(10.0, 0.0, BRAND_CELL, 0.61, 1.0);
    let plan = run_organisms(&mut pass, &[o], Some(&[key]), [10.0, 20.0, 0.0]);
    assert_eq!(plan.arch.len(), 1);
    let row = &plan.arch[0];
    assert!((row.hue - 0.61).abs() < 1e-5, "instance hue propagated");
    let canon_sat = cmet::canonical_from_key(key).sat_tint;
    assert!((row.pad - canon_sat.clamp(0.0, 0.49)).abs() < 1e-6, "pad == canonical sat_tint");
    assert!(row.pad > 0.001 && row.pad < 0.5, "pad strictly in the creature-gate band");
    // scale recipe rides the canonical body_length through RENDER_SCALE
    let tf = 0.62 + 0.38 * 0.8;
    let want = cmet::RENDER_SCALE * cmet::canonical_from_key(key).body_length * tf;
    assert!((row.radius - want).abs() < 1e-4, "hooked radius = RENDER_SCALE * canonical body_length");
}

/// diag_lod counts the last pass's near/mid/far lanes (extended over the
/// legacy topology: (live entries, meshes, rebuilds) + (near, mid, far)).
#[test]
fn diag_lod_reports_lane_counts() {
    let mut pass = OrganismPass::new();
    let orgs = [
        inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.0),   // d 10  -> near
        inst_at(130.0, 0.0, BRAND_CELL, 0.31, 1.0), // d 120 -> mid
        inst_at(320.0, 0.0, BRAND_CELL, 0.31, 1.0), // d 310 -> far blob
    ];
    run_organisms(&mut pass, &orgs, None, [10.0, 20.0, 0.0]);
    assert_eq!(pass.lod_counts(), (1, 1, 1));
    // an empty pass resets the counters
    run_organisms(&mut pass, &[], None, [10.0, 20.0, 0.0]);
    assert_eq!(pass.lod_counts(), (0, 0, 0));
    // the three diag APIs stay callable through the Morph façade type shape
    // (OrganismPass is the single source; Morph::diag_* just forwards).
    let (entries, meshes, rebuilds) = pass.diag();
    assert!(entries >= 1 && meshes >= 1 && rebuilds == 0, "diag tuple sane: ({entries}, {meshes}, {rebuilds})");
}

// ================================================== A3 locomotion (workstreams B+C)

use super::decomp::{
    gait_freq, leg_bucket_offset, leg_side_parity, motion_for, LocoTracks, Motion,
    LOCO_F_MAX, LOCO_F_MIN, LOCO_FULL, LOCO_IDLE_BREATHE, LOCO_IDLE_EPS, LOCO_WALK_BREATHE,
};
use alife_common::morphology::BodyPlan;

/// Struct extension: the stride docs stay in lockstep (22 x f32 = 88 bytes,
/// bytemuck Pod with zero-fill for never-animated lanes).
#[test]
fn morphinst_extension_matches_stride_docs() {
    assert_eq!(decomp::INST_FLOATS, 22);
    assert_eq!(decomp::INST_STRIDE, 88);
    assert_eq!(std::mem::size_of::<decomp::MorphInst>(), decomp::INST_STRIDE);
    let z = decomp::MorphInst::zeroed();
    assert!(z.motion == [0.0f32; 4] && z.drive == [0.0f32; 4], "zeroable defaults quiet");
}

/// Phase accumulation across frames: walking ids advance their stride phase
/// and estimate gait speed from the position delta / dt; stationary ids
/// decay to rest; teleports never spike the speed. Dead ids prune per pass.
#[test]
fn locomotion_phase_accumulates_walks_and_idles() {
    let mut lt = LocoTracks::new();
    lt.set_frame_dt(1.0 / 60.0);
    // ---- walk: 6 u per frame at 60 Hz => 360 u/s (gait fully engaged)
    lt.begin_pass();
    let (p0, s0) = lt.advance(7, 0.0, 0.0);
    assert_eq!(s0, 0.0, "new ids start at rest");
    assert_eq!(p0, 0.0, "idle ids freeze the phase (no drift)");
    lt.begin_pass();
    let (p1, s1) = lt.advance(7, 6.0, 0.0);
    assert!(s1 > 0.0, "some speed now: {s1}");
    let full = p1 + std::f32::consts::TAU * (1.0 / 60.0); // full-walk rate step
    // next walking frame advances the same step (gait fully engaged)
    let (p2, _) = lt.advance(7, 12.0, 0.0);
    assert!((p2 - full).abs() < 1e-4, "walking phase step {} vs {}", p2, full);
    // ---- idle: stationary frames decay the speed to zero, phase drift slows
    for _ in 0..120 {
        lt.begin_pass();
        lt.advance(7, 12.0, 0.0);
    }
    let (p_idle0, s_end) = lt.advance(7, 12.0, 0.0);
    assert!(s_end < 1e-3, "idle speed decays to ~0, got {s_end}");
    let (p_idle1, _) = lt.advance(7, 12.0, 0.0);
    assert_eq!(p_idle1, p_idle0, "parked limbs freeze (deterministic idle rows)");
    // ---- teleport: a big position jump resets speed instead of spiking it
    for i in 0..60 {
        lt.advance(7, 12.0 + i as f32 * 2.0, 0.0);
    }
    let (_, s_fast) = lt.advance(7, 12.0 + 50.0 * 2.0, 0.0);
    assert!(s_fast > LOCO_FULL * 0.5, "sprinting: {s_fast}");
    for i in 0..4 {
        let (_, s) = lt.advance(7, 12.0 + (50.0 + 900.0 * (i + 1) as f32) * 2.0, 0.0);
        assert!(s < 1e-3, "teleport resets speed, got {s}");
    }
    // ---- dead-id prune: only ids advanced this pass survive
    let mut lt2 = LocoTracks::new();
    lt2.begin_pass();
    lt2.advance(1, 0.0, 0.0);
    lt2.advance(2, 0.0, 0.0);
    lt2.begin_pass();
    lt2.advance(2, 0.0, 0.0);
    lt2.prune();
    lt2.begin_pass();
    // id 2 keeps advancing fine...
    let (_, s2) = lt2.advance(2, 0.0, 0.0);
    assert_eq!(s2, 0.0);
    // ...while pruned id 1 re-enters as a fresh-from-rest track (its new
    // position seeds px, so the stale delta never spikes the speed)
    let (p_dead, s_dead) = lt2.advance(1, 5.0, 0.0);
    assert!(s_dead < 1e-6, "pruned id restarted from rest");
    assert!(p_dead.abs() < std::f32::consts::TAU / 60.0 + 1e-6, "phase reseeds: {p_dead}");
}

/// Pure motion model: determinism, the documented formulas, idle quiescence
/// (zero motion except breathe) and the jointedness/armour damping.
#[test]
fn motion_model_is_pure_and_idle_is_quiet() {
    const BL: f32 = 1.4;
    let args = (BodyPlan::SegmentedArthropod as u8, BL, 18.0, 1.2345, 0.4, 0.2);
    let a = motion_for(args.0, args.1, args.2, args.3, args.4, args.5);
    let b = motion_for(args.0, args.1, args.2, args.3, args.4, args.5);
    assert_eq!(a, b, "pure function of its inputs");

    // bob = amp * sin(phase) at the documented amp stack
    let sp01 = (18.0 / LOCO_FULL).clamp(0.0, 1.0);
    let amp = super::decomp::LOCO_BOB * sp01
        * (1.0 - super::decomp::LOCO_JOINT_DAMP * 0.4)
        * 1.0 // BOB_PLAN_SCALE arthropod
        * BL;
    let want = amp * a.phase.sin();
    assert!((a.bob - want).abs() < 1e-6, "bob formula: {} vs {}", a.bob, want);
    assert!(a.pitch.abs() <= super::decomp::LOCO_PITCH_MAX + 1e-6);
    assert!(a.roll.abs() <= super::decomp::LOCO_ROLL_MAX + 1e-6);
    assert!(a.sway.abs() <= super::decomp::LOCO_SWAY + 1e-6);
    // stride amplitude at the documented stack (lift is a fraction of swing)
    let want_leg = super::decomp::LOCO_LEG_AMP * sp01
        * (1.0 - super::decomp::LOCO_LEG_JOINT_DAMP * 0.4)
        * BL;
    assert!((a.leg_amp - want_leg).abs() < 1e-6, "leg amp formula");
    assert!((a.leg_lift - want_leg * super::decomp::LOCO_LEG_LIFT).abs() < 1e-6);
    assert!(a.squash > 0.0, "walking squash (small body squash present)");

    // idle: everything zeros except the ambient breathe — determinism of the
    // quiescent latch (speed < LOCO_IDLE_EPS)
    let idle = motion_for(BodyPlan::Quadruped as u8, BL, LOCO_IDLE_EPS * 0.99, 2.5, 0.8, 0.9);
    let z = Motion { bob: 0.0, pitch: 0.0, roll: 0.0, phase: 2.5, leg_amp: 0.0, leg_lift: 0.0, squash: 0.0, breathe: LOCO_IDLE_BREATHE, sway: 0.0 };
    assert_eq!(idle, z, "idle quiescence: only breathe stays");

    // jointedness reduces bob (smoother gait), armour reduces squash
    let stiff = motion_for(0, BL, LOCO_FULL, 1.57, 0.0, 0.0);
    let floppy = motion_for(0, BL, LOCO_FULL, 1.57, 1.0, 0.0);
    assert!(stiff.bob.abs() > floppy.bob.abs(), "jointed bodies bob less");
    let bare = motion_for(0, BL, LOCO_FULL, 0.31, 0.0, 0.0);
    let plated = motion_for(0, BL, LOCO_FULL, 0.31, 0.0, 1.0);
    assert!(bare.squash > plated.squash, "armour stiffens the squash");
    // primitive squash-stretch is the strongest walk style
    let quad = motion_for(3, BL, LOCO_FULL, 0.31, 0.0, 0.0);
    assert!(bare.squash > quad.squash * 2.0, "primitive squash > limb-carry style");
}

/// Body-length extremes: the stride frequency clamp and amplitude bounds
/// hold from a tiny blob to the giant queen-length silhouette.
#[test]
fn stride_correctness_at_body_length_extremes() {
    assert_eq!(gait_freq(0.05), LOCO_F_MAX, "tiny bodies ride the fast cap");
    assert_eq!(gait_freq(16.0), LOCO_F_MIN, "giant bodies ride the slow cap");
    assert!((gait_freq(1.5) - 4.0).abs() < 1e-5, "mid-length creatures ~4 Hz");
    for plan in [0u8, 1, 2, 3] {
        for bl in [0.2f32, 0.5, 1.4, 3.0, 16.0] {
            let m = motion_for(plan, bl, LOCO_FULL, 0.77, 0.5, 0.5);
            assert!(m.bob.is_finite() && m.bob.abs() <= super::decomp::LOCO_BOB * bl + 1e-5,
                "plan {plan} bob bound at bl {bl}: {}", m.bob);
            assert!(m.leg_amp.is_finite() && m.leg_amp <= super::decomp::LOCO_LEG_AMP * bl + 1e-5,
                "plan {plan} leg bound at bl {bl}: {}", m.leg_amp);
            assert!(m.leg_lift <= m.leg_amp + 1e-6, "lift <= swing");
            assert!(m.pitch.is_finite() && m.roll.is_finite());
        }
    }
    // limbless Primitive carries no stride fields even at full speed
    let prim = motion_for(0, 1.4, LOCO_FULL, 0.5, 0.0, 0.0);
    assert_eq!(prim.leg_amp, 0.0, "Primitive has no legs to stride");
}

/// Gait styles: the VS leg-phase bucket table mirrored in Rust — quadruped
/// diagonal pairs (front-left == rear-right anti-phase to front-right ==
/// rear-left) and the arthropod tripod-ish mid-row on the fallback meshes.
#[test]
fn stride_plan_mapping_quadruped_diagonal_and_arthropod_parity() {
    // ---- pure table mirrors of the WGSL
    assert_eq!(leg_bucket_offset(0.16), 0.0, "front leg x>=0");
    assert_eq!(leg_bucket_offset(-0.34), std::f32::consts::PI, "rear leg x<-0.24");
    assert!((leg_bucket_offset(-0.1) - std::f32::consts::FRAC_PI_2).abs() < 1e-6, "mid row");
    assert_eq!(leg_side_parity(0.5), 0.0);
    assert!((leg_side_parity(-0.5) - std::f32::consts::PI).abs() < 1e-6);

    // ---- quadruped fallback mesh: legs resolve into a front pair (x >= 0)
    // and a rear pair (x <= -0.24) with NO limb verts straddling the mid row
    // — knees/feet lean but stay inside their pair's bucket, so the
    // per-vertex phase offsets stay coherent per leg
    let mesh = cmet::brand_fallback(3);
    let legs: Vec<(f32, f32, f32)> = mesh.iter()
        .filter(|t| t.aux[1] == cmet::CODE_LIMB && t.pos[1] < 0.0)
        .map(|t| (t.pos[0], t.pos[1], t.pos[2]))
        .collect();
    assert!(!legs.is_empty(), "quadruped fallback has legs");
    // ---- quadruped fallback mesh (tier 0 lean): front pair verts all sit at
    // x >= 0; the rear pair's hip/knee sit at x <= -0.24 while its FOOT TIPS
    // lean forward past the row boundary (y < -0.5, documented A3 limit —
    // lean-tier foot verts get the mid-row phase offset; stylized, tiny at
    // LOCO_LEG_AMP scale). Anything else straddling is a real bug.
    for &(x, y, _) in &legs {
        assert!(
            x >= 0.0 || x <= -0.24 || y < -0.5,
            "unexpected mid-band quadruped leg vert at x={x} y={y}"
        );
    }
    assert!(legs.iter().any(|&(x, _, _)| x > 0.0), "front leg pair attaches forward");
    assert!(legs.iter().any(|&(x, _, _)| x <= -0.24), "rear leg pair attaches behind");
    // diagonal-pair gait (FL+RR in phase, FR+RL anti-phase): documented hip
    // slots (+0.16 / -0.34) plus side parity
    let has_left_front = legs.iter().any(|&(x, _, z)| x > 0.0 && z > 0.0);
    let has_right_rear = legs.iter().any(|&(x, _, z)| x <= 0.0 && z <= 0.0);
    if has_left_front && has_right_rear {
        let ofl = leg_bucket_offset(0.16) + leg_side_parity(1.0);
        let orr = leg_bucket_offset(-0.34) + leg_side_parity(-1.0);
        assert!((ofl - orr).rem_euclid(std::f32::consts::TAU).abs() < 1e-6, "FL+RR diag pair");
        let ofr = leg_bucket_offset(0.16) + leg_side_parity(-1.0);
        let orl = leg_bucket_offset(-0.34) + leg_side_parity(1.0);
        assert!((ofr - orl).rem_euclid(std::f32::consts::TAU).abs() < 1e-6, "FR+RL diag pair");
        assert!(
            (((ofl.rem_euclid(std::f32::consts::TAU)) - (ofr.rem_euclid(std::f32::consts::TAU))).abs()
                - std::f32::consts::PI).abs() < 1e-5,
            "the two pair groups are anti-phase (trot)"
        );
    }

    // ---- arthropod fallback mesh: legs sit in the mid row (tripod-ish tread
    // via side parity), leg template y is below the trunk midline
    let mesh_a = cmet::brand_fallback(1);
    let aleg: Vec<_> = mesh_a.iter()
        .filter(|t| t.aux[1] == cmet::CODE_LIMB && t.pos[1] < 0.0)
        .map(|t| (t.pos[0], t.pos[2]))
        .collect();
    assert!(!aleg.is_empty(), "arthropod fallback has legs");
    for &(x, _) in &aleg {
        assert!(x.abs() < 0.24, "arthropod legs are mid/front row buckets, got {x}");
        assert!((leg_bucket_offset(x) - std::f32::consts::FRAC_PI_2).abs() < 1e-6
            || leg_bucket_offset(x) == 0.0, "mid or front, never rear");
    }
    // ---- multi-legged fallback spans all three rows (amble w/ quarter rows)
    let mesh_c = cmet::brand_fallback(2);
    let clegs: Vec<f32> = mesh_c.iter()
        .filter(|t| t.aux[1] == cmet::CODE_LIMB && t.pos[1] < 0.0)
        .map(|t| t.pos[0])
        .collect();
    assert!(clegs.iter().any(|&x| x >= 0.0), "front row present");
    assert!(clegs.iter().any(|&x| x < 0.0 && x >= -0.24), "mid row present");
    assert!(clegs.iter().any(|&x| x < -0.24), "rear row present");
    // antennae (CODE_LIMB above midline) are NOT legs
    let ants: Vec<_> = mesh_c.iter()
        .filter(|t| t.aux[1] == cmet::CODE_LIMB && t.pos[1] >= 0.0)
        .collect();
    assert!(!ants.is_empty(), "antennae exist as upper CODE_LIMB verts");
}

/// Idle rows: motion zero everywhere except `breathe`; far blob rows stay
/// byte-quiet; walking rows carry bob + stride (per-brand style) + heading sway.
#[test]
fn organism_rows_extend_with_locomotion_fields() {
    let mut pass = OrganismPass::new();
    pass.set_frame_dt(1.0 / 60.0);
    let mut idle = inst_at(40.0, 40.0, BRAND_COLONY, 0.31, 1.0);
    idle.id = 42;
    let p1 = run_organisms(&mut pass, &[idle], None, [40.0, 20.0, 40.0]);
    let p2 = run_organisms(&mut pass, &[idle], None, [40.0, 20.0, 40.0]);
    for plan in [&p1, &p2] {
        let row = &plan.arch[0];
        assert!(
            row.motion[0] == 0.0 && row.motion[1] == 0.0 && row.motion[2] == 0.0,
            "idle bob/pitch/roll zero, got {:?}", &row.motion[..3]
        );
        assert!(row.drive[0].abs() < 1e-6 && row.drive[1].abs() < 1e-6 && row.drive[2].abs() < 1e-6,
            "idle drive (swing/lift/squash) zero, got {:?}", row.drive);
        assert!((row.drive[3] - LOCO_IDLE_BREATHE).abs() < 1e-6, "idle breathe amplitude");
    }
    // phase is desynchronized by the species seed and frozen while idle
    // (deterministic replans: identical rows byte-for-byte between passes)
    assert_eq!(p2.arch[0].motion[3], p1.arch[0].motion[3], "idle phase frozen");

    // ---- far blob rows stay silent (pad 0 lane)
    let mut far = inst_at(320.0, 0.0, BRAND_CELL, 0.31, 1.0);
    far.id = 43;
    let pf = run_organisms(&mut pass, &[far], None, [10.0, 20.0, 0.0]);
    for b in &pf.blobs {
        assert!(b.motion == [0.0f32; 4] && b.drive == [0.0f32; 4], "blobs never animate");
    }

    // ---- walking: 0.3 u per frame at 60 Hz => 18 u/s
    let mut walk = inst_at(0.0, 0.0, BRAND_FLAGELLATE, 0.31, 1.0);
    walk.id = 44;
    let mut bobs = Vec::new();
    for f in 0..6 {
        walk.x = f as f32 * 0.3;
        let p = run_organisms(&mut pass, &[walk], None, [0.0, 20.0, 0.0]);
        let row = &p.arch[0];
        bobs.push(row.motion[0]);
        if f >= 1 {
            // stride engaged: legs swing, breathe shallows, sway rides heading
            assert!(row.drive[0] > 0.0, "arthropod legs swing while walking: {} f{}", row.drive[0], f);
            assert!(row.motion[0].abs() > 0.0 || row.motion[1].abs() > 0.0 || row.motion[2].abs() > 0.0,
                "walking rows move: f{f} {:?}", row.motion);
            assert!((row.drive[3] - LOCO_WALK_BREATHE).abs() < 1e-6, "ambient walk breathe f{f}");
            assert!(row.heading.abs() <= super::decomp::LOCO_SWAY + 1e-5, "heading sway bounded f{f} = {}", row.heading);
        }
    }
    assert!(bobs.iter().any(|b| b.abs() > 0.005), "bob amplitude over the walk cycle");

    // ---- gait style follows the BodyPlan beneath the mesh: limbless
    // Primitive rows carry no stride even at speed
    let mut run = inst_at(0.0, 0.0, BRAND_CELL, 0.31, 1.0);
    run.id = 45;
    let mut p = OrgPlan::new();
    for f in 0..3 {
        run.x = f as f32 * 0.4; // 24 u/s: fully engaged
        p = run_organisms(&mut pass, &[run], None, [0.0, 20.0, 0.0]);
    }
    assert_eq!(p.arch[0].drive[0], 0.0, "limbless Primitive stride stays zero");
    assert!(p.arch[0].motion[0] != 0.0 || p.arch[0].motion[1] != 0.0, "but body motion rides");
}

/// Queen lane (reserved template) and legacy planner rows keep the A3
/// extension fields zeroed: her settled gait is untouched; only the pad
/// routing she already had matters to the VS.
#[test]
fn queen_and_planner_rows_keep_locomotion_fields_zero() {
    let plan = decomp::plan_frame(
        &[queen_org()],
        [400.0, 40.0, 400.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    let qi = &plan.insts[(plan.tail_n + plan.shell_n) as usize];
    assert!(qi.motion == [0.0f32; 4] && qi.drive == [0.0f32; 4], "queen row quiet");
    assert!(qi.pad == 0.0 || qi.pad == 1.0, "queen pad routing unchanged");

    let ordinary = decomp::plan_frame(
        &[inst_at(400.0, 400.0, BRAND_FILAMENT, 0.62, 1.0)],
        [400.0, 40.0, 400.0],
        decomp::MODE_ORBIT,
        &mut |_, _| 0.0,
        &mut |_, _| true,
    );
    for i in &ordinary.insts {
        assert!(i.motion == [0.0f32; 4] && i.drive == [0.0f32; 4], "legacy lanes quiet");
        assert!(i.pad == 0.0, "legacy lanes are outside the creature gate");
    }
}

/// Idealized phase accounting used by the stride docs: at walking speed the
/// quadruped's legs resolve into two anti-phase pair groups (documented
/// decision: per-part animation exists only via the static material ids —
/// per-instance data cannot carry per-part phases, so the bucket + parity
/// tables do the splitting; the head nod rides CODE_HEAD).
#[test]
fn head_nod_choice_documented_and_head_code_present() {
    // CODE_HEAD exists in the creature material ids => the head nod rides it
    assert_eq!(cmet::CODE_HEAD, 6.25);
    let mesh = cmet::brand_fallback(3);
    assert!(mesh.iter().any(|t| t.aux[1] == cmet::CODE_HEAD), "quadruped has a head part");
    assert!(mesh.iter().any(|t| t.aux[1] == cmet::CODE_LIMB), "quadruped has limbs");
}
