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

// ----------------------------------------------------- WGSL compiles

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
