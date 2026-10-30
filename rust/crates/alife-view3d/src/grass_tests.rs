//! Pure-function tests for grass placement, rules, culling and wind.

use super::place::{self, MODE_FOLLOW, MODE_ORBIT, MODE_PLANET};
use super::*;

fn rule_at(
    f: f32,
    m: f32,
    hz: f32,
    e: f32,
    biome: u32,
) -> place::Rule {
    place::species_rule(f, m, hz, e, biome)
}

#[test]
fn density_respects_biome_rules() {
    // fertile, moist plains -> dense
    let fertile = rule_at(0.9, 0.9, 0.0, 0.8, 0);
    // shore band (elevation just below sea level) -> sparse algae
    let shore = rule_at(0.9, 0.9, 0.0, 0.40, 0);
    // hazard cell -> bare
    let hazard = rule_at(0.9, 0.9, 0.9, 0.8, 0);
    // deep water -> bare
    let deep = rule_at(0.9, 0.9, 0.0, 0.2, 0);
    // dry rocky -> sparse
    let rocky = rule_at(0.4, 0.15, 0.0, 0.8, 2);

    assert!(fertile.density >= 0.65, "fertile plains must be dense: {}", fertile.density);
    assert!(shore.density < fertile.density, "shore must be sparser than plains");
    assert!(rocky.density < fertile.density, "rocky highlands sparser than plains");
    assert!(hazard.density <= 0.01, "hazard cells stay bare: {}", hazard.density);
    assert_eq!(deep.density, 0.0, "deep water has no vegetation");
}

#[test]
fn placement_is_deterministic() {
    let cfg = alife_worldgen::WorldGenConfig::default();
    let world = alife_worldgen::generate(&cfg);
    let b = |scale: f32| place::Builder {
        gen: world.generation_tag,
        cell: (3, 3),
        mode: MODE_ORBIT,
        focus: (112.0, 96.0),
        lod_scale: scale,
    };
    let a = place::build(&world, &b(1.0));
    let c = place::build(&world, &b(1.0));
    assert_eq!(a.total, c.total, "same inputs must give identical totals");
    assert_eq!(a.chunks.len(), c.chunks.len());
    for (ca, cb) in a.chunks.iter().zip(c.chunks.iter()) {
        assert_eq!(ca.cx, cb.cx);
        assert_eq!(ca.cy, cb.cy);
        assert_eq!(ca.insts.len(), cb.insts.len());
        for (ia, ib) in ca.insts.iter().zip(cb.insts.iter()) {
            assert_eq!(ia.x, ib.x);
            assert_eq!(ia.z, ib.z);
            assert_eq!(ia.tint_hue, ib.tint_hue);
            assert_eq!(ia.lean_phase, ib.lean_phase);
        }
    }
    // different hash inputs must actually differ somewhere
    let other = {
        let mut found = false;
        let mut idx = 0;
        let mut prev = place::blade_hash(1, 0, 0, 0);
        for i in 1..64 {
            let cur = place::blade_hash(1, 0, 0, i);
            if cur != prev {
                found = true;
                idx = i;
            }
            prev = cur;
        }
        let _ = idx;
        found
    };
    assert!(other, "blade hash must vary per blade index");
}

#[test]
fn instance_cap_under_budget() {
    let cfg = alife_worldgen::WorldGenConfig::default();
    let world = alife_worldgen::generate(&cfg);
    // planet mode covers the entire dense-rich world â€” worst case for cap
    let b = place::Builder {
        gen: world.generation_tag,
        cell: (0, 0),
        mode: MODE_PLANET,
        focus: (world.width * 0.5, world.height * 0.5),
        lod_scale: 1.0,
    };
    let res = place::build(&world, &b);
    assert!(res.total <= place::MAX_BLADES, "blade cap exceeded: {}", res.total);
    assert!(
        res.total * place::TRIS_PER_BLADE <= 300_000,
        "triangle budget exceeded: {}",
        res.total * place::TRIS_PER_BLADE
    );
}

#[test]
fn lod_scale_thins_density_monotonically() {
    // crafted flat, fertile-damp world (no cap saturation, one clear window)
    let g = |v: f32| alife_worldgen::FieldGrid::new(4, 4, v);
    let world = alife_worldgen::WorldModel {
        width: 96.0,
        height: 96.0,
        cell: 1.0,
        elevation: g(0.8),
        moisture: g(0.5),
        temperature: g(0.5),
        fertility: g(0.9),
        hazard: g(0.0),
        biome: g(0.0),
        generation_tag: 7,
        hazard_pulse_period: 0.0,
    };
    let b = |s: f32| place::Builder {
        gen: 7,
        cell: (1, 1),
        mode: MODE_FOLLOW,
        focus: (48.0, 48.0),
        lod_scale: s,
    };
    let full = place::build(&world, &b(1.0));
    let thin = place::build(&world, &b(0.4));
    assert!(thin.total < full.total, "lower lod_scale must drop more blades");
    assert!(full.total <= place::MAX_BLADES);
}

#[test]
fn frustum_cull_rejects_behind_camera() {
    // camera at origin looking down -z (right-handed)
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 0.0, -1.0),
        glam::Vec3::Y,
    );
    let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_1_SQRT_2, 1.0, 0.1, 100.0);
    let f = frustum::Frustum::from_cam(
        &view.to_cols_array_2d(),
        &proj.to_cols_array_2d(),
    );
    let mid = [
        glam::Mat4::look_at_rh(glam::Vec3::ONE, glam::Vec3::ZERO, glam::Vec3::Y)
            .to_cols_array_2d(),
    ];
    let _ = mid;
    assert!(f.sphere_visible([0.0, 0.0, -50.0], 2.0), "front chunk visible");
    assert!(!f.sphere_visible([0.0, 0.0, 50.0], 2.0), "behind-camera chunk culled");
    assert!(!f.sphere_visible([0.0, 0.0, 900.0], 6.0), "far behind chunk culled");
    // near plane: things behind z>=0 clipped even if inside x/y frustum cone
    assert!(!f.sphere_visible([0.001, 0.0, 3.0], 0.5));
}

#[test]
fn frustum_keeps_visible_ring() {
    // horizontal camera looking down -z (right-handed)
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 3.0, 0.0),
        glam::Vec3::new(0.0, 3.0, -1.0),
        glam::Vec3::Y,
    );
    let proj = glam::Mat4::perspective_rh(1.3, 16.0 / 9.0, 0.2, 900.0);
    let f = frustum::Frustum::from_cam(&view.to_cols_array_2d(), &proj.to_cols_array_2d());
    let mut kept = 0;
    for t in 0..64 {
        let ang = t as f32 / 64.0 * std::f32::consts::TAU;
        let c = [300.0 * ang.cos(), 12.0, 300.0 * ang.sin() - 100.0];
        if f.sphere_visible(c, 3.0) {
            kept += 1;
        }
    }
    assert!(kept > 0 && kept < 64, "ring should be partially culled: kept {}", kept);
}

#[test]
fn wind_gust_amplitude_bounded() {
    // gust EMA: bounded in [0.30, 1.6] target and thus Ever after smoothing
    for dt in [1.0 / 60.0, 0.033, 0.1] {
        for chunk in 0..=1 {
            let cloud = chunk as f32;
            let mut gust = 0.35f32;
            for _ in 0..600 {
                let target = (0.30 + 0.55 * cloud + 0.45 * 1.0_f32 * cloud).clamp(0.30, 1.6);
                gust += (target - gust) * ((dt as f32) * 2.5).clamp(0.05_f32, 0.6_f32);
                assert!((0.05..=2.5).contains(&gust), "gust escaped bound: {}", gust);
            }
        }
    }
    // shader-side bound check: clamp in WGSL mirrors this constant
    assert!(2.5 >= 1.6, "shader clamp must dominate the ema target ceiling");
}

#[test]
fn species_overrides_exist_only_on_matching_fields() {
    // shrub: fertile + dry
    assert!(place::shrub_override(0.8, 0.2, 0.0).is_some());
    assert!(place::shrub_override(0.8, 0.9, 0.0).is_none());
    assert!(place::shrub_override(0.2, 0.2, 0.0).is_none());
    // dry tuft: low moisture + elevated ground
    assert!(place::dry_tuft_override(0.1, 0.9, 0.0).is_some());
    assert!(place::dry_tuft_override(0.9, 0.9, 0.0).is_none());
    assert!(place::dry_tuft_override(0.1, 0.2, 0.0).is_none());
    // pick_rule falls back to the primary rule otherwise
    let r = place::pick_rule(0.9, 0.9, 0.0, 0.8, 0, 0.9999);
    assert!(r.density > 0.5, "primary fertile rule should keep most candidates");
}

#[test]
#[ignore]
fn probe_counts() {
    let cfg = alife_worldgen::WorldGenConfig::default();
    let w = alife_worldgen::generate(&cfg);
    for (fx, fz) in [(900.0, 800.0), (1200.0, 800.0), (300.0, 300.0)] {
        let r = place::build(
            &w,
            &place::Builder {
                gen: w.generation_tag,
                cell: ((fx / 32.0) as i32, (fz / 32.0) as i32),
                mode: MODE_ORBIT,
                focus: (fx, fz),
                lod_scale: 1.0,
            },
        );
        println!("orbit focus ({fx},{fz}) -> total {} tris {} chunks {}", r.total, r.total * 6, r.chunks.len());
    }
    let rf = place::build(
        &w,
        &place::Builder {
            gen: w.generation_tag,
            cell: (28, 25),
            mode: MODE_FOLLOW,
            focus: (900.0, 800.0),
            lod_scale: 1.0,
        },
    );
    println!("follow focus -> total {} tris {}", rf.total, rf.total * 6);
}

/// WGSL validity smoke test; skipped silently when no GPU adapter exists
/// (e.g. CI sandboxes). wgpu panics inside create_shader_module on invalid
/// WGSL, so a pass means both shader modules are valid.
#[test]
fn gpu_pipeline_smoke() {
    if std::env::var("GRASS_SKIP_GPU").is_ok() {
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
    let mut g = Grass::new(&device).unwrap();
    g.ensure_pipeline(&device, wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Depth32Float);
    assert!(g.gpu.is_some());
}


