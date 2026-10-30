# RENDERER-RESEARCH.md — Native C++ renderer vs. existing in-process Rust/wgpu renderer

**Owner:** Agent E (research only). **Date:** 2026-10-30. **Status:** decision made below.

Context: existing renderer is Rust/wgpu (v22+ era) running in-process, integrated into egui via
paint callbacks. Scene: terrain chunks + instanced grass + organisms. Target hardware:
Intel Iris Xe iGPU (ADL platform, i5-1235U, 16 GB shared-memory laptop). Question: should we
build a native C++ renderer instead?

---

## 1. Findings — Intel Iris Xe (ICL/ADL iGPU) driver stability, per backend

### 1.1 Vulkan backend on Iris Xe — ACTIVE, UNFIXABLE driver bug (verified)

- **IGCIT #1309** ("vkCreateGraphicsPipelines returns VK_SUCCESS but sets pipeline to
  VK_NULL_HANDLE (GLSL SPIR-V)", Intel Iris Xe, i7-1265U, Vulkan 1.4.323, Win10 22H2):
  a *minimal* GLSL→SPIR-V pipeline creation returns `VK_SUCCESS` while writing
  `VK_NULL_HANDLE` — a hard Vulkan-spec violation. Intel engineer (Karen-Intel, Jan 9, 2026)
  closed it **"Unable to support"**, stating:

  > "Iris Xe products have reached legacy support mode, which means that Intel will provide
  > software support for affected products on critical fixes and security vulnerabilities only."

  Source: https://github.com/IGCIT/Intel-GPU-Community-Issue-Tracker-IGCIT/issues/1309 (fetched 2026-10-30).
  Intel's legacy-support policy page is linked from that reply:
  https://www.intel.com/content/www/us/en/support/articles/000101986/graphics.html (not fetched directly — **unverified content of that page**).

- **Critical implication:** this bug is *language-agnostic*. A hand-written C++ Vulkan
  renderer hits the exact same `vkCreateGraphicsPipelines` driver call with the exact same
  result. A C++ rewrite does not buy driver-correctness here.

### 1.2 Vulkan backend — historical wgpu-specific Intel issues (verified, mostly fixed/mitigated)

- **wgpu #3158** — instance creation with `Backends::VULKAN` hung indefinitely on an
  11th-gen (TGL GT2) Mesa/Intel Xe laptop; closed `not_planned` / needs-information (2024).
  https://github.com/gfx-rs/wgpu/issues/3158
- **wgpu #1672** — lockup in `ioctl` on Linux/Vulkan/Intel-Mesa caused by the same semaphore
  being signaled and waited ("relay" semaphore); Mesa issue 5508. Official workaround (also
  recorded in wgpu's own docs): **one submission per frame**; partially patched via #2212.
  https://github.com/gfx-rs/wgpu/issues/1672
- **wgpu official "Known Driver Issues" doc** (vendored page
  https://wgpu.rs/doc/src/wgpu/documentation/internals/known_driver_issues.rs.html) lists for
  Intel/Vulkan: Intel-Mesa semaphore signal+wait hang (Linux, Mesa 21.2.3), NULL
  `cmd_begin_debug_utils_label_ext` pointer, hidden-iGPU-for-apps-named-cube; for
  Intel/DirectX12: offset-after-resize artifact (wgpu-rs#647), `CreatePlacedResource` TDR on
  older HD Graphics (wgpu#1319); Intel/All: MSAA + sRGB resolve cannot be combined (wgpu#725);
  and a general Intel/Windows "buffer write not synchronized with draw" (wgpu#1060) that has
  no manual workaround.
- **wgpu #4599** — outdated Intel Windows drivers (pre-31.0.101.2115) fail robustness-feature
  shadow rendering in wgpu ≥0.15; user confirmed fixed after driver update.
  https://github.com/gfx-rs/wgpu/issues/4599 → **driver version pin matters on Iris Xe.**

### 1.3 Vulkan vs DX12 — which backend wgpu actually uses on Windows (verified, with one source contradiction)

- **wgpu #2719** ("Make DX12 the Default API on Windows", open as of 2026) gives the
  maintainers' rationale for *eventually* preferring DX12 on Windows:
  > "DXGI swapchains work a lot better on windows than vulkan WSI, particuarly AMD and
  > Intel's. They are much more stable and more well behaved."
  https://github.com/gfx-rs/wgpu/issues/2719
  However, in the same thread (Mar 2026) maintainer cwfitzgerald still declines the switch:
  "the shader story is still much worse on D3D, and performance is still broadly worse".
  **Conclusion: wgpu on Windows still prefers Vulkan, not DX12.**
- **Contradiction to flag:** gogpu/wgpu issue #24 (a third-party Go WebGPU implementation)
  claims "Rust wgpu uses DX12 by default on Windows". That claim is **contradicted** by the
  gfx-rs/wgpu thread above; treat the gogpu statement as unverified/incorrect.
  https://github.com/gogpu/wgpu/issues/24
- Practically: on the project's Iris Xe hardware, explicit backend selection is available
  (`wgpu::Backends::DX12`) and is the cheap mitigation if the Vulkan backend misbehaves.

### 1.4 C++ native renderer driver notes

- The Terrain3D project (C++ GDExtension under Godot, i.e. a native C++ renderer case study)
  has 4.5k stars and is maintained, but it is *specific* to the Godot engine's renderer and
  does not encode Iris Xe-specific knowledge. https://github.com/TokisanGames/Terrain3D
  Godot community discussions about Iris Xe instability exist, but no official Terrain3D doc
  page stating ICL/ADL-specific guidance was found in this research session (**unverified** —
  nothing citable found; not fabricated).

---

## 2. Findings — renderer-architecture precedents (technique reference for either path)

All four grass repos implement/derive from **Jahrmann & Wimmer 2017, "Responsive real-time
grass rendering for general 3D scenes"** (TU Wien). Note for our stack: wgpu has **no
tessellation shader stage**, so the Jahrmann–Wimmer style tessellation pipelines in these
repos translate in Rust/wgpu as compute-shader culling + vertex pulling (SSBO-indexed
attributes) or instanced blade billboards/meshes. The *techniques* transfer;
the *code* mostly cannot be copied (license table below).

| Repo | Language / stack | Technique | License (verified how) | Useful to us? |
|---|---|---|---|---|
| [TokisanGames/Terrain3D](https://github.com/TokisanGames/Terrain3D) | C++ GDExtension for Godot 4 | LOD terrain (up to 10 LODs, ~65.5×65.5 km regions), foliage instancing with LODs + shadow impostor, texture splatting | **MIT** (README "MIT license" + LICENSE.txt, fetched 2026-10-30) | Architecture inspiration only (LOD ladder, foliage instancing design); code is Godot-tied, not wgpu/C++-drop-in |
| [DeveloperDenis/RealTimeGrassRendering](https://github.com/DeveloperDenis/RealTimeGrassRendering) | C/C++ + OpenGL (GLSL) | Geometry-based grass blades, instancing, force-map wind, per-blade lighting, tessellation-level LOD near camera, distance culling | **Unverified.** Repo appears in the GitHub search index and the author's profile (developerdenis.com), but `GET api.github.com/repos/DeveloperDenis/RealTimeGrassRendering` returned 404 at research time — existence/status **unverified**; assume no code reuse | Read-as-reference only if repo revives; not usable otherwise |
| [bobhansky/VulkanGrass](https://github.com/bobhansky/VulkanGrass) | C++ / Vulkan (CMake, CIS 565 course project) | Compute-shader physics + orientation/distance/frustum culling into SSBO bound as vertex buffer, tessellation-control-shader LOD, RenderDoc-verified perf analysis (culling: 709→1110 fps in their dense scene) | **No license file found**: `GET api.github.com/repos/bobhansky/VulkanGrass/license` → 404 (2026-10-30). Treat as all-rights-reserved | Technique-only reference; do NOT copy code. Good culling-order perf data |
| [wayne-wu/vulkan-grass-rendering](https://github.com/wayne-wu/vulkan-grass-rendering) | C++ / Vulkan (fork of CIS565-Fall-2021 base) | Same Jahrmann–Wimmer family: compute-shader gravity/recovery/wind forces, orientation/frustum/distance culling, tess-eval curve interpolation | **No license file found**: `/license` endpoint → 404 (2026-10-30). Unlicensed | Technique-only reference; the base project (CIS565) has per-course terms — **unverified** |
| [rcm7133/Modern-Grass-Rendering](https://github.com/rcm7133/Modern-Grass-Rendering) | C# (Unity — per GitHub API `language: C#`) | >100,000 GPU-instanced grass blade meshes, distance-based LOD, custom GPU frustum culling | **None** (GitHub API `license: null`, 2026-10-30) | Concept reference only (instancing scale target; LOD+culling design), no code reuse |

License/technique table summary: **nothing here licenses C++ code reuse into our project**;
MIT Terrain3D permits code reuse but the code is Godot-engine-bound, so porting still means
a rewrite.

---

## 3. Decision framework — end-to-end calculus

Costs/benefits of a hypothetical C++ renderer on this project:

| Dimension | In-process Rust/wgpu (current) | Native C++ layer |
|---|---|---|
| Driver correctness on Iris Xe | Vulnerable to IGCIT #1309-class Vulkan driver bugs, **but has a fallback (force DX12/GLES backend)** | Raw Vulkan is vulnerable to the *same* Intel driver bugs with no abstraction escape hatch; D3D11/D3D12 path is a fresh engine |
| Integration | Zero bridge; scene data (chunk heights, grass instances, organism transforms) zero-copy shared with egui UI | Requires FFI boundary: marshalling thousands of per-frame instance matrices across ABI, or duplicating the scene graph in C++ |
| Memory | One GPU buffer set; iGPU shares system RAM, so buffers uploaded once | Second context/renderer duplicates vertex+instance buffers → double shared-memory pressure, which is the exact scarce resource on this hardware |
| Perf upside | naga/WGSL + bind-group overhead is real but measurable | Only wins if we beat wgpu's overhead *and* keep the bridge cost ≈ 0, which it won't be |
| Maintenance | One toolchain (Rust), egui paint callback already works | CMake/ABI churn on Windows, second bug surface |

**Decision (headline): NO C++ rewrite. Keep and harden the in-process Rust/wgpu renderer.**
Given the existing renderer already runs in-process, a C++ rewrite must show a **measured,
large, end-to-end** advantage. This research found none; on this specific Iris Xe hardware the
driver reality (*legacy support mode* per Intel's response) makes the Vulkan driver bugs a
platform property that C++ would inherit, while the *DX12 fallback and one-submission-per-frame
mitigation are only available inside the wgpu path*.

Standing bias for future decisions: benchmark, don't rewrite.

### 3.1 Hardening checklist for the current renderer (no C++)

1. Pin/downgrade-compatible Intel driver ≥ 31.0.101.2115 range (wgpu #4599 evidence) and record
   the exact driver version in the project docs.
2. Keep exactly **one wgpu submission per frame** (Intel/Mesa relay-semaphore hang workaround
   from wgpu #1672); avoid same-semaphore signal+wait patterns.
3. Do not combine MSAA with sRGB resolve (wgpu #725, Intel/All).
4. If pipeline creation fails or device-lost errors appear: profile with
   `wgpu::Backends::DX12` forced (swapchain stability per wgpu #2719) before considering any
   platform change; wrap backend selection in one config knob.

### 3.2 Revisit conditions (any one, with documented benchmarks)

+ **R1** — Reproducible wgpu failure on the project's Iris Xe that survives the above
  mitigations (forced DX12 AND GLES both fail; Vulkan NULL-pipeline per IGCIT #1309 observed
  on our own repro), blocking the shipped feature set.
+ **R2** — A side-by-side prototype shows ≥1.5–2× sustained end-to-end frame-time advantage
  where the C++ comparison includes the full FFI/bridge cost on the same scene (~terrain
  chunks + ≥100k grass blades-equivalent + organisms), published numbers in this doc's
  successor.
+ **R3** — A required rendering feature lands outside wgpu's reach (e.g. hardware
  tessellation stage or vendor-specific extensions).
+ **R4** — Project changes target hardware/deployment model away from Iris Xe-class shared-
  memory iGPUs, invalidating the memory-pressure argument in §3.

If R1 or R2 triggers, re-evaluate: export raw Vulkan/ABI from the Rust layer first (thin C++
shim) before any full rewrite.

---

*Citations fetched 2026-10-30 unless noted. Claims marked **unverified** were not confirmed
from a primary page and must be re-verified before use.*
