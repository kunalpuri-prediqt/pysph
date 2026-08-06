---
type: plan
id: 2026-08-05_warp-webgpu-fluid-surface-renderer
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-05T14:45:00 CEST
status: completed
depends_on: [2026-08-05_warp-gameplay-pbf-dam-break-column]
adrs: [ADR-0012, ADR-0013]
aspects: [warp-backend, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_game.py
  - pysph/base/tests/test_warp_game.py
within_boundary: true
---

# Plan: WebGPU gameplay fluid-surface renderer

## Goal

Turn the fast gameplay particle cloud into a visually continuous, richly
shaded water surface in the browser while preserving the existing VTK particle
view and every Adaptive/Uniform scientific path.

## Product contract

Gameplay receives two explicit visualization modes:

| View | Purpose | Renderer |
|---|---|---|
| Fluid surface | visually rich interactive water | client WebGPU |
| Particles | solver/debug inspection and fallback | existing server VTK |

Adaptive and Uniform remain particle/scalar scientific views. Surface mode is
display-only: it cannot alter simulation particles or be used as numerical
evidence.

## Approach

### 1. Generate render-only ellipsoid attributes in Warp

- Add persistent arrays for smoothed display position, covariance, principal
  axes, anisotropic scale, and neighbor count to the isolated gameplay solver.
- At snapshot cadence, use the current device hash grid to compute a bounded
  Yu--Turk-style neighborhood covariance and symmetric eigendecomposition.
- Fall back to isotropic scale for sparse or degenerate neighborhoods. Clamp
  every axis to an explicit spacing-relative interval and verify a finite,
  right-handed orthonormal frame.
- Do not feed smoothing or anisotropy into `x`, velocity, density projection,
  rigid coupling, or saved scientific observables. Record render parameters in
  the gameplay config/manifest.

### 2. Add a versioned packed gameplay-frame transport

- Pack fluid render centers, axes/scales, speed, and rigid pose into a compact
  float32 payload with version/count metadata; base64 is acceptable for this
  first local Trame slice but pack time and byte rate must be reported.
- Publish only in Gameplay + Fluid surface mode. Skip VTK scene update and JPEG
  generation on that path so server rendering does not remain hidden in the
  measured client loop.
- Preserve frame buffering/replay. A replayed surface frame must produce the
  same packed payload, and switching back to Particles must immediately render
  the retained snapshot through VTK.

### 3. Implement the local WebGPU multipass renderer

- Add a self-contained ES module loaded through Trame's client `Handler`; no
  CDN, npm bundle, or new Python/runtime dependency.
- Feature-detect `navigator.gpu`, request a device, handle resize/device loss,
  and report explicit capability/error state. Auto-fallback to Particles on
  failure.
- Render analytic ellipsoid splats into nearest-depth and additive-thickness
  textures, apply bounded bilateral depth smoothing, and reconstruct eye-space
  normals from neighboring samples.
- Composite water with Fresnel reflection, screen-space refraction,
  Beer--Lambert thickness absorption, shallow/deep tint, directional/specular
  light, and a procedural environment. Draw tank context and the rigid box.
- Provide orbit/pan/zoom/reset controls local to the canvas and keep camera
  behavior independent of the VTK view.

### 4. Add controls and measured telemetry

- Add `Fluid surface | Particles`, water color/absorption, refraction,
  roughness, splat scale, smoothing radius/iterations, and thickness controls
  under Gameplay visualization.
- Report client renderer, WebGPU adapter availability, packed bytes/frame,
  server pack median/p95, client upload median/p95, render/presentation
  median/p95, and dropped/stale frames. Do not reuse solver FPS labels.
- Keep a visible “screen-space approximation” note and expose a debug view for
  raw depth, thickness, normals, and final composite.

### 5. Validate the full path

- Warp oracles: solver positions are byte-identical before/after render
  attribute generation; centers/axes/scales are finite; frames orthonormal and
  right-handed; scales positive/bounded; sparse fallback deterministic.
- Transport oracles: header/version/count/length validation, round-trip exact
  float32 data, malformed payload rejection, replay identity, and gameplay-only
  publication.
- Node tests cover frame decode, camera matrices, resize math, shader-module
  presence/contracts, and WebGPU-unavailable fallback without needing a GPU.
- Browser acceptance on the editor Edge tab captures final/depth/thickness/
  normal views, camera interaction, resize, mode switching, and fallback.
- Re-run gameplay, studio, adaptive, generated-code, and NNPS regressions.

## Acceptance gates

- The fluid reads as one continuous body in the initial column, impact sheet,
  and settled pool; no persistent particle-sized holes at default camera and
  1,000-particle profile.
- Thin sheets do not collapse into grossly oversized blobs; anisotropy remains
  within configured bounds and never creates NaN/negative axes.
- Fluid surface mode does not call server VTK update/JPEG generation per live
  frame. Particles mode and all scientific views remain unchanged.
- WebGPU failure visibly and automatically returns to the particle view; the
  app remains usable.
- Editor-browser presentation at 1280x720, default 1,000 particles: median
  `<=16.7 ms`, p95 `<=33.3 ms` after warmup. Upload, render, and presentation
  timings are separated; no server-loop number substitutes for browser FPS.
- Packed gameplay frame is `<=96 KiB` at the default particle count and server
  pack median is `<=2 ms` on the RTX 5090 host.
- Existing gameplay solver frame gate remains green and the 240-frame run stays
  finite. Adaptive remains the default app mode and exact scientific source
  regressions pass.

## Expected files

- `pysph/base/warp_game.py`
- `pysph/base/tests/test_warp_game.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/assets/studio.css`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/assets/webgpu_surface.js`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_webgpu_surface.mjs`
- studio README, ADR/reference, experiment, review, and session-memory files.

## Risks

- Browser WebGPU cannot directly consume Warp CUDA memory; base64 state
  transport adds copies and allocation. This first slice measures the cost and
  leaves binary websocket/shared-native rendering as a later optimization.
- Browser automation may not expose a hardware adapter in headless mode, so
  shader/unit tests and hands-on GPU acceptance are both required.
- Screen-space refraction lacks offscreen information and can show edge
  artifacts. Debug targets and bounded parameters make failures diagnosable.
- Anisotropic covariance is unstable for sparse neighborhoods unless the
  isotropic fallback and eigenvalue clamps are strict.
- Raw WebGPU lifecycle/device-loss/camera code is larger than VTK integration;
  the ES module must remain isolated and feature-gated.

## Out of scope

- Ray tracing, OptiX, Vulkan, path tracing, denoising, or cinematic export.
- True secondary foam/spray/bubble particles, vorticity-confinement dynamics,
  surface tension, cohesion, adhesion, or FLIP/APIC.
- Replacing VTK for Adaptive/Uniform or changing scientific result schemas.
- Remote-network bandwidth optimization or CUDA/browser zero-copy.
- A production dependency/build declaration or generic PySPH public API.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-08-05T14:46:00 CEST
- Approval, verbatim quote:
  > approved
