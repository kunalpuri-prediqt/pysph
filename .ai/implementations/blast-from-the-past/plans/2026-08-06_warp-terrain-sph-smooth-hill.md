---
type: plan
id: 2026-08-06_warp-terrain-sph-smooth-hill
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-06T16:10:17 CEST
status: approved
depends_on:
  - 2026-07-31_warp-two-level-adaptive-dam-break-smoke
adrs: [ADR-0015]
aspects: [warp-backend, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_adaptive.py
  - pysph/base/tests/test_warp_adaptive.py
within_boundary: true
---

# Plan: Terrain SPH with a procedural smooth hill

## Goal

Add an isolated fifth `Terrain SPH` studio profile in which the existing 3D
uniform Warp WCSPH dam-break flows into and around a deterministic smooth hill.
The hill must be a physical stationary solid boundary and a smooth rendered
surface. It is a procedural development fixture, not NASADEM.

## Product contract

| Profile | Solver | Terrain/obstacle |
|---|---|---|
| Adaptive | 3D two-level WCSPH | established tank/selected obstacle |
| Uniform | 3D uniform WCSPH | established tank/selected obstacle |
| Gameplay | approximate 3D PBF | tank/floating box |
| Geospatial | 2D finite-volume SWE | synthetic/NASADEM heightfield |
| Terrain SPH | 3D uniform WCSPH | procedural Gaussian hill |

Terrain SPH output will identify `solver_family=terrain-wcsph`,
`resolution_mode=uniform`, and `obstacle_mode=hill`. It must not be presented
as geospatial, adaptive-resolution, or game-frame-rate evidence.

## Approach

### 1. Add a deterministic physical hill

- Extend `DamBreakConfig` with explicit hill center, height and horizontal
  Gaussian radii; accept `obstacle_mode=hill` without changing `fixed`.
- Sample a bounded filled mound on the existing `dx` lattice for stationary
  WCSPH solid particles. Exclude the floor layer already represented by the
  tank wall and keep all points inside the tank.
- Use the established fixed-solid WCSPH equations/NNPS path; do not add a new
  SPH formulation or alter fluid equations.
- Include hill parameters and particle count in snapshot/metrics/save output.

### 2. Render the same hill smoothly

- Add a dedicated VTK hill surface actor derived from the same analytic
  parameters, with normals and an earthy material.
- Keep the collision particles available for a debug/fallback path, but show
  the smooth actor by default in Terrain SPH.
- Preserve current box, floating-body, tank, WebGPU gameplay surface and
  WebGPU geospatial renderer behavior.

### 3. Add the Terrain SPH studio profile

- Add a fifth mode with a clearly labelled uniform-WCSPH warning and bounded
  defaults (`dx=0.1`, 250 steps).
- Route `solver_family=terrain-wcsph` explicitly through the existing
  `WarpDamBreakSimulation`; disable adaptive, floating-body, geospatial and
  gameplay-only controls.
- Expose hill center/height/radii in a focused `Terrain` panel, preserve other
  profile settings while switching, and retain pressure/speed coloring,
  playback, cancellation and saved-result restore.

### 4. Validate before visual tuning

- Unit-test deterministic finite hill samples, tank bounds, analytic envelope,
  mass/smoothing length and no overlap with the floor layer.
- Run a uniform WCSPH hill smoke and assert finite state, stationary hill
  coordinates, conserved fluid mass and a changed downstream flow relative to
  the no-hill reference.
- Exercise exact worker pause/step/resume/cancel/save routing.
- Run existing Warp/studio/WebGPU separation suites unchanged, then inspect an
  actual editor-browser run and record solver/render cost separately.

## Expected files

- `pysph/base/warp_adaptive.py`
- `pysph/base/tests/test_warp_adaptive.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/worker.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/vtk_scene.py`
- focused studio/worker tests, README, experiment, review and memory files

## Acceptance gates

- Hill samples are deterministic, finite, bounded, stationary and contained
  below the analytic Gaussian envelope to an explicit `dx` tolerance.
- A 250-step `dx=0.1` Terrain SPH run remains finite with relative fluid mass
  drift `<=1e-6`; hill geometry drift is exactly zero on the host snapshot.
- Flow/hill interaction differs measurably from the same no-hill uniform run;
  this is an interaction smoke, not a validation against field data.
- The rendered analytic surface matches configured center/height/radii, is
  visibly smooth in the editor, and does not masquerade as NASADEM.
- Existing Adaptive/Uniform/Gameplay/Geospatial, cancellation, transport,
  generated-code and NNPS regressions pass unchanged.
- No generic API/ABI, dependency/build/release configuration, generated
  equation source or non-Warp behavior changes.

## Risks and limits

- Fixed boundary particles approximate a smooth surface at `dx`; visual
  smoothness does not increase collision resolution.
- Uniform WCSPH may be slower than the gameplay PBF and regional SWE modes.
- A one-layer/filled fixed-solid treatment is an engineering boundary model,
  not terrain-boundary convergence evidence.
- Current adaptive WCSPH interface gates fail, so adaptive terrain SPH is out
  of scope for this slice.
- The procedural hill does not remove the Earthdata authentication requirement
  for later NASADEM work.

## Out of scope

- NASADEM download/import, arbitrary heightfields, full regional SPH, erosion,
  sediment, porous terrain, moving ground, adaptive terrain resolution, PBF/
  SWE coupling, or a new WebGPU SPH surface renderer.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-08-06T16:14:04 CEST
- Approval, verbatim quote:
  > approved
