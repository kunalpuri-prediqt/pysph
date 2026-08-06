---
type: plan
id: 2026-08-06_warp-nasadem-shallow-water-geospatial-demo
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-06T11:21:29 CEST
status: approved
depends_on:
  - 2026-08-05_warp-gameplay-pbf-dam-break-column
  - 2026-08-05_warp-webgpu-fluid-surface-renderer
adrs: [ADR-0012, ADR-0013, ADR-0014]
aspects: [warp-backend, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_shallow_water.py
  - pysph/base/tests/test_warp_shallow_water.py
within_boundary: true
---

# Plan: NASADEM shallow-water geospatial demo

## Goal

Add an isolated fourth studio profile that runs an interactive Warp
shallow-water simulation over a real, provenance-preserving NASADEM terrain
crop and renders it as a cinematic WebGPU landscape. The initial water event,
dam barrier, and breach are explicitly synthetic; the feature is not an
operational flood forecast or historical reconstruction.

## Product contract

| Profile | Domain/model | Claim boundary |
|---|---|---|
| Adaptive | 3D variable-resolution WCSPH | experimental scientific ladder |
| Uniform | 3D fixed-resolution WCSPH | scientific comparison |
| Gameplay | 3D fixed-budget PBF | visual interaction |
| Geospatial | 2D finite-volume SWE on NASADEM | regional terrain-flow prototype |

Geospatial output must identify `solver_family=geospatial-swe`, NASADEM source
metadata, a `synthetic_breach=true` flag, grid resolution, metric cell size,
and the limitations of a 30 m surface DEM.

## Approach

### 1. Add a reproducible real-terrain asset

- Use official `NASADEM_HGT.001` data for tile `N43E006` and crop a bounded
  region around the Malpasset valley, France. The UI must say “synthetic
  breach—not a historical reconstruction.”
- Add a preparation script that accepts the externally downloaded official
  tile, verifies its expected format, reprojects/crops to a local metric grid,
  records raw/cropped checksums, and emits a compact static terrain asset plus
  JSON provenance manifest.
- Record geographic bounds, source URL/DOI, acquisition date, source product,
  1-arc-second posting, EGM96 vertical datum, horizontal CRS/local origin,
  processing version, nodata treatment, and conditioning operations.
- Commit only the bounded processed crop and manifest, not credentials or the
  full 1-degree source tile. If official acquisition is blocked by Earthdata
  authentication, stop at the deterministic synthetic fixture and report the
  external-data blocker rather than substituting an unattributed DEM.

### 2. Implement the Warp shallow-water solver

- Create `pysph/base/warp_shallow_water.py` with a configuration and worker
  lifecycle compatible with the existing incremental solvers: `initialize`,
  `step`, `done`, `snapshot`, `metrics`, and `save`.
- Evolve conservative cell state `(h, hu, hv)` on a regular local-metric grid
  with double-buffered Warp arrays and a finite-volume Rusanov flux.
- Use hydrostatic reconstruction/bed-source balancing so a lake at rest over
  variable topography remains at rest to an explicit tolerance.
- Preserve non-negative depth across wet/dry fronts, zero momentum in dry
  cells, enforce a CFL-bounded timestep, and expose explicit closed/outflow
  boundary and Manning-friction controls.
- Initialize a reservoir water surface against a synthetic barrier, then open
  a configurable breach at `t=0`. Terrain and barrier are distinct arrays so
  the original DEM is never silently modified.
- Keep the hot step device-resident. Pull only bounded diagnostics and display
  snapshots at configured cadence.

### 3. Add compact terrain/water transport

- Send static terrain/provenance once per run and dynamic water depth/speed at
  display cadence through a versioned geospatial frame schema.
- Quantize bounded display fields explicitly (initially float16 or uint16 with
  scale/offset metadata); validate decoding error against the unquantized
  snapshot and reject malformed/non-finite frames.
- Separate and report solver, snapshot/readback, packing, wire bytes, browser
  upload, and renderer completion. Do not infer browser FPS from solver timing.
- Preserve bounded replay without retaining unbounded regional arrays.

### 4. Render a cinematic WebGPU landscape

- Add an isolated client WebGPU geospatial renderer rather than coupling the
  regular-grid path to the particle surface shader.
- Render terrain as a metric heightfield with slope/elevation-derived material,
  physically coherent normals, sun/sky lighting, atmospheric distance fog,
  and an optional solver-independent display-only vertical exaggeration.
- Render a continuous water heightfield over wet cells with thickness tint,
  grazing Fresnel, broad highlights, velocity/wet-front foam, shoreline fade,
  and terrain-aware occlusion. No ray tracing is required.
- Provide orbit/pan/zoom, reset, downstream/overview camera presets, playback,
  water-depth/speed debug views, and a Showcase mode that collapses studio
  sidebars without losing status or timeline access.
- Keep current WebGPU particle surface, VTK scientific views, and their
  fallbacks unchanged.

### 5. Integrate the fourth profile additively

- Extend worker routing only for `solver_family=geospatial-swe`.
- Add Geospatial selection, terrain/breach/solver controls, data provenance,
  solver health, wet area, volume drift, peak depth/speed, timing, and renderer
  telemetry. Disable irrelevant particle/refinement/rigid-body controls.
- Keep Adaptive as the initial studio profile and preserve all saved scientific
  and gameplay configuration when switching away and back.
- Save a self-describing NPZ/JSON result with terrain asset identity and enough
  state to restore the final view. Existing result loading remains compatible.

## Expected files

Host files are limited to the two boundary-approved `warp_*` paths in the
frontmatter. Additive implementation files are expected under:

- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/worker.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/geospatial_transport.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/assets/webgpu_geospatial.js`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/assets/studio.css`
- focused Python/Node tests, terrain preparation script, processed terrain
  asset/provenance, README, reference, experiment, review, and memory files.

## Tests and acceptance gates

### Terrain/provenance

- Processed crop dimensions, geographic bounds, local metric spacing, finite
  heights, checksum, datum, source product, and processing history match its
  manifest.
- Conditioning is deterministic and preserves a separately checksummed raw
  crop. Display vertical exaggeration never changes solver bed elevation.

### Numerical

- Lake at rest over non-flat bed: maximum velocity and free-surface drift stay
  below explicit fp32 tolerances after 200 steps.
- Closed-boundary wet case: relative water-volume drift `<=1e-4` after 500
  steps; dry cells never have negative depth or non-finite momentum.
- One-dimensional flat-bed dam break matches an independent CPU reference
  profile within a declared L1 tolerance and propagates in the expected
  direction.
- A deliberately invalid CFL/configuration fails clearly. Wet/dry, friction,
  boundary, save/restore, and worker cancel/pause/step paths have focused tests.

### Performance/presentation

- Default committed grid is bounded at 256x256 unless measurement justifies a
  different size. On the RTX 5090, warm solver median `<=16.7 ms` and p95
  `<=33.3 ms` over at least 120 frames.
- At a 1280x720 editor Edge viewport, WebGPU completion median `<=16.7 ms` and
  p95 `<=33.3 ms` after warmup. Upload and transport are reported separately.
- Wire bytes/frame and sustained local cadence are reported. Missing a gate is
  recorded as a failed prototype gate, not hidden by time dilation or sparse
  telemetry.
- Hands-on acceptance inspects overview/downstream cameras, breach opening,
  wet-front motion, terrain/water debug targets, replay, resize, Showcase mode,
  and fallback behavior.

### Separation

- Existing Adaptive, Uniform, Gameplay PBF, WebGPU particle-surface, generated
  code, NNPS, studio, transport, and cancellation regressions pass unchanged.
- No generic PySPH API/ABI, dependency/build/release configuration, or
  non-Warp behavior changes.

## Risks

- Official NASA Earthdata acquisition may require owner authentication. No
  credential may enter the repository or memory; data provenance is a kill
  gate, not an invitation to use a convenient third-party tile silently.
- Hydrostatic reconstruction and wet/dry positivity are necessary but do not
  make this a validated hazard model. Resolution, bathymetry, structures,
  infiltration, breach hydrograph, and calibration remain absent.
- Base64 transport can dominate a 256x256 grid even after quantization. The
  first slice measures and bounds it; binary transport is a likely follow-up.
- SRTM/NASADEM can carry vegetation/surface bias and cannot resolve the dam or
  narrow channels. Visual smoothing must not alter solver terrain silently.
- A beautiful renderer can overstate fidelity. Provenance and synthetic-event
  warnings remain visible in both controls and saved results.

## Out of scope

- Operational inundation forecasting, evacuation guidance, historical-event
  reconstruction, calibration, uncertainty quantification, or regulatory use.
- Real reservoir bathymetry, surveyed dam/breach geometry, rainfall/runoff,
  infiltration, sediment, erosion, debris, buildings, bridges, or casualties.
- Coupling the regional SWE grid to the 3D PBF/WCSPH solvers in this slice.
- Arbitrary live global tile search/download, Earthdata credential management,
  GIS authoring, map-server deployment, binary networking, or multiplayer.
- Ray tracing, path tracing, OptiX, satellite-image licensing work, or new
  Python/npm/runtime dependencies.

## Estimated effort

Large Tier-2 prototype: staged terrain acquisition/provenance, numerical core,
transport, WebGPU rendering, studio integration, hardware acceptance, and
review. Kill gates are official data provenance, lake-at-rest/positivity, and
bounded end-to-end local performance.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-08-06T11:24:57 CEST
- Approval, verbatim quote:
  > approved
