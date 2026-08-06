---
type: experiment
id: 2026-08-06_warp-nasadem-shallow-water
created: 2026-08-06T11:25:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: complete-with-external-data-blocker
last_checked: 2026-08-06T12:20:00 CEST
plan: 2026-08-06_warp-nasadem-shallow-water-geospatial-demo
---

# Experiment: Warp NASADEM shallow-water geospatial profile

## Purpose

Validate a bounded 256² conservative Warp shallow-water profile, compact
heightfield transport, and WebGPU terrain/water renderer without conflating
the synthetic event with historical or operational flood modelling.

## Data gate

CMR resolved official granule `NASADEM_HGT_n43e006`
(`G2816791562-LPCLOUD`) and the DOI/product metadata. The protected Earthdata
Cloud granule URL returned HTTP `401` because this environment has no Earthdata
Login credentials. Per the approved plan, no third-party DEM was substituted.

`prepare_nasadem.py` is ready for the owner-supplied official archive. It
strictly validates the 3601² big-endian HGT member, crops/resamples the bounded
AOI, records source/crop/asset checksums, CRS, EGM96 datum and processing, and
keeps the synthetic barrier and initial water separate. Until that asset is
present, the application visibly uses `synthetic-valley-fixture`.

## Numerical results

The exact spawned-worker smoke ran 300 steps at 256² on Warp 1.15.0, CUDA
driver 13.2, NVIDIA GeForce RTX 5090.

| Measurement | Result | Gate |
|---|---:|---:|
| Solver median | `0.346 ms` | `<=16.7 ms` pass |
| Solver p95 | `1.425 ms` | `<=33.3 ms` pass |
| Simulated / solver wall | `371.4×` | report |
| Relative volume drift | `6.04e-8` | `<=1e-4` pass |
| Peak depth / speed | `28.19 m / 18.86 m/s` | diagnostic |
| Final wet cells | `17,765 / 65,536` | diagnostic |
| Pause / step / resume | passed | pass |

Focused numerical oracles cover a 200-step lake at rest over non-flat bed,
500-step closed-boundary conservation and positivity, an independent fp64
one-dimensional Rusanov reference, wet-front direction, Manning damping,
closed/outflow policies, invalid CFL, deterministic synthetic terrain,
separate barrier storage, and pickle-free save/restore.

## Transport/browser results

The completed 600-step editor-browser run used a 593×656 CSS-pixel canvas at
device scale, 256² terrain/water grids, and 341.3 KiB base64 dynamic frames.
After the static-terrain/dynamic-frame ordering race was fixed, Edge reported
WebGPU `READY` and no shader fallback.

| Measurement | Result | Gate |
|---|---:|---:|
| Server pack median / p95 | `0.39 / 4.39 ms` | report |
| Browser upload median / p95 | `0.10 / 0.30 ms` (prior warm run) | report |
| WebGPU completion median / p95 | `7.70 / 19.80 ms` | `<=16.7 / <=33.3 ms` pass |
| Packed dynamic frame | `341.3 KiB` | bounded/report |
| Final relative volume drift | `1.19e-7` | pass |

Hands-on acceptance verified selection, launch, 600/600 completion, retained
timeline, orbit/reset canvas lifecycle, telemetry, warning text, no VTK/JPEG
work on geospatial frames, and a successful actual renderer recovery from the
fast-frame ordering race. The synthetic fixture remains visually smooth; real
ridge/drainage detail is explicitly blocked on the authenticated NASADEM tile.

## Validation summary

- Warp separation: `86 passed`, two installed-Warp ctypes deprecations.
- Studio/transport/importer: `47 passed`.
- WebGPU surface contracts: `6 passed`.
- WebGPU geospatial contracts: `5 passed`.
- Python compile and JavaScript syntax checks: passed.
- Exact 256² spawned worker: passed.

## Claim boundary

This is a depth-averaged regional terrain-flow prototype with a synthetic
barrier/breach. It omits bathymetry, surveyed dam geometry, calibrated breach
hydrograph, infiltration, rainfall/runoff, buildings and uncertainty. Timing
does not turn it into an operational inundation model.
