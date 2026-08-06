---
type: review
date: 2026-08-06
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-08-06_warp-terrain-sph-smooth-hill.md
adrs: [ADR-0015]
aspects_touched: [warp-backend, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_adaptive.py, pysph/base/tests/test_warp_adaptive.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - Terrain SPH procedural smooth hill

## Outcome

The studio now exposes a fifth `Terrain SPH` profile that runs genuine uniform
3D Warp WCSPH into a deterministic stationary Gaussian-hill boundary. The
collision hill consists of fixed solid particles; a separate smooth VTK actor
uses the same analytic parameters for presentation. Adaptive, Uniform,
Gameplay and Geospatial retain their existing solver/render routes.

The default 250-step `dx=0.1` hill run is finite, conserves fluid mass, keeps
the solid bitwise stationary, changes the flow relative to a flat reference,
and completes end to end in the editor browser. This is prototype interaction
evidence, not terrain-boundary convergence, real geospatial data, adaptive SPH,
or game-frame-rate evidence.

## Diff summary

- `warp_adaptive.py` adds bounded deterministic Gaussian hill construction,
  `terrain-wcsph`/`hill` configuration, fixed-solid initialization, and
  snapshot/metric/save identity without altering WCSPH equations.
- `test_warp_adaptive.py` covers deterministic sampling, tank/envelope/floor
  bounds, mass/smoothing length/density, finite evolution, stationary geometry
  and exact fluid-mass conservation.
- The studio adds the fifth two-row selector, isolated Terrain controls and
  warning, explicit worker routing/status, terrain metrics and saved-result
  restore.
- `vtk_scene.py` triangulates the same analytic center/height/radii into a
  4,753-point normal-generated earthy surface and hides its collision particles
  by default.
- Worker smoke, studio/scene tests, README, ADR-0015, matched comparison,
  current/daily/aspect/session memory and this review record the contract.
- Hands-on mode switching exposed and fixed a profile-state leak: after a
  Terrain result, `hill` could be forwarded to Gameplay and rejected. Terrain
  now forces `hill`, Gameplay forces `floating`, and scientific modes restore
  the user's prior `none`/`fixed`/`floating` selection.

## Visual/data flow

```text
hill center + height + radii
          |                     |
          v                     v
filled dx-lattice solids    smooth VTK mesh
          |                     |
          v                     v
existing fixed-solid WCSPH  editor presentation
```

The visual mesh cannot improve the collision model; both branches only share
the analytic parameters.

## Numerical and browser evidence

| Gate | Result |
|---|---:|
| Hill fluid / fixed-solid particles | `1,000 / 72` |
| 250-step finite state | pass |
| Relative fluid-mass drift | `0.0` |
| Hill coordinate drift | bitwise zero |
| Final particles changed by >1 mm | `503 / 1,000` |
| RMS / max position delta vs flat | `9.76 mm / 13.58 cm` |
| Surge-front delta vs flat | `-3.11 cm` |
| Warm flat / hill solver median | `5.34 / 8.13 ms` |
| Editor browser | `250 / 250`, completed |
| Cold browser wall time | `62.97 s` including ~1 min compile |

The editor screenshot was inspected at 1280×720: Terrain SPH is selected, the
pressure-coloured water front meets the smooth green leading slope, hill
metrics show 72 particles and configured `3.00 / 0.35 / 0.40 / 0.12 m`
parameters, and the run reports COMPLETED on the RTX 5090.

## Raw validation output

```text
$ python -m pytest -q pysph/base/tests/test_warp_adaptive.py \
    pysph/base/tests/test_warp_game.py \
    pysph/base/tests/test_warp_shallow_water.py \
    pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_nnps.py
88 passed, 2 warnings in 587.12s
```

The warnings are installed Warp ctypes layout deprecations under Python 3.14.

```text
$ PYTHONPATH=... python -m pytest -q test_studio.py \
    test_surface_transport.py test_geospatial_transport.py \
    test_prepare_nasadem.py
51 passed in 8.18s

$ node --test test_webgpu_surface.mjs test_webgpu_geospatial.mjs
2 suites passed, 0 failed

$ PYTHONPATH=... python worker_smoke.py --terrain
frames=13; pause=true; step=true; resume=true;
solver_family=terrain-wcsph; hill_particles=72; mass_drift=0.0;
all_finite=true

$ PYTHONPATH=... python experiments/.../run_comparison.py
hill_finite=true; hill_mass_drift_lte_1e_6=true;
hill_stationary=true; measurable_interaction=true
```

Generic preemptive cancellation remains covered by the studio worker unit test;
the Terrain worker smoke exercises the exact spawned pause/step/resume/save
path. Python compile checks, JavaScript syntax checks and `git diff --check`
pass.

The exact editor reproduction `Terrain SPH -> Gameplay -> Launch` now completes
250/250 with 7,020 fluid and 170 floating-body particles in 1.98 seconds and no
failure banner.

## Risks and unresolved questions

- Fixed solid particles approximate the analytic surface at `dx=0.1`; no
  boundary-resolution or convergence study exists.
- The 250-step front interacts with the leading slope but does not overtop the
  hill. Longer showcase dynamics would require a separately bounded run/tune.
- Warm WCSPH step cost is suitable for this local 1,000-particle prototype, but
  the cold generated-kernel compile dominates the first browser run and total
  wall telemetry does not separate it yet.
- NASADEM remains separately blocked on Earthdata authentication. This fixture
  must not be labelled as real terrain.
- Adaptive terrain SPH remains out of scope while variable-resolution
  consistency gates fail.

## Promotion state

Prototype-owner commit/push authorization is recorded verbatim below. Upstream
production/PR promotion still requires exact `@prabhu: LGTM`.

## Owner verdict

Prototype-owner authorization by @kunalpuri-prediqt at
2026-08-06T17:04:31 CEST, verbatim:

> ok. commit and push

This authorizes the cumulative prototype commit and push to the owner's fork;
it is not upstream promotion approval.
