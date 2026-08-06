---
type: review
date: 2026-08-05
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-08-05_warp-gameplay-pbf-dam-break-column.md
adrs: [ADR-0012]
aspects_touched: [warp-backend, gpu-nnps, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_game.py, pysph/base/tests/test_warp_game.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - gameplay PBF dam-break column

## Outcome

The studio now has a separate `Gameplay · fast` profile beside Adaptive and
Uniform. It is an isolated, explicitly approximate position-based fluid
solver with a fixed 60 Hz timestep, device hash-grid neighbors, three bounded
density projections, XSPH smoothing, analytic tank contact, and a bounded
impulse-coupled quaternion box. Adaptive remains the default; neither existing
WCSPH class nor generated equation source was changed.

This prototype passes its visual-stability, lifecycle, separation, and RTX
5090 frame-budget gates. A hands-on follow-up adds bounded Rich/Fast quality
presets, preemptive cancellation, and oriented-box floor contact. It is not
pressure, energy, hydrostatic, APR, or torque-validation evidence.

## Diff summary

- New host module: `pysph/base/warp_game.py` (989 lines).
- New focused host tests: `pysph/base/tests/test_warp_game.py` (189 lines).
- Studio changes add the third selector, approximation warning, gameplay
  controls/telemetry, worker routing, and a longer lifecycle smoke fixture.
- Hands-on review caught gameplay-only keys leaking into Adaptive/Uniform
  launch payloads. Configuration assembly now adds those keys only for
  `gameplay-pbf`; tests feed both scientific payloads directly through
  `DamBreakConfig.from_mapping` before the worker is spawned.
- Added ADR-0012, the approved/completed Tier-2 plan, the PBF primary-source
  reference, benchmark script/results, and implementation memory updates.
- Hands-on acceptance adds a `Rich` 7,020-particle local visual preset while
  retaining the measured 1,000-particle profile as `Fast`; gameplay spacing
  below `0.04 m` is rejected before worker launch.
- Cancellation now escalates from a brief cooperative request to process
  terminate/kill, so a long CUDA step cannot make the Cancel control inert.
- Rigid floor clamping now uses the rotated box's exact plane support extent,
  removing the bounding-sphere hover; the upright default starts at
  `z=0.10 m`.
- No dependency/build/release configuration, generic API/ABI, non-Warp host
  file, `warp_adaptive.py`, `warp_codegen.py`, or `warp_nnps.py` changed.

## Behavioral and numerical changes

- `solver_family=gameplay-pbf` selects the new fixed-resolution solver in the
  spawned worker. Missing/other solver-family values retain WCSPH routing.
- Gameplay pressure is intentionally zero/unavailable; the UI automatically
  switches to speed coloring and labels the approximation.
- Maximum applied projection travel is capped at `0.20 dx` per iteration and
  configuration validation guarantees the total allowance cannot outrun the
  neighbor skin between rebuilds.
- Fluid/body correction impulses are capped at `1.5 N·s` linear and
  `0.2 N·m·s` angular per substep; body angular speed is capped at `12 rad/s`.
  These are deliberate game-stability controls, not conservation claims.
- The `0.20 dx` tuning point gave lower frame-130 density-constraint RMS and a
  less energetic body response than `0.25/0.30 dx`. A four-second run remains
  finite with constraint RMS `0.07284` and body angular speed `1.167 rad/s`.
- Snapshots preserve `xyz`, `h`, `rho`, `p`, `speed`, `velocity`, `kind`, and
  `level`; NPZ and JSON identify `gameplay-pbf` and `approximate=true`.

## RTX 5090 evidence

Final benchmark: 1,000 fluid particles, three projections, one substep,
`dt=1/60 s`, XSPH coefficient `0.01`, 10 warm-up + 120 measured frames.

| Measurement | Result | Gate |
|---|---:|---:|
| Solver median | `1.049 ms` | `<=16.7 ms` pass |
| Solver p95 | `1.244 ms` | `<=33.3 ms` pass |
| Snapshot/readback median | `0.662 ms` | reported separately |
| Snapshot/readback p95 | `0.830 ms` | reported separately |
| Solver + snapshot median | `1.721 ms` | diagnostic |
| Solver + snapshot loop | `574.7 FPS` | not browser FPS |
| Compile + initialize | `97.189 ms` | diagnostic |

The primary artifact is `/tmp/pysph-gameplay-pbf-benchmark.json`. Transport,
JPEG/VTK work outside snapshot packing, and browser composition are excluded.

The separate Rich acceptance run used 7,020 fluid particles and completed 250
steps with solver median `2.345 ms`, surface pack median `0.580 ms`, and
WebGPU completion near `4.0 ms`. Its packed frame is `599,040 B`, so it is
local visual-quality evidence and does not replace the Fast transport gate.

## Raw validation output

```text
$ pytest -q pysph/base/tests/test_warp_game.py \
    pysph/base/tests/test_warp_adaptive.py \
    pysph/base/tests/test_warp_codegen.py \
    pysph/base/tests/test_warp_nnps.py
........................................................................ [ 91%]
.......                                                                  [100%]
79 passed, 2 warnings in 33.79s

$ pytest -q test_surface_transport.py test_studio.py
..................................                                       [100%]
34 passed

$ worker_smoke.py --gameplay
paused_once=true, stepped_once=true, resumed_once=true, status=completed
step=300, all_finite=true, rigid_device_error=0

$ worker_smoke.py
paused_once=true, stepped_once=true, resumed_once=true, status=completed
resolution_mode=adaptive, step=4, fluid_particles=2080, all_finite=true

$ python -m py_compile <all touched Python files>
[no output]

$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
[no output]
```

The warnings are Warp/Python 3.19-future ctypes layout deprecations from the
installed Warp package, not failures in these changes.

## Risks and unresolved questions

- PBF surface density error remains visible and limited corrections are
  frequent during violent flow. The controls make this bounded and auditable;
  they do not make it scientific SPH.
- The approximate body response does not exactly conserve energy or angular
  momentum. Linear/angular caps and damping are part of its product contract.
- CUDA atomic reduction order can vary the detailed body trajectory slightly;
  gates cover finite/bounded motion and shell rigidity, not replay identity.
- Browser presentation FPS and input-to-photon latency are unmeasured. Direct
  client GPU particle rendering is the logical next game-oriented experiment.
- The server currently streams all gameplay frames. Network/browser cost may
  require display decimation even though the solver is comfortably real-time.
- The Rich preset's `599,040 B` base64 frame exceeds the Fast preset's 96-KiB
  gate. Binary transport or display decimation is required before treating
  Rich as a remote profile.

## Visual aid

Waiver: the interactive studio itself is the relevant visual artifact; a
static diagram would not add information beyond its three labeled columns,
approximation warning, live viewport, and timing telemetry.

## Promotion state

Prototype-owner commit/push authorization is recorded verbatim below. This
does not authorize upstream promotion; exact `@prabhu: LGTM` remains required
for production/PR promotion.

## Owner verdict

Prototype-owner authorization by @kunalpuri-prediqt at
2026-08-06T17:04:31 CEST, verbatim:

> ok. commit and push

This authorizes the cumulative prototype commit and push to the owner's fork;
it is not upstream promotion approval.
