---
type: plan
id: 2026-08-05_warp-gameplay-pbf-dam-break-column
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-05T14:00:00 CEST
status: completed
depends_on:
  - 2026-07-31_trame-warp-dam-break-studio
  - 2026-08-04_warp-adaptive-floating-body-production
adrs: [ADR-0003, ADR-0005, ADR-0006, ADR-0008, ADR-0012]
aspects: [warp-backend, gpu-nnps, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_game.py
  - pysph/base/tests/test_warp_game.py
within_boundary: true
---

# Plan: gameplay PBF dam-break column

## Goal

Add a third `Gameplay · fast` column beside `Adaptive · two level` and
`Uniform` in the Warp studio. It will run a dedicated GPU position-based fluid
profile with an approximate floating box and a measurable interactive frame
budget, while leaving both scientific WCSPH paths unchanged.

## Product contract

The three columns answer different questions:

| Column | Numerical contract | Primary gate |
|---|---|---|
| Adaptive | variable-resolution WCSPH prototype | scientific ladder |
| Uniform | mass-matched WCSPH reference | scientific ladder |
| Gameplay | fixed-budget PBF approximation | visual stability + frame time |

Gameplay telemetry and manifests must carry an `approximate` warning and the
solver family `gameplay-pbf`. Its results cannot be presented as hydrostatic,
pressure, energy, or torque validation evidence.

## Approach

### 1. Add an isolated gameplay solver

- Create `pysph/base/warp_game.py` with a configuration and incremental
  simulation interface matching the worker-facing subset of
  `WarpDamBreakSimulation`: `initialize`, `step`, `done`, `snapshot`,
  `metrics`, and `save`.
- Reuse the existing dam-break initial particle geometry and snapshot schema,
  but retain only fixed-resolution fluid state and analytic tank boundaries in
  the gameplay hot path.
- Keep persistent Warp arrays for current/predicted position, velocity,
  density constraint, Lagrange multiplier, position correction, and error
  flags. No particle mutation or host pull is permitted inside a solver step.

### 2. Implement bounded PBF projection

- Predict one display step from velocity and gravity.
- Refresh a Warp `HashGrid` once on predicted device positions. The existing
  `UniformGridWarpNNPS.update(push=False)` still reads device coordinates back
  to the host to recompute bounds, so it cannot satisfy this plan's no-hot-path
  readback gate.
- For a configurable fixed iteration count (default 3), compute density/
  constraint gradients, solve regularized multipliers, and apply symmetric
  position corrections with bounded artificial pressure.
- Project the tank planes every iteration and guard maximum correction against
  a spacing-relative allowance. Fail clearly rather than silently using stale
  neighbors when the allowance is exceeded.
- Reconstruct velocity from corrected displacement and apply configurable
  XSPH-like damping. Use fixed `dt=1/60 s` by default; any substep count is an
  explicit cost knob.

### 3. Add an approximate interactive rigid box

- Represent the box with persistent centre, linear/angular velocity,
  quaternion, mass, and inertia, while rendering the existing shell sampling.
- Push penetrating fluid particles out of the oriented box and accumulate
  equal/opposite bounded correction impulses and torque in deterministic f64
  body reductions.
- Advance the body with gravity, damping, impulse response, and analytic tank
  contact. This is a game interaction model, not Liu pressure coupling.
- Record impulse clamp activity and body/bounds error telemetry so visual
  stability is auditable.

### 4. Route the worker and studio additively

- Extend the studio worker to select `WarpGameplayDamBreakSimulation` only
  when `solver_family=gameplay-pbf`; otherwise instantiate the existing
  `WarpDamBreakSimulation` exactly as today.
- Add the third segmented-button column and a short approximation badge/help
  panel. Selecting Gameplay applies game defaults without overwriting saved
  Adaptive/Uniform values when switching back.
- Surface simulated FPS, solver-frame median/p95, display/readback frame time,
  constraint error, projection iterations, and clamp count. Preserve
  pause/resume/single-step/cancel/replay and Pressure/Speed coloring; pressure
  is unavailable in Gameplay and automatically falls back to Speed.
- Keep output/restart schemas self-describing with solver family and profile
  parameters. Scientific result loading remains backward compatible.

### 5. Validate correctness, separation, and performance

- Independent NumPy fixture: density constraint error decreases after
  projection and pair corrections are symmetric before boundaries/damping.
- Bounds fixtures: floor, corner, and box projection remain finite; maximum
  correction guard fires on a deliberately invalid step.
- Fluid/body fixture: the box receives the opposite bounded impulse, moves in
  the expected direction, preserves its rigid shell, and stays within tank
  bounds over a short run.
- Source/separation tests prove Gameplay does not alter generated WCSPH source,
  `DamBreakConfig`, adaptive/uniform configuration, or their default UI state.
- Run focused gameplay tests, full adaptive/controller tests, studio tests,
  and relevant Warp NNPS/codegen regressions.
- Benchmark a warm 120-frame default run on the RTX 5090. Report compile time,
  solver-only median/p95, snapshot/readback time, streamed end-to-end FPS,
  particle count, iterations, and simulated/wall-time ratio.

## Acceptance gates

- Existing Adaptive and Uniform numerical/source tests remain unchanged within
  current tolerances; default app mode remains Adaptive.
- Gameplay state stays finite, inside analytic bounds, and below its explicit
  correction/rigid-error thresholds for the 120-frame case.
- Density-constraint RMS decreases across projection iterations on the
  independent fixture and the dam-break run.
- The approximate body visibly translates/rotates under a directed particle
  impact and preserves shell geometry within `1e-5` relative drift.
- Warm RTX 5090 solver-only frame time: median `<=16.7 ms`, p95 `<=33.3 ms` at
  the committed default particle count and three iterations. This is a failed
  gate, not a reason to hide work through time dilation, if it misses.
- Snapshot/readback and streamed FPS are reported separately. No claim of
  60 FPS is made from solver timing alone.
- Studio controls and saved-result loading pass; no NaN/device error or worker
  lifecycle regression.

## Expected files

Host changes are limited to the two `warp_*` files in frontmatter. Additive
studio changes are expected in:

- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/worker.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py`
- the studio README and experiment evidence.

ADR/reference, experiment, review, and session-memory files remain under the
implementation `.ai/` tree.

## Risks

- Three PBF iterations may be visually stable but too compressible; iteration
  count must remain visible rather than silently raised.
- Rebuilding neighbors only once per display substep is safe only while
  corrections remain bounded by the tested skin allowance.
- Approximate two-way body impulses can inject energy. Clamp/damping telemetry
  and directed fixtures are mandatory.
- At small particle counts, Python launch and server-side VTK/JPEG costs may
  dominate GPU math. The timing decomposition must identify that honestly.
- A direct GPU renderer is out of scope for this first column, so end-to-end
  browser FPS may miss even if solver FPS passes.

## Out of scope

- Replacing or weakening Adaptive/Uniform WCSPH.
- Scientific equivalence between PBF and WCSPH pressure, energy, torque, or
  hydrostatic results.
- Dynamic split/merge in Gameplay.
- FLIP/APIC, surface meshing, foam/spray, WebGPU/native rendering, multiplayer
  determinism, or engine integration.
- Generic PySPH public API/ABI changes or a production dependency declaration.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-08-05T14:01:32 CEST
- Approval, verbatim quote:
  > approved

## Outcome

Completed on 2026-08-05. The gameplay column is additive and the default
studio mode remains Adaptive WCSPH. The final profile uses a device Warp
`HashGrid`, three bounded PBF projections, explicit XSPH smoothing, analytic
tank contact, and bounded linear/angular gameplay-body response. The final
benchmark and validation evidence are recorded in experiment
`2026-08-05_warp-gameplay-pbf-dam-break`.
