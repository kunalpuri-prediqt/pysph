---
type: decision
id: ADR-0012
date: 2026-08-05
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0003, ADR-0005, ADR-0006, ADR-0008, ADR-0011]
depends_on: [ADR-0003, ADR-0005]
conflicts_with: []
---

# ADR-0012: Adopt a separate position-based gameplay solver profile

## Context

The adaptive WCSPH floating-body demo advances approximately `0.357 s` of
physical time in about `50 s` on the current RTX 5090 checkpoint. Its acoustic
and contact timestep restrictions require roughly 42 solver steps per 60 Hz
display interval. Host-orchestrated adaptation and frame readback add further
latency. Those costs are appropriate to a scientific prototype but miss an
interactive-game budget by orders of magnitude.

Real-time particle-fluid systems such as NVIDIA FleX use a position-based
solver with a small, fixed iteration budget and explicitly trade physical
accuracy for stability and interactivity. Relabeling a coarser WCSPH run as
"game mode" would preserve the same timestep architecture and conceal that
trade.

## Proposed decision

Add a separate, explicitly approximate `Gameplay` solver profile alongside
the existing `Adaptive` and `Uniform` scientific profiles:

- use fixed-resolution, fixed-capacity fluid particles and analytic tank
  planes; do not split, merge, or evolve a WCSPH equation of state;
- use a Position-Based Fluids density constraint with a fixed `1/60 s` display
  step, a bounded number of projection iterations, artificial pressure, and
  optional XSPH-like velocity smoothing;
- keep particle state and projection work on the GPU between display frames,
  rebuilding a Warp `HashGrid` directly from device positions once per
  substep;
- model the gameplay obstacle as an approximate floating rigid box receiving
  bounded impulses from fluid position corrections, with gravity, damping,
  and analytic tank contact;
- publish/read back snapshots only at display cadence and label all manifests,
  telemetry, and UI copy as `gameplay-pbf` / approximate;
- preserve the scientific WCSPH classes, generated kernels, validation gates,
  and default Adaptive selection unchanged.

## Rationale

This establishes an honest architectural comparison. The scientific columns
answer whether the discretization is faithful; Gameplay answers whether a
stable and visually useful dam-break interaction fits a frame budget. A fixed
projection iteration count makes cost predictable, avoids the acoustic CFL
restriction, and permits visibly incompressible motion at much larger display
steps. Reusing Warp's device hash grid, the existing worker protocol, snapshot
schema, and VTK scene keeps the experiment focused on solver architecture.

## Alternatives considered

- **Coarsen/lower `c0` in WCSPH and call it game mode.** Rejected: it remains
  acoustically restricted, becomes more compressible, and does not establish a
  distinct frame-budget architecture.
- **Implement FLIP/APIC first.** Deferred: a pressure grid, particle-grid
  transfer, sparse active cells, and boundary solve are a larger first slice.
- **Use only ballistic particles.** Rejected: it would be fast but would not
  preserve a meaningful fluid-volume constraint.
- **Replace the scientific solver.** Rejected: gameplay approximations must be
  additive and must not weaken the WCSPH evidence path.

## Consequences

- Gameplay results are not eligible for the hydrostatic, pressure, torque, or
  APR convergence claims used by Adaptive/Uniform.
- Approximate fluid/body projection will not conserve energy or angular
  momentum exactly; damping and bounded impulses are intentional controls.
- A single neighbor rebuild per substep assumes bounded corrections. A kill
  test must fail the mode if corrections exceed the neighbor skin/spacing
  allowance.
- Server-side VTK/JPEG streaming may remain slower than the solver. Solver and
  end-to-end frame timings must therefore be reported separately.
- If the warm solver misses its frame gate at the small default particle
  count, the implementation remains experimental and the bottleneck is
  profiled rather than hidden through time dilation.

## Acceptance gate

Keep this ADR Proposed until the gameplay solver has independent density-
constraint and bounds oracles, the approximate floating box moves without
escaping or producing non-finite state, existing scientific/studio regressions
pass, and a warm 120-frame RTX 5090 run reports solver-only median frame time
`<=16.7 ms` and p95 `<=33.3 ms`. End-to-end streamed FPS is reported but is not
allowed to masquerade as solver FPS.

## Follow-ups

- Implement plan `2026-08-05_warp-gameplay-pbf-dam-break-column`.
- If the PBF slice is accepted, evaluate direct client GPU rendering and then
  FLIP/APIC as a higher-fidelity gameplay alternative.

## Outcome

Accepted on 2026-08-05. The isolated solver passes symmetric pair-correction,
monotonic three-iteration density-constraint, tank-corner, bounded rigid-body,
snapshot-schema, and worker-lifecycle oracles. Applied correction allowances
are configuration-guarded to remain within the neighbor skin. The tuned
four-second default remains finite with zero rigid device error and exact
shell transformation.

The final warm RTX 5090 benchmark uses 1,000 fluid particles, three projection
iterations, `dt=1/60 s`, and XSPH coefficient `0.01`. Solver-only timing is
`1.049 ms` median and `1.244 ms` p95; snapshot/readback is `0.662 ms` median;
the server solver-plus-snapshot loop reaches `574.7 FPS`. This excludes Trame
transport and browser rendering and is not a scientific-fluid claim.
