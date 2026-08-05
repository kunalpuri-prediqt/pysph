---
type: review
date: 2026-08-04
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-08-04_warp-adaptive-floating-body-production.md
adrs: [ADR-0008, ADR-0009, ADR-0010, ADR-0011]
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_adaptive.py, pysph/base/warp_codegen.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_adaptive.py, pysph/base/tests/test_warp_codegen.py, pysph/base/tests/test_warp_nnps.py, pysph/base/tests/test_warp_sph.py]
review_mode: prototype-checkpoint
status: incomplete-failed-gates
---

# Review - adaptive Warp dam-break with floating body

## Outcome

The demo can now run adaptive fluid with a genuinely floating 6-DOF body,
axis-aligned tank contact, restartable rigid pose, live studio rendering and
telemetry. The obstacle modes are explicit: `none`, `fixed`, and `floating`.
This is a substantial engineering checkpoint, but it is not production APR.

## What passed

- Grid and multilevel rigid coupling share the same driver and exact mixed
  coarse/fine fluid-body neighbors match a brute-force oracle.
- Fluid adaptation preserves body coordinates, mass and identity across NNPS
  rebuilds.
- Contact kill tests cover no-contact, glancing, damped bounce, and corner
  multi-contact without energy creation.
- Full checkpoints restore fluid, body, controller allocator, adaptation
  history, rigid velocity/angular velocity, quaternion and reference geometry.
- A fresh 900-step case remains finite, has rigid device error zero, relative
  mass drift `2.95e-9`, zero geometric penetration, and rigid geometry drift
  `2.28e-7`.
- Controller/config tests: 16 passed. Studio tests: 19 passed. Focused Warp
  neighbor, rebuild, rigid grid/multilevel, contact, and long-rotation tests
  pass.
- Adaptive-only averaged-`HIJ` grad-`h` consistency now covers continuity,
  pressure force, and equal fluid/body reaction. Its beta/net-force oracle
  passes, as does a four-cycle translating split/merge fixture.
- The corrected step-900 rerun improves body-velocity disagreement from 8.45%
  to 6.95% while retaining `2.95e-9` mass drift and `2.28e-7` rigid geometry
  drift. The adaptive controller suite is now 16 passed and the focused
  coupled regression selection is 5 passed.
- A corrected mass-matched `dx=1/14`, `body_spacing=0.1` reference supersedes
  the earlier `dx=0.07` convergence screen. Against it, surge error is
  `0.09767 m`, maximum-height error `0.00213 m`, and body-COM distance
  `0.04582 m`; all three spacing gates pass. Fluid kinetic energy differs by
  5.94%, narrowly outside its 5% screen.
- Rigid telemetry now separates fluid/contact force and torque and integrates
  their linear/angular impulses. An asymmetric variable-`h` Liu fixture
  confirms exactly balanced net force and torque, ruling out a reaction-sign
  defect.
- Exact destination/source kernel-gradient ownership is implemented
  additively for adaptive variable-`h` blocks. All three supported kernels in
  fp32/fp64 pass unequal-support oracles, equal `h` reduces to the established
  symmetric form, and fluid/fluid plus fluid/body force/torque conserve.

## What failed or remains

- Against the mass-matched fine reference, angular response differs 95.21%,
  integrated fluid angular impulse 58.72%, and sampled contact angular impulse
  88.20%. The adaptive pressure range is `-22.10..35.43 kPa` versus
  `-7.80..23.84 kPa` uniform fine. Uniform coarse-to-fine angular velocity
  itself differs 86.76%, so this is not solely an adaptation defect.
- The current averaged-`HIJ` beta correction worsens a manufactured
  multilevel hydrostatic interface test: horizontal residual RMS rises 12.74%
  and vertical residual RMS 26.75%. It is conservative, but not yet an
  accepted variable-resolution pressure formulation.
- The exact separate-`h` follow-up also fails the padded interface screen:
  horizontal RMS is 24.16% and vertical RMS 19.34% worse than uncorrected.
  Its matched-time equilibrium run retains a roughly 2.3--2.5 cm lateral
  drift that changes sign with refinement.
- Separate-`h` step 900 improves surge error to `0.07604 m`, height error to
  `0.00148 m`, and body-COM distance to `0.03927 m`, but kinetic-energy
  difference rises to 9.79%, fluid angular-impulse difference to 63.15%, and
  the pressure minimum to `-25.68 kPa`. It has 3,388 particles and no merge.
- A nominal half-submerged floating equilibrium drifts laterally and
  vertically at both tested resolutions. Refinement improves vertical and
  rotational drift, but the lateral displacement changes sign and remains
  about 2.6 cm, exposing particle/surface quadrature asymmetry.
- The adaptive case has 3,376 fluid particles versus 2,744 uniform fine and
  produces no flow-driven merge; the 4x particle gate fails.
- Adaptation still pulls/reconstructs the fluid ParticleArray on the host.
  Device pool/free-list allocation and compaction are not implemented.
- Optional shifting is host `O(N^2)` work and is now disabled by default.
  Formal warm-cache GPU timing, memory-capacity demonstration, full CPU
  Application parity, and hands-on studio acceptance remain incomplete.
- The next scientific question is no longer kernel-gradient ownership. It is
  which interface consistency, particle-volume, free-surface, density, and
  rigid-surface quadrature corrections are required in addition to the exact
  pair form.

## Promotion state

Status is deliberately `incomplete-failed-gates`. ADR-0008 and ADR-0010 are
accepted engineering decisions; ADR-0009 and ADR-0011 remain Proposed. No
production or upstream promotion is authorized, and exact `@prabhu: LGTM`
remains required for any later promotion.
