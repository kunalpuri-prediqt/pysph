---
type: plan
id: 2026-07-31_warp-two-level-adaptive-dam-break-smoke
author: @kunalpuri-prediqt
agent: codex
created: 2026-07-31T11:45:00 CEST
status: completed
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_adaptive.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_adaptive.py
within_boundary: true
---

# Plan: Warp two-level adaptive dam-break smoke

## Goal

Run a small 3D fixed-obstacle dam-break in which fluid particles dynamically
split on entering a geometry-defined fine region and merge on leaving it. SPH
steps and variable-resolution neighbor traversal remain on NVIDIA Warp; this
first smoke checkpoint may orchestrate mutation at explicit host checkpoints.

This is an engineering demonstration of the complete adaptation lifecycle, not
a scientifically validated APR formulation or a performance claim.

## Context

The branch already has a working Warp 3D dam-break, mixed-`m`/mixed-`h` WCSPH
equations, and exact `MultilevelGridWarpNNPS` traversal. The RTX 5090 baseline
now runs with Warp 1.15.0. What is missing is particle identity, split/merge
property transfer, mutation orchestration, and a runner that rebuilds every
dependent device structure after particle-count changes.

The earlier 3D stencil experiment did not reproduce the published Vacondio
density-error table. Therefore this smoke will use a clearly labelled
equal-mass eight-child octant stencil only to validate software mechanics. It
must not be promoted as the production split operator.

## Approach

1. Add a small adaptive-fluid controller with stable particle IDs, parent IDs,
   and two discrete levels.
2. Split one coarse parent into eight children at octant offsets. Halve `h`,
   divide mass by eight, and copy/reconstruct all WCSPH and saved-stage
   properties. Preserve mass and linear momentum.
3. Merge only a complete deterministic eight-sibling family outside the fine
   region. Use center-of-mass position and mass-weighted velocity/state.
4. At a configurable adaptation cadence, classify particles using a
   geometry-defined box around the obstacle, apply split/merge, replace the
   fluid arrays, and rebuild `MultilevelGridWarpNNPS` plus saved step state.
5. Add a sibling experiment runner rather than modifying the existing uniform
   dam-break runner. Record particle/level history, split/merge counts, mass and
   momentum residuals, timestep history, finiteness, and obstacle metrics.
6. Run three gates: a split-then-merge conservation test, a one-step mixed-level
   WCSPH test, and a short adaptive obstacle dam-break on the RTX 5090.

## Files expected to change

- `pysph/base/warp_adaptive.py` (new): two-level identity and conservative
  split/merge mechanics plus checkpoint orchestration.
- `pysph/base/warp_sph.py`: add a backward-compatible dam-break
  `neighbor_mode` selector so the existing generated equation blocks can use
  the already-reviewed multilevel traversal.
- `pysph/base/tests/test_warp_adaptive.py` (new): conservation, deterministic
  ownership, complete/incomplete family, and repeated-cycle tests.
- `.ai/implementations/blast-from-the-past/experiments/2026-07-31_warp-two-level-adaptive-dam-break-smoke/`
  (new): sibling runner, commands, metrics, and outputs.
- Implementation memory, review, and closeout artifacts required by the local
  operating contract.

No generic PySPH API, Cython ABI, existing uniform runner, dependency file, or
release configuration will change.

## Tests / validation

- Exact particle-count and property-shape checks after every mutation.
- Relative mass and linear-momentum residual `<= 1e-6` in fp32-scaled tests.
- Split followed by merge recovers the parent center of mass and velocity.
- Multilevel accepted-neighbor sets match brute force after mutation.
- Short obstacle dam-break has finite state, zero Warp device error, both
  levels populated, and at least one recorded split event.
- If the flow produces complete sibling families leaving the target box, record
  a merge event; independently require the deterministic merge-cycle test.
- Run focused Warp adaptive, NNPS, codegen, and multilevel SPH tests separately
  because of the documented monolithic JIT accumulation issue.

## Risks

- Host-orchestrated mutation introduces transfers and allocations, so timing is
  not representative of the planned device-resident particle pool.
- The octant stencil may introduce density noise and anisotropy. Results are
  software smoke evidence only.
- Coarse fixed walls next to fine fluid may contaminate boundary accuracy.
- A global timestep remains limited by fine particles.
- Warp 1.15.0 is newer than the branch's recorded 1.14.0 baseline and may expose
  compatibility differences.

## Out of scope

- Production split weights or resolution-transition correction.
- Device-side allocation, scan, compaction, and capacity growth.
- Particle shifting, free-surface correction, and scientifically converged
  obstacle-pressure validation.
- More than two levels, adaptive solid particles, moving bodies, local
  timesteps, multi-GPU, or performance claims.

## Estimated effort

One prototype checkpoint, approximately 300-600 lines of implementation/tests
plus a sibling experiment runner. Stop after the short finite adaptive run and
review; production device-resident APR remains a separate plan.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-07-31T11:54:25 CEST
- Approval, verbatim quote:
  > APPROVED
