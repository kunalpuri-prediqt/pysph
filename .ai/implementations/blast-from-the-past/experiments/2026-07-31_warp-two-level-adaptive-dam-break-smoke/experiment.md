---
type: experiment
id: 2026-07-31_warp-two-level-adaptive-dam-break-smoke
created: 2026-07-31T11:54:25 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: completed
last_checked: 2026-07-31T12:55:00 CEST
---

# Experiment: Warp two-level adaptive dam-break smoke

## Purpose

Exercise the complete host-orchestrated split/merge lifecycle around the
device-resident Warp WCSPH step and multilevel NNPS. This is an engineering
smoke case, not a validated APR formulation.

## Configuration

- 3D fixed-obstacle dam-break.
- Two smoothing-length levels with ratio 2.
- Equal-mass eight-child octant split.
- Complete-family deterministic merge.
- Geometry-defined fine box around the impact region.
- Warp 1.15.0 on beast02's RTX 5090 (`sm_120`).

## Gates

- Pure split/merge suite passes.
- At least one split event and both levels populated.
- All state finite.
- Relative mass drift no larger than `1e-6`.
- Multilevel GPU dam-break completes without device error.

## Results

The end-to-end adaptive run passed every engineering gate on beast02:

```text
Warp 1.15.0
device: NVIDIA GeForce RTX 5090 (sm_120)
steps: 250
simulated time: 0.13162134085182803 s
elapsed cached wall time: 4.474121003000619 s
adaptation checkpoints: 25
split parents: 103
merged families in this flow: 0
final particles: 1,721 fluid (897 coarse + 824 fine)
mass: 1000.0
relative mass drift: 0.0
rho range: [985.0015258789062, 1015.136474609375]
pressure range: [-15226.88671875, 17075.619140625] Pa
all finite: true
```

The independent controller suite exercises complete-family merge and a
split/merge round trip; the flow did not move a complete fine family outside
the refinement box during its 0.132 s simulated interval. The 250-step run is
therefore split-path evidence, while merge behavior is unit-test evidence.

Validation:

```text
adaptive + studio unit tests: 19 passed in 0.93s
Warp NNPS: 35 passed in 0.77s
Warp code generation: 10 passed in 15.08s
Warp SPH: 58 passed in 2339.87s
```

The result archive and manifest are
`/tmp/pysph-adaptive-250step.{npz,json}`. The committed review image is
`adaptive-250step-resolution.png`.

## Interpretation

This closes the software-mechanics smoke checkpoint: mutation, stable family
identity, exact mass transfer, multilevel NNPS rebuild, generated multilevel
WCSPH traversal, and repeated GPU stepping work together.

It does not validate the equal-mass octant stencil as a production APR method.
The global timestep is limited by the fine level, fixed boundaries remain
coarse, and 250 steps stop before the main obstacle impact. A production APR
checkpoint still needs a literature-reconciled transfer operator, transition
correction/shifting, device-side allocation/compaction, and convergence
evidence.
