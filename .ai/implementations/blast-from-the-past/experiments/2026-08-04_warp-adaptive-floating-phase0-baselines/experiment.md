---
type: experiment
id: 2026-08-04_warp-adaptive-floating-phase0-baselines
created: 2026-08-04T15:15:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: completed
last_checked: 2026-08-04T15:45:00 CEST
plan: 2026-08-04_warp-adaptive-floating-body-production
---

# Experiment: adaptive-floating Phase 0 baselines

## Purpose

Freeze reproducible pre-composition behavior for the three existing Warp paths
before the adaptive-fluid and floating-body implementations are combined:

1. uniform fixed-obstacle dam-break;
2. two-level adaptive fixed-obstacle dam-break;
3. uniform floating-box collision-free transient.

These are engineering baselines, not the later scientific/convergence cases.

## Environment

- Warp 1.15.0, CUDA Toolkit 12.9, driver 13.2.
- NVIDIA GeForce RTX 5090, 32 GiB, `sm_120`.
- Warp cache: `/tmp/pysph-warp-phase0/1.15.0`.

## Configuration

- `dx=0.1`, Wendland, `alpha=0.25`, XSPH `0.5`, CFL `0.3`, one boundary
  layer, adaptive timestep with 50-step startup damping.
- 20 steps for each baseline to keep the cold-cache checkpoint bounded.
- Adaptive case: adaptation every 5 steps, maximum 128 splits/checkpoint,
  fine box `1.75,2.8,-0.3,0.3,0.0,0.65`.
- Floating case: body density 500 kg/m3, initial box center `(2.35,0,0.30)`;
  horizon is deliberately collision-free.

## Commands

Commands use `WARP_CACHE_PATH=/tmp/pysph-warp-phase0` and write complete NPZ/
JSON state under `/tmp/pysph-phase0-*`. Raw terminal output and compact result
summaries are recorded after completion.

## Gates

- All three paths remain finite with zero reported device error where exposed.
- Adaptive case contains both levels, records a split, and has mass drift no
  larger than `1e-6`.
- Floating case moves under computed fluid/body forces and retains relative
  rigid geometry within `1e-5`.

## Results

All three baseline gates passed:

```text
case                         fluid      solids       sim time       result
uniform fixed                1000       3824 + 4     0.003095018    finite
adaptive fixed               1630       3824         0.001547523    finite
uniform floating             1000       3824 + 44    0.003095018    finite
```

- Uniform fixed: mass drift zero, peak pressure `199.5003 Pa`; cold elapsed
  `154.445 s`.
- Adaptive fixed: 910 coarse + 720 fine, 90 split parents, zero merges and zero
  mass drift; elapsed `7.207 s`.
- Uniform floating: COM displacement
  `(2.384e-8, 0, -4.695e-5) m`, vertical COM velocity `-0.0303613 m/s`,
  relative geometry drift `2.164857e-7`, vertical body force `-87.8915 N`, and
  rigid device error zero.

Artifacts:

- `/tmp/pysph-phase0-uniform-fixed.{npz,json}`
- `/tmp/pysph-phase0-adaptive-fixed.{npz,json}`
- `/tmp/pysph-phase0-uniform-floating.npz`

The floating runner initially exposed Python 3.14's missing stdlib
`distutils`; preloading setuptools before PySPH imports fixed the compatibility
path and is now persisted in the runner.
