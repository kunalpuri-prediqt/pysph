---
type: experiment
id: 2026-08-06_warp-terrain-sph-smooth-hill
created: 2026-08-06T16:35:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: complete
last_checked: 2026-08-06T16:46:00 CEST
plan: 2026-08-06_warp-terrain-sph-smooth-hill
---

# Experiment: Terrain SPH procedural smooth hill

## Purpose

Check that the fifth studio profile is genuine uniform 3D Warp WCSPH with a
stationary solid hill, and that the hill changes the flow relative to the same
flat-tank initial condition. This is an interaction smoke, not terrain or
field-data validation.

## Method

Run matched `dx=0.1`, 250-step flat and Gaussian-hill cases. Require finite
hill state, relative fluid-mass drift no greater than `1e-6`, bitwise stationary
hill coordinates, and at least one final fluid-particle position differing by
more than `1 mm` from the flat reference. Record solver cost separately.

Primary artifact: `/tmp/pysph-terrain-sph-comparison.json`.

## Results

Both matched cases completed 250 steps on the NVIDIA GeForce RTX 5090 with
Warp 1.15.0. The hill case used 1,000 fluid particles and 72 stationary hill
boundary particles. It remained finite with exactly zero reported relative
fluid-mass drift and bitwise-identical initial/final hill coordinates.

| Measurement | Flat | Hill |
|---|---:|---:|
| Warm solver median | `5.34 ms` | `8.13 ms` |
| Surge front | `2.503 m` | `2.472 m` |
| Fluid kinetic energy | `320.97 J` | `278.71 J` |

Relative to the flat reference, 503/1,000 final fluid positions changed by
more than `1 mm`; RMS position change was `9.76 mm`, the maximum was `13.58 cm`,
and the surge front was `3.11 cm` behind. All interaction gates pass. The
comparison establishes numerical influence from the fixed boundary only; it
does not validate terrain-boundary convergence or field behavior.

The first browser launch with a separate `/tmp` Warp cache completed 250/250
in `62.97 s`, dominated by a one-time approximately 60-second generated-kernel
compile. The cached experiment numbers above isolate warm solver-step cost.
