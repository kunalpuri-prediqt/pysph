---
type: experiment
id: 2026-08-05_warp-gameplay-pbf-dam-break
created: 2026-08-05T14:10:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: complete
last_checked: 2026-08-06T11:09:00 CEST
plan: 2026-08-05_warp-gameplay-pbf-dam-break-column
---

# Experiment: gameplay PBF dam-break

## Purpose

Measure the explicitly approximate `gameplay-pbf` column against its visual
stability and interactive frame-budget gates without treating it as WCSPH
pressure, energy, hydrostatic, or torque evidence.

## Method

Run the committed default of `dx=0.1 m`, `dt=1/60 s`, one substep and three
projection iterations. After ten warm-up frames, time 120 completed solver
frames. Pull a renderer snapshot after each frame and report that transfer
cost separately from the solver. The current stream-loop number includes the
solver and snapshot packing/readback, but excludes Trame transport and browser
rendering.

```bash
env PYTHONPATH=. WARP_CACHE_PATH=/tmp/pysph-warp-game-cache \
  /home/kunalp/.pqt_venv_e0b41259/bin/python \
  .ai/implementations/blast-from-the-past/experiments/\
2026-08-05_warp-gameplay-pbf-dam-break/run_benchmark.py
```

Primary artifact: `/tmp/pysph-gameplay-pbf-benchmark.json`.

## Gates

- finite state, zero device error, and rigid-shell drift no greater than
  `1e-5`;
- warm solver median no greater than `16.7 ms` and p95 no greater than
  `33.3 ms`;
- report snapshot/readback and stream-loop timings independently;
- retain the approximation warning and do not infer scientific equivalence.

## Results

Final hardware: NVIDIA GeForce RTX 5090, Warp 1.15.0, CUDA driver 13.2.
The measured profile has 1,000 fluid particles, three projections, one
substep, `dt=1/60 s`, correction allowance `0.20 dx`, and XSPH coefficient
`0.01`.

| Measurement | Result |
|---|---:|
| Compile + initialize | `97.189 ms` |
| Warm solver median | `1.049 ms` |
| Warm solver p95 | `1.244 ms` |
| Snapshot/readback median | `0.662 ms` |
| Snapshot/readback p95 | `0.830 ms` |
| Solver + snapshot median | `1.721 ms` |
| Solver + snapshot p95 | `2.013 ms` |
| Solver + snapshot loop | `574.7 FPS` |
| Simulated / solver wall time | `15.85x` |

All timing gates pass. The final 120-frame sample is finite, has zero rigid
device error and zero shell-transform drift, and lowers density-constraint RMS
from `0.17485` initially to `0.06253`. A separate four-second/240-frame run
remains finite with constraint RMS `0.07284`, body angular speed
`1.167 rad/s`, and solver median/p95 `0.940/1.244 ms`.

The `0.20 dx` correction cap was selected from a `0.20/0.25/0.30` sweep. At
frame 130 it produced the lowest constraint RMS (`0.059` in that sweep) and
the least energetic body response. A linear/angular impulse cap and
`12 rad/s` angular-speed cap prevent the unbounded spin seen in the first
prototype. Clamp counts remain visible as `Limited corrections`; they are a
relaxation diagnostic, not a scientific residual.

Validation:

- gameplay GPU oracles: `10 passed`;
- studio plus transport suite: `34 passed`;
- adaptive controller: `16 passed`;
- Warp generator: `17 passed`;
- Warp NNPS: `36 passed`;
- gameplay worker pause/step/resume/save smoke: passed at 300 steps.

The measured server loop excludes Trame/WebSocket transport, VTK/JPEG work
outside snapshot packing, and browser rendering. It must not be described as
574 browser FPS or as physically equivalent to WCSPH.

## Hands-on follow-up: bounded rich presentation

The original `dx=0.10` profile remains the `Fast` preset and the transport
gate above remains its baseline. A separate `Rich` preset now uses `dx=0.05`
and 7,020 fluid particles to reduce visible screen-space particle lobes. On
the same RTX 5090 it completed 250 steps with solver median `2.345 ms`, server
pack median `0.580 ms`, WebGPU completion near `4.0 ms`, and a `599,040 B`
packed frame. The Rich payload deliberately does not satisfy the Fast
96-KiB transport gate; it is local visual-quality evidence and motivates
binary transport before remote use.

Hands-on acceptance also found and corrected two gameplay defects. Worker
cancellation is now preemptive after a short cooperative grace period; an
actual browser cancellation stopped a 10,000-step Rich run at step 85 and
returned the studio to `cancelled` in about 1.2 seconds. Rigid tank contact now
uses the rotated box's plane support extent instead of its bounding-sphere
radius, and the default box starts upright on the floor at `z=0.10 m` rather
than visibly hovering. Gameplay configuration rejects `dx < 0.04 m`, which
prevents the accidental unbounded 975,100-particle launch that exposed the
cancellation defect.
