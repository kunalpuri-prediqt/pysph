---
type: experiment
id: 2026-07-31_trame-warp-dam-break-studio
created: 2026-07-31T11:54:25 CEST
author: @kunalpuri-prediqt
aspect: host-integration
status: completed
last_checked: 2026-07-31T12:55:00 CEST
---

# Experiment: Trame Warp dam-break studio

## Purpose

Validate the local browser application, spawned CUDA worker, interactive VTK
scene, pause/step/resume protocol, bounded replay transport, and persisted run
outputs for the adaptive dam-break prototype.

## Environment

```text
host: beast02
GPU: NVIDIA GeForce RTX 5090, sm_120, 32 GiB
warp-lang: 1.15.0
vtk: 9.6.2
trame: 3.13.2
trame-vtk: 2.11.15
trame-vuetify: 3.2.5
```

All application packages were added with pip to the active PrediQT venv. No
Spack package or configuration was changed.

## Results

The exact spawned-worker protocol completed a four-step adaptive CUDA run:

```text
statuses: initializing -> running -> paused -> paused -> running -> completed
frames received: 5
pause requested: true
single step requested: true
resume requested: true
fluid particles: 1,630 (910 coarse + 720 fine)
split parents: 90
relative mass drift: 0.0
all finite: true
runtime device: NVIDIA GeForce RTX 5090
```

The worker wrote and re-read:

- `/tmp/pysph-studio-worker-smoke.npz`
- `/tmp/pysph-studio-worker-smoke.json`
- `/tmp/pysph-studio-worker-smoke-summary.json`

Unit coverage includes pre-launch configuration validation, stale-frame
dropping, bounded replay behavior, VTK actor/scalar updates, and camera
preservation:

```text
adaptive + studio unit tests: 19 passed in 0.93s
py_compile: exit 0
```

The Trame application object builds, and a server bound to
`127.0.0.1:12345` served its HTML shell successfully. Automated Windows Edge
capture from WSL could not complete Trame's WebSocket loopback connection, so
the full browser interaction pass remains a user-side acceptance check. The
underlying VTK scene rendered successfully through EGL and produced
`../2026-07-31_warp-two-level-adaptive-dam-break-smoke/adaptive-250step-resolution.png`.

## Interpretation

The local application plumbing and GPU control path are ready for interactive
use on beast02. The bounded queue intentionally drops stale visualization
frames without slowing the solver, while the ring buffer retains recent
received frames for replay.

This is a local single-user prototype. It has no authentication, remote job
scheduler, public deployment, or guarantee that every solver step is retained
as a replay frame.
