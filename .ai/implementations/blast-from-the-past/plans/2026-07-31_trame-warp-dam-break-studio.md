---
type: plan
id: 2026-07-31_trame-warp-dam-break-studio
author: @kunalpuri-prediqt
agent: codex
created: 2026-07-31T11:50:00 CEST
status: completed
depends_on:
  - 2026-07-31_warp-two-level-adaptive-dam-break-smoke
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: []
within_boundary: true
---

# Plan: Trame Warp dam-break studio

## Goal

Build a polished local browser application for configuring, running, pausing,
inspecting, and replaying the Warp GPU dam-break simulations on beast02. The
application will support the existing uniform-resolution solver immediately and
the two-level split/merge solver after its dependent checkpoint lands.

## Context

The RTX 5090 successfully runs the existing fixed-obstacle Warp dam-break with
Warp 1.15.0. VTK 9.6.2 and PyVista 0.48.4 are installed with pip. Trame is not
installed yet.

Trame supports both browser-local vtk.js rendering and server-side VTK remote
rendering through `VtkRemoteLocalView`. A local view gives fluid camera
interaction for the initial small cases; remote view avoids transferring large
particle datasets when cases grow. A separate solver process is required so
CUDA stepping and JIT compilation cannot block Trame's event loop.

This is a local application because the solver and server need direct access to
beast02's GPU. No public deployment or OpenAI hosting configuration is present.

## Approach

### 1. Application shell and visual design

- Use Trame 3 and Vuetify 3 with a custom dark scientific-visualization theme.
- Build a responsive three-panel layout:
  - compact simulation controls;
  - full-height interactive 3D particle viewport;
  - collapsible telemetry/timeline panel.
- Add clear states for idle, compiling, running, paused, completed, cancelled,
  and failed.
- Keep controls keyboard accessible and usable on a laptop-sized browser.

### 2. Solver configuration

- Resolution mode: uniform or adaptive two-level.
- Geometry: spacing, smoothing-length ratio, boundary layers, obstacle toggle.
- Physics: kernel, viscosity, XSPH, gravity, density, sound speed, CFL, and
  adaptive/global timestep controls.
- Run controls: end step/time, visualization stride, adaptation cadence, fine
  region, output path, and bounded replay-buffer size.
- Validate incompatible values in the UI before starting a worker.
- Lock run-defining controls while active; allow visualization settings to
  change at any time.

### 3. Non-blocking GPU worker

- Launch the Warp solver in a fresh `multiprocessing` process using `spawn`.
- Send commands through a control queue/event: start, pause, resume, single
  step, cancel, and shutdown.
- Return structured telemetry and downsampled/full snapshots through a bounded
  result queue so a slow browser never blocks CUDA stepping.
- Initialize CUDA only inside the worker. Surface Warp compile progress and
  exceptions in the UI.
- Pause at a step boundary and retain particle/device state; resume without
  rebuilding the case.

### 4. Interactive VTK particle view

- Maintain VTK `PolyData` for fluid, walls, and obstacle.
- Render particles with `vtkPointGaussianMapper`; scale by smoothing length or
  a user-selected visual radius.
- Color by pressure, density, speed, resolution level, or particle kind.
- Add scalar bar, axes, ground grid, obstacle visibility, wall opacity, point
  size, background, and camera-reset controls.
- Use `VtkRemoteLocalView`, defaulting to local rendering for the small smoke
  case and exposing an explicit local/remote toggle.
- Update coordinates and selected scalar arrays without resetting the camera.
- Support orbit, pan, zoom, pause/resume, single-step, and snapshot download.

### 5. Telemetry and replay

- Display simulated time, step, timestep, particles per level, split/merge
  counts, mass residual, GPU step rate, maximum pressure, and queue health.
- Keep a bounded frame ring buffer and expose a timeline slider after or while
  paused. Returning to "live" resumes newest-frame display.
- Save the final NPZ and a compact JSON run manifest containing all selected
  options and software/hardware metadata.

### 6. Launch and documentation

- Provide a single pip-only requirements file for Trame packages.
- Provide a launch script that activates no environment itself and documents:
  `source /home/kunalp/prediqt/activate`, then the app command and URL.
- Default Warp cache/output locations to writable user-configurable paths;
  tests use `/tmp`.

## Files expected to change

- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/`
  - Trame application, Vuetify layout/theme, VTK scene, worker protocol,
    solver adapter, tests, requirements, and README.
- `.ai/implementations/blast-from-the-past/experiments/2026-07-31_trame-warp-dam-break-studio/`
  - GPU smoke manifests, screenshots, validation notes, and measured update
    cadence.
- Implementation memory, review, and closeout artifacts required by the local
  operating contract.

The prototype will import the approved `pysph/base/warp_*.py` APIs but will not
modify generic PySPH tools, public API/ABI, setup metadata, or release files.

## Tests / validation

- Unit-test configuration validation and worker state-machine transitions.
- Test bounded queue behavior and stale-frame dropping.
- Test pause, single-step, resume, cancel, worker exception propagation, and
  clean shutdown.
- Test VTK data updates preserve point/scalar lengths and camera state.
- Run a CPU/mock worker UI test without CUDA.
- Run the one-step uniform obstacle case on the RTX 5090 through the app.
- After the APR dependency lands, run a short two-level case through the app
  and verify both levels, split telemetry, conservation metrics, and finiteness.
- Manually verify browser orbit/pan/zoom, local/remote toggle, scalar selection,
  pause inspection, replay scrubbing, and responsive layout.
- Capture one final app screenshot and a JSON run manifest for review.

## Risks

- Sending every particle every step will overwhelm the UI; snapshot cadence and
  a bounded latest-frame queue are mandatory.
- Browser-local rendering requires dataset transfer; remote rendering may
  compete with CUDA for the same GPU. The explicit toggle lets us measure both.
- VTK objects and Warp arrays are not process-safe; only NumPy snapshot payloads
  and scalar telemetry cross the process boundary.
- Python 3.14/Compyle 0.9.1 needs setuptools' pip-provided distutils
  compatibility preload until Compyle is updated.
- Cold Warp JIT compilation can take minutes and must appear as progress rather
  than a frozen application.

## Out of scope

- Public/cloud deployment, authentication, multi-user scheduling, and remote
  execution on other hosts.
- Streaming million-particle frames at solver-step frequency.
- Editing physics parameters in the middle of an active run.
- Production promotion into `pysph.tools` or the installed PySPH CLI.
- Scientifically validated APR claims; those remain gated by the adaptive
  solver plan.

## Estimated effort

Two prototype checkpoints:

1. application shell, worker protocol, uniform solver, interactive VTK view;
2. adaptive controls/telemetry, replay, polish, documentation, and GPU review.

Expected scope is approximately 700-1,200 lines across the app, worker,
visualization, tests, and styling.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-07-31T11:54:25 CEST
- Approval, verbatim quote:
  > APPROVED
