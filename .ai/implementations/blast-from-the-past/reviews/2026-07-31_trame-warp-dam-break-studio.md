---
type: review
date: 2026-07-31
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-07-31_trame-warp-dam-break-studio.md
adrs: [ADR-0007]
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: []
review_mode: prototype-owner
status: prototype-approved
---

# Review - Trame Warp dam-break studio

## Outcome

The local Trame/VTK studio is ready to run uniform or two-level adaptive
dam-break cases. It exposes run-defining controls, a responsive dark UI, a
camera-preserving server-rendered particle view, pause/resume/single-step/
cancel, live telemetry, scalar coloring, and buffered replay.

CUDA runs in a spawned process; only NumPy snapshots and metrics cross the
process boundary. The worker writes the final NPZ and a JSON manifest with
configuration, adaptation history, metrics, and GPU/Warp metadata.

## Diff summary

- Added the Trame 3/Vuetify 3 application and custom responsive theme.
- Added a VTK scene with fluid, wall, obstacle, axes, ground plane, scalar bar,
  point-Gaussian particles, and camera-preserving updates.
- Added a spawn-safe GPU worker, bounded latest-frame result queue, pause/step/
  resume/cancel protocol, and 120-frame replay buffer.
- Added app-side run validation, adaptive controls/telemetry, uniform/adaptive
  solver selection, and NPZ/JSON persistence.
- Added pip-only requirements, launch documentation, unit tests, worker smoke,
  and result renderer.

## Boundary and dependencies

All app and experiment files live under the implementation memory boundary.
No generic PySPH API/ABI, build/release configuration, or Spack package changed.
The venv received pip-only `trame`, `trame-vtk`, and `trame-vuetify`.

## Behavioral validation

```text
$ python -m pytest -q pysph/base/tests/test_warp_adaptive.py \
    .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py
19 passed in 0.93s

$ python -m py_compile pysph/base/warp_adaptive.py \
    .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/*.py
(exit 0)

$ python .ai/implementations/blast-from-the-past/apps/\
warp_dam_break_studio/worker_smoke.py
statuses: initializing, running, paused, paused, running, completed
frames: 5
paused_once=true, stepped_once=true, resumed_once=true
fluid=1630, coarse=910, fine=720, split_parents=90
mass_drift=0.0, all_finite=true
runtime.device_name=NVIDIA GeForce RTX 5090

$ curl http://127.0.0.1:12345/
HTTP/HTML response received from the Trame server
```

The smoke also re-read the worker-written JSON manifest and asserted the finite
result and exact worker-reported runtime device metadata.

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
(exit 0)
```

## Boundary amendment

- `implementation.md` boundary section updated: n-a
- Amendments log entry: n-a

## Visual aid

```text
browser (Vuetify + vtk.js/remote VTK)
          | commands                    ^ snapshots + telemetry
          v                             |
Trame server ---- bounded queue ---- spawned CUDA worker
     |                                      |
VTK PolyData/ring replay          Warp adaptive simulation
                                             |
                                  final NPZ + JSON manifest
```

The rendered particle viewport is:
![Adaptive resolution viewport](../experiments/2026-07-31_warp-two-level-adaptive-dam-break-smoke/adaptive-250step-resolution.png)

## Risks and incomplete work

- Browser-local VTK serialization failed on the deployed scene's Gaussian
  particle/scalar-bar props. The viewport now deliberately uses Trame remote
  rendering; browser orbit/pan/zoom remain interactive while the server renders
  pixels and avoids transferring the full particle geometry.
- The first remote-only deployment exposed a collapsed Vuetify main-content
  height: telemetry rendered, but the VTK element had no usable viewport. The
  app now uses Vuetify's full-height layout/container pattern and ships the
  current server render as an image fallback. The last completed NPZ is restored
  after restart so a visible frame does not depend on launching another run.
- Trame's default five-minute no-client timeout stopped the transient GCP
  service between browser refreshes. The app now starts with `timeout=0`; the
  deployment unit also uses systemd restart-on-failure semantics.
- Browser acceptance then showed that Vuetify's `fill-height` container was
  flex-centering a roughly 10-pixel VTK child. The content now uses an explicit
  full-height CSS grid with center viewport and collapsible right details panel;
  the existing left drawer remains collapsible. The oversized axes prop was
  removed, the obstacle now matches the solver's 0.16 × 0.40 × 0.161 geometry,
  and scalar-bar dimensions/fonts are capped relative to the viewport.
- The Gaussian splat renderer made the coarse `dx=0.1` particle lattice look
  like a glowing solid block. Fluid now uses instanced sphere glyphs scaled by
  each particle's `h`, with adaptive resolution as the default scalar. Critical
  grid/overlay geometry is duplicated inline because browser acceptance showed
  the class stylesheet was not consistently applied after Trame reconnects.
- VTK emits a benign headless `DISPLAY=:0` warning before selecting EGL in the
  non-browser renderer.
- Snapshot transport is intentionally lossy under browser backpressure; the
  solver is authoritative and the final NPZ remains complete.
- The app is local/single-user only and has no authentication or job scheduler.
- A dedicated viewport screenshot-download button was not added; final result
  data and the offline VTK renderer are available.

## Unresolved questions

- After hands-on use, should the default run be extended from 250 to 500 steps
  so the wave reaches the obstacle despite the fine-level global timestep?
- Should a later app checkpoint add WebSocket compression or server-side
  decimation for substantially larger particle counts?

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > ok. my VM is at gcloud compute ssh gcp-prediqt-rtx6000x1     --zone=us-central1-b
  > -- commit and push here -- go tot the VM and fire up the app so I can connect from the browser here
  > - 2026-07-31T14:41:36 CEST

Prototype-owner authorization permits only a local `prototype:` commit.
Cumulative exact `@prabhu: LGTM` remains required before upstream promotion.

## Addendum - 2026-07-31 - UI polish pass

Scope: viewport/interaction polish only. No solver, worker, transport, or
numerical behavior changed; the CUDA worker protocol and NPZ/JSON persistence
are untouched.

Changes:

- Added a floating viewport control cluster: a reset-view button that restores
  the default camera, and a toggle that hides/shows the color scale.
- Rebuilt the scalar bar as a full-height vertical color scale on the right edge
  (`SetOrientationToVertical`, height 0.84, `SetBarRatio(0.26)`), replacing the
  short bottom stub. It is hidable via `ParticleScene.set_colorbar_visible`.
- Removed the redundant "live particle field" step/particle HUD; the bottom
  timeline already reports progress.
- Refined the dark theme: layered radial/linear background glows, glassmorphic
  toolbar/drawer/panels, hover lift on metric cards, and consistent base color
  across inline viewport backgrounds. Deepened the VTK gradient background.
- Extended `test_studio.py` to cover colorbar visibility and the presence of the
  viewport control cluster / absence of the old HUD.

Validation for this pass (studio tests + offline render run on the RTX 6000 VM):

```text
$ python -m py_compile app.py vtk_scene.py worker.py test_studio.py render_result.py
(exit 0)
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
$ git diff --check
(exit 0)
```

Prototype owner: @kunalpuri-prediqt. Prototype authorization, verbatim quote:

> make it beautiful and more interactive

> 4. Commit locally with a `prototype:` subject.
> 5. Push to:
> `kunalpuri-prediqt/blast-from-the-past-sync`

Authorization is a local `prototype:` commit and push to the owner's fork for
VM deployment. Cumulative exact `@prabhu: LGTM` remains required before upstream
promotion.
