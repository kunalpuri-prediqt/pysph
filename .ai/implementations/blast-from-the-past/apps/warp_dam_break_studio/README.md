# PySPH Warp Dam-Break Studio

Local Trame/VTK application for the uniform and two-level adaptive Warp
dam-break prototypes.

## Install

Use the PrediQT Python environment and pip only:

```bash
source /home/kunalp/prediqt/activate
python -m pip install -r \
  .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/requirements.txt
```

## Run

```bash
source /home/kunalp/prediqt/activate
export PYTHONPATH="$PWD"
export WARP_CACHE_PATH=/tmp/pysph-warp-cache
python \
  .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py \
  --port 1234
```

Open `http://127.0.0.1:1234/`.

The app launches every simulation in a fresh spawned process. The interactive
VTK viewport is rendered on the GPU server and streamed through Trame, avoiding
browser-local serialization limits for Gaussian particle actors. A JPEG frame
is also carried in application state as a visible fallback, and the latest NPZ
is restored when the service restarts.

The offscreen render window is resized from the browser through a
`SizeObserver` bound to `viewport_size`, so the streamed frame matches the
panel aspect instead of being letterboxed, and the camera reframes the tank
for the current aspect on every resize.

The stylesheet lives in `assets/studio.css` and is registered with
`server.enable_module({"serve": ..., "styles": ...})`. It cannot be inlined
with `html.Style(...)`: Vue's runtime template compiler discards `<style>`
tags, so an inline block is silently dropped and only inline element styles
survive.

The workspace has a collapsible control drawer, a dedicated full-height 3D
center panel, and a collapsible run-details panel on the right. The drawer is
split into a scrolling section area (Run card plus Adaptive region, Physics and
Visualization panels) and a pinned footer holding the status strip, the launch
button, the pause/resume/step transport and cancel. The toolbar carries a brand
lockup, a status pill that pulses while running, and a determinate progress bar
driven by the current step. The run-details panel groups telemetry into
Particles / Solver / Health tiles, with mass drift and peak pressure
colour-coded from their own values.

The viewport carries a floating control cluster with a
reset-view button that restores the default camera and a toggle that hides or
shows the color scale; when shown, the scale is a full-height vertical bar on
the right edge labelled by the active field. An idle empty-state card explains
that no run has started, and a busy pill appears while the CUDA worker starts.
Pause takes effect at the next complete solver-step boundary. The 3D camera
remains interactive while paused. The bottom timeline is a live progress
indicator while a run is active and only becomes an interactive replay scrubber
once the run completes; scrubbing replays buffered frames and never rewinds the
GPU solver.

Fluid particles are rendered as instanced sphere glyphs whose radii follow each
particle's smoothing length, so coarse and split particles remain visually
distinct. Adaptive resolution is the default color field. A subtle grid floor
and a wireframe outline of the real container (161/30 long, 0.5 wide, 1.5 tall)
mark the tank so the flow and the fixed obstacle have spatial context.

The Trame server disables its idle auto-shutdown so an SSH-forwarded deployment
remains available between browser sessions.

The adaptive mode is an engineering smoke demonstration. Its equal-mass
eight-child stencil and host-side mutation checkpoints are not a validated or
performance-ready APR formulation.
