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
is restored when the service restarts. The workspace has a collapsible control
drawer, a dedicated full-height 3D center panel, and a collapsible run-details
panel on the right. The viewport carries a floating control cluster with a
reset-view button that restores the default camera and a toggle that hides or
shows the color scale; when shown, the scale is a full-height vertical bar on
the right edge. The redundant live step/particle overlay was removed because the
bottom timeline already reports progress. Pause takes
effect at the next complete solver-step boundary. The 3D camera remains
interactive while paused, and the timeline can inspect buffered frames without
altering GPU solver state.

Fluid particles are rendered as instanced sphere glyphs whose radii follow each
particle's smoothing length, so coarse and split particles remain visually
distinct. Adaptive resolution is the default color field.

The Trame server disables its idle auto-shutdown so an SSH-forwarded deployment
remains available between browser sessions.

The adaptive mode is an engineering smoke demonstration. Its equal-mass
eight-child stencil and host-side mutation checkpoints are not a validated or
performance-ready APR formulation.
