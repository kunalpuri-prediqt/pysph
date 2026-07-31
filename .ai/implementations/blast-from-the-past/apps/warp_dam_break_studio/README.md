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

The app launches every simulation in a fresh spawned process. Pause takes
effect at the next complete solver-step boundary. The 3D camera remains
interactive while paused, and the timeline can inspect buffered frames without
altering GPU solver state.

The adaptive mode is an engineering smoke demonstration. Its equal-mass
eight-child stencil and host-side mutation checkpoints are not a validated or
performance-ready APR formulation.
