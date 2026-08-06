# PySPH Warp Dam-Break Studio

Local Trame/WebGPU/VTK application for five Warp flow profiles: uniform WCSPH,
two-level adaptive WCSPH, explicitly approximate gameplay PBF, a depth-averaged
geospatial shallow-water solver, and uniform 3D Terrain SPH over a procedural
Gaussian hill.

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

`Terrain SPH` routes the established uniform Warp WCSPH solver through a
stationary filled solid boundary sampled from a configurable Gaussian hill.
The VTK viewport renders the same analytic hill as a smooth surface while the
hidden boundary particles remain the collision model. This profile is neither
NASADEM nor adaptive-resolution or game-frame-rate evidence.

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

Gameplay defaults to a separate client WebGPU `Fluid surface` view. The Warp
solver produces display-only smoothed positions and bounded anisotropic axes;
the studio packs each fluid particle into four float32 vectors and sends that
versioned frame through Trame state. The browser then renders analytic
ellipsoid depth and additive thickness, applies bilateral depth smoothing,
reconstructs normals, and composites Fresnel reflection, refraction,
Beer--Lambert absorption, a procedural tank floor, and the moving rigid box.
This surface is a visual approximation and never feeds back into PBF state.

`Particles` retains the existing VTK/JPEG view and is the automatic fallback
when WebGPU initialization or shader compilation fails. Adaptive and Uniform
always use VTK. Surface mode deliberately skips VTK scene updates and JPEG
encoding per frame; its orbit/pan/zoom camera and resize loop are browser-local.
The Visualization panel exposes optical controls and final/depth/thickness/
normal targets. Run details separate server pack, browser upload, and WebGPU
completion timings from solver frame cost.

Geospatial uses a separate regular-grid WebGPU renderer and a conservative
finite-volume shallow-water solver. Static terrain is quantized and sent once;
dynamic depth/speed fields use a versioned interleaved uint16 frame. Terrain
and water are metric heightfields with renderer-only vertical exaggeration,
slope/elevation material, contours, sun/fog lighting, thickness tint, Fresnel,
shoreline fade, and velocity-derived foam. Its telemetry reports solver,
server packing, wire size, browser upload, and submitted-GPU completion
separately. The profile is explicitly depth-averaged and is not coupled to the
3D particle solvers.

Until an authenticated official tile is prepared, the profile uses a
deterministic synthetic valley and says so in the viewport. To prepare the
approved real-terrain asset, download the official Earthdata granule
`NASADEM_HGT_n43e006.zip` and run:

```bash
python prepare_nasadem.py \
  /path/to/NASADEM_HGT_n43e006.zip \
  assets/nasadem-malpasset-256.npz \
  --acquired YYYY-MM-DD
```

The tool requires the exact 3601×3601 big-endian HGT member, rejects an
unexpected tile/shape, crops the bounded Malpasset-area AOI to a 256² local
metric grid, and emits an adjacent provenance JSON with official URL/DOI,
CMR/granule IDs, WGS84 bounds, EGM96 datum, checksums, processing history, and
limitations. The NASA elevations, synthetic barrier, and synthetic initial
water are separate arrays. Credentials and the full one-degree source archive
must never be committed. The event is a **synthetic breach—not a historical
reconstruction** and is not an inundation forecast.

Gameplay quality is explicit: `Rich · 7k particles` uses `dx=0.05` and is the
visual default, while `Fast · 1k particles` preserves the original `dx=0.10`
frame-budget benchmark. Returning to Adaptive or Uniform restores the spacing
that was active for the scientific column. Gameplay rejects spacing below
`0.04 m` before spawning a worker so a mistyped field cannot create an
unbounded particle count. Cancel first requests a cooperative stop and then
terminates a worker that is stuck inside a long GPU step.

Fluid particles are rendered as instanced sphere glyphs whose radii follow each
particle's smoothing length, so coarse and split particles remain visually
distinct. Adaptive resolution is the default color field. A subtle grid floor
and a wireframe outline of the real container (161/30 long, 0.5 wide, 1.5 tall)
mark the tank so the flow and the fixed obstacle have spatial context.

The Trame server disables its idle auto-shutdown so an SSH-forwarded deployment
remains available between browser sessions.

## Solver columns

- **Adaptive · two level** is the variable-resolution WCSPH research path.
- **Uniform** is its fixed-resolution scientific reference.
- **Gameplay · fast** is a separate fixed-particle Position-Based Fluids
  profile with a fixed display timestep, bounded projection iterations,
  analytic tank constraints, and approximate impulse coupling to the floating
  box. It reports solver median/p95 frame time separately from snapshot and
  browser streaming cost. Gameplay output is labelled `gameplay-pbf` and is not
  pressure, energy, hydrostatic, or APR validation evidence.
- **Geospatial** is a 2D finite-volume shallow-water profile on a bounded local
  metric grid. It evolves `(h, hu, hv)` with Rusanov fluxes, hydrostatic
  reconstruction, wet/dry positivity, CFL stepping, explicit boundary policy,
  and Manning friction. It is a regional terrain-flow prototype, not a hazard
  model.

The floating gameplay box starts on the floor. Tank contact uses the support
extent of its current oriented box, rather than a conservative bounding sphere,
so an upright box no longer hovers above the floor and rotated corners remain
inside the analytic tank.

Pressure coloring automatically switches to Speed when Gameplay is selected,
because the position-based solver does not evolve a physical pressure field.

The adaptive mode remains an engineering prototype with explicit failed
scientific gates. Gameplay is a visual-interaction prototype and Geospatial is
a depth-averaged regional prototype; all have different claim boundaries and
none is a production solver.
