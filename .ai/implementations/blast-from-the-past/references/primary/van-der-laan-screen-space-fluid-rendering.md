---
type: reference-note
id: van-der-laan-screen-space-fluid-rendering
created: 2026-08-05T14:45:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [host-integration, validation-benchmarks]
---

# Reference: van der Laan et al., Screen Space Fluid Rendering

## Citation

W. J. van der Laan, S. Green, and M. Sainz, “Screen Space Fluid
Rendering with Curvature Flow,” I3D 2009, pp. 91--98,
DOI `10.1145/1507149.1507164`.

## TL;DR

Render particle fluids without polygonization by splatting visible particles
to screen-space depth, smoothing the depth surface on the GPU, reconstructing
normals, and shading the result. The method has an explicit real-time
quality/cost tradeoff and avoids marching-cubes grid artifacts.

## Bearing on blast-from-the-past

This is the primary algorithmic precedent for turning the gameplay PBF cloud
into a continuous visual surface. It affects display only and does not justify
changing solver positions, density, or validation metrics.

## Algorithms to use

- Render particle/ellipsoid depth and additive thickness targets.
- Smooth visible depth with a bounded edge-preserving/curvature-style pass.
- Recover eye-space normals from neighboring depth samples.
- Shade and composite only the nearest visible fluid surface.
- Keep resolution and smoothing iterations explicit performance knobs.

## Questions raised

- Is bilateral depth smoothing sufficient for the first WebGPU slice, or is
  the full curvature-flow update materially better at the target resolution?
- What thickness/absorption mapping remains stable as particle spacing changes?

## Verdict

Adopt the screen-space pass structure. Validate visual continuity and browser
frame time independently from the gameplay solver.
