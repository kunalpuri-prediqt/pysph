# Known Issues - warp-backend

- [2026-08-05] The accepted `gameplay-pbf` path is an isolated visual solver,
  not a Warp implementation of the scientific PySPH equation stack. Its fixed
  timestep, bounded projections, XSPH smoothing, and impulse-coupled box must
  not be used as hydrostatic, pressure, energy, or torque evidence.
- [2026-08-05] Gameplay render smoothing and anisotropy are display-only Warp
  outputs. The browser screen-space surface is view-dependent, can miss
  offscreen refraction, and must not be interpreted as a solver observable or
  a world-space free-surface reconstruction.
- [resolved 2026-08-06] Gameplay floor contact originally clamped the body COM
  using the box's bounding-sphere radius, so the upright box visibly hovered.
  Contact now computes the rotated box's support extent along each tank plane;
  this remains an approximate impulse-coupled game body, not rigid-contact
  validation evidence.

- [2026-08-05] Adaptive variable-resolution equations can now request exact
  destination/source smoothing-length gradients while uniform equations keep
  their averaged-`HIJ` source unchanged. The exact form passes local oracles
  but still fails the manufactured hydrostatic interface gate, so it remains
  experimental rather than a production consistency claim.
- [2026-08-06] Terrain SPH uses a filled fixed-solid particle mound at the
  selected `dx`; its smooth VTK actor does not increase collision resolution.
  No terrain-boundary convergence claim has been established.
