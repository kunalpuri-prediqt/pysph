---
type: decision
id: ADR-0013
date: 2026-08-05
author: @kunalpuri-prediqt
scope: host-integration
status: Accepted
supersedes: []
relates_to: [ADR-0006, ADR-0012]
depends_on: [ADR-0012]
conflicts_with: []
---

# ADR-0013: Render gameplay fluid with client WebGPU screen-space reconstruction

## Context

The accepted gameplay PBF solver completes a 1,000-particle frame in about
`1.05 ms` on the RTX 5090, while the studio still renders particle spheres
through server-side VTK and streams images. The result is interactive but
looks granular, and the server rendering/encoding path prevents the browser
GPU from reconstructing or shading a continuous water surface.

Real-time particle-fluid renderers commonly splat visible particles into
screen-space depth and thickness targets, smooth depth, reconstruct normals,
and shade the resulting surface. NVIDIA FleX exposes smoothed positions and
anisotropy specifically to support ellipsoid splatting and screen-space
surface reconstruction. The van der Laan--Green--Sainz curvature-flow method
provides the direct real-time rendering precedent; Yu--Turk provides the
anisotropic-kernel basis for thin sheets and smooth particle surfaces.

## Decision

Add a separate client-side WebGPU surface renderer for Gameplay while keeping
the existing VTK particle view for Adaptive, Uniform, debugging, and fallback:

- compute display-only smoothed positions and bounded anisotropic ellipsoid
  axes on the Warp device at snapshot cadence without feeding them back into
  the PBF state;
- transmit a versioned packed gameplay frame containing fluid render
  attributes and rigid pose, initially through Trame's synchronized state;
- use a local ES module and raw WebGPU, with no CDN or new runtime package, to
  render depth/thickness splats, smooth depth, reconstruct normals, and shade
  water with Fresnel reflection, screen-space refraction, Beer--Lambert
  absorption, depth tint, and a procedural environment;
- expose `Fluid surface` / `Particles` only for Gameplay; surface mode skips
  server VTK/JPEG rendering, owns its camera and resize loop, and reports
  feature availability plus presentation/upload timing;
- feature-detect WebGPU and automatically fall back to the existing particle
  view with an explicit message when it is unavailable or device setup fails.

## Rationale

Screen-space reconstruction delivers the largest visual improvement before
ray tracing: it removes the bead appearance, supports thickness-dependent
color and refraction, and scales with visible pixels rather than a dynamic
world-space mesh. Client rendering also removes server image encoding from the
gameplay presentation loop. Raw WebGPU keeps the prototype self-contained and
gives direct control over multipass textures and timing; a large renderer
framework would not remove the need for custom fluid passes.

Keeping VTK intact preserves the established scientific inspection path and a
reliable fallback. Display-only anisotropy avoids contaminating gameplay
solver dynamics with reconstruction smoothing.

## Alternatives considered

- **Improve VTK sphere materials only.** Rejected: still exposes particles and
  retains the server image-stream bottleneck.
- **Marching cubes and a triangle mesh.** Deferred: world-space extraction and
  topology updates cost more and introduce grid resolution/artifact choices
  before screen-space quality is known.
- **Three.js WebGPURenderer.** Deferred for the first slice: it is a sound
  future scene-management option, but the core fluid passes remain custom and
  would add a JS dependency/bundling decision now.
- **OptiX/full ray tracing first.** Deferred: the RTX 5090 can run it, but a
  dynamic surface/BVH plus server streaming is a larger path and does not first
  solve the missing surface representation.
- **Replace VTK globally.** Rejected: Adaptive/Uniform scientific inspection
  and existing camera/scalar behavior must remain stable.

## Consequences

- Browser WebGPU and server CUDA do not share memory in this architecture;
  render attributes cross a device-to-host and Trame transport boundary. Pack,
  upload, render, and presentation timing must be separated.
- Screen-space refraction cannot represent offscreen scene information and is
  view-dependent. These are accepted gameplay-rendering limitations.
- Raw WebGPU increases shader and lifecycle code that requires browser-level
  validation in addition to Python/Node tests.
- Surface mode can be substantially faster than VTK streaming, but no FPS
  claim is accepted until measured in the editor browser.
- The renderer is eligible only for gameplay visual evidence. It cannot hide
  or reinterpret scientific particle fields.

## Follow-ups

- Plan `2026-08-05_warp-webgpu-fluid-surface-renderer` is implemented. Edge
  151 on the NVIDIA Blackwell WebGPU adapter measured `3.30/5.80 ms`
  median/p95 GPU completion at a `1280x720` browser viewport; the 85,336-byte
  packed frame measured `0.182/0.224 ms` server median/p95 packing.
- After the surface renderer passes, evaluate vorticity confinement and real
  foam/spray particles as separate dynamics decisions.
- Evaluate OptiX only as an optional cinematic/hybrid mode after a stable
  reconstructed surface and measured browser path exist.
