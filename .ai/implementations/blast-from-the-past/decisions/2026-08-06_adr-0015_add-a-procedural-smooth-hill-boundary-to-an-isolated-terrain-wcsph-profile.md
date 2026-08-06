---
type: decision
id: ADR-0015
date: 2026-08-06
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0012, ADR-0014]
depends_on: []
conflicts_with: []
---

# ADR-0015: Add a procedural smooth-hill boundary to an isolated terrain WCSPH profile

## Context

The regional Geospatial profile uses a depth-averaged finite-volume solver,
while the owner now wants actual three-dimensional SPH water. Official
NASADEM acquisition remains blocked on Earthdata authentication, but terrain
coupling and presentation can be developed first against a deterministic
procedural shape. Replacing the existing dam-break box silently would destroy
its validation baseline; coupling the hill to the regional SWE profile would
not satisfy the request for SPH.

## Decision

Add an isolated `Terrain SPH` studio profile that reuses the current uniform
Warp WCSPH solver and its fixed-solid interaction, but replaces the cuboid
obstacle with a bounded Gaussian hill sampled as stationary solid particles.
Render the matching analytic surface as a smooth VTK terrain actor while the
solver continues to collide with the sampled boundary particles. The hill is
procedural and metrically explicit; it is not described as NASADEM.

## Rationale

This provides genuine 3D SPH water, keeps the already-tested uniform WCSPH
formulation and neighbor search, avoids credentials, and gives terrain
boundary/visualization infrastructure a deterministic oracle. An additive
profile preserves the scientific dam-break comparisons and the separate
regional shallow-water prototype.

## Alternatives considered

- **Replace Geospatial SWE with WCSPH.** Rejected because regional 3D particle
  cost is unbounded and it discards a working, distinctly labelled profile.
- **Change the existing fixed obstacle into a hill.** Rejected because old
  manifests/tests and box-comparison evidence require stable semantics.
- **Render a hill without collision particles.** Rejected because the owner
  requested SPH terrain interaction, not a visual backdrop.
- **Wait for NASADEM.** Rejected for this checkpoint; a deterministic hill can
  validate the SPH coupling without weakening the later provenance gate.

## Consequences

- The first terrain is analytic and smooth, not geospatial evidence.
- Uniform WCSPH is the baseline; adaptive terrain SPH remains a later gated
  step because current variable-resolution interface tests still fail.
- Solver and visual surfaces come from the same parameters, but the collision
  surface remains particle-discretized at `dx`.
- The new profile may not meet a game frame budget. Solver and browser costs
  must be reported rather than inferred from the approximate PBF mode.
- Existing Adaptive, Uniform, Gameplay and Geospatial behavior remains intact.

## Follow-ups

- Implement approved plan `2026-08-06_warp-terrain-sph-smooth-hill`.
- After validation, evaluate a prepared NASADEM crop and bounded SPH domain in
  a separate plan.
