---
type: decision
id: ADR-0014
date: 2026-08-06
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0005, ADR-0012, ADR-0013]
depends_on: [ADR-0012, ADR-0013]
conflicts_with: []
---

# ADR-0014: Use NASADEM terrain with a Warp shallow-water geospatial gameplay profile

## Context

The tank-based gameplay PBF solver is fast and interactive, but it cannot
represent kilometre-scale terrain without an impractical particle count. NASA
provides the 1-arc-second, approximately 30 m NASADEM reprocessing of SRTM;
`NASADEM_HGT.001` supplies merged void-filled heights relative to the EGM96
geoid. A cropped real terrain asset can make the demo spatially meaningful,
but it does not supply reservoir bathymetry, dam geometry, buildings, or a
historical initial condition.

At this scale, a depth-averaged structured-grid solver is the appropriate
interactive model. It must remain separate from both the 3D scientific WCSPH
columns and the approximate 3D gameplay PBF column, and it must not be
presented as an operational inundation forecast.

## Decision

Add a fourth, isolated `Geospatial` profile built from:

- a versioned, reproducibly cropped NASADEM terrain asset with source product,
  tile, geographic bounds, EGM96 datum, checksum, and processing metadata;
- a conservative 2D shallow-water finite-volume solver in a new
  `pysph/base/warp_*.py` module, using positivity-preserving dry-state handling,
  hydrostatic reconstruction for lake-at-rest balance, bounded CFL stepping,
  and explicit friction/boundary policy;
- a clearly synthetic reservoir barrier and breach over the real terrain;
- a client WebGPU heightfield renderer for terrain, water, wet/dry fronts,
  velocity-derived foam, lighting, atmosphere, and a local cinematic camera;
- a versioned quantized height/speed frame transport whose solver, packing,
  wire, upload, and browser-render costs are reported separately.

The first preset will use a bounded crop around the Malpasset valley in France
from NASADEM tile `N43E006`, labelled “synthetic breach—not a historical
reconstruction.” Vertical exaggeration, if exposed, is renderer-only; solver
coordinates and elevations remain metrically consistent.

## Rationale

Shallow-water equations retain the dominant gravity-wave and terrain-routing
behavior needed for a valley-scale interactive flood while reducing the state
from a 3D particle volume to a 2D grid. Warp can keep this grid state and
finite-volume update on CUDA. WebGPU can render the same regular topology
directly, avoiding particle splats and dynamic meshing. The separation and
labelling keep real terrain provenance from implying that missing hydraulic
inputs are real.

## Alternatives considered

- **Run the existing PBF/WCSPH solver over the terrain.** Rejected for the
  regional profile: the particle count and timestep cost are mismatched to a
  30 m DEM and kilometre-scale domain.
- **Run the shallow-water solver entirely in browser WebGPU.** Deferred: it
  would reduce transport but would bypass the Warp/PySPH experiment's CUDA
  backend objective.
- **Use original SRTMGL1 instead of NASADEM.** Rejected for the first preset:
  NASADEM is NASA's improved reprocessing and carries useful source/precision
  layers, while retaining the same approximate 30 m scale.
- **Claim a historical Malpasset reconstruction.** Rejected: NASADEM lacks the
  pre-failure dam, reservoir bathymetry, validated breach hydrograph, and
  historical built environment.
- **Couple the current 3D PBF hero region immediately.** Deferred until the
  regional solver, transport, rendering, and scale conventions pass alone.

## Consequences

- The new solver has its own numerical validation: well-balanced rest,
  positivity, closed-boundary mass conservation, dry-front behavior, and CFL
  stability. WCSPH/PBF tests cannot substitute for those gates.
- A server-CUDA/browser-WebGPU copy remains. Quantization and display cadence
  bound the first local prototype, but binary transport may still be required.
- Terrain at 30 m cannot resolve narrow channels, structures, or dam geometry;
  the UI and saved manifests must carry this limitation.
- Hydrologic conditioning changes the raw DEM and must be reproducible and
  documented. Raw and conditioned checksums remain distinct.
- Generic PySPH API/ABI and existing solver behavior remain unchanged.

## Follow-ups

- Implement approved plan `2026-08-06_warp-nasadem-shallow-water-geospatial-demo`.
- After the regional profile passes, evaluate local 3D PBF coupling, buildings,
  higher-resolution regional DEMs, and binary transport in separate plans.
