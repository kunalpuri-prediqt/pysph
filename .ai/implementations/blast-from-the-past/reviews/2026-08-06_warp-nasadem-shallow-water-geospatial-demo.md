---
type: review
date: 2026-08-06
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-08-06_warp-nasadem-shallow-water-geospatial-demo.md
adrs: [ADR-0012, ADR-0013, ADR-0014]
aspects_touched: [warp-backend, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_shallow_water.py, pysph/base/tests/test_warp_shallow_water.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - NASADEM shallow-water geospatial demo

## Outcome

The studio now has an isolated fourth `Geospatial` profile: a conservative
2D finite-volume shallow-water solver on Warp, compact static/dynamic grid
transport, and a dedicated WebGPU landscape renderer. It does not alter the
Adaptive, Uniform or Gameplay solver contracts.

The implementation and synthetic fallback pass numerical, worker, transport,
browser and performance gates. The official NASADEM crop is not present:
Earthdata returned `401` and no owner credential/archive is available. The UI
therefore labels the current terrain `synthetic-valley-fixture`. The importer
and automatic asset routing are complete; adding the authenticated official
archive is the remaining external-data step.

## Diff summary

- `warp_shallow_water.py` adds double-buffered device-resident `(h, hu, hv)`,
  hydrostatic reconstruction, Rusanov fluxes, balanced bed sources, CFL
  reduction, wet/dry positivity, Manning friction, explicit boundaries,
  metrics and self-describing save output.
- Terrain elevation and synthetic barrier are separate arrays. Display-only
  vertical exaggeration never changes solver state.
- `prepare_nasadem.py` validates the exact official N43E006 HGT archive and
  emits a bounded NPZ plus provenance/checksum manifest without credentials or
  the full tile.
- `geospatial_transport.py` separates a one-time quantized terrain frame from
  interleaved uint16 depth/speed frames and rejects malformed/non-finite data.
- `webgpu_geospatial.js` renders regular terrain and water heightfields with
  normals, terrain material/contours, lighting/fog, thickness tint, Fresnel,
  restrained highlights, shoreline/speed foam and browser-local camera.
- `app.py` adds the fourth mode, isolated controls, terrain-flow health and
  timing telemetry, replay/showcase lifecycle, automatic prepared-asset
  detection, and a robust terrain-before-water synchronization contract.
- Focused Python/Node tests, README, ADR-0014, primary reference, experiment,
  daily/session memory and this review record the evidence and limitations.

## Numerical/browser evidence

| Gate | Result |
|---|---:|
| 256² solver median / p95 | `0.346 / 1.425 ms` |
| 300-step worker volume drift | `6.04e-8` |
| 600-step browser volume drift | `1.19e-7` |
| WebGPU completion median / p95 | `7.70 / 19.80 ms` |
| Dynamic base64 frame | `341.3 KiB` |
| Warp separation | `86 passed` |
| Studio/transport/importer | `47 passed` |
| Node WebGPU | `11 passed` |

The solver also passes lake-at-rest, 500-step conservation/positivity,
independent CPU Rusanov, wet-front, friction, boundary, invalid-config,
barrier separation and save/restore oracles. The exact worker lifecycle passes
pause, single-step, resume and completion.

## Risks and blocker

- **External blocker:** supply an authenticated official
  `NASADEM_HGT_n43e006.zip` to run the importer. No unverified replacement may
  be used.
- The synthetic test valley is smooth and cannot demonstrate the visual value
  of real drainage/ridge detail. It is intentionally not passed off as SRTM.
- Base64 costs 341.3 KiB per 256² dynamic frame. Local performance passes, but
  binary transport is the likely remote follow-up.
- The model omits bathymetry, surveyed structures, calibrated breach flow,
  infiltration/runoff and uncertainty. A beautiful rendering must not broaden
  its claim boundary.
- WebGPU completion excludes display scan-out and remote network latency.

## Promotion state

Prototype-owner commit/push authorization is recorded verbatim below with the
real-data gate explicitly open. Upstream production/PR promotion still
requires exact `@prabhu: LGTM`.

## Owner verdict

Prototype-owner authorization by @kunalpuri-prediqt at
2026-08-06T17:04:31 CEST, verbatim:

> ok. commit and push

This authorizes the cumulative prototype commit and push to the owner's fork;
the authenticated NASADEM data blocker remains open and this is not upstream
promotion approval.
