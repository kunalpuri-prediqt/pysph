---
type: reference-note
id: nasadem-hgt
created: 2026-08-06T12:10:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, validation-benchmarks, host-integration]
---

# Reference: NASADEM Merged DEM Global 1 arc second V001

## Citation

NASA JPL, *NASADEM Merged DEM Global 1 arc second V001*, 2020,
distributed by NASA EOSDIS Land Processes DAAC,
DOI `10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001`.

Official user guide: `https://lpdaac.usgs.gov/documents/2237/NASADEM_User_Guide_V13.pdf`.

## Product facts used here

- `NASADEM_HGT` is a void-filled merged DEM delivered as two-byte signed
  integer elevations in metres relative to the EGM96 geoid.
- Posting is one arc-second, approximately 30 m, in geographic WGS84 tiles.
- The approved first tile is `N43E006`; CMR resolves producer granule
  `NASADEM_HGT_n43e006` as `G2816791562-LPCLOUD`.
- The official Earthdata Cloud URL is protected by Earthdata Login. A `401`
  response is an authentication blocker, not permission to substitute a
  third-party elevation source.

## Bearing on blast-from-the-past

The terrain is a real surface DEM, but the barrier, breach, reservoir, and
water event are synthetic. The 30 m product does not provide bathymetry or
resolve the Malpasset dam, narrow channels, buildings, or historical state.
The studio must say “synthetic breach—not a historical reconstruction” and
must not present the result as an inundation forecast.

## Verdict

Use only a bounded, checksummed crop prepared from the authenticated official
granule. Preserve WGS84 bounds, EGM96 datum, acquisition date, source and
processed checksums, operations, and limitations in the adjacent manifest.
