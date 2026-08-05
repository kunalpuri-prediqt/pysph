---
type: experiment
id: 2026-08-04_warp-icosa13-transfer-and-shifting
created: 2026-08-04T17:00:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: completed
last_checked: 2026-08-04T17:20:00 CEST
plan: 2026-08-04_warp-adaptive-floating-body-production
---

# Experiment: icosa13 transfer, reconstruction, and shifting

## Purpose

Replace the equal-mass octant smoke transfer with the PySPH/Warp-convention
icosahedral candidate while retaining an explicit block on premature
production acceptance.

## Oracle results

`test_warp_adaptive.py` now checks:

- 13 daughter count and exact family mass;
- shell/center minimum-to-maximum mass ratio `0.656566`;
- exact linear scalar reconstruction on a 3D fixture;
- mass and linear momentum conservation;
- complete-family merge back to parent mass and smoothing length;
- split/merge hysteresis;
- bounded shifting with constant-field and linear-field preservation.

All 15 adaptive controller/config tests pass in `0.25 s`.

## Coupled GPU result

The 20-step adaptive/floating/contact-enabled case at `dx=0.1` produced:

```text
fluid particles:          2080 (910 coarse, 1170 fine)
split parents:            90
mass drift:               1.3411045e-9
max shift:                5.4342337e-5 m
latest momentum residual: 1.8600360e-8
rigid geometry drift:     2.1648573e-7
rigid device error:       0
contact penetration:      0
all finite:               true
```

Artifact: `/tmp/pysph-phase3-icosa13-shift-adaptive-floating.npz` with matching
JSON manifest.

## Conclusion

The transfer candidate passes its local conservation/reconstruction gates and
the short coupled smoke. ADR-0009 remains Proposed because repeated
hydrostatic/translating boundary crossings and a selected variable-resolution
consistency operator are still outstanding.
