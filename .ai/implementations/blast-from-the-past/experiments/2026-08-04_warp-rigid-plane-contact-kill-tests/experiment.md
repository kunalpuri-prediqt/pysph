---
type: experiment
id: 2026-08-04_warp-rigid-plane-contact-kill-tests
created: 2026-08-04T16:20:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: completed
last_checked: 2026-08-04T16:35:00 CEST
plan: 2026-08-04_warp-adaptive-floating-body-production
---

# Experiment: Warp rigid plane-contact kill tests

## Purpose

Select or reject the axis-aligned spring-dashpot/Coulomb contact candidate
before using it in the adaptive floating-body dam-break.

## Environment

- Warp 1.15.0, CUDA Toolkit 12.9, driver 13.2.
- NVIDIA GeForce RTX 5090, 32 GiB, `sm_120`.
- Cache `/tmp/pysph-warp-phase2/1.15.0`.

## Fixtures and gates

The focused `test_warp_sph.py` contact cases cover:

1. no-contact body: all contact force components and penetration exactly zero;
2. glancing floor impact: positive normal reaction and friction opposing slide;
3. damped box bounce: positive rebound velocity, final mechanical energy no
   greater than initial energy within `2e-3`, and penetration below `0.025 m`;
4. simultaneous lower x/y/z corner contact: all three reactions positive and
   measured geometric penetration `0.01 m`.

## Result

Command:

```text
WARP_CACHE_PATH=/tmp/pysph-warp-phase2 pytest -q \
  test_warp_sph.py::{three focused contact tests}
```

Result: `3 passed` in `0.74 s` on the warm cache. The earlier two-test cold
run also passed in `5.11 s`.

The contact-enabled 20-step adaptive floating-body dam-break additionally
remained finite with zero mass drift, rigid error zero, geometry drift
`2.164857e-7`, contact force `(0,0,0)`, and penetration zero. Its COM, velocity,
pressure, timestep, and simulated time match the pre-contact composed run.

## Decision

The candidate passes the bounded kill gates and is selected in ADR-0008.
Long-impact penetration and settling remain Phase-5 validation, not evidence
provided by this short collision-free transient.
