---
type: experiment
id: 2026-07-13_warp-multilevel-nnps-dense-memory-gate
created: 2026-07-13T07:19:08 CET
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: completed
last_checked: 2026-07-14T00:05:57 CEST
---

# Experiment: Warp multilevel NNPS dense memory gate

## Purpose

Evaluate ADR-0007's decision gate: whether dense per-level cell arrays remain
smaller than the saved fp32 WCSPH particle state on representative localized
refinement layouts, while preserving exact neighbors and producing a real warm
build-plus-fused-consumer benefit.

## Setup

- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB, `sm_89`.
- Warp 1.14.0, CUDA toolkit 12.9, driver API 13.2, Python 3.14.4.
- fp32 only, per owner direction; five levels, `h_ref=0.03`, ratio 2,
  `radius_scale=2`.
- Three representative shapes: compact 3D block, connected thin slab (small
  and large), and two disconnected compact refinement patches.
- Five warm repetitions after an untimed consumer warm-up. JSON files beside
  this note contain raw samples and correctness hashes.
- Dense persistent bytes are computed from the actual allocated arrays and
  cross-checked against the CUDA mempool delta:
  `8*total_cells + 8*nparticles + 36*nlevels`. Saved WCSPH state is the seven
  fp32 PEC arrays `x0/y0/z0/u0/v0/w0/rho0` (`28*nparticles`). The sparse number
  is a representation estimate (uint64 key + int32 start/count per occupied
  cell), not a implemented sparse backend or a peak-transient claim.

## Hypothesis

Dense per-level storage will remain below the saved WCSPH state for connected
localized refinement and reduce fused-consumer time, but disconnected fine
regions may make a level's AABB mostly empty and trigger ADR-0007's sparse
fallback.

## Execution

```text
$PY benchmark_dense_memory.py --case compact --repeats 5 --output compact.json
$PY benchmark_dense_memory.py --case slab --repeats 5 --output slab.json
$PY benchmark_dense_memory.py --case large-slab --repeats 5 --output large-slab.json
$PY benchmark_dense_memory.py --case two-patches --repeats 5 --output two-patches.json
```

## Results

All cases used identical accepted neighbor counts in multilevel and uniform
grid paths. Summation-density aggregate differences were fp32 accumulation
order only (relative delta `4.1e-9` to `6.9e-9` in the recorded connected
cases); the fused-output parity is covered directly by the new 3D fp32 test.

```text
case          particles  dense/saved  candidates ML/uniform  build+fused median
compact            2205       0.463   0.619M / 4.831M        6.354 / 4.525 ms
slab               1883       0.538   0.276M / 3.519M        4.390 / 3.563 ms
large-slab         4808       0.494   0.775M / 23.050M       7.937 / 9.814 ms
two-patches         694      26.784   0.122M / 0.236M        7.206 / 2.662 ms
```

Headline connected case (`large-slab`): exact `75,966` accepted pairs;
candidate work fell 29.8x; the multilevel fused consumer was 1.43x faster and
build-plus-fused time was 1.24x faster (19.1% lower). Persistent dense storage
was 66,476 bytes, exactly matching the CUDA mempool delta, versus 134,624 bytes
of saved state. Per-build metadata readback was 164 bytes.

Kill case (`two-patches`): exact `10,956` accepted pairs, but the fine-level
AABB covered the empty space between the patches and allocated 64,343 cells.
Persistent dense storage was 520,476 bytes versus only 19,432 bytes of saved
state (26.8x over the gate) and a 9,860-byte sparse sorted representation
estimate (52.8x dense/sparse). Both build and consumer were slower than the
uniform grid.

## Conclusion

The dense-per-level prototype is effective for connected refinement and passes
the correctness, candidate-scaling, and mixed-case runtime gates. It fails the
memory decision gate for a realistic topology class: multiple disconnected
refinement regions. ADR-0007 therefore remains **Proposed**. A production
multilevel NNPS must use sparse keyed cells for sparse levels, or a measured
per-level dense/sparse hybrid; dense-only storage cannot be accepted.

## Follow-ups

- Add a sparse key/sort oracle and compare exact persistent plus transient
  memory, build time, and fused traversal time against dense storage.
- Prefer a per-level hybrid decision based on `dense_cells / occupied_cells`
  so compact levels retain the faster/simple dense path.
- Keep the current dense class as the correctness/performance prototype; do not
  expose its representation as a production APR default.
