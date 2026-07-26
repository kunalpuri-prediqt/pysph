---
type: review
date: 2026-07-26
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-07-06_warp-multilevel-gpu-nnps.md
adrs: [ADR-0007]
aspects_touched: [gpu-nnps, warp-backend, particle-memory, validation-benchmarks]
host_files: [pysph/base/warp_multilevel_nnps.py, pysph/base/warp_codegen.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_nnps.py, pysph/base/tests/test_warp_sph.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - Warp multilevel sparse keyed cells and per-level hybrid

## Outcome

The dense-memory blocker is resolved. `MultilevelGridWarpNNPS` now builds a
device-resident sorted-cell-key oracle, measures occupied cells per level, and
selects dense storage when `logical_cells / occupied_cells <= 4` or sparse
sorted keys otherwise. Exact traversal is shared by diagnostic neighbor caches
and generated SPH equation groups.

ADR-0007 moves from Proposed to Accepted. This does not implement runtime
particle splitting/merging and is not production promotion approval.

## Diff summary

- Added GPU logical-cell key generation, radix sort, run-length encoding,
  per-level occupancy reduction, stable sparse compaction, compact dense
  storage, and device `lower_bound` traversal.
- Added configurable `sparse_cell_ratio=4.0`, storage diagnostics, and the
  disconnected two-patch exact-key/exact-neighbor regression.
- Extended generated multilevel equation kernels and their launch contract with
  dense/sparse per-level routing.
- Forced the 3D generated summation-density test through sparse storage.
- Added a four-layout benchmark with raw repeated timings, exact persistent
  bytes, accepted pairs, fp32 output deltas, and hardware metadata.

## Boundary and behavior

All host files are approved `pysph/base/warp_*.py` or focused
`pysph/base/tests/test_warp_*.py` files. No generic public API/ABI,
non-Warp behavior, dependency, build, or release file changes.

The `MultilevelGridWarpNNPS` prototype gains one optional constructor keyword.
Existing uniform-grid, flat, grid, and periodic generated modes are unchanged.
Multilevel periodic domains remain explicitly refused.

## Numerical correctness

- Device sparse keys/counts match an independently reconstructed NumPy key set.
- The disconnected hybrid neighbor set matches brute force for every particle.
- Hybrid and forced-dense benchmark paths have identical accepted-pair totals
  in all four layouts.
- Benchmark fused differences are fp32 accumulation order: relative L2
  `4.29e-8` to `6.18e-8`; peak-normalized maximum absolute delta remains below
  `2e-7`.
- Forced-sparse 3D generated summation density matches the uniform grid.
- Fused pressure/viscosity/continuity/XSPH outputs, CFL/force factors, and final
  adaptive timestep match the uniform grid under the existing fp32 gate.

## Validation output

```text
$PY -m pytest -q pysph/base/tests/test_warp_nnps.py
35 passed, 2 warnings in 4.92s

$PY -m pytest -q pysph/base/tests/test_warp_codegen.py
10 passed, 2 warnings in 1.28s

$PY -m pytest -q pysph/base/tests/test_warp_sph.py \
  -k summation_density_multilevel_matches_grid_3d
1 passed, 57 deselected, 2 warnings in 15.88s

$PY -m pytest -q pysph/base/tests/test_warp_sph.py \
  -k multilevel_fused_wcsph
1 passed, 57 deselected, 2 warnings in 348.40s

$PY -m pytest -q pysph/base/tests/test_warp_sph.py -k multilevel
4 passed, 54 deselected, 2 warnings in 391.85s

$PY -m py_compile benchmark_sparse_hybrid.py
(exit 0)

$ python scripts/update-decision-graph.py
Generated decisions/index.json and decisions/graph.md

$ python scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
(exit 0)
```

The SPH processes were deliberately isolated because the documented WSL2
single-process PTX-JIT accumulation limitation remains.

## Measured decision gate

| Layout | Level mode (fine/coarse) | Accepted pairs | Hybrid bytes | Forced-dense bytes | Saved WCSPH state | Hybrid / dense build+fused |
|---|---|---:|---:|---:|---:|---:|
| Compact | dense / sparse | 41,231 | 25,976 | 28,640 | 61,740 | 5.861 / 4.558 ms |
| Slab | dense / sparse | 30,921 | 25,768 | 28,432 | 52,724 | 4.134 / 3.774 ms |
| Large slab | dense / sparse | 75,966 | 63,888 | 66,552 | 134,624 | 5.057 / 4.107 ms |
| Two patches | sparse / sparse | 10,956 | 8,900 | 520,552 | 19,432 | 7.318 / 3.330 ms |

The two-patch kill case now uses 1.71% of forced-dense memory and 45.8% of
saved WCSPH state. Connected fine levels retain dense storage.

## Risks and incomplete work

- Sparse traversal fixes allocation but scans the logical query cell range and
  performs a lower-bound lookup per cell. It is 3.65x slower than forced dense
  for the tiny disconnected fused case and 2.20x slower build+fused. Sparse is
  a memory fallback, not a speed claim.
- The ratio threshold of 4 is evidence-backed for these fixtures but remains a
  configurable prototype policy, not a universal hardware-tuned constant.
- Cell keys remain int32, matching the existing flattened-cell contract. The
  implementation fails loudly if logical cells exceed int32 capacity.
- Per-update `O(nlevels)` metadata/occupancy readback remains a prototype
  allowance.
- Runtime allocation, split/merge, conservation, shifting, and adaptive
  dam-break integration remain out of scope for this reviewed checkpoint.

## Visual aid

```text
particle h -> GPU level/AABB reduction -> logical cell keys -> radix sort/RLE
                                                   |
                         logical / occupied <= 4 --+--> dense scan/scatter
                                                   |
                         logical / occupied > 4 ---+--> sparse keys/ranges
                                                                  |
destination query -> per-level cell range -> dense index OR sparse lower_bound
                  -> exact symmetric cutoff -> generated SPH equation blocks
```

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > ok. commit. closeout and push - 2026-07-26T22:14:12 CEST

Prototype-owner authorization permits only a local `prototype:` commit.
Cumulative exact `@prabhu: LGTM` remains required before upstream promotion.
