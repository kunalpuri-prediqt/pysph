---
type: experiment
id: 2026-07-26_warp-multilevel-nnps-sparse-hybrid
created: 2026-07-26T21:36:14 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: completed
last_checked: 2026-07-26T22:03:10 CEST
---

# Experiment: Warp multilevel NNPS sparse keyed cells and hybrid selection

## Purpose

Resolve ADR-0007's dense-memory failure by measuring a device-built sorted
cell-key representation and a per-level dense/sparse hybrid.

## Setup

- Reuse the four fp32 layouts and SPH consumer from the completed dense-memory
  gate so results remain comparable.
- Build logical int32 cell keys on the GPU, radix-sort particle/key pairs,
  run-length encode occupied cells, and read back only per-level occupied-cell
  counts.
- Select sparse storage when `logical_cells / occupied_cells > 4`; otherwise
  retain the dense per-level fast path.
- Compare the hybrid against a forced-dense instance under the same new
  generated traversal.

## Correctness gates

- Independently reconstruct sparse keys and counts in a focused test.
- Exact neighbor-set parity against brute force on disconnected patches.
- Forced-sparse generated 3D summation density parity with the uniform grid.
- Existing multilevel NNPS and codegen suites remain green.

## Execution

```text
$PY benchmark_sparse_hybrid.py --case compact --repeats 5 --output compact.json
$PY benchmark_sparse_hybrid.py --case slab --repeats 5 --output slab.json
$PY benchmark_sparse_hybrid.py --case large-slab --repeats 5 --output large-slab.json
$PY benchmark_sparse_hybrid.py --case two-patches --repeats 5 --output two-patches.json
```

## Results

All hybrid and forced-dense cases produced identical accepted-pair totals.
Fused-output differences are fp32 accumulation order: relative L2 deltas were
`4.25e-8` to `7.56e-8`, and maximum absolute delta divided by the reference
peak was `1.51e-7` to `1.94e-7`.

```text
case          mode (fine/coarse)  hybrid/dense bytes  saved state  build+fused
compact       dense/sparse         25,976 / 28,640       61,740   5.861 / 4.558 ms
slab          dense/sparse         25,768 / 28,432       52,724   4.134 / 3.774 ms
large-slab    dense/sparse         63,888 / 66,552      134,624   5.057 / 4.107 ms
two-patches   sparse/sparse         8,900 / 520,552      19,432   7.318 / 3.330 ms
```

The disconnected kill case now passes the memory gate: persistent NNPS state
is 8,900 bytes, 1.71% of forced dense and 45.8% of the saved WCSPH state.
Sparse lower-bound traversal is 3.65x slower than dense for the fused consumer
on this tiny case, so sparse is a memory-safety fallback rather than a general
speed path. Connected fine levels remain dense; their fused times are within
about 0--7% of forced dense, while build adds the occupancy sort/count cost.

## Conclusion

Use a per-level hybrid selected by `logical_cells / occupied_cells > 4`.
Preserve dense storage for compact/connected levels and use sorted int32 keys
for sparse levels. The threshold is a prototype default, configurable on the
class. This resolves ADR-0007's dense-memory blocker without claiming sparse
traversal is faster.
