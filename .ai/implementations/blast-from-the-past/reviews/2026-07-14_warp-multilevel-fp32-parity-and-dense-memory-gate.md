---
type: review
date: 2026-07-14
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-07-06_warp-multilevel-gpu-nnps.md
adrs: [ADR-0007]
aspects_touched: [gpu-nnps, warp-backend, particle-memory, validation-benchmarks]
host_files: [pysph/base/warp_multilevel_nnps.py, pysph/base/tests/test_warp_sph.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - Warp multilevel fp32 parity and dense-memory gate

## Diff summary

- Added one fp32 3D mixed-resolution parity test covering the generated fused
  pressure/viscosity/continuity/XSPH group, per-particle CFL/force factors, and
  the finalized adaptive timestep in `neighbor_mode='multilevel'` versus the
  uniform-grid path.
- Corrected the multilevel class docstring: level assignment and AABB/max-h
  reductions are already device-side; only O(nlevels) metadata is read back.
- Added a repeatable dense-memory/performance experiment and four raw JSON
  results (`compact`, `slab`, `large-slab`, `two-patches`).
- Recorded the fp32-only owner amendment, dense-memory kill result, ADR-0007
  Proposed status, sparse/hybrid follow-up, and current/aspect memory.
- No production kernel, launcher, public API/ABI, generic PySPH behavior,
  dependency, build, or release file changed.

## Aspects touched and host files modified

- `gpu-nnps`: dense representation decision gate and sparse/hybrid next step.
- `warp-backend`: fp32 generated-group and adaptive-reduction validation.
- `particle-memory`: exact persistent device-byte accounting and mempool check.
- `validation-benchmarks`: commands, raw samples, hardware, hashes, timing.
- Host files: `pysph/base/tests/test_warp_sph.py` (70 test/helper lines) and
  `pysph/base/warp_multilevel_nnps.py` (documentation correction only).
- Boundary remains truthful: both are approved `warp_*` prototype/test files.

## Behavioral / numerical changes

None in the runtime backend. The generated multilevel routing already existed;
this checkpoint proves its remaining fp32 consumers. The new test compares all
nine fused/CFL output arrays plus the final scalar timestep with fp32-derived
tolerances (`rtol=2e-4`, `atol=2e-6`). The owner explicitly removed fp64 from
the delivery gate; dtype-generic existing code was not removed or expanded.

Benchmark density aggregates differ only by neighbor accumulation order
(`1.2e-9` to `6.9e-9` relative in recorded connected cases). Accepted neighbor
pair totals are exact in every layout.

## Tests / validation run

```text
$PY -m pytest -q pysph/base/tests/test_warp_sph.py \
  -k multilevel_fused_wcsph
1 passed, 57 deselected, 2 warnings in 227.40s

$PY -m pytest -q pysph/base/tests/test_warp_sph.py -k multilevel
4 passed, 54 deselected, 2 warnings in 274.05s

$PY -m pytest -q pysph/base/tests/test_warp_nnps.py
34 passed, 2 warnings in 2.27s

$PY -m pytest -q pysph/base/tests/test_warp_codegen.py
10 passed, 2 warnings in 34.25s

$PY -m py_compile experiments/.../benchmark_dense_memory.py
(exit 0, no output)

$PY -m pytest -q pysph/base/tests/test_warp_sph.py
Inconclusive infrastructure run: emitted passing progress, then stayed at 100%
CPU inside the final in-process Warp PTX-JIT segment. Terminated after 2,988s
without a pytest summary. This is the documented module-accumulation limitation;
the isolated four-test multilevel gate above passes.
```

Raw benchmark commands used `benchmark_dense_memory.py --case <case>
--repeats 5`; complete samples and hashes are in the experiment JSON files.

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/update-decision-graph.py
Generated decisions/index.json and decisions/graph.md
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
$ git diff --check
(exit 0, no output)
```

## Boundary amendment

- implementation.md boundary section updated: n-a
- Amendments log entry: n-a

## Visual aid

| fp32 layout | Particles | Exact accepted pairs | Candidates ML / uniform | Dense / saved state | Build + fused ML / uniform |
|---|---:|---:|---:|---:|---:|
| Compact block | 2,205 | 41,231 | 0.619M / 4.831M | 0.463x | 6.354 / 4.525 ms |
| Thin slab | 1,883 | 30,921 | 0.276M / 3.519M | 0.538x | 4.390 / 3.563 ms |
| Large thin slab | 4,808 | 75,966 | 0.775M / 23.050M | 0.494x | 7.937 / 9.814 ms |
| Two patches | 694 | 10,956 | 0.122M / 0.236M | 26.784x | 7.206 / 2.662 ms |

The connected large slab passes the runtime gate (19.1% lower combined time).
The disconnected topology decisively kills dense-only production acceptance.

## Risks

- Dense per-level memory follows the level AABB, not occupied cells. Multiple
  disconnected refinement regions can allocate mostly empty grids and be both
  larger and slower than the uniform path.
- The sparse figure is a persistent-representation estimate, not an implemented
  or peak-transient measurement. Sparse key/sort implementation must measure
  sorting scratch, build time, and traversal before choosing it.
- Adding the large multilevel group makes the already-known single-process
  PTX-JIT accumulation limit observable in `test_warp_sph.py`; focused process
  isolation is still required on this WSL2 dev host.
- This is prototype evidence, not production completion or an upstream claim.

## Unresolved questions

- Sorted cell/Morton keys, a hash table, or a per-level hybrid?
- What occupancy threshold should select dense versus sparse per level?
- Can generated kernels dispatch per level without divergent hot-loop overhead?

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > ok. commit - 2026-07-14T07:44:37 CEST

Prototype-owner authorization does not permit upstream publication or
production promotion; cumulative `@prabhu: LGTM` remains required for that.
