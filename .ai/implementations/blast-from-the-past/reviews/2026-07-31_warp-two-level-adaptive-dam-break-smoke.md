---
type: review
date: 2026-07-31
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-07-31_warp-two-level-adaptive-dam-break-smoke.md
adrs: [ADR-0007]
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_adaptive.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_adaptive.py]
review_mode: prototype-owner
status: prototype-approved
---

# Review - Warp two-level adaptive dam-break smoke

## Outcome

The approved engineering checkpoint works end to end on beast02's RTX 5090.
Coarse particles split into deterministic eight-child families at host
adaptation checkpoints; the rebuilt two-level NNPS feeds the generated Warp
WCSPH path between checkpoints. Exact controller tests cover complete-family
merge. This is prototype software evidence, not production APR validation.

## Diff summary

- Added `TwoLevelAdaptiveController` with stable particle/family IDs,
  deterministic octant split, complete-family merge, and conservation
  diagnostics.
- Added `DamBreakConfig` and incremental `WarpDamBreakSimulation` with
  initialization, step, snapshot, telemetry, and NPZ persistence.
- Added a backward-compatible `neighbor_mode` selector to
  `wc_sph_dam_break_step`; its default remains the existing uniform grid.
- Added nine focused controller/config tests and an adaptive experiment runner.
- Added a VTK resolution-level image from the completed 250-step result.

## Boundary and behavior

Every host file is an approved `pysph/base/warp_*.py` or focused
`pysph/base/tests/test_warp_*.py` file. No generic public API/ABI,
non-Warp behavior, build metadata, release configuration, or Spack state
changed.

The equal-mass octant stencil is deliberately labeled an engineering smoke
operator. Adaptation pulls/rebuilds the fluid host array and multilevel NNPS at
explicit checkpoints; GPU stepping remains device-resident between them.

## Numerical behavior

- A split produces eight children at `h/2` and `m/8`, copying velocity and
  WCSPH state.
- A merge is permitted only for a complete deterministic eight-sibling family.
- Controller gates verify mass and linear-momentum conservation to `1e-6`.
- The 250-step flow retained exactly `mass=1000.0`, relative drift `0.0`, and
  finite position, velocity, density, and pressure.
- Final mixed resolution was 897 coarse + 824 fine fluid particles.
- The flow produced 103 split parents and no complete-family merge in its
  short simulated interval; merge is independently unit-tested.

## Validation output

```text
$ python -m pytest -q pysph/base/tests/test_warp_adaptive.py \
    .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py
19 passed in 0.93s

$ python -c '<setuptools + pytest wrapper>' -q \
    pysph/base/tests/test_warp_nnps.py
35 passed, 2 warnings in 0.77s

$ python -c '<setuptools + pytest wrapper>' -q \
    pysph/base/tests/test_warp_codegen.py
10 passed, 2 warnings in 15.08s

$ python -c '<setuptools + pytest wrapper>' -q \
    pysph/base/tests/test_warp_sph.py
58 passed, 2 warnings in 2339.87s (0:38:59)

$ python experiments/.../run_adaptive.py --steps 250 --adapt-every 10 \
    --output /tmp/pysph-adaptive-250step.npz
step=250, time=0.13162134085182803, particles=1721,
coarse=897, fine=824, split_parents=103, merged_families=0,
mass_drift=0.0, all_finite=true, elapsed=4.474121003000619s
```

The six regression warnings are the same two `warp-lang` ctypes layout
deprecations repeated once per isolated Python process under Python 3.14.

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
(exit 0)
```

## Boundary amendment

- `implementation.md` boundary section updated: n-a
- Amendments log entry: n-a

## Visual aid

![Adaptive 250-step resolution view](../experiments/2026-07-31_warp-two-level-adaptive-dam-break-smoke/adaptive-250step-resolution.png)

```text
host checkpoint -> classify coarse/fine -> split/complete-family merge
       -> replace fluid ParticleArray -> rebuild MultilevelGridWarpNNPS
       -> generated multilevel WCSPH steps on CUDA -> next checkpoint
```

## Risks and incomplete work

- The octant transfer stencil is not literature-reconciled or converged.
- Host mutation/reallocation and global fine-level timestep are not a
  performance design.
- Fixed boundaries remain coarse, with no transition correction or shifting.
- The 250-step run reaches 0.132 s and does not yet capture the principal
  obstacle-impact transient.
- Device allocation/compaction, more than two levels, and scientifically
  validated pressure results remain future work.

## Unresolved questions

- Which published split/merge transfer operator and correction terms should
  replace the equal-mass smoke stencil?
- Should the next checkpoint prioritize device-side mutation or a
  longer/converged physics comparison?

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > ok. my VM is at gcloud compute ssh gcp-prediqt-rtx6000x1     --zone=us-central1-b
  > -- commit and push here -- go tot the VM and fire up the app so I can connect from the browser here
  > - 2026-07-31T14:41:36 CEST

Prototype-owner authorization permits only a local `prototype:` commit.
Cumulative exact `@prabhu: LGTM` remains required before upstream promotion.
