# Current - blast-from-the-past

Updated: 2026-07-14T00:05:57 CEST by codex

**Status:** The Warp backend has device-mirrored particle state, grid-direct
3D WCSPH, generated/fused equation groups, periodic neighbors, validated 2D
elliptical-drop and 3D dam-break cases, and ADR-0006 P0-P3 floating-rigid-body
support. P2 (`bb3843f7`) keeps moment reduction, 3x3 angular solve, RK2 state,
and rigid motion on the GPU. P3 adds deterministic two-pass Liu fluid/rigid
coupling, static rigid number density, device density/body-force staging, and
the sibling `wc_sph_dam_break_rigid_step` without modifying the fixed-wall
driver.

**Active aspects:** warp-backend, gpu-nnps, particle-memory,
validation-benchmarks, host-integration.

**In-flight experiment:**
`experiments/2026-06-19_warp-floating-body-rigid` (ADR-0006). P0-P3 are done;
P3 is approved by @prabhu and committed locally. P4 contact/long-horizon fidelity and the
photorealistic animation remain.

`experiments/2026-07-06_warp-adaptive-particle-resolution-p0` is active under
the approved dynamic-APR plan. The first checkpoint proves the existing
multi-solid driver can run a fixed Kleefsman obstacle: a warm 1,000-fluid +
3,824-wall + 4-obstacle case ran 250 steps to `t=0.258455`, remained finite,
developed 150.147 kPa maximum obstacle pressure, and kept obstacle device
coordinates bit-identical.

The multilevel-GPU-NNPS milestone (`plans/2026-07-06_warp-multilevel-gpu-nnps.md`,
status in-progress) has landed steps 1-3 and ADR-0007 (Proposed). New module
`pysph/base/warp_multilevel_nnps.py` provides `MultilevelGridWarpNNPS`: discrete
half-open smoothing-length levels, a flattened per-level global cell list, and
exact variable-stencil cross-level traversal. Construction is device-resident
(GPU level assignment + count/max-h/AABB reductions, only O(nlevels) scalar
readback). All eight kill-gate fixtures pass, a synthetic localized-refinement
fixture shows ~9x lower candidate work (197k vs 1.77M pairs) with identical
accepted sets, and `warp_codegen` gained `neighbor_mode='multilevel'` so
generated SPH equation groups consume the multilevel structure directly
(summation density matches the uniform grid in 2D and 3D). Multilevel + periodic
is refused. fp32 adaptive-timestep plus fused pressure/viscosity/continuity/XSPH
multilevel parity now passes in 3D. The owner explicitly deferred fp64.

The dense-memory gate rejected dense-only storage as a universal production
representation. A connected 4,808-particle slab used 49.4% of saved-state
memory, cut candidates 29.8x, and lowered warm build-plus-fused time 19.1%.
But two disconnected fine patches allocated 64,343 cells for 694 particles:
26.8x saved-state memory and 52.8x the sorted-sparse estimate, with worse
runtime. ADR-0007 remains Proposed pending sparse keyed cells or a per-level
dense/sparse hybrid.

**Latest validation:** fp32 multilevel SPH subset: `4 passed`; Warp NNPS:
`34 passed`; codegen: `10 passed`. The monolithic SPH file hit the documented
in-process PTX-JIT accumulation limit after 2,988s; focused validation is the
operational gate on this WSL2 host. Final P3 Warp SPH suite before this addition:
`54 passed, 2 warnings`. The 7,458-particle coupled first-plunge transient ran
241 steps to `t=0.200603`,
remained finite with device error 0, moved/rotated the body from computed fluid
reaction, and preserved relative geometry to `1.90e-6`. The review image and
metrics are in
`reviews/2026-06-20_warp-liu-fluid-rigid-coupling-p3.md`.

**Open approvals:** P3 and 3D dam-break reviews are approved by @prabhu. PR
#435 remains an upstream publication item, not a local review blocker.

**Next action:** Continue the multilevel milestone with a sparse keyed-cell
oracle and a measured per-level dense/sparse hybrid; preserve exact traversal
and the connected-level dense fast path before moving ADR-0007 to Accepted.
Run the three warp
test files SEPARATELY (`test_warp_nnps.py` 34, `test_warp_codegen.py` 10,
`test_warp_sph.py` 57) -- the combined single-process command hangs
pre-existingly on the WSL2 PTX-JIT. P0 stencil-convention reconciliation remains
a prerequisite for production APR weights; floating-body P4 remains queued.

**Known validation limitation:** Compyle 0.9.1 on Python 3.14 cannot run the
shipped CPU rigid Application (`ast.Str` removal). P3 uses direct NumPy
primitive parity plus end-to-end GPU smoke/transient validation; the review
discloses that a monolithic hand-staged EPEC oracle was not added.
