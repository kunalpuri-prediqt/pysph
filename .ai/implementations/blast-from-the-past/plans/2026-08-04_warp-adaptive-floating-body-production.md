---
type: plan
id: 2026-08-04_warp-adaptive-floating-body-production
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-04T14:50:00 CEST
status: approved
depends_on:
  - 2026-06-20_warp-liu-fluid-rigid-coupling-p3
  - 2026-07-06_warp-dynamic-adaptive-particle-resolution
  - 2026-07-31_warp-two-level-adaptive-dam-break-smoke
  - 2026-07-31_trame-warp-dam-break-studio
adrs: [ADR-0006, ADR-0007]
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_adaptive.py
  - pysph/base/warp_device_helper.py
  - pysph/base/warp_multilevel_nnps.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_adaptive.py
  - pysph/base/tests/test_warp_device_helper.py
  - pysph/base/tests/test_warp_nnps.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: production-track adaptive Warp dam-break with a floating body

## Goal

Turn the current two separate prototypes—host-orchestrated adaptive fluid with
a fixed obstacle, and uniform-resolution fluid with a Liu-coupled 6-DOF box—
into one Warp simulation and studio mode with adaptive fluid, a floating rigid
body, wall contact, production-oriented particle mutation, and explicit
scientific and performance gates.

The requested scope is the full ten-item review follow-up. Completion means the
composition works, the long impact/contact case is validated, and every
correctness/performance target is reported as passed or missed. It does not
permit relabelling a missed target as success.

## Existing foundations

- ADR-0006: device-resident rigid state, f64 force/torque reduction, RK2 motion,
  and deterministic two-pass Liu coupling.
- ADR-0007: exact device-built multilevel neighbor traversal with per-level
  dense/sparse storage.
- `WarpDamBreakSimulation`: incremental adaptive fixed-obstacle runner with
  stable particle/family identity and checkpoint persistence.
- Trame studio: spawned CUDA worker, snapshots, replay, telemetry, and
  uniform/adaptive configuration.

## Non-negotiable invariants

- Existing uniform fixed-wall and uniform rigid-body behavior remains covered
  by regression tests; new paths are additive or explicitly parameterized.
- Fluid, rigid, and fixed-wall arrays remain distinct. Fluid adaptation must
  never compact, split, merge, or reinterpret rigid/body particle identity.
- Rigid geometry is advanced only through the rigid transform, never through a
  fluid PEC particle stage.
- Accepted neighbor sets remain exact under mixed fluid/body smoothing lengths.
- Adaptation conserves mass and linear momentum within the gates below.
- Device errors and non-finite state fail the run immediately with actionable
  diagnostics.
- Performance claims use warm-cache repeated runs on the same GPU and include
  adaptation, contact, output cadence, active particles, and step count.

## Phase 0 — freeze baselines and resolve design choices

1. Re-run and archive three baselines on the selected CUDA host:
   - uniform fixed-obstacle dam-break;
   - two-level adaptive fixed-obstacle smoke;
   - uniform floating-box collision-free transient.
2. Add deterministic small fixtures that record fluid, wall, and body fields,
   rigid COM/velocity/angular velocity, conserved quantities, accepted neighbor
   sets, and step timestep before composition changes.
3. Audit primary/reference implementations for:
   - the split/merge reconstruction and variable-resolution consistency term;
   - particle shifting/free-surface limiting;
   - PySPH rigid-wall collision/contact and timestep restrictions.
4. Run kill tests for candidate transfer and contact formulations before
   selecting them. Record the selected transfer/correction/contact model in new
   ADRs; regenerate the decision graph. Do not encode a production choice from
   visual plausibility alone.

Checkpoint gate:

- All current focused Warp suites pass and baseline artifacts are reproducible.
- Candidate contact does not add energy in a one-body drop/bounce fixture beyond
  the documented numerical tolerance.
- Candidate split/merge reproduces constant fields and has measured linear-field
  error/convergence before solver integration.

## Phase 1 — compose adaptive fluid and the floating body

1. Add an explicit obstacle mode to `DamBreakConfig`: `none`, `fixed`, or
   `floating`. Preserve `with_obstacle` loading compatibility for old manifests.
2. Construct the floating box as a rigid ParticleArray with physical mass,
   body identity, Liu properties, and a persistent `WarpRigidBodyState`.
3. Generalize `wc_sph_dam_break_rigid_step` and its equation launches to accept
   `neighbor_mode`, including `multilevel`, instead of hard-coding `grid`.
4. Route fluid, wall, and body through one `MultilevelGridWarpNNPS` in adaptive
   mode. Check exact cross-array neighbors against brute force for coarse/fine
   fluid interacting with the rigid surface.
5. When fluid adaptation changes particle count, replace only the fluid array,
   rebuild the multilevel NNPS, preserve compact rigid state, and explicitly
   re-establish any NNPS-dependent cached/pre-pass state. Assert that body
   particle count, mass, identity, and relative geometry are unchanged.
6. Extend snapshots, metrics, NPZ/JSON persistence, and restart loading with
   body coordinates/velocities plus COM, linear/angular velocity, force/torque,
   orientation state, and rigid error flags.
7. Add studio controls for obstacle mode and body density/initial placement;
   show body telemetry and render the current moving body rather than a static
   context cube when floating mode is selected.

Checkpoint gate:

- One-step and repeated-step uniform-grid rigid results remain within existing
  fp32/fp64 tolerances.
- Multilevel and uniform-grid rigid equation outputs match on an identical
  mixed-resolution fixture.
- A short adaptive floating-body run contains both fluid levels, records a split,
  moves/rotates the body under computed fluid reaction, remains finite, and
  preserves rigid pair distances to relative `1e-5`.
- Pause/resume/single-step/replay and save/restore work in floating mode.

## Phase 2 — add rigid-wall collision/contact

1. Implement the Phase-0-selected contact formulation as an additive Warp path
   between rigid-body particles/state and fixed walls.
2. Accumulate contact force/torque deterministically into the existing f64 rigid
   reduction path. Do not reintroduce nondeterministic fp32 source atomics.
3. Add penetration, restitution/damping, friction (if selected by the ADR), and
   contact-state diagnostics. Add a contact-aware timestep restriction if the
   kill tests show the fluid CFL criterion is insufficient.
4. Validate drop, slide, glancing impact, corner/multiple-contact, and no-contact
   fixtures before enabling contact in the dam-break case.
5. Add a behavior-stability guard for EPEC/RK2 force evaluation and rigid update
   placement across predictor/corrector stages.

Checkpoint gate:

- No persistent wall penetration greater than `0.25 * dx_fine` in the fixture
  suite unless the selected reference model establishes a stricter threshold.
- Contact forces remain finite; static/no-contact cases receive zero contact
  impulse; damping does not create mechanical energy.
- The collision-free P3 transient remains within its baseline tolerances.

## Phase 3 — replace smoke transfer with scientifically reconciled APR

1. Implement the selected 3D split operator and complete-family conservative
   merge with reconstruction of all WCSPH, saved-stage, and identity fields.
2. Add sibling variable-resolution equation blocks for the selected partition/
   kernel-consistency correction; leave uniform generated blocks unchanged.
3. Implement iterative particle shifting/regularization and first-order property
   correction with free-surface and fixed/rigid-solid limiters.
4. Add target-level hysteresis and bounded adjacent resolution jumps to prevent
   split/merge thrashing near the body and moving free surface.
5. Keep solids at a compatible fixed sampling for this checkpoint; dynamically
   adapting rigid or wall particles remains out of scope.

Checkpoint gate:

- Constant fields reproduce through repeated split/merge/shift to roundoff.
- Linear-field and kernel-summation error decreases under refinement.
- Hydrostatic and translating-fluid fixtures repeatedly cross the refinement
  boundary without secular density/pressure growth.
- Per-event relative mass and linear-momentum residual are each `<= 1e-5`.
- A uniform target level through the APR sibling path matches the existing
  uniform solver within fp32/fp64 tolerance.

## Phase 4 — move adaptation allocation and compaction onto the GPU

1. Add a device particle-pool/free-list representation with explicit capacity,
   stable IDs, family ownership, allocation status, and old/new index maps.
2. Implement device classification, deterministic split/merge candidate
   selection, scan allocation, property transfer, and compaction.
3. Integrate pool capacity growth with `WarpDeviceHelper` without changing the
   generic ParticleArray public API or Cython ABI.
4. Rebind/rebuild multilevel NNPS structures from compacted device arrays while
   keeping rigid compact state and particle identity valid.
5. Remove full per-particle host pulls and ParticleArray reconstruction from
   steady adaptation checkpoints. Retain explicit checkpoint pulls only for
   output and validation.
6. Eliminate or bound the remaining per-update multilevel metadata readback;
   measure sparse lookup overhead and optimize occupied-range traversal if it
   dominates the adaptive case.

Checkpoint gate:

- A guard fails if steady adaptation pulls per-particle fluid properties or
  reconstructs the host ParticleArray.
- Pool exhaustion/capacity growth is deterministic and preserves every live
  property and stable ID.
- Device and NumPy oracle mutations match for split, merge, simultaneous mixed
  operations, and repeated compaction.
- Conservation and scientific Phase-3 gates remain satisfied.

## Phase 5 — long impact/contact and merge validation

1. Run the adaptive floating-body dam-break through surge impact, sustained
   fluid/body coupling, first wall contact, rebound/settling, and enough
   downstream motion to exercise flow-driven complete-family merges.
2. Save matched checkpoints for adaptive, uniform-coarse, and uniform-fine
   cases using identical geometry, body mass, physics, timestep policy, and
   output cadence.
3. Record fluid observables: surge front, maximum height, density/pressure
   ranges, kinetic energy, mass/momentum, neighbor counts, and transition error.
4. Record body observables: COM path, orientation, linear/angular velocity,
   fluid/contact impulses, penetration, total force/torque, geometry drift, and
   device error.
5. Run the shipped CPU rigid-body Application when the Compyle/Python runtime is
   compatible. Until then, keep the faithful NumPy primitive parity and mark
   full CPU-Application parity as blocked rather than silently omitted.

Scientific gates:

- All state finite and rigid device error zero.
- Total relative mass drift attributable to adaptation `<= 1e-5`.
- Surge front and maximum height within two finest-particle spacings of the
  uniform-fine case at matched checkpoints.
- Developed-flow kinetic energy and time-windowed body pressure/fluid impulse
  within 5% of uniform-fine. Report peak pressure separately; do not use it as
  the primary chaotic-flow metric without convergence.
- Body COM within two finest-particle spacings and integrated linear/angular
  impulse within 5% of uniform-fine over the comparison window.
- Rigid geometry drift `<= 1e-5`, contact penetration within the Phase-2 gate,
  and no persistent void/banding or unbounded density error at level interfaces.
- At least one flow-driven merge occurs; deterministic controller-only merge
  evidence is not sufficient for this phase.

## Phase 6 — performance, scale, studio acceptance, and promotion review

1. Benchmark adaptive versus uniform-fine on the same GPU with warm JIT cache,
   at least three measured repetitions, and separately report:
   physics time, NNPS time, adaptation time, contact time, snapshot/output time,
   peak active particles, peak device memory, step count, and simulated time.
2. Measure the impact of the global fine-particle timestep. If it prevents a
   runtime win, report the result and identify local/multirate stepping as a
   later project rather than hiding the miss.
3. Run a capacity demonstration whose uniform-fine equivalent exceeds device
   memory while the adaptive case fits. Clearly label the uniform number as an
   estimate and anchor correctness to the smaller runnable comparison.
4. Complete hands-on studio acceptance: fixed/floating/none modes, adaptive and
   uniform runs, live moving-body rendering, pause/step/resume/cancel, replay,
   telemetry, failure reporting, NPZ/JSON restoration, and responsive layout.
5. Produce phase reviews and a cumulative promotion review. No upstream merge
   or production-complete claim occurs without exact `@prabhu: LGTM`.

Scale gates:

- At least 4x fewer peak active fluid particles than equivalent uniform-fine.
- At least 2x lower measured time-to-solution than a runnable uniform-fine case
  on the same GPU. A miss is an honest failed gate, not a blocker to documenting
  correctness or memory savings.
- The adaptive capacity case completes within measured device memory.

## Files expected to change

Host files are limited to the frontmatter list and remain within the existing
Warp prototype boundary. The studio and experiment work lives under:

- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/`
- `.ai/implementations/blast-from-the-past/experiments/`
- `.ai/implementations/blast-from-the-past/references/`
- required plans, ADRs, reviews, session logs, closeouts, and aspect memory.

If contact or CPU parity requires changing generic PySPH equations, dependency
metadata, Cython ABI, shipped examples, or public interfaces, stop and amend
this plan with `within_boundary: false` before touching those files.

## Validation commands

- Focused controller, device-helper, NNPS, codegen, SPH, and studio tests at
  every checkpoint.
- NumPy/brute-force oracles for transfer, compaction, cross-level/cross-array
  neighbors, contact, fluid/rigid reaction, and conservation.
- Existing uniform regression and generated-source/cache-stability guards.
- Separate GPU experiment commands with hardware, Warp version, precision,
  commit, configuration, cache state, and raw output recorded.
- `validate-memory.py` and `git diff --check` before every review.

The large Warp suites may be separated by process because of the documented
module/JIT accumulation behavior; every omitted test or unavailable external
runtime must be recorded explicitly.

## Risks

- This combines two individually reviewed prototypes; a plausible animation can
  hide cross-level force, transfer, or contact errors.
- Contact stiffness can introduce a timestep smaller than fluid CFL and erase
  the expected performance gain.
- A global fine-level timestep may yield memory savings without the 2x runtime
  target.
- Device compaction can invalidate rigid mappings, stable IDs, cached arrays,
  saved integrator state, and replay/output assumptions.
- Split/merge and shifting near a moving solid can create voids, penetration,
  density spikes, or non-conservative impulse transfer.
- Long fp32 free-surface trajectories are chaotic; validation emphasizes
  converged/integrated observables rather than pointwise particle identity.
- Full CPU rigid-body parity currently depends on resolving the recorded
  Compyle/Python compatibility problem.

## Out of scope

- Adaptive rigid or wall particles.
- Local/asynchronous/multirate timestepping.
- Multi-GPU/MPI domain decomposition and dynamic load balancing.
- Public multi-user deployment or authentication.
- Arbitrary continuous smoothing lengths or solution/ML-driven refinement.
- Photorealistic rendering beyond scientific studio/review output.
- Generic PySPH API/ABI changes without an approved boundary amendment.

## Review checkpoints and commit policy

Each phase gets its own raw validation outputs and review artifact. Prototype
commits may use the owner exception only while every restriction in the
implementation contract remains true. Any dependency/build/public-interface or
out-of-boundary change uses promotion review. A cumulative exact
`@prabhu: LGTM` remains mandatory before upstream promotion.

## Execution checkpoint — 2026-08-04

- Phase 0: completed baselines and kill tests.
- Phase 1: engineering composition, persistence, restart, and studio paths
  implemented; focused gates pass.
- Phase 2: implemented and accepted through ADR-0008; focused gates pass.
- Phase 3: transfer/reconstruction/shifting candidate implemented as proposed
  ADR-0009, but its full consistency and boundary-crossing gates fail or remain
  unverified. It is not accepted.
- Phase 4: not implemented. Host pulls and ParticleArray reconstruction remain.
- Phase 5: fresh corrected step-900 comparison completed. Finite, mass,
  penetration, maximum-height, body-COM, and rigid-geometry gates pass. Surge,
  body response/impulse, flow-driven merge, and particle-count gates fail;
  energy/interface and full CPU-Application gates remain incomplete.
- Phase 6: formal warm performance/capacity and hands-on acceptance remain
  incomplete. No production/upstream promotion is authorized.

See review `2026-08-04_warp-adaptive-floating-body-checkpoint` and experiment
`2026-08-04_warp-adaptive-floating-long-comparison`.

## Approval

- [x] Scope requested in chat as items 1–10.
- [x] Plan posted in chat.
- [x] Explicit approval received after this artifact exists.
- Approved by: @kunalpuri-prediqt at 2026-08-04T15:12:27 CEST
- Approval, verbatim quote:
  > approved
