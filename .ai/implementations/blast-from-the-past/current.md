# Current - blast-from-the-past

Updated: 2026-08-06T16:46:00 CEST by codex

**Status:** The Warp dam-break composes dynamic two-level fluid resolution, a
Liu-coupled floating 6-DOF body, tank contact, restartable quaternion pose, and
studio controls/telemetry. Exact separate destination/source smoothing-length
gradients are now implemented and locally conservative, but they fail the
hydrostatic-interface and matched-time equilibrium gates. The step-900 rerun
passes surge, height, COM, finite/mass/contact-geometry gates while failing
energy, angular impulse, merge, and particle reduction. This remains an
experimental prototype, not production APR.

**Gameplay checkpoint (2026-08-06):** ADR-0012 is Accepted for a separate,
explicitly approximate `Gameplay · fast` studio column. The measured Fast
profile has 1,000 fluid particles; Rich uses 7,020 for local presentation.
Both use a device Warp `HashGrid`, three bounded density projections per
`1/60 s` frame, XSPH smoothing, analytic tank bounds, and a bounded
impulse-coupled quaternion box. Rich completed 250 steps with solver median
`2.345 ms`; Fast remains the 96-KiB transport baseline. Gameplay rejects
spacing below `0.04 m`, Cancel preempts a CUDA-bound worker after a short
grace period, and rotated-box support replaces the hovering bounding-sphere
floor clamp. Full Warp separation is `79 passed`; studio/transport is `34`.

**Gameplay presentation checkpoint (2026-08-06):** ADR-0013 is Accepted.
Gameplay defaults to a display-only client WebGPU screen-space surface with an
explicit VTK Particles fallback; Adaptive/Uniform remain VTK-only. Warp emits
smoothed centers and bounded anisotropic frames without mutating PBF state.
At a 1280x720 Edge 151 viewport on the NVIDIA Blackwell adapter, WebGPU
completion measured `3.30/5.80 ms` median/p95, upload `0.00/0.10 ms`, server
packing `0.182/0.224 ms`, and the Fast versioned frame is 85,336 bytes with zero
dropped/stale replay frames. Final/depth/thickness/normal views, moving body,
camera, resize, replay, and Surface/Particles switching passed. Node `6`,
studio/transport `34`, and Warp separation `79` tests pass. The tuned Rich
view renders near `4.0 ms` but its 599,040-byte base64 frame is local-only
quality evidence pending binary transport or display decimation. Review:
`reviews/2026-08-05_warp-webgpu-fluid-surface-renderer.md`.

**Geospatial checkpoint (2026-08-06):** ADR-0014 is Accepted for an isolated
fourth `Geospatial` profile. A 256² Warp finite-volume SWE path evolves
`(h, hu, hv)` with Rusanov fluxes, hydrostatic reconstruction, CFL stepping,
wet/dry positivity, Manning friction and explicit boundaries; the regular
terrain/water grids use a dedicated WebGPU renderer. The exact worker measured
`0.346/1.425 ms` solver median/p95 and `6.04e-8` volume drift over 300 steps;
the editor browser measured `7.70/19.80 ms` WebGPU completion median/p95 with
a 341.3-KiB dynamic frame. Warp separation is `86 passed`, studio/transport/
importer `47`, and Node WebGPU `11`. Official NASADEM N43E006 acquisition is
blocked by Earthdata HTTP 401; the app therefore visibly uses the deterministic
fixture. The strict importer and automatic prepared-asset routing are ready,
and no third-party DEM was substituted. Review:
`reviews/2026-08-06_warp-nasadem-shallow-water-geospatial-demo.md`.

**Terrain SPH checkpoint (2026-08-06):** ADR-0015 is Accepted for an isolated
fifth `Terrain SPH` profile. It routes the established uniform 3D Warp WCSPH
solver through 72 stationary solid particles sampled beneath a bounded
Gaussian hill and renders the same parameters as a smooth VTK surface. A
matched 250-step `dx=0.1` comparison is finite with zero fluid-mass drift and
bitwise-stationary hill coordinates; versus the flat case, 503/1,000 fluid
particles move by more than 1 mm, RMS change is 9.76 mm, maximum change is
13.58 cm, and the surge front is 3.11 cm behind. Warm hill cost is 8.13 ms/step
versus 5.34 ms flat on the RTX 5090. The editor-browser run completed 250/250
and shows the physical water front against the smooth hill. This is a
procedural interaction smoke, not NASADEM, terrain convergence, adaptation, or
game-frame-rate evidence. Review:
`reviews/2026-08-06_warp-terrain-sph-smooth-hill.md`.

**Composition checkpoint (2026-08-04):** `DamBreakConfig.obstacle_mode` now
selects `none`, `fixed`, or `floating`. Floating mode routes adaptive fluid,
wall, and body through one multilevel NNPS; the rigid driver accepts grid or
multilevel traversal. Snapshots, telemetry, NPZ restart, and the studio include
the moving body. A rebuild regression prevents stale host coordinates from
rewinding the body at adaptation checkpoints.

**Rigid/contact checkpoint:** ADR-0008 selects deterministic axis-aligned plane
contact with a stiffness timestep restriction. ADR-0010 selects a persistent
f64 quaternion plus immutable body-frame particle coordinates. The fresh
900-step adaptive case has zero reported geometric penetration, rigid error
zero, and relative geometry drift `2.27694e-7`.

**APR/long comparison:** ADR-0009's icosa13 transfer, first-order
reconstruction, complete-family merge, and adaptive-only averaged-`HIJ`
grad-`h` consistency pass local oracles, including four repeated translating
split/merge cycles. Tuned defaults keep the static refinement box, disable
quadratic shifting, and use zero merge hysteresis. At corrected step 900 the
adaptive result still has 3,376 fluid particles versus 2,744 for mass-matched
uniform `dx=1/14`. Surge error is 0.09767 m, height error 0.00213 m, body-COM
distance 0.04582 m, and kinetic-energy difference 5.94%; only energy narrowly
misses its 5% screen. Fluid angular impulse differs 58.72% and final body
angular velocity 95.21%, with no complete flow-driven family merge. See
experiments `2026-08-04_warp-adaptive-floating-long-comparison` and
`2026-08-05_warp-physics-validation-ladder`.

**Physics ladder (2026-08-05):** Rigid telemetry now separates fluid/contact
force and torque and integrates their impulses. Equal-and-opposite Liu
reaction conserves net force and torque in an asymmetric variable-`h` oracle,
but the current averaged-`HIJ` beta approximation increases manufactured
interface horizontal residual RMS by 12.74% and vertical hydrostatic residual
RMS by 26.75%. The adaptive pressure range is `-22.10..35.43 kPa`, versus
`-7.80..23.84 kPa` uniform fine. Floating-equilibrium refinement reduces
vertical/rotational drift but leaves sign-changing lateral drift.

**Separate-`h` follow-up:** ADR-0011 adds opt-in generated `gradi`/`gradj`
factors and uses their exact ownership in beta, continuity, pressure, and Liu
reaction. Kernel oracles pass for cubic, Gaussian, and Wendland kernels in
fp32/fp64; equal-`h`, unequal-support, fluid force/torque, and rigid reaction
oracles pass. The padded interface still worsens versus uncorrected by 24.16%
horizontally and 19.34% vertically. Matched-time equilibrium improves vertical
and rotational drift under refinement but lateral displacement changes from
`+2.233 cm` to `-2.466 cm`. At adaptive step 900, surge error is `0.07604 m`,
height error `0.00148 m`, and COM distance `0.03927 m`, while kinetic energy
differs 9.79%, fluid angular impulse 63.15%, body angular velocity 82.71%, and
the negative pressure tail reaches `-25.68 kPa`. The operator remains an
experimental mathematically faithful path; no coefficient tuning was used.

**Latest work (2026-08-02):** Tier-2 plan
`2026-08-02_warp-studio-visual-refresh`, review
`reviews/2026-08-02_warp-studio-visual-refresh.md`, session log
`updates/session-logs/2026-08-02_2320.md`, daily closeout
`updates/daily/2026-08-02.md`. Three defects fixed alongside the refresh:
the stylesheet had never been applied because Vue's runtime template compiler
discards `<style>` tags (now served from `assets/studio.css` through
`server.enable_module`); the toolbar's flex-grow title slot centred the brand;
and the offscreen render window kept its construction size, letterboxing the
streamed frame (now driven by `client.SizeObserver` plus aspect-aware
`ResetCamera`). Studio suite `16 passed`; `validate-memory` PASS.

**Adaptive dam-break checkpoint:** `pysph/base/warp_adaptive.py` adds stable
particle/family identity, deterministic equal-mass eight-child split,
complete-family merge, incremental Warp stepping, snapshots, telemetry, and
NPZ persistence. Adaptation is host-orchestrated at explicit checkpoints;
generated WCSPH equations and `MultilevelGridWarpNNPS` run on CUDA between
them. The equal-mass stencil is software-smoke evidence, not production APR.

The RTX 5090 250-step case completed in 4.474 cached seconds to simulated
`t=0.131621`, with 1,721 fluid particles (897 coarse + 824 fine), 103 split
parents, exact mass 1,000, zero relative mass drift, and finite state. No
complete family left the fine region during this short run; merge is covered
by deterministic controller tests.

**Browser studio:** The local Trame 3/Vuetify 3 app under
`apps/warp_dam_break_studio/` exposes Adaptive/Uniform/Gameplay/Geospatial/
Terrain SPH,
physics and refinement controls, VTK and isolated WebGPU viewports, scalar
coloring, pause/resume/single-step/cancel, telemetry, and bounded replay.
CUDA runs in a spawned worker; final NPZ and JSON manifest include runtime GPU
metadata. The worker protocol passed pause/step/resume on the RTX 5090.

**Review state:** prototype-owner authorization recorded:

- `reviews/2026-07-31_warp-two-level-adaptive-dam-break-smoke.md`
- `reviews/2026-07-31_trame-warp-dam-break-studio.md`
- `reviews/2026-08-02_warp-studio-visual-refresh.md`

Prototype authorization permits commits and pushes to the owner's fork.
Cumulative exact `@prabhu: LGTM` remains required before upstream promotion or
merge into a production/release branch.

**Known limitations:** the proposed transfer and both tested
variable-resolution consistency operators have local conservation evidence,
but production-flow coarsening, hydrostatic/torque convergence, equilibrium,
energy/angular impulse, and particle reduction still fail. The exact
separate-`h` form does not by itself correct the mixed-resolution interface.
Adaptation still performs full host fluid
pulls/reconstruction; device pool/free-list allocation and compaction are
absent; optional shifting is host `O(N^2)` and disabled by default. The 4x
particle and flow-driven merge gates fail. Formal warm GPU timing/memory
scaling, full CPU Application parity, and hands-on studio acceptance remain
incomplete.

**Active plans:** Tier-2 plan
`2026-08-06_warp-terrain-sph-smooth-hill` is implemented and ready for owner
publication; prototype-owner commit/push authorization is recorded. Tier-2 plan
`2026-08-06_warp-nasadem-shallow-water-geospatial-demo` is implemented through
the synthetic fallback with prototype-owner commit/push authorization;
authenticated official NASADEM acquisition/crop remains its explicit
external-data blocker. Tier-2 plan
`2026-08-05_warp-webgpu-fluid-surface-renderer` and its parent
`2026-08-05_warp-gameplay-pbf-dam-break-column` are completed and
prototype-owner commit/push authorization is recorded. Non-local latency,
binary transport,
and secondary spray/foam remain follow-ups. Tier-2 plan
`2026-08-05_warp-separate-h-variable-resolution-gradients` was approved
verbatim with `"approved"` at 2026-08-05T12:05:00 CEST and is implemented with
failed physical gates. Its parent
`2026-08-04_warp-adaptive-floating-body-production` remains incomplete: the
transfer candidate is not accepted, device-resident mutation is unimplemented,
and the long validation gates fail. Checkpoint review:
`reviews/2026-08-04_warp-adaptive-floating-body-checkpoint.md`.
