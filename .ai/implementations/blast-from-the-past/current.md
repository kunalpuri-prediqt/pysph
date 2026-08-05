# Current - blast-from-the-past

Updated: 2026-08-05T14:35:00 CEST by codex

**Status:** The Warp dam-break composes dynamic two-level fluid resolution, a
Liu-coupled floating 6-DOF body, tank contact, restartable quaternion pose, and
studio controls/telemetry. Exact separate destination/source smoothing-length
gradients are now implemented and locally conservative, but they fail the
hydrostatic-interface and matched-time equilibrium gates. The step-900 rerun
passes surge, height, COM, finite/mass/contact-geometry gates while failing
energy, angular impulse, merge, and particle reduction. This remains an
experimental prototype, not production APR.

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
`apps/warp_dam_break_studio/` exposes uniform/adaptive options, physics and
refinement controls, a VTK particle viewport, scalar coloring, local/remote
rendering, pause/resume/single-step/cancel, telemetry, and bounded replay.
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
`2026-08-05_warp-separate-h-variable-resolution-gradients` was approved
verbatim with `"approved"` at 2026-08-05T12:05:00 CEST and is implemented with
failed physical gates. Its parent
`2026-08-04_warp-adaptive-floating-body-production` remains incomplete: the
transfer candidate is not accepted, device-resident mutation is unimplemented,
and the long validation gates fail. Checkpoint review:
`reviews/2026-08-04_warp-adaptive-floating-body-checkpoint.md`.
