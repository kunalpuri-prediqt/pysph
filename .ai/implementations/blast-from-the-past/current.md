# Current - blast-from-the-past

Updated: 2026-07-31T12:55:00 CEST by codex

**Status:** The Warp backend now has a reviewed engineering checkpoint for
dynamic two-level particle resolution and a local Trame/VTK browser studio.
Both have prototype-owner authorization for a local `prototype:` commit.

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

**Latest validation:** adaptive/app unit suite `19 passed`; Warp NNPS
`35 passed`; Warp codegen `10 passed`; complete Warp SPH file `58 passed` in
38:59. The isolated Warp files emit only the existing two Python 3.14 ctypes
deprecations from `warp-lang`. Memory validation and `git diff --check` pass.

**Review state:** prototype-owner authorization recorded:

- `reviews/2026-07-31_warp-two-level-adaptive-dam-break-smoke.md`
- `reviews/2026-07-31_trame-warp-dam-break-studio.md`

Prototype authorization permits the requested commit and push to the owner's
fork for VM deployment. Cumulative exact `@prabhu: LGTM` remains required
before upstream promotion or merge into a production/release branch.

**Known limitations:** production APR still needs a reconciled transfer
operator, transition correction/shifting, device-side allocation/compaction,
longer obstacle-impact and convergence evidence. Automated Windows Edge capture
could not complete Trame's WSL loopback WebSocket, so hands-on browser
orbit/pan/zoom and local/remote switching remain an acceptance check.

**Next action:** push the reviewed commit, deploy it to
`gcp-prediqt-rtx6000x1`, and run the app through a local SSH tunnel. A later
Tier-2 plan should choose between device-resident adaptation and the
literature/convergence checkpoint.
