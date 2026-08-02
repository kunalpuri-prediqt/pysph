# Current - blast-from-the-past

Updated: 2026-08-02T23:58:00 CEST by copilot

**Status:** The Warp backend has a reviewed engineering checkpoint for dynamic
two-level particle resolution and a local Trame/VTK browser studio. The studio
has just had an approved visual refresh with prototype-owner authorization to
commit and push to the owner's fork.

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

**Known limitations:** production APR still needs a reconciled transfer
operator, transition correction/shifting, device-side allocation/compaction,
longer obstacle-impact and convergence evidence. The refreshed studio UI is
verified on one browser only, and below 1050px the details panel still covers
the viewport.

**Next action:** deploy the refreshed studio to `gcp-prediqt-rtx6000x1` and
re-run the tunnel acceptance pass. A later Tier-2 plan should choose between
device-resident adaptation and the literature/convergence checkpoint.
