---
type: review
date: 2026-08-02
user: @kunalpuri-prediqt
agent: copilot
plan: plans/2026-08-02_warp-studio-visual-refresh.md
adrs: []
aspects_touched: [host-integration]
host_files: []
review_mode: prototype-owner
status: prototype-approved
within_boundary: true
---

# Review: Warp studio visual refresh

## Diff summary

```
 apps/warp_dam_break_studio/README.md          |  40 +-
 apps/warp_dam_break_studio/app.py             | 466 ++++++++++++--------
 apps/warp_dam_break_studio/test_studio.py     |  41 +
 apps/warp_dam_break_studio/vtk_scene.py       |  37 +-
 apps/warp_dam_break_studio/assets/studio.css  | new
 plans/2026-08-02_warp-studio-visual-refresh.md| new
 updates/session-logs/2026-08-02_2320.md       | new
```

All files are inside `.ai/implementations/blast-from-the-past/`. No host PySPH
file is touched and no dependency is added.

## Aspects and host files

- Aspect: `host-integration`.
- Host files: none.
- Boundary: unchanged; `within_boundary: true`.

## What changed

Approved presentation work:

- Design tokens, typography scale and tabular numerals; consistent focus rings.
- Toolbar: brand lockup, status pill that pulses while running, determinate
  progress bar driven by the existing `step` / `step_total` state.
- Drawer: scrolling section area (Run card plus Adaptive region, Physics,
  Visualization panels) and a pinned footer holding the status strip, gradient
  launch button, grouped pause/resume/step transport and cancel. Resolution mode
  is a segmented control driven by the existing `mode_items`.
- Viewport: idle empty-state card, busy pill while the CUDA worker starts,
  restyled control cluster and timeline chrome. LIVE/REPLAY semantics and the
  disabled-while-running scrubber are unchanged.
- Run details: Particles / Solver / Health tiles in a two-column grid, with
  mass drift and peak pressure colour-coded from their own values.

Defects found and fixed during the pass (all presentation-layer):

1. **The stylesheet had never been applied in a browser.** Vue's runtime
   template compiler discards `<style>` tags, so `html.Style(css)` inside the
   layout was silently dropped and every previous version ran on inline element
   styles alone. The stylesheet now lives in `assets/studio.css` and is
   registered with `server.enable_module({"serve": ..., "styles": ...})`.
2. **Brand lockup rendered centred.** `SinglePageWithDrawerLayout` keeps a
   flex-grow toolbar title slot ahead of appended children; it is hidden with
   `d-none`.
3. **Viewport letterboxed.** The offscreen render window kept its construction
   size, so the streamed JPEG never matched the panel. A `client.SizeObserver`
   bound to `viewport_size` resizes the render window and reframes the camera.
   Its payload is `{"size": {...}, "pixelRatio": ...}`, not a flat rect.
4. **Camera framing was aspect-blind.** `_set_camera` now calls
   `ResetCamera(*TANK_BOUNDS)` after orienting the camera.
5. Colour-scale title clipped at the right edge (bar moved to 0.885) and two
   metric labels wrapped (`white-space: nowrap`, "Simulated time" shortened).

## Behavioural / numerical changes

None. `worker.py`, the command protocol, `FrameBuffer`, `validate_run_config`,
the NPZ/manifest formats and every state key are untouched. No solver, kernel,
integrator, neighbour-search or timestep code is in the diff. The only
non-cosmetic runtime effect is that the offscreen render window is now resized
to the browser panel, which changes rendered image dimensions only.

## Raw validation output

```
$ python -m pytest test_studio.py -q
................                                                         [100%]
16 passed in 1.46s

$ curl -s -o /dev/null -w "%{http_code} %{content_type} %{size_download}\n" \
    http://127.0.0.1:1234/studio_assets/studio.css
200 text/css 9538

$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
diff clean
```

Two new tests cover the refreshed chrome markup and the state keys it binds to,
and one covers the render-window resize handler including its rejection of
undersized and empty payloads.

## Visual aid

Hands-on browser pass on the local RTX 4060 instance at
`http://127.0.0.1:1234/`, captured at three points during the session:
unstyled (defect 1 reproduced), styled but letterboxed with the brand centred
(defects 2 and 3 reproduced), and the final state with the stylesheet applied,
the brand left-aligned, the tank filling and framed in the viewport, the colour
scale unclipped and metric labels on one line. Screenshots were reviewed in the
session but are not committed to `.ai/`.

## Risks

- The visual refresh is only verified on one browser and one window size range.
  Narrow windows still push the details panel over the viewport by the
  pre-existing `max-width: 1050px` rule.
- `client.SizeObserver` re-renders and re-captures a JPEG on every viewport
  resize. Rapid dragging of the window will issue extra offscreen renders; it is
  bounded by the observer's own coalescing but was not load-tested.
- The stylesheet is now a separate served asset, so a stale browser cache can
  show old styling until a hard reload.

## Unresolved questions

- Should the details panel become a proper overlay drawer below 1050px instead
  of covering the viewport?
- Should the render window track `devicePixelRatio` for sharper frames on HiDPI
  displays, at the cost of larger JPEG payloads?

## Prototype-owner conditions

- Exploratory prototype work inside the approved boundary: `.ai/` memory and
  `apps/warp_dam_break_studio/` only.
- The Tier-2 plan `2026-08-02_warp-studio-visual-refresh` is recorded and
  approved.
- No generic public API/ABI, non-Warp host behaviour, dependency, build or
  release configuration is altered, and no file outside the boundary is touched.
- Performance and behaviour claims here are prototype evidence, not production
  results.
- Commit subject begins with `prototype:`.

## Verdict

Prototype-owner authorization, @kunalpuri-prediqt, 2026-08-02 23:58 CEST,
verbatim:

> approved, commit and push

This is not promotion approval. A cumulative promotion review with exact
`@prabhu: LGTM` remains required before any of this reaches an upstream PR, a
production/release branch, or is presented as completed production work.
