---
type: plan
id: 2026-08-02_warp-studio-visual-refresh
author: @kunalpuri-prediqt
agent: copilot
created: 2026-08-02T23:30:00 CEST
status: approved
depends_on:
  - 2026-07-31_trame-warp-dam-break-studio
aspects: [host-integration]
host_files: []
within_boundary: true
status_note: implemented, awaiting owner acceptance and review artifact
---

# Plan: Warp studio visual refresh

## Goal

Make the Warp dam-break studio look and feel like a finished product rather than
a default Vuetify form: a coherent design system, clearer information hierarchy,
better empty/loading states, and calmer, more readable telemetry. Presentation
only - no solver, worker, protocol, or state-key changes.

## Context

The app currently works end to end (launch, pause/step/resume, replay, tank and
obstacle context, graceful blow-up handling). The remaining gap is visual: the
drawer is a stack of default-density fields, the right panel is an undifferentiated
list of label/value pairs, the idle viewport shows only an empty wireframe with no
explanation, and the toolbar carries no run progress. The owner asked for a
beautification pass over the whole look and feel.

## Approach

### 1. Design system pass

- Consolidate the ad-hoc CSS into a small token set: background layers, surface
  levels, hairline, accent (cyan), warm accent (orange), text primary/secondary/
  muted, radius scale, elevation scale.
- Define a typography scale: eyebrow, section title, body, metric label, metric
  value. Use tabular/monospace numerals for all numeric telemetry so values stop
  jittering as they update.
- Consistent focus-visible rings and hover transitions on every interactive
  element.

### 2. Toolbar

- Brand lockup (mark + "PySPH · Warp Studio" + small "Warp / CUDA" subtitle).
- Status pill with a soft pulse animation while `running`, static otherwise.
- Thin determinate progress bar across the bottom edge of the toolbar driven by
  the existing `step` / `step_total` state.
- Keep the existing reset-camera and details-panel toggle buttons, restyled.

### 3. Control drawer

- Replace the flat stack with titled section cards (Run, Adaptive region,
  Physics, Visualization) that share one density, one field variant, and icons
  in each section header.
- Resolution mode becomes a two-option segmented control instead of a select.
- "Fixed obstacle" moves inside the Run card next to the geometry inputs.
- Primary action becomes a gradient "Launch GPU run" button; transport controls
  (pause / resume / step) become one grouped, icon-labelled button bar with
  Cancel as a quiet destructive text action beneath.
- Status detail and error alert restyled as an inline status strip pinned at the
  bottom of the drawer.

### 4. Viewport

- Idle empty state: a centred, non-blocking card over the viewport explaining
  that no run has started and pointing at Launch. Hidden as soon as frames exist.
- Busy state: a compact "compiling / initializing CUDA worker" indicator using
  the existing `status` values, replacing the current silent gap.
- Restyle the floating control cluster and the timeline chrome to match the new
  token set; keep the existing LIVE/REPLAY semantics and the rule that the
  slider is disabled while a run is active.

### 5. Run-details panel

- Two-column metric grid for compact scalars, full-width rows for the important
  ones, grouped under Particles / Solver / Health headers with a divider.
- Each metric gets a unit suffix and a subdued icon; mass drift and peak
  pressure get colour-coded health treatment based on existing values only.
- Panel header restyled with the collapse affordance aligned to the new scale.

### 6. Scene polish (vtk_scene.py, minimal)

- Slightly deeper gradient background and a softer floor-grid colour so the
  particle colour map reads better.
- Colour-scale label typography and width tuned to the new chrome.

No change to actor construction, glyph scaling, scalar ranges, or camera logic.

## Files expected to change

- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/app.py`
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/vtk_scene.py`
  (background/colour-scale styling only)
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py`
  (markup assertions updated to the new structure)
- `.ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/README.md`
- Session log, daily closeout, review, and `current.md` per the operating
  contract.

No host PySPH file is touched. No new dependency is added.

## Explicit non-goals

- No change to `worker.py`, the command protocol, `FrameBuffer`, config
  validation, NPZ/manifest formats, or any state key name.
- No change to solver behaviour, numerics, or performance.
- No new Python or JS dependency; Vuetify 3 components and CSS only.

## Tests / validation

- `python -m pytest .ai/implementations/blast-from-the-past/apps/warp_dam_break_studio/test_studio.py`
  (currently `13 passed`).
- `python -m py_compile` on the touched modules.
- Offline `render_result.py` render to confirm the scene changes.
- Hands-on browser pass at `http://127.0.0.1:1234/`: idle empty state, launch,
  running progress bar, pause/step/resume, completion, replay scrub, panel
  collapse, and a narrow-window check.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
  and `git diff --check`.

## Risks

- Trame/Vuetify markup changes can break the existing DOM assertions in
  `test_studio.py`; those assertions are updated deliberately, not deleted.
- Purely visual regressions are only caught hands-on, so the browser pass is a
  required acceptance step, not optional.
- The empty-state and progress overlays read existing state only; if a status
  value is ever added they degrade to hidden rather than erroring.

## Approval

@kunalpuri-prediqt, 2026-08-02 23:30 CEST, verbatim:

> approved

## Deviations found during implementation

Three defects surfaced that the plan had not anticipated. All are inside the
approved boundary and are presentation-layer only:

1. `html.Style(...)` never worked - Vue's runtime template compiler discards
   `<style>` tags, so the studio stylesheet had never been applied in a browser.
   The stylesheet moved to `assets/studio.css`, served through
   `server.enable_module`.
2. The toolbar's built-in flex-grow title slot pushed the brand lockup to the
   centre; it is now hidden with `d-none`.
3. The offscreen render window kept its construction size, so the streamed frame
   was letterboxed. A `client.SizeObserver` bound to `viewport_size` now resizes
   the render window and reframes the camera, and `_set_camera` fits the tank
   bounds for the current aspect.
