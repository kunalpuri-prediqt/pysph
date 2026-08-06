# Known Issues - host-integration

- [2026-08-02] Trame/Vue's runtime template compiler discards `<style>` tags,
  so `html.Style(...)` inside a layout is silently dropped. Client CSS must be
  served and registered through `server.enable_module({"serve": ...,
  "styles": ...})` (or `client.Style`). The studio ran without its stylesheet
  from creation until this was found.
- [2026-08-02] `client.SizeObserver` writes
  `{"size": {"x", "y", "width", "height"}, "pixelRatio", "dpi"}` to state, not
  a flat rect. Reading `width`/`height` at the top level silently no-ops.
- [2026-08-05] Trame-client 3.12's `Handler(variable=...)` automatic watcher
  exposes a slot-local ref with that name rather than binding the same-named
  server state in this app. Use a distinct slot variable plus a nested
  `ClientStateChange(value=(state_name,))` that calls `handler.run($event)`.
  Reusing the state name shadows it and leaves the renderer stuck at WAITING.
- [2026-08-05] `register_external_script(Path(...))` copies into the installed
  `trame_client` package. The studio redirects that generated serving cache to
  `/tmp/pysph-trame-client-scripts` so immutable environments remain usable.
- [resolved 2026-08-05] Windows Edge 151 completed the WSL loopback websocket,
  CUDA worker, WebGPU shader/pipeline creation, captures, replay, orbit/zoom,
  resize, and Surface/Particles switching through a CDP hardware session.
- [resolved 2026-08-06] Cooperative worker cancellation could not interrupt a
  long CUDA step. `SolverWorker.cancel()` now allows a short cooperative grace
  period, then terminates/kills and releases the spawned worker; an actual
  browser run returned to `cancelled` in about 1.2 seconds.
- [2026-08-06] Gameplay rejects `dx < 0.04 m`. The bound prevents accidental
  near-million-particle launches, but it is a product safety limit rather than
  a memory model; future quality profiles still need explicit particle/byte
  budgets.
- [2026-08-06] A cold Terrain SPH browser worker using the separate `/tmp`
  Warp cache spent about one minute compiling a generated 3D WCSPH module.
  Cached steps are fast, but the UI currently reports the cold compile inside
  total run wall time rather than as separate initialization telemetry.
- [resolved 2026-08-06] Completed Terrain metrics left `obstacle_mode=hill` in
  shared UI state, so switching directly to Gameplay forwarded an unsupported
  obstacle. Profiles now force their own hill/floating identities and preserve
  the scientific obstacle selection independently.
