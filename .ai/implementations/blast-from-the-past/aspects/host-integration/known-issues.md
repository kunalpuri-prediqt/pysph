# Known Issues - host-integration

- [2026-08-02] Trame/Vue's runtime template compiler discards `<style>` tags,
  so `html.Style(...)` inside a layout is silently dropped. Client CSS must be
  served and registered through `server.enable_module({"serve": ...,
  "styles": ...})` (or `client.Style`). The studio ran without its stylesheet
  from creation until this was found.
- [2026-08-02] `client.SizeObserver` writes
  `{"size": {"x", "y", "width", "height"}, "pixelRatio", "dpi"}` to state, not
  a flat rect. Reading `width`/`height` at the top level silently no-ops.
- [2026-07-31] Automated Windows Edge capture could fetch the Trame app served
  inside WSL but did not complete the loopback WebSocket. The local server,
  spawned CUDA worker, and EGL VTK scene are validated; hands-on browser
  orbit/pan/zoom and local/remote switching remain an acceptance check.
