# Open Questions - host-integration

- [open] Should Warp be an optional dependency, an extra, or only an experiment dependency at first?
- [open] Which CI or local validation tier should eventually exercise Warp?
- [closed 2026-08-05] Gameplay now defaults to direct client WebGPU
  screen-space reconstruction; VTK/JPEG remains its explicit particle fallback
  and the only renderer for Adaptive/Uniform.
- [open 2026-08-05] If gameplay frames move beyond the local 1,000-particle
  profile, should base64 Trame state be replaced by binary websocket transport
  or a native shared-memory presentation path?
- [open 2026-08-06] Should a future arbitrary terrain-heightfield WCSPH path
  remain a studio-local profile or gain a reusable Warp terrain-boundary API?
