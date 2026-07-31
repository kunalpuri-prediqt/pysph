# Known Issues - host-integration

- [2026-07-31] Automated Windows Edge capture could fetch the Trame app served
  inside WSL but did not complete the loopback WebSocket. The local server,
  spawned CUDA worker, and EGL VTK scene are validated; hands-on browser
  orbit/pan/zoom and local/remote switching remain an acceptance check.
