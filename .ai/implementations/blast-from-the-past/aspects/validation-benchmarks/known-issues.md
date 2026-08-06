# Known Issues - validation-benchmarks

- [2026-08-05] Gameplay's `574.7 FPS` measurement is the server
  solver-plus-snapshot loop, excluding Trame transport and browser rendering.
  It is not an observed browser frame rate and must remain separate from the
  WCSPH scientific validation ladder.
- [2026-08-05] The WebGPU `3.30/5.80 ms` median/p95 measurement awaits
  `GPUQueue.onSubmittedWorkDone` in a local Edge 151 hardware session. It is a
  renderer-completion measure, not end-to-end input-to-photon or remote-network
  throughput. The render canvas was `1265x656` inside a `1280x720` browser.
- [2026-08-06] The Rich 7,020-particle local preset renders near `4.0 ms` but
  packs `599,040 B` per base64 frame, far above the Fast 96-KiB transport gate.
  It is visual-quality evidence only until binary transport or display
  decimation is implemented.
- [2026-08-06] The geospatial 256² dynamic base64 frame is 341.3 KiB. Local
  solver and WebGPU gates pass, but remote cadence needs binary transport or
  explicit display decimation; solver timing is not browser FPS.
- [2026-08-06] Official NASADEM N43E006 acquisition is blocked by Earthdata
  authentication (HTTP 401). The current visual acceptance uses the labelled
  deterministic fixture and cannot be cited as real-terrain visual evidence.
- [2026-08-06] The Terrain SPH flat/hill delta proves that the fixed hill
  affects the flow, but it is not an accuracy benchmark. Its 8.13 ms warm
  step excludes first-use generated-kernel compilation, Trame transport and
  VTK rendering.

- [2026-08-05] The old uniform-fine `dx=0.07` floating dam-break artifact has
  `941.192 kg` fluid versus `1000 kg` adaptive and is not a mass-matched
  convergence reference. Use `dx=1/14` and explicitly fix
  `body_spacing=0.1`.
- [2026-08-05] Uniform coarse-to-fine rigid angular velocity differs 86.76%
  at step 900. Adaptive angular-response comparisons cannot be interpreted as
  an APR-only error until rigid pressure/surface quadrature converges.
- [2026-08-05] The current averaged-`HIJ` beta approximation increases the
  manufactured multilevel-interface horizontal residual RMS by 12.74% and
  vertical hydrostatic residual RMS by 26.75%.
- [2026-08-05] Exact separate destination/source gradients pass independent
  kernel and conservation oracles but do not cure the interface error. In the
  padded fixture they increase horizontal RMS by 24.16% and vertical RMS by
  19.34% versus uncorrected. At matched time, floating-equilibrium lateral
  drift remains about 2.3--2.5 cm and changes sign with refinement.
