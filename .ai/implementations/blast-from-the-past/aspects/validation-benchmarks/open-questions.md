# Open Questions - validation-benchmarks

- [open] What does "blazing fast" mean in concrete speedup, throughput, and hardware terms?
- [closed 2026-08-05] For the gameplay PBF column, the concrete solver gate is
  warm median `<=16.7 ms` and p95 `<=33.3 ms` at 1,000 particles and three
  projections on the RTX 5090. Final result: `1.049/1.244 ms`; snapshot
  readback is reported separately.
- [closed 2026-08-05] The new gameplay WebGPU path at a 1280x720 Edge viewport
  measured `3.30/5.80 ms` median/p95 GPU completion, `0.00/0.10 ms` upload,
  `0.182/0.224 ms` server pack, and zero dropped/stale replay frames. This is
  local presentation evidence, not network input-to-photon latency.
- [open 2026-08-05] What measured input-to-photon latency and sustained live
  cadence does the WebGPU path achieve across a non-local Trame connection?
- [open] What correctness and timing thresholds should promote elliptical drop
  from smoke workload to first published particle-dynamics benchmark?
- [closed 2026-06-17] For apples-to-apples resolved Application comparisons,
  the Warp runner should use PySPH-like adaptive timestep policy: `n_damp`
  growth and temporary output-time landing caps. The old initial-`dt` capped
  policy remains available as `--warp-timestep-policy current` for diagnostics.
- [open 2026-07-31] Which literature-reconciled split/merge operator,
  transition correction, and convergence case should gate promotion of the
  two-level adaptive dam-break beyond engineering-smoke status?
- [closed 2026-08-05] The direct variational destination/source-gradient pair
  form is implemented and locally conservative, but it fails the hydrostatic
  screen. Separate `h` alone is not the missing complete consistency treatment.
- [open 2026-08-05] Which interface consistency, particle-volume, free-surface,
  or density-reinitialization treatment removes the mixed-resolution pressure
  residual without sacrificing exact fluid/rigid force and torque conservation?
- [open 2026-08-05] What rigid surface quadrature/body sampling is required
  for floating-equilibrium and angular-impulse convergence independent of the
  fluid lattice orientation?
- [open 2026-08-06] Which canonical SPH flow-over-topography case and error
  norms should replace final-particle displacement as the Terrain SPH
  promotion gate?
