# Open Questions - validation-benchmarks

- [open] What does "blazing fast" mean in concrete speedup, throughput, and hardware terms?
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
