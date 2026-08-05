# Known Issues - validation-benchmarks

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
