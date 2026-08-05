---
type: experiment
id: 2026-08-05_warp-physics-validation-ladder
created: 2026-08-05T11:00:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: completed-with-failed-gates
last_checked: 2026-08-05T14:35:00 CEST
plan: 2026-08-04_warp-adaptive-floating-body-production
---

# Experiment: adaptive floating-body physics validation ladder

## Purpose

Replace the earlier non-mass-matched convergence screen with a fair uniform
reference, decompose the fluid/contact force and torque acting on the floating
body, and test whether the current averaged-`HIJ` variable-resolution
correction improves hydrostatic interface balance.

## Method

- Use `dx=1/14 m` for the uniform-fine reference. It tiles the
  `2 x 0.5 x 1 m` fluid block with 2,744 particles and gives the same
  discretized `1000 kg` mass as adaptive `dx=0.1`.
- Fix `body_spacing=0.1 m` in every comparison. Deriving it from fluid `dx`
  changes both rigid sampling and the contact timestep, so it is not a valid
  resolution comparison.
- Run uniform coarse, uniform fine, corrected adaptive, and uncorrected
  adaptive cases through step 900 at approximately `t=0.357 s`.
- Sample the rigid force and torque every ten steps and trapezoid-integrate
  fluid and contact linear/angular impulses.
- Use a manufactured linear-hydrostatic multilevel field to compare the
  uncorrected averaged-`HIJ` gradient with its beta-corrected form.
- Release a half-submerged density-`500 kg/m^3` body in a nominally
  hydrostatic tank at two fluid resolutions, with wall contact disabled.

Scripts:

- `run_interface_balance.py`
- `run_float_equilibrium.py`

Primary artifacts:

- `/tmp/pysph-physics-uniform-massmatched-history-step900.{npz,json}`
- `/tmp/pysph-physics-uniform-coarse-dx010-bodydx010-step900.{npz,json}`
- `/tmp/pysph-physics-adaptive-varh-static-history-step900.{npz,json}`
- `/tmp/pysph-physics-adaptive-uncorrected-history-step900.{npz,json}`
- `/tmp/pysph-physics-interface-balance.json`
- `/tmp/pysph-physics-float-equilibrium-dx010.json`
- `/tmp/pysph-physics-float-equilibrium-dx1over12.json`
- `/tmp/pysph-physics-interface-balance-separate-h-padded.json`
- `/tmp/pysph-physics-float-equilibrium-dx010-t035.json`
- `/tmp/pysph-physics-float-equilibrium-dx1over12-t035.json`
- `/tmp/pysph-physics-adaptive-separate-h-history-step900.{npz,json}`

## Mass-matched dam-break result

| Observable | Corrected adaptive | Uniform fine | Difference/result |
|---|---:|---:|---|
| Fluid mass | 1000.000003 kg | 1000.000037 kg | matched |
| Active fluid particles | 3,376 | 2,744 | fail particle-reduction gate |
| Surge front | 2.799340 m | 2.701674 m | 0.097667 m; pass two-fine-spacing gate |
| Maximum height | 0.963760 m | 0.965888 m | 0.002128 m; pass |
| Fluid kinetic energy | 423.118 J | 449.843 J | 5.94%; narrowly fails 5% screen |
| Fluid momentum | - | - | 13.24% relative difference |
| Body COM | - | - | 0.04582 m; pass spacing gate |
| Body velocity | - | - | 9.18% relative difference |
| Body angular velocity | - | - | 95.21% relative difference; fail |
| Integrated fluid impulse | - | - | 8.61% relative difference |
| Integrated fluid angular impulse | - | - | 58.72% relative difference; fail |
| Contact impulse | - | - | 8.52% relative difference |
| Sampled contact angular impulse | - | - | 88.20% relative difference; fail |
| Pressure range | -22.10..35.43 kPa | -7.80..23.84 kPa | excessive adaptive pressure tail |

The corrected adaptive run remains finite with relative mass drift
`2.95e-9`. It has no flow-driven complete-family merge at the endpoint.

Uniform coarse-to-fine angular velocity is itself not converged: its relative
difference is 86.76%, while body velocity differs by 10.44%. The large
adaptive angular discrepancy therefore cannot be attributed to split/merge
alone. It is a pressure/rigid-surface quadrature sensitivity exposed most
strongly by torque.

## Corrected versus uncorrected adaptive A/B

The beta-corrected run improves fluid momentum, body velocity, fluid linear
impulse, and sampled contact angular impulse slightly. The uncorrected run
improves final angular velocity, integrated fluid angular impulse, kinetic
energy, and the negative pressure tail slightly. Neither is a decisive
scientific winner.

The manufactured interface test is more direct. At the mixed-resolution
interface, applying the current averaged-`HIJ` beta correction changes:

| Residual | Uncorrected | Corrected | Corrected / uncorrected |
|---|---:|---:|---:|
| Horizontal acceleration RMS | 5.37077 | 6.05494 | 1.1274 |
| Vertical hydrostatic residual RMS | 0.405565 | 0.514072 | 1.2675 |

The current approximation therefore worsens this hydrostatic interface
fixture. The conservative Liu reaction oracle still gives zero net force and
zero net torque for an asymmetric variable-`h` fluid/body fixture, so the
reaction sign and moment arm are not the defect.

## Floating-equilibrium screen

| Fluid spacing | Simulated time | COM displacement `(x,y,z)` m | Final angular velocity rad/s |
|---|---:|---|---|
| 0.100000 m | 0.4441 s | `(0.00368, 0.03012, 0.04842)` | `(-0.4273, 0.0022, 0.0059)` |
| 0.083333 m | 0.3705 s | `(-0.00030, -0.02604, 0.02197)` | `(0.0688, -0.0136, 0.0008)` |

Vertical and rotational drift improve with refinement, but lateral drift
remains about 2.6 cm and changes sign. The nominal equilibrium is therefore
not resolution-independent; particle/surface sampling symmetry remains a
material error source. The unequal final times make this a diagnostic screen,
not a formal convergence order.

## Conclusion

The fair mass-matched comparison clears the surge, height, COM, finite-state,
mass, and rigid-geometry screens. The principal unresolved physics is angular
momentum transfer from a noisy pressure field to a sparsely sampled rigid
surface. The current averaged-`HIJ` beta approximation is conservative but is
not hydrostatically accepted and should not be tuned further as though it were
the final variable-resolution formulation.

The next mathematical correction is to expose separate destination and source
kernel gradients and implement the full variable-resolution pair form, then
repeat the interface, equilibrium, and step-900 ladder. That change reaches
the Warp code generator, which is outside the approved plan's host-file scope;
no speculative production correction was made here.

Final focused validation: three Warp beta/Liu/rebuild tests passed, the full
adaptive controller suite passed 16 tests, the studio suite passed 19 tests,
all touched Python files compiled, `validate-memory: PASS`, and
`git diff --check` was clean.

## Separate-h formulation rerun

The approved follow-up plan implemented exact destination/source kernel
gradient ownership in generated Warp equations. The local mathematical gates
pass: all three kernels in fp32/fp64 match independent evaluations for unequal
supports; equal `h` reduces to the symmetric form; and asymmetric 3D
fluid/fluid and fluid/body fixtures conserve total force and torque.

The padded manufactured fixture removes truncated-support contamination and
compares all three formulations on the same field:

| Residual RMS | Uncorrected | Averaged `HIJ` | Separate `h` | Separate / uncorrected |
|---|---:|---:|---:|---:|
| Horizontal acceleration | 5.370774 | 5.934755 | 6.668305 | 1.2416 |
| Vertical hydrostatic | 0.405565 | 0.423651 | 0.484013 | 1.1934 |

The exact pair form therefore fails the primary hydrostatic gate and is also
worse than the averaged approximation in this fixture.

Matched-time (`t=0.35 s`) floating equilibrium improves vertical displacement
from `0.04490 m` at `dx=0.1` to `0.02182 m` at `dx=1/12`, and the final angular
speed falls from `0.26965` to `0.01664 rad/s`. Lateral displacement changes
from `+0.02233 m` to `-0.02466 m`, however, so the resolution-independent
equilibrium gate still fails.

At step 900, separate `h` remains finite with `2.97e-9` relative mass drift,
zero rigid error, no penetration, and `3.47e-7` geometry drift. Against the
mass-matched uniform-fine reference:

| Observable | Separate `h` result | Gate |
|---|---:|---|
| Active fluid particles | 3,388 versus 2,744 | fail |
| Surge error | 0.07604 m | pass |
| Maximum-height error | 0.00148 m | pass |
| Body-COM distance | 0.03927 m | pass |
| Fluid kinetic-energy difference | 9.79% | fail |
| Fluid linear-momentum difference | 14.17% | fail 5% |
| Body angular-velocity difference | 82.71% | fail |
| Integrated fluid impulse difference | 7.41% | fail 5% |
| Integrated fluid angular-impulse difference | 63.15% | fail |
| Sampled contact impulse difference | 4.40% | pass 5% |
| Contact impulse difference | 6.42% | fail 5% |
| Pressure range | -25.68..25.58 kPa | excessive negative tail |

There is still no complete flow-driven merge. Relative to averaged `HIJ`, the
new form improves surge, height, COM, body velocity/angular velocity, fluid
linear impulse, and contact impulse, but worsens kinetic energy, fluid angular
impulse, and the negative pressure tail. It is thus locally correct without
being physically accepted as the complete APR formulation. ADR-0011 remains
Proposed and no empirical coefficient tuning was performed.

Follow-up validation: the dedicated new-oracle selection passed 11 tests, the
full generator suite passed 17, the isolated multilevel regression passed,
the adaptive controller suite passed 16, and the studio suite passed 19. One
combined Warp invocation was interrupted after 25 minutes while compiling;
its two outstanding cases passed in isolated selections.
