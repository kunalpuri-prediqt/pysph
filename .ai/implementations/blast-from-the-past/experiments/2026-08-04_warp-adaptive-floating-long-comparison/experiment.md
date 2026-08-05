---
type: experiment
id: 2026-08-04_warp-adaptive-floating-long-comparison
created: 2026-08-04T18:00:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: completed-with-failed-gates
last_checked: 2026-08-05T10:25:00 CEST
plan: 2026-08-04_warp-adaptive-floating-body-production
---

# Experiment: long adaptive floating-body comparison

## Purpose

Run the composed adaptive-fluid/floating-body/contact case far enough to
exercise impact and compare its principal observables with a matched
quaternion-based uniform-fine reference. Report failed gates explicitly.

## Correctness defects found before comparison

Two defects invalidated earlier long artifacts:

1. Rebuilding the multilevel NNPS pushed stale host body coordinates and
   rewound the moving body at every adaptation checkpoint. The runner now
   synchronizes the body host mirror before each rebuild and asserts rigid
   coordinate/mass/identity preservation.
2. Incremental particle rotation accumulated geometry drift above `1e-5`.
   ADR-0010 replaces it with quaternion pose integration and reconstruction
   from immutable body-frame coordinates.

Artifacts whose ancestry predates either fix are not scientific evidence:
`pysph-phase5-adaptive-floating-impact-250`,
`pysph-phase5-adaptive-floating-contact-600`,
`pysph-restartable-adaptive-floating-step600`,
`pysph-rigid-rebuild-fix-step700`, and
`pysph-phase5-first-contact-step900` under `/tmp`.

## Fresh corrected results at step 900

Both cases use the quaternion rigid integrator, the same body dimensions,
density and `body_spacing=0.1`, the same fluid/contact model, and simulated
time approximately `0.357140 s`. The adaptive case starts at coarse `dx=0.1`;
the reference is uniform `dx=0.07`. Their discretized initial fluid masses are
not identical (`1000.0` versus `941.192`), so this is a demanding convergence
screen rather than a claim of normalized continuum parity.

| Observable | Adaptive | Uniform fine | Gate/result |
|---|---:|---:|---|
| Active fluid particles | 3,376 | 2,744 | fail; adaptive has 23% more |
| Surge front | 2.80641 m | 2.64261 m | fail; error 0.16380 m > 0.14 m |
| Maximum height | 0.963785 m | 0.946718 m | pass; error 0.01707 m |
| Body COM distance | - | - | pass; difference 0.10573 m < 0.14 m |
| Body velocity difference | - | - | fail; 8.45% of reference speed |
| Body angular-velocity difference | - | - | fail; 147% of reference norm |
| Contact-impulse difference | - | - | fail; 22.6% of reference norm |
| Relative fluid mass drift | `2.95e-9` | 0 | pass |
| Rigid geometry drift | `2.27694e-7` | `2.24395e-7` | pass |
| Max geometric penetration | 0 | 0 | pass |
| Rigid device error / finite | 0 / true | 0 / true | pass |
| Flow-driven merged families | 0 | n/a | fail |

Adaptive artifact:
`/tmp/pysph-phase5-quaternion-adaptive-step900.npz` and JSON manifest.
Uniform-fine artifact:
`/tmp/pysph-phase5-quaternion-uniform-fine-dx070-step900.npz` and JSON
manifest.

The adaptive run was split at step 600 and resumed successfully with its saved
quaternion/reference state. Its two reported runner intervals total about
`136.60 s`; the freshly compiled uniform-fine runner reported `163.22 s`.
These are not the required three-repeat warm GPU timing measurements and do
not establish the 2x performance gate.

## Conclusion

The composition, contact, restart, finite-state, mass, penetration, COM,
height, and rigid-geometry gates work. The proposed APR path fails the surge,
body-response/impulse, flow-driven merge, and 4x particle-reduction gates.
Kinetic-energy, interface-error, full CPU-Application parity, device mutation,
and formal warm performance gates remain unverified. ADR-0009 therefore stays
Proposed and the implementation must not be promoted as production APR.

## 2026-08-05 variable-resolution consistency rerun

The adaptive-only equation path now computes a conservative grad-`h`
partition factor before each acceleration evaluation. Because the Warp
backend currently exposes its established averaged-`HIJ` gradient rather than
separate destination/source gradients, the implementation consistently uses
that averaged kernel in the beta pre-pass, continuity equation, symmetric
pressure force, and Liu fluid/body reaction. The unchanged uniform path does
not enable the correction.

The tuned transfer settings retain the static refinement box and icosa13
stencil while setting merge hysteresis and shifting iterations to zero. This
avoids host `O(N^2)` shifting and gave the closest screened response without
changing refinement placement. The corrected 900-step artifact is
`/tmp/pysph-varh-tune-h0-shift0-900.npz` with its JSON manifest.

| Observable | Corrected adaptive | Earlier adaptive | Uniform fine | Result |
|---|---:|---:|---:|---|
| Active fluid particles | 3,376 | 3,376 | 2,744 | fail |
| Surge front | 2.79934 m | 2.80641 m | 2.64261 m | improved; error 0.15672 m, still fail |
| Maximum height | 0.963760 m | 0.963785 m | 0.946718 m | pass |
| Body COM distance from reference | 0.10389 m | 0.10573 m | 0 | pass |
| Body velocity difference | 6.95% | 8.45% | 0 | improved; pass 8% screen |
| Body angular-velocity difference | 151% | 147% | 0 | fail |
| Contact-impulse difference | 22.2% | 22.6% | 0 | fail |
| Relative fluid mass drift | `2.95e-9` | `2.95e-9` | 0 | pass |
| Rigid geometry drift | `2.28e-7` | `2.28e-7` | `2.24e-7` | pass |
| Flow-driven merged families | 0 | 0 | n/a | fail at endpoint |

The corrected adaptive endpoint has beta range `0.2061..1.1139`, fluid
kinetic energy `423.1176`, and fluid momentum
`[475.9701, -1.0230, -276.2776]`.

A continuation through step 1100 reached surge `x=3.07014` and remained
finite with mass drift `3.38e-9`, but still recorded no merge. Inspection
showed 47 families with at least one daughter beyond `x=2.8`, while no complete
13-daughter family had cleared it; the largest family minimum was `x=2.76219`.
Thus the production flow has not yet satisfied the complete-family merge gate.
The controller's explicit translating-boundary fixture does exercise four
split/merge/return cycles and preserves mass, momentum, constant density, and
linear hydrostatic pressure.

Focused validation: adaptive controller suite `16 passed`; coupled dam-break,
grid/multilevel rigid, rebuild, and variable-`h` oracle suite `5 passed`. The
new oracle independently checks beta values and net pressure force.

## 2026-08-05 mass-matched correction and physics ladder

The `dx=0.07` reference above is superseded for scientific comparison. It
contains only `941.192 kg` of fluid, while the adaptive case contains
`1000 kg`. The mass-matched reference uses `dx=1/14` and explicitly fixes
`body_spacing=0.1`; otherwise changing fluid resolution also changes rigid
sampling and the contact timestep.

| Observable | Corrected adaptive | Mass-matched uniform fine | Result |
|---|---:|---:|---|
| Active fluid particles | 3,376 | 2,744 | fail particle-reduction gate |
| Surge front | 2.799340 m | 2.701674 m | pass; error 0.097667 m |
| Maximum height | 0.963760 m | 0.965888 m | pass; error 0.002128 m |
| Fluid kinetic energy | 423.118 J | 449.843 J | 5.94%; narrowly fails 5% screen |
| Fluid momentum difference | - | - | 13.24% |
| Body COM distance | - | - | pass; 0.04582 m |
| Body velocity difference | - | - | 9.18% |
| Body angular-velocity difference | - | - | fail; 95.21% |
| Fluid impulse difference | - | - | 8.61% |
| Fluid angular-impulse difference | - | - | fail; 58.72% |
| Contact-impulse difference | - | - | 8.52% |
| Sampled contact angular-impulse difference | - | - | fail; 88.20% |

Artifacts are
`/tmp/pysph-physics-adaptive-varh-static-history-step900.{npz,json}` and
`/tmp/pysph-physics-uniform-massmatched-history-step900.{npz,json}`.

The pressure range is `-22.10..35.43 kPa` adaptive versus
`-7.80..23.84 kPa` uniform fine. Uniform `dx=0.1` to `dx=1/14` angular
velocity also differs by 86.76%, showing that rigid torque is not converged
even without adaptation. A manufactured hydrostatic interface test finds that
the averaged-`HIJ` beta approximation increases horizontal residual RMS by
12.74% and vertical hydrostatic residual RMS by 26.75%. A nominal floating
equilibrium likewise drifts, although vertical/rotational drift improves when
fluid spacing is reduced from `0.1` to `1/12`.

The fair reference therefore clears the earlier surge conclusion but exposes
pressure-induced angular impulse as the dominant physics gap. See experiment
`2026-08-05_warp-physics-validation-ladder` for the full decomposition and
artifacts. The next correction requires separate destination/source kernel
gradients rather than further tuning of the averaged-`HIJ` approximation.
