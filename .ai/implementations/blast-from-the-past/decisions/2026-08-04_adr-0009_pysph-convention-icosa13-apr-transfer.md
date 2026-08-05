---
type: decision
id: ADR-0009
date: 2026-08-04
author: @kunalpuri-prediqt
scope: particle-memory
status: Proposed
supersedes: []
relates_to: [ADR-0005, ADR-0007, ADR-0008]
depends_on: [ADR-0007]
conflicts_with: []
---

# ADR-0009: PySPH-convention icosa13 APR transfer

## Context

The original adaptive dam-break smoke used eight equal-mass octant daughters,
piecewise-constant property copying, one shared split/merge boundary, and no
particle regularization. It conserved mass and momentum but was explicitly not
a production APR transfer.

Vacondio et al.'s selected 3D layout is a 12-vertex icosahedral shell plus one
central daughter at offset `0.65 h_parent` and daughter smoothing length
`0.70 h_parent`. A converged numerical reproduction using PySPH/Warp's actual
3D Wendland C2 convention confirms that layout beats cubic-plus-center and
equal masses, but does not reproduce the paper table's exact error or mass
ratio. Treating the published table as an exact oracle would therefore encode
an unresolved convention mismatch.

## Proposed decision

For the PySPH/Warp implementation, use the converged convention-specific
constrained optimum:

- 12 shell daughters of mass fraction `0.0739476670674313`;
- one central daughter of mass fraction `0.1126279951908244`;
- shell offset `0.65 h_parent` and daughter `h = 0.70 h_parent`;
- local least-squares first-order reconstruction for continuous scalar fields;
- mass-weighted complete-family merge of exactly 13 daughters;
- separate split/merge bounds with configurable hysteresis;
- bounded repulsive shifting of fine particles, suppressed for low external
  neighbor count and prevented from shifting into the moving rigid surface;
- first-order correction after shifting and explicit linear-momentum removal;
- an adaptive-only grad-`h` partition factor applied consistently to
  continuity, symmetric pressure force, and fluid/rigid pressure reaction,
  using the Warp backend's averaged-`HIJ` kernel convention.

The octant-eight operator remains available only for manifest/test
compatibility. New adaptive dam-break configurations default to `icosa13`.
Long-run screening leaves bounded shifting available but defaults it off and
uses zero merge hysteresis; this avoids host quadratic work and does not delay
a complete family's eligibility to merge.

## Rationale

This makes the convention mismatch explicit and testable. The chosen masses
minimize density reconstruction error for the kernel implementation actually
executed by PySPH/Warp, while symmetry makes the daughter centroid exact.
First-order reconstruction eliminates the piecewise-constant linear-field
error, complete-family merge is conservative, hysteresis prevents immediate
boundary thrashing, and shifting has free-surface/rigid limiters.

## Alternatives considered

- **Keep octant-eight.** Retained as a smoke fixture, rejected as the new
  default because its converged density error is higher and it has no central
  daughter.
- **Copy the paper's unverified mass ratio.** Rejected until the kernel/stencil
  convention difference is identified.
- **Equal masses.** Rejected by the density reconstruction kill test.

## Current evidence and acceptance gate

NumPy oracle tests pass for exact mass and linear-momentum conservation,
constant-field preservation, linear-field reproduction, 13-member complete
merge, hysteresis, bounded shifting, and post-shift correction. A 20-step
adaptive floating-body GPU smoke stays finite with 910 coarse + 1,170 fine
particles, zero rigid error, relative mass drift `1.3411e-9`, and maximum shift
`5.4342e-5 m`.

The repeated translating-boundary fixture now passes four complete
split/merge/return cycles while preserving mass, momentum, constant density,
and linear hydrostatic pressure. A generated-kernel oracle validates the
averaged-`HIJ` beta factor and pairwise pressure-force conservation. A finite
900-step adaptive/floating rerun improves the body-velocity difference from
8.45% to 6.95%, but still fails surge, angular-response, contact-impulse,
particle-reduction, and production-flow merge gates. The run's front ends
just inside the refinement boundary; a step-1100 continuation has partial but
no complete families beyond it.

ADR-0011 subsequently implemented the exact separate destination/source
gradient form. It passes local kernel and conservation oracles but performs
worse than the uncorrected operator in the padded hydrostatic-interface
fixture and still fails matched-time equilibrium and long-run energy/angular
impulse gates. The earlier averaged-`HIJ` factor and the exact separate-`h`
path are therefore both experimental evidence, not accepted production APR.

This ADR therefore remains **Proposed**. The transfer and consistency
operators have local evidence, but the whole production APR method does not
yet have the required convergence, flow-driven coarsening, device-resident
mutation, and performance evidence.
