---
type: decision
id: ADR-0011
date: 2026-08-05
author: @kunalpuri-prediqt
scope: warp-backend
status: Proposed
supersedes: []
relates_to: [ADR-0003, ADR-0006, ADR-0009]
depends_on: [ADR-0003, ADR-0006]
conflicts_with: []
---

# ADR-0011: Separate-h gradients for variable-resolution WCSPH

## Context

ADR-0009 introduced an adaptive-only grad-`h` partition factor but evaluated
all pair terms with the Warp backend's pre-existing averaged-`HIJ` gradient.
That approximation is pair conservative and improved some long dam-break
linear observables, but a manufactured multilevel hydrostatic fixture found
that it increased horizontal acceleration residual RMS by 12.74% and vertical
hydrostatic residual RMS by 26.75% relative to the uncorrected operator.

PySPH's equation evaluator distinguishes `DWI`, evaluated with destination
`h_i`, `DWJ`, evaluated with source `h_j`, and `DWIJ`, evaluated with their
average. Existing variational unequal-`h` PySPH equations use `DWI` and `DWJ`
as separate terms. The generated Warp equations currently expose only the
scalar radial factor corresponding to `DWIJ`.

## Proposed decision

- Add generator-owned destination/source radial gradient factors evaluated at
  `d_h[i]` and `s_h[j]`. Keep the existing averaged `grad` quantity unchanged.
- Compute the adaptive partition factor and continuity equation with the
  destination-`h` gradient.
- Compute conservative pressure acceleration with the destination pressure
  coefficient multiplying the destination-`h` gradient and the source pressure
  coefficient multiplying the source-`h` gradient.
- Apply the identical two-term pair force to Liu fluid/rigid coupling and its
  exact reverse reaction, with solid `beta_h=1`.
- Request the new geometry only from adaptive variable-`h` equation blocks.
  Uniform WCSPH, artificial viscosity, XSPH, and other averaged-`HIJ` equations
  retain their generated source and runtime cost.

## Rationale

This is the direct mapping of PySPH's `DWI`/`DWJ` semantics into the Warp
generator and restores the smoothing-length ownership required by the
variational pair form. Pair antisymmetry is preserved because reversing a pair
swaps both pressure coefficients and gradient ownership while reversing the
separation vector. A source outside one particle's kernel support contributes
zero through that gradient but may still contribute through the other,
provided the exact max-support neighbor traversal retains the pair.

## Alternatives considered

- **Keep averaged `HIJ`.** Rejected as the production candidate because it
  fails the manufactured hydrostatic-interface screen.
- **Average the two separate gradients before applying both pressure terms.**
  Rejected because it discards coefficient/gradient ownership and is not the
  target variational form.
- **Change every generated equation to separate gradients.** Rejected because
  viscosity, XSPH, and established uniform equations intentionally use their
  current `HIJ` convention and should not pay additional derivative cost.
- **Tune beta or pressure coefficients empirically.** Rejected because it does
  not repair the mathematical inconsistency and would overfit one benchmark.

## Consequences

- Adaptive pressure interactions evaluate two kernel derivatives instead of
  one; uniform paths retain one.
- Generated-kernel structural signatures change only for blocks requesting the
  new quantities.
- Unequal-support pairs become a mandatory oracle for flat, grid, and
  multilevel traversal.
- Correct local conservation does not guarantee rigid torque convergence;
  surface quadrature remains independently measurable.

## Acceptance gate

Keep this ADR Proposed until separate gradients match PySPH kernel oracles,
fluid/fluid and fluid/body total force and torque conserve, equal `h` reduces
to the old symmetric result, and the manufactured interface is no worse than
the uncorrected operator. Long dam-break and equilibrium misses remain explicit
prototype limitations rather than reasons to alter the formula.

## Validation outcome

The implementation passes the independent kernel, equal-`h`, unequal-support,
fluid/fluid conservation, and fluid/body force/torque oracles. The padded
manufactured interface nevertheless rejects it as a production formulation:

| Residual RMS | Uncorrected | Averaged `HIJ` | Separate `h` |
|---|---:|---:|---:|
| Horizontal acceleration | 5.370774 | 5.934755 | 6.668305 |
| Vertical hydrostatic | 0.405565 | 0.423651 | 0.484013 |

Separate `h` is 24.16% and 19.34% worse than uncorrected, respectively. A
matched-time floating-equilibrium refinement also retains a sign-changing
lateral drift. ADR-0011 therefore remains **Proposed**: the operator is kept
as a mathematically faithful experimental path, but it is not accepted as the
complete variable-resolution consistency treatment and its coefficients will
not be tuned to these fixtures.

## Follow-ups

- Implement and validate plan
  `2026-08-05_warp-separate-h-variable-resolution-gradients`.
- If the local operator passes but floating equilibrium does not converge,
  isolate rigid surface quadrature/body sampling in a separate plan.
