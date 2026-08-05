---
type: plan
id: 2026-08-05_warp-separate-h-variable-resolution-gradients
author: @kunalpuri-prediqt
agent: codex
created: 2026-08-05T12:00:00 CEST
status: approved
depends_on:
  - 2026-08-04_warp-adaptive-floating-body-production
adrs: [ADR-0003, ADR-0006, ADR-0009, ADR-0011]
aspects: [warp-backend, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_codegen.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: separate-h conservative variable-resolution gradients

## Goal

Replace the adaptive solver's averaged-`HIJ` grad-`h` approximation with the
variationally consistent separate-smoothing-length pair form, while leaving
the established uniform WCSPH equation source and cache signatures unchanged.
Repeat the manufactured interface, floating-equilibrium, and mass-matched
step-900 validation ladder and report every scientific gate as passed or
failed.

## Context

The physics ladder in experiment
`2026-08-05_warp-physics-validation-ladder` found that the current correction
is pair conservative but increases manufactured multilevel-interface
horizontal acceleration RMS by 12.74% and vertical hydrostatic residual RMS
by 26.75%. Bulk surge, height, and body COM pass against the corrected
mass-matched fine reference, whereas body angular velocity and integrated
fluid angular impulse differ by 95.21% and 58.72%.

PySPH's equation evaluator defines `DWI` with destination `h_i`, `DWJ` with
source `h_j`, and `DWIJ` with averaged `HIJ`. Existing PySPH equations that
need a variationally conservative variable-`h` pair use the first two
separately. Vacondio et al.'s selected APR formulation likewise requires a
variationally consistent WCSPH scheme for unequal particle masses and
smoothing lengths. The target discrete operators are:

```text
beta_i = -(1 / (d rho_i)) sum_j m_j r_ij . grad_i W_ij(h_i)

d rho_i/dt = (1 / beta_i) sum_j m_j v_ij . grad_i W_ij(h_i)

d v_i/dt = -sum_j m_j [
    p_i/(beta_i rho_i^2) grad_i W_ij(h_i)
  + p_j/(beta_j rho_j^2) grad_i W_ij(h_j)
]
```

Here `d` is spatial dimension. The fluid/rigid Liu passes must use the same
two pair terms and exact equal-and-opposite reaction. Solid `beta` remains
one unless a later formulation explicitly evolves solid smoothing length.

## Approach

### 1. Record the formulation decision

- Add an ADR selecting the separate-`h` variational pair form and explicitly
  rejecting the averaged-`HIJ` beta approximation for production APR.
- Regenerate the decision index and graph with the repository script.
- Extend the primary reference note with the exact operator mapping and the
  distinction between PySPH vector `DWI`/`DWJ` and Warp's scalar radial
  gradient factors.

### 2. Extend generated shared geometry additively

- Add separate destination/source radial gradient factors to
  `SHARED_QUANTITIES`, named to make their `h_i`/`h_j` ownership explicit.
- Emit each factor from the existing kernel router evaluated at `d_h[i]` or
  `s_h[j]`; equation snippets continue multiplying by `(dx, dy, dz)`.
- Auto-add position and smoothing arrays when either factor is requested.
- Preserve existing `hij`, `grad`, and `wij` generation byte-for-byte for
  groups that do not request the new quantities.
- Keep flat, grid, multilevel, periodic-grid, fp32, fp64, kernel selection, and
  generated-kernel cache behavior structurally correct.

### 3. Replace only the adaptive variable-`h` blocks

- Compute `beta_i` and adaptive continuity from the destination-`h` gradient.
- Compute adaptive pressure acceleration as the sum of destination-pressure /
  destination-gradient and source-pressure / source-gradient terms.
- Apply the same formulation to fluid/rigid Liu acceleration and the reverse
  body-force pass. Ensure rigid `beta_h=1` is present where required.
- Keep uniform `PressureGradient`, `ContinuityEquation`, Liu coupling,
  artificial viscosity, XSPH, EOS, and all fixed-`HIJ` paths unchanged.
- Retain the static refinement region and current split/merge/shifting
  settings; this plan changes the SPH operator, not the adaptation policy.

### 4. Add source and numerical oracles

- Generator-source tests prove that requesting separate gradients emits both
  `h_i` and `h_j` evaluations, while an existing uniform block's generated
  source and structural metadata do not change.
- Compare fp32/fp64 separate gradients with PySPH kernel `DWI`/`DWJ` values for
  unequal `h_i != h_j`, including a pair that lies inside only the larger
  support.
- Update the beta oracle to use destination `h_i` and compare flat/grid/
  multilevel results where applicable.
- Compare adaptive continuity and pressure acceleration with an independent
  NumPy/PySPH unequal-`h` oracle.
- Prove pair and whole-fixture linear-force and angular-torque conservation
  for fluid/fluid and Liu fluid/body interactions.
- Prove that equal smoothing lengths reduce to the prior symmetric result
  within fp32/fp64 tolerance.

### 5. Rerun the physics ladder

- Rerun the manufactured hydrostatic multilevel interface for uncorrected,
  old averaged-`HIJ`, and new separate-`h` forms.
- Rerun floating equilibrium at `dx=0.1` and `dx=1/12` to a common simulated
  end time and record COM/velocity/angular-velocity histories.
- Rerun corrected adaptive step 900 against the existing mass-matched
  `dx=1/14`, `body_spacing=0.1` uniform reference.
- Record pressure range, kinetic energy, fluid momentum, body COM/velocity/
  angular velocity, and integrated fluid/contact linear/angular impulses.
- Update the experiment, checkpoint review, daily/session memory, and current
  status with superseding artifacts and explicit failed gates.

## Acceptance gates

- Existing generated uniform equation source/metadata and focused uniform
  numerical tests are unchanged within their current tolerances.
- Separate gradients match independent PySPH kernel evaluations for all three
  supported kernels, fp32/fp64, and unequal smoothing lengths.
- Variable-`h` fluid/fluid and fluid/body fixtures conserve total linear force
  and total torque to the established precision-scaled tolerances.
- The manufactured interface horizontal and vertical residual RMS must not be
  worse than the uncorrected operator. Improvement over the old averaged-`HIJ`
  result alone is insufficient for production acceptance.
- Floating-equilibrium drift must decrease under refinement at a common final
  time; any sign-changing lateral bias remains a reported failed gate.
- The mass-matched long-run gates remain those of the approved production
  plan: finite state, adaptation mass drift `<=1e-5`, surge/height/COM within
  two fine spacings, and developed-flow energy plus integrated linear/angular
  impulse within 5% of uniform fine.
- No NaN/device error, rigid geometry regression, contact penetration
  regression, or loss of repeated split/merge conservation.

If the exact separate-`h` formulation passes local conservation oracles but
still fails a physical ladder gate, retain the mathematically correct
experimental path, document the failure, and do not tune coefficients to make
the benchmark pass.

## Files expected to change

Host changes are limited to the four files in frontmatter. Experiment scripts,
artifacts, ADR/reference notes, reviews, and session memory remain under
`.ai/implementations/blast-from-the-past/`.

## Tests / validation

- Focused `test_warp_codegen.py` source/cache/launch selection.
- Focused and then relevant full `test_warp_sph.py` selection on CUDA.
- `test_warp_adaptive.py` and studio suite as regressions even though their
  host files are not expected to change.
- Manufactured interface, matched-time floating equilibrium, and adaptive
  step-900 GPU runs.
- Touched-file `py_compile`, `validate-memory.py`, and `git diff --check`.

## Risks

- A pair can be accepted because only the larger smoothing-length support
  contains it. The multilevel traversal must still visit the pair, and the
  smaller-support gradient must evaluate exactly to zero without dropping the
  larger-support term.
- Two kernel derivative evaluations increase adaptive equation cost; uniform
  kernels must not pay this overhead.
- Correcting the fluid operator may expose an independent rigid-surface
  quadrature error. The force/torque decomposition will distinguish these.
- Pressure and torque are chaotic after impact. Conclusions will use matched
  histories and integrated observables, not a single peak alone.

## Out of scope

- Moving or removing the static refinement box.
- New split/merge stencils, shifting, free-surface correction, or dynamically
  adapting wall/body particles.
- Device-resident allocation/compaction and formal performance promotion.
- Changing generic PySPH equation-evaluator `DWI`/`DWJ` semantics or public
  APIs outside the Warp prototype boundary.
- Claiming production completion if hydrostatic, equilibrium, angular impulse,
  coarsening, or particle-reduction gates still fail.

## Estimated effort

One implementation/kill-test session plus GPU validation time for the two
equilibrium runs and one 900-step adaptive rerun.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-08-05T12:05:00 CEST
- Approval, verbatim quote:
  > approved
