---
type: reference-note
id: macklin-muller-position-based-fluids
created: 2026-08-05T14:00:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, validation-benchmarks]
---

# Reference: Macklin and Müller, Position Based Fluids

## Citation

M. Macklin and M. Müller, “Position Based Fluids,” ACM Transactions on
Graphics 32(4), 2013, DOI `10.1145/2461912.2461984`.

Official implementation context: NVIDIA FleX 1.1 documentation,
`https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/flex/manual.html`.

## Bearing on blast-from-the-past

The method enforces an incompressibility-like density constraint by directly
projecting particle positions. A small fixed number of solver iterations gives
stable visual fluid motion at timesteps much larger than the acoustic timestep
used by WCSPH. NVIDIA FleX documents the same broad position-based architecture
for real-time GPU particle effects.

This is authority for an explicitly approximate Gameplay column, not a reason
to alter or validate the scientific WCSPH implementation against game metrics.

## MVP mapping

- Predict positions from velocity and gravity.
- Build a GPU uniform-grid neighbor list over predicted positions.
- Compute density constraint and per-particle Lagrange multiplier.
- Apply symmetric position corrections with bounded artificial pressure.
- Project analytic tank and rigid-box collision constraints.
- Update velocity from corrected displacement and apply optional smoothing.
- Use a fixed iteration/frame budget and report solver timing independently
  from rendering/readback.

## Open limitations

- The paper does not define this prototype's fluid-to-rigid impulse clamp or
  analytic buoyancy approximation; those remain game controls and must be
  labeled accordingly.
- A one-rebuild-per-substep implementation needs an explicit correction-bound
  guard because projected particles move after neighbor construction.
