---
type: reference-note
id: yu-turk-anisotropic-fluid-surfaces
created: 2026-08-05T14:45:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, host-integration, validation-benchmarks]
---

# Reference: Yu and Turk, Anisotropic Fluid Surfaces

## Citation

J. Yu and G. Turk, “Reconstructing Surfaces of Particle-Based Fluids Using
Anisotropic Kernels,” ACM SIGGRAPH/Eurographics Symposium on Computer
Animation, 2010, pp. 217--225.

## TL;DR

Use neighborhood covariance/PCA to orient and scale anisotropic kernels, plus
a display-only smoothing step, so reconstructed particle-fluid surfaces better
preserve smooth bodies, thin streams, and sharp features than isotropic blobs.

## Bearing on blast-from-the-past

The gameplay Warp hash grid already provides the neighborhood needed to
compute smoothed display positions and covariance eigenvectors. These values
must remain a render-only side channel and never overwrite PBF state.

## Algorithms to use

- Compute weighted neighborhood centers and covariance matrices.
- Use symmetric eigendecomposition to obtain principal axes.
- Clamp eigenvalue ratios and kernel scale for sparse/free-surface particles.
- Fall back to an isotropic kernel below a minimum neighbor count.
- Keep smoothing and anisotropy parameters visible in the result metadata.

## Questions raised

- Which bounded anisotropy ratios look continuous without erasing thin sheets?
- What minimum neighbor count gives a stable eigensystem at this low particle
  resolution?

## Verdict

Adopt a bounded display-only variant and validate finite orthonormal axes,
positive scales, and exact non-mutation of solver positions.
