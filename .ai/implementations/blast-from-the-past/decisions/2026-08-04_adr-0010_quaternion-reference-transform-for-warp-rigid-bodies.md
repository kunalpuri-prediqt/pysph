---
type: decision
id: ADR-0010
date: 2026-08-04
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0006, ADR-0008]
depends_on: [ADR-0006]
conflicts_with: []
---

# ADR-0010: quaternion reference transform for Warp rigid bodies

## Context

ADR-0006 advanced each rigid particle incrementally from its instantaneous
position and angular velocity. That is adequate for short transients, but a
corrected 900-step adaptive dam-break measured relative-geometry drift of
`1.28e-5`, outside the production plan's `1e-5` gate. The same body is also
rebuilt into the NNPS after fluid adaptation, so its pose must be persistent
and restartable independently of host particle mirrors.

## Decision

Store one normalized f64 quaternion per body and immutable f64 body-frame
reference coordinates per rigid particle. Integrate the quaternion from the
device angular velocity at each rigid stage, then reconstruct position and
velocity from the exact rigid transform:

- `x_i = cm + R(q) r_i`;
- `v_i = vc + omega cross (R(q) r_i)`.

Persist current/saved quaternions and reference coordinates in restartable
checkpoints. A legacy checkpoint without those fields uses its current pose as
the new reference pose. Before rebuilding an NNPS at an adaptation checkpoint,
pull the moving body's device coordinates to its host mirror and assert that
the rebuild changes neither body coordinates, mass, nor identity.

## Rationale

Reconstruction from a shared pose makes rigidity an invariant rather than an
accumulated numerical approximation. A quaternion is compact, has a simple
device update, composes naturally with ADR-0006's persistent state, and avoids
the drift of per-particle incremental rotation. Synchronizing the body before
the existing push-oriented NNPS constructor prevents adaptation from rewinding
the body to stale host coordinates.

## Consequences

- Rigid checkpoints now contain quaternion and body-frame reference fields.
- Restart trajectories retain fp32-level NNPS rebuild differences, but no
  longer lose the rigid pose.
- The rigid-body state has more persistent device arrays; the public PySPH
  ParticleArray API and non-Warp paths are unchanged.
- Arbitrary deformable-body motion remains out of scope.

## Evidence

- A 500-stage high-angular-velocity test preserves relative geometry below
  `1e-6` and exercises save/restore state.
- The fresh 900-step adaptive floating-body run has geometry drift
  `2.27694e-7`; the matched uniform-fine run has `2.24395e-7`.
- A focused regression rebuilds the adaptive NNPS after moving the body and
  verifies that the device coordinates are not rewound.
