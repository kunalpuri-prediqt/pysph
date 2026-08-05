---
type: decision
id: ADR-0008
date: 2026-08-04
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0005, ADR-0006, ADR-0007]
depends_on: [ADR-0006]
conflicts_with: []
---

# ADR-0008: axis-aligned plane contact for Warp rigid bodies

## Context

The adaptive floating-body dam-break composes ADR-0006's deterministic Liu
fluid/rigid coupling with ADR-0007's multilevel traversal. The rigid shell must
also collide with the fixed tank. PySPH's existing `RigidBodyWallCollision`
uses a linear normal spring, restitution-derived dashpot, tangential damping,
and Coulomb limiting against wall normals. The dam-break tank is an
axis-aligned box, so evaluating its six inner planes directly avoids duplicate
forces from multilayer wall particles and does not couple contact correctness
to SPH neighbor sampling.

Kill tests require zero force away from a wall, an opposing normal/friction
force during glancing impact, simultaneous corner contact, bounded
penetration, and no mechanical-energy gain in a damped bounce.

## Decision

Adopt an additive per-rigid-particle contact kernel against six axis-aligned
inner planes:

- activate contact when the particle-center gap is less than its configured
  contact radius;
- use `F_n = max(0, k_n overlap - gamma_n v_n)`, with damping ratio derived
  from the configured coefficient of restitution;
- use tangential viscous damping capped by `mu * F_n`;
- sum simultaneous plane reactions locally per particle, then feed the result
  into ADR-0006's existing deterministic f64 body force/torque reduction;
- report contact force and geometric penetration separately (the contact-radius
  overlap is not itself reported as wall penetration);
- restrict the explicit step to `safety * sqrt(m_min / k_n)` whenever contact
  is enabled.

The initial dam-break defaults are `k_n=5e4`, restitution `0.3`, friction
`0.2`, safety `0.2`, and contact radius `dx/2`. These are engineering defaults,
not a calibrated material model.

## Rationale

Per-particle plane evaluation has no source atomics, is deterministic at the
contact-force stage, handles corners in one launch, and represents the actual
tank boundary independently of wall-particle resolution. The chosen law is in
the same model family as PySPH's existing wall collision while correcting two
prototype weaknesses for this case: damping uses each contact particle's mass,
and the timestep is explicitly restricted by contact stiffness.

## Alternatives considered

- **Contact against wall particles.** Rejected for this tank because multiple
  wall layers and corners would require contact ownership/deduplication and
  make force depend on wall sampling.
- **Penalty without damping or friction.** Rejected by the bounce and glancing
  impact requirements.
- **Impulse/complementarity solver.** Deferred; it is substantially more
  machinery than needed for the single convex box and six static planes.

## Consequences

- The contact path is specific to axis-aligned bounds. General moving or curved
  collision geometry needs a later broad/narrow-phase design.
- Contact diagnostics are persisted and surfaced by the studio.
- The force kernel supports fp32 and fp64 particle arrays; compact rigid
  moments remain f64.
- Passing kill fixtures cover no-contact, drop/bounce, glancing/slide, and
  corner multi-contact. The long dam-break still has to establish its measured
  penetration and settling behavior.

## Evidence

Experiment `2026-08-04_warp-rigid-plane-contact-kill-tests` records the focused
GPU fixtures. The collision-free 20-step adaptive floating-body run is
numerically unchanged with contact enabled and reports zero contact force and
zero penetration.
