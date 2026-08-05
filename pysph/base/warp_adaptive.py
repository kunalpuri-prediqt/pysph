"""Two-level adaptive-resolution smoke support for the Warp WCSPH path.

This module deliberately separates two concerns:

``TwoLevelAdaptiveController``
    A deterministic NumPy oracle for conservative split/merge, first-order
    reconstruction, hysteresis, and limited particle shifting. It retains the
    original eight-child smoke stencil for compatibility and provides the
    PySPH/Warp-convention 13-daughter icosahedral production candidate.

``WarpDamBreakSimulation``
    An incremental 3D dam-break runner used by the browser application and the
    adaptive experiment.  Physics steps run on NVIDIA Warp.  In adaptive mode
    the runner rebuilds the fluid ``ParticleArray`` and
    ``MultilevelGridWarpNNPS`` only at explicit adaptation checkpoints.

The exact published Vacondio mass ratio remains unreproduced under PySPH's
Wendland convention. The 13-daughter candidate therefore uses the converged
convention-specific constrained optimum recorded by the implementation memory,
not an unverified transcription of the paper's table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import setuptools  # noqa: F401

# Python 3.14 removed stdlib distutils while Compyle 0.9.1 still imports it.
# Importing pip-provided setuptools first activates its compatibility module.

from pysph.base.utils import DEFAULT_PROPS, get_particle_array
from pysph.base.warp_multilevel_nnps import MultilevelGridWarpNNPS
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import (
    create_rigid_body_state,
    wc_sph_dam_break_rigid_step,
    wc_sph_dam_break_step,
)
from pysph.examples._db_geometry import DamBreak3DGeometry


H = 1.0
GRAVITY = 9.81
REF_C0 = 10.0 * math.sqrt(2.0 * GRAVITY * 0.55)

_OCTANTS = np.asarray(
    [
        (sx, sy, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ],
    dtype=np.float64,
)

_PHI = 0.5 * (1.0 + np.sqrt(5.0))
_ICOSAHEDRON = np.unique(np.asarray([
    point
    for aa in (-1.0, 1.0)
    for bb in (-_PHI, _PHI)
    for point in ((0.0, aa, bb), (aa, bb, 0.0), (bb, 0.0, aa))
], dtype=np.float64), axis=0)
_ICOSAHEDRON /= np.linalg.norm(_ICOSAHEDRON, axis=1)[:, None]
_ICOSA13_OFFSETS = np.vstack((_ICOSAHEDRON, np.zeros((1, 3))))
_ICOSA13_MASS_FRACTIONS = np.concatenate((
    np.full(12, 0.0739476670674313),
    np.asarray([0.1126279951908244]),
))

_VECTOR_COMPONENTS = (
    ("u", "v", "w"),
    ("au", "av", "aw"),
    ("ax", "ay", "az"),
    ("u0", "v0", "w0"),
)


def damp_factor(count, n_damp):
    """Return PySPH's startup timestep damping factor."""
    if n_damp > 0 and count < n_damp:
        return 0.5 * (
            math.sin(math.pi * (-0.5 + (count + 1) / float(n_damp))) + 1.0
        )
    return 1.0


def _inside_bounds(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & (z >= zmin)
        & (z <= zmax)
    )


def _relative_residual(before, after, scale):
    return float(abs(after - before) / max(float(scale), 1.0e-30))


@dataclass
class AdaptationStats:
    """Metrics for one adaptation checkpoint."""

    particles_before: int
    particles_after: int
    split_parents: int = 0
    created_children: int = 0
    merged_families: int = 0
    removed_children: int = 0
    mass_before: float = 0.0
    mass_after: float = 0.0
    mass_residual: float = 0.0
    momentum_residual: float = 0.0
    shifted_particles: int = 0
    max_shift: float = 0.0

    def to_dict(self):
        return asdict(self)


class TwoLevelAdaptiveController:
    """Deterministic host oracle for two-level split/merge mechanics.

    Parameters
    ----------
    hdx:
        Ratio between smoothing length and particle spacing.
    fine_bounds:
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` target box.
    max_splits_per_adapt:
        Safety cap which bounds the eight-fold growth at one checkpoint.
    """

    metadata_props = ("level", "family_id")

    def __init__(
        self,
        hdx=1.3,
        fine_bounds=(1.75, 2.8, -0.3, 0.3, 0.0, 0.65),
        max_splits_per_adapt=128,
        split_stencil="octant8",
        hysteresis=0.0,
        shift_iterations=0,
        shift_coefficient=0.02,
    ):
        self.hdx = float(hdx)
        if self.hdx <= 0.0:
            raise ValueError("hdx must be positive")
        if len(fine_bounds) != 6:
            raise ValueError("fine_bounds must contain six values")
        self.fine_bounds = tuple(float(x) for x in fine_bounds)
        if not (
            self.fine_bounds[0] < self.fine_bounds[1]
            and self.fine_bounds[2] < self.fine_bounds[3]
            and self.fine_bounds[4] < self.fine_bounds[5]
        ):
            raise ValueError("fine_bounds must have increasing min/max pairs")
        self.max_splits_per_adapt = int(max_splits_per_adapt)
        if self.max_splits_per_adapt < 1:
            raise ValueError("max_splits_per_adapt must be >= 1")
        if split_stencil not in {"octant8", "icosa13"}:
            raise ValueError("split_stencil must be 'octant8' or 'icosa13'")
        self.split_stencil = split_stencil
        self.hysteresis = float(hysteresis)
        if self.hysteresis < 0.0:
            raise ValueError("hysteresis must be nonnegative")
        self.shift_iterations = int(shift_iterations)
        self.shift_coefficient = float(shift_coefficient)
        if self.shift_iterations < 0 or self.shift_coefficient < 0.0:
            raise ValueError("shift settings must be nonnegative")
        self.next_pid = 1

    @property
    def merge_bounds(self):
        value = self.hysteresis
        xmin, xmax, ymin, ymax, zmin, zmax = self.fine_bounds
        return (
            xmin - value, xmax + value,
            ymin - value, ymax + value,
            zmin - value, zmax + value,
        )

    @property
    def child_count(self):
        return 13 if self.split_stencil == "icosa13" else 8

    def _reconstruction_gradient(self, state, name, parent_index):
        """Least-squares first-order gradient for one scalar parent field."""
        center = np.asarray([
            state[axis][parent_index] for axis in ("x", "y", "z")
        ], dtype=np.float64)
        xyz = np.column_stack((state["x"], state["y"], state["z"]))
        delta = xyz - center
        radius = 2.0 * float(state["h"][parent_index])
        selected = (
            (np.linalg.norm(delta, axis=1) > 0.0)
            & (np.linalg.norm(delta, axis=1) <= radius)
        )
        matrix = delta[selected]
        if len(matrix) < 3 or np.linalg.matrix_rank(matrix) < 3:
            return np.zeros(3)
        values = np.asarray(state[name], dtype=np.float64)
        rhs = values[selected] - values[parent_index]
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def initialize_state(self, state: Mapping[str, np.ndarray]):
        """Copy ``state`` and attach stable particle/family metadata."""
        result = {
            name: np.asarray(values).copy()
            for name, values in state.items()
        }
        if not result:
            raise ValueError("particle state is empty")
        lengths = {len(values) for values in result.values()}
        if len(lengths) != 1:
            raise ValueError("all particle properties must have equal length")
        n = lengths.pop()
        if "pid" not in result or np.unique(result["pid"]).size != n:
            result["pid"] = np.arange(1, n + 1, dtype=np.int32)
        else:
            result["pid"] = np.asarray(result["pid"], dtype=np.int32)
        if "level" not in result:
            result["level"] = np.zeros(n, dtype=np.float64)
        if "family_id" not in result:
            result["family_id"] = result["pid"].astype(np.float64)
        self.next_pid = max(
            self.next_pid,
            int(np.max(result["pid"], initial=0)) + 1,
        )
        return result

    @staticmethod
    def invariants(state):
        """Return conserved mass and linear momentum from a host state."""
        mass = np.asarray(state["m"], dtype=np.float64)
        momentum = np.asarray(
            [
                np.sum(mass * np.asarray(state[name], dtype=np.float64))
                for name in ("u", "v", "w")
            ]
        )
        return float(np.sum(mass)), momentum

    def _split(self, state):
        level = np.asarray(state["level"]).astype(np.int32)
        inside = _inside_bounds(
            state["x"], state["y"], state["z"], self.fine_bounds
        )
        split_indices = np.flatnonzero((level == 0) & inside)
        split_indices = split_indices[: self.max_splits_per_adapt]
        if split_indices.size == 0:
            return state, 0

        keep = np.ones(len(level), dtype=bool)
        keep[split_indices] = False
        child_count = self.child_count
        nchild = split_indices.size * child_count
        child_data = {}
        for name, values in state.items():
            values = np.asarray(values)
            child_data[name] = np.repeat(values[split_indices], child_count)

        if self.split_stencil == "icosa13":
            offsets = (
                np.asarray(state["h"])[split_indices, None, None]
                * 0.65 * _ICOSA13_OFFSETS[None, :, :]
            ).reshape(nchild, 3)
            mass_fractions = np.tile(
                _ICOSA13_MASS_FRACTIONS, split_indices.size
            )
            child_h = 0.70 * np.repeat(
                np.asarray(state["h"])[split_indices], child_count
            )
        else:
            spacing = np.asarray(state["h"])[split_indices] / self.hdx
            offsets = (
                spacing[:, None, None] * 0.25 * _OCTANTS[None, :, :]
            ).reshape(nchild, 3)
            mass_fractions = np.full(nchild, 1.0 / child_count)
            child_h = 0.5 * np.repeat(
                np.asarray(state["h"])[split_indices], child_count
            )
        for axis, column in zip(("x", "y", "z"), range(3)):
            child_data[axis] = (
                np.repeat(
                    np.asarray(state[axis])[split_indices], child_count
                )
                + offsets[:, column]
            )
        child_data["m"] = child_data["m"] * mass_fractions
        child_data["h"] = child_h
        child_data["level"] = np.ones(
            nchild, dtype=np.asarray(state["level"]).dtype
        )
        families = np.repeat(
            np.asarray(state["pid"])[split_indices], child_count
        )
        child_data["family_id"] = families.astype(
            np.asarray(state["family_id"]).dtype
        )
        child_data["pid"] = np.arange(
            self.next_pid, self.next_pid + nchild, dtype=np.int32
        )
        self.next_pid += nchild

        # Reconstruct scalar fields to first order. Symmetric mass-weighted
        # offsets keep each family's linear momentum exactly at the parent
        # value while avoiding piecewise-constant daughter fields.
        excluded = {
            "x", "y", "z", "x0", "y0", "z0", "m", "h", "pid",
            "gid", "tag", "level", "family_id",
        }
        for name, values in state.items():
            if name in excluded or not np.issubdtype(values.dtype, np.floating):
                continue
            reconstructed = []
            for local, parent_index in enumerate(split_indices):
                gradient = self._reconstruction_gradient(
                    state, name, parent_index
                )
                family_offsets = offsets[
                    local * child_count:(local + 1) * child_count
                ]
                reconstructed.append(
                    float(values[parent_index]) + family_offsets @ gradient
                )
            child_data[name] = np.concatenate(reconstructed)

        # Saved PEC stage coordinates should describe the newly created
        # children, not the removed parent's center.
        for current, saved in (("x", "x0"), ("y", "y0"), ("z", "z0")):
            if saved in child_data:
                child_data[saved] = child_data[current].copy()
        for current, saved in (("u", "u0"), ("v", "v0"), ("w", "w0")):
            if saved in child_data:
                child_data[saved] = child_data[current].copy()
        if "rho0" in child_data:
            child_data["rho0"] = child_data["rho"].copy()

        result = {}
        for name, values in state.items():
            values = np.asarray(values)
            result[name] = np.concatenate(
                [values[keep], np.asarray(child_data[name], dtype=values.dtype)]
            )
        return result, int(split_indices.size)

    def _merge(self, state):
        level = np.asarray(state["level"]).astype(np.int32)
        outside = ~_inside_bounds(
            state["x"], state["y"], state["z"], self.merge_bounds
        )
        candidates = np.flatnonzero((level == 1) & outside)
        if candidates.size == 0:
            return state, 0

        families = np.asarray(state["family_id"]).astype(np.int64)
        selected = []
        family_groups = []
        for family in np.unique(families[candidates]):
            group = np.flatnonzero((level == 1) & (families == family))
            if group.size == self.child_count and np.all(outside[group]):
                selected.extend(group.tolist())
                family_groups.append((int(family), group))
        if not family_groups:
            return state, 0

        keep = np.ones(len(level), dtype=bool)
        keep[np.asarray(selected, dtype=np.intp)] = False
        parents = {name: [] for name in state}
        discrete = {"pid", "gid", "tag", "level", "family_id"}
        for family, group in family_groups:
            masses = np.asarray(state["m"], dtype=np.float64)[group]
            total_mass = float(np.sum(masses))
            weights = masses / total_mass
            for name, values in state.items():
                values = np.asarray(values)
                if name == "m":
                    value = total_mass
                elif name == "h":
                    factor = 0.70 if self.split_stencil == "icosa13" else 0.5
                    value = float(np.max(values[group])) / factor
                elif name == "pid" or name == "family_id":
                    value = family
                elif name == "level":
                    value = 0
                elif name in discrete:
                    value = values[group[0]]
                else:
                    value = float(np.sum(
                        np.asarray(values[group], dtype=np.float64) * weights
                    ))
                parents[name].append(value)

        result = {}
        for name, values in state.items():
            values = np.asarray(values)
            result[name] = np.concatenate(
                [values[keep], np.asarray(parents[name], dtype=values.dtype)]
            )
        return result, len(family_groups)

    def _shift(self, state, solid_xyz=None):
        """Regularize fine particles with conservative, limited shifts."""
        if self.shift_iterations == 0 or self.shift_coefficient == 0.0:
            return state, 0, 0.0
        result = {name: np.asarray(values).copy()
                  for name, values in state.items()}
        level = np.asarray(result["level"], dtype=np.int32)
        fine = np.flatnonzero(level == 1)
        if fine.size == 0:
            return result, 0, 0.0
        families = np.asarray(result["family_id"], dtype=np.int64)
        mass = np.asarray(result["m"], dtype=np.float64)
        solid_xyz = (
            None if solid_xyz is None
            else np.asarray(solid_xyz, dtype=np.float64).reshape(-1, 3)
        )
        shifted = np.zeros(len(level), dtype=bool)
        max_shift = 0.0
        correct_fields = tuple(
            name for name in ("rho", "p", "cs", "u", "v", "w")
            if name in result
        )
        for _ in range(self.shift_iterations):
            xyz = np.column_stack((result["x"], result["y"], result["z"]))
            shifts = np.zeros_like(xyz, dtype=np.float64)
            gradients = {
                name: {
                    int(index): self._reconstruction_gradient(
                        result, name, int(index)
                    )
                    for index in fine
                }
                for name in correct_fields
            }
            for index in fine:
                delta = xyz[index] - xyz
                distance = np.linalg.norm(delta, axis=1)
                support = 2.0 * float(result["h"][index])
                neighbors = (distance > 1.0e-14) & (distance < support)
                external = neighbors & (families != families[index])
                # A low external-neighbor count identifies a free surface or
                # isolated daughter family; leave it untouched.
                if np.count_nonzero(external) < 6:
                    continue
                q = distance[neighbors] / support
                direction = delta[neighbors] / distance[neighbors, None]
                raw = np.sum(((1.0 - q) ** 2)[:, None] * direction, axis=0)
                raw /= max(np.count_nonzero(neighbors), 1)
                shift = self.shift_coefficient * float(
                    result["h"][index]
                ) * raw
                limit = 0.05 * float(result["h"][index])
                magnitude = np.linalg.norm(shift)
                if magnitude > limit:
                    shift *= limit / magnitude
                if solid_xyz is not None and len(solid_xyz):
                    separation = xyz[index] - solid_xyz
                    nearest = int(np.argmin(np.linalg.norm(separation, axis=1)))
                    away = separation[nearest]
                    away_norm = np.linalg.norm(away)
                    if away_norm < 1.5 * float(result["h"][index]):
                        away /= max(away_norm, 1.0e-30)
                        into_solid = np.dot(shift, away)
                        if into_solid < 0.0:
                            shift -= into_solid * away
                shifts[index] = shift

            # Preserve each complete family's mass-weighted centroid. This
            # prevents the regularizer itself from advecting fluid mass.
            for family in np.unique(families[fine]):
                group = np.flatnonzero((level == 1) & (families == family))
                weights = mass[group] / np.sum(mass[group])
                shifts[group] -= np.sum(
                    weights[:, None] * shifts[group], axis=0
                )
            magnitude = np.linalg.norm(shifts, axis=1)
            moved = magnitude > 0.0
            shifted |= moved
            max_shift = max(max_shift, float(np.max(magnitude, initial=0.0)))
            for axis, column in zip(("x", "y", "z"), range(3)):
                result[axis] += shifts[:, column]
            for name in correct_fields:
                before_momentum = None
                if name in {"u", "v", "w"}:
                    before_momentum = float(np.sum(mass * result[name]))
                for index in fine:
                    result[name][index] += np.dot(
                        gradients[name][int(index)], shifts[index]
                    )
                if before_momentum is not None:
                    correction = (
                        np.sum(mass * result[name]) - before_momentum
                    ) / np.sum(mass)
                    result[name] -= correction
            for current, saved in (
                ("x", "x0"), ("y", "y0"), ("z", "z0"),
                ("u", "u0"), ("v", "v0"), ("w", "w0"),
                ("rho", "rho0"),
            ):
                if saved in result and current in result:
                    result[saved] = result[current].copy()
        return result, int(np.count_nonzero(shifted)), max_shift

    def adapt(self, state: Mapping[str, np.ndarray], solid_xyz=None):
        """Merge complete outside families, then split coarse inside parents."""
        state = self.initialize_state(state)
        mass_before, momentum_before = self.invariants(state)
        particles_before = len(state["x"])
        merged, nmerge = self._merge(state)
        adapted, nsplit = self._split(merged)
        adapted, nshift, max_shift = self._shift(
            adapted, solid_xyz=solid_xyz
        )
        mass_after, momentum_after = self.invariants(adapted)
        momentum_scale = np.sum(
            np.asarray(state["m"], dtype=np.float64)
            * np.sqrt(
                np.asarray(state["u"], dtype=np.float64) ** 2
                + np.asarray(state["v"], dtype=np.float64) ** 2
                + np.asarray(state["w"], dtype=np.float64) ** 2
            )
        )
        stats = AdaptationStats(
            particles_before=particles_before,
            particles_after=len(adapted["x"]),
            split_parents=nsplit,
            created_children=self.child_count * nsplit,
            merged_families=nmerge,
            removed_children=self.child_count * nmerge,
            mass_before=mass_before,
            mass_after=mass_after,
            mass_residual=_relative_residual(
                mass_before, mass_after, abs(mass_before)
            ),
            momentum_residual=float(
                np.linalg.norm(momentum_after - momentum_before)
                / max(float(momentum_scale), 1.0e-30)
            ),
            shifted_particles=nshift,
            max_shift=max_shift,
        )
        return adapted, stats


def particle_state(pa, pull=True):
    """Return all scalar particle properties as independent NumPy arrays."""
    if pa.gpu is not None and getattr(pa.gpu, "backend", None) == "warp":
        result = {}
        for name in pa.gpu.properties:
            data = pa.gpu.get_device_array(name).get()
            stride = pa.stride.get(name, 1)
            if stride != 1:
                raise ValueError(
                    "adaptive smoke supports scalar properties only; "
                    f"{name!r} has stride {stride}"
                )
            result[name] = data.copy()
        if pull:
            for name, data in result.items():
                ary = pa.properties[name]
                if ary.length != data.size:
                    ary.resize(data.size)
                ary.set_data(data)
            pa.set_num_real_particles(len(result["x"]))
        return result
    return {
        name: ary.get_npy_array().copy()
        for name, ary in pa.properties.items()
    }


def warp_particle_array_from_state(state, name="fluid", device=None):
    """Build a Warp ``ParticleArray`` from a complete host state."""
    state = {key: np.asarray(value) for key, value in state.items()}
    extras = sorted(set(state).difference(DEFAULT_PROPS))
    pa = get_particle_array(
        name=name,
        additional_props=extras,
        backend="warp",
        **state,
    )
    if device is not None and str(pa.gpu.device) != str(device):
        # ParticleArray's backend constructor chooses Warp's current device.
        # Rebinding through the normal helper keeps this function useful when
        # callers select a non-default CUDA device.
        from pysph.base.warp_device_helper import WarpDeviceHelper

        pa.set_device_helper(WarpDeviceHelper(pa, backend="warp", device=device))
    return pa


@dataclass
class DamBreakConfig:
    """Serializable configuration shared by the experiment and web worker."""

    resolution_mode: str = "adaptive"
    dx: float = 0.1
    hdx: float = 1.3
    steps: int = 40
    rho0: float = 1000.0
    c0: float = REF_C0
    p0: float = 0.0
    gamma: float = 7.0
    alpha: float = 0.25
    beta: float = 0.0
    variable_h_correction: bool = True
    kernel: str = "wendland"
    radius_scale: float = 2.0
    xsph_eps: float | None = 0.5
    gz: float = -GRAVITY
    n_damp: int = 50
    nboundary_layers: int = 1
    adaptive_dt: bool = True
    cfl: float = 0.3
    dt: float | None = None
    dt_min: float = 0.0
    dt_max: float | None = None
    # ``with_obstacle`` is retained for old manifests.  New callers should use
    # obstacle_mode so a solid can be absent, fixed, or a coupled rigid body.
    obstacle_mode: str | None = None
    with_obstacle: bool | None = True
    obstacle_center_x: float = 3.0
    body_density: float = 500.0
    body_center_x: float = 2.35
    body_center_y: float = 0.0
    body_center_z: float = 0.30
    body_length: float = 0.32
    body_width: float = 0.28
    body_height: float = 0.20
    body_spacing: float | None = None
    contact_enabled: bool = True
    contact_radius: float | None = None
    contact_stiffness: float = 5.0e4
    contact_restitution: float = 0.3
    contact_friction: float = 0.2
    contact_dt_safety: float = 0.2
    contact_bounds: tuple[float, float, float, float, float, float] = (
        0.0, 161.0 / 30.0, -0.25, 0.25, 0.0, 1.5,
    )
    adapt_every: int = 5
    max_splits_per_adapt: int = 128
    split_stencil: str = "icosa13"
    adapt_hysteresis: float = 0.0
    shift_iterations: int = 0
    shift_coefficient: float = 0.02
    fine_bounds: tuple[float, float, float, float, float, float] = (
        1.75,
        2.8,
        -0.3,
        0.3,
        0.0,
        0.65,
    )
    device: str | None = None

    def validate(self):
        if self.resolution_mode not in {"uniform", "adaptive"}:
            raise ValueError("resolution_mode must be 'uniform' or 'adaptive'")
        if self.dx <= 0 or self.hdx <= 0:
            raise ValueError("dx and hdx must be positive")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.kernel not in {"cubic", "gaussian", "wendland"}:
            raise ValueError("unsupported kernel")
        if self.adapt_every < 1:
            raise ValueError("adapt_every must be >= 1")
        if self.split_stencil not in {"octant8", "icosa13"}:
            raise ValueError("split_stencil must be 'octant8' or 'icosa13'")
        if self.adapt_hysteresis < 0.0:
            raise ValueError("adapt_hysteresis must be nonnegative")
        if self.shift_iterations < 0 or self.shift_coefficient < 0.0:
            raise ValueError("shift settings must be nonnegative")
        if self.cfl <= 0:
            raise ValueError("cfl must be positive")
        if self.obstacle_mode is None:
            self.obstacle_mode = "fixed" if self.with_obstacle else "none"
        if self.obstacle_mode not in {"none", "fixed", "floating"}:
            raise ValueError(
                "obstacle_mode must be 'none', 'fixed', or 'floating'"
            )
        self.with_obstacle = self.obstacle_mode != "none"
        if self.body_density <= 0.0:
            raise ValueError("body_density must be positive")
        if min(self.body_length, self.body_width, self.body_height) <= 0.0:
            raise ValueError("floating body dimensions must be positive")
        if self.body_spacing is not None and self.body_spacing <= 0.0:
            raise ValueError("body_spacing must be positive")
        if self.contact_radius is not None and self.contact_radius < 0.0:
            raise ValueError("contact_radius must be nonnegative")
        if self.contact_stiffness <= 0.0 or self.contact_dt_safety <= 0.0:
            raise ValueError("contact stiffness and timestep safety must be positive")
        if not 0.0 < self.contact_restitution <= 1.0:
            raise ValueError("contact_restitution must be in (0, 1]")
        if self.contact_friction < 0.0:
            raise ValueError("contact_friction must be nonnegative")
        if len(self.contact_bounds) != 6:
            raise ValueError("contact_bounds must contain six values")
        if not (
            self.contact_bounds[0] < self.contact_bounds[1]
            and self.contact_bounds[2] < self.contact_bounds[3]
            and self.contact_bounds[4] < self.contact_bounds[5]
        ):
            raise ValueError(
                "contact_bounds must have increasing min/max pairs"
            )
        return self

    def to_dict(self):
        # Normalize the legacy ``with_obstacle`` flag before serializing so a
        # round trip has one unambiguous obstacle representation.
        self.validate()
        result = asdict(self)
        result["fine_bounds"] = list(self.fine_bounds)
        result["contact_bounds"] = list(self.contact_bounds)
        return result

    @classmethod
    def from_mapping(cls, values):
        data = dict(values)
        if "fine_bounds" in data:
            data["fine_bounds"] = tuple(data["fine_bounds"])
        if "contact_bounds" in data:
            data["contact_bounds"] = tuple(data["contact_bounds"])
        return cls(**data).validate()


class WarpDamBreakSimulation:
    """Incremental uniform/adaptive Warp dam-break simulation."""

    def __init__(self, config: DamBreakConfig | Mapping | None = None):
        if config is None:
            config = DamBreakConfig()
        elif not isinstance(config, DamBreakConfig):
            config = DamBreakConfig.from_mapping(config)
        self.config = config.validate()
        self.h0 = self.config.hdx * self.config.dx
        co = 10.0 * math.sqrt(2.0 * GRAVITY * H)
        reference_dt = 0.25 * self.h0 / (1.1 * co)
        self.seed_dt = (
            reference_dt if self.config.dt is None else float(self.config.dt)
        )
        self.dt_max = (
            float("inf")
            if self.config.dt_max is None
            else float(self.config.dt_max)
        )
        self.controller = TwoLevelAdaptiveController(
            hdx=self.config.hdx,
            fine_bounds=self.config.fine_bounds,
            max_splits_per_adapt=self.config.max_splits_per_adapt,
            split_stencil=self.config.split_stencil,
            hysteresis=self.config.adapt_hysteresis,
            shift_iterations=self.config.shift_iterations,
            shift_coefficient=self.config.shift_coefficient,
        )
        self.particles = None
        self.particle_names = []
        self.nnps = None
        self.rigid_state = None
        self._rigid_initial_xyz = None
        self._rigid_initial_distances = None
        self.neighbor_mode = "grid"
        self.step_count = 0
        self.time = 0.0
        self.dt_history = []
        self.adaptation_history = []
        self.initial_mass = None
        self._push_next = True
        self._initialized = False

    def _geometry(self):
        return DamBreak3DGeometry(
            container_height=1.5 * H,
            container_width=H / 2.0,
            container_length=161 * H / 30.0,
            fluid_column_height=H,
            fluid_column_width=H / 2.0,
            fluid_column_length=2.0 * H,
            dx=self.config.dx,
            nboundary_layers=self.config.nboundary_layers,
            hdx=self.config.hdx,
            rho0=self.config.rho0,
            with_obstacle=self.config.obstacle_mode == "fixed",
            obstacle_center_x=self.config.obstacle_center_x,
        )

    def _floating_body(self):
        """Create a shell-sampled rigid box with physical total mass."""
        size = np.asarray(
            [
                self.config.body_length,
                self.config.body_width,
                self.config.body_height,
            ],
            dtype=np.float64,
        )
        axes = []
        spacing = (
            self.config.dx
            if self.config.body_spacing is None else self.config.body_spacing
        )
        for length in size:
            count = max(3, int(round(float(length) / spacing)) + 1)
            axes.append(np.linspace(-0.5 * length, 0.5 * length, count))
        ix, iy, iz = np.meshgrid(
            np.arange(len(axes[0])),
            np.arange(len(axes[1])),
            np.arange(len(axes[2])),
            indexing="ij",
        )
        shell = (
            (ix == 0)
            | (ix == len(axes[0]) - 1)
            | (iy == 0)
            | (iy == len(axes[1]) - 1)
            | (iz == 0)
            | (iz == len(axes[2]) - 1)
        )
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        xyz = np.column_stack((xx[shell], yy[shell], zz[shell]))
        xyz += np.asarray(
            [
                self.config.body_center_x,
                self.config.body_center_y,
                self.config.body_center_z,
            ]
        )
        count = len(xyz)
        total_mass = self.config.body_density * float(np.prod(size))
        zeros = np.zeros(count, dtype=np.float64)
        body = get_particle_array(
            name="body",
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            h=np.full(count, self.h0),
            m=np.full(count, total_mass / count),
            rho=np.full(count, self.config.rho0),
            rho0=np.full(count, self.config.rho0),
            p=zeros.copy(),
            cs=np.full(count, self.config.c0),
            u=zeros.copy(),
            v=zeros.copy(),
            w=zeros.copy(),
            arho=zeros.copy(),
            beta_h=np.ones(count, dtype=np.float64),
            V=zeros.copy(),
            fx=zeros.copy(),
            fy=zeros.copy(),
            fz=zeros.copy(),
            body_id=np.zeros(count, dtype=np.int32),
            backend="warp",
        )
        if self.config.device is not None and str(body.gpu.device) != str(
            self.config.device
        ):
            from pysph.base.warp_device_helper import WarpDeviceHelper

            body.set_device_helper(
                WarpDeviceHelper(
                    body, backend="warp", device=self.config.device
                )
            )
        return body

    def _to_warp(self, source, name):
        n = source.get_number_of_particles()
        zeros = np.zeros(n, dtype=np.float64)
        return get_particle_array(
            name=name,
            x=np.asarray(source.x, dtype=np.float64).copy(),
            y=np.asarray(source.y, dtype=np.float64).copy(),
            z=np.asarray(source.z, dtype=np.float64).copy(),
            h=np.asarray(source.h, dtype=np.float64).copy(),
            m=np.asarray(source.m, dtype=np.float64).copy(),
            rho=np.full(n, self.config.rho0, dtype=np.float64),
            p=zeros.copy(),
            cs=np.full(n, self.config.c0, dtype=np.float64),
            u=zeros.copy(),
            v=zeros.copy(),
            w=zeros.copy(),
            au=zeros.copy(),
            av=zeros.copy(),
            aw=zeros.copy(),
            arho=zeros.copy(),
            beta_h=np.ones(n, dtype=np.float64),
            ax=zeros.copy(),
            ay=zeros.copy(),
            az=zeros.copy(),
            x0=zeros.copy(),
            y0=zeros.copy(),
            z0=zeros.copy(),
            u0=zeros.copy(),
            v0=zeros.copy(),
            w0=zeros.copy(),
            rho0=zeros.copy(),
            backend="warp",
        )

    def _build_nnps(self):
        if self.config.resolution_mode == "adaptive":
            self.nnps = MultilevelGridWarpNNPS(
                dim=3,
                particles=self.particles,
                radius_scale=self.config.radius_scale,
                h_ref=self.h0 / 2.0,
                level_ratio=2.0,
                nlevels=2,
                device=self.config.device,
            )
            self.neighbor_mode = "multilevel"
        else:
            self.nnps = UniformGridWarpNNPS(
                dim=3,
                particles=self.particles,
                radius_scale=self.config.radius_scale,
                device=self.config.device,
            )
            self.neighbor_mode = "grid"

    def _adapt(self):
        fluid = self.particles[0]
        state = particle_state(fluid)
        solid_xyz = None
        rigid_before = None
        if self.rigid_state is not None:
            # NNPS constructors preserve their historical push=True behavior.
            # Synchronize the moving body's host mirror first so rebuilding
            # after fluid mutation can never rewind rigid coordinates.
            body = particle_state(self.particles[2], pull=True)
            solid_xyz = np.column_stack((body["x"], body["y"], body["z"]))
            rigid_before = {
                name: np.asarray(body[name]).copy()
                for name in ("x", "y", "z", "m", "body_id")
            }
        state, stats = self.controller.adapt(state, solid_xyz=solid_xyz)
        self.particles[0] = warp_particle_array_from_state(
            state, name="fluid", device=self.config.device
        )
        self._build_nnps()
        if rigid_before is not None:
            rigid_after = particle_state(self.particles[2], pull=False)
            for name, expected in rigid_before.items():
                if not np.array_equal(rigid_after[name], expected):
                    raise RuntimeError(
                        f"fluid adaptation changed rigid property {name!r}"
                    )
        self._push_next = False
        entry = {
            "step": self.step_count,
            "time": self.time,
            **stats.to_dict(),
        }
        self.adaptation_history.append(entry)
        return entry

    def initialize(self):
        if self._initialized:
            return self.snapshot(include_solids=True)
        cpu_particles = self._geometry().create_particles()
        names = ("fluid", "wall", "obstacle")
        self.particles = [
            self._to_warp(pa, names[index])
            for index, pa in enumerate(cpu_particles)
        ]
        self.particle_names = list(names[: len(self.particles)])
        if self.config.obstacle_mode == "floating":
            body = self._floating_body()
            self.particles.append(body)
            self.particle_names.append("body")
            self.rigid_state = create_rigid_body_state(
                body, nbody=1, device=self.config.device
            )
            xyz = np.column_stack((body.x, body.y, body.z))
            self._rigid_initial_xyz = xyz.copy()
            self._rigid_initial_distances = np.linalg.norm(
                xyz - xyz[0], axis=1
            )
        initial_state = particle_state(self.particles[0], pull=False)
        initial_state = self.controller.initialize_state(initial_state)
        self.initial_mass = float(np.sum(initial_state["m"], dtype=np.float64))
        self.particles[0] = warp_particle_array_from_state(
            initial_state, name="fluid", device=self.config.device
        )
        self._build_nnps()
        self._push_next = False
        if self.config.resolution_mode == "adaptive":
            self._adapt()
        self._initialized = True
        return self.snapshot(include_solids=True)

    @property
    def done(self):
        return self.step_count >= self.config.steps

    def step(self):
        if not self._initialized:
            self.initialize()
        if self.done:
            return None
        scale = damp_factor(self.step_count, self.config.n_damp)
        common = dict(
            dt=self.seed_dt,
            rho0=self.config.rho0,
            c0=self.config.c0,
            p0=self.config.p0,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
            xsph_eps=self.config.xsph_eps,
            gx=0.0,
            gy=0.0,
            gz=self.config.gz,
            adaptive_dt=self.config.adaptive_dt,
            cfl=self.config.cfl,
            dt_min=self.config.dt_min,
            dt_max=self.dt_max,
            adaptive_dt_scale=scale,
            step_dt_max=self.dt_max,
            push=self._push_next,
            return_dt=True,
            neighbor_mode=self.neighbor_mode,
            variable_h_correction=(
                self.config.variable_h_correction
                and self.config.resolution_mode == "adaptive"
            ),
        )
        if self.config.obstacle_mode == "floating":
            contact_bounds = (
                self.config.contact_bounds
                if self.config.contact_enabled else None
            )
            dt_used = wc_sph_dam_break_rigid_step(
                self.nnps,
                self.rigid_state,
                fluid_index=0,
                wall_indices=(1,),
                rigid_index=2,
                contact_bounds=contact_bounds,
                contact_radius=(
                    0.5 * self.config.dx
                    if self.config.contact_radius is None
                    else self.config.contact_radius
                ),
                contact_stiffness=self.config.contact_stiffness,
                contact_restitution=self.config.contact_restitution,
                contact_friction=self.config.contact_friction,
                contact_dt_safety=self.config.contact_dt_safety,
                **common,
            )
        else:
            common["gravity_ramp"] = 1.0
            dt_used = wc_sph_dam_break_step(
                self.nnps,
                fluid_index=0,
                solid_indices=tuple(range(1, len(self.particles))),
                **common,
            )
        self._push_next = False
        self.dt_history.append(float(dt_used))
        self.time += float(dt_used)
        self.step_count += 1
        adaptation = None
        if (
            self.config.resolution_mode == "adaptive"
            and self.step_count % self.config.adapt_every == 0
            and not self.done
        ):
            adaptation = self._adapt()
        return {
            "step": self.step_count,
            "time": self.time,
            "dt": float(dt_used),
            "adaptation": adaptation,
        }

    def snapshot(self, include_solids=True):
        if not self._initialized and self.particles is None:
            raise RuntimeError("simulation is not initialized")
        arrays = []
        kinds = []
        levels = []
        names = self.particle_names
        selected = self.particles if include_solids else self.particles[:1]
        for index, pa in enumerate(selected):
            state = particle_state(pa, pull=False)
            n = len(state["x"])
            speed = np.sqrt(
                state["u"] ** 2 + state["v"] ** 2 + state["w"] ** 2
            )
            arrays.append(
                np.column_stack(
                    [
                        state["x"],
                        state["y"],
                        state["z"],
                        state["h"],
                        state["rho"],
                        state["p"],
                        speed,
                        state["u"],
                        state["v"],
                        state["w"],
                    ]
                ).astype(np.float32, copy=False)
            )
            kinds.append(np.full(n, index, dtype=np.uint8))
            if index == 0 and "level" in state:
                levels.append(np.asarray(state["level"], dtype=np.uint8))
            else:
                levels.append(np.full(n, 2, dtype=np.uint8))
        packed = np.concatenate(arrays, axis=0)
        return {
            "step": self.step_count,
            "time": self.time,
            "xyz": np.ascontiguousarray(packed[:, :3]),
            "h": np.ascontiguousarray(packed[:, 3]),
            "rho": np.ascontiguousarray(packed[:, 4]),
            "p": np.ascontiguousarray(packed[:, 5]),
            "speed": np.ascontiguousarray(packed[:, 6]),
            "velocity": np.ascontiguousarray(packed[:, 7:10]),
            "kind": np.concatenate(kinds),
            "level": np.concatenate(levels),
            "counts": {
                names[index]: int(
                    self.particles[index].get_number_of_particles()
                )
                for index in range(len(self.particles))
            },
            "obstacle_mode": self.config.obstacle_mode,
        }

    def _rigid_metrics(self):
        if self.rigid_state is None:
            return {
                "body_particles": 0,
                "body_mass": 0.0,
                "body_cm": None,
                "body_vc": None,
                "body_omega": None,
                "body_force": None,
                "body_torque": None,
                "body_fluid_force": None,
                "body_fluid_torque": None,
                "body_orientation": None,
                "body_quaternion": None,
                "body_geometry_drift": 0.0,
                "contact_force": None,
                "contact_torque": None,
                "contact_impulse": None,
                "contact_max_penetration": 0.0,
                "contact_occurred": False,
                "rigid_device_error": 0,
            }
        body = particle_state(self.particles[2], pull=False)
        xyz = np.column_stack((body["x"], body["y"], body["z"]))
        mass = np.asarray(body["m"], dtype=np.float64)
        cm = np.sum(mass[:, None] * xyz, axis=0) / np.sum(mass)
        distances = np.linalg.norm(xyz - xyz[0], axis=1)
        scale = max(float(np.max(self._rigid_initial_distances)), 1.0e-30)
        drift = float(
            np.max(np.abs(distances - self._rigid_initial_distances)) / scale
        )
        initial_cm = np.sum(
            mass[:, None] * self._rigid_initial_xyz, axis=0
        ) / np.sum(mass)
        reference = self._rigid_initial_xyz - initial_cm
        current = xyz - cm
        covariance = reference.T @ (mass[:, None] * current)
        uu, _, vv_t = np.linalg.svd(covariance)
        orientation = vv_t.T @ uu.T
        if np.linalg.det(orientation) < 0.0:
            vv_t[-1] *= -1.0
            orientation = vv_t.T @ uu.T
        force = np.column_stack((body["fx"], body["fy"], body["fz"]))
        contact_force_particles = np.column_stack((
            body.get("contact_fx", np.zeros(len(xyz))),
            body.get("contact_fy", np.zeros(len(xyz))),
            body.get("contact_fz", np.zeros(len(xyz))),
        ))
        total_force = np.sum(force, axis=0)
        contact_force = np.sum(contact_force_particles, axis=0)
        total_torque = self.rigid_state.torque.numpy().reshape(-1, 3)[0]
        contact_torque = np.sum(
            np.cross(xyz - cm, contact_force_particles), axis=0
        )
        gravity_force = np.asarray(
            [0.0, 0.0, float(np.sum(mass)) * self.config.gz]
        )
        return {
            "body_particles": len(xyz),
            "body_mass": float(np.sum(mass)),
            "body_cm": cm.tolist(),
            "body_vc": self.rigid_state.vc.numpy().reshape(-1, 3)[0].tolist(),
            "body_omega": self.rigid_state.omega.numpy().reshape(-1, 3)[0].tolist(),
            "body_force": total_force.tolist(),
            "body_torque": total_torque.tolist(),
            "body_fluid_force": (
                total_force - gravity_force - contact_force
            ).tolist(),
            "body_fluid_torque": (total_torque - contact_torque).tolist(),
            "body_orientation": orientation.tolist(),
            "body_quaternion": self.rigid_state.q.numpy().reshape(-1, 4)[0].tolist(),
            "body_geometry_drift": drift,
            "contact_force": contact_force.tolist(),
            "contact_torque": contact_torque.tolist(),
            "contact_impulse": np.sum(
                np.column_stack((
                    body.get("contact_impulse_x", np.zeros(len(xyz))),
                    body.get("contact_impulse_y", np.zeros(len(xyz))),
                    body.get("contact_impulse_z", np.zeros(len(xyz))),
                )),
                axis=0,
            ).tolist(),
            "contact_max_penetration": float(np.max(
                body.get("max_penetration_history", np.zeros(len(xyz)))
            )),
            "contact_occurred": bool(np.any(
                np.abs(body.get("contact_impulse_x", np.zeros(len(xyz))))
                + np.abs(body.get("contact_impulse_y", np.zeros(len(xyz))))
                + np.abs(body.get("contact_impulse_z", np.zeros(len(xyz))))
                > 0.0
            )),
            "rigid_device_error": int(self.rigid_state.error.numpy()[0]),
        }

    def metrics(self):
        snap = self.snapshot(include_solids=True)
        fluid = particle_state(self.particles[0], pull=False)
        mass = float(np.sum(fluid["m"], dtype=np.float64))
        finite = all(
            np.all(np.isfinite(fluid[name]))
            for name in ("x", "y", "z", "rho", "p", "u", "v", "w")
        )
        level = np.asarray(
            fluid.get("level", np.zeros(len(fluid["x"]))), dtype=np.int32
        )
        velocity = np.column_stack(
            (fluid["u"], fluid["v"], fluid["w"])
        ).astype(np.float64, copy=False)
        particle_mass = np.asarray(fluid["m"], dtype=np.float64)
        fluid_momentum = np.sum(
            particle_mass[:, None] * velocity, axis=0, dtype=np.float64
        )
        fluid_kinetic_energy = 0.5 * np.sum(
            particle_mass * np.sum(velocity * velocity, axis=1),
            dtype=np.float64,
        )
        latest_adapt = (
            self.adaptation_history[-1] if self.adaptation_history else None
        )
        return {
            "obstacle_mode": self.config.obstacle_mode,
            "resolution_mode": self.config.resolution_mode,
            "step": self.step_count,
            "steps": self.config.steps,
            "time": self.time,
            "dt_last": self.dt_history[-1] if self.dt_history else None,
            "fluid_particles": len(fluid["x"]),
            "fine_particles": int(np.count_nonzero(level == 1)),
            "coarse_particles": int(np.count_nonzero(level == 0)),
            "wall_particles": snap["counts"].get("wall", 0),
            "obstacle_particles": snap["counts"].get("obstacle", 0),
            "mass": mass,
            "mass_drift": _relative_residual(
                self.initial_mass, mass, abs(self.initial_mass)
            ),
            "fluid_momentum": fluid_momentum.tolist(),
            "fluid_kinetic_energy": float(fluid_kinetic_energy),
            "rho_min": float(np.min(fluid["rho"])),
            "rho_max": float(np.max(fluid["rho"])),
            "p_min": float(np.min(fluid["p"])),
            "p_max": float(np.max(fluid["p"])),
            "beta_h_min": float(np.min(
                fluid.get("beta_h", np.ones(len(fluid["x"])))
            )),
            "beta_h_max": float(np.max(
                fluid.get("beta_h", np.ones(len(fluid["x"])))
            )),
            "surge_front_x": float(np.max(fluid["x"])),
            "max_height": float(np.max(fluid["z"])),
            "all_finite": bool(finite),
            "adaptation_events": len(self.adaptation_history),
            "split_parents": int(sum(
                x["split_parents"] for x in self.adaptation_history
            )),
            "merged_families": int(sum(
                x["merged_families"] for x in self.adaptation_history
            )),
            "shifted_particles": int(
                latest_adapt.get("shifted_particles", 0)
                if latest_adapt else 0
            ),
            "max_shift": float(max(
                (x.get("max_shift", 0.0) for x in self.adaptation_history),
                default=0.0,
            )),
            "latest_adaptation": latest_adapt,
            **self._rigid_metrics(),
        }

    def run(
        self,
        snapshot_stride=1,
        callback: Callable[[dict, dict], None] | None = None,
    ):
        self.initialize()
        if callback is not None:
            callback(self.snapshot(), self.metrics())
        while not self.done:
            self.step()
            if (
                callback is not None
                and (
                    self.step_count % max(int(snapshot_stride), 1) == 0
                    or self.done
                )
            ):
                callback(self.snapshot(), self.metrics())
        return self.metrics()

    def save(self, path):
        """Save visualization plus complete restartable solver state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot(include_solids=True)
        metrics = self.metrics()
        particle_states = [
            particle_state(pa, pull=False) for pa in self.particles
        ]
        restart = {
            f"particle_{index}_{name}": values
            for index, state in enumerate(particle_states)
            for name, values in state.items()
        }
        schemas = [sorted(state) for state in particle_states]
        rigid = {}
        if self.rigid_state is not None:
            for name in ("vc", "omega", "vc0", "omega0", "q", "q0", "error"):
                rigid[f"rigid_{name}"] = getattr(
                    self.rigid_state, name
                ).numpy()
            rigid["rigid_initial_xyz"] = self._rigid_initial_xyz
            rigid["rigid_reference_xyz"] = np.column_stack((
                self.rigid_state.reference_x.numpy(),
                self.rigid_state.reference_y.numpy(),
                self.rigid_state.reference_z.numpy(),
            ))
        np.savez(
            path,
            **{
                key: value
                for key, value in snapshot.items()
                if isinstance(value, np.ndarray)
            },
            dt_history=np.asarray(self.dt_history),
            metrics=json.dumps(metrics, sort_keys=True),
            config=json.dumps(self.config.to_dict(), sort_keys=True),
            adaptation_history=json.dumps(
                self.adaptation_history, sort_keys=True
            ),
            particle_names=json.dumps(self.particle_names),
            particle_schemas=json.dumps(schemas),
            initial_mass=np.asarray(self.initial_mass),
            controller_next_pid=np.asarray(self.controller.next_pid),
            **restart,
            **rigid,
        )
        return path

    @classmethod
    def load(cls, path, steps=None):
        """Restore a complete checkpoint saved by :meth:`save`.

        ``steps`` may extend the final target step for a continuation run.
        Visualization-only legacy files fail explicitly instead of silently
        constructing an incomplete physical state.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = {
                "config", "metrics", "particle_names", "particle_schemas",
                "initial_mass", "controller_next_pid",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    "checkpoint is not restartable; missing " + ", ".join(missing)
                )
            config = DamBreakConfig.from_mapping(
                json.loads(str(data["config"].item()))
            )
            metrics = json.loads(str(data["metrics"].item()))
            names = json.loads(str(data["particle_names"].item()))
            schemas = json.loads(str(data["particle_schemas"].item()))
            if steps is not None:
                config.steps = int(steps)
                config.validate()
            simulation = cls(config)
            simulation.particle_names = list(names)
            simulation.particles = []
            for index, (name, schema) in enumerate(zip(names, schemas)):
                state = {
                    prop: np.asarray(data[f"particle_{index}_{prop}"])
                    for prop in schema
                }
                simulation.particles.append(
                    warp_particle_array_from_state(
                        state, name=name, device=config.device
                    )
                )
            simulation.initial_mass = float(data["initial_mass"])
            simulation.controller.next_pid = int(data["controller_next_pid"])
            simulation.step_count = int(metrics["step"])
            simulation.time = float(metrics["time"])
            simulation.dt_history = np.asarray(
                data.get("dt_history", np.asarray([])), dtype=np.float64
            ).tolist()
            simulation.adaptation_history = json.loads(
                str(data["adaptation_history"].item())
            )
            if config.obstacle_mode == "floating":
                body = simulation.particles[2]
                vc = np.asarray(data["rigid_vc"]).reshape(-1, 3)
                omega = np.asarray(data["rigid_omega"]).reshape(-1, 3)
                q = (
                    np.asarray(data["rigid_q"]).reshape(-1, 4)
                    if "rigid_q" in data else None
                )
                if "rigid_reference_xyz" in data:
                    reference_xyz = np.asarray(data["rigid_reference_xyz"])
                else:
                    body_xyz = np.column_stack((body.x, body.y, body.z))
                    body_mass = np.asarray(body.m, dtype=np.float64)
                    body_cm = np.sum(
                        body_mass[:, None] * body_xyz, axis=0
                    ) / np.sum(body_mass)
                    reference_xyz = body_xyz - body_cm
                simulation.rigid_state = create_rigid_body_state(
                    body, nbody=len(vc), vc=vc, omega=omega, q=q,
                    reference_xyz=reference_xyz,
                    device=config.device,
                )
                simulation._rigid_initial_xyz = np.asarray(
                    data["rigid_initial_xyz"], dtype=np.float64
                )
                simulation._rigid_initial_distances = np.linalg.norm(
                    simulation._rigid_initial_xyz
                    - simulation._rigid_initial_xyz[0], axis=1
                )
            simulation._build_nnps()
            simulation._push_next = False
            simulation._initialized = True
            return simulation
