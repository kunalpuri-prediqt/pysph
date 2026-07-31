"""Two-level adaptive-resolution smoke support for the Warp WCSPH path.

This module deliberately separates two concerns:

``TwoLevelAdaptiveController``
    A deterministic NumPy oracle for conservative eight-child splitting and
    complete-family merging.  It is intentionally host-orchestrated: the first
    checkpoint validates the particle lifecycle before a later device-pool
    implementation moves allocation and compaction onto the GPU.

``WarpDamBreakSimulation``
    An incremental 3D dam-break runner used by the browser application and the
    adaptive experiment.  Physics steps run on NVIDIA Warp.  In adaptive mode
    the runner rebuilds the fluid ``ParticleArray`` and
    ``MultilevelGridWarpNNPS`` only at explicit adaptation checkpoints.

The equal-mass octant stencil is an engineering smoke fixture, not the
production APR split operator.  The Vacondio/PySPH stencil convention mismatch
recorded by the implementation memory remains unresolved.
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
from pysph.base.warp_sph import wc_sph_dam_break_step
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
        self.next_pid = 1

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
        nchild = split_indices.size * 8
        child_data = {}
        for name, values in state.items():
            values = np.asarray(values)
            child_data[name] = np.repeat(values[split_indices], 8)

        spacing = np.asarray(state["h"])[split_indices] / self.hdx
        offsets = (
            spacing[:, None, None] * 0.25 * _OCTANTS[None, :, :]
        ).reshape(nchild, 3)
        for axis, column in zip(("x", "y", "z"), range(3)):
            child_data[axis] = (
                np.repeat(np.asarray(state[axis])[split_indices], 8)
                + offsets[:, column]
            )
        child_data["m"] = child_data["m"] / 8.0
        child_data["h"] = child_data["h"] / 2.0
        child_data["level"] = np.ones(
            nchild, dtype=np.asarray(state["level"]).dtype
        )
        families = np.repeat(
            np.asarray(state["pid"])[split_indices], 8
        )
        child_data["family_id"] = families.astype(
            np.asarray(state["family_id"]).dtype
        )
        child_data["pid"] = np.arange(
            self.next_pid, self.next_pid + nchild, dtype=np.int32
        )
        self.next_pid += nchild

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
            state["x"], state["y"], state["z"], self.fine_bounds
        )
        candidates = np.flatnonzero((level == 1) & outside)
        if candidates.size == 0:
            return state, 0

        families = np.asarray(state["family_id"]).astype(np.int64)
        selected = []
        family_groups = []
        for family in np.unique(families[candidates]):
            group = np.flatnonzero((level == 1) & (families == family))
            if group.size == 8 and np.all(outside[group]):
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
                    value = 2.0 * float(np.max(values[group]))
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

    def adapt(self, state: Mapping[str, np.ndarray]):
        """Merge complete outside families, then split coarse inside parents."""
        state = self.initialize_state(state)
        mass_before, momentum_before = self.invariants(state)
        particles_before = len(state["x"])
        merged, nmerge = self._merge(state)
        adapted, nsplit = self._split(merged)
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
            created_children=8 * nsplit,
            merged_families=nmerge,
            removed_children=8 * nmerge,
            mass_before=mass_before,
            mass_after=mass_after,
            mass_residual=_relative_residual(
                mass_before, mass_after, abs(mass_before)
            ),
            momentum_residual=float(
                np.linalg.norm(momentum_after - momentum_before)
                / max(float(momentum_scale), 1.0e-30)
            ),
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
    with_obstacle: bool = True
    adapt_every: int = 5
    max_splits_per_adapt: int = 128
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
        if self.cfl <= 0:
            raise ValueError("cfl must be positive")
        return self

    def to_dict(self):
        result = asdict(self)
        result["fine_bounds"] = list(self.fine_bounds)
        return result

    @classmethod
    def from_mapping(cls, values):
        data = dict(values)
        if "fine_bounds" in data:
            data["fine_bounds"] = tuple(data["fine_bounds"])
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
        )
        self.particles = None
        self.nnps = None
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
            with_obstacle=self.config.with_obstacle,
        )

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
        state, stats = self.controller.adapt(state)
        self.particles[0] = warp_particle_array_from_state(
            state, name="fluid", device=self.config.device
        )
        self._build_nnps()
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
        dt_used = wc_sph_dam_break_step(
            self.nnps,
            fluid_index=0,
            solid_indices=tuple(range(1, len(self.particles))),
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
            gravity_ramp=1.0,
            adaptive_dt=self.config.adaptive_dt,
            cfl=self.config.cfl,
            dt_min=self.config.dt_min,
            dt_max=self.dt_max,
            adaptive_dt_scale=scale,
            step_dt_max=self.dt_max,
            push=self._push_next,
            return_dt=True,
            neighbor_mode=self.neighbor_mode,
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
        names = ("fluid", "wall", "obstacle")
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
            "kind": np.concatenate(kinds),
            "level": np.concatenate(levels),
            "counts": {
                names[index]: int(
                    self.particles[index].get_number_of_particles()
                )
                for index in range(len(self.particles))
            },
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
        latest_adapt = (
            self.adaptation_history[-1] if self.adaptation_history else None
        )
        return {
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
            "rho_min": float(np.min(fluid["rho"])),
            "rho_max": float(np.max(fluid["rho"])),
            "p_min": float(np.min(fluid["p"])),
            "p_max": float(np.max(fluid["p"])),
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
            "latest_adaptation": latest_adapt,
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
        """Save the latest visualization state and reproducibility manifest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot(include_solids=True)
        metrics = self.metrics()
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
        )
        return path
