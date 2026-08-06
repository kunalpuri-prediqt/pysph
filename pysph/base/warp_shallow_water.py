"""Experimental Warp finite-volume shallow-water gameplay profile.

This module is intentionally isolated from PySPH's scientific WCSPH stack.
It evolves depth and depth-integrated momentum on a regular metric grid for
regional, depth-averaged visualization.  It is not an inundation forecast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None


if wp is not None:

    @wp.struct
    class _HydroFlux:
        value: wp.vec3
        h_left: wp.float32
        h_right: wp.float32


    @wp.func
    def _sample_state(
            h: wp.array(dtype=wp.float32),
            hu: wp.array(dtype=wp.float32),
            hv: wp.array(dtype=wp.float32),
            nx: int, ny: int, ix: int, iy: int,
            closed_boundary: int):
        cx = wp.max(0, wp.min(nx - 1, ix))
        cy = wp.max(0, wp.min(ny - 1, iy))
        index = cy * nx + cx
        mx = hu[index]
        my = hv[index]
        if closed_boundary != 0:
            if ix < 0 or ix >= nx:
                mx = -mx
            if iy < 0 or iy >= ny:
                my = -my
        return wp.vec3(h[index], mx, my)


    @wp.func
    def _sample_bed(
            bed: wp.array(dtype=wp.float32),
            nx: int, ny: int, ix: int, iy: int):
        cx = wp.max(0, wp.min(nx - 1, ix))
        cy = wp.max(0, wp.min(ny - 1, iy))
        return bed[cy * nx + cx]


    @wp.func
    def _hydrostatic_x_flux(
            left: wp.vec3, z_left: wp.float32,
            right: wp.vec3, z_right: wp.float32,
            gravity: wp.float32, dry: wp.float32):
        result = _HydroFlux()
        h_left = wp.max(left[0], 0.0)
        h_right = wp.max(right[0], 0.0)
        u_left = 0.0
        v_left = 0.0
        u_right = 0.0
        v_right = 0.0
        if h_left > dry:
            u_left = left[1] / h_left
            v_left = left[2] / h_left
        if h_right > dry:
            u_right = right[1] / h_right
            v_right = right[2] / h_right
        z_star = wp.max(z_left, z_right)
        hs_left = wp.max(0.0, h_left + z_left - z_star)
        hs_right = wp.max(0.0, h_right + z_right - z_star)
        q_left = wp.vec3(
            hs_left, hs_left * u_left, hs_left * v_left
        )
        q_right = wp.vec3(
            hs_right, hs_right * u_right, hs_right * v_right
        )
        f_left = wp.vec3(
            q_left[1],
            q_left[1] * u_left + 0.5 * gravity * hs_left * hs_left,
            q_left[1] * v_left,
        )
        f_right = wp.vec3(
            q_right[1],
            q_right[1] * u_right + 0.5 * gravity * hs_right * hs_right,
            q_right[1] * v_right,
        )
        wave_left = wp.abs(u_left) + wp.sqrt(gravity * hs_left)
        wave_right = wp.abs(u_right) + wp.sqrt(gravity * hs_right)
        wave = wp.max(wave_left, wave_right)
        result.value = 0.5 * (f_left + f_right) - 0.5 * wave * (
            q_right - q_left
        )
        result.h_left = hs_left
        result.h_right = hs_right
        return result


    @wp.func
    def _hydrostatic_y_flux(
            lower: wp.vec3, z_lower: wp.float32,
            upper: wp.vec3, z_upper: wp.float32,
            gravity: wp.float32, dry: wp.float32):
        result = _HydroFlux()
        h_lower = wp.max(lower[0], 0.0)
        h_upper = wp.max(upper[0], 0.0)
        u_lower = 0.0
        v_lower = 0.0
        u_upper = 0.0
        v_upper = 0.0
        if h_lower > dry:
            u_lower = lower[1] / h_lower
            v_lower = lower[2] / h_lower
        if h_upper > dry:
            u_upper = upper[1] / h_upper
            v_upper = upper[2] / h_upper
        z_star = wp.max(z_lower, z_upper)
        hs_lower = wp.max(0.0, h_lower + z_lower - z_star)
        hs_upper = wp.max(0.0, h_upper + z_upper - z_star)
        q_lower = wp.vec3(
            hs_lower, hs_lower * u_lower, hs_lower * v_lower
        )
        q_upper = wp.vec3(
            hs_upper, hs_upper * u_upper, hs_upper * v_upper
        )
        f_lower = wp.vec3(
            q_lower[2],
            q_lower[2] * u_lower,
            q_lower[2] * v_lower + 0.5 * gravity * hs_lower * hs_lower,
        )
        f_upper = wp.vec3(
            q_upper[2],
            q_upper[2] * u_upper,
            q_upper[2] * v_upper + 0.5 * gravity * hs_upper * hs_upper,
        )
        wave_lower = wp.abs(v_lower) + wp.sqrt(gravity * hs_lower)
        wave_upper = wp.abs(v_upper) + wp.sqrt(gravity * hs_upper)
        wave = wp.max(wave_lower, wave_upper)
        result.value = 0.5 * (f_lower + f_upper) - 0.5 * wave * (
            q_upper - q_lower
        )
        result.h_left = hs_lower
        result.h_right = hs_upper
        return result


    @wp.kernel
    def _maximum_wave_rate(
            h: wp.array(dtype=wp.float32),
            hu: wp.array(dtype=wp.float32),
            hv: wp.array(dtype=wp.float32),
            gravity: wp.float32, dry: wp.float32,
            inv_dx: wp.float32, inv_dy: wp.float32,
            maximum: wp.array(dtype=wp.float32)):
        index = wp.tid()
        depth = h[index]
        if depth > dry:
            u = hu[index] / depth
            v = hv[index] / depth
            c = wp.sqrt(gravity * depth)
            rate = (wp.abs(u) + c) * inv_dx + (
                wp.abs(v) + c
            ) * inv_dy
            wp.atomic_max(maximum, 0, rate)


    @wp.kernel
    def _finite_volume_step(
            bed: wp.array(dtype=wp.float32),
            h: wp.array(dtype=wp.float32),
            hu: wp.array(dtype=wp.float32),
            hv: wp.array(dtype=wp.float32),
            out_h: wp.array(dtype=wp.float32),
            out_hu: wp.array(dtype=wp.float32),
            out_hv: wp.array(dtype=wp.float32),
            nx: int, ny: int,
            dx: wp.float32, dy: wp.float32, dt: wp.float32,
            gravity: wp.float32, dry: wp.float32,
            manning: wp.float32, closed_boundary: int):
        ix, iy = wp.tid()
        index = iy * nx + ix
        state = wp.vec3(h[index], hu[index], hv[index])
        z = bed[index]

        state_right = _sample_state(
            h, hu, hv, nx, ny, ix + 1, iy, closed_boundary
        )
        z_right = _sample_bed(bed, nx, ny, ix + 1, iy)
        flux_right = _hydrostatic_x_flux(
            state, z, state_right, z_right, gravity, dry
        )

        state_left = _sample_state(
            h, hu, hv, nx, ny, ix - 1, iy, closed_boundary
        )
        z_left = _sample_bed(bed, nx, ny, ix - 1, iy)
        flux_left = _hydrostatic_x_flux(
            state_left, z_left, state, z, gravity, dry
        )

        state_upper = _sample_state(
            h, hu, hv, nx, ny, ix, iy + 1, closed_boundary
        )
        z_upper = _sample_bed(bed, nx, ny, ix, iy + 1)
        flux_upper = _hydrostatic_y_flux(
            state, z, state_upper, z_upper, gravity, dry
        )

        state_lower = _sample_state(
            h, hu, hv, nx, ny, ix, iy - 1, closed_boundary
        )
        z_lower = _sample_bed(bed, nx, ny, ix, iy - 1)
        flux_lower = _hydrostatic_y_flux(
            state_lower, z_lower, state, z, gravity, dry
        )

        source_right = 0.5 * gravity * (
            state[0] * state[0]
            - flux_right.h_left * flux_right.h_left
        )
        source_left = 0.5 * gravity * (
            state[0] * state[0]
            - flux_left.h_right * flux_left.h_right
        )
        source_upper = 0.5 * gravity * (
            state[0] * state[0]
            - flux_upper.h_left * flux_upper.h_left
        )
        source_lower = 0.5 * gravity * (
            state[0] * state[0]
            - flux_lower.h_right * flux_lower.h_right
        )

        depth = state[0] - dt / dx * (
            flux_right.value[0] - flux_left.value[0]
        ) - dt / dy * (
            flux_upper.value[0] - flux_lower.value[0]
        )
        momentum_x = state[1] - dt / dx * (
            flux_right.value[1] + source_right
            - flux_left.value[1] - source_left
        ) - dt / dy * (
            flux_upper.value[1] - flux_lower.value[1]
        )
        momentum_y = state[2] - dt / dx * (
            flux_right.value[2] - flux_left.value[2]
        ) - dt / dy * (
            flux_upper.value[2] + source_upper
            - flux_lower.value[2] - source_lower
        )

        if depth <= 0.0:
            depth = 0.0
            momentum_x = 0.0
            momentum_y = 0.0
        elif depth <= dry:
            # Preserve conservative positive film depth.  It is dynamically
            # dry, so remove momentum without deleting water volume.
            momentum_x = 0.0
            momentum_y = 0.0
        elif manning > 0.0:
            speed = wp.sqrt(
                momentum_x * momentum_x + momentum_y * momentum_y
            ) / depth
            friction = 1.0 + dt * gravity * manning * manning * speed / (
                wp.pow(depth, 1.3333333333)
            )
            momentum_x = momentum_x / friction
            momentum_y = momentum_y / friction

        out_h[index] = depth
        out_hu[index] = momentum_x
        out_hv[index] = momentum_y


@dataclass
class ShallowWaterConfig:
    """Serializable controls for the geospatial shallow-water profile."""

    steps: int = 600
    grid_nx: int = 256
    grid_ny: int = 256
    cell_size_x: float = 30.0
    cell_size_y: float = 30.0
    gravity: float = 9.81
    cfl: float = 0.35
    dt_max: float = 0.5
    dry_depth: float = 0.01
    manning: float = 0.025
    boundary: str = "closed"
    terrain_path: str | None = None
    terrain_id: str = "synthetic-valley-fixture"
    synthetic_breach: bool = True
    device: str | None = None

    def validate(self):
        if self.steps < 1:
            raise ValueError("shallow-water steps must be positive")
        if self.grid_nx < 4 or self.grid_ny < 4:
            raise ValueError("shallow-water grid dimensions must be >= 4")
        if self.cell_size_x <= 0.0 or self.cell_size_y <= 0.0:
            raise ValueError("shallow-water cell sizes must be positive")
        if self.gravity <= 0.0 or not 0.0 < self.cfl <= 0.5:
            raise ValueError("gravity must be positive and CFL in (0, 0.5]")
        if self.dt_max <= 0.0 or self.dry_depth <= 0.0:
            raise ValueError("dt_max and dry_depth must be positive")
        if self.manning < 0.0:
            raise ValueError("Manning friction cannot be negative")
        if self.boundary not in {"closed", "outflow"}:
            raise ValueError("boundary must be 'closed' or 'outflow'")
        return self

    def to_dict(self):
        result = asdict(self)
        result.update({
            "solver_family": "geospatial-swe",
            "approximate": True,
            "depth_averaged": True,
        })
        return result

    @classmethod
    def from_mapping(cls, values: Mapping):
        data = dict(values)
        data.pop("solver_family", None)
        data.pop("approximate", None)
        data.pop("depth_averaged", None)
        allowed = cls.__dataclass_fields__
        return cls(**{
            key: value for key, value in data.items() if key in allowed
        }).validate()


def make_synthetic_valley(nx, ny, dx=30.0, dy=30.0):
    """Return a deterministic valley/breach fixture for tests and bootstrapping."""
    x = np.linspace(0.0, 1.0, int(nx), dtype=np.float32)
    y = np.linspace(-1.0, 1.0, int(ny), dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    raw_bed = (
        38.0 * (1.0 - xx)
        + 54.0 * yy * yy
        + 3.0 * np.sin(4.0 * np.pi * xx) * np.exp(-2.5 * yy * yy)
    ).astype(np.float32)
    bed = raw_bed.copy()
    dam_index = max(2, min(int(nx) - 3, int(round(0.34 * (nx - 1)))))
    breach_half_width = max(1, int(round(0.055 * ny)))
    center = ny // 2
    barrier = np.zeros_like(bed)
    barrier[:, dam_index] = 42.0
    barrier[
        center - breach_half_width:center + breach_half_width + 1,
        dam_index,
    ] = 0.0
    bed += barrier
    reservoir_surface = 58.0
    initial_depth = np.where(
        xx < x[dam_index],
        np.maximum(reservoir_surface - raw_bed, 0.0),
        0.0,
    ).astype(np.float32)
    return {
        "bed": np.ascontiguousarray(bed),
        "raw_bed": np.ascontiguousarray(raw_bed),
        "barrier": np.ascontiguousarray(barrier),
        "initial_depth": np.ascontiguousarray(initial_depth),
        "cell_size_x": float(dx),
        "cell_size_y": float(dy),
        "terrain_id": "synthetic-valley-fixture",
        "synthetic": True,
    }


def load_terrain_asset(path):
    """Load a processed terrain asset without allowing pickle content."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"bed", "initial_depth"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                "terrain asset is missing " + ", ".join(sorted(missing))
            )
        result = {name: np.asarray(data[name]) for name in data.files}
    bed = np.asarray(result["bed"], dtype=np.float32)
    depth = np.asarray(result["initial_depth"], dtype=np.float32)
    if bed.ndim != 2 or depth.shape != bed.shape:
        raise ValueError("terrain bed/depth arrays must be matching 2D grids")
    if not np.all(np.isfinite(bed)) or not np.all(np.isfinite(depth)):
        raise ValueError("terrain asset contains non-finite values")
    if np.min(depth) < 0.0:
        raise ValueError("terrain asset contains negative water depth")
    result["bed"] = np.ascontiguousarray(bed)
    result["initial_depth"] = np.ascontiguousarray(depth)
    return result


class WarpShallowWaterSimulation:
    """Incremental conservative depth-averaged terrain-flow simulation."""

    def __init__(
            self, config: ShallowWaterConfig | Mapping | None = None,
            bed=None, initial_depth=None):
        if wp is None:  # pragma: no cover
            raise ImportError("warp is required for shallow-water simulation")
        if config is None:
            config = ShallowWaterConfig()
        elif not isinstance(config, ShallowWaterConfig):
            config = ShallowWaterConfig.from_mapping(config)
        self.config = config.validate()
        self.device = wp.get_device(self.config.device)
        self._provided_bed = bed
        self._provided_depth = initial_depth
        self.step_count = 0
        self.time = 0.0
        self.dt_history = []
        self.step_wall_history = []
        self.adaptation_history = []
        self._initialized = False
        self._host_cache_step = -1
        self._host_cache = None

    @property
    def done(self):
        return self.step_count >= self.config.steps

    def _terrain(self):
        if self._provided_bed is not None:
            bed = np.asarray(self._provided_bed, dtype=np.float32)
            if self._provided_depth is None:
                depth = np.zeros_like(bed)
            else:
                depth = np.asarray(self._provided_depth, dtype=np.float32)
            return {
                "bed": np.ascontiguousarray(bed),
                "initial_depth": np.ascontiguousarray(depth),
                "terrain_id": self.config.terrain_id,
                "synthetic": self.config.terrain_id.startswith("synthetic"),
            }
        if self.config.terrain_path:
            return load_terrain_asset(self.config.terrain_path)
        return make_synthetic_valley(
            self.config.grid_nx, self.config.grid_ny,
            self.config.cell_size_x, self.config.cell_size_y,
        )

    def initialize(self):
        if self._initialized:
            return self.snapshot()
        terrain = self._terrain()
        terrain_bed = np.asarray(
            terrain.get("raw_bed", terrain["bed"]), dtype=np.float32
        )
        barrier = np.asarray(
            terrain.get("barrier", np.zeros_like(terrain_bed)),
            dtype=np.float32,
        )
        bed = terrain_bed + barrier
        depth = np.asarray(terrain["initial_depth"], dtype=np.float32)
        if (bed.ndim != 2 or bed.shape != depth.shape
                or barrier.shape != bed.shape):
            raise ValueError(
                "terrain, barrier, and initial depth must be matching 2D arrays"
            )
        if (not np.all(np.isfinite(bed))
                or not np.all(np.isfinite(depth))):
            raise ValueError("terrain state must be finite")
        if np.min(depth) < 0.0:
            raise ValueError("initial depth cannot be negative")
        self.ny, self.nx = bed.shape
        self.config.grid_nx = self.nx
        self.config.grid_ny = self.ny
        if "cell_size_x" in terrain:
            self.config.cell_size_x = float(np.asarray(
                terrain["cell_size_x"]
            ).item())
        if "cell_size_y" in terrain:
            self.config.cell_size_y = float(np.asarray(
                terrain["cell_size_y"]
            ).item())
        self.terrain_id = str(terrain.get(
            "terrain_id", self.config.terrain_id
        ))
        self.synthetic_terrain = bool(terrain.get("synthetic", False))
        self.terrain_bed_host = np.ascontiguousarray(terrain_bed)
        self.barrier_host = np.ascontiguousarray(barrier)
        self.bed_host = np.ascontiguousarray(bed)
        self.initial_depth_host = np.ascontiguousarray(depth)
        zeros = np.zeros_like(depth)
        self.bed = wp.array(
            bed.ravel(), dtype=wp.float32, device=self.device
        )
        self.h = wp.array(
            depth.ravel(), dtype=wp.float32, device=self.device
        )
        self.hu = wp.array(
            zeros.ravel(), dtype=wp.float32, device=self.device
        )
        self.hv = wp.array(
            zeros.ravel(), dtype=wp.float32, device=self.device
        )
        count = self.nx * self.ny
        self.next_h = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.next_hu = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.next_hv = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.maximum_rate = wp.zeros(
            1, dtype=wp.float32, device=self.device
        )
        self.initial_volume = float(
            np.sum(depth, dtype=np.float64)
            * self.config.cell_size_x * self.config.cell_size_y
        )
        self._initialized = True
        return self.snapshot()

    def _stable_dt(self):
        self.maximum_rate.zero_()
        wp.launch(
            _maximum_wave_rate,
            dim=self.nx * self.ny,
            inputs=[
                self.h, self.hu, self.hv,
                np.float32(self.config.gravity),
                np.float32(self.config.dry_depth),
                np.float32(1.0 / self.config.cell_size_x),
                np.float32(1.0 / self.config.cell_size_y),
                self.maximum_rate,
            ],
            device=self.device,
        )
        maximum = float(self.maximum_rate.numpy()[0])
        if not np.isfinite(maximum):
            raise OverflowError("non-finite shallow-water wave speed")
        if maximum <= 1.0e-12:
            return float(self.config.dt_max)
        return min(float(self.config.dt_max), self.config.cfl / maximum)

    def step(self):
        if not self._initialized:
            self.initialize()
        if self.done:
            return None
        started = time.perf_counter()
        dt = self._stable_dt()
        wp.launch(
            _finite_volume_step,
            dim=(self.nx, self.ny),
            inputs=[
                self.bed, self.h, self.hu, self.hv,
                self.next_h, self.next_hu, self.next_hv,
                self.nx, self.ny,
                np.float32(self.config.cell_size_x),
                np.float32(self.config.cell_size_y),
                np.float32(dt), np.float32(self.config.gravity),
                np.float32(self.config.dry_depth),
                np.float32(self.config.manning),
                int(self.config.boundary == "closed"),
            ],
            device=self.device,
        )
        self.h, self.next_h = self.next_h, self.h
        self.hu, self.next_hu = self.next_hu, self.hu
        self.hv, self.next_hv = self.next_hv, self.hv
        wp.synchronize_device(self.device)
        self.step_count += 1
        self.time += dt
        self.dt_history.append(dt)
        self.step_wall_history.append(time.perf_counter() - started)
        self._host_cache_step = -1
        return {"step": self.step_count, "time": self.time, "dt": dt}

    def _host_state(self):
        if self._host_cache_step != self.step_count:
            depth = self.h.numpy().reshape(self.ny, self.nx)
            momentum_x = self.hu.numpy().reshape(self.ny, self.nx)
            momentum_y = self.hv.numpy().reshape(self.ny, self.nx)
            speed = np.zeros_like(depth)
            wet = depth > self.config.dry_depth
            speed[wet] = np.sqrt(
                momentum_x[wet] * momentum_x[wet]
                + momentum_y[wet] * momentum_y[wet]
            ) / depth[wet]
            self._host_cache = {
                "depth": np.ascontiguousarray(depth),
                "momentum_x": np.ascontiguousarray(momentum_x),
                "momentum_y": np.ascontiguousarray(momentum_y),
                "speed": np.ascontiguousarray(speed),
            }
            self._host_cache_step = self.step_count
        return self._host_cache

    def snapshot(self):
        state = self._host_state()
        return {
            "step": self.step_count,
            "time": self.time,
            "bed": self.bed_host.copy(),
            "terrain_bed": self.terrain_bed_host.copy(),
            "synthetic_barrier": self.barrier_host.copy(),
            "water_depth": state["depth"].copy(),
            "water_surface": np.ascontiguousarray(
                self.bed_host + state["depth"]
            ),
            "water_speed": state["speed"].copy(),
            "momentum_x": state["momentum_x"].copy(),
            "momentum_y": state["momentum_y"].copy(),
            "solver_family": "geospatial-swe",
            "terrain_id": self.terrain_id,
            "synthetic_breach": self.config.synthetic_breach,
            "grid_shape": [self.ny, self.nx],
            "cell_size": [
                self.config.cell_size_x, self.config.cell_size_y,
            ],
            "dry_depth": self.config.dry_depth,
        }

    def metrics(self):
        state = self._host_state()
        depth = state["depth"]
        speed = state["speed"]
        volume = float(
            np.sum(depth, dtype=np.float64)
            * self.config.cell_size_x * self.config.cell_size_y
        )
        initial = self.initial_volume
        volume_drift = 0.0 if initial == 0.0 else (volume - initial) / initial
        wet = depth > self.config.dry_depth
        history = np.asarray(self.step_wall_history, dtype=np.float64)
        finite = all(np.all(np.isfinite(value)) for value in (
            depth, state["momentum_x"], state["momentum_y"], speed,
        ))
        return {
            "solver_family": "geospatial-swe",
            "approximate": True,
            "depth_averaged": True,
            "synthetic_breach": self.config.synthetic_breach,
            "approximation_warning": (
                "Depth-averaged terrain-flow prototype with a synthetic "
                "breach; not an inundation forecast"
            ),
            "terrain_id": self.terrain_id,
            "synthetic_terrain": self.synthetic_terrain,
            "resolution_mode": "geospatial",
            "step": self.step_count,
            "steps": self.config.steps,
            "time": self.time,
            "dt_last": self.dt_history[-1] if self.dt_history else None,
            "grid_shape": [self.ny, self.nx],
            "cell_size": [
                self.config.cell_size_x, self.config.cell_size_y,
            ],
            "dry_depth": self.config.dry_depth,
            "fluid_particles": 0,
            "fine_particles": 0,
            "coarse_particles": 0,
            "water_volume": volume,
            "initial_water_volume": initial,
            "volume_drift": volume_drift,
            "mass_drift": volume_drift,
            "wet_cells": int(np.count_nonzero(wet)),
            "wet_area": float(
                np.count_nonzero(wet)
                * self.config.cell_size_x * self.config.cell_size_y
            ),
            "peak_depth": float(np.max(depth)),
            "peak_speed": float(np.max(speed)),
            "all_finite": bool(finite and np.min(depth) >= 0.0),
            "solver_frame_median_ms": (
                None if not history.size
                else 1000.0 * float(np.median(history))
            ),
            "solver_frame_p95_ms": (
                None if not history.size
                else 1000.0 * float(np.percentile(history, 95))
            ),
            "simulated_to_wall_ratio": (
                None if not history.size
                else self.time / float(np.sum(history))
            ),
            "adaptation_events": 0,
            "split_parents": 0,
            "merged_families": 0,
            "shifted_particles": 0,
            "max_shift": 0.0,
            "body_particles": 0,
            "body_mass": 0.0,
            "rigid_device_error": 0,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        metrics = self.metrics()
        np.savez_compressed(
            path,
            **{
                key: value for key, value in snapshot.items()
                if isinstance(value, np.ndarray)
            },
            dt_history=np.asarray(self.dt_history),
            metrics=json.dumps(metrics, sort_keys=True),
            config=json.dumps(self.config.to_dict(), sort_keys=True),
        )
        return path
