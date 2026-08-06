"""Fixed-budget GPU gameplay fluid for the Warp dam-break studio.

This module is deliberately separate from the scientific WCSPH path.  It uses
Position-Based Fluids-style density projection, analytic tank constraints and
an approximate impulse-coupled rigid box.  Its contract is visual stability
and measured frame cost, not pressure, energy or hydrostatic convergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import setuptools  # noqa: F401

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from pysph.examples._db_geometry import DamBreak3DGeometry


H = 1.0
GRAVITY = 9.81
TANK_BOUNDS = (0.0, 161.0 / 30.0, -0.25, 0.25, 0.0, 1.5)


if wp is not None:
    @wp.func
    def _poly6(r2: wp.float32, h: wp.float32):
        value = wp.float32(0.0)
        h2 = h * h
        if r2 < h2:
            x = h2 - r2
            pi = wp.float32(3.141592653589793)
            value = wp.float32(315.0) * x * x * x / (
                wp.float32(64.0) * pi * h * h * h * h * h * h * h * h * h
            )
        return value


    @wp.func
    def _spiky_gradient(r: wp.vec3, h: wp.float32):
        grad = wp.vec3(0.0, 0.0, 0.0)
        distance = wp.length(r)
        if distance > wp.float32(1.0e-7) and distance < h:
            pi = wp.float32(3.141592653589793)
            scale = -wp.float32(45.0) * (h - distance) * (h - distance) / (
                pi * h * h * h * h * h * h * distance
            )
            grad = scale * r
        return grad


    @wp.func
    def _clamp_vec_length(value: wp.vec3, limit: wp.float32):
        magnitude = wp.length(value)
        result = value
        if magnitude > limit and magnitude > wp.float32(1.0e-12):
            result = value * (limit / magnitude)
        return result


    @wp.kernel
    def _predict_fluid(
            x: wp.array(dtype=wp.vec3),
            x_prev: wp.array(dtype=wp.vec3),
            velocity: wp.array(dtype=wp.vec3),
            dt: wp.float32,
            gravity_z: wp.float32,
            speed_limit: wp.float32):
        i = wp.tid()
        position = x[i]
        vel = velocity[i] + wp.vec3(0.0, 0.0, gravity_z * dt)
        vel = _clamp_vec_length(vel, speed_limit)
        x_prev[i] = position
        x[i] = position + dt * vel
        velocity[i] = vel


    @wp.kernel
    def _solve_lambdas(
            grid: wp.uint64,
            x: wp.array(dtype=wp.vec3),
            density: wp.array(dtype=wp.float32),
            constraint: wp.array(dtype=wp.float32),
            lambdas: wp.array(dtype=wp.float32),
            mass: wp.float32,
            rest_density: wp.float32,
            support: wp.float32,
            query_radius: wp.float32,
            epsilon: wp.float32):
        i = wp.tid()
        xi = x[i]
        rho = wp.float32(0.0)
        grad_i = wp.vec3(0.0, 0.0, 0.0)
        grad_norm2 = wp.float32(0.0)
        neighbors = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors:
            rij = xi - x[j]
            r2 = wp.dot(rij, rij)
            if r2 < support * support:
                rho += mass * _poly6(r2, support)
                if j != i:
                    grad_j = -(mass / rest_density) * _spiky_gradient(
                        rij, support
                    )
                    grad_i -= grad_j
                    grad_norm2 += wp.dot(grad_j, grad_j)
        c = rho / rest_density - wp.float32(1.0)
        grad_norm2 += wp.dot(grad_i, grad_i)
        density[i] = rho
        constraint[i] = c
        lambdas[i] = -c / (grad_norm2 + epsilon)


    @wp.kernel
    def _solve_position_delta(
            grid: wp.uint64,
            x: wp.array(dtype=wp.vec3),
            lambdas: wp.array(dtype=wp.float32),
            delta: wp.array(dtype=wp.vec3),
            mass: wp.float32,
            rest_density: wp.float32,
            support: wp.float32,
            query_radius: wp.float32,
            artificial_pressure: wp.float32,
            artificial_power: wp.float32):
        i = wp.tid()
        xi = x[i]
        correction = wp.vec3(0.0, 0.0, 0.0)
        q = wp.float32(0.3) * support
        wq = _poly6(q * q, support)
        neighbors = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors:
            if j != i:
                rij = xi - x[j]
                r2 = wp.dot(rij, rij)
                if r2 < support * support:
                    ratio = _poly6(r2, support) / wq
                    scorr = -artificial_pressure * wp.pow(
                        ratio, artificial_power
                    )
                    correction += (
                        lambdas[i] + lambdas[j] + scorr
                    ) * mass * _spiky_gradient(rij, support)
        delta[i] = correction / rest_density


    @wp.kernel
    def _apply_delta_and_tank(
            x: wp.array(dtype=wp.vec3),
            delta: wp.array(dtype=wp.vec3),
            max_correction_seen: wp.array(dtype=wp.float32),
            clamp_count: wp.array(dtype=wp.int32),
            max_correction: wp.float32,
            particle_radius: wp.float32,
            xmin: wp.float32, xmax: wp.float32,
            ymin: wp.float32, ymax: wp.float32,
            zmin: wp.float32, zmax: wp.float32):
        i = wp.tid()
        raw = delta[i]
        raw_length = wp.length(raw)
        wp.atomic_max(max_correction_seen, 0, raw_length)
        applied = raw
        if raw_length > max_correction:
            applied = _clamp_vec_length(raw, max_correction)
            wp.atomic_add(clamp_count, 0, 1)
        position = x[i] + applied
        position = wp.vec3(
            wp.clamp(position[0], xmin + particle_radius,
                     xmax - particle_radius),
            wp.clamp(position[1], ymin + particle_radius,
                     ymax - particle_radius),
            wp.clamp(position[2], zmin + particle_radius,
                     zmax - particle_radius),
        )
        x[i] = position


    @wp.kernel
    def _measure_density(
            grid: wp.uint64,
            x: wp.array(dtype=wp.vec3),
            density: wp.array(dtype=wp.float32),
            constraint: wp.array(dtype=wp.float32),
            mass: wp.float32,
            rest_density: wp.float32,
            support: wp.float32,
            query_radius: wp.float32):
        i = wp.tid()
        xi = x[i]
        rho = wp.float32(0.0)
        neighbors = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors:
            rij = xi - x[j]
            r2 = wp.dot(rij, rij)
            if r2 < support * support:
                rho += mass * _poly6(r2, support)
        density[i] = rho
        constraint[i] = rho / rest_density - wp.float32(1.0)


    @wp.kernel
    def _update_velocity(
            x: wp.array(dtype=wp.vec3),
            x_prev: wp.array(dtype=wp.vec3),
            velocity: wp.array(dtype=wp.vec3),
            dt_inv: wp.float32,
            damping: wp.float32,
            speed_limit: wp.float32):
        i = wp.tid()
        vel = damping * (x[i] - x_prev[i]) * dt_inv
        velocity[i] = _clamp_vec_length(vel, speed_limit)


    @wp.kernel
    def _solve_xsph_delta(
            grid: wp.uint64,
            x: wp.array(dtype=wp.vec3),
            velocity: wp.array(dtype=wp.vec3),
            density: wp.array(dtype=wp.float32),
            delta: wp.array(dtype=wp.vec3),
            mass: wp.float32,
            support: wp.float32,
            query_radius: wp.float32,
            coefficient: wp.float32):
        i = wp.tid()
        xi = x[i]
        vi = velocity[i]
        smoothing = wp.vec3(0.0, 0.0, 0.0)
        neighbors = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors:
            if j != i:
                rij = xi - x[j]
                r2 = wp.dot(rij, rij)
                if r2 < support * support:
                    rhoj = wp.max(density[j], wp.float32(1.0e-6))
                    smoothing += (
                        mass / rhoj * (velocity[j] - vi)
                        * _poly6(r2, support)
                    )
        delta[i] = coefficient * smoothing


    @wp.kernel
    def _apply_xsph_delta(
            velocity: wp.array(dtype=wp.vec3),
            delta: wp.array(dtype=wp.vec3),
            speed_limit: wp.float32):
        i = wp.tid()
        velocity[i] = _clamp_vec_length(
            velocity[i] + delta[i], speed_limit
        )


    @wp.kernel
    def _compute_render_covariance(
            grid: wp.uint64,
            x: wp.array(dtype=wp.vec3),
            center: wp.array(dtype=wp.vec3),
            covariance: wp.array(dtype=wp.mat33),
            neighbor_count: wp.array(dtype=wp.int32),
            support: wp.float32,
            query_radius: wp.float32,
            smoothing: wp.float32):
        i = wp.tid()
        xi = x[i]
        weighted_center = wp.vec3(0.0, 0.0, 0.0)
        weight_sum = wp.float32(0.0)
        count = wp.int32(0)
        neighbors = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors:
            offset = xi - x[j]
            r2 = wp.dot(offset, offset)
            if r2 < support * support:
                weight = _poly6(r2, support)
                weighted_center += weight * x[j]
                weight_sum += weight
                count += wp.int32(1)
        mean = xi
        if weight_sum > wp.float32(1.0e-12):
            mean = weighted_center / weight_sum
        display_center = (wp.float32(1.0) - smoothing) * xi + smoothing * mean

        c00 = wp.float32(0.0)
        c01 = wp.float32(0.0)
        c02 = wp.float32(0.0)
        c11 = wp.float32(0.0)
        c12 = wp.float32(0.0)
        c22 = wp.float32(0.0)
        covariance_weight = wp.float32(0.0)
        neighbors2 = wp.hash_grid_query(grid, xi, query_radius)
        for j in neighbors2:
            offset = xi - x[j]
            r2 = wp.dot(offset, offset)
            if r2 < support * support:
                weight = _poly6(r2, support)
                d = x[j] - mean
                c00 += weight * d[0] * d[0]
                c01 += weight * d[0] * d[1]
                c02 += weight * d[0] * d[2]
                c11 += weight * d[1] * d[1]
                c12 += weight * d[1] * d[2]
                c22 += weight * d[2] * d[2]
                covariance_weight += weight
        if covariance_weight > wp.float32(1.0e-12):
            inv_weight = wp.float32(1.0) / covariance_weight
            c00 *= inv_weight
            c01 *= inv_weight
            c02 *= inv_weight
            c11 *= inv_weight
            c12 *= inv_weight
            c22 *= inv_weight
        center[i] = display_center
        covariance[i] = wp.mat33(
            c00, c01, c02,
            c01, c11, c12,
            c02, c12, c22,
        )
        neighbor_count[i] = count


    @wp.kernel
    def _solve_render_anisotropy(
            covariance: wp.array(dtype=wp.mat33),
            neighbor_count: wp.array(dtype=wp.int32),
            axes: wp.array(dtype=wp.mat33),
            scale: wp.array(dtype=wp.vec3),
            minimum_neighbors: wp.int32,
            base_radius: wp.float32,
            minimum_ratio: wp.float32,
            maximum_ratio: wp.float32):
        i = wp.tid()
        frame = wp.identity(n=3, dtype=wp.float32)
        ellipsoid_scale = wp.vec3(base_radius, base_radius, base_radius)
        if neighbor_count[i] >= minimum_neighbors:
            eigenvalues = wp.vec3(0.0, 0.0, 0.0)
            wp.eig3(covariance[i], frame, eigenvalues)
            floor = base_radius * base_radius * wp.float32(1.0e-4)
            e0 = wp.max(eigenvalues[0], floor)
            e1 = wp.max(eigenvalues[1], floor)
            e2 = wp.max(eigenvalues[2], floor)
            mean_eigenvalue = (e0 + e1 + e2) / wp.float32(3.0)
            r0 = wp.sqrt(e0 / mean_eigenvalue)
            r1 = wp.sqrt(e1 / mean_eigenvalue)
            r2 = wp.sqrt(e2 / mean_eigenvalue)
            r0 = wp.clamp(r0, minimum_ratio, maximum_ratio)
            r1 = wp.clamp(r1, minimum_ratio, maximum_ratio)
            r2 = wp.clamp(r2, minimum_ratio, maximum_ratio)
            volume_scale = wp.pow(
                wp.max(r0 * r1 * r2, wp.float32(1.0e-8)),
                -wp.float32(1.0) / wp.float32(3.0),
            )
            r0 = wp.clamp(
                r0 * volume_scale, minimum_ratio, maximum_ratio
            )
            r1 = wp.clamp(
                r1 * volume_scale, minimum_ratio, maximum_ratio
            )
            r2 = wp.clamp(
                r2 * volume_scale, minimum_ratio, maximum_ratio
            )
            if wp.determinant(frame) < wp.float32(0.0):
                frame = wp.mat33(
                    frame[0, 0], frame[0, 1], -frame[0, 2],
                    frame[1, 0], frame[1, 1], -frame[1, 2],
                    frame[2, 0], frame[2, 1], -frame[2, 2],
                )
            ellipsoid_scale = base_radius * wp.vec3(r0, r1, r2)
        axes[i] = frame
        scale[i] = ellipsoid_scale


    @wp.kernel
    def _reset_diagnostics(
            max_correction_seen: wp.array(dtype=wp.float32),
            clamp_count: wp.array(dtype=wp.int32),
            impulse: wp.array(dtype=wp.float64),
            torque: wp.array(dtype=wp.float64)):
        max_correction_seen[0] = wp.float32(0.0)
        clamp_count[0] = wp.int32(0)
        for axis in range(3):
            impulse[axis] = wp.float64(0.0)
            torque[axis] = wp.float64(0.0)


    @wp.kernel
    def _predict_body(
            cm: wp.array(dtype=wp.vec3d),
            velocity: wp.array(dtype=wp.vec3d),
            omega: wp.array(dtype=wp.vec3d),
            orientation: wp.array(dtype=wp.quatd),
            half_size: wp.vec3d,
            dt: wp.float64,
            gravity_z: wp.float64,
            linear_damping: wp.float64,
            angular_damping: wp.float64,
            xmin: wp.float64, xmax: wp.float64,
            ymin: wp.float64, ymax: wp.float64,
            zmin: wp.float64, zmax: wp.float64):
        center = cm[0]
        vel = linear_damping * (
            velocity[0] + wp.vec3d(0.0, 0.0, gravity_z * dt)
        )
        spin = angular_damping * omega[0]
        center += dt * vel
        angle = wp.length(spin) * dt
        q = orientation[0]
        if angle > wp.float64(1.0e-12):
            dq = wp.quat_from_axis_angle(spin / wp.length(spin), angle)
            q = wp.normalize(dq * q)
        axis_x = wp.quat_rotate(
            q, wp.vec3d(half_size[0], 0.0, 0.0)
        )
        axis_y = wp.quat_rotate(
            q, wp.vec3d(0.0, half_size[1], 0.0)
        )
        axis_z = wp.quat_rotate(
            q, wp.vec3d(0.0, 0.0, half_size[2])
        )
        support = wp.vec3d(
            wp.abs(axis_x[0]) + wp.abs(axis_y[0]) + wp.abs(axis_z[0]),
            wp.abs(axis_x[1]) + wp.abs(axis_y[1]) + wp.abs(axis_z[1]),
            wp.abs(axis_x[2]) + wp.abs(axis_y[2]) + wp.abs(axis_z[2]),
        )
        center = wp.vec3d(
            wp.clamp(center[0], xmin + support[0], xmax - support[0]),
            wp.clamp(center[1], ymin + support[1], ymax - support[1]),
            wp.clamp(center[2], zmin + support[2], zmax - support[2]),
        )
        if (center[2] <= zmin + support[2] + wp.float64(1.0e-8)
                and vel[2] < 0.0):
            vel = wp.vec3d(
                vel[0] * wp.float64(0.8),
                vel[1] * wp.float64(0.8),
                -vel[2] * wp.float64(0.25),
            )
        cm[0] = center
        velocity[0] = vel
        omega[0] = spin
        orientation[0] = q


    @wp.kernel
    def _project_body(
            x: wp.array(dtype=wp.vec3),
            cm: wp.array(dtype=wp.vec3d),
            orientation: wp.array(dtype=wp.quatd),
            impulse: wp.array(dtype=wp.float64),
            torque: wp.array(dtype=wp.float64),
            half_size: wp.vec3d,
            particle_radius: wp.float64,
            particle_mass: wp.float64,
            dt_inv: wp.float64,
            correction_limit: wp.float64,
            clamp_count: wp.array(dtype=wp.int32)):
        i = wp.tid()
        q = orientation[0]
        center = cm[0]
        world = wp.vec3d(
            wp.float64(x[i][0]), wp.float64(x[i][1]), wp.float64(x[i][2])
        )
        local = wp.quat_rotate_inv(q, world - center)
        extent = half_size + wp.vec3d(
            particle_radius, particle_radius, particle_radius
        )
        if (wp.abs(local[0]) < extent[0]
                and wp.abs(local[1]) < extent[1]
                and wp.abs(local[2]) < extent[2]):
            px = extent[0] - wp.abs(local[0])
            py = extent[1] - wp.abs(local[1])
            pz = extent[2] - wp.abs(local[2])
            correction_local = wp.vec3d(0.0, 0.0, 0.0)
            if px <= py and px <= pz:
                correction_local = wp.vec3d(wp.sign(local[0]) * px, 0.0, 0.0)
            elif py <= pz:
                correction_local = wp.vec3d(0.0, wp.sign(local[1]) * py, 0.0)
            else:
                correction_local = wp.vec3d(0.0, 0.0, wp.sign(local[2]) * pz)
            magnitude = wp.length(correction_local)
            if magnitude > correction_limit:
                correction_local *= correction_limit / magnitude
                wp.atomic_add(clamp_count, 0, 1)
            correction_world = wp.quat_rotate(q, correction_local)
            corrected = world + correction_world
            x[i] = wp.vec3(
                wp.float32(corrected[0]), wp.float32(corrected[1]),
                wp.float32(corrected[2])
            )
            particle_impulse = -particle_mass * dt_inv * correction_world
            lever = world - center
            particle_torque = wp.cross(lever, particle_impulse)
            for axis in range(3):
                wp.atomic_add(impulse, axis, particle_impulse[axis])
                wp.atomic_add(torque, axis, particle_torque[axis])


    @wp.kernel
    def _apply_body_impulse(
            cm: wp.array(dtype=wp.vec3d),
            velocity: wp.array(dtype=wp.vec3d),
            omega: wp.array(dtype=wp.vec3d),
            impulse: wp.array(dtype=wp.float64),
            torque: wp.array(dtype=wp.float64),
            inv_mass: wp.float64,
            inv_inertia: wp.vec3d,
            impulse_limit: wp.float64,
            angular_impulse_limit: wp.float64,
            angular_speed_limit: wp.float64,
            clamp_count: wp.array(dtype=wp.int32)):
        linear_impulse = wp.vec3d(impulse[0], impulse[1], impulse[2])
        magnitude = wp.length(linear_impulse)
        if magnitude > impulse_limit and magnitude > wp.float64(1.0e-12):
            linear_impulse *= impulse_limit / magnitude
            wp.atomic_add(clamp_count, 0, 1)
        angular_impulse = wp.vec3d(torque[0], torque[1], torque[2])
        angular_magnitude = wp.length(angular_impulse)
        if (angular_magnitude > angular_impulse_limit
                and angular_magnitude > wp.float64(1.0e-12)):
            angular_impulse *= angular_impulse_limit / angular_magnitude
            wp.atomic_add(clamp_count, 0, 1)
        velocity[0] += inv_mass * linear_impulse
        spin = omega[0] + wp.cw_mul(inv_inertia, angular_impulse)
        spin_magnitude = wp.length(spin)
        if (spin_magnitude > angular_speed_limit
                and spin_magnitude > wp.float64(1.0e-12)):
            spin *= angular_speed_limit / spin_magnitude
            wp.atomic_add(clamp_count, 0, 1)
        omega[0] = spin
        for axis in range(3):
            impulse[axis] = linear_impulse[axis]
            torque[axis] = angular_impulse[axis]


    @wp.kernel
    def _transform_body_shell(
            reference: wp.array(dtype=wp.vec3d),
            cm: wp.array(dtype=wp.vec3d),
            velocity: wp.array(dtype=wp.vec3d),
            omega: wp.array(dtype=wp.vec3d),
            orientation: wp.array(dtype=wp.quatd),
            out_x: wp.array(dtype=wp.vec3),
            out_v: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        rotated = wp.quat_rotate(orientation[0], reference[i])
        world = cm[0] + rotated
        vel = velocity[0] + wp.cross(omega[0], rotated)
        out_x[i] = wp.vec3(
            wp.float32(world[0]), wp.float32(world[1]), wp.float32(world[2])
        )
        out_v[i] = wp.vec3(
            wp.float32(vel[0]), wp.float32(vel[1]), wp.float32(vel[2])
        )


@dataclass
class GameplayDamBreakConfig:
    """Serializable controls for the explicitly approximate gameplay mode."""

    steps: int = 240
    dx: float = 0.1
    hdx: float = 2.0
    dt: float = 1.0 / 60.0
    substeps: int = 1
    projection_iterations: int = 3
    rest_density: float = 1000.0
    lambda_epsilon: float = 1.0e-6
    artificial_pressure: float = 0.001
    artificial_power: float = 4.0
    velocity_damping: float = 0.997
    xsph_coefficient: float = 0.01
    render_smoothing: float = 0.90
    render_splat_scale: float = 1.25
    render_minimum_neighbors: int = 6
    render_anisotropy_min: float = 0.50
    render_anisotropy_max: float = 2.00
    speed_limit: float = 12.0
    correction_fraction: float = 0.20
    neighbor_skin_fraction: float = 1.0
    gz: float = -GRAVITY
    obstacle_mode: str = "floating"
    body_density: float = 500.0
    body_center_x: float = 2.35
    body_center_y: float = 0.0
    body_center_z: float = 0.10
    body_length: float = 0.32
    body_width: float = 0.28
    body_height: float = 0.20
    body_linear_damping: float = 0.995
    body_angular_damping: float = 0.99
    body_impulse_limit: float = 1.5
    body_angular_impulse_limit: float = 0.2
    body_angular_speed_limit: float = 12.0
    device: str | None = None

    def validate(self):
        if self.steps < 1 or self.dx <= 0.0 or self.hdx <= 1.0:
            raise ValueError("gameplay steps/dx/hdx are invalid")
        if self.dt <= 0.0 or self.substeps < 1:
            raise ValueError("gameplay timestep/substeps are invalid")
        if self.projection_iterations < 1 or self.projection_iterations > 12:
            raise ValueError("projection_iterations must be in [1, 12]")
        if self.rest_density <= 0.0 or self.lambda_epsilon <= 0.0:
            raise ValueError("density controls must be positive")
        if not 0.0 < self.velocity_damping <= 1.0:
            raise ValueError("velocity_damping must be in (0, 1]")
        if not 0.0 <= self.xsph_coefficient <= 1.0:
            raise ValueError("xsph_coefficient must be in [0, 1]")
        if not 0.0 <= self.render_smoothing <= 1.0:
            raise ValueError("render_smoothing must be in [0, 1]")
        if self.render_splat_scale <= 0.0:
            raise ValueError("render_splat_scale must be positive")
        if self.render_minimum_neighbors < 1:
            raise ValueError("render_minimum_neighbors must be positive")
        if (self.render_anisotropy_min <= 0.0
                or self.render_anisotropy_max < 1.0
                or self.render_anisotropy_min > 1.0
                or self.render_anisotropy_min >= self.render_anisotropy_max):
            raise ValueError("render anisotropy bounds are invalid")
        if self.speed_limit <= 0.0 or self.correction_fraction <= 0.0:
            raise ValueError("gameplay limits must be positive")
        if (self.projection_iterations * self.correction_fraction
                > self.neighbor_skin_fraction + 1.0e-12):
            raise ValueError(
                "projection correction allowance exceeds the neighbor skin"
            )
        if self.obstacle_mode not in {"none", "fixed", "floating"}:
            raise ValueError("unsupported gameplay obstacle mode")
        if self.body_density <= 0.0:
            raise ValueError("body density must be positive")
        if (self.body_impulse_limit <= 0.0
                or self.body_angular_impulse_limit <= 0.0
                or self.body_angular_speed_limit <= 0.0):
            raise ValueError("body impulse and speed limits must be positive")
        if min(self.body_length, self.body_width, self.body_height) <= 0.0:
            raise ValueError("body dimensions must be positive")
        return self

    def to_dict(self):
        result = asdict(self)
        result["solver_family"] = "gameplay-pbf"
        result["approximate"] = True
        return result

    @classmethod
    def from_mapping(cls, values: Mapping):
        data = dict(values)
        data.pop("solver_family", None)
        data.pop("approximate", None)
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in allowed}).validate()


class WarpGameplayDamBreakSimulation:
    """Incremental fixed-budget PBF gameplay simulation."""

    def __init__(self, config: GameplayDamBreakConfig | Mapping | None = None):
        if wp is None:  # pragma: no cover
            raise ImportError("warp is required for gameplay simulation")
        if config is None:
            config = GameplayDamBreakConfig()
        elif not isinstance(config, GameplayDamBreakConfig):
            config = GameplayDamBreakConfig.from_mapping(config)
        self.config = config.validate()
        self.device = wp.get_device(self.config.device)
        self.support = np.float32(self.config.hdx * self.config.dx)
        self.skin = np.float32(
            self.config.neighbor_skin_fraction * self.config.dx
        )
        self.query_radius = np.float32(self.support + self.skin)
        self.particle_radius = np.float32(0.45 * self.config.dx)
        self.mass = np.float32(self.config.rest_density * self.config.dx ** 3)
        self.step_count = 0
        self.time = 0.0
        self.dt_history = []
        self.step_wall_history = []
        self.render_attribute_wall_history = []
        self.adaptation_history = []
        self._initialized = False
        self._host_cache_step = -1
        self._host_cache = None

    @property
    def done(self):
        return self.step_count >= self.config.steps

    def _geometry(self):
        return DamBreak3DGeometry(
            container_height=1.5 * H,
            container_width=H / 2.0,
            container_length=161 * H / 30.0,
            fluid_column_height=H,
            fluid_column_width=H / 2.0,
            fluid_column_length=2.0 * H,
            dx=self.config.dx,
            nboundary_layers=1,
            hdx=1.3,
            rho0=self.config.rest_density,
            with_obstacle=self.config.obstacle_mode == "fixed",
            obstacle_center_x=3.0,
        )

    def _body_shell(self):
        size = np.asarray([
            self.config.body_length, self.config.body_width,
            self.config.body_height,
        ], dtype=np.float64)
        axes = []
        for length in size:
            count = max(3, int(round(float(length) / self.config.dx)) + 1)
            axes.append(np.linspace(-0.5 * length, 0.5 * length, count))
        ix, iy, iz = np.meshgrid(
            np.arange(len(axes[0])), np.arange(len(axes[1])),
            np.arange(len(axes[2])), indexing="ij",
        )
        shell = (
            (ix == 0) | (ix == len(axes[0]) - 1)
            | (iy == 0) | (iy == len(axes[1]) - 1)
            | (iz == 0) | (iz == len(axes[2]) - 1)
        )
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        return np.column_stack((xx[shell], yy[shell], zz[shell]))

    def initialize(self):
        if self._initialized:
            return self.snapshot()
        particles = self._geometry().create_particles()
        fluid = particles[0]
        fluid_xyz = np.column_stack((fluid.x, fluid.y, fluid.z)).astype(np.float32)
        n = len(fluid_xyz)
        self.fluid_count = n
        self.wall_xyz = np.column_stack((
            particles[1].x, particles[1].y, particles[1].z,
        )).astype(np.float32)
        self.wall_h = np.full(len(self.wall_xyz), self.config.dx, dtype=np.float32)
        if self.config.obstacle_mode == "fixed" and len(particles) > 2:
            self.fixed_obstacle_xyz = np.column_stack((
                particles[2].x, particles[2].y, particles[2].z,
            )).astype(np.float32)
        else:
            self.fixed_obstacle_xyz = np.empty((0, 3), dtype=np.float32)

        zeros3 = np.zeros_like(fluid_xyz)
        self.x = wp.array(fluid_xyz, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.array(fluid_xyz, dtype=wp.vec3, device=self.device)
        self.velocity = wp.array(zeros3, dtype=wp.vec3, device=self.device)
        self.delta = wp.array(zeros3, dtype=wp.vec3, device=self.device)
        self.density = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.constraint = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.lambdas = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.render_center = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.render_covariance = wp.zeros(
            n, dtype=wp.mat33, device=self.device
        )
        self.render_axes = wp.zeros(n, dtype=wp.mat33, device=self.device)
        self.render_scale = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.render_neighbor_count = wp.zeros(
            n, dtype=wp.int32, device=self.device
        )
        self.max_correction_seen = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.clamp_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        grid_dim = max(32, int(math.ceil(n ** (1.0 / 3.0))) * 4)
        self.grid = wp.HashGrid(
            grid_dim, grid_dim, grid_dim, device=self.device, dtype=wp.float32
        )

        shell = self._body_shell()
        self.body_reference_host = shell.copy()
        self.body_reference = wp.array(shell, dtype=wp.vec3d, device=self.device)
        self.body_x = wp.zeros(len(shell), dtype=wp.vec3, device=self.device)
        self.body_v = wp.zeros(len(shell), dtype=wp.vec3, device=self.device)
        center = np.asarray([[
            self.config.body_center_x, self.config.body_center_y,
            self.config.body_center_z,
        ]], dtype=np.float64)
        self.body_cm = wp.array(center, dtype=wp.vec3d, device=self.device)
        self.body_velocity = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.body_omega = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.body_orientation = wp.array(
            [wp.quatd(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatd,
            device=self.device,
        )
        self.body_impulse = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.body_torque = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.body_half_size = np.asarray([
            0.5 * self.config.body_length, 0.5 * self.config.body_width,
            0.5 * self.config.body_height,
        ], dtype=np.float64)
        self.body_mass = self.config.body_density * (
            self.config.body_length * self.config.body_width
            * self.config.body_height
        )
        sx, sy, sz = (
            self.config.body_length, self.config.body_width,
            self.config.body_height,
        )
        inertia = self.body_mass * np.asarray([
            (sy * sy + sz * sz) / 12.0,
            (sx * sx + sz * sz) / 12.0,
            (sx * sx + sy * sy) / 12.0,
        ])
        self.body_inv_inertia = 1.0 / inertia
        self._update_body_shell()

        self.grid.build(self.x, self.query_radius)
        wp.launch(
            _measure_density, dim=n,
            inputs=[
                self.grid.id, self.x, self.density, self.constraint,
                self.mass, np.float32(self.config.rest_density), self.support,
                self.query_radius,
            ], device=self.device,
        )
        wp.synchronize_device(self.device)
        self._initialized = True
        return self.snapshot()

    def _update_body_shell(self):
        wp.launch(
            _transform_body_shell, dim=len(self.body_reference_host),
            inputs=[
                self.body_reference, self.body_cm, self.body_velocity,
                self.body_omega, self.body_orientation, self.body_x,
                self.body_v,
            ], device=self.device,
        )

    def _substep(self, dt):
        bounds = tuple(np.float64(x) for x in TANK_BOUNDS)
        wp.launch(
            _reset_diagnostics, dim=1,
            inputs=[
                self.max_correction_seen, self.clamp_count,
                self.body_impulse, self.body_torque,
            ], device=self.device,
        )
        if self.config.obstacle_mode == "floating":
            wp.launch(
                _predict_body, dim=1,
                inputs=[
                    self.body_cm, self.body_velocity, self.body_omega,
                    self.body_orientation, wp.vec3d(*self.body_half_size),
                    np.float64(dt), np.float64(self.config.gz),
                    np.float64(self.config.body_linear_damping),
                    np.float64(self.config.body_angular_damping), *bounds,
                ], device=self.device,
            )
        wp.launch(
            _predict_fluid, dim=self.fluid_count,
            inputs=[
                self.x, self.x_prev, self.velocity, np.float32(dt),
                np.float32(self.config.gz), np.float32(self.config.speed_limit),
            ], device=self.device,
        )
        self.grid.build(self.x, self.query_radius)
        for _ in range(self.config.projection_iterations):
            wp.launch(
                _solve_lambdas, dim=self.fluid_count,
                inputs=[
                    self.grid.id, self.x, self.density, self.constraint,
                    self.lambdas, self.mass,
                    np.float32(self.config.rest_density), self.support,
                    self.query_radius, np.float32(self.config.lambda_epsilon),
                ], device=self.device,
            )
            wp.launch(
                _solve_position_delta, dim=self.fluid_count,
                inputs=[
                    self.grid.id, self.x, self.lambdas, self.delta, self.mass,
                    np.float32(self.config.rest_density), self.support,
                    self.query_radius,
                    np.float32(self.config.artificial_pressure),
                    np.float32(self.config.artificial_power),
                ], device=self.device,
            )
            wp.launch(
                _apply_delta_and_tank, dim=self.fluid_count,
                inputs=[
                    self.x, self.delta, self.max_correction_seen,
                    self.clamp_count,
                    np.float32(self.config.correction_fraction * self.config.dx),
                    self.particle_radius,
                    *[np.float32(x) for x in TANK_BOUNDS],
                ], device=self.device,
            )
            if self.config.obstacle_mode == "floating":
                wp.launch(
                    _project_body, dim=self.fluid_count,
                    inputs=[
                        self.x, self.body_cm, self.body_orientation,
                        self.body_impulse, self.body_torque,
                        wp.vec3d(*self.body_half_size),
                        np.float64(self.particle_radius),
                        np.float64(self.mass), np.float64(1.0 / dt),
                        np.float64(
                            self.config.correction_fraction * self.config.dx
                        ), self.clamp_count,
                    ], device=self.device,
                )
        self.grid.build(self.x, self.query_radius)
        wp.launch(
            _measure_density, dim=self.fluid_count,
            inputs=[
                self.grid.id, self.x, self.density, self.constraint,
                self.mass, np.float32(self.config.rest_density), self.support,
                self.query_radius,
            ], device=self.device,
        )
        wp.launch(
            _update_velocity, dim=self.fluid_count,
            inputs=[
                self.x, self.x_prev, self.velocity, np.float32(1.0 / dt),
                np.float32(self.config.velocity_damping),
                np.float32(self.config.speed_limit),
            ], device=self.device,
        )
        if self.config.xsph_coefficient > 0.0:
            wp.launch(
                _solve_xsph_delta, dim=self.fluid_count,
                inputs=[
                    self.grid.id, self.x, self.velocity, self.density,
                    self.delta, self.mass, self.support, self.query_radius,
                    np.float32(self.config.xsph_coefficient),
                ], device=self.device,
            )
            wp.launch(
                _apply_xsph_delta, dim=self.fluid_count,
                inputs=[
                    self.velocity, self.delta,
                    np.float32(self.config.speed_limit),
                ], device=self.device,
            )
        if self.config.obstacle_mode == "floating":
            wp.launch(
                _apply_body_impulse, dim=1,
                inputs=[
                    self.body_cm, self.body_velocity, self.body_omega,
                    self.body_impulse, self.body_torque,
                    np.float64(1.0 / self.body_mass),
                    wp.vec3d(*self.body_inv_inertia),
                    np.float64(self.config.body_impulse_limit),
                    np.float64(self.config.body_angular_impulse_limit),
                    np.float64(self.config.body_angular_speed_limit),
                    self.clamp_count,
                ], device=self.device,
            )
        self._update_body_shell()

    def step(self):
        if not self._initialized:
            self.initialize()
        if self.done:
            return None
        started = time.perf_counter()
        sub_dt = self.config.dt / self.config.substeps
        for _ in range(self.config.substeps):
            self._substep(sub_dt)
        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - started
        self.step_wall_history.append(elapsed)
        self.dt_history.append(self.config.dt)
        self.time += self.config.dt
        self.step_count += 1
        self._host_cache_step = -1
        return {
            "step": self.step_count,
            "time": self.time,
            "dt": self.config.dt,
            "adaptation": None,
            "solver_wall_seconds": elapsed,
        }

    def _host_state(self):
        if self._host_cache_step == self.step_count and self._host_cache is not None:
            return self._host_cache
        fluid_x = self.x.numpy().astype(np.float32, copy=False)
        fluid_v = self.velocity.numpy().astype(np.float32, copy=False)
        density = self.density.numpy().astype(np.float32, copy=False)
        constraint = self.constraint.numpy().astype(np.float32, copy=False)
        body_x = self.body_x.numpy().astype(np.float32, copy=False)
        body_v = self.body_v.numpy().astype(np.float32, copy=False)
        state = {
            "fluid_x": fluid_x,
            "fluid_v": fluid_v,
            "density": density,
            "constraint": constraint,
            "body_x": body_x,
            "body_v": body_v,
            "body_cm": self.body_cm.numpy()[0],
            "body_velocity": self.body_velocity.numpy()[0],
            "body_omega": self.body_omega.numpy()[0],
            "body_orientation": self.body_orientation.numpy()[0],
            "max_correction": float(self.max_correction_seen.numpy()[0]),
            "clamp_count": int(self.clamp_count.numpy()[0]),
        }
        self._host_cache_step = self.step_count
        self._host_cache = state
        return state

    def render_attributes(self):
        """Return display-only smoothed ellipsoids without mutating PBF state."""
        if not self._initialized:
            self.initialize()
        started = time.perf_counter()
        self.grid.build(self.x, self.query_radius)
        wp.launch(
            _compute_render_covariance, dim=self.fluid_count,
            inputs=[
                self.grid.id, self.x, self.render_center,
                self.render_covariance, self.render_neighbor_count,
                self.support, self.query_radius,
                np.float32(self.config.render_smoothing),
            ], device=self.device,
        )
        wp.launch(
            _solve_render_anisotropy, dim=self.fluid_count,
            inputs=[
                self.render_covariance, self.render_neighbor_count,
                self.render_axes, self.render_scale,
                np.int32(self.config.render_minimum_neighbors),
                np.float32(
                    self.config.render_splat_scale * self.config.dx
                ),
                np.float32(self.config.render_anisotropy_min),
                np.float32(self.config.render_anisotropy_max),
            ], device=self.device,
        )
        result = {
            "render_xyz": self.render_center.numpy().astype(
                np.float32, copy=False
            ),
            "render_axes": self.render_axes.numpy().astype(
                np.float32, copy=False
            ),
            "render_scale": self.render_scale.numpy().astype(
                np.float32, copy=False
            ),
            "render_neighbors": self.render_neighbor_count.numpy().astype(
                np.int32, copy=False
            ),
        }
        self.render_attribute_wall_history.append(
            time.perf_counter() - started
        )
        return result

    def snapshot(self, include_solids=True):
        state = self._host_state()
        render = self.render_attributes()
        arrays = [state["fluid_x"]]
        velocity = [state["fluid_v"]]
        h = [np.full(self.fluid_count, self.support, dtype=np.float32)]
        rho = [state["density"]]
        p = [np.zeros(self.fluid_count, dtype=np.float32)]
        kind = [np.zeros(self.fluid_count, dtype=np.uint8)]
        level = [np.zeros(self.fluid_count, dtype=np.uint8)]
        counts = {"fluid": self.fluid_count}
        if include_solids:
            arrays.append(self.wall_xyz)
            velocity.append(np.zeros_like(self.wall_xyz))
            h.append(self.wall_h)
            rho.append(np.full(len(self.wall_xyz), self.config.rest_density, dtype=np.float32))
            p.append(np.zeros(len(self.wall_xyz), dtype=np.float32))
            kind.append(np.ones(len(self.wall_xyz), dtype=np.uint8))
            level.append(np.full(len(self.wall_xyz), 2, dtype=np.uint8))
            counts["wall"] = len(self.wall_xyz)
            if self.config.obstacle_mode == "floating":
                obstacle_x = state["body_x"]
                obstacle_v = state["body_v"]
            else:
                obstacle_x = self.fixed_obstacle_xyz
                obstacle_v = np.zeros_like(obstacle_x)
            if len(obstacle_x):
                arrays.append(obstacle_x)
                velocity.append(obstacle_v)
                h.append(np.full(len(obstacle_x), self.config.dx, dtype=np.float32))
                rho.append(np.full(len(obstacle_x), self.config.rest_density, dtype=np.float32))
                p.append(np.zeros(len(obstacle_x), dtype=np.float32))
                kind.append(np.full(len(obstacle_x), 2, dtype=np.uint8))
                level.append(np.full(len(obstacle_x), 2, dtype=np.uint8))
                counts["body" if self.config.obstacle_mode == "floating" else "obstacle"] = len(obstacle_x)
        packed_x = np.concatenate(arrays)
        packed_v = np.concatenate(velocity)
        snapshot = {
            "step": self.step_count,
            "time": self.time,
            "xyz": np.ascontiguousarray(packed_x),
            "h": np.concatenate(h),
            "rho": np.concatenate(rho),
            "p": np.concatenate(p),
            "speed": np.linalg.norm(packed_v, axis=1).astype(np.float32),
            "velocity": np.ascontiguousarray(packed_v),
            "kind": np.concatenate(kind),
            "level": np.concatenate(level),
            "counts": counts,
            "obstacle_mode": self.config.obstacle_mode,
            "solver_family": "gameplay-pbf",
        }
        snapshot.update(render)
        return snapshot

    def metrics(self):
        state = self._host_state()
        density = state["density"]
        constraint = state["constraint"]
        velocity = state["fluid_v"].astype(np.float64)
        finite = all(np.all(np.isfinite(value)) for value in (
            state["fluid_x"], velocity, density, constraint,
            state["body_x"], state["body_velocity"], state["body_omega"],
        ))
        history = np.asarray(self.step_wall_history, dtype=np.float64)
        median = float(np.median(history)) if history.size else None
        p95 = float(np.percentile(history, 95)) if history.size else None
        render_history = np.asarray(
            self.render_attribute_wall_history, dtype=np.float64
        )
        mass = float(self.mass) * self.fluid_count
        body_cm = state["body_cm"]
        body_v = state["body_velocity"]
        body_omega = state["body_omega"]
        return {
            "solver_family": "gameplay-pbf",
            "approximate": True,
            "approximation_warning": (
                "Position-based gameplay fluid; not scientific WCSPH output"
            ),
            "obstacle_mode": self.config.obstacle_mode,
            "resolution_mode": "gameplay",
            "step": self.step_count,
            "steps": self.config.steps,
            "time": self.time,
            "dt_last": self.dt_history[-1] if self.dt_history else None,
            "fluid_particles": self.fluid_count,
            "fine_particles": 0,
            "coarse_particles": self.fluid_count,
            "wall_particles": len(self.wall_xyz),
            "obstacle_particles": len(self.fixed_obstacle_xyz),
            "mass": mass,
            "mass_drift": 0.0,
            "fluid_momentum": (
                float(self.mass) * np.sum(velocity, axis=0)
            ).tolist(),
            "fluid_kinetic_energy": float(
                0.5 * float(self.mass) * np.sum(velocity * velocity)
            ),
            "rho_min": float(np.min(density)),
            "rho_max": float(np.max(density)),
            "p_min": 0.0,
            "p_max": 0.0,
            "surge_front_x": float(np.max(state["fluid_x"][:, 0])),
            "max_height": float(np.max(state["fluid_x"][:, 2])),
            "all_finite": bool(finite),
            "adaptation_events": 0,
            "split_parents": 0,
            "merged_families": 0,
            "shifted_particles": 0,
            "max_shift": 0.0,
            "projection_iterations": self.config.projection_iterations,
            "constraint_rms": float(np.sqrt(np.mean(constraint * constraint))),
            "constraint_max_abs": float(np.max(np.abs(constraint))),
            "max_correction": state["max_correction"],
            "correction_clamps": state["clamp_count"],
            "solver_frame_median_ms": None if median is None else 1000.0 * median,
            "solver_frame_p95_ms": None if p95 is None else 1000.0 * p95,
            "simulated_to_wall_ratio": None if not history.size else (
                self.time / float(np.sum(history))
            ),
            "render_attribute_median_ms": (
                None if not render_history.size
                else 1000.0 * float(np.median(render_history))
            ),
            "render_attribute_p95_ms": (
                None if not render_history.size
                else 1000.0 * float(np.percentile(render_history, 95))
            ),
            "body_particles": len(self.body_reference_host) if self.config.obstacle_mode == "floating" else 0,
            "body_mass": self.body_mass if self.config.obstacle_mode == "floating" else 0.0,
            "body_cm": body_cm.tolist() if self.config.obstacle_mode == "floating" else None,
            "body_vc": body_v.tolist() if self.config.obstacle_mode == "floating" else None,
            "body_omega": body_omega.tolist() if self.config.obstacle_mode == "floating" else None,
            "body_orientation": state["body_orientation"].tolist() if self.config.obstacle_mode == "floating" else None,
            "body_geometry_drift": 0.0,
            "contact_force": None,
            "contact_impulse": self.body_impulse.numpy().tolist() if self.config.obstacle_mode == "floating" else None,
            "contact_max_penetration": 0.0,
            "rigid_device_error": 0,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        metrics = self.metrics()
        np.savez(
            path,
            **{key: value for key, value in snapshot.items() if isinstance(value, np.ndarray)},
            dt_history=np.asarray(self.dt_history),
            metrics=json.dumps(metrics, sort_keys=True),
            config=json.dumps(self.config.to_dict(), sort_keys=True),
        )
        return path
