#!/usr/bin/env python3
"""Hydrostatic half-submerged floating-box equilibrium diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import setuptools  # noqa: F401  # activates distutils compatibility on 3.14

from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.base.warp_adaptive import DamBreakConfig, WarpDamBreakSimulation
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import (
    create_rigid_body_state,
    wc_sph_dam_break_rigid_step,
)


GRAVITY = 9.81
RHO0 = 1000.0
C0 = 10.0 * math.sqrt(2.0 * GRAVITY * 0.55)


def hydrostatic_density(z, free_surface=1.0):
    pressure = RHO0 * GRAVITY * np.maximum(free_surface - z, 0.0)
    return RHO0 * (1.0 + 7.0 * pressure / (RHO0 * C0**2)) ** (1.0 / 7.0)


def run(dx, steps=400, tf=None):
    config = DamBreakConfig(
        resolution_mode="uniform",
        obstacle_mode="floating",
        dx=dx,
        steps=steps,
        body_spacing=0.1,
        body_center_x=0.5,
        body_center_y=0.0,
        body_center_z=1.0,
        body_density=500.0,
        contact_enabled=False,
        n_damp=50,
    )
    simulation = WarpDamBreakSimulation(config)
    geometry = DamBreak3DGeometry(
        container_height=1.5,
        container_width=0.5,
        container_length=1.0,
        fluid_column_height=1.0,
        fluid_column_width=0.5,
        fluid_column_length=1.0,
        dx=dx,
        nboundary_layers=1,
        hdx=config.hdx,
        rho0=RHO0,
        with_obstacle=False,
    )
    cpu_fluid, cpu_wall = geometry.create_particles()
    inside_body = (
        (np.abs(cpu_fluid.x - config.body_center_x)
         < 0.5 * config.body_length)
        & (np.abs(cpu_fluid.y - config.body_center_y)
           < 0.5 * config.body_width)
        & (np.abs(cpu_fluid.z - config.body_center_z)
           < 0.5 * config.body_height)
    )
    cpu_fluid.remove_particles(np.flatnonzero(inside_body))
    fluid = simulation._to_warp(cpu_fluid, "fluid")
    wall = simulation._to_warp(cpu_wall, "wall")
    body = simulation._floating_body()
    fluid.rho[:] = hydrostatic_density(np.asarray(fluid.z))
    wall.rho[:] = hydrostatic_density(np.asarray(wall.z))
    particles = [fluid, wall, body]
    nnps = UniformGridWarpNNPS(
        dim=3, particles=particles, radius_scale=config.radius_scale
    )
    rigid_state = create_rigid_body_state(body, nbody=1)
    body_xyz = np.column_stack((body.x, body.y, body.z))
    initial_cm = np.average(body_xyz, axis=0, weights=body.m)
    initial_mass = float(np.sum(fluid.m, dtype=np.float64))
    time_now = 0.0
    history = []
    completed_steps = 0
    for step in range(steps):
        if tf is not None and time_now >= tf:
            break
        if config.n_damp > 0 and step < config.n_damp:
            scale = 0.5 * (
                math.sin(math.pi * (-0.5 + (step + 1) / config.n_damp)) + 1.0
            )
        else:
            scale = 1.0
        step_dt_max = simulation.dt_max
        if tf is not None:
            step_dt_max = min(step_dt_max, max(tf - time_now, 0.0))
        dt = wc_sph_dam_break_rigid_step(
            nnps,
            rigid_state,
            fluid_index=0,
            wall_indices=(1,),
            rigid_index=2,
            dt=simulation.seed_dt,
            rho0=RHO0,
            c0=C0,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            kernel=config.kernel,
            xsph_eps=config.xsph_eps,
            gz=config.gz,
            adaptive_dt=True,
            cfl=config.cfl,
            dt_min=0.0,
            dt_max=simulation.dt_max,
            adaptive_dt_scale=scale,
            step_dt_max=step_dt_max,
            push=(step == 0),
            return_dt=True,
            neighbor_mode="grid",
            contact_bounds=None,
        )
        time_now += float(dt)
        completed_steps = step + 1
        if (step % 10 == 0 or step + 1 == steps or
                (tf is not None and time_now >= tf)):
            history.append({
                "step": step + 1,
                "time": time_now,
                "cm": np.asarray(rigid_state.cm.numpy()).reshape(-1, 3)[0].tolist(),
                "vc": np.asarray(rigid_state.vc.numpy()).reshape(-1, 3)[0].tolist(),
                "omega": np.asarray(rigid_state.omega.numpy()).reshape(-1, 3)[0].tolist(),
            })
    fluid.gpu.pull("rho", "p", "u", "v", "w")
    final_cm = np.asarray(rigid_state.cm.numpy()).reshape(-1, 3)[0]
    final_vc = np.asarray(rigid_state.vc.numpy()).reshape(-1, 3)[0]
    final_omega = np.asarray(rigid_state.omega.numpy()).reshape(-1, 3)[0]
    cm_history = np.asarray([item["cm"] for item in history])
    report = {
        "dx": dx,
        "steps": completed_steps,
        "step_limit": steps,
        "target_time": tf,
        "time": time_now,
        "fluid_particles": fluid.get_number_of_particles(),
        "wall_particles": wall.get_number_of_particles(),
        "body_particles": body.get_number_of_particles(),
        "fluid_mass": initial_mass,
        "initial_cm": initial_cm.tolist(),
        "final_cm": final_cm.tolist(),
        "cm_displacement": (final_cm - initial_cm).tolist(),
        "max_vertical_excursion": float(np.max(np.abs(
            cm_history[:, 2] - initial_cm[2]
        ))),
        "final_vc": final_vc.tolist(),
        "final_omega": final_omega.tolist(),
        "rho_min": float(np.min(fluid.rho)),
        "rho_max": float(np.max(fluid.rho)),
        "p_min": float(np.min(fluid.p)),
        "p_max": float(np.max(fluid.p)),
        "all_finite": bool(all(np.isfinite(value).all() for value in (
            fluid.rho, fluid.p, fluid.u, fluid.v, fluid.w,
            final_cm, final_vc, final_omega,
        ))),
        "rigid_device_error": int(rigid_state.error.numpy()[0]),
        "history": history,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tf", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.dx, args.steps, args.tf)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not report["all_finite"] or report["rigid_device_error"]:
        raise SystemExit("floating-equilibrium diagnostic failed")


if __name__ == "__main__":
    main()
