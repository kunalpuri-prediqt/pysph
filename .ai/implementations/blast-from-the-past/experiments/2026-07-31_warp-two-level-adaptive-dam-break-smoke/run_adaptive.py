#!/usr/bin/env python3
"""Run the two-level Warp dam-break engineering smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pysph.base.warp_adaptive import DamBreakConfig, WarpDamBreakSimulation


def _bounds(value):
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "fine bounds require xmin,xmax,ymin,ymax,zmin,zmax"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("uniform", "adaptive"),
                        default="adaptive")
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--adapt-every", type=int, default=5)
    parser.add_argument("--max-splits", type=int, default=128)
    parser.add_argument(
        "--split-stencil", choices=("octant8", "icosa13"),
        default="icosa13",
    )
    parser.add_argument("--adapt-hysteresis", type=float, default=0.0)
    parser.add_argument("--shift-iterations", type=int, default=0)
    parser.add_argument("--shift-coefficient", type=float, default=0.02)
    parser.add_argument(
        "--no-variable-h-correction", action="store_true",
        help="disable adaptive grad-h consistency terms for an A/B run",
    )
    parser.add_argument(
        "--fine-bounds",
        type=_bounds,
        default=(1.75, 2.8, -0.3, 0.3, 0.0, 0.65),
    )
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument(
        "--obstacle-mode", choices=("none", "fixed", "floating"),
        default=None,
    )
    parser.add_argument("--no-obstacle", action="store_true")
    parser.add_argument("--body-density", type=float, default=500.0)
    parser.add_argument("--body-center-x", type=float, default=2.35)
    parser.add_argument("--body-center-z", type=float, default=0.30)
    parser.add_argument("--body-spacing", type=float, default=None)
    parser.add_argument("--restart", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/pysph-adaptive-smoke.npz"))
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    obstacle_mode = args.obstacle_mode
    if obstacle_mode is None:
        obstacle_mode = "none" if args.no_obstacle else "fixed"
    if args.restart is not None:
        simulation = WarpDamBreakSimulation.load(
            args.restart, steps=args.steps
        )
        config = simulation.config
    else:
        config = DamBreakConfig(
            resolution_mode=args.mode,
            dx=args.dx,
            steps=args.steps,
            adapt_every=args.adapt_every,
            max_splits_per_adapt=args.max_splits,
            split_stencil=args.split_stencil,
            adapt_hysteresis=args.adapt_hysteresis,
            shift_iterations=args.shift_iterations,
            shift_coefficient=args.shift_coefficient,
            variable_h_correction=not args.no_variable_h_correction,
            fine_bounds=args.fine_bounds,
            obstacle_mode=obstacle_mode,
            body_density=args.body_density,
            body_center_x=args.body_center_x,
            body_center_z=args.body_center_z,
            body_spacing=args.body_spacing,
        )
        simulation = WarpDamBreakSimulation(config)
    started = time.perf_counter()
    physics_history = []

    def report(snapshot, metrics):
        physics_history.append({
            key: metrics.get(key)
            for key in (
                "step", "time", "fluid_kinetic_energy", "fluid_momentum",
                "p_min", "p_max", "beta_h_min", "beta_h_max",
                "body_cm", "body_vc", "body_omega", "body_fluid_force",
                "body_fluid_torque", "contact_force", "contact_torque",
                "contact_impulse",
            )
        })
        print(json.dumps({
            "step": metrics["step"],
            "time": metrics["time"],
            "fluid_particles": metrics["fluid_particles"],
            "coarse_particles": metrics["coarse_particles"],
            "fine_particles": metrics["fine_particles"],
            "split_parents": metrics["split_parents"],
            "merged_families": metrics["merged_families"],
            "shifted_particles": metrics["shifted_particles"],
            "max_shift": metrics["max_shift"],
            "mass_drift": metrics["mass_drift"],
            "beta_h_min": metrics["beta_h_min"],
            "beta_h_max": metrics["beta_h_max"],
            "all_finite": metrics["all_finite"],
            "obstacle_mode": metrics["obstacle_mode"],
            "body_cm": metrics["body_cm"],
            "body_geometry_drift": metrics["body_geometry_drift"],
            "rigid_device_error": metrics["rigid_device_error"],
        }, sort_keys=True), flush=True)

    metrics = simulation.run(
        snapshot_stride=args.snapshot_stride,
        callback=report,
    )
    times = np.asarray([item["time"] for item in physics_history])
    for source, target in (
        ("body_fluid_force", "body_fluid_impulse"),
        ("body_fluid_torque", "body_fluid_angular_impulse"),
        ("contact_force", "sampled_contact_impulse"),
        ("contact_torque", "sampled_contact_angular_impulse"),
    ):
        values = [item[source] for item in physics_history]
        if len(times) > 1 and all(value is not None for value in values):
            metrics[target] = np.trapezoid(
                np.asarray(values, dtype=np.float64), times, axis=0
            ).tolist()
        else:
            metrics[target] = None
    simulation.save(args.output)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["output"] = str(args.output)
    manifest = args.manifest or args.output.with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "config": config.to_dict(),
        "metrics": metrics,
        "physics_history": physics_history,
        "adaptation_history": simulation.adaptation_history,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["all_finite"]:
        raise SystemExit("adaptive smoke produced non-finite state")
    if config.resolution_mode == "adaptive":
        if metrics["fine_particles"] <= 0:
            raise SystemExit("adaptive smoke produced no fine particles")
        if metrics["split_parents"] <= 0:
            raise SystemExit("adaptive smoke recorded no split event")
        if metrics["mass_drift"] > 1.0e-6:
            raise SystemExit("adaptive smoke exceeded mass drift gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
