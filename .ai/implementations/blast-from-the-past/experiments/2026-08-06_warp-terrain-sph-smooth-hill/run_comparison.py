"""Compare the procedural-hill Terrain SPH profile with a flat reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from pysph.base.warp_adaptive import DamBreakConfig, WarpDamBreakSimulation


def run(obstacle_mode, steps):
    simulation = WarpDamBreakSimulation(DamBreakConfig(
        solver_family="terrain-wcsph" if obstacle_mode == "hill" else "wcsph",
        resolution_mode="uniform",
        obstacle_mode=obstacle_mode,
        dx=0.1,
        steps=steps,
        device="cuda:0",
    ))
    started = time.perf_counter()
    initial = simulation.initialize()
    initialize_seconds = time.perf_counter() - started
    hill_mask = initial["kind"] == 2
    hill_initial = initial["xyz"][hill_mask].copy()
    step_seconds = []
    for _ in range(steps):
        tick = time.perf_counter()
        simulation.step()
        step_seconds.append(time.perf_counter() - tick)
    final = simulation.snapshot(include_solids=True)
    hill_final = final["xyz"][final["kind"] == 2]
    return {
        "simulation": simulation,
        "initial": initial,
        "final": final,
        "metrics": simulation.metrics(),
        "initialize_seconds": initialize_seconds,
        "step_seconds": step_seconds,
        "hill_stationary": bool(
            len(hill_initial) == len(hill_final)
            and np.array_equal(hill_initial, hill_final)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--output", default="/tmp/pysph-terrain-sph-comparison.json",
    )
    args = parser.parse_args()

    flat = run("none", args.steps)
    hill = run("hill", args.steps)
    flat_xyz = flat["final"]["xyz"][flat["final"]["kind"] == 0]
    hill_xyz = hill["final"]["xyz"][hill["final"]["kind"] == 0]
    displacement = np.linalg.norm(hill_xyz - flat_xyz, axis=1)
    result = {
        "config": {
            "dx": 0.1,
            "steps": args.steps,
            "hill": hill["metrics"]["hill"],
        },
        "flat_metrics": flat["metrics"],
        "hill_metrics": hill["metrics"],
        "comparison": {
            "particle_position_rms_delta_m": float(np.sqrt(np.mean(
                displacement * displacement
            ))),
            "particle_position_max_delta_m": float(np.max(displacement)),
            "particles_changed_gt_1mm": int(np.count_nonzero(
                displacement > 1.0e-3
            )),
            "surge_front_delta_m": float(
                hill["metrics"]["surge_front_x"]
                - flat["metrics"]["surge_front_x"]
            ),
            "kinetic_energy_delta_j": float(
                hill["metrics"]["fluid_kinetic_energy"]
                - flat["metrics"]["fluid_kinetic_energy"]
            ),
        },
        "timing": {
            "flat_initialize_s": flat["initialize_seconds"],
            "hill_initialize_s": hill["initialize_seconds"],
            "flat_step_median_ms": 1000.0 * statistics.median(
                flat["step_seconds"]
            ),
            "hill_step_median_ms": 1000.0 * statistics.median(
                hill["step_seconds"]
            ),
        },
        "gates": {
            "hill_finite": hill["metrics"]["all_finite"],
            "hill_mass_drift_lte_1e_6": (
                abs(hill["metrics"]["mass_drift"]) <= 1.0e-6
            ),
            "hill_stationary": hill["hill_stationary"],
            "measurable_interaction": bool(np.max(displacement) > 1.0e-3),
        },
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(result["gates"].values()):
        raise SystemExit("one or more Terrain SPH acceptance gates failed")


if __name__ == "__main__":
    main()
