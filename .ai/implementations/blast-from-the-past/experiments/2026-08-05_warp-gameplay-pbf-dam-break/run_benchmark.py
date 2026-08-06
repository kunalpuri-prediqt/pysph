"""Benchmark the isolated gameplay PBF dam-break profile.

The simulation synchronizes each solver step, so the solver samples measure
completed Warp work.  Snapshot samples are timed separately and include the
device-to-host copies needed by the current Trame renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import warp as wp

from pysph.base.warp_game import (
    GameplayDamBreakConfig,
    WarpGameplayDamBreakSimulation,
)


def percentile(values, fraction):
    return float(np.percentile(np.asarray(values), 100.0 * fraction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output", default="/tmp/pysph-gameplay-pbf-benchmark.json"
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    config = GameplayDamBreakConfig(
        steps=args.warmup + args.frames,
        device=args.device,
    )
    simulation = WarpGameplayDamBreakSimulation(config)

    compile_started = time.perf_counter()
    simulation.initialize()
    compile_seconds = time.perf_counter() - compile_started

    for _ in range(args.warmup):
        simulation.step()

    solver_seconds = []
    snapshot_seconds = []
    end_to_end_seconds = []
    for _ in range(args.frames):
        frame_started = time.perf_counter()
        solver_started = time.perf_counter()
        simulation.step()
        solver_seconds.append(time.perf_counter() - solver_started)

        snapshot_started = time.perf_counter()
        simulation.snapshot()
        snapshot_seconds.append(time.perf_counter() - snapshot_started)
        end_to_end_seconds.append(time.perf_counter() - frame_started)

    metrics = simulation.metrics()
    device = simulation.device
    result = {
        "config": config.to_dict(),
        "hardware": {
            "host": platform.node(),
            "processor": platform.processor(),
            "warp_version": wp.__version__,
            "device": device.alias,
            "device_name": device.name,
            "device_arch": device.arch,
        },
        "samples": {
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
            "particles": metrics["fluid_particles"],
            "projection_iterations": metrics["projection_iterations"],
        },
        "timing": {
            "compile_and_initialize_ms": 1000.0 * compile_seconds,
            "solver_median_ms": 1000.0 * statistics.median(solver_seconds),
            "solver_p95_ms": 1000.0 * percentile(solver_seconds, 0.95),
            "snapshot_median_ms": 1000.0 * statistics.median(snapshot_seconds),
            "snapshot_p95_ms": 1000.0 * percentile(snapshot_seconds, 0.95),
            "stream_loop_median_ms": 1000.0 * statistics.median(end_to_end_seconds),
            "stream_loop_p95_ms": 1000.0 * percentile(end_to_end_seconds, 0.95),
            "stream_loop_fps": args.frames / sum(end_to_end_seconds),
            "simulated_to_solver_wall_ratio": (
                args.frames * config.dt / sum(solver_seconds)
            ),
        },
        "final_metrics": metrics,
        "gates": {
            "all_finite": metrics["all_finite"],
            "solver_median_lte_16_7_ms": (
                1000.0 * statistics.median(solver_seconds) <= 16.7
            ),
            "solver_p95_lte_33_3_ms": (
                1000.0 * percentile(solver_seconds, 0.95) <= 33.3
            ),
            "rigid_device_error_zero": metrics["rigid_device_error"] == 0,
            "body_geometry_drift_lte_1e_5": (
                metrics["body_geometry_drift"] <= 1.0e-5
            ),
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
