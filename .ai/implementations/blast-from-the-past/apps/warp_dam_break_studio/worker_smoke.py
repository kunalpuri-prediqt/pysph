#!/usr/bin/env python3
"""Exercise the exact spawned-worker protocol used by the Trame controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from worker import SolverWorker


def main():
    parser = argparse.ArgumentParser()
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--gameplay", action="store_true")
    profile.add_argument("--geospatial", action="store_true")
    profile.add_argument("--terrain", action="store_true")
    args = parser.parse_args()
    worker = SolverWorker()
    messages = []
    paused_once = False
    stepped_once = False
    resumed_once = False
    deadline = time.monotonic() + 120.0
    try:
        config = {
            "resolution_mode": "adaptive",
            "dx": 0.1,
            "steps": 4,
            "adapt_every": 2,
            "max_splits_per_adapt": 64,
            "with_obstacle": True,
            "fine_bounds": [1.75, 2.8, -0.3, 0.3, 0.0, 0.65],
            "output": "/tmp/pysph-studio-worker-smoke.npz",
        }
        if args.gameplay:
            config.update({
                "solver_family": "gameplay-pbf",
                "resolution_mode": "uniform",
                "dx": 0.2,
                # Leave enough frames for the polling client to exercise the
                # pause/step/resume lifecycle even when the GPU solver runs
                # substantially faster than real time.
                "steps": 300,
                "projection_iterations": 2,
                "dt": 1.0 / 60.0,
                "obstacle_mode": "floating",
                "output": "/tmp/pysph-studio-worker-gameplay-smoke.npz",
            })
        elif args.geospatial:
            config = {
                "solver_family": "geospatial-swe",
                "steps": 300,
                "grid_nx": 256,
                "grid_ny": 256,
                "cell_size_x": 30.0,
                "cell_size_y": 30.0,
                "cfl": 0.35,
                "dt_max": 0.5,
                "dry_depth": 0.01,
                "manning": 0.025,
                "boundary": "closed",
                "terrain_id": "synthetic-valley-fixture",
                "synthetic_breach": True,
                "output": "/tmp/pysph-studio-worker-geospatial-smoke.npz",
            }
        elif args.terrain:
            config.update({
                "solver_family": "terrain-wcsph",
                "resolution_mode": "uniform",
                "steps": 12,
                "obstacle_mode": "hill",
                "hill_center_x": 3.0,
                "hill_center_y": 0.0,
                "hill_height": 0.35,
                "hill_radius_x": 0.40,
                "hill_radius_y": 0.12,
                "output": "/tmp/pysph-studio-worker-terrain-smoke.npz",
            })
        worker.start(config, snapshot_stride=1)
        completed = None
        while time.monotonic() < deadline:
            batch = worker.poll()
            messages.extend(batch)
            for message in batch:
                if (
                    message.get("type") == "frame"
                    and not paused_once
                ):
                    worker.send("pause")
                    paused_once = True
                if (
                    message.get("status") == "paused"
                    and paused_once
                    and not stepped_once
                ):
                    worker.send("step")
                    stepped_once = True
                elif (
                    message.get("status") == "paused"
                    and stepped_once
                    and not resumed_once
                ):
                    worker.send("resume")
                    resumed_once = True
                if message.get("type") == "completed":
                    completed = message
                if message.get("type") == "error":
                    raise RuntimeError(message["error"] + "\n" +
                                       message.get("traceback", ""))
            if completed is not None:
                break
            time.sleep(0.02)
        if completed is None:
            raise RuntimeError("worker smoke timed out")
        metrics = completed["metrics"]
        summary = {
            "statuses": [
                message.get("status")
                for message in messages
                if message.get("status")
            ],
            "frames": sum(
                message.get("type") == "frame" for message in messages
            ),
            "paused_once": paused_once,
            "stepped_once": stepped_once,
            "resumed_once": resumed_once,
            "metrics": metrics,
        }
        manifest = json.loads(Path(metrics["manifest"]).read_text())
        assert manifest["metrics"]["all_finite"]
        assert manifest["metrics"]["runtime"]["device_name"] == (
            metrics["runtime"]["device_name"]
        )
        summary_name = (
            "geospatial" if args.geospatial
            else "terrain" if args.terrain
            else "gameplay" if args.gameplay
            else "adaptive"
        )
        Path(
            f"/tmp/pysph-studio-worker-{summary_name}-smoke-summary.json"
        ).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        assert metrics["all_finite"]
        if args.gameplay:
            assert metrics["solver_family"] == "gameplay-pbf"
            assert metrics["approximate"]
        elif args.geospatial:
            assert metrics["solver_family"] == "geospatial-swe"
            assert metrics["depth_averaged"]
            assert metrics["synthetic_terrain"]
            assert metrics["grid_shape"] == [256, 256]
        elif args.terrain:
            assert metrics["solver_family"] == "terrain-wcsph"
            assert metrics["hill_particles"] > 0
            assert metrics["obstacle_mode"] == "hill"
        else:
            assert metrics["fine_particles"] > 0
        assert metrics["mass_drift"] <= 1.0e-6
        assert paused_once and stepped_once and resumed_once
    finally:
        worker.close()


if __name__ == "__main__":
    main()
