#!/usr/bin/env python3
"""Exercise the exact spawned-worker protocol used by the Trame controls."""

from __future__ import annotations

import json
from pathlib import Path
import time

from worker import SolverWorker


def main():
    worker = SolverWorker()
    messages = []
    paused_once = False
    stepped_once = False
    resumed_once = False
    deadline = time.monotonic() + 120.0
    try:
        worker.start({
            "resolution_mode": "adaptive",
            "dx": 0.1,
            "steps": 4,
            "adapt_every": 2,
            "max_splits_per_adapt": 64,
            "with_obstacle": True,
            "fine_bounds": [1.75, 2.8, -0.3, 0.3, 0.0, 0.65],
            "output": "/tmp/pysph-studio-worker-smoke.npz",
        }, snapshot_stride=1)
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
        Path("/tmp/pysph-studio-worker-smoke-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        assert metrics["all_finite"]
        assert metrics["fine_particles"] > 0
        assert metrics["mass_drift"] <= 1.0e-6
        assert paused_once and stepped_once and resumed_once
    finally:
        worker.close()


if __name__ == "__main__":
    main()
