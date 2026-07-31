#!/usr/bin/env python3
"""Run the two-level Warp dam-break engineering smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
        "--fine-bounds",
        type=_bounds,
        default=(1.75, 2.8, -0.3, 0.3, 0.0, 0.65),
    )
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--no-obstacle", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/pysph-adaptive-smoke.npz"))
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DamBreakConfig(
        resolution_mode=args.mode,
        dx=args.dx,
        steps=args.steps,
        adapt_every=args.adapt_every,
        max_splits_per_adapt=args.max_splits,
        fine_bounds=args.fine_bounds,
        with_obstacle=not args.no_obstacle,
    )
    simulation = WarpDamBreakSimulation(config)
    started = time.perf_counter()

    def report(snapshot, metrics):
        print(json.dumps({
            "step": metrics["step"],
            "time": metrics["time"],
            "fluid_particles": metrics["fluid_particles"],
            "coarse_particles": metrics["coarse_particles"],
            "fine_particles": metrics["fine_particles"],
            "split_parents": metrics["split_parents"],
            "merged_families": metrics["merged_families"],
            "mass_drift": metrics["mass_drift"],
            "all_finite": metrics["all_finite"],
        }, sort_keys=True), flush=True)

    metrics = simulation.run(
        snapshot_stride=args.snapshot_stride,
        callback=report,
    )
    simulation.save(args.output)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["output"] = str(args.output)
    manifest = args.manifest or args.output.with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "config": config.to_dict(),
        "metrics": metrics,
        "adaptation_history": simulation.adaptation_history,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["all_finite"]:
        raise SystemExit("adaptive smoke produced non-finite state")
    if args.mode == "adaptive":
        if metrics["fine_particles"] <= 0:
            raise SystemExit("adaptive smoke produced no fine particles")
        if metrics["split_parents"] <= 0:
            raise SystemExit("adaptive smoke recorded no split event")
        if metrics["mass_drift"] > 1.0e-6:
            raise SystemExit("adaptive smoke exceeded mass drift gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
