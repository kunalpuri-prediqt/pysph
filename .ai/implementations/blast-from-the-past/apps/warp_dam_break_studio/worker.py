"""Spawn-safe worker protocol for the Trame dam-break studio."""

from __future__ import annotations

from collections import deque
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time
import traceback


TERMINAL_STATES = {"completed", "cancelled", "failed"}

_INSTABILITY_HINT = (
    "Numerical instability: the simulation diverged (pressure/velocity became "
    "non-finite). Try a lower CFL, a smaller dx, coarser adaptation, or fewer "
    "steps."
)


def put_latest(queue, message):
    """Put a message without allowing visualization backpressure to block."""
    try:
        queue.put_nowait(message)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(message)
    except Full:
        # Another producer cannot exist, but tolerate platform queue feeder
        # timing without ever blocking the CUDA worker.
        pass


def _worker_main(config, command_queue, result_queue, snapshot_stride):
    """Run one simulation; CUDA is imported and initialized only in here."""
    try:
        put_latest(result_queue, {
            "type": "status",
            "status": "initializing",
            "detail": "Building particles and initializing Warp",
        })
        from pysph.base.warp_adaptive import (
            DamBreakConfig,
            WarpDamBreakSimulation,
        )

        solver_config = dict(config)
        output = solver_config.pop("output", None)
        simulation = WarpDamBreakSimulation(
            DamBreakConfig.from_mapping(solver_config)
        )
        started = time.perf_counter()
        first = simulation.initialize()
        put_latest(result_queue, {
            "type": "frame",
            "snapshot": first,
            "metrics": simulation.metrics(),
            "wall_seconds": time.perf_counter() - started,
        })
        put_latest(result_queue, {
            "type": "status",
            "status": "running",
            "detail": "GPU solver running",
        })

        paused = False
        cancelled = False
        single_step = False
        while not simulation.done and not cancelled:
            while True:
                try:
                    command = command_queue.get_nowait()
                except Empty:
                    break
                action = command.get("action")
                if action == "pause":
                    paused = True
                    put_latest(result_queue, {
                        "type": "status",
                        "status": "paused",
                        "detail": "Paused at a solver step boundary",
                    })
                elif action == "resume":
                    paused = False
                    single_step = False
                    put_latest(result_queue, {
                        "type": "status",
                        "status": "running",
                        "detail": "GPU solver running",
                    })
                elif action == "step":
                    paused = True
                    single_step = True
                elif action in {"cancel", "shutdown"}:
                    cancelled = True

            if cancelled:
                break
            if paused and not single_step:
                try:
                    command = command_queue.get(timeout=0.1)
                    command_queue.put_nowait(command)
                except (Empty, Full):
                    pass
                continue

            tick = time.perf_counter()
            try:
                progress = simulation.step()
            except OverflowError as exc:
                raise RuntimeError(_INSTABILITY_HINT) from exc
            if progress is not None and not math.isfinite(
                progress.get("dt", 0.0)
            ):
                raise RuntimeError(_INSTABILITY_HINT)
            step_seconds = time.perf_counter() - tick
            should_publish = (
                simulation.step_count % max(int(snapshot_stride), 1) == 0
                or simulation.done
                or single_step
            )
            if should_publish:
                metrics = simulation.metrics()
                if not metrics.get("all_finite", True):
                    raise RuntimeError(_INSTABILITY_HINT)
                metrics["step_wall_seconds"] = step_seconds
                metrics["steps_per_second"] = (
                    1.0 / step_seconds if step_seconds > 0 else None
                )
                put_latest(result_queue, {
                    "type": "frame",
                    "snapshot": simulation.snapshot(),
                    "metrics": metrics,
                    "wall_seconds": time.perf_counter() - started,
                })
            if single_step:
                single_step = False
                put_latest(result_queue, {
                    "type": "status",
                    "status": "paused",
                    "detail": "Single step completed",
                })

        if cancelled:
            put_latest(result_queue, {
                "type": "status",
                "status": "cancelled",
                "detail": "Run cancelled",
            })
        else:
            metrics = simulation.metrics()
            import warp as wp

            device = wp.get_device(simulation.config.device)
            metrics["runtime"] = {
                "warp_version": wp.__version__,
                "device": str(device),
                "device_name": getattr(device, "name", str(device)),
                "device_arch": getattr(device, "arch", None),
            }
            if output:
                simulation.save(output)
                metrics["output"] = output
                manifest = Path(output).with_suffix(".json")
                manifest.write_text(json.dumps({
                    "config": simulation.config.to_dict(),
                    "metrics": metrics,
                    "adaptation_history": simulation.adaptation_history,
                }, indent=2, sort_keys=True) + "\n")
                metrics["manifest"] = str(manifest)
            put_latest(result_queue, {
                "type": "completed",
                "status": "completed",
                "metrics": metrics,
                "snapshot": simulation.snapshot(),
                "wall_seconds": time.perf_counter() - started,
            })
    except BaseException as exc:
        put_latest(result_queue, {
            "type": "error",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


class SolverWorker:
    """Controller-side lifecycle for one spawned solver process."""

    def __init__(self, context="spawn", result_queue_size=3):
        self.context_name = context
        self.result_queue_size = int(result_queue_size)
        self.process = None
        self.command_queue = None
        self.result_queue = None

    @property
    def alive(self):
        return self.process is not None and self.process.is_alive()

    def start(self, config, snapshot_stride=1):
        self.close()
        context = mp.get_context(self.context_name)
        self.command_queue = context.Queue(maxsize=16)
        self.result_queue = context.Queue(maxsize=self.result_queue_size)
        self.process = context.Process(
            target=_worker_main,
            args=(
                dict(config),
                self.command_queue,
                self.result_queue,
                int(snapshot_stride),
            ),
            name="pysph-warp-dam-break",
            daemon=True,
        )
        self.process.start()

    def send(self, action):
        if self.command_queue is None:
            return False
        try:
            self.command_queue.put_nowait({"action": action})
            return True
        except Full:
            return False

    def poll(self, limit=32):
        messages = []
        if self.result_queue is None:
            return messages
        for _ in range(limit):
            try:
                messages.append(self.result_queue.get_nowait())
            except Empty:
                break
        return messages

    def close(self, timeout=2.0):
        if self.process is not None:
            if self.process.is_alive() and self.command_queue is not None:
                self.send("shutdown")
                self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout)
            self.process.close()
        for queue in (self.command_queue, self.result_queue):
            if queue is not None:
                queue.close()
                queue.join_thread()
        self.process = None
        self.command_queue = None
        self.result_queue = None


class FrameBuffer:
    """Bounded replay buffer with explicit live-frame semantics."""

    def __init__(self, capacity=120):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.frames = deque(maxlen=self.capacity)

    def append(self, snapshot, metrics):
        self.frames.append((snapshot, metrics))
        return len(self.frames) - 1

    def clear(self):
        self.frames.clear()

    def get(self, index):
        if not self.frames:
            raise IndexError("frame buffer is empty")
        index = max(0, min(int(index), len(self.frames) - 1))
        return self.frames[index]

    @property
    def latest(self):
        return self.get(len(self.frames) - 1)

    def __len__(self):
        return len(self.frames)
