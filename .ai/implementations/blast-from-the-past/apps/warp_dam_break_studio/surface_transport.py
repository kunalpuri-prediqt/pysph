"""Versioned, bounded transport for gameplay WebGPU surface frames."""

from __future__ import annotations

import base64
import binascii
import time

import numpy as np


SURFACE_FRAME_VERSION = 1
SURFACE_FRAME_STRIDE_FLOATS = 16
SURFACE_FRAME_STRIDE_BYTES = 4 * SURFACE_FRAME_STRIDE_FLOATS
DEFAULT_TANK_BOUNDS = (0.0, 161.0 / 30.0, -0.25, 0.25, 0.0, 1.5)


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def pack_surface_frame(snapshot, metrics, body_dimensions, tank_bounds=None):
    """Pack one display-only gameplay snapshot into a JSON-safe state value.

    Each particle occupies four vec4<f32> values: center/speed followed by the
    three *scaled* ellipsoid basis columns.  The otherwise unused first basis
    ``w`` component carries the neighborhood count for debug visualization.
    """
    started = time.perf_counter()
    centers = np.asarray(snapshot.get("render_xyz"), dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("surface frames require render_xyz with shape (N, 3)")
    count = len(centers)
    centers = _finite_array(centers, (count, 3), "render_xyz")
    axes = _finite_array(
        snapshot.get("render_axes"), (count, 3, 3), "render_axes"
    )
    scale = _finite_array(
        snapshot.get("render_scale"), (count, 3), "render_scale"
    )
    if np.any(scale <= 0.0):
        raise ValueError("render_scale must be strictly positive")
    speed = _finite_array(
        np.asarray(snapshot.get("speed"), dtype=np.float32)[:count],
        (count,), "speed",
    )
    neighbors = np.asarray(
        snapshot.get("render_neighbors", np.zeros(count)), dtype=np.float32
    )
    neighbors = _finite_array(neighbors, (count,), "render_neighbors")

    scaled_axes = axes * scale[:, np.newaxis, :]
    particles = np.zeros(
        (count, SURFACE_FRAME_STRIDE_FLOATS), dtype=np.float32
    )
    particles[:, :3] = centers
    particles[:, 3] = speed
    particles[:, 4:7] = scaled_axes[:, :, 0]
    particles[:, 7] = neighbors
    particles[:, 8:11] = scaled_axes[:, :, 1]
    particles[:, 12:15] = scaled_axes[:, :, 2]
    particles = np.ascontiguousarray(particles, dtype="<f4")
    raw = particles.tobytes()
    payload = base64.b64encode(raw).decode("ascii")

    bounds = tuple(DEFAULT_TANK_BOUNDS if tank_bounds is None else tank_bounds)
    if len(bounds) != 6 or not np.all(np.isfinite(bounds)):
        raise ValueError("tank bounds must contain six finite values")
    dimensions = _finite_array(body_dimensions, (3,), "body_dimensions")
    body_cm = metrics.get("body_cm")
    body_orientation = metrics.get("body_orientation")
    body_visible = (
        snapshot.get("obstacle_mode", metrics.get("obstacle_mode"))
        == "floating" and body_cm is not None and body_orientation is not None
    )
    if body_visible:
        body_cm = _finite_array(body_cm, (3,), "body_cm").tolist()
        body_orientation = _finite_array(
            body_orientation, (4,), "body_orientation"
        ).tolist()
    else:
        body_cm = [0.0, 0.0, 0.0]
        body_orientation = [0.0, 0.0, 0.0, 1.0]

    return {
        "version": SURFACE_FRAME_VERSION,
        "count": count,
        "stride_floats": SURFACE_FRAME_STRIDE_FLOATS,
        "payload": payload,
        "raw_bytes": len(raw),
        "wire_bytes": len(payload),
        "step": int(metrics.get("step", snapshot.get("step", 0))),
        "time": float(metrics.get("time", snapshot.get("time", 0.0))),
        "tank_bounds": [float(value) for value in bounds],
        "body": {
            "visible": body_visible,
            "center": body_cm,
            "orientation": body_orientation,
            "half_size": (0.5 * dimensions).tolist(),
        },
        "pack_ms": (time.perf_counter() - started) * 1000.0,
    }


def unpack_surface_frame(frame):
    """Validate and decode a surface frame for tests and diagnostics."""
    if not isinstance(frame, dict):
        raise ValueError("surface frame must be a mapping")
    if frame.get("version") != SURFACE_FRAME_VERSION:
        raise ValueError("unsupported surface frame version")
    count = frame.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("surface frame count must be a non-negative integer")
    if frame.get("stride_floats") != SURFACE_FRAME_STRIDE_FLOATS:
        raise ValueError("unsupported surface frame stride")
    payload = frame.get("payload")
    if not isinstance(payload, str):
        raise ValueError("surface frame payload must be base64 text")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("surface frame payload is not valid base64") from exc
    expected_bytes = count * SURFACE_FRAME_STRIDE_BYTES
    if len(raw) != expected_bytes:
        raise ValueError(
            f"surface frame payload has {len(raw)} bytes, expected "
            f"{expected_bytes}"
        )
    particles = np.frombuffer(raw, dtype="<f4").reshape(
        count, SURFACE_FRAME_STRIDE_FLOATS
    )
    if not np.all(np.isfinite(particles)):
        raise ValueError("surface frame contains non-finite particle values")
    return particles
