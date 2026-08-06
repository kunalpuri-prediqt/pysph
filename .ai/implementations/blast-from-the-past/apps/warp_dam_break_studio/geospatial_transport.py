"""Versioned, JSON-safe transport for regular-grid terrain and water fields."""

from __future__ import annotations

import base64

import numpy as np


GEOSPATIAL_FRAME_VERSION = 1


def _finite_grid(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2D grid")
    return np.ascontiguousarray(result)


def _quantize_u16(value):
    value = np.asarray(value, dtype=np.float32)
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    scale = (maximum - minimum) / 65535.0
    if scale <= 0.0:
        scale = 1.0
    encoded = np.rint((value - minimum) / scale).clip(
        0.0, 65535.0
    ).astype("<u2")
    return encoded, minimum, scale


def _decode_u16(payload, shape, minimum, scale, name):
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid {name} base64 payload") from exc
    expected = int(np.prod(shape)) * 2
    if len(raw) != expected:
        raise ValueError(
            f"{name} payload has {len(raw)} bytes; expected {expected}"
        )
    encoded = np.frombuffer(raw, dtype="<u2").reshape(shape)
    return encoded.astype(np.float32) * float(scale) + float(minimum)


def pack_geospatial_terrain(snapshot, provenance=None):
    bed = _finite_grid(snapshot.get("bed"), "bed")
    encoded, minimum, scale = _quantize_u16(bed)
    payload = base64.b64encode(encoded.tobytes(order="C")).decode("ascii")
    return {
        "version": GEOSPATIAL_FRAME_VERSION,
        "kind": "terrain",
        "encoding": "u16-le",
        "shape": list(bed.shape),
        "minimum": minimum,
        "scale": scale,
        "data": payload,
        "packed_bytes": len(payload),
        "cell_size": list(snapshot.get("cell_size", [1.0, 1.0])),
        "terrain_id": str(snapshot.get("terrain_id", "unknown")),
        "provenance": dict(provenance or {}),
    }


def unpack_geospatial_terrain(frame):
    if int(frame.get("version", -1)) != GEOSPATIAL_FRAME_VERSION:
        raise ValueError("unsupported geospatial terrain version")
    if frame.get("kind") != "terrain" or frame.get("encoding") != "u16-le":
        raise ValueError("unsupported geospatial terrain encoding")
    shape = tuple(int(value) for value in frame.get("shape", ()))
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("invalid geospatial terrain shape")
    return _decode_u16(
        frame.get("data", ""), shape,
        frame.get("minimum", 0.0), frame.get("scale", 0.0), "terrain",
    )


def pack_geospatial_frame(snapshot, metrics):
    depth = _finite_grid(snapshot.get("water_depth"), "water_depth")
    speed = _finite_grid(snapshot.get("water_speed"), "water_speed")
    if speed.shape != depth.shape:
        raise ValueError("water depth and speed grids must have matching shape")
    if np.min(depth) < 0.0 or np.min(speed) < 0.0:
        raise ValueError("water display fields cannot be negative")
    depth_encoded, depth_minimum, depth_scale = _quantize_u16(depth)
    speed_encoded, speed_minimum, speed_scale = _quantize_u16(speed)
    packed = np.empty(depth.size * 2, dtype="<u2")
    packed[0::2] = depth_encoded.ravel()
    packed[1::2] = speed_encoded.ravel()
    payload = base64.b64encode(packed.tobytes(order="C")).decode("ascii")
    return {
        "version": GEOSPATIAL_FRAME_VERSION,
        "kind": "water",
        "encoding": "interleaved-u16-le",
        "shape": list(depth.shape),
        "depth_minimum": depth_minimum,
        "depth_scale": depth_scale,
        "speed_minimum": speed_minimum,
        "speed_scale": speed_scale,
        "data": payload,
        "packed_bytes": len(payload),
        "step": int(metrics.get("step", snapshot.get("step", 0))),
        "time": float(metrics.get("time", snapshot.get("time", 0.0))),
        "dry_depth": float(metrics.get("dry_depth", 0.01)),
    }


def unpack_geospatial_frame(frame):
    if int(frame.get("version", -1)) != GEOSPATIAL_FRAME_VERSION:
        raise ValueError("unsupported geospatial water version")
    if (frame.get("kind") != "water"
            or frame.get("encoding") != "interleaved-u16-le"):
        raise ValueError("unsupported geospatial water encoding")
    shape = tuple(int(value) for value in frame.get("shape", ()))
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("invalid geospatial water shape")
    try:
        raw = base64.b64decode(frame.get("data", ""), validate=True)
    except Exception as exc:
        raise ValueError("invalid geospatial water base64 payload") from exc
    expected = int(np.prod(shape)) * 4
    if len(raw) != expected:
        raise ValueError(
            f"water payload has {len(raw)} bytes; expected {expected}"
        )
    encoded = np.frombuffer(raw, dtype="<u2").reshape(-1, 2)
    depth = encoded[:, 0].astype(np.float32).reshape(shape)
    speed = encoded[:, 1].astype(np.float32).reshape(shape)
    depth = depth * float(frame.get("depth_scale", 0.0)) + float(
        frame.get("depth_minimum", 0.0)
    )
    speed = speed * float(frame.get("speed_scale", 0.0)) + float(
        frame.get("speed_minimum", 0.0)
    )
    return depth, speed
