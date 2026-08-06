import copy

import numpy as np
import pytest

from surface_transport import (
    SURFACE_FRAME_STRIDE_BYTES,
    pack_surface_frame,
    unpack_surface_frame,
)


def _surface_snapshot(count=3):
    axes = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], count, axis=0)
    return {
        "step": 7,
        "time": 0.125,
        "render_xyz": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "render_axes": axes,
        "render_scale": np.asarray(
            [[0.10, 0.15, 0.20]] * count, dtype=np.float32
        ),
        "render_neighbors": np.arange(count, dtype=np.int32) + 6,
        "speed": np.linspace(0.0, 2.0, count, dtype=np.float32),
        "obstacle_mode": "floating",
    }


def _metrics():
    return {
        "step": 7,
        "time": 0.125,
        "body_cm": [2.3, 0.0, 0.4],
        "body_orientation": [0.0, 0.0, 0.0, 1.0],
    }


def test_surface_transport_round_trips_exact_float32_layout():
    frame = pack_surface_frame(
        _surface_snapshot(), _metrics(), [0.32, 0.28, 0.20]
    )
    decoded = unpack_surface_frame(frame)
    assert frame["raw_bytes"] == 3 * SURFACE_FRAME_STRIDE_BYTES
    np.testing.assert_array_equal(decoded[:, :3], _surface_snapshot()["render_xyz"])
    np.testing.assert_array_equal(decoded[:, 3], _surface_snapshot()["speed"])
    np.testing.assert_array_equal(
        decoded[:, 4:7], np.asarray([[0.1, 0.0, 0.0]] * 3, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        decoded[:, 8:11], np.asarray([[0.0, 0.15, 0.0]] * 3, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        decoded[:, 12:15], np.asarray([[0.0, 0.0, 0.2]] * 3, dtype=np.float32)
    )
    np.testing.assert_array_equal(decoded[:, 7], [6.0, 7.0, 8.0])
    assert frame["body"]["visible"] is True
    assert frame["body"]["half_size"] == pytest.approx([0.16, 0.14, 0.10])


def test_surface_transport_is_replay_deterministic_except_timing():
    first = pack_surface_frame(
        _surface_snapshot(), _metrics(), [0.32, 0.28, 0.20]
    )
    second = pack_surface_frame(
        _surface_snapshot(), _metrics(), [0.32, 0.28, 0.20]
    )
    first.pop("pack_ms")
    second.pop("pack_ms")
    assert first == second


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 99, "version"),
        ("count", -1, "count"),
        ("stride_floats", 12, "stride"),
        ("payload", "not base64!", "base64"),
    ],
)
def test_surface_transport_rejects_malformed_headers(field, value, message):
    frame = pack_surface_frame(
        _surface_snapshot(), _metrics(), [0.32, 0.28, 0.20]
    )
    frame[field] = value
    with pytest.raises(ValueError, match=message):
        unpack_surface_frame(frame)


def test_surface_transport_rejects_truncated_or_nonfinite_payload():
    frame = pack_surface_frame(
        _surface_snapshot(), _metrics(), [0.32, 0.28, 0.20]
    )
    truncated = copy.deepcopy(frame)
    truncated["payload"] = truncated["payload"][:-4]
    with pytest.raises(ValueError, match="bytes"):
        unpack_surface_frame(truncated)

    snapshot = _surface_snapshot()
    snapshot["render_xyz"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pack_surface_frame(snapshot, _metrics(), [0.32, 0.28, 0.20])


def test_default_thousand_particle_frame_stays_under_wire_budget():
    frame = pack_surface_frame(
        _surface_snapshot(1000), _metrics(), [0.32, 0.28, 0.20]
    )
    assert frame["raw_bytes"] == 64_000
    assert frame["wire_bytes"] <= 96 * 1024
