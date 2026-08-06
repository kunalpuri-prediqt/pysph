import base64

import numpy as np
import pytest

from geospatial_transport import (
    pack_geospatial_frame,
    pack_geospatial_terrain,
    unpack_geospatial_frame,
    unpack_geospatial_terrain,
)


def test_terrain_quantization_is_bounded_and_preserves_metadata():
    bed = np.linspace(10.0, 210.0, 63, dtype=np.float32).reshape(7, 9)
    frame = pack_geospatial_terrain({
        "bed": bed,
        "cell_size": [30.0, 31.0],
        "terrain_id": "nasadem-test",
    }, provenance={"product": "NASADEM_HGT.001"})
    decoded = unpack_geospatial_terrain(frame)

    assert frame["terrain_id"] == "nasadem-test"
    assert frame["provenance"]["product"] == "NASADEM_HGT.001"
    assert frame["packed_bytes"] == len(frame["data"])
    assert np.max(np.abs(decoded - bed)) <= frame["scale"] * 0.51


def test_water_frame_round_trip_respects_per_field_quantization():
    depth = np.linspace(0.0, 12.0, 80, dtype=np.float32).reshape(8, 10)
    speed = np.sqrt(depth).astype(np.float32)
    frame = pack_geospatial_frame(
        {"water_depth": depth, "water_speed": speed, "step": 3},
        {"step": 3, "time": 1.25},
    )
    decoded_depth, decoded_speed = unpack_geospatial_frame(frame)

    assert frame["step"] == 3
    assert frame["time"] == pytest.approx(1.25)
    assert np.max(np.abs(decoded_depth - depth)) <= (
        frame["depth_scale"] * 0.51
    )
    assert np.max(np.abs(decoded_speed - speed)) <= (
        frame["speed_scale"] * 0.51
    )


def test_geospatial_decoder_rejects_version_length_and_base64_errors():
    depth = np.ones((4, 5), dtype=np.float32)
    frame = pack_geospatial_frame(
        {"water_depth": depth, "water_speed": depth}, {"step": 0}
    )
    invalid = dict(frame, version=99)
    with pytest.raises(ValueError, match="version"):
        unpack_geospatial_frame(invalid)
    invalid = dict(frame, data=base64.b64encode(b"short").decode("ascii"))
    with pytest.raises(ValueError, match="expected"):
        unpack_geospatial_frame(invalid)
    invalid = dict(frame, data="not base64")
    with pytest.raises(ValueError, match="base64"):
        unpack_geospatial_frame(invalid)


def test_geospatial_packer_rejects_nonfinite_or_mismatched_grids():
    depth = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="matching"):
        pack_geospatial_frame(
            {"water_depth": depth, "water_speed": np.ones((5, 4))}, {}
        )
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack_geospatial_frame(
            {"water_depth": depth, "water_speed": np.ones((4, 5))}, {}
        )
