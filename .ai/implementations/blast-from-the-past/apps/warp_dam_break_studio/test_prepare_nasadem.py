import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from prepare_nasadem import (
    SOURCE_BYTES,
    prepare,
    read_official_hgt,
)


def _fixture_archive(path):
    # A deterministic north/south gradient compresses to a small fixture while
    # retaining the official 3601-square, big-endian NASADEM HGT format.
    rows = np.linspace(700, 20, 3601, dtype=np.int16)
    tile = np.repeat(rows[:, None], 3601, axis=1).astype(">i2")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("n43e006.hgt", tile.tobytes(order="C"))


def test_prepare_emits_bounded_metric_asset_and_provenance(tmp_path):
    source = tmp_path / "NASADEM_HGT_n43e006.zip"
    output = tmp_path / "malpasset.npz"
    _fixture_archive(source)
    _, manifest_path, manifest = prepare(
        source, output, "2026-08-06", grid_size=128,
    )
    assert manifest_path.is_file()
    assert manifest["source"]["product"] == "NASADEM_HGT.001"
    assert manifest["source"]["vertical_datum"] == "EGM96 geoid"
    assert manifest["crop"]["shape"] == [128, 128]
    assert min(manifest["crop"]["cell_size_m"]) > 0.0
    assert manifest["synthetic_event"]["synthetic_breach"] is True
    assert manifest["synthetic_event"]["historical_reconstruction"] is False
    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest["crop"]["asset_sha256"]
    with np.load(output, allow_pickle=False) as data:
        assert data["bed"].shape == (128, 128)
        assert data["barrier"].shape == data["bed"].shape
        assert data["initial_depth"].shape == data["bed"].shape
        assert np.all(np.isfinite(data["bed"]))
        np.testing.assert_array_equal(data["bed"], data["raw_bed"])
        assert np.count_nonzero(data["barrier"]) > 0
        assert np.min(data["initial_depth"]) >= 0.0


def test_reader_rejects_wrong_member_size(tmp_path):
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("n43e006.hgt", b"\0" * 16)
    with pytest.raises(ValueError, match=str(SOURCE_BYTES)):
        read_official_hgt(source)


def test_prepare_rejects_wrong_tile_and_unbounded_grid(tmp_path):
    source = tmp_path / "wrong.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("n44e006.hgt", b"")
    with pytest.raises(ValueError, match="n43e006.hgt"):
        read_official_hgt(source)

    source = tmp_path / "NASADEM_HGT_n43e006.zip"
    _fixture_archive(source)
    with pytest.raises(ValueError, match="bounded"):
        prepare(source, tmp_path / "bad.npz", "2026-08-06", grid_size=512)
