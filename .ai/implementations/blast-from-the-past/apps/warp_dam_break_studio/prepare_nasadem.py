#!/usr/bin/env python3
"""Prepare the bounded NASADEM N43E006 terrain used by the studio.

The input must be the official ``NASADEM_HGT_n43e006.zip`` granule acquired
from NASA Earthdata.  This script never downloads data or handles credentials.
It emits a bounded metric-grid NPZ and an adjacent provenance JSON document.
The reservoir barrier and breach are deliberately synthetic and remain in a
separate array from the resampled NASADEM elevations.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np


TILE = "N43E006"
PRODUCT = "NASADEM_HGT.001"
PRODUCER_GRANULE_ID = "NASADEM_HGT_n43e006"
CMR_GRANULE_ID = "G2816791562-LPCLOUD"
SOURCE_DOI = "10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001"
SOURCE_URL = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
    "NASADEM_HGT.001/NASADEM_HGT_n43e006/"
    "NASADEM_HGT_n43e006.zip"
)
SOURCE_SHAPE = (3601, 3601)
SOURCE_BYTES = SOURCE_SHAPE[0] * SOURCE_SHAPE[1] * 2
TILE_BOUNDS = (6.0, 43.0, 7.0, 44.0)
DEFAULT_BOUNDS = (6.70, 43.47, 6.82, 43.58)
PROCESSING_VERSION = 1


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }, sort_keys=True).encode("ascii")
    return _sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def read_official_hgt(source_zip):
    """Read and strictly validate the expected 3601-square HGT member."""
    source_zip = Path(source_zip)
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    with zipfile.ZipFile(source_zip) as archive:
        candidates = [
            info for info in archive.infolist()
            if Path(info.filename).name.lower() == "n43e006.hgt"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "official NASADEM archive must contain exactly one "
                "n43e006.hgt member"
            )
        member = candidates[0]
        if member.file_size != SOURCE_BYTES:
            raise ValueError(
                f"n43e006.hgt has {member.file_size} bytes; expected "
                f"{SOURCE_BYTES} for a 3601x3601 one-arc-second tile"
            )
        raw = archive.read(member)
    elevation = np.frombuffer(raw, dtype=">i2").reshape(SOURCE_SHAPE)
    if np.any(elevation == -32768):
        raise ValueError("NASADEM HGT contains an invalid -32768 sample")
    return elevation, {
        "archive_sha256": _sha256_file(source_zip),
        "hgt_member": member.filename,
        "hgt_sha256": _sha256_bytes(raw),
        "archive_bytes": source_zip.stat().st_size,
    }


def _validate_bounds(bounds):
    west, south, east, north = (float(value) for value in bounds)
    if not (6.0 <= west < east <= 7.0 and 43.0 <= south < north <= 44.0):
        raise ValueError("crop bounds must lie inside NASADEM tile N43E006")
    return west, south, east, north


def _bilinear_resample(elevation, bounds, grid_size):
    west, south, east, north = _validate_bounds(bounds)
    grid_size = int(grid_size)
    if grid_size not in {128, 256}:
        raise ValueError("output grid must use the bounded 128 or 256 preset")

    longitude = np.linspace(west, east, grid_size, dtype=np.float64)
    latitude = np.linspace(south, north, grid_size, dtype=np.float64)
    source_x = (longitude - TILE_BOUNDS[0]) * 3600.0
    source_y = (TILE_BOUNDS[3] - latitude) * 3600.0
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, SOURCE_SHAPE[1] - 1)
    y1 = np.minimum(y0 + 1, SOURCE_SHAPE[0] - 1)
    fx = (source_x - x0)[None, :]
    fy = (source_y - y0)[:, None]
    z00 = elevation[np.ix_(y0, x0)].astype(np.float64)
    z10 = elevation[np.ix_(y0, x1)].astype(np.float64)
    z01 = elevation[np.ix_(y1, x0)].astype(np.float64)
    z11 = elevation[np.ix_(y1, x1)].astype(np.float64)
    bed = (
        (1.0 - fx) * (1.0 - fy) * z00
        + fx * (1.0 - fy) * z10
        + (1.0 - fx) * fy * z01
        + fx * fy * z11
    )
    # Although source rows run north-to-south, ``source_y`` was evaluated for
    # output latitudes ordered south-to-north, so the sampled result already
    # uses increasing local northing.
    bed = bed.astype(np.float32)

    center_latitude = 0.5 * (south + north)
    metres_per_degree_latitude = 111132.92 - 559.82 * math.cos(
        2.0 * math.radians(center_latitude)
    )
    metres_per_degree_longitude = 111412.84 * math.cos(
        math.radians(center_latitude)
    ) - 93.5 * math.cos(3.0 * math.radians(center_latitude))
    dx = (east - west) * metres_per_degree_longitude / (grid_size - 1)
    dy = (north - south) * metres_per_degree_latitude / (grid_size - 1)
    return np.ascontiguousarray(bed), float(dx), float(dy)


def _synthetic_event(terrain):
    """Create a deterministic display event without modifying terrain."""
    ny, nx = terrain.shape
    barrier = np.zeros_like(terrain, dtype=np.float32)
    dam_x = max(2, min(nx - 3, int(round(0.38 * (nx - 1)))))
    center = ny // 2
    breach_half_width = max(1, int(round(0.045 * ny)))
    barrier[:, dam_x] = 36.0
    barrier[
        center - breach_half_width:center + breach_half_width + 1, dam_x
    ] = 0.0

    upstream = terrain[:, :dam_x]
    reservoir_surface = float(np.percentile(upstream, 35.0) + 28.0)
    initial_depth = np.zeros_like(terrain, dtype=np.float32)
    initial_depth[:, :dam_x] = np.maximum(
        reservoir_surface - upstream, 0.0
    )
    return barrier, initial_depth, {
        "barrier_column": dam_x,
        "barrier_height_m": 36.0,
        "breach_center_row": center,
        "breach_half_width_cells": breach_half_width,
        "reservoir_surface_elevation_m_egm96": reservoir_surface,
    }


def prepare(source_zip, output, acquired, bounds=DEFAULT_BOUNDS,
            grid_size=256):
    acquired_date = date.fromisoformat(str(acquired))
    elevation, source_metadata = read_official_hgt(source_zip)
    bounds = _validate_bounds(bounds)
    terrain, dx, dy = _bilinear_resample(elevation, bounds, grid_size)
    if not np.all(np.isfinite(terrain)):
        raise ValueError("processed NASADEM crop contains non-finite heights")
    barrier, initial_depth, event = _synthetic_event(terrain)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        bed=terrain,
        raw_bed=terrain,
        barrier=barrier,
        initial_depth=initial_depth,
        cell_size_x=np.float64(dx),
        cell_size_y=np.float64(dy),
        terrain_id=np.asarray(PRODUCER_GRANULE_ID),
        synthetic=np.asarray(False),
    )
    west, south, east, north = bounds
    manifest = {
        "schema_version": 1,
        "processing_version": PROCESSING_VERSION,
        "terrain_id": PRODUCER_GRANULE_ID,
        "source": {
            "product": PRODUCT,
            "tile": TILE,
            "producer_granule_id": PRODUCER_GRANULE_ID,
            "cmr_granule_id": CMR_GRANULE_ID,
            "doi": SOURCE_DOI,
            "url": SOURCE_URL,
            "acquired": acquired_date.isoformat(),
            "posting": "1 arc-second (approximately 30 m)",
            "horizontal_crs": "EPSG:4326 (WGS 84)",
            "vertical_datum": "EGM96 geoid",
            "elevation_units": "metres",
            **source_metadata,
        },
        "crop": {
            "bounds_wgs84": {
                "west": west, "south": south,
                "east": east, "north": north,
            },
            "local_origin_wgs84": [west, south],
            "local_crs": (
                "local equirectangular east/north metric grid; distances "
                "evaluated at crop-center latitude"
            ),
            "shape": list(terrain.shape),
            "cell_size_m": [dx, dy],
            "bed_min_m_egm96": float(np.min(terrain)),
            "bed_max_m_egm96": float(np.max(terrain)),
            "bed_sha256": _array_sha256(terrain),
            "barrier_sha256": _array_sha256(barrier),
            "initial_depth_sha256": _array_sha256(initial_depth),
            "asset_sha256": _sha256_file(output),
        },
        "processing": [
            "validate exact 3601x3601 big-endian signed-int16 HGT member",
            "crop WGS84 bounds and bilinearly sample to a bounded metric grid",
            "preserve NASADEM elevations without smoothing or display exaggeration",
            "store synthetic barrier and initial water depth as separate arrays",
        ],
        "nodata_treatment": (
            "none: NASADEM_HGT is void-filled; -32768 is rejected"
        ),
        "synthetic_event": {
            "synthetic_breach": True,
            "historical_reconstruction": False,
            "warning": "synthetic breach—not a historical reconstruction",
            **event,
        },
        "limitations": [
            "30 m surface DEM cannot resolve the dam, narrow channels, or buildings",
            "no bathymetry, calibrated breach hydrograph, infiltration, or uncertainty",
            "not an inundation forecast or safety decision product",
        ],
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--acquired", required=True,
        help="Earthdata acquisition date in YYYY-MM-DD format",
    )
    parser.add_argument("--grid-size", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "--bounds", type=float, nargs=4, metavar=("W", "S", "E", "N"),
        default=DEFAULT_BOUNDS,
    )
    args = parser.parse_args()
    _, manifest_path, manifest = prepare(
        args.source_zip, args.output, args.acquired,
        bounds=args.bounds, grid_size=args.grid_size,
    )
    print(json.dumps({
        "asset": str(args.output),
        "manifest": str(manifest_path),
        "shape": manifest["crop"]["shape"],
        "cell_size_m": manifest["crop"]["cell_size_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
