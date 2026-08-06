import json

import numpy as np
import pytest


pytest.importorskip("warp")

from pysph.base.warp_shallow_water import (
    ShallowWaterConfig,
    WarpShallowWaterSimulation,
    load_terrain_asset,
    make_synthetic_valley,
)


def _simulation(bed, depth, **updates):
    ny, nx = np.asarray(bed).shape
    values = dict(
        steps=4, grid_nx=nx, grid_ny=ny,
        cell_size_x=10.0, cell_size_y=10.0,
        cfl=0.30, dt_max=0.05, dry_depth=1.0e-4,
        manning=0.0, boundary="closed", device="cuda:0",
        terrain_id="synthetic-test",
    )
    values.update(updates)
    return WarpShallowWaterSimulation(
        ShallowWaterConfig(**values), bed=bed, initial_depth=depth
    )


def _cpu_rusanov_step(depth, momentum, dt, dx, gravity=9.81):
    """Independent fp64 one-dimensional flat-bed Rusanov reference."""
    depth = np.asarray(depth, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    count = depth.size
    flux = np.zeros((count + 1, 2), dtype=np.float64)
    for interface in range(count + 1):
        if interface == 0:
            hl = hr = depth[0]
            ql, qr = -momentum[0], momentum[0]
        elif interface == count:
            hl = hr = depth[-1]
            ql, qr = momentum[-1], -momentum[-1]
        else:
            hl, hr = depth[interface - 1], depth[interface]
            ql, qr = momentum[interface - 1], momentum[interface]
        ul = ql / hl if hl > 1.0e-4 else 0.0
        ur = qr / hr if hr > 1.0e-4 else 0.0
        fl = np.asarray([ql, ql * ul + 0.5 * gravity * hl * hl])
        fr = np.asarray([qr, qr * ur + 0.5 * gravity * hr * hr])
        wave = max(abs(ul) + np.sqrt(gravity * hl),
                   abs(ur) + np.sqrt(gravity * hr))
        flux[interface] = 0.5 * (fl + fr) - 0.5 * wave * np.asarray(
            [hr - hl, qr - ql]
        )
    state = np.column_stack((depth, momentum))
    state -= dt / dx * (flux[1:] - flux[:-1])
    state[state[:, 0] <= 0.0] = 0.0
    return state[:, 0], state[:, 1]


def test_config_is_self_describing_and_rejects_invalid_cfl():
    config = ShallowWaterConfig(grid_nx=8, grid_ny=6).validate()
    encoded = config.to_dict()
    assert encoded["solver_family"] == "geospatial-swe"
    assert encoded["depth_averaged"] is True
    assert ShallowWaterConfig.from_mapping(encoded).grid_nx == 8
    json.dumps(encoded)
    with pytest.raises(ValueError, match="CFL"):
        ShallowWaterConfig(cfl=0.7).validate()


def test_synthetic_valley_is_deterministic_and_contains_a_breach():
    first = make_synthetic_valley(64, 48)
    second = make_synthetic_valley(64, 48)
    for name in ("bed", "raw_bed", "barrier", "initial_depth"):
        np.testing.assert_array_equal(first[name], second[name])
    barrier = first["barrier"]
    assert np.max(barrier) > 0.0
    assert np.count_nonzero(barrier == 0.0) > 0
    assert np.max(first["initial_depth"]) > 0.0

    simulation = WarpShallowWaterSimulation(ShallowWaterConfig(
        steps=1, grid_nx=64, grid_ny=48, device="cuda:0",
    ))
    snapshot = simulation.initialize()
    np.testing.assert_array_equal(
        snapshot["bed"],
        snapshot["terrain_bed"] + snapshot["synthetic_barrier"],
    )
    assert np.count_nonzero(snapshot["synthetic_barrier"]) > 0


def test_lake_at_rest_stays_balanced_over_nonflat_bed():
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    bed = 0.4 * np.sin(np.pi * xx) ** 2 + 0.2 * yy
    surface = 2.0
    depth = (surface - bed).astype(np.float32)
    simulation = _simulation(bed, depth, steps=200, dt_max=0.02)
    simulation.initialize()
    for _ in range(simulation.config.steps):
        simulation.step()
    snapshot = simulation.snapshot()
    metrics = simulation.metrics()
    speed = snapshot["water_speed"]
    surface_error = snapshot["water_surface"] - surface

    assert metrics["all_finite"]
    assert np.max(speed) < 2.0e-5
    assert np.max(np.abs(surface_error)) < 2.0e-5


def test_closed_flat_bed_dam_break_is_positive_and_conservative():
    bed = np.zeros((20, 48), dtype=np.float32)
    depth = np.zeros_like(bed)
    depth[:, :18] = 2.0
    simulation = _simulation(bed, depth, steps=500, dt_max=0.03)
    simulation.initialize()
    for _ in range(simulation.config.steps):
        simulation.step()
    snapshot = simulation.snapshot()
    metrics = simulation.metrics()

    assert metrics["all_finite"]
    assert np.min(snapshot["water_depth"]) >= 0.0
    assert abs(metrics["volume_drift"]) <= 1.0e-4
    assert np.count_nonzero(snapshot["water_depth"][:, 18:] > 1.0e-3) > 0


def test_flat_dam_break_matches_independent_cpu_rusanov_profile():
    bed = np.zeros((6, 64), dtype=np.float32)
    depth = np.zeros_like(bed)
    depth[:, :22] = 1.8
    simulation = _simulation(bed, depth, steps=80, dt_max=0.025)
    simulation.initialize()
    reference_h = depth[0].astype(np.float64)
    reference_q = np.zeros(64, dtype=np.float64)
    for _ in range(simulation.config.steps):
        progress = simulation.step()
        reference_h, reference_q = _cpu_rusanov_step(
            reference_h, reference_q, progress["dt"],
            simulation.config.cell_size_x,
        )
    snapshot = simulation.snapshot()
    assert np.mean(np.abs(snapshot["water_depth"][2] - reference_h)) < 2.0e-5
    assert np.mean(np.abs(snapshot["momentum_x"][2] - reference_q)) < 3.0e-5
    assert np.count_nonzero(snapshot["water_depth"][2, 22:] > 1.0e-3) > 0


def test_manning_friction_reduces_peak_speed_and_outflow_stays_finite():
    bed = np.zeros((12, 48), dtype=np.float32)
    depth = np.zeros_like(bed)
    depth[:, :16] = 2.0
    inviscid = _simulation(bed, depth, steps=120, manning=0.0)
    frictional = _simulation(bed, depth, steps=120, manning=0.08)
    outflow = _simulation(
        bed, depth, steps=120, manning=0.03, boundary="outflow"
    )
    for simulation in (inviscid, frictional, outflow):
        simulation.initialize()
        for _ in range(simulation.config.steps):
            simulation.step()
        assert simulation.metrics()["all_finite"]
    assert frictional.metrics()["peak_speed"] < inviscid.metrics()["peak_speed"]


def test_save_and_processed_asset_loader_are_pickle_free(tmp_path):
    bed = np.zeros((8, 10), dtype=np.float32)
    depth = np.ones_like(bed)
    simulation = _simulation(bed, depth, steps=1)
    simulation.initialize()
    simulation.step()
    output = simulation.save(tmp_path / "result.npz")
    with np.load(output, allow_pickle=False) as data:
        assert "bed" in data
        assert "water_depth" in data
        metrics = json.loads(str(data["metrics"]))
        assert metrics["solver_family"] == "geospatial-swe"

    terrain_path = tmp_path / "terrain.npz"
    barrier = np.zeros_like(bed)
    barrier[:, 4] = 0.5
    np.savez_compressed(
        terrain_path, bed=bed, barrier=barrier, initial_depth=depth,
        cell_size_x=np.float64(28.0), cell_size_y=np.float64(31.0),
        terrain_id=np.asarray("NASADEM_HGT_n43e006"),
        synthetic=np.asarray(False),
    )
    terrain = load_terrain_asset(terrain_path)
    np.testing.assert_array_equal(terrain["bed"], bed)
    np.testing.assert_array_equal(terrain["initial_depth"], depth)
    restored = WarpShallowWaterSimulation(ShallowWaterConfig(
        steps=1, terrain_path=str(terrain_path), device="cuda:0",
    ))
    restored_snapshot = restored.initialize()
    np.testing.assert_array_equal(restored_snapshot["terrain_bed"], bed)
    np.testing.assert_array_equal(restored_snapshot["synthetic_barrier"], barrier)
    assert restored.config.cell_size_x == pytest.approx(28.0)
    assert restored.config.cell_size_y == pytest.approx(31.0)
    assert restored.metrics()["synthetic_terrain"] is False
