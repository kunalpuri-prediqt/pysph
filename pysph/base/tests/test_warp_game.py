import json

import numpy as np
import pytest


wp = pytest.importorskip("warp")

from pysph.base.warp_game import (
    GameplayDamBreakConfig,
    TANK_BOUNDS,
    WarpGameplayDamBreakSimulation,
    _solve_lambdas,
    _solve_position_delta,
)


def _simulation(**updates):
    values = dict(
        dx=0.2, steps=2, projection_iterations=2,
        dt=1.0 / 60.0, device="cuda:0",
    )
    values.update(updates)
    return WarpGameplayDamBreakSimulation(GameplayDamBreakConfig(**values))


def test_gameplay_config_is_self_describing_and_validated():
    config = GameplayDamBreakConfig(dx=0.2, steps=4).validate()
    encoded = config.to_dict()
    assert encoded["solver_family"] == "gameplay-pbf"
    assert encoded["approximate"] is True
    assert GameplayDamBreakConfig.from_mapping(encoded).dx == 0.2
    assert config.body_center_z == pytest.approx(0.10)
    json.dumps(encoded)
    with pytest.raises(ValueError, match="projection_iterations"):
        GameplayDamBreakConfig(projection_iterations=0).validate()
    with pytest.raises(ValueError, match="neighbor skin"):
        GameplayDamBreakConfig(projection_iterations=6).validate()
    with pytest.raises(ValueError, match="xsph_coefficient"):
        GameplayDamBreakConfig(xsph_coefficient=1.1).validate()


def test_gameplay_projection_reduces_density_constraint_and_stays_bounded():
    simulation = _simulation(obstacle_mode="none")
    simulation.initialize()
    before = simulation.metrics()["constraint_rms"]
    simulation.step()
    metrics = simulation.metrics()
    xyz = simulation.snapshot(include_solids=False)["xyz"]

    assert metrics["all_finite"]
    assert metrics["constraint_rms"] < before
    assert metrics["solver_family"] == "gameplay-pbf"
    assert metrics["approximate"] is True
    xmin, xmax, ymin, ymax, zmin, zmax = TANK_BOUNDS
    radius = float(simulation.particle_radius)
    assert np.min(xyz[:, 0]) >= xmin + radius - 2.0e-6
    assert np.max(xyz[:, 0]) <= xmax - radius + 2.0e-6
    assert np.min(xyz[:, 1]) >= ymin + radius - 2.0e-6
    assert np.max(xyz[:, 1]) <= ymax - radius + 2.0e-6
    assert np.min(xyz[:, 2]) >= zmin + radius - 2.0e-6
    assert np.max(xyz[:, 2]) <= zmax - radius + 2.0e-6
    assert metrics["max_correction"] >= 0.0


def test_render_anisotropy_is_bounded_orthonormal_and_display_only():
    simulation = _simulation(obstacle_mode="none")
    simulation.initialize()
    before = simulation.x.numpy().copy()
    attributes = simulation.render_attributes()
    after = simulation.x.numpy()
    axes = attributes["render_axes"]
    scale = attributes["render_scale"]
    base = simulation.config.render_splat_scale * simulation.config.dx

    np.testing.assert_array_equal(after, before)
    assert attributes["render_xyz"].shape == before.shape
    assert axes.shape == (simulation.fluid_count, 3, 3)
    assert scale.shape == before.shape
    assert np.all(np.isfinite(attributes["render_xyz"]))
    assert np.all(np.isfinite(axes))
    assert np.all(np.isfinite(scale))
    identity = np.broadcast_to(np.eye(3), axes.shape)
    np.testing.assert_allclose(
        np.swapaxes(axes, 1, 2) @ axes, identity, atol=2.0e-4
    )
    assert np.all(np.linalg.det(axes) > 0.999)
    assert np.min(scale) >= (
        base * simulation.config.render_anisotropy_min - 2.0e-6
    )
    assert np.max(scale) <= (
        base * simulation.config.render_anisotropy_max + 2.0e-6
    )


def test_sparse_render_neighborhood_uses_deterministic_isotropic_fallback():
    simulation = _simulation(obstacle_mode="none")
    simulation.initialize()
    xyz = simulation.x.numpy()
    xyz[0] = np.asarray([5.0, 0.0, 1.3], dtype=np.float32)
    simulation.x = wp.array(xyz, dtype=wp.vec3, device=simulation.device)
    first = simulation.render_attributes()
    second = simulation.render_attributes()
    base = simulation.config.render_splat_scale * simulation.config.dx

    assert first["render_neighbors"][0] == 1
    np.testing.assert_array_equal(first["render_axes"][0], np.eye(3))
    np.testing.assert_allclose(first["render_scale"][0], base)
    np.testing.assert_array_equal(
        first["render_axes"][0], second["render_axes"][0]
    )
    np.testing.assert_array_equal(
        first["render_scale"][0], second["render_scale"][0]
    )


def test_three_projection_iterations_reduce_constraint_monotonically():
    residuals = []
    initial = None
    for iterations in (1, 2, 3):
        simulation = _simulation(
            obstacle_mode="none", gz=0.0,
            projection_iterations=iterations,
        )
        simulation.initialize()
        initial = simulation.metrics()["constraint_rms"]
        simulation.step()
        residuals.append(simulation.metrics()["constraint_rms"])

    assert residuals[0] < initial
    assert residuals[1] < residuals[0]
    assert residuals[2] < residuals[1]


def test_unbounded_pair_position_correction_is_symmetric():
    device = wp.get_device("cuda:0")
    positions = np.asarray([
        [1.0, 0.0, 0.5],
        [1.2, 0.0, 0.5],
    ], dtype=np.float32)
    x = wp.array(positions, dtype=wp.vec3, device=device)
    density = wp.zeros(2, dtype=wp.float32, device=device)
    constraint = wp.zeros(2, dtype=wp.float32, device=device)
    lambdas = wp.zeros(2, dtype=wp.float32, device=device)
    delta = wp.zeros(2, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device, dtype=wp.float32)
    support = np.float32(0.4)
    query_radius = np.float32(0.6)
    grid.build(x, query_radius)
    wp.launch(
        _solve_lambdas, dim=2,
        inputs=[
            grid.id, x, density, constraint, lambdas, np.float32(8.0),
            np.float32(1000.0), support, query_radius,
            np.float32(1.0e-6),
        ], device=device,
    )
    wp.launch(
        _solve_position_delta, dim=2,
        inputs=[
            grid.id, x, lambdas, delta, np.float32(8.0),
            np.float32(1000.0), support, query_radius,
            np.float32(0.001), np.float32(4.0),
        ], device=device,
    )
    correction = delta.numpy()

    assert np.linalg.norm(correction[0]) > 0.0
    np.testing.assert_allclose(correction[0], -correction[1], atol=1.0e-7)


def test_tank_projection_recovers_a_particle_outside_a_floor_corner():
    simulation = _simulation(obstacle_mode="none", gz=0.0)
    simulation.initialize()
    xyz = simulation.x.numpy()
    xyz[0] = np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    simulation.x = wp.array(xyz, dtype=wp.vec3, device=simulation.device)
    simulation.x_prev = wp.array(xyz, dtype=wp.vec3, device=simulation.device)
    simulation.step()
    corrected = simulation.snapshot(include_solids=False)["xyz"][0]
    radius = float(simulation.particle_radius)

    assert corrected[0] >= TANK_BOUNDS[0] + radius - 2.0e-6
    assert corrected[1] >= TANK_BOUNDS[2] + radius - 2.0e-6
    assert corrected[2] >= TANK_BOUNDS[4] + radius - 2.0e-6
    assert np.all(np.isfinite(corrected))


def test_gameplay_fluid_collision_moves_rigid_box_and_preserves_shell():
    simulation = _simulation(obstacle_mode="floating")
    simulation.initialize()
    xyz = simulation.x.numpy()
    velocity = simulation.velocity.numpy()
    center = np.asarray([
        simulation.config.body_center_x,
        simulation.config.body_center_y,
        simulation.config.body_center_z,
    ], dtype=np.float32)
    # Put a compact packet just inside the left face, moving right. The
    # position projection pushes it left and applies the opposite impulse to
    # the gameplay body.
    packet = 1
    xyz[:packet] = center + np.asarray([-0.14, 0.0, 0.0], dtype=np.float32)
    velocity[:packet] = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    simulation.x = wp.array(xyz, dtype=wp.vec3, device=simulation.device)
    simulation.x_prev = wp.array(xyz, dtype=wp.vec3, device=simulation.device)
    simulation.velocity = wp.array(
        velocity, dtype=wp.vec3, device=simulation.device
    )
    simulation.step()
    metrics = simulation.metrics()
    shell = simulation.snapshot()["xyz"][-metrics["body_particles"]:]
    distance = np.linalg.norm(shell - shell[0], axis=1)
    reference = simulation.body_reference_host
    reference_distance = np.linalg.norm(reference - reference[0], axis=1)

    assert metrics["all_finite"]
    assert metrics["body_vc"][0] > 0.0
    assert np.linalg.norm(metrics["body_omega"]) <= (
        simulation.config.body_angular_speed_limit + 1.0e-10
    )
    assert np.linalg.norm(metrics["contact_impulse"]) <= (
        simulation.config.body_impulse_limit + 1.0e-10
    )
    np.testing.assert_allclose(distance, reference_distance, atol=2.0e-6)
    assert TANK_BOUNDS[0] < metrics["body_cm"][0] < TANK_BOUNDS[1]
    assert TANK_BOUNDS[4] < metrics["body_cm"][2] < TANK_BOUNDS[5]


def test_gameplay_box_uses_oriented_support_instead_of_hovering_sphere():
    simulation = _simulation(obstacle_mode="floating", body_center_z=0.10)
    simulation.initialize()
    angle = np.pi / 4.0
    simulation.body_orientation = wp.array(
        [wp.quatd(0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0))],
        dtype=wp.quatd, device=simulation.device,
    )
    simulation.step()
    metrics = simulation.metrics()
    shell = simulation.snapshot()["xyz"][-metrics["body_particles"]:]
    expected_support = (
        simulation.body_half_size[0] * np.sin(angle)
        + simulation.body_half_size[2] * np.cos(angle)
    )

    assert metrics["body_cm"][2] == pytest.approx(
        expected_support, abs=2.0e-6
    )
    assert np.min(shell[:, 2]) >= TANK_BOUNDS[4] - 2.0e-6
    assert metrics["body_cm"][2] < np.linalg.norm(simulation.body_half_size)


def test_gameplay_save_uses_existing_snapshot_schema(tmp_path):
    simulation = _simulation(obstacle_mode="none", steps=1)
    simulation.initialize()
    simulation.step()
    path = simulation.save(tmp_path / "gameplay.npz")
    with np.load(path, allow_pickle=False) as data:
        for name in ("xyz", "h", "rho", "p", "speed", "kind", "level"):
            assert name in data
        metrics = json.loads(str(data["metrics"].item()))
        config = json.loads(str(data["config"].item()))
    assert metrics["solver_family"] == "gameplay-pbf"
    assert config["solver_family"] == "gameplay-pbf"
