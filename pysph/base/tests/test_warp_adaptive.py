import numpy as np
import pytest

from pysph.base.warp_adaptive import (
    DamBreakConfig,
    TwoLevelAdaptiveController,
    WarpDamBreakSimulation,
    gaussian_hill_height,
    make_gaussian_hill_particles,
    particle_state,
)


def _state(n=1, x=0.5):
    x = np.full(n, x, dtype=np.float64)
    values = np.arange(n, dtype=np.float64)
    return {
        "x": x,
        "y": 0.1 * values,
        "z": 0.05 * values,
        "h": np.full(n, 0.2),
        "m": np.linspace(2.0, 2.0 + n - 1, n),
        "rho": np.full(n, 1000.0),
        "p": np.full(n, 12.0),
        "cs": np.full(n, 20.0),
        "u": np.full(n, 1.25),
        "v": np.full(n, -0.5),
        "w": np.full(n, 0.75),
        "au": np.zeros(n),
        "av": np.zeros(n),
        "aw": np.zeros(n),
        "arho": np.zeros(n),
        "ax": np.zeros(n),
        "ay": np.zeros(n),
        "az": np.zeros(n),
        "x0": x.copy(),
        "y0": 0.1 * values,
        "z0": 0.05 * values,
        "u0": np.full(n, 1.25),
        "v0": np.full(n, -0.5),
        "w0": np.full(n, 0.75),
        "rho0": np.full(n, 1000.0),
        "pid": np.zeros(n, dtype=np.int32),
        "gid": np.full(n, np.iinfo(np.uint32).max, dtype=np.uint32),
        "tag": np.zeros(n, dtype=np.int32),
    }


def _controller(max_splits=8):
    return TwoLevelAdaptiveController(
        hdx=1.0,
        fine_bounds=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        max_splits_per_adapt=max_splits,
    )


def test_initialization_assigns_unique_ids_and_metadata():
    state = _controller().initialize_state(_state(4))
    assert np.array_equal(state["pid"], [1, 2, 3, 4])
    assert np.array_equal(state["family_id"], [1, 2, 3, 4])
    assert np.array_equal(state["level"], [0, 0, 0, 0])


def test_octant_split_preserves_mass_and_linear_momentum():
    controller = _controller()
    before = controller.initialize_state(_state())
    mass0, momentum0 = controller.invariants(before)

    after, stats = controller.adapt(before)
    mass1, momentum1 = controller.invariants(after)

    assert stats.split_parents == 1
    assert stats.created_children == 8
    assert len(after["x"]) == 8
    assert np.all(after["level"] == 1)
    assert np.all(after["m"] == before["m"][0] / 8.0)
    assert np.all(after["h"] == before["h"][0] / 2.0)
    assert np.isclose(mass1, mass0)
    assert np.allclose(momentum1, momentum0)
    assert stats.mass_residual <= 1.0e-15
    assert stats.momentum_residual <= 1.0e-15


def test_octant_split_is_centered_and_isotropic():
    controller = _controller()
    after, _ = controller.adapt(_state())
    mass = after["m"]
    for axis, center in (("x", 0.5), ("y", 0.0), ("z", 0.0)):
        assert np.isclose(np.sum(mass * after[axis]) / np.sum(mass), center)
        delta = np.unique(np.round(abs(after[axis] - center), 12))
        assert np.array_equal(delta, [0.05])


def test_split_then_complete_family_merge_recovers_parent_state():
    controller = _controller()
    original = controller.initialize_state(_state())
    split, split_stats = controller.adapt(original)
    fine = split["level"] == 1
    split["x"][fine] += 2.0
    split["x0"][fine] += 2.0

    merged, merge_stats = controller.adapt(split)

    assert split_stats.split_parents == 1
    assert merge_stats.merged_families == 1
    assert merge_stats.removed_children == 8
    assert len(merged["x"]) == 1
    assert merged["level"][0] == 0
    assert np.isclose(merged["m"][0], original["m"][0])
    assert np.isclose(merged["h"][0], original["h"][0])
    for name in ("u", "v", "w", "rho", "p", "cs"):
        assert np.isclose(merged[name][0], original[name][0])
    assert merge_stats.mass_residual <= 1.0e-15
    assert merge_stats.momentum_residual <= 1.0e-15


def test_incomplete_family_is_not_merged():
    controller = _controller()
    split, _ = controller.adapt(_state())
    state = {name: values[:-1] for name, values in split.items()}
    state["x"] += 2.0
    merged, stats = controller.adapt(state)
    assert stats.merged_families == 0
    assert len(merged["x"]) == 7


def test_split_cap_bounds_particle_growth():
    controller = _controller(max_splits=2)
    after, stats = controller.adapt(_state(5))
    assert stats.split_parents == 2
    assert len(after["x"]) == 5 - 2 + 16


def test_icosa13_split_preserves_invariants_and_reconstructs_linear_field():
    state = _state(7)
    state["x"] = np.asarray([0.5, 0.6, 0.4, 0.5, 0.5, 0.5, 0.5])
    state["y"] = np.asarray([0.0, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0])
    state["z"] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1])
    state["x0"] = state["x"].copy()
    state["y0"] = state["y"].copy()
    state["z0"] = state["z"].copy()
    state["p"] = 7.0 + 2.0 * state["x"] - 3.0 * state["y"] + 0.5 * state["z"]
    state["u"] = -1.0 + state["x"] + 0.25 * state["y"]
    state["u0"] = state["u"].copy()
    controller = TwoLevelAdaptiveController(
        hdx=1.0,
        fine_bounds=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        max_splits_per_adapt=1,
        split_stencil="icosa13",
    )
    before_mass, before_momentum = controller.invariants(state)
    after, stats = controller.adapt(state)
    family = after["family_id"] == 1
    expected_p = (
        7.0 + 2.0 * after["x"][family] - 3.0 * after["y"][family]
        + 0.5 * after["z"][family]
    )
    after_mass, after_momentum = controller.invariants(after)
    assert stats.created_children == 13
    assert np.count_nonzero(family) == 13
    assert np.isclose(after["m"][family].sum(), state["m"][0])
    assert np.isclose(after["m"][family].min() / after["m"][family].max(),
                      0.656566, rtol=2e-5)
    assert np.allclose(after["p"][family], expected_p, atol=1e-12)
    assert np.isclose(before_mass, after_mass)
    assert np.allclose(before_momentum, after_momentum, atol=1e-12)


def test_icosa13_complete_family_merge_recovers_parent():
    controller = TwoLevelAdaptiveController(
        hdx=1.0, split_stencil="icosa13",
        fine_bounds=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )
    original = controller.initialize_state(_state())
    split, _ = controller.adapt(original)
    split["x"] += 2.0
    split["x0"] += 2.0
    merged, stats = controller.adapt(split)
    assert stats.merged_families == 1
    assert stats.removed_children == 13
    assert len(merged["x"]) == 1
    assert np.isclose(merged["m"][0], original["m"][0])
    assert np.isclose(merged["h"][0], original["h"][0])


def test_merge_hysteresis_prevents_boundary_thrashing():
    controller = TwoLevelAdaptiveController(
        hdx=1.0, hysteresis=0.2,
        fine_bounds=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )
    split, _ = controller.adapt(_state())
    split["x"] += 0.6
    held, stats = controller.adapt(split)
    assert stats.merged_families == 0
    assert len(held["x"]) == 8
    held["x"] += 0.2
    merged, stats = controller.adapt(held)
    assert stats.merged_families == 1
    assert len(merged["x"]) == 1


def test_icosa13_shifting_preserves_constant_and_linear_fields():
    state = _state(7)
    state["x"] = np.asarray([0.5, 0.62, 0.38, 0.5, 0.5, 0.5, 0.5])
    state["y"] = np.asarray([0.0, 0.0, 0.0, 0.12, -0.12, 0.0, 0.0])
    state["z"] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.12, -0.12])
    for axis in ("x", "y", "z"):
        state[f"{axis}0"] = state[axis].copy()
    state["rho"][:] = 1000.0
    state["p"] = 2.0 * state["x"] - state["y"] + 0.5 * state["z"]
    controller = TwoLevelAdaptiveController(
        hdx=1.0, split_stencil="icosa13", max_splits_per_adapt=2,
        shift_iterations=1, shift_coefficient=0.05,
        fine_bounds=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )
    mass0, momentum0 = controller.invariants(state)
    shifted, stats = controller.adapt(state)
    expected = (
        2.0 * shifted["x"] - shifted["y"] + 0.5 * shifted["z"]
    )
    mass1, momentum1 = controller.invariants(shifted)
    assert stats.shifted_particles > 0
    assert stats.max_shift <= 0.05 * np.max(shifted["h"])
    assert np.all(shifted["rho"] == 1000.0)
    assert np.allclose(shifted["p"], expected, atol=1e-11)
    assert np.isclose(mass1, mass0)
    assert np.allclose(momentum1, momentum0, atol=1e-12)


def test_particles_outside_target_region_remain_coarse():
    controller = _controller()
    after, stats = controller.adapt(_state(x=3.0))
    assert stats.split_parents == 0
    assert len(after["x"]) == 1
    assert after["level"][0] == 0


def test_config_round_trip_and_validation():
    config = DamBreakConfig(
        resolution_mode="adaptive",
        steps=3,
        fine_bounds=(1, 2, -1, 1, 0, 1),
    )
    restored = DamBreakConfig.from_mapping(config.to_dict())
    assert restored == config
    with pytest.raises(ValueError, match="resolution_mode"):
        DamBreakConfig(resolution_mode="mystery").validate()
    with pytest.raises(ValueError, match="steps"):
        DamBreakConfig(steps=0).validate()


def test_config_normalizes_legacy_and_explicit_obstacle_modes():
    fixed = DamBreakConfig.from_mapping({"with_obstacle": True})
    absent = DamBreakConfig.from_mapping({"with_obstacle": False})
    floating = DamBreakConfig.from_mapping({
        "with_obstacle": False,
        "obstacle_mode": "floating",
    })
    assert fixed.obstacle_mode == "fixed"
    assert absent.obstacle_mode == "none"
    assert floating.obstacle_mode == "floating"
    assert floating.with_obstacle is True
    hill = DamBreakConfig.from_mapping({
        "solver_family": "terrain-wcsph",
        "resolution_mode": "uniform",
        "obstacle_mode": "hill",
    })
    assert hill.obstacle_mode == "hill"
    assert hill.solver_family == "terrain-wcsph"
    defaults = DamBreakConfig().to_dict()
    assert defaults["obstacle_mode"] == "fixed"
    assert defaults["variable_h_correction"] is True
    assert defaults["adapt_hysteresis"] == 0.0
    assert defaults["shift_iterations"] == 0


def test_config_rejects_invalid_floating_body_values():
    with pytest.raises(ValueError, match="obstacle_mode"):
        DamBreakConfig(obstacle_mode="drifting").validate()
    with pytest.raises(ValueError, match="body_density"):
        DamBreakConfig(body_density=0.0).validate()
    with pytest.raises(ValueError, match="dimensions"):
        DamBreakConfig(body_height=0.0).validate()
    with pytest.raises(ValueError, match="contact_restitution"):
        DamBreakConfig(contact_restitution=0.0).validate()
    with pytest.raises(ValueError, match="stiffness"):
        DamBreakConfig(contact_stiffness=0.0).validate()
    with pytest.raises(ValueError, match="increasing"):
        DamBreakConfig(
            contact_bounds=(1.0, 0.0, -0.25, 0.25, 0.0, 1.5)
        ).validate()
    with pytest.raises(ValueError, match="hill height"):
        DamBreakConfig(hill_height=0.0).validate()


def test_gaussian_hill_samples_are_deterministic_bounded_and_floor_exclusive():
    first = make_gaussian_hill_particles(dx=0.05)
    second = make_gaussian_hill_particles(dx=0.05)
    np.testing.assert_array_equal(first, second)
    assert first.shape[1] == 3
    assert np.all(np.isfinite(first))
    assert np.min(first[:, 2]) >= 0.05
    assert np.max(first[:, 2]) <= 0.35 + 0.05
    assert np.min(first[:, 0]) > 0.0
    assert np.max(first[:, 0]) < 161.0 / 30.0
    assert np.min(first[:, 1]) > -0.25
    assert np.max(first[:, 1]) < 0.25
    envelope = gaussian_hill_height(first[:, 0], first[:, 1])
    assert np.all(first[:, 2] <= envelope + 0.25 * 0.05 + 1.0e-12)
    with pytest.raises(ValueError, match="center"):
        make_gaussian_hill_particles(dx=0.05, center_x=-1.0)
    with pytest.raises(ValueError, match="tank height"):
        make_gaussian_hill_particles(dx=0.05, height=1.6)


def test_uniform_terrain_wcsph_hill_is_stationary_and_finite():
    simulation = WarpDamBreakSimulation(DamBreakConfig(
        solver_family="terrain-wcsph",
        resolution_mode="uniform",
        obstacle_mode="hill",
        dx=0.1,
        steps=8,
        n_damp=2,
        device="cuda:0",
    ))
    initial = simulation.initialize()
    mask = initial["kind"] == 2
    hill_before = initial["xyz"][mask].copy()
    assert len(hill_before) == initial["counts"]["obstacle"] > 0
    assert initial["hill"]["height"] == pytest.approx(0.35)
    hill_state = particle_state(simulation.particles[2])
    np.testing.assert_allclose(hill_state["h"], 0.13)
    np.testing.assert_allclose(hill_state["m"], 1.0)
    np.testing.assert_allclose(hill_state["rho"], 1000.0)
    for _ in range(simulation.config.steps):
        simulation.step()
    final = simulation.snapshot(include_solids=True)
    metrics = simulation.metrics()
    np.testing.assert_array_equal(final["xyz"][final["kind"] == 2], hill_before)
    assert metrics["solver_family"] == "terrain-wcsph"
    assert metrics["hill_particles"] == len(hill_before)
    assert metrics["all_finite"]
    assert abs(metrics["mass_drift"]) <= 1.0e-12


def test_invalid_controller_configuration_is_rejected():
    with pytest.raises(ValueError, match="hdx"):
        TwoLevelAdaptiveController(hdx=0)
    with pytest.raises(ValueError, match="six"):
        TwoLevelAdaptiveController(fine_bounds=(0, 1))
    with pytest.raises(ValueError, match="increasing"):
        TwoLevelAdaptiveController(
            fine_bounds=(1, 0, -1, 1, -1, 1)
        )


def _crossing_cube_state():
    coordinates = np.asarray([
        (x, y, z)
        for x in (0.35, 0.50, 0.65)
        for y in (-0.15, 0.0, 0.15)
        for z in (0.20, 0.35, 0.50)
    ])
    state = _state(len(coordinates))
    for column, name in enumerate(("x", "y", "z")):
        state[name] = coordinates[:, column].copy()
        state[name + "0"] = coordinates[:, column].copy()
    state["m"][:] = 1.0
    state["h"][:] = 0.20
    state["u"][:] = 0.75
    state["v"][:] = -0.10
    state["w"][:] = 0.05
    state["u0"] = state["u"].copy()
    state["v0"] = state["v"].copy()
    state["w0"] = state["w"].copy()
    state["rho"][:] = 1000.0
    state["rho0"] = state["rho"].copy()
    state["p"] = 12000.0 - 9810.0 * state["z"]
    return state


def test_repeated_translating_split_merge_crossings_preserve_state():
    controller = TwoLevelAdaptiveController(
        hdx=1.3,
        fine_bounds=(0.25, 0.75, -0.25, 0.25, 0.10, 0.60),
        max_splits_per_adapt=64,
        split_stencil="icosa13",
        hysteresis=0.05,
        shift_iterations=0,
    )
    state = controller.initialize_state(_crossing_cube_state())
    mass0, momentum0 = controller.invariants(state)

    for _ in range(4):
        state, split = controller.adapt(state)
        assert split.split_parents == 27
        assert split.mass_residual <= 5.0e-9
        assert split.momentum_residual <= 5.0e-8
        assert np.allclose(state["rho"], 1000.0)
        np.testing.assert_allclose(
            state["p"], 12000.0 - 9810.0 * state["z"],
            rtol=0.0, atol=2.0e-10,
        )

        state["x"] += 1.5
        state["x0"] += 1.5
        state, merged = controller.adapt(state)
        assert merged.merged_families == 27
        assert len(state["x"]) == 27
        assert np.all(state["level"] == 0)
        np.testing.assert_allclose(
            state["p"], 12000.0 - 9810.0 * state["z"],
            rtol=0.0, atol=2.0e-10,
        )
        state["x"] -= 1.5
        state["x0"] -= 1.5

    mass1, momentum1 = controller.invariants(state)
    assert np.isclose(mass1, mass0, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(momentum1, momentum0, rtol=0.0, atol=2.0e-12)
