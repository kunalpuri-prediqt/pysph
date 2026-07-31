import numpy as np
import pytest

from pysph.base.warp_adaptive import (
    DamBreakConfig,
    TwoLevelAdaptiveController,
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


def test_invalid_controller_configuration_is_rejected():
    with pytest.raises(ValueError, match="hdx"):
        TwoLevelAdaptiveController(hdx=0)
    with pytest.raises(ValueError, match="six"):
        TwoLevelAdaptiveController(fine_bounds=(0, 1))
    with pytest.raises(ValueError, match="increasing"):
        TwoLevelAdaptiveController(
            fine_bounds=(1, 0, -1, 1, -1, 1)
        )
