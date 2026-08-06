import base64
import json
from queue import Queue

import numpy as np
import pytest
from trame.app import get_server
from vtkmodules.util.numpy_support import vtk_to_numpy

from pysph.base.warp_adaptive import DamBreakConfig

from app import WarpDamBreakStudio, load_saved_result, validate_run_config
from vtk_scene import ParticleScene
from worker import FrameBuffer, SolverWorker, put_latest


def _snapshot(obstacle_mode="fixed"):
    return {
        "xyz": np.asarray([
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.2],
        ], dtype=np.float32),
        "h": np.asarray([0.1, 0.05, 0.1, 0.1], dtype=np.float32),
        "rho": np.asarray([1000, 1002, 1000, 1000], dtype=np.float32),
        "p": np.asarray([0, 50, 0, 20], dtype=np.float32),
        "speed": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "level": np.asarray([0, 1, 2, 2], dtype=np.uint8),
        "kind": np.asarray([0, 0, 1, 2], dtype=np.uint8),
        "obstacle_mode": obstacle_mode,
    }


def _gameplay_snapshot():
    snapshot = _snapshot(obstacle_mode="floating")
    count = 2
    snapshot.update({
        "solver_family": "gameplay-pbf",
        "render_xyz": snapshot["xyz"][:count].copy(),
        "render_axes": np.repeat(
            np.eye(3, dtype=np.float32)[None, :, :], count, axis=0
        ),
        "render_scale": np.full((count, 3), 0.125, dtype=np.float32),
        "render_neighbors": np.full(count, 9, dtype=np.int32),
    })
    return snapshot


def _geospatial_snapshot():
    bed = np.linspace(18.0, 4.0, 80, dtype=np.float32).reshape(8, 10)
    depth = np.zeros_like(bed)
    depth[:, :4] = 2.0
    speed = np.zeros_like(bed)
    speed[:, 3:6] = 1.25
    return {
        "solver_family": "geospatial-swe",
        "terrain_id": "synthetic-valley-fixture",
        "bed": bed,
        "water_depth": depth,
        "water_surface": bed + depth,
        "water_speed": speed,
        "cell_size": [30.0, 30.0],
        "grid_shape": [8, 10],
        "step": 4,
        "time": 1.5,
    }


def _hill_snapshot():
    snapshot = _snapshot(obstacle_mode="hill")
    snapshot["hill"] = {
        "center_x": 3.0,
        "center_y": 0.0,
        "height": 0.35,
        "radius_x": 0.40,
        "radius_y": 0.12,
    }
    return snapshot


def _config():
    return {
        "dx": 0.1,
        "steps": 10,
        "adapt_every": 2,
        "max_splits_per_adapt": 16,
        "fine_bounds": [1.0, 2.0, -0.3, 0.3, 0.0, 0.6],
        "output": "/tmp/result.npz",
    }


def test_run_config_validation_accepts_valid_values():
    config = _config()
    assert validate_run_config(config, snapshot_stride=2) is config


def test_run_config_validation_accepts_gameplay_profile():
    config = _config()
    config.update(
        solver_family="gameplay-pbf", dt=1.0 / 60.0,
        projection_iterations=3,
    )
    assert validate_run_config(config, snapshot_stride=1) is config
    config["projection_iterations"] = 0
    with pytest.raises(ValueError, match="projection iterations"):
        validate_run_config(config, snapshot_stride=1)


def test_run_config_validation_accepts_bounded_geospatial_profile():
    config = {
        "solver_family": "geospatial-swe",
        "steps": 10,
        "grid_nx": 128,
        "grid_ny": 128,
        "cfl": 0.35,
        "dt_max": 0.5,
        "manning": 0.025,
        "output": "/tmp/geospatial-result.npz",
    }
    assert validate_run_config(config, snapshot_stride=1) is config
    config["grid_nx"] = 512
    with pytest.raises(ValueError, match="bounded preset"):
        validate_run_config(config, snapshot_stride=1)
    config["grid_nx"] = 128
    config["cfl"] = 0.75
    with pytest.raises(ValueError, match="Geospatial CFL"):
        validate_run_config(config, snapshot_stride=1)


def test_run_config_validation_rejects_unbounded_gameplay_particle_count():
    config = _config()
    config.update(
        solver_family="gameplay-pbf", dx=0.01, dt=1.0 / 60.0,
        projection_iterations=3,
    )
    with pytest.raises(ValueError, match="at least 0.04 m"):
        validate_run_config(config, snapshot_stride=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dx", 0.0, "Particle spacing"),
        ("steps", 0, "Steps"),
        ("adapt_every", 0, "Adaptation cadence"),
        ("max_splits_per_adapt", 0, "Maximum splits"),
        ("output", "", "Result path"),
    ],
)
def test_run_config_validation_rejects_invalid_values(field, value, message):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_run_config(config, snapshot_stride=2)


def test_run_config_validation_rejects_bad_stride_and_bounds():
    with pytest.raises(ValueError, match="Visualization stride"):
        validate_run_config(_config(), snapshot_stride=0)
    config = _config()
    config["fine_bounds"][1] = config["fine_bounds"][0]
    with pytest.raises(ValueError, match="Adaptive region bounds"):
        validate_run_config(config, snapshot_stride=1)


def test_run_config_validation_rejects_bad_floating_body_values():
    config = _config()
    config["obstacle_mode"] = "drifting"
    with pytest.raises(ValueError, match="Obstacle mode"):
        validate_run_config(config, snapshot_stride=1)
    config = _config()
    config.update(obstacle_mode="floating", body_density=0.0)
    with pytest.raises(ValueError, match="density"):
        validate_run_config(config, snapshot_stride=1)
    config = _config()
    config.update(obstacle_mode="floating", body_height=0.0)
    with pytest.raises(ValueError, match="dimensions"):
        validate_run_config(config, snapshot_stride=1)
    config = _config()
    config.update(obstacle_mode="hill", hill_height=0.0,
                  hill_radius_x=0.4, hill_radius_y=0.12)
    with pytest.raises(ValueError, match="Hill height"):
        validate_run_config(config, snapshot_stride=1)


def test_put_latest_drops_stale_message():
    queue = Queue(maxsize=1)
    put_latest(queue, {"value": 1})
    put_latest(queue, {"value": 2})
    assert queue.get_nowait()["value"] == 2


def test_frame_buffer_is_bounded_and_index_clamped():
    frames = FrameBuffer(capacity=2)
    for index in range(3):
        frames.append({"step": index}, {"step": index})
    assert len(frames) == 2
    assert frames.get(-10)[0]["step"] == 1
    assert frames.get(10)[0]["step"] == 2


def test_particle_scene_updates_all_actors_and_preserves_camera():
    scene = ParticleScene()
    camera = scene.renderer.GetActiveCamera()
    position = camera.GetPosition()
    scene.update(_snapshot())
    assert scene.scalar == "pressure"
    assert scene.fluid.polydata.GetNumberOfPoints() == 2
    assert scene.fluid.mapper.IsA("vtkGlyph3DMapper")
    assert scene.wall.polydata.GetNumberOfPoints() == 1
    assert scene.obstacle.polydata.GetNumberOfPoints() == 1
    assert scene.obstacle.actor.GetVisibility() == 0
    assert scene.context_obstacle.GetVisibility() == 1
    assert camera.GetPosition() == position
    scene.set_scalar("resolution")
    assert scene.scalar_bar.GetTitle() == "Refinement"
    scene.set_scalar("pressure")
    assert "Pressure" in scene.scalar_bar.GetTitle()
    scene.set_scalar("resolution")
    scene.set_particle_scale(0.06)
    scene.set_wall_opacity(0.4)
    scene.set_obstacle_visible(False)
    assert scene.obstacle.actor.GetVisibility() == 0
    assert scene.context_obstacle.GetVisibility() == 0
    scene.set_colorbar_visible(False)
    assert scene.scalar_bar.GetVisibility() == 0
    scene.set_colorbar_visible(True)
    assert scene.scalar_bar.GetVisibility() == 1


def test_particle_scene_draws_floating_body_as_particles():
    scene = ParticleScene()
    scene.update(_snapshot(obstacle_mode="floating"))
    assert scene.obstacle.mapper.IsA("vtkGlyph3DMapper")
    assert scene.obstacle.polydata.GetPointData().GetArray("h") is not None
    assert scene.obstacle.actor.GetVisibility() == 1
    assert scene.context_obstacle.GetVisibility() == 0
    scene.set_obstacle_visible(False)
    assert scene.obstacle.actor.GetVisibility() == 0
    scene.set_obstacle_visible(True)
    assert scene.obstacle.actor.GetVisibility() == 1
    scene.set_obstacle_mode("none")
    scene.set_obstacle_visible(True)
    assert scene.obstacle.actor.GetVisibility() == 0
    assert scene.context_obstacle.GetVisibility() == 0


def test_particle_scene_draws_smooth_hill_instead_of_boundary_particles():
    scene = ParticleScene()
    scene.update(_hill_snapshot())
    assert scene.hill_polydata.GetNumberOfPoints() == 97 * 49
    points = vtk_to_numpy(scene.hill_polydata.GetPoints().GetData())
    assert np.max(points[:, 2]) == pytest.approx(0.35, rel=1.0e-5)
    centerline = points[np.isclose(points[:, 1], 0.0)]
    radius_sample = centerline[np.argmin(np.abs(centerline[:, 0] - 3.4))]
    assert radius_sample[2] == pytest.approx(
        0.35 * np.exp(-0.5), rel=1.0e-5,
    )
    assert scene.hill_actor.GetVisibility() == 1
    assert scene.obstacle.actor.GetVisibility() == 0
    assert scene.context_obstacle.GetVisibility() == 0
    scene.set_obstacle_visible(False)
    assert scene.hill_actor.GetVisibility() == 0


def test_particle_scene_produces_browser_fallback_image():
    scene = ParticleScene()
    scene.update(_snapshot())
    uri = scene.jpeg_data_uri(quality=75)
    prefix, encoded = uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded).startswith(b"\xff\xd8\xff")


def test_load_saved_result_restores_snapshot_and_metrics(tmp_path):
    path = tmp_path / "result.npz"
    snapshot = _snapshot()
    snapshot.pop("obstacle_mode")
    expected_metrics = {
        "step": 12, "steps": 12, "obstacle_mode": "floating",
    }
    np.savez(path, **snapshot, metrics=json.dumps(expected_metrics))
    loaded_snapshot, metrics = load_saved_result(path)
    assert metrics == expected_metrics
    assert loaded_snapshot["obstacle_mode"] == "floating"
    np.testing.assert_array_equal(loaded_snapshot["xyz"], snapshot["xyz"])


def test_load_saved_result_restores_geospatial_heightfields(tmp_path):
    path = tmp_path / "terrain-result.npz"
    snapshot = _geospatial_snapshot()
    metrics = {
        "solver_family": "geospatial-swe",
        "terrain_id": snapshot["terrain_id"],
        "step": snapshot["step"],
        "time": snapshot["time"],
        "cell_size": snapshot["cell_size"],
    }
    np.savez(
        path,
        bed=snapshot["bed"],
        water_depth=snapshot["water_depth"],
        water_speed=snapshot["water_speed"],
        metrics=json.dumps(metrics),
    )
    loaded_snapshot, loaded_metrics = load_saved_result(path)
    assert loaded_metrics == metrics
    assert loaded_snapshot["solver_family"] == "geospatial-swe"
    np.testing.assert_array_equal(loaded_snapshot["bed"], snapshot["bed"])
    np.testing.assert_array_equal(
        loaded_snapshot["water_surface"],
        snapshot["bed"] + snapshot["water_depth"],
    )


def test_load_saved_result_restores_terrain_hill_parameters(tmp_path):
    path = tmp_path / "terrain-sph-result.npz"
    snapshot = _hill_snapshot()
    hill = snapshot.pop("hill")
    snapshot.pop("obstacle_mode")
    metrics = {
        "solver_family": "terrain-wcsph",
        "obstacle_mode": "hill",
        "hill": hill,
        "hill_particles": 1,
    }
    np.savez(path, **snapshot, metrics=json.dumps(metrics))
    loaded_snapshot, loaded_metrics = load_saved_result(path)
    assert loaded_metrics == metrics
    assert loaded_snapshot["obstacle_mode"] == "hill"
    assert loaded_snapshot["hill"] == hill


def test_studio_uses_explicit_three_panel_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-layout-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    markup = studio.ui.html
    assert 'class="studio-workspace"' in markup
    assert 'class="viewport-wrap"' in markup
    assert 'class="details-panel"' in markup
    assert 'class="viewport-controls"' in markup
    assert 'class="scalar-toggle"' in markup
    assert "Pressure" in markup
    assert "Speed" in markup
    assert "Reset view" in markup
    assert "timeline-label" in markup
    assert "viewport-hud" not in markup
    assert "viewport-container" not in markup
    assert "position:relative;display:grid" in markup
    assert "position:absolute;inset:0;width:100%;height:100%" in markup


def test_studio_has_refreshed_chrome(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-chrome-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    markup = studio.ui.html
    assert 'class="brand"' in markup
    assert "toolbar-progress" in markup
    assert 'class="drawer-scroll"' in markup
    assert 'class="drawer-footer"' in markup
    assert "launch-btn" in markup
    assert 'class="transport mt-3"' in markup
    assert "status-dot status-" in markup
    assert 'class="viewport-empty"' in markup
    assert 'class="viewport-busy"' in markup
    assert 'class="details-grid"' in markup
    assert "metric span-2" in markup
    assert "metric-value num" in markup
    assert "details-rule" in markup


def test_studio_states_used_by_new_chrome_are_present(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-state-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    for key in ("step", "step_total", "status", "status_detail", "run_active",
                "frame_image", "mode_items", "resolution_mode", "viewport_size",
                "obstacle_mode", "obstacle_items", "body_geometry_drift",
                "geospatial_terrain", "geospatial_frame",
                "geospatial_webgpu_status", "water_volume", "hill_particles",
                "hill_center_x", "hill_height", "hill_radius_x"):
        assert studio.state.has(key)


def test_studio_exposes_floating_body_controls(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-floating-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    markup = studio.ui.html
    assert "Floating body" in markup
    assert "body_density" in markup
    assert "body_geometry_drift" in markup
    assert "rigid_device_error" in markup
    assert "contact_max_penetration" in markup


def test_studio_exposes_separate_gameplay_column_and_config(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-gameplay-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    values = [item["value"] for item in studio.state.mode_items]
    assert values == [
        "adaptive", "uniform", "gameplay", "geospatial", "terrain",
    ]
    scientific_config = studio._config()
    assert studio.state.resolution_mode == "adaptive"
    assert scientific_config["solver_family"] == "wcsph"
    assert scientific_config["resolution_mode"] == "adaptive"
    for gameplay_key in (
        "substeps", "projection_iterations", "velocity_damping",
        "xsph_coefficient",
    ):
        assert gameplay_key not in scientific_config
    studio.state.resolution_mode = "uniform"
    studio._on_resolution_mode("uniform")
    uniform_config = studio._config()
    assert uniform_config["solver_family"] == "wcsph"
    assert uniform_config["resolution_mode"] == "uniform"
    assert "substeps" not in uniform_config
    for scientific_payload in (scientific_config, uniform_config):
        solver_payload = dict(scientific_payload)
        solver_payload.pop("solver_family")
        solver_payload.pop("output")
        parsed = DamBreakConfig.from_mapping(solver_payload)
        assert parsed.resolution_mode == scientific_payload["resolution_mode"]
    studio.state.resolution_mode = "gameplay"
    studio._on_resolution_mode("gameplay")
    config = studio._config()
    assert config["solver_family"] == "gameplay-pbf"
    assert config["resolution_mode"] == "uniform"
    assert config["dx"] == pytest.approx(0.05)
    assert config["dt"] == pytest.approx(1.0 / 60.0)
    assert config["projection_iterations"] == 3
    assert config["xsph_coefficient"] == pytest.approx(0.01)
    assert studio.state.scalar == "speed"
    markup = studio.ui.html
    assert "Gameplay profile" in markup
    assert "Approximate PBF" in markup
    assert "GAMEPLAY BUDGET" in markup
    assert "solver_frame_median_ms" in markup


def test_studio_exposes_isolated_terrain_wcsph_profile(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-terrain-wcsph-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.resolution_mode = "terrain"
    studio._on_resolution_mode("terrain")
    config = studio._config()
    assert config["solver_family"] == "terrain-wcsph"
    assert config["resolution_mode"] == "uniform"
    assert config["obstacle_mode"] == "hill"
    assert config["dx"] == pytest.approx(0.1)
    assert config["hill_height"] == pytest.approx(0.35)
    solver_payload = dict(config)
    solver_payload.pop("output")
    parsed = DamBreakConfig.from_mapping(solver_payload)
    assert parsed.solver_family == "terrain-wcsph"
    assert parsed.obstacle_mode == "hill"
    assert "Terrain SPH" in studio.ui.html
    assert "procedural smooth hill" in studio.ui.html


def test_terrain_obstacle_state_cannot_leak_into_other_solver_profiles(
        monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-terrain-profile-isolation", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.obstacle_mode = "fixed"
    studio._on_obstacle_mode("fixed")

    studio.state.resolution_mode = "terrain"
    studio._on_resolution_mode("terrain")
    assert studio.state.obstacle_mode == "hill"
    assert studio._config()["obstacle_mode"] == "hill"

    studio.state.resolution_mode = "gameplay"
    studio._on_resolution_mode("gameplay")
    assert studio.state.obstacle_mode == "floating"
    assert studio._config()["obstacle_mode"] == "floating"

    studio.state.resolution_mode = "uniform"
    studio._on_resolution_mode("uniform")
    assert studio.state.obstacle_mode == "fixed"
    assert studio._config()["obstacle_mode"] == "fixed"


def test_studio_exposes_geospatial_profile_and_isolated_config(
        monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-geospatial-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.resolution_mode = "geospatial"
    studio._on_resolution_mode("geospatial")
    config = studio._config()
    assert config["solver_family"] == "geospatial-swe"
    assert config["grid_nx"] == config["grid_ny"] == 256
    assert config["steps"] == 600
    assert config["terrain_id"] == "synthetic-valley-fixture"
    assert config["synthetic_breach"] is True
    assert "dx" not in config
    assert studio.state.geospatial_quality == "detailed"
    markup = studio.ui.html
    assert "geospatial-canvas" in markup
    assert "Geospatial profile" in markup
    assert "GEOSPATIAL BUDGET" in markup
    assert "Depth-averaged synthetic breach" in markup


def test_studio_automatically_routes_a_prepared_nasadem_asset(
        monkeypatch, tmp_path):
    asset = tmp_path / "nasadem-malpasset-256.npz"
    np.savez_compressed(
        asset,
        bed=np.zeros((8, 8), dtype=np.float32),
        initial_depth=np.zeros((8, 8), dtype=np.float32),
    )
    asset.with_suffix(".json").write_text(json.dumps({
        "source": {"product": "NASADEM_HGT.001"},
        "crop": {"shape": [8, 8]},
    }))
    monkeypatch.setattr("app.DEFAULT_GEOSPATIAL_ASSET", asset)
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-nasadem-asset-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.resolution_mode = "geospatial"
    studio._on_resolution_mode("geospatial")
    config = studio._config()
    assert config["terrain_path"] == str(asset)
    assert config["terrain_id"] == "NASADEM_HGT_n43e006"
    assert "processed crop loaded" in studio.state.geospatial_data_status
    assert studio._geospatial_provenance["source"]["product"] == (
        "NASADEM_HGT.001"
    )


def test_geospatial_surface_publishes_static_and_dynamic_frames_without_vtk(
        monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-geospatial-routing-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.resolution_mode = "geospatial"
    monkeypatch.setattr(
        studio.scene, "update",
        lambda *_: pytest.fail("geospatial path must not update VTK"),
    )
    monkeypatch.setattr(
        studio.scene, "jpeg_data_uri",
        lambda *_: pytest.fail("geospatial path must not encode a JPEG"),
    )
    snapshot = _geospatial_snapshot()
    metrics = {
        "step": 4,
        "steps": 10,
        "time": 1.5,
        "solver_family": "geospatial-swe",
        "terrain_id": snapshot["terrain_id"],
        "synthetic_terrain": True,
        "wet_cells": 32,
        "wet_area": 28800.0,
        "water_volume": 57600.0,
        "volume_drift": 1.0e-6,
        "peak_depth": 2.0,
        "peak_speed": 1.25,
    }
    studio._show_frame(snapshot, metrics)
    assert studio.state.geospatial_terrain["shape"] == [8, 10]
    assert studio.state.geospatial_frame["shape"] == [8, 10]
    assert studio.state.geospatial_packed_bytes > 0
    assert studio.state.water_volume == pytest.approx(57600.0)
    assert studio.state.frame_image == ""


def test_geospatial_renderer_failure_is_visible(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-geospatial-failure-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio._on_geospatial_failure({"error": "No WebGPU adapter"})
    assert studio.state.geospatial_webgpu_status == "unavailable"
    assert studio.state.geospatial_error == "No WebGPU adapter"
    assert studio.state.status_detail == "Geospatial WebGPU renderer unavailable"


def test_gameplay_quality_presets_restore_scientific_spacing(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-gameplay-quality-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.dx = 0.08
    studio.state.resolution_mode = "gameplay"
    studio._on_resolution_mode("gameplay")
    assert studio.state.dx == pytest.approx(0.05)
    studio.state.gameplay_quality = "fast"
    studio._on_gameplay_quality("fast")
    assert studio.state.dx == pytest.approx(0.10)
    studio.state.resolution_mode = "adaptive"
    studio._on_resolution_mode("adaptive")
    assert studio.state.dx == pytest.approx(0.08)
    assert "gameplay_quality_items" in studio.ui.html


def test_gameplay_surface_publishes_packed_frame_without_vtk(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-surface-routing-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio.state.resolution_mode = "gameplay"
    studio.state.gameplay_view = "surface"
    monkeypatch.setattr(
        studio.scene, "update",
        lambda *_: pytest.fail("surface path must not update VTK"),
    )
    monkeypatch.setattr(
        studio.scene, "jpeg_data_uri",
        lambda *_: pytest.fail("surface path must not encode a JPEG"),
    )
    metrics = {
        "step": 4, "time": 0.1, "solver_family": "gameplay-pbf",
        "body_cm": [2.3, 0.0, 0.4],
        "body_orientation": [0.0, 0.0, 0.0, 1.0],
    }
    studio._show_frame(_gameplay_snapshot(), metrics)
    assert studio.state.surface_frame["count"] == 2
    assert studio.state.surface_packed_bytes > 0
    assert studio.state.frame_image == ""


def test_gameplay_surface_controls_and_fallback_are_exposed(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-surface-ui-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    markup = studio.ui.html
    assert "gameplay-surface-canvas" in markup
    assert "gameplay_view_items" in markup
    assert "Gameplay view" in markup
    assert "Screen-space visual approximation" in markup
    assert "surface_debug_view" in markup
    assert "SURFACE RENDERER" in markup
    assert "surface_render_median_ms" in markup
    assert "surface_pack_median_ms" in markup
    assert studio.state.gameplay_view == "surface"
    studio.state.resolution_mode = "gameplay"
    studio._on_surface_failure({"error": "No adapter"})
    assert studio.state.gameplay_view == "particles"
    assert studio.state.webgpu_status == "unavailable"
    assert studio.state.webgpu_error == "No adapter"


def test_viewport_size_resizes_render_window(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-resize-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    studio._on_viewport_size({"size": {"width": 900, "height": 700}})
    assert tuple(studio.scene.render_window.GetSize()) == (900, 700)
    studio._on_viewport_size({"size": {"width": 10, "height": 10}})
    assert tuple(studio.scene.render_window.GetSize()) == (900, 700)
    studio._on_viewport_size(None)
    assert tuple(studio.scene.render_window.GetSize()) == (900, 700)


def test_worker_cancel_preempts_a_process_stuck_inside_solver_step():
    class Process:
        def __init__(self):
            self.running = True
            self.terminated = False
            self.closed = False

        def is_alive(self):
            return self.running

        def join(self, _timeout):
            pass

        def terminate(self):
            self.terminated = True
            self.running = False

        def kill(self):
            self.running = False

        def close(self):
            self.closed = True

    class Queue:
        def __init__(self):
            self.messages = []

        def put_nowait(self, message):
            self.messages.append(message)

        def close(self):
            pass

        def join_thread(self):
            pass

    worker = SolverWorker()
    process = Process()
    commands = Queue()
    worker.process = process
    worker.command_queue = commands
    worker.result_queue = Queue()

    assert worker.cancel(timeout=0.0)
    assert commands.messages == [{"action": "cancel"}]
    assert process.terminated
    assert process.closed
    assert worker.process is None
