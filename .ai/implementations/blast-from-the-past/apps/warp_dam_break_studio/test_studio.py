import base64
import json
from queue import Queue

import numpy as np
import pytest
from trame.app import get_server

from app import WarpDamBreakStudio, load_saved_result, validate_run_config
from vtk_scene import ParticleScene
from worker import FrameBuffer, put_latest


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
                "obstacle_mode", "obstacle_items", "body_geometry_drift"):
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
