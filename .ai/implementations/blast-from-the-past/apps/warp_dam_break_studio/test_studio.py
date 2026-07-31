import base64
import json
from queue import Queue

import numpy as np
import pytest
from trame.app import get_server

from app import WarpDamBreakStudio, load_saved_result, validate_run_config
from vtk_scene import ParticleScene
from worker import FrameBuffer, put_latest


def _snapshot():
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
    assert scene.fluid.polydata.GetNumberOfPoints() == 2
    assert scene.wall.polydata.GetNumberOfPoints() == 1
    assert scene.obstacle.polydata.GetNumberOfPoints() == 1
    assert camera.GetPosition() == position
    scene.set_scalar("resolution")
    scene.set_particle_scale(0.06)
    scene.set_wall_opacity(0.4)
    scene.set_obstacle_visible(False)
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
    np.savez(path, **snapshot, metrics=json.dumps({"step": 12, "steps": 12}))
    loaded_snapshot, metrics = load_saved_result(path)
    assert metrics == {"step": 12, "steps": 12}
    np.testing.assert_array_equal(loaded_snapshot["xyz"], snapshot["xyz"])


def test_studio_uses_explicit_three_panel_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr("app.DEFAULT_OUTPUT", str(tmp_path / "missing.npz"))
    server = get_server("studio-layout-test", client_type="vue3")
    studio = WarpDamBreakStudio(server=server)
    markup = studio.ui.html
    assert 'class="studio-workspace"' in markup
    assert 'class="viewport-wrap"' in markup
    assert 'class="details-panel"' in markup
    assert "viewport-container" not in markup
