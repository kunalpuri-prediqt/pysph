#!/usr/bin/env python3
"""PySPH Warp Dam-Break Studio: local Trame/VTK browser application."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import time

import numpy as np
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html, vtk, vuetify3 as v3
from trame_client import external_script_handler as _script_handler
from trame_client import module as _client_module

from geospatial_transport import (
    pack_geospatial_frame,
    pack_geospatial_terrain,
)
from surface_transport import pack_surface_frame
from vtk_scene import ParticleScene, SCALARS
from worker import FrameBuffer, SolverWorker, TERMINAL_STATES


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "/tmp/pysph-dam-break-studio.npz"
DEFAULT_GEOSPATIAL_ASSET = APP_DIR / "assets" / "nasadem-malpasset-256.npz"
SNAPSHOT_ARRAYS = ("xyz", "h", "rho", "p", "speed", "kind", "level")
OPTIONAL_SNAPSHOT_ARRAYS = (
    "velocity", "render_xyz", "render_axes", "render_scale",
    "render_neighbors",
)
MIN_RENDER_SIZE = (320, 240)
MAX_RENDER_SIZE = (2400, 1600)
# Trame's external-script helper otherwise copies modules into site-packages,
# which is both immutable in packaged deployments and the wrong home for app
# runtime data.  Point its generated serving cache at a process-local /tmp tree.
CLIENT_SCRIPT_CACHE = Path("/tmp/pysph-trame-client-scripts")
CLIENT_SCRIPT_ES_CACHE = CLIENT_SCRIPT_CACHE / "es"
CLIENT_SCRIPT_UMD_CACHE = CLIENT_SCRIPT_CACHE / "umd"
CLIENT_SCRIPT_ES_CACHE.mkdir(parents=True, exist_ok=True)
CLIENT_SCRIPT_UMD_CACHE.mkdir(parents=True, exist_ok=True)
_client_module.USER_PROVIDED_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_CACHE
_client_module.USER_PROVIDED_ES_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_ES_CACHE
_client_module.USER_PROVIDED_UMD_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_UMD_CACHE
_script_handler.USER_PROVIDED_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_CACHE
_script_handler.USER_PROVIDED_ES_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_ES_CACHE
_script_handler.USER_PROVIDED_UMD_SCRIPTS_DIR_PATH = CLIENT_SCRIPT_UMD_CACHE
WEBGPU_SURFACE_SCRIPT = client.register_external_script(
    APP_DIR / "assets" / "webgpu_surface.js",
    function_names=[
        "updateSurfaceFrame", "updateSurfaceControls", "resetSurfaceCamera",
    ],
    name="warp-dam-break-webgpu-surface",
)
WEBGPU_GEOSPATIAL_SCRIPT = client.register_external_script(
    APP_DIR / "assets" / "webgpu_geospatial.js",
    function_names=[
        "updateGeospatialTerrain", "updateGeospatialFrame",
        "updateGeospatialControls", "resetGeospatialCamera",
    ],
    name="warp-dam-break-webgpu-geospatial",
)


def validate_run_config(config, snapshot_stride):
    """Validate run-defining values before a worker process is spawned."""
    if config.get("solver_family") == "geospatial-swe":
        if int(config.get("steps", 0)) < 1:
            raise ValueError("Steps must be at least one")
        if (int(config.get("grid_nx", 0)) not in {128, 256}
                or int(config.get("grid_ny", 0)) not in {128, 256}):
            raise ValueError("Geospatial grid must use a bounded preset")
        if not 0.0 < float(config.get("cfl", 0.0)) <= 0.5:
            raise ValueError("Geospatial CFL must be in (0, 0.5]")
        if float(config.get("dt_max", 0.0)) <= 0.0:
            raise ValueError("Geospatial maximum timestep must be positive")
        if float(config.get("manning", -1.0)) < 0.0:
            raise ValueError("Geospatial Manning friction cannot be negative")
        if int(snapshot_stride) < 1:
            raise ValueError("Visualization stride must be at least one")
        if not str(config.get("output", "")).strip():
            raise ValueError("Result path must not be empty")
        return config
    if config["dx"] <= 0:
        raise ValueError("Particle spacing must be positive")
    if config["steps"] < 1:
        raise ValueError("Steps must be at least one")
    if config.get("solver_family") == "gameplay-pbf":
        if float(config["dx"]) < 0.04:
            raise ValueError(
                "Gameplay spacing must be at least 0.04 m; use a bounded "
                "quality preset instead of launching an unbounded particle count"
            )
        if int(config.get("projection_iterations", 0)) < 1:
            raise ValueError("Gameplay projection iterations must be positive")
        if float(config.get("dt", 0.0)) <= 0.0:
            raise ValueError("Gameplay timestep must be positive")
        if not 0.0 <= float(config.get("xsph_coefficient", 0.0)) <= 1.0:
            raise ValueError("Gameplay XSPH smoothing must be in [0, 1]")
    if config["adapt_every"] < 1:
        raise ValueError("Adaptation cadence must be at least one")
    if config["max_splits_per_adapt"] < 1:
        raise ValueError("Maximum splits must be at least one")
    if int(snapshot_stride) < 1:
        raise ValueError("Visualization stride must be at least one")
    if config.get("obstacle_mode", "fixed") not in {
        "none", "fixed", "floating", "hill",
    }:
        raise ValueError("Obstacle mode must be none, fixed, floating, or hill")
    if float(config.get("body_density", 500.0)) <= 0.0:
        raise ValueError("Floating-body density must be positive")
    body_dimensions = (
        float(config.get("body_length", 0.32)),
        float(config.get("body_width", 0.28)),
        float(config.get("body_height", 0.20)),
    )
    if min(body_dimensions) <= 0.0:
        raise ValueError("Floating-body dimensions must be positive")
    if config.get("obstacle_mode") == "hill":
        hill_dimensions = (
            float(config.get("hill_height", 0.0)),
            float(config.get("hill_radius_x", 0.0)),
            float(config.get("hill_radius_y", 0.0)),
        )
        if min(hill_dimensions) <= 0.0:
            raise ValueError("Hill height and radii must be positive")
        if hill_dimensions[0] > 1.5:
            raise ValueError("Hill height must not exceed the tank height")
        dx = float(config["dx"])
        center_x = float(config.get("hill_center_x", 0.0))
        center_y = float(config.get("hill_center_y", 0.0))
        if not (
            dx <= center_x <= 161.0 / 30.0 - dx
            and -0.25 + dx <= center_y <= 0.25 - dx
        ):
            raise ValueError("Hill center must lie inside the tank interior")
    bounds = config["fine_bounds"]
    if bounds[0] >= bounds[1] or bounds[5] <= bounds[4]:
        raise ValueError("Adaptive region bounds are invalid")
    if not str(config["output"]).strip():
        raise ValueError("Result path must not be empty")
    return config


def load_saved_result(path):
    """Load the final particle frame and metrics from a solver NPZ."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "metrics" in data:
            metrics = json.loads(str(data["metrics"].item()))
        else:
            return None
        if metrics.get("solver_family") == "geospatial-swe":
            required = ("bed", "water_depth", "water_speed")
            if any(name not in data for name in required):
                return None
            snapshot = {name: np.asarray(data[name]) for name in required}
            snapshot.update({
                "water_surface": (
                    snapshot["bed"] + snapshot["water_depth"]
                ),
                "step": metrics.get("step", 0),
                "time": metrics.get("time", 0.0),
                "terrain_id": metrics.get("terrain_id", "unknown"),
                "solver_family": "geospatial-swe",
                "cell_size": metrics.get("cell_size", [30.0, 30.0]),
                "grid_shape": list(snapshot["bed"].shape),
            })
            return snapshot, metrics
        if any(name not in data for name in SNAPSHOT_ARRAYS):
            return None
        snapshot = {name: np.asarray(data[name]) for name in SNAPSHOT_ARRAYS}
        snapshot.update({
            name: np.asarray(data[name])
            for name in OPTIONAL_SNAPSHOT_ARRAYS
            if name in data
        })
    manifest = path.with_suffix(".json")
    if manifest.is_file():
        manifest_data = json.loads(manifest.read_text())
        metrics.update(manifest_data.get("metrics", {}))
        config = manifest_data.get("config", {})
        metrics.setdefault("obstacle_mode", config.get("obstacle_mode", "fixed"))
    snapshot["obstacle_mode"] = metrics.get("obstacle_mode", "fixed")
    snapshot["hill"] = metrics.get("hill")
    return snapshot, metrics


class WarpDamBreakStudio:
    def __init__(self, server=None):
        self.server = server or get_server(
            "warp-dam-break-studio", client_type="vue3"
        )
        self.state = self.server.state
        self.ctrl = self.server.controller
        self.scene = ParticleScene()
        self.worker = SolverWorker()
        self.frames = FrameBuffer(120)
        self._surface_pack_times = deque(maxlen=240)
        self._geospatial_pack_times = deque(maxlen=240)
        self._geospatial_terrain_key = None
        self._scientific_dx = 0.1
        self._scientific_obstacle_mode = "floating"
        self._particle_steps = 250
        self._last_resolution_mode = "adaptive"
        self._geospatial_asset = (
            DEFAULT_GEOSPATIAL_ASSET
            if DEFAULT_GEOSPATIAL_ASSET.is_file() else None
        )
        self._geospatial_provenance = {}
        if self._geospatial_asset is not None:
            manifest = self._geospatial_asset.with_suffix(".json")
            if manifest.is_file():
                self._geospatial_provenance = json.loads(
                    manifest.read_text(encoding="utf-8")
                )
        self._configure_state()
        self._bind_controller()
        self._build_ui()
        self._restore_previous_result()
        self.ctrl.on_server_ready.add(self._on_server_ready)
        self.ctrl.on_server_ready.add_task(self._poll_worker)
        self.ctrl.on_server_exited.add(self.close)

    def _configure_state(self):
        self.state.update({
            "status": "idle",
            "status_detail": "Ready for a GPU run",
            "run_active": False,
            "paused": False,
            "resolution_mode": "adaptive",
            "dx": 0.1,
            "steps": 250,
            "adapt_every": 10,
            "max_splits": 128,
            "snapshot_stride": 5,
            "obstacle_mode": "floating",
            "obstacle_items": [
                {"title": "Floating body", "value": "floating"},
                {"title": "Fixed obstacle", "value": "fixed"},
                {"title": "No obstacle", "value": "none"},
            ],
            "hill_center_x": 3.0,
            "hill_center_y": 0.0,
            "hill_height": 0.35,
            "hill_radius_x": 0.40,
            "hill_radius_y": 0.12,
            "body_density": 500.0,
            "body_center_x": 2.35,
            "body_center_z": 0.10,
            "body_length": 0.32,
            "body_width": 0.28,
            "body_height": 0.20,
            "kernel": "wendland",
            "alpha": 0.25,
            "xsph_eps": 0.5,
            "cfl": 0.3,
            "n_damp": 50,
            "game_dt": 1.0 / 60.0,
            "gameplay_quality": "rich",
            "gameplay_quality_items": [
                {"title": "Rich · 7k particles", "value": "rich"},
                {"title": "Fast · 1k particles", "value": "fast"},
            ],
            "game_substeps": 1,
            "projection_iterations": 3,
            "game_velocity_damping": 0.997,
            "game_xsph": 0.01,
            "fine_xmin": 1.75,
            "fine_xmax": 2.8,
            "fine_zmax": 0.65,
            "scalar": "pressure",
            "scalar_items": [
                {"title": name.title(), "value": name}
                for name in SCALARS
            ],
            "kernel_items": [
                {"title": "Wendland C2", "value": "wendland"},
                {"title": "Cubic spline", "value": "cubic"},
                {"title": "Gaussian", "value": "gaussian"},
            ],
            "mode_items": [
                {"title": "Adaptive · two level", "value": "adaptive"},
                {"title": "Uniform", "value": "uniform"},
                {"title": "Gameplay · fast", "value": "gameplay"},
                {"title": "Geospatial", "value": "geospatial"},
                {"title": "Terrain SPH", "value": "terrain"},
            ],
            "particle_scale": 0.55,
            "gameplay_view": "surface",
            "gameplay_view_items": [
                {"title": "Fluid surface", "value": "surface"},
                {"title": "Particles", "value": "particles"},
            ],
            "surface_frame": None,
            "surface_controls": None,
            "surface_camera_reset": 0,
            "surface_water_color": "#169dc4",
            "surface_absorption": 0.9,
            "surface_refraction": 0.012,
            "surface_roughness": 0.25,
            "surface_splat_scale": 1.05,
            "surface_smoothing_radius": 8,
            "surface_smoothing_iterations": 4,
            "surface_thickness": 0.65,
            "surface_debug_view": "final",
            "surface_debug_items": [
                {"title": "Final composite", "value": "final"},
                {"title": "Raw depth", "value": "depth"},
                {"title": "Thickness", "value": "thickness"},
                {"title": "Normals", "value": "normals"},
            ],
            "webgpu_status": "waiting",
            "webgpu_error": "",
            "surface_packed_bytes": 0,
            "surface_pack_median_ms": None,
            "surface_pack_p95_ms": None,
            "surface_upload_median_ms": None,
            "surface_upload_p95_ms": None,
            "surface_render_median_ms": None,
            "surface_render_p95_ms": None,
            "surface_dropped_frames": 0,
            "surface_stale_frames": 0,
            "geospatial_quality": "detailed",
            "geospatial_quality_items": [
                {"title": "Detailed · 256²", "value": "detailed"},
                {"title": "Preview · 128²", "value": "preview"},
            ],
            "geospatial_grid": 256,
            "geospatial_cell_size": 30.0,
            "geospatial_cfl": 0.35,
            "geospatial_dt_max": 0.5,
            "geospatial_manning": 0.025,
            "geospatial_dry_depth": 0.01,
            "geospatial_vertical_exaggeration": 4.5,
            "geospatial_absorption": 0.08,
            "geospatial_foam_start": 2.0,
            "geospatial_foam_end": 8.0,
            "geospatial_terrain_id": (
                "NASADEM_HGT_n43e006" if self._geospatial_asset
                else "synthetic-valley-fixture"
            ),
            "geospatial_data_status": (
                "Official NASADEM N43E006 processed crop loaded"
                if self._geospatial_asset else
                "Earthdata login required for NASADEM N43E006"
            ),
            "geospatial_terrain": None,
            "geospatial_frame": None,
            "geospatial_controls": None,
            "geospatial_camera_reset": 0,
            "geospatial_webgpu_status": "waiting",
            "geospatial_error": "",
            "geospatial_packed_bytes": 0,
            "geospatial_pack_median_ms": None,
            "geospatial_pack_p95_ms": None,
            "geospatial_upload_median_ms": None,
            "geospatial_upload_p95_ms": None,
            "geospatial_render_median_ms": None,
            "geospatial_render_p95_ms": None,
            "wet_cells": 0,
            "wet_area": 0.0,
            "water_volume": 0.0,
            "volume_drift": 0.0,
            "peak_depth": 0.0,
            "peak_speed": 0.0,
            "showcase_mode": False,
            "wall_opacity": 0.20,
            "obstacle_visible": True,
            "colorbar_visible": True,
            "right_panel_open": True,
            "frame_index": 0,
            "frame_max": 0,
            "live_view": True,
            "step": 0,
            "step_total": 250,
            "sim_time": 0.0,
            "dt_last": None,
            "fluid_particles": 0,
            "hill_particles": 0,
            "fine_particles": 0,
            "coarse_particles": 0,
            "split_parents": 0,
            "merged_families": 0,
            "shifted_particles": 0,
            "max_shift": 0.0,
            "mass_drift": 0.0,
            "p_max": 0.0,
            "body_particles": 0,
            "body_mass": 0.0,
            "body_cm": None,
            "body_vc": None,
            "body_omega": None,
            "body_orientation": None,
            "body_geometry_drift": 0.0,
            "contact_force": None,
            "contact_impulse": None,
            "contact_max_penetration": 0.0,
            "rigid_device_error": 0,
            "steps_per_second": None,
            "solver_family": "wcsph",
            "approximate": False,
            "constraint_rms": 0.0,
            "projection_iterations_live": 0,
            "correction_clamps": 0,
            "solver_frame_median_ms": None,
            "solver_frame_p95_ms": None,
            "simulated_to_wall_ratio": None,
            "device_name": "NVIDIA GPU",
            "error_text": "",
            "output_path": DEFAULT_OUTPUT,
            "frame_image": "",
        })

    def _bind_controller(self):
        self.ctrl.start_run = self.start_run
        self.ctrl.pause_run = self.pause_run
        self.ctrl.resume_run = self.resume_run
        self.ctrl.single_step = self.single_step
        self.ctrl.cancel_run = self.cancel_run
        self.ctrl.reset_camera = self.reset_camera
        self.state.change("scalar")(self._on_scalar)
        self.state.change("particle_scale")(self._on_particle_scale)
        self.state.change("wall_opacity")(self._on_wall_opacity)
        self.state.change("obstacle_visible")(self._on_obstacle_visible)
        self.state.change("obstacle_mode")(self._on_obstacle_mode)
        self.state.change("resolution_mode")(self._on_resolution_mode)
        self.state.change("gameplay_quality")(self._on_gameplay_quality)
        self.state.change("gameplay_view")(self._on_gameplay_view)
        self.state.change("geospatial_quality")(self._on_geospatial_quality)
        for control_name in (
            "surface_water_color", "surface_absorption", "surface_refraction",
            "surface_roughness", "surface_splat_scale",
            "surface_smoothing_radius", "surface_smoothing_iterations",
            "surface_thickness", "surface_debug_view",
        ):
            self.state.change(control_name)(self._on_surface_control)
        for control_name in (
            "geospatial_vertical_exaggeration", "geospatial_absorption",
            "geospatial_foam_start", "geospatial_foam_end",
            "geospatial_dry_depth",
        ):
            self.state.change(control_name)(self._on_geospatial_control)
        self.state.change("colorbar_visible")(self._on_colorbar_visible)
        self.state.change("frame_index")(self._on_frame_index)
        self.state.change("viewport_size")(self._on_viewport_size)

    def _config(self):
        gameplay = self.state.resolution_mode == "gameplay"
        geospatial = self.state.resolution_mode == "geospatial"
        terrain = self.state.resolution_mode == "terrain"
        if geospatial:
            return validate_run_config({
                "solver_family": "geospatial-swe",
                "steps": int(self.state.steps),
                "grid_nx": int(self.state.geospatial_grid),
                "grid_ny": int(self.state.geospatial_grid),
                "cell_size_x": float(self.state.geospatial_cell_size),
                "cell_size_y": float(self.state.geospatial_cell_size),
                "gravity": 9.81,
                "cfl": float(self.state.geospatial_cfl),
                "dt_max": float(self.state.geospatial_dt_max),
                "dry_depth": float(self.state.geospatial_dry_depth),
                "manning": float(self.state.geospatial_manning),
                "boundary": "closed",
                "terrain_path": (
                    None if self._geospatial_asset is None
                    else str(self._geospatial_asset)
                ),
                "terrain_id": self.state.geospatial_terrain_id,
                "synthetic_breach": True,
                "output": self.state.output_path,
            }, self.state.snapshot_stride)
        config = {
            "solver_family": (
                "gameplay-pbf" if gameplay
                else "terrain-wcsph" if terrain else "wcsph"
            ),
            "resolution_mode": (
                "uniform" if gameplay or terrain
                else self.state.resolution_mode
            ),
            "dx": float(self.state.dx),
            "steps": int(self.state.steps),
            "adapt_every": int(self.state.adapt_every),
            "max_splits_per_adapt": int(self.state.max_splits),
            "obstacle_mode": (
                "hill" if terrain
                else "floating" if gameplay
                else self.state.obstacle_mode
            ),
            "hill_center_x": float(self.state.hill_center_x),
            "hill_center_y": float(self.state.hill_center_y),
            "hill_height": float(self.state.hill_height),
            "hill_radius_x": float(self.state.hill_radius_x),
            "hill_radius_y": float(self.state.hill_radius_y),
            "body_density": float(self.state.body_density),
            "body_center_x": float(self.state.body_center_x),
            "body_center_z": float(self.state.body_center_z),
            "body_length": float(self.state.body_length),
            "body_width": float(self.state.body_width),
            "body_height": float(self.state.body_height),
            "kernel": self.state.kernel,
            "alpha": float(self.state.alpha),
            "xsph_eps": float(self.state.xsph_eps),
            "cfl": float(self.state.cfl),
            "n_damp": int(self.state.n_damp),
            "dt": None,
            "fine_bounds": [
                float(self.state.fine_xmin),
                float(self.state.fine_xmax),
                -0.3,
                0.3,
                0.0,
                float(self.state.fine_zmax),
            ],
            "output": self.state.output_path,
        }
        if gameplay:
            config.update({
                "dt": float(self.state.game_dt),
                "substeps": int(self.state.game_substeps),
                "projection_iterations": int(
                    self.state.projection_iterations
                ),
                "velocity_damping": float(
                    self.state.game_velocity_damping
                ),
                "xsph_coefficient": float(self.state.game_xsph),
            })
        return validate_run_config(config, self.state.snapshot_stride)

    def start_run(self):
        if self.worker.alive:
            return
        self.frames.clear()
        if self._geospatial_active():
            # A restored result can carry the same terrain identity as the new
            # run.  Force the one-time static upload so a freshly mounted or
            # refreshed browser renderer never receives water alone.
            self._geospatial_terrain_key = None
            self.state.geospatial_terrain = None
            self.state.geospatial_frame = None
        self.state.update({
            "status": "initializing",
            "status_detail": "Starting isolated CUDA worker",
            "run_active": True,
            "paused": False,
            "error_text": "",
            "frame_index": 0,
            "frame_max": 0,
            "live_view": True,
            "step": 0,
            "step_total": int(self.state.steps),
        })
        try:
            self.worker.start(
                self._config(),
                snapshot_stride=(
                    1 if self.state.resolution_mode in {"gameplay", "geospatial"}
                    else int(self.state.snapshot_stride)
                ),
            )
        except Exception as exc:
            self.state.update({
                "status": "failed",
                "status_detail": "Worker could not start",
                "run_active": False,
                "error_text": f"{type(exc).__name__}: {exc}",
            })

    def pause_run(self):
        if self.worker.send("pause"):
            self.state.status_detail = "Pause requested"

    def resume_run(self):
        if self.worker.send("resume"):
            self.state.update({
                "paused": False,
                "live_view": True,
                "status_detail": "Resume requested",
            })

    def single_step(self):
        if self.worker.send("step"):
            self.state.status_detail = "Single step requested"

    def cancel_run(self):
        if self.worker.cancel():
            self.state.update({
                "status": "cancelled",
                "status_detail": "Run cancelled",
                "run_active": False,
                "paused": False,
            })

    def reset_camera(self):
        if self._geospatial_active():
            self.state.geospatial_camera_reset += 1
            return
        if self._surface_active():
            self.state.surface_camera_reset += 1
            return
        self.scene.reset_camera()
        self._refresh_view()

    def _surface_active(self):
        return (
            self.state.resolution_mode == "gameplay"
            and self.state.gameplay_view == "surface"
        )

    def _geospatial_active(self):
        return self.state.resolution_mode == "geospatial"

    def _geospatial_control_values(self):
        return {
            "vertical_exaggeration": float(
                self.state.geospatial_vertical_exaggeration
            ),
            "absorption": float(self.state.geospatial_absorption),
            "foam_start": float(self.state.geospatial_foam_start),
            "foam_end": float(self.state.geospatial_foam_end),
            "dry_depth": float(self.state.geospatial_dry_depth),
        }

    def _on_geospatial_control(self, **_):
        self.state.geospatial_controls = self._geospatial_control_values()

    def _on_geospatial_result(self, outputs=None, **_):
        outputs = outputs or {}
        self.state.update({
            "geospatial_webgpu_status": outputs.get("status", "ready"),
            "geospatial_error": outputs.get("error", ""),
            "geospatial_upload_median_ms": outputs.get("upload_median_ms"),
            "geospatial_upload_p95_ms": outputs.get("upload_p95_ms"),
            "geospatial_render_median_ms": outputs.get("render_median_ms"),
            "geospatial_render_p95_ms": outputs.get("render_p95_ms"),
        })

    def _on_geospatial_failure(self, outputs=None, **_):
        outputs = outputs or {}
        error = outputs.get("error", "WebGPU geospatial renderer unavailable")
        self.state.update({
            "geospatial_webgpu_status": "unavailable",
            "geospatial_error": error,
            "status_detail": "Geospatial WebGPU renderer unavailable",
        })

    def _surface_control_values(self):
        return {
            "water_color": self.state.surface_water_color,
            "absorption": float(self.state.surface_absorption),
            "refraction": float(self.state.surface_refraction),
            "roughness": float(self.state.surface_roughness),
            "splat_scale": float(self.state.surface_splat_scale),
            "smoothing_radius": int(self.state.surface_smoothing_radius),
            "smoothing_iterations": int(
                self.state.surface_smoothing_iterations
            ),
            "thickness": float(self.state.surface_thickness),
            "debug_view": self.state.surface_debug_view,
            "obstacle_visible": bool(self.state.obstacle_visible),
        }

    def _on_surface_control(self, **_):
        self.state.surface_controls = self._surface_control_values()

    def _on_surface_result(self, outputs=None, **_):
        outputs = outputs or {}
        self.state.update({
            "webgpu_status": outputs.get("status", "ready"),
            "webgpu_error": outputs.get("error", ""),
            "surface_upload_median_ms": outputs.get("upload_median_ms"),
            "surface_upload_p95_ms": outputs.get("upload_p95_ms"),
            "surface_render_median_ms": outputs.get("render_median_ms"),
            "surface_render_p95_ms": outputs.get("render_p95_ms"),
            "surface_dropped_frames": outputs.get("dropped_frames", 0),
            "surface_stale_frames": outputs.get("stale_frames", 0),
        })

    def _on_surface_failure(self, outputs=None, **_):
        outputs = outputs or {}
        error = outputs.get("error", "WebGPU is unavailable in this browser")
        self.state.update({
            "webgpu_status": "unavailable",
            "webgpu_error": error,
            "gameplay_view": "particles",
            "status_detail": "WebGPU unavailable; using particle view",
        })

    def _on_gameplay_view(self, gameplay_view, **_):
        if not len(self.frames):
            return
        snapshot, metrics = self.frames.get(self.state.frame_index)
        self._show_frame(snapshot, metrics, update_metrics=False)

    def _on_scalar(self, scalar, **_):
        self.scene.set_scalar(scalar)
        self._refresh_view()

    def _on_particle_scale(self, particle_scale, **_):
        self.scene.set_particle_scale(particle_scale)
        self._refresh_view()

    def _on_wall_opacity(self, wall_opacity, **_):
        self.scene.set_wall_opacity(wall_opacity)
        self._refresh_view()

    def _on_obstacle_visible(self, obstacle_visible, **_):
        self.scene.set_obstacle_visible(obstacle_visible)
        self._on_surface_control()
        self._refresh_view()

    def _on_obstacle_mode(self, obstacle_mode, **_):
        if (
            self.state.resolution_mode in {"adaptive", "uniform"}
            and obstacle_mode in {"none", "fixed", "floating"}
        ):
            self._scientific_obstacle_mode = obstacle_mode
        self.scene.set_obstacle_mode(obstacle_mode)
        self.scene.set_obstacle_visible(self.state.obstacle_visible)
        self._refresh_view()

    def _on_resolution_mode(self, resolution_mode, **_):
        previous_mode = self._last_resolution_mode
        if resolution_mode == "gameplay":
            if previous_mode not in {"gameplay", "geospatial"}:
                self._scientific_dx = float(self.state.dx)
            if previous_mode != "geospatial":
                self._particle_steps = int(self.state.steps)
            self._apply_gameplay_quality()
            self.state.steps = self._particle_steps
            self.state.obstacle_mode = "floating"
        elif resolution_mode == "geospatial":
            if previous_mode != "geospatial":
                self._particle_steps = int(self.state.steps)
            self.state.steps = 600
            self._apply_geospatial_quality()
        elif resolution_mode == "terrain":
            if previous_mode in {"gameplay", "geospatial"}:
                self.state.dx = self._scientific_dx
                self.state.steps = self._particle_steps
            self.state.obstacle_mode = "hill"
        elif previous_mode in {"gameplay", "geospatial"}:
            self.state.dx = self._scientific_dx
            self.state.steps = self._particle_steps
            self.state.obstacle_mode = self._scientific_obstacle_mode
        elif resolution_mode in {"adaptive", "uniform"}:
            self.state.obstacle_mode = self._scientific_obstacle_mode
        self._last_resolution_mode = resolution_mode
        if resolution_mode == "gameplay" and self.state.scalar == "pressure":
            self.state.scalar = "speed"
        self.state.status_detail = {
            "gameplay": "Approximate fixed-budget PBF profile",
            "geospatial": "Depth-averaged synthetic terrain-flow profile",
            "terrain": "Uniform 3D WCSPH over a procedural smooth hill",
        }.get(resolution_mode, "Ready for a scientific GPU run")
        if len(self.frames):
            snapshot, metrics = self.frames.get(self.state.frame_index)
            family = metrics.get("solver_family", "wcsph")
            compatible = {
                "adaptive": family == "wcsph",
                "uniform": family == "wcsph",
                "gameplay": family == "gameplay-pbf",
                "geospatial": family == "geospatial-swe",
                "terrain": family == "terrain-wcsph",
            }.get(resolution_mode, False)
            if compatible:
                self._show_frame(snapshot, metrics, update_metrics=False)
            else:
                self.state.frame_image = ""

    def _apply_gameplay_quality(self):
        spacing = {"rich": 0.05, "fast": 0.10}
        self.state.dx = spacing.get(self.state.gameplay_quality, 0.05)

    def _on_gameplay_quality(self, gameplay_quality, **_):
        if self.state.run_active or self.state.resolution_mode != "gameplay":
            return
        self._apply_gameplay_quality()

    def _apply_geospatial_quality(self):
        self.state.geospatial_grid = {
            "detailed": 256, "preview": 128,
        }.get(self.state.geospatial_quality, 256)

    def _on_geospatial_quality(self, geospatial_quality, **_):
        if self.state.run_active or self.state.resolution_mode != "geospatial":
            return
        self._apply_geospatial_quality()

    def _on_colorbar_visible(self, colorbar_visible, **_):
        self.scene.set_colorbar_visible(colorbar_visible)
        self._refresh_view()

    def _on_viewport_size(self, viewport_size, **_):
        """Match the offscreen render window to the browser panel.

        Without this the server renders at a fixed aspect and the frame is
        letterboxed inside the viewport.
        """
        if (not viewport_size or self._surface_active()
                or self._geospatial_active()):
            return
        size = viewport_size.get("size") or {}
        width = int(size.get("width") or 0)
        height = int(size.get("height") or 0)
        if width < MIN_RENDER_SIZE[0] or height < MIN_RENDER_SIZE[1]:
            return
        width = min(width, MAX_RENDER_SIZE[0])
        height = min(height, MAX_RENDER_SIZE[1])
        if tuple(self.scene.render_window.GetSize()) == (width, height):
            return
        self.scene.render_window.SetSize(width, height)
        self.scene.reset_camera()
        self._refresh_view()

    def _on_frame_index(self, frame_index, **_):
        if not len(self.frames):
            return
        index = max(0, min(int(frame_index), len(self.frames) - 1))
        snapshot, metrics = self.frames.get(index)
        self.state.live_view = index == len(self.frames) - 1
        self._show_frame(snapshot, metrics, update_metrics=False)

    def _show_frame(self, snapshot, metrics, update_metrics=True):
        if update_metrics:
            self.state.update({
                "step": metrics.get("step", 0),
                "sim_time": metrics.get("time", 0.0),
                "dt_last": metrics.get("dt_last"),
                "fluid_particles": metrics.get("fluid_particles", 0),
                "hill_particles": metrics.get("hill_particles", 0),
                "fine_particles": metrics.get("fine_particles", 0),
                "coarse_particles": metrics.get("coarse_particles", 0),
                "split_parents": metrics.get("split_parents", 0),
                "merged_families": metrics.get("merged_families", 0),
                "shifted_particles": metrics.get("shifted_particles", 0),
                "max_shift": metrics.get("max_shift", 0.0),
                "mass_drift": metrics.get("mass_drift", 0.0),
                "p_max": metrics.get("p_max", 0.0),
                "obstacle_mode": metrics.get(
                    "obstacle_mode", snapshot.get("obstacle_mode", "fixed")
                ),
                "body_particles": metrics.get("body_particles", 0),
                "body_mass": metrics.get("body_mass", 0.0),
                "body_cm": metrics.get("body_cm"),
                "body_vc": metrics.get("body_vc"),
                "body_omega": metrics.get("body_omega"),
                "body_orientation": metrics.get("body_orientation"),
                "body_geometry_drift": metrics.get(
                    "body_geometry_drift", 0.0
                ),
                "contact_force": metrics.get("contact_force"),
                "contact_impulse": metrics.get("contact_impulse"),
                "contact_max_penetration": metrics.get(
                    "contact_max_penetration", 0.0
                ),
                "rigid_device_error": metrics.get("rigid_device_error", 0),
                "steps_per_second": metrics.get("steps_per_second"),
                "solver_family": metrics.get("solver_family", "wcsph"),
                "approximate": metrics.get("approximate", False),
                "constraint_rms": metrics.get("constraint_rms", 0.0),
                "projection_iterations_live": metrics.get(
                    "projection_iterations", 0
                ),
                "correction_clamps": metrics.get("correction_clamps", 0),
                "solver_frame_median_ms": metrics.get(
                    "solver_frame_median_ms"
                ),
                "solver_frame_p95_ms": metrics.get("solver_frame_p95_ms"),
                "simulated_to_wall_ratio": metrics.get(
                    "simulated_to_wall_ratio"
                ),
                "wet_cells": metrics.get("wet_cells", 0),
                "wet_area": metrics.get("wet_area", 0.0),
                "water_volume": metrics.get("water_volume", 0.0),
                "volume_drift": metrics.get("volume_drift", 0.0),
                "peak_depth": metrics.get("peak_depth", 0.0),
                "peak_speed": metrics.get("peak_speed", 0.0),
                "geospatial_terrain_id": metrics.get(
                    "terrain_id", self.state.geospatial_terrain_id
                ),
                "device_name": metrics.get("runtime", {}).get(
                    "device_name", self.state.device_name
                ),
            })
        if self._surface_active() and "render_xyz" in snapshot:
            frame = pack_surface_frame(
                snapshot,
                metrics,
                [
                    float(self.state.body_length),
                    float(self.state.body_width),
                    float(self.state.body_height),
                ],
            )
            self._surface_pack_times.append(frame["pack_ms"])
            pack_times = np.asarray(self._surface_pack_times)
            self.state.update({
                "surface_frame": frame,
                "surface_packed_bytes": frame["wire_bytes"],
                "surface_pack_median_ms": float(np.median(pack_times)),
                "surface_pack_p95_ms": float(np.percentile(pack_times, 95)),
                "frame_image": "",
            })
            return
        if self._geospatial_active() and "water_depth" in snapshot:
            terrain_key = (
                snapshot.get("terrain_id"),
                tuple(np.asarray(snapshot["bed"]).shape),
                float(np.min(snapshot["bed"])),
                float(np.max(snapshot["bed"])),
            )
            if terrain_key != self._geospatial_terrain_key:
                self.state.geospatial_terrain = pack_geospatial_terrain(
                    snapshot,
                    provenance=(
                        {
                            "product": "deterministic synthetic fixture",
                            "synthetic_breach": True,
                        }
                        if metrics.get("synthetic_terrain", True)
                        else {
                            "product": "NASADEM_HGT.001",
                            "synthetic_breach": True,
                            "source": self._geospatial_provenance.get(
                                "source", {}
                            ),
                            "crop": self._geospatial_provenance.get(
                                "crop", {}
                            ),
                        }
                    ),
                )
                self._geospatial_terrain_key = terrain_key
            started = time.perf_counter()
            frame = pack_geospatial_frame(snapshot, metrics)
            self._geospatial_pack_times.append(
                1000.0 * (time.perf_counter() - started)
            )
            pack_times = np.asarray(self._geospatial_pack_times)
            self.state.update({
                "geospatial_frame": frame,
                "geospatial_packed_bytes": frame["packed_bytes"],
                "geospatial_pack_median_ms": float(np.median(pack_times)),
                "geospatial_pack_p95_ms": float(
                    np.percentile(pack_times, 95)
                ),
                "frame_image": "",
            })
            return
        self.scene.update(snapshot)
        self.state.frame_image = self.scene.jpeg_data_uri()
        self.ctrl.view_update()

    def _refresh_view(self, **_):
        if self._surface_active() or self._geospatial_active():
            return
        self.scene.render_window.Render()
        self.state.frame_image = self.scene.jpeg_data_uri()
        self.ctrl.view_update()

    def _restore_previous_result(self):
        try:
            saved = load_saved_result(self.state.output_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        if saved is None:
            return
        snapshot, metrics = saved
        if metrics.get("solver_family") == "geospatial-swe":
            self.state.resolution_mode = "geospatial"
        elif metrics.get("solver_family") == "terrain-wcsph":
            self.state.resolution_mode = "terrain"
        index = self.frames.append(snapshot, metrics)
        self.state.update({
            "status": "completed",
            "status_detail": "Restored the latest completed GPU result",
            "frame_index": index,
            "frame_max": index,
            "step_total": metrics.get("steps", metrics.get("step", 0)),
        })
        self._show_frame(snapshot, metrics)

    def _on_server_ready(self, **_):
        self.state.surface_controls = self._surface_control_values()
        self.state.geospatial_controls = self._geospatial_control_values()
        if self._surface_active() or self._geospatial_active():
            return
        self.ctrl.view_resize()
        # The client viewport aspect is only known now, so reframe the tank.
        self.scene.reset_camera()
        self.ctrl.view_update()

    def _handle_message(self, message):
        kind = message.get("type")
        if kind == "status":
            status = message["status"]
            self.state.update({
                "status": status,
                "status_detail": message.get("detail", ""),
                "paused": status == "paused",
                "run_active": status not in TERMINAL_STATES,
            })
            return
        if kind in {"frame", "completed"}:
            snapshot = message["snapshot"]
            metrics = message["metrics"]
            index = self.frames.append(snapshot, metrics)
            with self.state:
                self.state.frame_max = len(self.frames) - 1
                if self.state.live_view or kind == "completed":
                    self.state.frame_index = index
                    self.state.live_view = True
                    self._show_frame(snapshot, metrics)
                if kind == "completed":
                    self.state.status = "completed"
                    self.state.status_detail = (
                        f"Completed in {message.get('wall_seconds', 0):.2f} s"
                    )
                    self.state.run_active = False
                    self.state.paused = False
            return
        if kind == "error":
            self.state.update({
                "status": "failed",
                "status_detail": "Solver worker failed",
                "run_active": False,
                "paused": False,
                "error_text": (
                    message.get("error", "Unknown worker error")
                    + "\n\n"
                    + message.get("traceback", "")
                ),
            })

    async def _poll_worker(self, **_):
        while True:
            await asyncio.sleep(0.05)
            messages = self.worker.poll()
            if messages:
                with self.state:
                    for message in messages:
                        self._handle_message(message)

    def close(self, **_):
        self.worker.close()

    def _build_ui(self):
        # Vue's runtime template compiler drops inline <style> tags, so the
        # stylesheet has to be served and registered as a client module.
        self.server.enable_module({
            "serve": {"studio_assets": str(APP_DIR / "assets")},
            "styles": ["studio_assets/studio.css"],
        })
        with SinglePageWithDrawerLayout(
            self.server, full_height=True, theme="dark"
        ) as layout:
            layout.root["classes"] = "studio-shell"
            layout.root["style"] = "height:100vh; min-height:100vh;"
            layout.toolbar["classes"] = "studio-toolbar"
            layout.drawer["classes"] = "studio-drawer"
            layout.drawer["v_show"] = ("!showcase_mode",)
            layout.content["classes"] = "studio-main"
            layout.content["style"] = (
                "height:100vh;max-height:100vh;overflow:hidden;"
                "background:#060a15;"
            )
            layout.drawer["width"] = 372
            layout.title.set_text("")
            # The built-in toolbar title is a flex-grow slot; leaving it in
            # place pushes the brand lockup to the centre of the bar.
            layout.title["classes"] = "d-none"
            with layout.toolbar:
                with html.Div(classes="brand"):
                    html.Div("SPH", classes="brand-mark")
                    with html.Div():
                        html.Div("PySPH · Warp Studio", classes="brand-name")
                        html.Div("Warp · CUDA", classes="brand-sub")
                v3.VSpacer()
                v3.VChip(
                    text=("status.toUpperCase()",),
                    color=(
                        "status === 'running' ? 'cyan' : "
                        "status === 'paused' ? 'amber' : "
                        "status === 'failed' ? 'error' : "
                        "status === 'completed' ? 'success' : 'blue-grey'"
                    ),
                    variant="tonal",
                    size="small",
                    classes=(
                        "status === 'running' ? "
                        "'status-pill is-running mr-3' : 'status-pill mr-3'"
                    ),
                )
                v3.VBtn(
                    icon="mdi-crosshairs-gps",
                    variant="text",
                    click=self.ctrl.reset_camera,
                    title="Reset camera",
                )
                v3.VBtn(
                    icon="mdi-dock-right",
                    variant="text",
                    click="right_panel_open = !right_panel_open",
                    title="Toggle run details",
                )
                v3.VProgressLinear(
                    model_value=(
                        "step_total ? "
                        "Math.min(100, (step / step_total) * 100) : 0",
                    ),
                    color="cyan",
                    height=2,
                    classes="toolbar-progress",
                    v_show=("run_active",),
                )
            with layout.drawer:
                with html.Div(classes="drawer-scroll"):
                    html.Div("GPU SIMULATION", classes="eyebrow mb-1")
                    html.H2(
                        "{{ resolution_mode === 'geospatial' ? "
                        "'Terrain-flow controls' : resolution_mode === "
                        "'terrain' ? 'Terrain SPH controls' : "
                        "'Dam-break controls' }}",
                        classes="text-h5 mb-1",
                    )
                    html.P(
                        "{{ resolution_mode === 'geospatial' ? "
                        "'Route a synthetic breach across a regional heightfield.' : "
                        "resolution_mode === 'terrain' ? "
                        "'Run genuine 3D WCSPH into a procedural smooth hill.' : "
                        "'Configure the physics, then inspect every particle "
                        "without leaving the browser.' }}",
                        classes="text-body-2 text-medium-emphasis mb-5",
                    )
                    with html.Div(classes="card"):
                        with html.Div(classes="card-head"):
                            v3.VIcon("mdi-tune-variant", size="16")
                            html.Span("Run")
                        with v3.VBtnToggle(
                            v_model=("resolution_mode", "adaptive"),
                            mandatory=True,
                            divided=True,
                            density="comfortable",
                            color="cyan",
                            classes="segmented mb-4",
                            disabled=("run_active",),
                        ):
                            v3.VBtn(
                                v_for="item in mode_items",
                                key="item.value",
                                value=("item.value",),
                                text=("item.title",),
                            )
                        v3.VAlert(
                            text=(
                                "'Approximate PBF: optimized for visual "
                                "interaction and frame cost, not scientific "
                                "pressure or energy results.'",
                            ),
                            type="warning",
                            variant="tonal",
                            density="compact",
                            classes="mb-4",
                            v_show=("resolution_mode === 'gameplay'",),
                        )
                        v3.VAlert(
                            text=(
                                "geospatial_terrain_id === "
                                "'synthetic-valley-fixture' ? "
                                "'Depth-averaged synthetic breach. The visible "
                                "terrain is the deterministic fixture until "
                                "official NASADEM N43E006 is authenticated; "
                                "this is not a historical reconstruction or "
                                "inundation forecast.' : "
                                "'Depth-averaged synthetic breach over an "
                                "official NASADEM N43E006 crop—not a historical "
                                "reconstruction or inundation forecast.'",
                            ),
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mb-4",
                            v_show=("resolution_mode === 'geospatial'",),
                        )
                        v3.VAlert(
                            text=(
                                "'Uniform 3D WCSPH with a stationary physical "
                                "Gaussian-hill boundary. This procedural hill "
                                "is not NASADEM or game-frame-rate evidence.'",
                            ),
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mb-4",
                            v_show=("resolution_mode === 'terrain'",),
                        )
                        with v3.VRow(dense=True):
                            with v3.VCol(
                                cols=6,
                                v_show=("resolution_mode !== 'geospatial'",),
                            ):
                                v3.VTextField(
                                    v_model=("dx", 0.1),
                                    label="Spacing dx",
                                    type="number",
                                    step=0.01,
                                    variant="outlined",
                                    density="compact",
                                    prepend_inner_icon="mdi-dots-grid",
                                    disabled=("run_active",),
                                )
                            with v3.VCol(
                                cols=(
                                    "resolution_mode === 'geospatial' ? 12 : 6",
                                ),
                            ):
                                v3.VTextField(
                                    v_model=("steps", 250),
                                    label="Steps",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    prepend_inner_icon="mdi-step-forward",
                                    disabled=("run_active",),
                                )
                        v3.VSelect(
                            v_model=("obstacle_mode", "floating"),
                            items=("obstacle_items",),
                            label="Obstacle",
                            variant="outlined",
                            prepend_inner_icon="mdi-cube-outline",
                            disabled=("run_active",),
                            density="compact",
                            classes="mb-2",
                            v_show=(
                                "resolution_mode !== 'geospatial' && "
                                "resolution_mode !== 'terrain'",
                            ),
                        )
                    with v3.VExpansionPanels(
                        variant="accordion", classes="mb-4", flat=True
                    ):
                        with v3.VExpansionPanel(
                            title="Gameplay profile",
                            v_show=("resolution_mode === 'gameplay'",),
                        ):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    v_model=("gameplay_quality", "rich"),
                                    items=("gameplay_quality_items",),
                                    label="Gameplay quality",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("game_dt", 1.0 / 60.0),
                                            label="Frame Δt",
                                            type="number",
                                            step=0.001,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("game_substeps", 1),
                                            label="Substeps",
                                            type="number",
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                v3.VSlider(
                                    v_model=("projection_iterations", 3),
                                    min=1,
                                    max=4,
                                    step=1,
                                    label="PBF projections",
                                    color="orange",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("game_velocity_damping", 0.997),
                                    min=0.95,
                                    max=1.0,
                                    step=0.001,
                                    label="Velocity retention",
                                    color="orange",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("game_xsph", 0.01),
                                    min=0.0,
                                    max=0.1,
                                    step=0.005,
                                    label="XSPH smoothing",
                                    color="orange",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(
                            title="Geospatial profile",
                            v_show=("resolution_mode === 'geospatial'",),
                        ):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    v_model=("geospatial_quality", "detailed"),
                                    items=("geospatial_quality_items",),
                                    label="Grid quality",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VTextField(
                                    model_value=("geospatial_data_status",),
                                    label="Terrain source",
                                    variant="outlined",
                                    density="compact",
                                    readonly=True,
                                )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("geospatial_cfl", 0.35),
                                            label="CFL",
                                            type="number",
                                            step=0.01,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("geospatial_dt_max", 0.5),
                                            label="Max Δt (s)",
                                            type="number",
                                            step=0.05,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                v3.VSlider(
                                    v_model=("geospatial_manning", 0.025),
                                    min=0.0,
                                    max=0.08,
                                    step=0.005,
                                    label="Manning friction",
                                    color="cyan",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(
                            title="Terrain",
                            v_show=("resolution_mode === 'terrain'",),
                        ):
                            with v3.VExpansionPanelText():
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("hill_center_x", 3.0),
                                            label="Hill center x",
                                            type="number", step=0.05,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("hill_height", 0.35),
                                            label="Hill height",
                                            type="number", step=0.05,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("hill_radius_x", 0.40),
                                            label="Length radius",
                                            type="number", step=0.05,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("hill_radius_y", 0.12),
                                            label="Width radius",
                                            type="number", step=0.01,
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                v3.VTextField(
                                    v_model=("hill_center_y", 0.0),
                                    label="Hill center y",
                                    type="number", step=0.01,
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(
                            title="Adaptive region",
                            v_show=("resolution_mode === 'adaptive'",),
                        ):
                            with v3.VExpansionPanelText():
                                v3.VTextField(
                                    v_model=("adapt_every", 10),
                                    label="Adapt every N steps",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VTextField(
                                    v_model=("max_splits", 128),
                                    label="Max splits / checkpoint",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VTextField(
                                    v_model=("snapshot_stride", 5),
                                    label="Visualize every N steps",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("fine_xmin", 1.9),
                                            label="Fine x min",
                                            type="number",
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("fine_xmax", 3.5),
                                            label="Fine x max",
                                            type="number",
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                v3.VTextField(
                                    v_model=("fine_zmax", 0.65),
                                    label="Fine-region height",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(
                            title="Physics",
                            v_show=("resolution_mode !== 'geospatial'",),
                        ):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    v_model=("kernel", "wendland"),
                                    items=("kernel_items",),
                                    label="SPH kernel",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("alpha", 0.25),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="Viscosity α",
                                    color="cyan",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("xsph_eps", 0.5),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="XSPH ε",
                                    color="cyan",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(
                            title="Floating body",
                            v_show=(
                                "obstacle_mode === 'floating' && "
                                "resolution_mode !== 'geospatial' && "
                                "resolution_mode !== 'terrain'",
                            ),
                        ):
                            with v3.VExpansionPanelText():
                                v3.VTextField(
                                    v_model=("body_density", 500.0),
                                    label="Density",
                                    type="number",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("body_center_x", 2.35),
                                            label="Center x",
                                            type="number",
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("body_center_z", 0.10),
                                            label="Center z",
                                            type="number",
                                            variant="outlined",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                with v3.VRow(dense=True):
                                    for key, label, value in (
                                        ("body_length", "Length", 0.32),
                                        ("body_width", "Width", 0.28),
                                        ("body_height", "Height", 0.20),
                                    ):
                                        with v3.VCol(cols=4):
                                            v3.VTextField(
                                                v_model=(key, value),
                                                label=label,
                                                type="number",
                                                variant="outlined",
                                                density="compact",
                                                disabled=("run_active",),
                                            )
                        with v3.VExpansionPanel(title="Visualization"):
                            with v3.VExpansionPanelText():
                                with html.Div(
                                    v_show=("resolution_mode === 'geospatial'",),
                                ):
                                    v3.VAlert(
                                        text="Renderer-only terrain exaggeration never changes solver elevations.",
                                        type="info",
                                        variant="tonal",
                                        density="compact",
                                        classes="mb-3",
                                    )
                                    v3.VSlider(
                                        v_model=(
                                            "geospatial_vertical_exaggeration",
                                            4.5,
                                        ),
                                        min=0.5,
                                        max=8.0,
                                        step=0.1,
                                        label="Vertical exaggeration",
                                        color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("geospatial_absorption", 0.08),
                                        min=0.01,
                                        max=0.3,
                                        step=0.01,
                                        label="Water absorption",
                                        color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("geospatial_foam_start", 2.0),
                                        min=0.0,
                                        max=10.0,
                                        step=0.5,
                                        label="Foam speed start",
                                        color="cyan",
                                        thumb_label=True,
                                    )
                                v3.VSelect(
                                    v_model=("gameplay_view", "surface"),
                                    items=("gameplay_view_items",),
                                    label="Gameplay view",
                                    variant="outlined",
                                    density="compact",
                                    v_show=("resolution_mode === 'gameplay'",),
                                )
                                with html.Div(
                                    v_show=(
                                        "resolution_mode === 'gameplay' && "
                                        "gameplay_view === 'surface'",
                                    ),
                                ):
                                    v3.VAlert(
                                        text="Screen-space visual approximation; not a scientific observable.",
                                        type="info",
                                        variant="tonal",
                                        density="compact",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        v_model=("surface_water_color", "#169dc4"),
                                        label="Water colour",
                                        variant="outlined",
                                        density="compact",
                                    )
                                    v3.VSlider(
                                        v_model=("surface_absorption", 0.9),
                                        min=0.2, max=5.0, step=0.1,
                                        label="Absorption", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_refraction", 0.012),
                                        min=0.0, max=0.10, step=0.005,
                                        label="Refraction", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_roughness", 0.25),
                                        min=0.0, max=0.7, step=0.025,
                                        label="Roughness", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_splat_scale", 1.05),
                                        min=0.65, max=1.8, step=0.05,
                                        label="Surface reach", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_smoothing_radius", 8),
                                        min=1, max=8, step=1,
                                        label="Bilateral radius", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_smoothing_iterations", 4),
                                        min=0, max=4, step=1,
                                        label="Smoothing passes", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSlider(
                                        v_model=("surface_thickness", 0.65),
                                        min=0.1, max=2.5, step=0.05,
                                        label="Optical thickness", color="cyan",
                                        thumb_label=True,
                                    )
                                    v3.VSelect(
                                        v_model=("surface_debug_view", "final"),
                                        items=("surface_debug_items",),
                                        label="Surface target",
                                        variant="outlined",
                                        density="compact",
                                    )
                                v3.VSelect(
                                    v_model=("scalar", "pressure"),
                                    items=("scalar_items",),
                                    label="Color particles by",
                                    variant="outlined",
                                    density="compact",
                                    v_show=(
                                        "resolution_mode !== 'geospatial' && "
                                        "(gameplay_view !== 'surface' || "
                                        "resolution_mode !== 'gameplay')",
                                    ),
                                )
                                v3.VSlider(
                                    v_model=("particle_scale", 0.55),
                                    min=0.2,
                                    max=0.9,
                                    step=0.05,
                                    label="Particle size",
                                    color="cyan",
                                    thumb_label=True,
                                    v_show=(
                                        "resolution_mode !== 'geospatial' && "
                                        "(gameplay_view !== 'surface' || "
                                        "resolution_mode !== 'gameplay')",
                                    ),
                                )
                                v3.VSlider(
                                    v_model=("wall_opacity", 0.2),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="Wall opacity",
                                    color="cyan",
                                    thumb_label=True,
                                    v_show=("resolution_mode !== 'geospatial'",),
                                )
                                v3.VChip(
                                    text=(
                                        "resolution_mode === 'geospatial' ? "
                                        "'WebGPU terrain rendering' : "
                                        "resolution_mode === 'gameplay' && "
                                        "gameplay_view === 'surface' ? "
                                        "'WebGPU client rendering' : "
                                        "'RTX server rendering'",
                                    ),
                                    prepend_icon=(
                                        "resolution_mode === 'gameplay' && gameplay_view === 'surface' "
                                        "? 'mdi-web' : 'mdi-server'",
                                    ),
                                    color="cyan",
                                    variant="tonal",
                                    size="small",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    v_model=("output_path", DEFAULT_OUTPUT),
                                    label="Result NPZ",
                                    variant="outlined",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                with html.Div(classes="drawer-footer"):
                    with html.Div(classes="status-strip mb-3"):
                        html.Span(classes=("'status-dot status-' + status",))
                        html.Div(
                            "{{ status_detail }}", classes="status-text"
                        )
                    v3.VBtn(
                        text="Launch GPU run",
                        prepend_icon="mdi-rocket-launch",
                        block=True,
                        size="large",
                        classes="launch-btn",
                        click=self.ctrl.start_run,
                        disabled=("run_active",),
                    )
                    with html.Div(classes="transport mt-3"):
                        v3.VBtn(
                            text="Pause",
                            prepend_icon="mdi-pause",
                            variant="tonal",
                            size="small",
                            click=self.ctrl.pause_run,
                            disabled=("!run_active || paused",),
                        )
                        v3.VBtn(
                            text="Resume",
                            prepend_icon="mdi-play",
                            variant="tonal",
                            size="small",
                            click=self.ctrl.resume_run,
                            disabled=("!run_active || !paused",),
                        )
                        v3.VBtn(
                            text="Step",
                            prepend_icon="mdi-debug-step-over",
                            variant="tonal",
                            size="small",
                            click=self.ctrl.single_step,
                            disabled=("!run_active || !paused",),
                        )
                    v3.VBtn(
                        text="Cancel run",
                        block=True,
                        variant="text",
                        size="small",
                        color="error",
                        classes="mt-2",
                        click=self.ctrl.cancel_run,
                        disabled=("!run_active",),
                    )
                    v3.VAlert(
                        text=("error_text",),
                        type="error",
                        variant="tonal",
                        density="compact",
                        classes="mt-3",
                        v_show=("error_text.length > 0",),
                    )
            with layout.content:
                with html.Div(
                    classes="studio-workspace",
                    style=(
                        "position:relative;display:grid;"
                        "grid-template-columns:minmax(0,1fr) auto;"
                        "width:100%;height:calc(100vh - 64px);"
                        "min-height:420px;overflow:hidden;"
                    ),
                ):
                    with html.Div(
                        classes="viewport-wrap",
                        style=(
                            "position:relative;width:100%;height:100%;"
                            "min-width:0;min-height:0;overflow:hidden;"
                            "background:#060a15;"
                        ),
                    ):
                        html.Canvas(
                            id="geospatial-canvas",
                            classes="surface-canvas geospatial-canvas",
                            v_show=("resolution_mode === 'geospatial'",),
                        )
                        html.Canvas(
                            id="gameplay-surface-canvas",
                            classes="surface-canvas",
                            v_show=(
                                "resolution_mode === 'gameplay' && "
                                "gameplay_view === 'surface'",
                            ),
                        )
                        with client.Handler(
                            function=WEBGPU_SURFACE_SCRIPT.function(
                                "updateSurfaceFrame"
                            ),
                            variable="surface_handler_input",
                            trigger_on_change=False,
                            success=(self._on_surface_result, "[$event]"),
                            failure=(self._on_surface_failure, "[$event]"),
                        ) as surface_frame_handler:
                            client.ClientStateChange(
                                value=("surface_frame",),
                                change=surface_frame_handler.run("$event"),
                            )
                        with client.Handler(
                            function=WEBGPU_SURFACE_SCRIPT.function(
                                "updateSurfaceControls"
                            ),
                            variable="surface_control_handler_input",
                            trigger_on_change=False,
                        ) as surface_control_handler:
                            client.ClientStateChange(
                                value=("surface_controls",),
                                change=surface_control_handler.run("$event"),
                                trigger_on_create=True,
                            )
                        with client.Handler(
                            function=WEBGPU_SURFACE_SCRIPT.function(
                                "resetSurfaceCamera"
                            ),
                            variable="surface_reset_handler_input",
                            trigger_on_change=False,
                        ) as surface_reset_handler:
                            client.ClientStateChange(
                                value=("surface_camera_reset",),
                                change=surface_reset_handler.run("$event"),
                            )
                        with client.Handler(
                            function=WEBGPU_GEOSPATIAL_SCRIPT.function(
                                "updateGeospatialTerrain"
                            ),
                            variable="geospatial_terrain_handler_input",
                            trigger_on_change=False,
                            success=(self._on_geospatial_result, "[$event]"),
                            failure=(self._on_geospatial_failure, "[$event]"),
                        ) as geospatial_terrain_handler:
                            client.ClientStateChange(
                                value=("geospatial_terrain",),
                                change=geospatial_terrain_handler.run("$event"),
                                trigger_on_create=True,
                            )
                        with client.Handler(
                            function=WEBGPU_GEOSPATIAL_SCRIPT.function(
                                "updateGeospatialFrame"
                            ),
                            variable="geospatial_frame_handler_input",
                            trigger_on_change=False,
                            success=(self._on_geospatial_result, "[$event]"),
                            failure=(self._on_geospatial_failure, "[$event]"),
                        ) as geospatial_frame_handler:
                            client.ClientStateChange(
                                value=("geospatial_frame",),
                                change=geospatial_frame_handler.run("$event"),
                                trigger_on_create=True,
                            )
                        with client.Handler(
                            function=WEBGPU_GEOSPATIAL_SCRIPT.function(
                                "updateGeospatialControls"
                            ),
                            variable="geospatial_control_handler_input",
                            trigger_on_change=False,
                        ) as geospatial_control_handler:
                            client.ClientStateChange(
                                value=("geospatial_controls",),
                                change=geospatial_control_handler.run("$event"),
                                trigger_on_create=True,
                            )
                        with client.Handler(
                            function=WEBGPU_GEOSPATIAL_SCRIPT.function(
                                "resetGeospatialCamera"
                            ),
                            variable="geospatial_reset_handler_input",
                            trigger_on_change=False,
                        ) as geospatial_reset_handler:
                            client.ClientStateChange(
                                value=("geospatial_camera_reset",),
                                change=geospatial_reset_handler.run("$event"),
                            )
                        html.Img(
                            src=("frame_image",),
                            v_show=(
                                "frame_image.length > 0 && resolution_mode !== "
                                "'geospatial' && !(resolution_mode === "
                                "'gameplay' && gameplay_view === 'surface')",
                            ),
                            classes="viewport-fallback",
                            alt="Rendered adaptive particle field",
                            style=(
                                "position:absolute;inset:0;width:100%;height:100%;"
                                "object-fit:contain;z-index:2;pointer-events:none;"
                                "background:#060a15;"
                            ),
                        )
                        view = vtk.VtkRemoteView(
                            self.scene.render_window,
                            ref="view",
                            interactive_ratio=0.65,
                            still_ratio=1,
                            interactive_quality=70,
                            still_quality=95,
                            classes="viewport-remote",
                            v_show=(
                                "resolution_mode !== 'geospatial' && "
                                "(resolution_mode !== 'gameplay' || "
                                "gameplay_view !== 'surface')",
                            ),
                            EndAnimation=self._refresh_view,
                            style=(
                                "position:absolute;inset:0;width:100%;height:100%;"
                                "z-index:1;background:transparent;"
                            ),
                        )
                        self.ctrl.view_update = view.update
                        self.ctrl.view_resize = view.resize
                        self.ctrl.view_reset_camera = view.reset_camera
                        client.SizeObserver(
                            "viewport_size", classes="viewport-sizer"
                        )
                        with html.Div(classes="viewport-controls"):
                            with v3.VBtnToggle(
                                v_model=("scalar", "pressure"),
                                mandatory=True,
                                divided=True,
                                density="compact",
                                color="cyan",
                                classes="scalar-toggle",
                                v_show=(
                                    "resolution_mode !== 'geospatial' && "
                                    "(resolution_mode !== 'gameplay' || "
                                    "gameplay_view !== 'surface')",
                                ),
                            ):
                                v3.VBtn(
                                    "Pressure",
                                    value="pressure",
                                    prepend_icon="mdi-gauge",
                                    size="small",
                                )
                            with v3.VBtnToggle(
                                v_model=("gameplay_view", "surface"),
                                mandatory=True,
                                divided=True,
                                density="compact",
                                color="cyan",
                                classes="surface-toggle",
                                v_show=("resolution_mode === 'gameplay'",),
                            ):
                                v3.VBtn("Surface", value="surface", size="small")
                                v3.VBtn("Particles", value="particles", size="small")
                                v3.VBtn(
                                    "Speed",
                                    value="speed",
                                    prepend_icon="mdi-speedometer",
                                    size="small",
                                )
                            v3.VBtn(
                                "Reset view",
                                prepend_icon="mdi-crosshairs-gps",
                                variant="tonal",
                                size="small",
                                color="cyan",
                                classes="viewport-btn",
                                click=self.ctrl.reset_camera,
                            )
                            v3.VBtn(
                                text=(
                                    "showcase_mode ? 'Controls' : 'Showcase'",
                                ),
                                prepend_icon="mdi-fullscreen",
                                variant="tonal",
                                size="small",
                                color="cyan",
                                classes="viewport-btn",
                                click=(
                                    "showcase_mode = !showcase_mode; "
                                    "right_panel_open = false",
                                ),
                                v_show=("resolution_mode === 'geospatial'",),
                            )
                            v3.VBtn(
                                icon=(
                                    "colorbar_visible ? 'mdi-eye' : 'mdi-eye-off'"
                                ),
                                variant="tonal",
                                size="small",
                                classes="viewport-btn",
                                click="colorbar_visible = !colorbar_visible",
                                title="Show or hide the color scale",
                                v_show=(
                                    "resolution_mode !== 'geospatial' && "
                                    "(resolution_mode !== 'gameplay' || "
                                    "gameplay_view !== 'surface')",
                                ),
                            )
                        with html.Div(
                            classes="surface-status",
                            v_show=(
                                "resolution_mode === 'gameplay' && "
                                "gameplay_view === 'surface'",
                            ),
                        ):
                            html.Span("{{ webgpu_status.toUpperCase() }}")
                            html.Span("SCREEN-SPACE WATER")
                        with html.Div(
                            classes="surface-status",
                            v_show=("resolution_mode === 'geospatial'",),
                        ):
                            html.Span(
                                "{{ geospatial_webgpu_status.toUpperCase() }}"
                            )
                            html.Span("DEPTH-AVERAGED TERRAIN FLOW")
                        v3.VAlert(
                            text=("webgpu_error",),
                            type="warning",
                            variant="tonal",
                            density="compact",
                            classes="surface-error",
                            v_show=("webgpu_error.length > 0",),
                        )
                        v3.VAlert(
                            text=("geospatial_error",),
                            type="warning",
                            variant="tonal",
                            density="compact",
                            classes="surface-error",
                            v_show=(
                                "resolution_mode === 'geospatial' && "
                                "geospatial_error.length > 0",
                            ),
                        )
                        with html.Div(
                            classes="viewport-busy",
                            v_show=("status === 'initializing'",),
                        ):
                            v3.VProgressCircular(
                                indeterminate=True,
                                size=18,
                                width=2,
                                color="cyan",
                            )
                            html.Span(
                                "{{ status_detail || 'Starting CUDA worker' }}"
                            )
                        with html.Div(
                            classes="viewport-empty",
                            v_show=(
                                "frame_image.length === 0 && !surface_frame && "
                                "!geospatial_frame && !run_active",
                            ),
                        ):
                            with html.Div(classes="empty-card"):
                                v3.VIcon(
                                    "mdi-waves", size="42", color="#4fb9d8"
                                )
                                html.Div(
                                    "No particles yet", classes="empty-title"
                                )
                                html.P(
                                    "The tank, floor grid and obstacle are "
                                    "drawn for context. Press Launch GPU run "
                                    "to fill the tank with fluid particles.",
                                    classes="empty-body",
                                )
                        with html.Div(
                            classes="timeline",
                            style=(
                                "position:absolute;left:24px;right:24px;bottom:18px;"
                                "z-index:4;background:rgba(7,16,31,.84);"
                                "border:1px solid rgba(140,180,230,.14);"
                                "border-radius:16px;padding:4px 18px 0;"
                            ),
                        ):
                            html.Div(
                                "{{ live_view ? 'LIVE' : 'REPLAY' }} · "
                                "frame {{ (frame_index + 1).toLocaleString() }}"
                                " / {{ (frame_max + 1).toLocaleString() }}",
                                classes="timeline-label",
                            )
                            v3.VSlider(
                                v_model=("frame_index", 0),
                                min=0,
                                max=("frame_max", 0),
                                step=1,
                                hide_details=True,
                                color="cyan",
                                disabled=("run_active",),
                                prepend_icon=(
                                    "live_view ? 'mdi-access-point' : "
                                    "'mdi-history'"
                                ),
                            )
                    with html.Div(
                        classes="details-panel",
                        v_show=("right_panel_open && !showcase_mode",),
                        style=(
                            "width:300px;height:100%;overflow-y:auto;padding:18px;"
                            "background:rgba(12,24,44,.96);"
                            "border-left:1px solid rgba(140,180,230,.14);"
                            "box-shadow:-16px 0 36px rgba(0,0,0,.18);"
                        ),
                    ):
                        with html.Div(
                            classes="d-flex align-center justify-space-between mb-3"
                        ):
                            with html.Div():
                                html.Div("SIMULATION", classes="eyebrow")
                                html.H3("Run details", classes="text-h6")
                            v3.VBtn(
                                icon="mdi-chevron-right",
                                variant="text",
                                size="small",
                                click="right_panel_open = false",
                                title="Collapse details",
                            )
                        with html.Div(classes="status-strip mb-4"):
                            v3.VIcon("mdi-expansion-card-variant", size="15")
                            html.Div(
                                "{{ device_name }}", classes="status-text"
                            )
                        html.Div(
                            "PARTICLES", classes="eyebrow mb-2",
                            v_show=("resolution_mode !== 'geospatial'",),
                        )
                        with html.Div(
                            classes="details-grid",
                            v_show=("resolution_mode !== 'geospatial'",),
                        ):
                            self._metric(
                                "Fluid", "mdi-water",
                                "fluid_particles.toLocaleString()",
                            )
                            self._metric(
                                "Step", "mdi-step-forward",
                                "step.toLocaleString() + ' / ' + "
                                "step_total.toLocaleString()",
                            )
                            self._metric(
                                "Fine / coarse", "mdi-grid",
                                "fine_particles.toLocaleString() + ' / ' + "
                                "coarse_particles.toLocaleString()",
                                span=True,
                            )
                            self._metric(
                                "Split / merged", "mdi-call-split",
                                "split_parents.toLocaleString() + ' / ' + "
                                "merged_families.toLocaleString()",
                                span=True,
                            )
                            self._metric(
                                "Shifted / max shift", "mdi-vector-polyline",
                                "shifted_particles.toLocaleString() + ' / ' + "
                                "Number(max_shift).toExponential(2) + ' m'",
                                span=True,
                            )
                        with html.Div(
                            v_show=("resolution_mode === 'terrain'",),
                        ):
                            html.Div(classes="details-rule")
                            html.Div("PROCEDURAL TERRAIN", classes="eyebrow mb-2")
                            with html.Div(classes="details-grid"):
                                self._metric(
                                    "Hill particles", "mdi-image-filter-hdr",
                                    "hill_particles.toLocaleString()",
                                )
                                self._metric(
                                    "Hill height", "mdi-arrow-expand-up",
                                    "Number(hill_height).toFixed(2) + ' m'",
                                )
                                self._metric(
                                    "Center x", "mdi-axis-x-arrow",
                                    "Number(hill_center_x).toFixed(2) + ' m'",
                                )
                                self._metric(
                                    "Radii", "mdi-chart-bell-curve",
                                    "Number(hill_radius_x).toFixed(2) + ' / ' + "
                                    "Number(hill_radius_y).toFixed(2) + ' m'",
                                )
                        with html.Div(
                            v_show=("resolution_mode === 'geospatial'",),
                        ):
                            html.Div("TERRAIN FLOW", classes="eyebrow mb-2")
                            with html.Div(classes="details-grid"):
                                self._metric(
                                    "Grid", "mdi-grid-large",
                                    "geospatial_grid.toLocaleString() + '²'",
                                )
                                self._metric(
                                    "Step", "mdi-step-forward",
                                    "step.toLocaleString() + ' / ' + "
                                    "step_total.toLocaleString()",
                                )
                                self._metric(
                                    "Wet cells", "mdi-waves-arrow-right",
                                    "wet_cells.toLocaleString()",
                                )
                                self._metric(
                                    "Wet area", "mdi-map-marker-radius",
                                    "(wet_area / 1e6).toFixed(2) + ' km²'",
                                )
                                self._metric(
                                    "Peak depth", "mdi-arrow-collapse-down",
                                    "Number(peak_depth).toFixed(2) + ' m'",
                                )
                                self._metric(
                                    "Peak speed", "mdi-speedometer",
                                    "Number(peak_speed).toFixed(2) + ' m/s'",
                                )
                                self._metric(
                                    "Volume drift", "mdi-scale-balance",
                                    "Number(volume_drift).toExponential(2)",
                                    tone="Math.abs(Number(volume_drift)) > 1e-4 "
                                         "? 'bad' : 'ok'",
                                )
                                self._metric(
                                    "Terrain", "mdi-image-filter-hdr",
                                    "geospatial_terrain_id",
                                )
                        with html.Div(
                            v_show=(
                                "obstacle_mode === 'floating' && "
                                "resolution_mode !== 'geospatial' && "
                                "resolution_mode !== 'terrain'",
                            ),
                        ):
                            html.Div(classes="details-rule")
                            html.Div("RIGID BODY", classes="eyebrow mb-2")
                            with html.Div(classes="details-grid"):
                                self._metric(
                                    "Body particles", "mdi-cube-scan",
                                    "body_particles.toLocaleString()",
                                )
                                self._metric(
                                    "Body mass", "mdi-weight",
                                    "Number(body_mass).toFixed(3) + ' kg'",
                                )
                                self._metric(
                                    "Geometry drift", "mdi-ruler-square",
                                    "Number(body_geometry_drift).toExponential(2)",
                                    tone="Number(body_geometry_drift) > 1e-4 ? "
                                         "'bad' : Number(body_geometry_drift) > "
                                         "1e-6 ? 'warn' : 'ok'",
                                )
                                self._metric(
                                    "Center of mass", "mdi-axis-arrow",
                                    "body_cm ? body_cm.map(v => "
                                    "Number(v).toFixed(3)).join(', ') : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Linear velocity", "mdi-speedometer-medium",
                                    "body_vc ? body_vc.map(v => "
                                    "Number(v).toExponential(2)).join(', ') : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Angular velocity", "mdi-rotate-orbit",
                                    "body_omega ? body_omega.map(v => "
                                    "Number(v).toExponential(2)).join(', ') : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Contact force", "mdi-vector-combine",
                                    "contact_force ? contact_force.map(v => "
                                    "Number(v).toExponential(2)).join(', ') : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Contact impulse", "mdi-chart-timeline-variant",
                                    "contact_impulse ? contact_impulse.map(v => "
                                    "Number(v).toExponential(2)).join(', ') : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Max penetration", "mdi-arrow-collapse-down",
                                    "Number(contact_max_penetration).toExponential(2) + ' m'",
                                    span=True,
                                    tone="Number(contact_max_penetration) > "
                                         "0.025 ? 'bad' : "
                                         "Number(contact_max_penetration) > "
                                         "0 ? 'warn' : 'ok'",
                                )
                                self._metric(
                                    "Device error", "mdi-alert-circle-outline",
                                    "rigid_device_error.toLocaleString()",
                                    span=True,
                                    tone="Number(rigid_device_error) !== 0 ? "
                                         "'bad' : 'ok'",
                                )
                        html.Div(classes="details-rule")
                        html.Div("SOLVER", classes="eyebrow mb-2")
                        with html.Div(classes="details-grid"):
                            self._metric(
                                "State", "mdi-pulse", "status.toUpperCase()",
                            )
                            self._metric(
                                "Throughput", "mdi-speedometer",
                                "steps_per_second ? "
                                "Number(steps_per_second).toFixed(1) + ' step/s'"
                                " : '—'",
                            )
                            self._metric(
                                "Sim. time", "mdi-clock-outline",
                                "Number(sim_time).toExponential(3) + ' s'",
                            )
                            self._metric(
                                "Last Δt", "mdi-timer-sand",
                                "dt_last ? "
                                "Number(dt_last).toExponential(2) + ' s' : '—'",
                            )
                        with html.Div(
                            v_show=("solver_family === 'gameplay-pbf'",),
                        ):
                            html.Div(classes="details-rule")
                            html.Div(
                                "GAMEPLAY BUDGET", classes="eyebrow mb-2"
                            )
                            with html.Div(classes="details-grid"):
                                self._metric(
                                    "Projection RMS", "mdi-waveform",
                                    "Number(constraint_rms).toExponential(2)",
                                )
                                self._metric(
                                    "Iterations", "mdi-repeat",
                                    "projection_iterations_live.toLocaleString()",
                                )
                                self._metric(
                                    "Median frame", "mdi-timer-outline",
                                    "solver_frame_median_ms ? "
                                    "Number(solver_frame_median_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    tone="Number(solver_frame_median_ms) > 16.7 "
                                         "? 'warn' : 'ok'",
                                )
                                self._metric(
                                    "P95 frame", "mdi-timer-alert-outline",
                                    "solver_frame_p95_ms ? "
                                    "Number(solver_frame_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    tone="Number(solver_frame_p95_ms) > 33.3 "
                                         "? 'bad' : 'ok'",
                                )
                                self._metric(
                                    "Limited corrections", "mdi-arrow-collapse",
                                    "correction_clamps.toLocaleString()",
                                )
                                self._metric(
                                    "Sim / wall", "mdi-speedometer",
                                    "simulated_to_wall_ratio ? "
                                    "Number(simulated_to_wall_ratio).toFixed(2) + "
                                    "'×' : '—'",
                                )
                            with html.Div(
                                classes="details-rule",
                                v_show=("gameplay_view === 'surface'",),
                            ):
                                pass
                            html.Div(
                                "SURFACE RENDERER",
                                classes="eyebrow mb-2",
                                v_show=("gameplay_view === 'surface'",),
                            )
                            with html.Div(
                                classes="details-grid",
                                v_show=("gameplay_view === 'surface'",),
                            ):
                                self._metric(
                                    "WebGPU", "mdi-chip",
                                    "webgpu_status.toUpperCase()",
                                    tone="webgpu_status === 'ready' ? 'ok' : "
                                         "webgpu_status === 'unavailable' ? "
                                         "'bad' : 'warn'",
                                )
                                self._metric(
                                    "Packed frame", "mdi-package-variant",
                                    "surface_packed_bytes ? "
                                    "(surface_packed_bytes / 1024).toFixed(1) + "
                                    "' KiB' : '—'",
                                )
                                self._metric(
                                    "Pack med / p95", "mdi-server-network",
                                    "surface_pack_median_ms != null ? "
                                    "Number(surface_pack_median_ms).toFixed(2) + "
                                    "' / ' + Number(surface_pack_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Upload med / p95", "mdi-upload-network",
                                    "surface_upload_median_ms != null ? "
                                    "Number(surface_upload_median_ms).toFixed(2) + "
                                    "' / ' + Number(surface_upload_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Render med / p95", "mdi-monitor-dashboard",
                                    "surface_render_median_ms != null ? "
                                    "Number(surface_render_median_ms).toFixed(2) + "
                                    "' / ' + Number(surface_render_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                    tone="Number(surface_render_p95_ms) > 33.3 ? "
                                         "'bad' : Number(surface_render_median_ms) "
                                         "> 16.7 ? 'warn' : 'ok'",
                                )
                                self._metric(
                                    "Dropped / stale", "mdi-image-sync-outline",
                                    "surface_dropped_frames.toLocaleString() + "
                                    "' / ' + surface_stale_frames.toLocaleString()",
                                    span=True,
                                    tone="surface_dropped_frames > 0 || "
                                         "surface_stale_frames > 0 ? 'warn' : 'ok'",
                                )
                        with html.Div(
                            v_show=("solver_family === 'geospatial-swe'",),
                        ):
                            html.Div(classes="details-rule")
                            html.Div(
                                "GEOSPATIAL BUDGET", classes="eyebrow mb-2"
                            )
                            with html.Div(classes="details-grid"):
                                self._metric(
                                    "Median solver", "mdi-timer-outline",
                                    "solver_frame_median_ms != null ? "
                                    "Number(solver_frame_median_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    tone="Number(solver_frame_median_ms) > 16.7 "
                                         "? 'warn' : 'ok'",
                                )
                                self._metric(
                                    "P95 solver", "mdi-timer-alert-outline",
                                    "solver_frame_p95_ms != null ? "
                                    "Number(solver_frame_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    tone="Number(solver_frame_p95_ms) > 33.3 "
                                         "? 'bad' : 'ok'",
                                )
                                self._metric(
                                    "Packed frame", "mdi-package-variant",
                                    "geospatial_packed_bytes ? "
                                    "(geospatial_packed_bytes / 1024).toFixed(1) + "
                                    "' KiB' : '—'",
                                )
                                self._metric(
                                    "Pack med / p95", "mdi-server-network",
                                    "geospatial_pack_median_ms != null ? "
                                    "Number(geospatial_pack_median_ms).toFixed(2) + "
                                    "' / ' + Number(geospatial_pack_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Upload med / p95", "mdi-upload-network",
                                    "geospatial_upload_median_ms != null ? "
                                    "Number(geospatial_upload_median_ms).toFixed(2) + "
                                    "' / ' + Number(geospatial_upload_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                )
                                self._metric(
                                    "Render med / p95", "mdi-monitor-dashboard",
                                    "geospatial_render_median_ms != null ? "
                                    "Number(geospatial_render_median_ms).toFixed(2) + "
                                    "' / ' + Number(geospatial_render_p95_ms).toFixed(2) + "
                                    "' ms' : '—'",
                                    span=True,
                                    tone="Number(geospatial_render_p95_ms) > 33.3 ? "
                                         "'bad' : Number(geospatial_render_median_ms) "
                                         "> 16.7 ? 'warn' : 'ok'",
                                )
                        html.Div(classes="details-rule")
                        html.Div("HEALTH", classes="eyebrow mb-2")
                        with html.Div(classes="details-grid"):
                            self._metric(
                                "Mass drift", "mdi-scale-balance",
                                "Number(mass_drift).toExponential(2)",
                                tone="Math.abs(Number(mass_drift)) > 1e-4 ? "
                                     "'bad' : Math.abs(Number(mass_drift)) > "
                                     "1e-9 ? 'warn' : 'ok'",
                            )
                            self._metric(
                                "Peak pressure", "mdi-gauge",
                                "Number(p_max).toExponential(2) + ' Pa'",
                                tone="Number(p_max) > 1e8 ? 'bad' : "
                                     "Number(p_max) > 1e6 ? 'warn' : 'ok'",
                            )
            layout.footer.hide()
        self.ui = layout

    @staticmethod
    def _metric(label, icon, expression, span=False, tone=None):
        """One telemetry tile; `tone` is a JS expression returning a health class."""
        classes = "metric span-2" if span else "metric"
        with html.Div(classes=classes):
            with html.Div(classes="metric-label"):
                v3.VIcon(icon, size="13")
                html.Span(label)
            html.Div(
                f"{{{{ {expression} }}}}",
                classes=(
                    (f"'metric-value num ' + ({tone})",) if tone
                    else "metric-value num"
                ),
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = WarpDamBreakStudio()
    app.server.start(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        timeout=0,
    )


if __name__ == "__main__":
    main()
