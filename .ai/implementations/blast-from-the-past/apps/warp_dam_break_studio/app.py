#!/usr/bin/env python3
"""PySPH Warp Dam-Break Studio: local Trame/VTK browser application."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html, vtk, vuetify3 as v3

from vtk_scene import ParticleScene, SCALARS
from worker import FrameBuffer, SolverWorker, TERMINAL_STATES


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "/tmp/pysph-dam-break-studio.npz"
SNAPSHOT_ARRAYS = ("xyz", "h", "rho", "p", "speed", "kind", "level")
OPTIONAL_SNAPSHOT_ARRAYS = ("velocity",)
MIN_RENDER_SIZE = (320, 240)
MAX_RENDER_SIZE = (2400, 1600)


def validate_run_config(config, snapshot_stride):
    """Validate run-defining values before a worker process is spawned."""
    if config["dx"] <= 0:
        raise ValueError("Particle spacing must be positive")
    if config["steps"] < 1:
        raise ValueError("Steps must be at least one")
    if config["adapt_every"] < 1:
        raise ValueError("Adaptation cadence must be at least one")
    if config["max_splits_per_adapt"] < 1:
        raise ValueError("Maximum splits must be at least one")
    if int(snapshot_stride) < 1:
        raise ValueError("Visualization stride must be at least one")
    if config.get("obstacle_mode", "fixed") not in {
        "none", "fixed", "floating",
    }:
        raise ValueError("Obstacle mode must be none, fixed, or floating")
    if float(config.get("body_density", 500.0)) <= 0.0:
        raise ValueError("Floating-body density must be positive")
    body_dimensions = (
        float(config.get("body_length", 0.32)),
        float(config.get("body_width", 0.28)),
        float(config.get("body_height", 0.20)),
    )
    if min(body_dimensions) <= 0.0:
        raise ValueError("Floating-body dimensions must be positive")
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
        if any(name not in data for name in SNAPSHOT_ARRAYS):
            return None
        snapshot = {name: np.asarray(data[name]) for name in SNAPSHOT_ARRAYS}
        snapshot.update({
            name: np.asarray(data[name])
            for name in OPTIONAL_SNAPSHOT_ARRAYS
            if name in data
        })
        metrics = json.loads(str(data["metrics"].item()))
    manifest = path.with_suffix(".json")
    if manifest.is_file():
        manifest_data = json.loads(manifest.read_text())
        metrics.update(manifest_data.get("metrics", {}))
        config = manifest_data.get("config", {})
        metrics.setdefault("obstacle_mode", config.get("obstacle_mode", "fixed"))
    snapshot["obstacle_mode"] = metrics.get("obstacle_mode", "fixed")
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
            "body_density": 500.0,
            "body_center_x": 2.35,
            "body_center_z": 0.30,
            "body_length": 0.32,
            "body_width": 0.28,
            "body_height": 0.20,
            "kernel": "wendland",
            "alpha": 0.25,
            "xsph_eps": 0.5,
            "cfl": 0.3,
            "n_damp": 50,
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
            ],
            "particle_scale": 0.55,
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
        self.state.change("colorbar_visible")(self._on_colorbar_visible)
        self.state.change("frame_index")(self._on_frame_index)
        self.state.change("viewport_size")(self._on_viewport_size)

    def _config(self):
        config = {
            "resolution_mode": self.state.resolution_mode,
            "dx": float(self.state.dx),
            "steps": int(self.state.steps),
            "adapt_every": int(self.state.adapt_every),
            "max_splits_per_adapt": int(self.state.max_splits),
            "obstacle_mode": self.state.obstacle_mode,
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
        return validate_run_config(config, self.state.snapshot_stride)

    def start_run(self):
        if self.worker.alive:
            return
        self.frames.clear()
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
                snapshot_stride=int(self.state.snapshot_stride),
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
        if self.worker.send("cancel"):
            self.state.status_detail = "Cancellation requested"

    def reset_camera(self):
        self.scene.reset_camera()
        self._refresh_view()

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
        self._refresh_view()

    def _on_obstacle_mode(self, obstacle_mode, **_):
        self.scene.set_obstacle_mode(obstacle_mode)
        self.scene.set_obstacle_visible(self.state.obstacle_visible)
        self._refresh_view()

    def _on_colorbar_visible(self, colorbar_visible, **_):
        self.scene.set_colorbar_visible(colorbar_visible)
        self._refresh_view()

    def _on_viewport_size(self, viewport_size, **_):
        """Match the offscreen render window to the browser panel.

        Without this the server renders at a fixed aspect and the frame is
        letterboxed inside the viewport.
        """
        if not viewport_size:
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
        self.scene.update(snapshot)
        if update_metrics:
            self.state.update({
                "step": metrics.get("step", 0),
                "sim_time": metrics.get("time", 0.0),
                "dt_last": metrics.get("dt_last"),
                "fluid_particles": metrics.get("fluid_particles", 0),
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
                "device_name": metrics.get("runtime", {}).get(
                    "device_name", self.state.device_name
                ),
            })
        self.state.frame_image = self.scene.jpeg_data_uri()
        self.ctrl.view_update()

    def _refresh_view(self, **_):
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
                    html.H2("Dam-break controls", classes="text-h5 mb-1")
                    html.P(
                        "Configure the physics, then inspect every particle "
                        "without leaving the browser.",
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
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
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
                            with v3.VCol(cols=6):
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
                        )
                    with v3.VExpansionPanels(
                        variant="accordion", classes="mb-4", flat=True
                    ):
                        with v3.VExpansionPanel(title="Adaptive region"):
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
                        with v3.VExpansionPanel(title="Physics"):
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
                            v_show=("obstacle_mode === 'floating'",),
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
                                            v_model=("body_center_z", 0.30),
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
                                v3.VSelect(
                                    v_model=("scalar", "pressure"),
                                    items=("scalar_items",),
                                    label="Color particles by",
                                    variant="outlined",
                                    density="compact",
                                )
                                v3.VSlider(
                                    v_model=("particle_scale", 0.55),
                                    min=0.2,
                                    max=0.9,
                                    step=0.05,
                                    label="Particle size",
                                    color="cyan",
                                    thumb_label=True,
                                )
                                v3.VSlider(
                                    v_model=("wall_opacity", 0.2),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="Wall opacity",
                                    color="cyan",
                                    thumb_label=True,
                                )
                                v3.VChip(
                                    text="RTX server rendering",
                                    prepend_icon="mdi-server",
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
                        html.Img(
                            src=("frame_image",),
                            v_show=("frame_image.length > 0",),
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
                            ):
                                v3.VBtn(
                                    "Pressure",
                                    value="pressure",
                                    prepend_icon="mdi-gauge",
                                    size="small",
                                )
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
                                icon=(
                                    "colorbar_visible ? 'mdi-eye' : 'mdi-eye-off'"
                                ),
                                variant="tonal",
                                size="small",
                                classes="viewport-btn",
                                click="colorbar_visible = !colorbar_visible",
                                title="Show or hide the color scale",
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
                            v_show=("frame_image.length === 0 && !run_active",),
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
                        v_show=("right_panel_open",),
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
                        html.Div("PARTICLES", classes="eyebrow mb-2")
                        with html.Div(classes="details-grid"):
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
                            v_show=("obstacle_mode === 'floating'",),
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
