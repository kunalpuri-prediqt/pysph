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
from trame.widgets import html, vtk, vuetify3 as v3

from vtk_scene import ParticleScene, SCALARS
from worker import FrameBuffer, SolverWorker, TERMINAL_STATES


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "/tmp/pysph-dam-break-studio.npz"
SNAPSHOT_ARRAYS = ("xyz", "h", "rho", "p", "speed", "kind", "level")


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
        metrics = json.loads(str(data["metrics"].item()))
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
            "with_obstacle": True,
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
            "particle_scale": 0.045,
            "wall_opacity": 0.20,
            "obstacle_visible": True,
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
            "mass_drift": 0.0,
            "p_max": 0.0,
            "steps_per_second": None,
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
        self.state.change("frame_index")(self._on_frame_index)

    def _config(self):
        config = {
            "resolution_mode": self.state.resolution_mode,
            "dx": float(self.state.dx),
            "steps": int(self.state.steps),
            "adapt_every": int(self.state.adapt_every),
            "max_splits_per_adapt": int(self.state.max_splits),
            "with_obstacle": bool(self.state.with_obstacle),
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
                "mass_drift": metrics.get("mass_drift", 0.0),
                "p_max": metrics.get("p_max", 0.0),
                "steps_per_second": metrics.get("steps_per_second"),
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
        css = """
        :root {
          --studio-bg: #07101f;
          --studio-panel: rgba(12, 24, 44, .92);
          --studio-line: rgba(140, 180, 230, .14);
          --studio-cyan: #42d8f5;
          --studio-orange: #ff7345;
        }
        html, body, #app { background: var(--studio-bg); overflow: hidden; }
        .studio-shell { background:
          radial-gradient(circle at 70% 0%, rgba(28, 92, 155, .20), transparent 38%),
          #07101f; }
        .studio-toolbar {
          backdrop-filter: blur(18px);
          border-bottom: 1px solid var(--studio-line) !important;
          background: rgba(7, 16, 31, .82) !important;
        }
        .studio-drawer {
          background: var(--studio-panel) !important;
          border-right: 1px solid var(--studio-line) !important;
        }
        .eyebrow { color: #70dff5; letter-spacing: .16em; font-size: .68rem;
          text-transform: uppercase; font-weight: 700; }
        .metric { border: 1px solid var(--studio-line); background:
          linear-gradient(145deg, rgba(25, 48, 78, .60), rgba(10, 21, 39, .72));
          border-radius: 14px; padding: 10px 12px; min-height: 68px; }
        .metric-label { color: #8ca4c2; font-size: .68rem; text-transform:
          uppercase; letter-spacing: .08em; }
        .metric-value { color: #f4f8ff; font-size: 1.15rem; font-weight: 650; }
        .studio-main { height: 100vh !important; max-height: 100vh !important;
          overflow: hidden; background: #07101f; }
        .viewport-container { position: relative; height: calc(100vh - 64px)
          !important; min-height: calc(100vh - 64px); overflow: hidden; }
        .viewport-wrap { position: absolute; inset: 0; min-height: 420px;
          overflow: hidden; background: #07101f; }
        .viewport-fallback { position: absolute; inset: 0; width: 100%;
          height: 100%; object-fit: contain; z-index: 2; pointer-events: none;
          background: #07101f; }
        .viewport-remote { position: absolute !important; inset: 0; width: 100%;
          height: 100%; z-index: 1; background: transparent !important; }
        .viewport-hud { position: absolute; top: 18px; left: 18px; z-index: 3;
          background: rgba(7, 16, 31, .72); border: 1px solid var(--studio-line);
          backdrop-filter: blur(14px); border-radius: 16px; padding: 12px 15px;
          pointer-events: none; }
        .timeline { position: absolute; left: 24px; right: 24px; bottom: 18px;
          z-index: 4; background: rgba(7, 16, 31, .84);
          border: 1px solid var(--studio-line); backdrop-filter: blur(14px);
          border-radius: 16px; padding: 4px 18px 0; }
        """
        with SinglePageWithDrawerLayout(
            self.server, full_height=True, theme="dark"
        ) as layout:
            layout.root["classes"] = "studio-shell"
            layout.root["style"] = "height:100vh; min-height:100vh;"
            layout.toolbar["classes"] = "studio-toolbar"
            layout.drawer["classes"] = "studio-drawer"
            layout.content["classes"] = "studio-main"
            layout.drawer["width"] = 368
            layout.title.set_text("PySPH · Warp Studio")
            with layout.toolbar:
                v3.VSpacer()
                v3.VChip(
                    text=("status.toUpperCase()",),
                    color=(
                        "status === 'running' ? 'cyan' : "
                        "status === 'failed' ? 'error' : "
                        "status === 'completed' ? 'success' : 'blue-grey'"
                    ),
                    variant="tonal",
                    size="small",
                    classes="mr-3",
                )
                v3.VBtn(
                    icon="mdi-crosshairs-gps",
                    variant="text",
                    click=self.ctrl.reset_camera,
                    title="Reset camera",
                )
            with layout.drawer:
                with v3.VContainer(classes="pa-5"):
                    html.Div("GPU SIMULATION", classes="eyebrow mb-1")
                    html.H2("Dam-break controls", classes="text-h5 mb-1")
                    html.P(
                        "Configure the physics, then inspect every particle "
                        "without leaving the browser.",
                        classes="text-body-2 text-medium-emphasis mb-5",
                    )
                    v3.VSelect(
                        v_model=("resolution_mode", "adaptive"),
                        items=("mode_items",),
                        label="Resolution mode",
                        variant="outlined",
                        density="compact",
                        disabled=("run_active",),
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
                                disabled=("run_active",),
                            )
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("steps", 250),
                                label="Steps",
                                type="number",
                                variant="outlined",
                                density="compact",
                                disabled=("run_active",),
                            )
                    with v3.VExpansionPanels(variant="accordion", classes="mb-4"):
                        with v3.VExpansionPanel(title="Adaptive region"):
                            with v3.VExpansionPanelText():
                                v3.VTextField(
                                    v_model=("adapt_every", 10),
                                    label="Adapt every N steps",
                                    type="number",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VTextField(
                                    v_model=("max_splits", 128),
                                    label="Max splits / checkpoint",
                                    type="number",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VTextField(
                                    v_model=("snapshot_stride", 5),
                                    label="Visualize every N steps",
                                    type="number",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("fine_xmin", 1.75),
                                            label="Fine x min",
                                            type="number",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                    with v3.VCol(cols=6):
                                        v3.VTextField(
                                            v_model=("fine_xmax", 2.8),
                                            label="Fine x max",
                                            type="number",
                                            density="compact",
                                            disabled=("run_active",),
                                        )
                                v3.VTextField(
                                    v_model=("fine_zmax", 0.65),
                                    label="Fine-region height",
                                    type="number",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(title="Physics"):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    v_model=("kernel", "wendland"),
                                    items=("kernel_items",),
                                    label="SPH kernel",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("alpha", 0.25),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="Viscosity α",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                                v3.VSlider(
                                    v_model=("xsph_eps", 0.5),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="XSPH ε",
                                    thumb_label=True,
                                    disabled=("run_active",),
                                )
                        with v3.VExpansionPanel(title="Visualization"):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    v_model=("scalar", "pressure"),
                                    items=("scalar_items",),
                                    label="Color particles by",
                                    density="compact",
                                )
                                v3.VSlider(
                                    v_model=("particle_scale", 0.045),
                                    min=0.01,
                                    max=0.12,
                                    step=0.005,
                                    label="Particle radius",
                                    thumb_label=True,
                                )
                                v3.VSlider(
                                    v_model=("wall_opacity", 0.2),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    label="Wall opacity",
                                    thumb_label=True,
                                )
                                v3.VChip(
                                    text="RTX server rendering",
                                    prepend_icon="mdi-server",
                                    color="cyan",
                                    variant="tonal",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    v_model=("output_path", DEFAULT_OUTPUT),
                                    label="Result NPZ",
                                    density="compact",
                                    disabled=("run_active",),
                                )
                    v3.VSwitch(
                        v_model=("with_obstacle", True),
                        label="Fixed obstacle",
                        color="deep-orange",
                        disabled=("run_active",),
                        density="compact",
                    )
                    with v3.VRow(dense=True, classes="mt-2"):
                        with v3.VCol(cols=12):
                            v3.VBtn(
                                text="Launch GPU run",
                                prepend_icon="mdi-rocket-launch",
                                color="cyan",
                                block=True,
                                size="large",
                                click=self.ctrl.start_run,
                                disabled=("run_active",),
                            )
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=4):
                            v3.VBtn(
                                text="Pause",
                                block=True,
                                variant="tonal",
                                click=self.ctrl.pause_run,
                                disabled=("!run_active || paused",),
                            )
                        with v3.VCol(cols=4):
                            v3.VBtn(
                                text="Resume",
                                block=True,
                                variant="tonal",
                                click=self.ctrl.resume_run,
                                disabled=("!run_active || !paused",),
                            )
                        with v3.VCol(cols=4):
                            v3.VBtn(
                                text="Step",
                                block=True,
                                variant="tonal",
                                click=self.ctrl.single_step,
                                disabled=("!run_active || !paused",),
                            )
                    v3.VBtn(
                        text="Cancel run",
                        block=True,
                        variant="text",
                        color="error",
                        click=self.ctrl.cancel_run,
                        disabled=("!run_active",),
                    )
                    html.Div(
                        "{{ status_detail }}",
                        classes="text-caption text-medium-emphasis mt-3",
                    )
                    v3.VAlert(
                        text=("error_text",),
                        type="error",
                        variant="tonal",
                        classes="mt-3",
                        v_show=("error_text.length > 0",),
                    )
            with layout.content:
                html.Style(css)
                with v3.VContainer(
                    fluid=True, classes="pa-0 fill-height viewport-container"
                ):
                    with html.Div(classes="viewport-wrap"):
                        html.Img(
                            src=("frame_image",),
                            v_show=("frame_image.length > 0",),
                            classes="viewport-fallback",
                            alt="Rendered adaptive particle field",
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
                        )
                        self.ctrl.view_update = view.update
                        self.ctrl.view_resize = view.resize
                        self.ctrl.view_reset_camera = view.reset_camera
                        with html.Div(classes="viewport-hud"):
                            html.Div("LIVE PARTICLE FIELD", classes="eyebrow")
                            html.Div(
                                "{{ step.toLocaleString() }} / "
                                "{{ step_total.toLocaleString() }} steps",
                                classes="text-h6",
                            )
                            html.Div(
                                "t = {{ sim_time.toExponential(3) }} s · "
                                "{{ fluid_particles.toLocaleString() }} fluid",
                                classes="text-caption text-medium-emphasis",
                            )
                        with html.Div(classes="timeline"):
                            v3.VSlider(
                                v_model=("frame_index", 0),
                                min=0,
                                max=("frame_max", 0),
                                step=1,
                                hide_details=True,
                                color="cyan",
                                prepend_icon=(
                                    "live_view ? 'mdi-access-point' : "
                                    "'mdi-history'"
                                ),
                            )
                    with html.Div(
                        style=(
                            "position:absolute; right:20px; top:20px; width:220px; "
                            "z-index:5; display:grid; gap:8px;"
                        )
                    ):
                        for label, expression in (
                            ("Fine / coarse", "fine_particles + ' / ' + coarse_particles"),
                            ("Split / merged", "split_parents + ' / ' + merged_families"),
                            ("Mass drift", "Number(mass_drift).toExponential(2)"),
                            ("Peak pressure", "Number(p_max).toExponential(2) + ' Pa'"),
                        ):
                            with html.Div(classes="metric"):
                                html.Div(label, classes="metric-label")
                                html.Div(
                                    f"{{{{ {expression} }}}}",
                                    classes="metric-value",
                                )
            layout.footer.hide()
        self.ui = layout


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
