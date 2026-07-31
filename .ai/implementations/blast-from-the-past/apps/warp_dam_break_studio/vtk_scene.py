"""VTK particle scene used by the Trame studio."""

from __future__ import annotations

import base64

import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkPlaneSource
from vtkmodules.vtkIOImage import vtkJPEGWriter
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPointGaussianMapper,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401


SCALARS = {
    "pressure": "p",
    "density": "rho",
    "speed": "speed",
    "resolution": "level",
}


class PointCloud:
    def __init__(self, color=(0.2, 0.6, 1.0), opacity=1.0,
                 gaussian=True, scale_factor=0.035):
        self.polydata = vtkPolyData()
        self.points = vtkPoints()
        self.polydata.SetPoints(self.points)
        if gaussian:
            self.mapper = vtkPointGaussianMapper()
            self.mapper.SetScaleFactor(scale_factor)
            self.mapper.EmissiveOff()
        else:
            self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputData(self.polydata)
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(*color)
        self.actor.GetProperty().SetOpacity(opacity)
        self.actor.GetProperty().SetPointSize(3.0)
        self.arrays = {}

    def update(self, xyz, arrays=None):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        vtk_xyz = numpy_to_vtk(xyz, deep=True)
        vtk_xyz.SetNumberOfComponents(3)
        self.points.SetData(vtk_xyz)
        self.points.Modified()
        arrays = arrays or {}
        point_data = self.polydata.GetPointData()
        for name in list(self.arrays):
            if name not in arrays:
                point_data.RemoveArray(name)
                self.arrays.pop(name, None)
        for name, values in arrays.items():
            values = np.ascontiguousarray(values, dtype=np.float32)
            vtk_values = numpy_to_vtk(values, deep=True)
            vtk_values.SetName(name)
            existing = point_data.GetArray(name)
            if existing is not None:
                point_data.RemoveArray(name)
            point_data.AddArray(vtk_values)
            self.arrays[name] = vtk_values
        self.polydata.Modified()


class ParticleScene:
    """Three particle actors plus a lightweight flume context."""

    def __init__(self):
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.018, 0.027, 0.055)
        self.renderer.SetBackground2(0.07, 0.105, 0.16)
        self.renderer.GradientBackgroundOn()
        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1280, 760)
        self.render_window.SetOffScreenRendering(1)
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        self.fluid = PointCloud(
            color=(0.15, 0.62, 1.0), opacity=0.94,
            gaussian=True, scale_factor=0.045,
        )
        self.wall = PointCloud(
            color=(0.55, 0.66, 0.82), opacity=0.20,
            gaussian=False,
        )
        self.obstacle = PointCloud(
            color=(1.0, 0.45, 0.12), opacity=0.96,
            gaussian=True, scale_factor=0.07,
        )
        for cloud in (self.wall, self.fluid, self.obstacle):
            self.renderer.AddActor(cloud.actor)

        self.lookup = vtkColorTransferFunction()
        self.lookup.SetColorSpaceToDiverging()
        self.fluid.mapper.SetLookupTable(self.lookup)
        self.fluid.mapper.SetScalarModeToUsePointFieldData()
        self.fluid.mapper.ScalarVisibilityOn()
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lookup)
        self.scalar_bar.SetTitle("Pressure")
        self.scalar_bar.SetNumberOfLabels(4)
        self.scalar_bar.SetPosition(0.82, 0.08)
        self.scalar_bar.SetWidth(0.12)
        self.scalar_bar.SetHeight(0.34)
        self.scalar_bar.GetTitleTextProperty().SetColor(0.85, 0.92, 1.0)
        self.scalar_bar.GetLabelTextProperty().SetColor(0.72, 0.82, 0.94)
        self.renderer.AddViewProp(self.scalar_bar)
        self.scalar = "pressure"
        self._add_context()
        self._set_camera()

    def _add_context(self):
        plane = vtkPlaneSource()
        plane.SetOrigin(0.0, -0.32, 0.0)
        plane.SetPoint1(5.4, -0.32, 0.0)
        plane.SetPoint2(0.0, 0.32, 0.0)
        plane.SetXResolution(27)
        plane.SetYResolution(4)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.18, 0.30, 0.46)
        actor.GetProperty().SetOpacity(0.24)
        self.renderer.AddActor(actor)

        cube = vtkCubeSource()
        cube.SetCenter(2.5, 0.0, 0.3)
        cube.SetXLength(0.12)
        cube.SetYLength(0.36)
        cube.SetZLength(0.6)
        cube_mapper = vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube.GetOutputPort())
        cube_actor = vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetColor(0.95, 0.25, 0.08)
        cube_actor.GetProperty().SetOpacity(0.12)
        self.renderer.AddActor(cube_actor)

        axes = vtkAxesActor()
        axes.SetTotalLength(0.35, 0.35, 0.35)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.035)
        axes.SetPosition(0.0, -0.32, 0.0)
        self.renderer.AddActor(axes)

    def _set_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(4.4, -4.7, 2.8)
        camera.SetFocalPoint(2.25, 0.0, 0.45)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetClippingRange(0.1, 30.0)

    def reset_camera(self):
        self._set_camera()
        self.renderer.ResetCameraClippingRange()

    def set_scalar(self, scalar):
        if scalar not in SCALARS:
            raise ValueError(f"unknown scalar {scalar!r}")
        self.scalar = scalar
        array_name = SCALARS[scalar]
        self.scalar_bar.SetTitle(scalar.title())
        self.fluid.mapper.SelectColorArray(array_name)
        self._update_lookup(array_name)

    def _update_lookup(self, array_name):
        vtk_array = self.fluid.polydata.GetPointData().GetArray(array_name)
        if vtk_array is None or vtk_array.GetNumberOfTuples() == 0:
            return
        lo, hi = vtk_array.GetRange()
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self.lookup.RemoveAllPoints()
        if array_name == "level":
            self.lookup.AddRGBPoint(0.0, 0.10, 0.52, 0.92)
            self.lookup.AddRGBPoint(1.0, 0.98, 0.28, 0.20)
        else:
            mid = 0.5 * (lo + hi)
            self.lookup.AddRGBPoint(lo, 0.08, 0.28, 0.72)
            self.lookup.AddRGBPoint(mid, 0.08, 0.82, 0.92)
            self.lookup.AddRGBPoint(hi, 1.0, 0.28, 0.12)
        self.fluid.mapper.SetScalarRange(lo, hi)
        self.lookup.Build()

    def update(self, snapshot):
        kind = np.asarray(snapshot["kind"])
        xyz = np.asarray(snapshot["xyz"])
        arrays = {
            "p": np.asarray(snapshot["p"]),
            "rho": np.asarray(snapshot["rho"]),
            "speed": np.asarray(snapshot["speed"]),
            "level": np.asarray(snapshot["level"]),
            "h": np.asarray(snapshot["h"]),
        }
        fluid_mask = kind == 0
        wall_mask = kind == 1
        obstacle_mask = kind == 2
        self.fluid.update(
            xyz[fluid_mask],
            {name: value[fluid_mask] for name, value in arrays.items()},
        )
        self.wall.update(xyz[wall_mask])
        self.obstacle.update(xyz[obstacle_mask])
        self.set_scalar(self.scalar)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def set_particle_scale(self, value):
        self.fluid.mapper.SetScaleFactor(float(value))
        self.obstacle.mapper.SetScaleFactor(float(value) * 1.5)

    def set_wall_opacity(self, value):
        self.wall.actor.GetProperty().SetOpacity(float(value))

    def set_obstacle_visible(self, visible):
        self.obstacle.actor.SetVisibility(bool(visible))

    def jpeg_data_uri(self, quality=82):
        """Capture the current render as a browser-safe fallback image."""
        capture = vtkWindowToImageFilter()
        capture.SetInput(self.render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtkJPEGWriter()
        writer.SetInputConnection(capture.GetOutputPort())
        writer.SetQuality(int(quality))
        writer.WriteToMemoryOn()
        writer.Write()
        payload = vtk_to_numpy(writer.GetResult()).tobytes()
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
