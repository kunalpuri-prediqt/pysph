"""VTK particle scene used by the Trame studio."""

from __future__ import annotations

import base64

import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkPlaneSource, vtkSphereSource
from vtkmodules.vtkIOImage import vtkJPEGWriter
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkGlyph3DMapper,
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

SCALAR_TITLES = {
    "pressure": "Pressure (Pa)",
    "density": "Density",
    "speed": "Speed (m/s)",
    "resolution": "Refinement",
}


class PointCloud:
    def __init__(self, color=(0.2, 0.6, 1.0), opacity=1.0,
                 gaussian=True, spheres=False, scale_factor=0.035):
        self.polydata = vtkPolyData()
        self.points = vtkPoints()
        self.polydata.SetPoints(self.points)
        if spheres:
            self.source = vtkSphereSource()
            self.source.SetRadius(0.5)
            self.source.SetThetaResolution(10)
            self.source.SetPhiResolution(8)
            self.mapper = vtkGlyph3DMapper()
            self.mapper.SetSourceConnection(self.source.GetOutputPort())
            self.mapper.SetScaleArray("h")
            self.mapper.SetScaleModeToScaleByMagnitude()
            self.mapper.SetScaleFactor(scale_factor)
            self.mapper.ScalingOn()
        elif gaussian:
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

    def set_scale_factor(self, value):
        if hasattr(self.mapper, "SetScaleFactor"):
            self.mapper.SetScaleFactor(float(value))

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
        self.renderer.SetBackground(0.012, 0.020, 0.043)
        self.renderer.SetBackground2(0.043, 0.086, 0.132)
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
            gaussian=False, spheres=True, scale_factor=0.55,
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
        self.scalar_bar.SetTitle("Refinement")
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetLabelFormat("%.2g")
        self.scalar_bar.SetOrientationToVertical()
        self.scalar_bar.SetPosition(0.918, 0.08)
        self.scalar_bar.SetWidth(0.06)
        self.scalar_bar.SetHeight(0.84)
        self.scalar_bar.SetMaximumWidthInPixels(96)
        self.scalar_bar.SetMaximumHeightInPixels(4000)
        self.scalar_bar.SetBarRatio(0.26)
        self.scalar_bar.SetTextPad(6)
        self.scalar_bar.UnconstrainedFontSizeOn()
        self.scalar_bar.GetTitleTextProperty().SetColor(0.90, 0.95, 1.0)
        self.scalar_bar.GetTitleTextProperty().SetFontSize(15)
        self.scalar_bar.GetTitleTextProperty().SetBold(1)
        self.scalar_bar.GetTitleTextProperty().SetItalic(0)
        self.scalar_bar.GetLabelTextProperty().SetColor(0.76, 0.86, 0.98)
        self.scalar_bar.GetLabelTextProperty().SetFontSize(12)
        self.scalar_bar.GetLabelTextProperty().SetBold(0)
        self.scalar_bar.GetLabelTextProperty().SetItalic(0)
        self.renderer.AddViewProp(self.scalar_bar)
        self.scalar = "resolution"
        self._add_context()
        self._set_camera()

    def _add_context(self):
        # Real container extents (DamBreak3DGeometry, H = 1): the tank is
        # 161/30 long, 0.5 wide, 1.5 tall.
        x_len, y_half, z_top = 161.0 / 30.0, 0.25, 1.5

        floor = vtkPlaneSource()
        floor.SetOrigin(0.0, -y_half, 0.0)
        floor.SetPoint1(x_len, -y_half, 0.0)
        floor.SetPoint2(0.0, y_half, 0.0)
        floor_mapper = vtkPolyDataMapper()
        floor_mapper.SetInputConnection(floor.GetOutputPort())
        floor_actor = vtkActor()
        floor_actor.SetMapper(floor_mapper)
        floor_actor.GetProperty().SetColor(0.09, 0.16, 0.27)
        floor_actor.GetProperty().SetOpacity(0.55)
        floor_actor.GetProperty().SetAmbient(0.6)
        floor_actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(floor_actor)

        grid = vtkPlaneSource()
        grid.SetOrigin(0.0, -y_half, 0.002)
        grid.SetPoint1(x_len, -y_half, 0.002)
        grid.SetPoint2(0.0, y_half, 0.002)
        grid.SetXResolution(28)
        grid.SetYResolution(3)
        grid_mapper = vtkPolyDataMapper()
        grid_mapper.SetInputConnection(grid.GetOutputPort())
        grid_actor = vtkActor()
        grid_actor.SetMapper(grid_mapper)
        grid_actor.GetProperty().SetRepresentationToWireframe()
        grid_actor.GetProperty().SetColor(0.30, 0.46, 0.64)
        grid_actor.GetProperty().SetOpacity(0.30)
        grid_actor.GetProperty().SetLineWidth(1.0)
        grid_actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(grid_actor)

        tank = vtkCubeSource()
        tank.SetBounds(0.0, x_len, -y_half, y_half, 0.0, z_top)
        tank_mapper = vtkPolyDataMapper()
        tank_mapper.SetInputConnection(tank.GetOutputPort())
        tank_actor = vtkActor()
        tank_actor.SetMapper(tank_mapper)
        tank_actor.GetProperty().SetRepresentationToWireframe()
        tank_actor.GetProperty().SetColor(0.36, 0.54, 0.74)
        tank_actor.GetProperty().SetOpacity(0.34)
        tank_actor.GetProperty().SetLineWidth(1.4)
        tank_actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(tank_actor)

        cube = vtkCubeSource()
        cube.SetCenter(3.0, 0.0, 0.0805)
        cube.SetXLength(0.16)
        cube.SetYLength(0.4)
        cube.SetZLength(0.161)
        cube_mapper = vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube.GetOutputPort())
        cube_actor = vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetColor(0.95, 0.25, 0.08)
        cube_actor.GetProperty().SetOpacity(0.82)
        cube_actor.GetProperty().EdgeVisibilityOn()
        cube_actor.GetProperty().SetEdgeColor(1.0, 0.62, 0.22)
        self.renderer.AddActor(cube_actor)
        self.context_obstacle = cube_actor
        self.context_obstacle_source = cube

    def _fit_context_obstacle(self, points):
        """Match the drawn obstacle box to the actual obstacle particles."""
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        self.context_obstacle_source.SetBounds(
            float(lo[0]), float(hi[0]),
            float(lo[1]), float(hi[1]),
            0.0, float(hi[2]),
        )

    def _set_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(5.0, -5.6, 3.2)
        camera.SetFocalPoint(2.5, 0.0, 0.55)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetClippingRange(0.1, 40.0)

    def reset_camera(self):
        self._set_camera()
        self.renderer.ResetCameraClippingRange()

    def set_colorbar_visible(self, visible):
        self.scalar_bar.SetVisibility(bool(visible))

    def set_scalar(self, scalar):
        if scalar not in SCALARS:
            raise ValueError(f"unknown scalar {scalar!r}")
        self.scalar = scalar
        array_name = SCALARS[scalar]
        self.scalar_bar.SetTitle(SCALAR_TITLES.get(scalar, scalar.title()))
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
        self.obstacle.actor.SetVisibility(False)
        if np.any(obstacle_mask):
            self._fit_context_obstacle(xyz[obstacle_mask])
        self.context_obstacle.SetVisibility(bool(np.any(obstacle_mask)))
        self.set_scalar(self.scalar)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def set_particle_scale(self, value):
        self.fluid.set_scale_factor(value)

    def set_wall_opacity(self, value):
        self.wall.actor.GetProperty().SetOpacity(float(value))

    def set_obstacle_visible(self, visible):
        self.obstacle.actor.SetVisibility(False)
        self.context_obstacle.SetVisibility(bool(visible))

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
