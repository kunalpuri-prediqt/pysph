#!/usr/bin/env python3
"""Render a studio NPZ snapshot to PNG using the application's VTK scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

from vtk_scene import ParticleScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--scalar",
        choices=("pressure", "density", "speed", "resolution"),
        default="resolution",
    )
    args = parser.parse_args()
    with np.load(args.input) as data:
        snapshot = {
            name: data[name]
            for name in ("xyz", "h", "rho", "p", "speed", "kind", "level")
        }
    scene = ParticleScene()
    scene.scalar = args.scalar
    scene.set_wall_opacity(0.10)
    scene.update(snapshot)
    scene.render_window.SetSize(1600, 900)
    scene.render_window.Render()
    capture = vtkWindowToImageFilter()
    capture.SetInput(scene.render_window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkPNGWriter()
    writer.SetFileName(str(args.output))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    print(args.output)


if __name__ == "__main__":
    main()
