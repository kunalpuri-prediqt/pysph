#!/usr/bin/env python3
"""Manufactured hydrostatic balance across an icosa13 resolution interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pysph.base.warp_adaptive import (
    TwoLevelAdaptiveController,
    warp_particle_array_from_state,
)
from pysph.base.warp_multilevel_nnps import MultilevelGridWarpNNPS
from pysph.base.warp_codegen import WarpEquation
from pysph.base.warp_sph import (
    VariableHPressureGradient,
    _run_equation_group,
    compute_pressure_gradient,
    compute_variable_h_beta,
)


class AveragedGradHCorrection(WarpEquation):
    """Historical averaged-HIJ beta retained only for the A/B diagnostic."""
    src_arrays = ("m",)
    dst_arrays = ("rho",)
    out_arrays = ("beta_h",)
    scalars = ("dim_inv",)
    requires = ("rij2", "grad")

    def loop(self):
        return (
            "        _acc_beta_h += -s_m[j] * rij2 * grad * dim_inv"
            " / d_rho[i]"
        )

    def post_loop(self):
        return (
            "    if wp.abs(d_beta_h[i]) < TYPE(1.0e-8):\n"
            "        d_beta_h[i] = TYPE(1.0)"
        )


class AveragedVariableHPressureGradient(WarpEquation):
    """Historical averaged-HIJ pressure correction for comparison only."""
    src_arrays = ("m", "rho", "p", "beta_h")
    dst_arrays = ("rho", "p", "beta_h")
    out_arrays = ("au", "av", "aw")
    requires = ("dx", "dy", "dz", "grad")

    def initialize(self):
        return (
            "    vh_rhoi21_ = TYPE(1.0) / (d_rho[i] * d_rho[i])\n"
            "    vh_tmpi_ = d_p[i] * vh_rhoi21_ / d_beta_h[i]"
        )

    def loop(self):
        return (
            "        vh_rhoj21_ = TYPE(1.0) / (s_rho[j] * s_rho[j])\n"
            "        vh_pg_ = vh_tmpi_"
            " + s_p[j] * vh_rhoj21_ / s_beta_h[j]\n"
            "        vh_fac_ = -s_m[j] * vh_pg_ * grad\n"
            "        _acc_au += vh_fac_ * dx\n"
            "        _acc_av += vh_fac_ * dy\n"
            "        _acc_aw += vh_fac_ * dz"
        )


def manufactured_state(dx=0.1, rho0=1000.0, gravity=9.81):
    # Pad every measured region by more than the coarse 2h support. The
    # original +/-0.3 m cube contaminated the nominal interface mask with
    # truncated-support boundary error (coarse support is 0.26 m).
    xy_axis = np.arange(-0.6, 0.6001, dx)
    z_axis = np.arange(-0.1, 0.9001, dx)
    x, y, z = np.meshgrid(xy_axis, xy_axis, z_axis, indexing="ij")
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    count = len(xyz)
    zeros = np.zeros(count)
    state = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "x0": xyz[:, 0].copy(),
        "y0": xyz[:, 1].copy(),
        "z0": xyz[:, 2].copy(),
        "h": np.full(count, 1.3 * dx),
        "m": np.full(count, rho0 * dx**3),
        "rho": np.full(count, rho0),
        "rho0": np.full(count, rho0),
        "p": rho0 * gravity * (0.8 - xyz[:, 2]),
        "u": zeros.copy(),
        "v": zeros.copy(),
        "w": zeros.copy(),
        "u0": zeros.copy(),
        "v0": zeros.copy(),
        "w0": zeros.copy(),
        "au": zeros.copy(),
        "av": zeros.copy(),
        "aw": zeros.copy(),
        "beta_h": np.ones(count),
    }
    controller = TwoLevelAdaptiveController(
        hdx=1.3,
        fine_bounds=(-0.001, 1.0, -1.0, 1.0, -1.0, 1.0),
        max_splits_per_adapt=count,
        split_stencil="icosa13",
        hysteresis=0.0,
        shift_iterations=0,
    )
    return controller.adapt(state)[0]


def pressure_acceleration(state, mode):
    pa = warp_particle_array_from_state(state, name="fluid")
    nnps = MultilevelGridWarpNNPS(
        dim=3,
        particles=[pa],
        radius_scale=2.0,
        h_ref=0.065,
        level_ratio=2.0,
        nlevels=2,
    )
    if mode == "separate_h":
        compute_variable_h_beta(
            nnps, 0, [0], kernel="wendland", push=False,
            neighbor_mode="multilevel",
        )
        _run_equation_group(
            nnps, 0, 0, [VariableHPressureGradient()],
            kernel="wendland", neighbor_mode="multilevel",
        )
    elif mode == "averaged_hij":
        _run_equation_group(
            nnps, 0, 0, [AveragedGradHCorrection()],
            scalar_values={"dim_inv": 1.0 / 3.0},
            kernel="wendland", neighbor_mode="multilevel",
        )
        _run_equation_group(
            nnps, 0, 0, [AveragedVariableHPressureGradient()],
            kernel="wendland", neighbor_mode="multilevel",
        )
    elif mode == "uncorrected":
        compute_pressure_gradient(
            nnps, 0, 0, kernel="wendland", push=False,
            neighbor_mode="multilevel",
        )
    else:
        raise ValueError("mode must be uncorrected, averaged_hij, or separate_h")
    pa.gpu.pull("au", "av", "aw", "beta_h")
    return {
        "au": np.asarray(pa.au, dtype=np.float64),
        "av": np.asarray(pa.av, dtype=np.float64),
        "aw": np.asarray(pa.aw, dtype=np.float64),
        "beta_h": np.asarray(pa.beta_h, dtype=np.float64),
    }


def summarize(state, result, gravity=9.81):
    x = np.asarray(state["x"])
    y = np.asarray(state["y"])
    z = np.asarray(state["z"])
    level = np.asarray(state["level"], dtype=np.int32)
    interior = (np.abs(y) <= 0.11) & (z >= 0.28) & (z <= 0.52)
    masks = {
        "coarse_bulk": interior & (x < -0.18),
        "interface": interior & (x >= -0.18) & (x <= 0.14),
        "fine_bulk": interior & (x > 0.14) & (x < 0.27),
    }
    horizontal = np.hypot(result["au"], result["av"])
    vertical_residual = result["aw"] - gravity
    groups = {}
    for name, mask in masks.items():
        groups[name] = {
            "particles": int(np.count_nonzero(mask)),
            "horizontal_rms": float(np.sqrt(np.mean(horizontal[mask] ** 2))),
            "vertical_residual_rms": float(np.sqrt(
                np.mean(vertical_residual[mask] ** 2)
            )),
            "vertical_residual_bias": float(np.mean(vertical_residual[mask])),
        }
    return {
        "particles": len(x),
        "coarse_particles": int(np.count_nonzero(level == 0)),
        "fine_particles": int(np.count_nonzero(level == 1)),
        "beta_h_min": float(np.min(result["beta_h"])),
        "beta_h_max": float(np.max(result["beta_h"])),
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    state = manufactured_state()
    report = {mode: summarize(state, pressure_acceleration(state, mode))
              for mode in ("uncorrected", "averaged_hij", "separate_h")}
    before = report["uncorrected"]["groups"]["interface"]
    report["interface_change_vs_uncorrected"] = {}
    for mode in ("averaged_hij", "separate_h"):
        after = report[mode]["groups"]["interface"]
        report["interface_change_vs_uncorrected"][mode] = {
            "horizontal_rms_ratio": (
                after["horizontal_rms"] / before["horizontal_rms"]
            ),
            "vertical_residual_rms_ratio": (
                after["vertical_residual_rms"]
                / before["vertical_residual_rms"]
            ),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
