#!/usr/bin/env python3
"""Measure the ADR-0007 dense-per-level memory and warm-runtime gate."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from compyle.api import get_config
import warp as wp

from pysph.base.utils import get_particle_array
from pysph.base.warp_multilevel_nnps import MultilevelGridWarpNNPS
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import (
    compute_summation_density, compute_wcsph_accel_continuity,
)


H_REF = 0.03
LEVEL_RATIO = 2.0
NLEVELS = 5
RADIUS_SCALE = 2.0


def _block(n, offset):
    q = np.arange(n, dtype=np.float32) * np.float32(0.04)
    x, y, z = np.meshgrid(q, q, q, indexing='ij')
    off = np.asarray(offset, dtype=np.float32)
    return (x.ravel() + off[0], y.ravel() + off[1], z.ravel() + off[2])


def make_case(name):
    if name == 'compact':
        fine = _block(13, (0.0, 0.0, 0.0))
    elif name == 'slab':
        q = np.arange(25, dtype=np.float32) * np.float32(0.04)
        r = np.arange(3, dtype=np.float32) * np.float32(0.04)
        fine = tuple(a.ravel() for a in np.meshgrid(q, q, r, indexing='ij'))
    elif name == 'large-slab':
        q = np.arange(40, dtype=np.float32) * np.float32(0.04)
        r = np.arange(3, dtype=np.float32) * np.float32(0.04)
        fine = tuple(a.ravel() for a in np.meshgrid(q, q, r, indexing='ij'))
    elif name == 'two-patches':
        a = _block(7, (0.0, 0.0, 0.0))
        b = _block(7, (2.0, 2.0, 2.0))
        fine = tuple(np.concatenate((u, v)) for u, v in zip(a, b))
    else:
        raise ValueError(name)

    corners = np.asarray([
        (x, y, z) for x in (0.0, 4.0)
        for y in (0.0, 4.0) for z in (0.0, 4.0)
    ], dtype=np.float32)
    x = np.concatenate((fine[0], corners[:, 0]))
    y = np.concatenate((fine[1], corners[:, 1]))
    z = np.concatenate((fine[2], corners[:, 2]))
    h = np.concatenate((
        np.full(fine[0].size, H_REF, dtype=np.float32),
        np.full(corners.shape[0], 0.48, dtype=np.float32),
    ))
    return x, y, z, h


def make_pa(name, arrays):
    x, y, z, h = arrays
    phase = np.arange(x.size, dtype=np.float32)
    return get_particle_array(
        name=name, x=x, y=y, z=z, h=h,
        m=np.ones(x.size, dtype=np.float32),
        rho=np.ones(x.size, dtype=np.float32),
        p=np.float32(1.0) + np.float32(0.01) * (phase % 11),
        cs=np.full(x.size, 5.0, dtype=np.float32),
        u=np.float32(0.01) * (phase % 7),
        v=np.float32(0.01) * (phase % 5),
        w=np.float32(0.01) * (phase % 3),
        au=np.zeros(x.size, dtype=np.float32),
        av=np.zeros(x.size, dtype=np.float32),
        aw=np.zeros(x.size, dtype=np.float32),
        arho=np.zeros(x.size, dtype=np.float32),
        ax=np.zeros(x.size, dtype=np.float32),
        ay=np.zeros(x.size, dtype=np.float32),
        az=np.zeros(x.size, dtype=np.float32),
        backend='warp',
    )


def uniform_candidate_pairs(nnps):
    grid = nnps._build_grid(0)
    counts = grid['counts'].numpy()
    b, cs = nnps._bounds, nnps.cell_size
    nx, ny, nz = b['nx'], b['ny'], b['nz']
    gpu = nnps.particles[0].gpu
    x, y, z = gpu.x.get(), gpu.y.get(), gpu.z.get()

    def cell0(c, cmin, n):
        return min(max(int(np.floor((c - cmin) / cs)), 0), n - 1)

    total = 0
    for px, py, pz in zip(x, y, z):
        ix0 = cell0(px, b['xmin'], nx)
        iy0 = cell0(py, b['ymin'], ny)
        iz0 = cell0(pz, b['zmin'], nz)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ix, iy, iz = ix0 + dx, iy0 + dy, iz0 + dz
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        total += int(counts[ix + iy * nx + iz * nx * ny])
    return total


def timed_build(nnps, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        nnps.update(push=False)
        if isinstance(nnps, MultilevelGridWarpNNPS):
            nnps._build_multilevel(0)
        else:
            nnps._build_grid(0)
        wp.synchronize_device(nnps.device)
        samples.append(time.perf_counter() - start)
    return samples


def timed_density(nnps, mode, repeats):
    compute_summation_density(nnps, push=False, neighbor_mode=mode)
    wp.synchronize_device(nnps.device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute_summation_density(nnps, push=False, neighbor_mode=mode)
        wp.synchronize_device(nnps.device)
        samples.append(time.perf_counter() - start)
    rho = nnps.particles[0].gpu.rho.get()
    digest = hashlib.sha256(rho.tobytes()).hexdigest()
    return samples, digest, float(np.sum(rho, dtype=np.float64))


def timed_fused(nnps, mode, repeats):
    kwargs = dict(
        alpha=0.1, beta=0.0, eps=0.5, c0=5.0,
        kernel='cubic', push=False, neighbor_mode=mode,
    )
    compute_wcsph_accel_continuity(nnps, **kwargs)
    wp.synchronize_device(nnps.device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute_wcsph_accel_continuity(nnps, **kwargs)
        wp.synchronize_device(nnps.device)
        samples.append(time.perf_counter() - start)
    gpu = nnps.particles[0].gpu
    outputs = np.column_stack([
        gpu.get_device_array(name).get()
        for name in ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')
    ])
    return (samples, hashlib.sha256(outputs.tobytes()).hexdigest(),
            float(np.sum(outputs, dtype=np.float64)))


def stats(values):
    return {
        'samples_s': values,
        'median_s': float(np.median(values)),
        'min_s': float(np.min(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=(
        'compact', 'slab', 'large-slab', 'two-patches'),
                        required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output')
    args = parser.parse_args()

    cfg = get_config()
    old_double = cfg.use_double
    cfg.use_double = False
    try:
        arrays = make_case(args.case)
        ml_pa = make_pa('ml', arrays)
        uniform_pa = make_pa('uniform', arrays)
    finally:
        cfg.use_double = old_double

    ml = MultilevelGridWarpNNPS(
        dim=3, particles=[ml_pa], radius_scale=RADIUS_SCALE,
        h_ref=H_REF, level_ratio=LEVEL_RATIO, nlevels=NLEVELS,
    )
    uniform = UniformGridWarpNNPS(
        dim=3, particles=[uniform_pa], radius_scale=RADIUS_SCALE,
    )
    device = wp.get_device(ml.device)
    pool_before = wp.get_mempool_used_mem_current(device)
    ml_build = timed_build(ml, args.repeats)
    pool_after_ml = wp.get_mempool_used_mem_current(device)
    uniform_build = timed_build(uniform, args.repeats)
    pool_after_uniform = wp.get_mempool_used_mem_current(device)

    ml_data = ml._build_multilevel(0)
    n = arrays[0].size
    total_cells = int(ml_data['total_cells'])
    # Persistent dense representation: counts+starts per cell, cell particle
    # index + level assignment per particle, and 9 fp32/int32 level arrays.
    dense_persistent_bytes = 8 * total_cells + 8 * n + 36 * NLEVELS
    saved_state_bytes = 7 * 4 * n  # x0,y0,z0,u0,v0,w0,rho0 in WCSPH PEC.
    counts = ml_data['counts'].numpy()
    occupied_cells = int(np.count_nonzero(counts))
    # Comparison estimate for a sorted sparse representation: uint64 key plus
    # int32 start/count per occupied cell, same particle/level arrays+metadata.
    sparse_estimate_bytes = 16 * occupied_cells + 8 * n + 36 * NLEVELS
    metadata_readback_bytes = 32 * NLEVELS + 4

    ml_cache = ml.build_neighbor_cache_gpu(0, 0)
    uniform_cache = uniform.build_neighbor_cache_gpu(0, 0)
    ml_fused, ml_fused_hash, ml_fused_sum = timed_fused(
        ml, 'multilevel', args.repeats)
    uniform_fused, uniform_fused_hash, uniform_fused_sum = timed_fused(
        uniform, 'grid', args.repeats)
    ml_density, ml_hash, ml_sum = timed_density(
        ml, 'multilevel', args.repeats)
    uniform_density, uniform_hash, uniform_sum = timed_density(
        uniform, 'grid', args.repeats)

    result = {
        'case': args.case,
        'precision': 'fp32',
        'particles': int(n),
        'fine_particles': int(n - 8),
        'levels': NLEVELS,
        'device': device.name,
        'device_arch': device.arch,
        'warp_version': wp.__version__,
        'python': platform.python_version(),
        'total_dense_cells': total_cells,
        'occupied_cells': occupied_cells,
        'dense_persistent_bytes': dense_persistent_bytes,
        'sparse_sorted_estimate_bytes': sparse_estimate_bytes,
        'saved_wcsph_state_bytes': saved_state_bytes,
        'dense_over_saved_state': dense_persistent_bytes / saved_state_bytes,
        'dense_over_sparse_estimate': (
            dense_persistent_bytes / sparse_estimate_bytes),
        'metadata_readback_bytes_per_build': metadata_readback_bytes,
        'mempool_current_bytes': {
            'before_builds': pool_before,
            'after_multilevel': pool_after_ml,
            'after_uniform': pool_after_uniform,
            'multilevel_delta': pool_after_ml - pool_before,
        },
        'candidate_pairs': {
            'multilevel': int(ml.candidate_pairs(0, 0)),
            'uniform': int(uniform_candidate_pairs(uniform)),
        },
        'accepted_pairs': {
            'multilevel': int(ml_cache['total_neighbors']),
            'uniform': int(uniform_cache['total_neighbors']),
        },
        'build': {
            'multilevel': stats(ml_build),
            'uniform': stats(uniform_build),
        },
        'summation_density': {
            'multilevel': stats(ml_density),
            'uniform': stats(uniform_density),
            'multilevel_sha256': ml_hash,
            'uniform_sha256': uniform_hash,
            'multilevel_sum': ml_sum,
            'uniform_sum': uniform_sum,
            'sum_relative_delta': abs(ml_sum - uniform_sum) / abs(uniform_sum),
        },
        'fused_wcsph': {
            'multilevel': stats(ml_fused),
            'uniform': stats(uniform_fused),
            'multilevel_sha256': ml_fused_hash,
            'uniform_sha256': uniform_fused_hash,
            'multilevel_sum': ml_fused_sum,
            'uniform_sum': uniform_fused_sum,
            'sum_relative_delta': (
                abs(ml_fused_sum - uniform_fused_sum)
                / max(abs(uniform_fused_sum), 1.0e-30)),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
