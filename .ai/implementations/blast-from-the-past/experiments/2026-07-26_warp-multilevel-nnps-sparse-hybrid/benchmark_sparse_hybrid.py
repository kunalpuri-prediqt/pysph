#!/usr/bin/env python3
"""Measure sparse/hybrid multilevel NNPS memory, build, and fused traversal."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import warp as wp

from pysph.base.warp_multilevel_nnps import MultilevelGridWarpNNPS


HERE = Path(__file__).resolve().parent
OLD_SCRIPT = (
    HERE.parent / '2026-07-13_warp-multilevel-nnps-dense-memory-gate'
    / 'benchmark_dense_memory.py'
)
SPEC = importlib.util.spec_from_file_location('dense_gate', OLD_SCRIPT)
DENSE_GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DENSE_GATE)


def timed_build(nnps, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        nnps.update(push=False)
        nnps._build_multilevel(0)
        wp.synchronize_device(nnps.device)
        samples.append(time.perf_counter() - start)
    return samples


def actual_storage_bytes(ml):
    arrays = (
        'level_of', 'starts', 'counts', 'cell_particles', 'sparse_keys',
        'sparse_starts', 'sparse_counts', 'sparse_particles',
        'origin_x', 'origin_y', 'origin_z', 'cell_size', 'nx', 'ny', 'nz',
        'dense_offset', 'storage_mode', 'virtual_offset', 'level_cells',
        'support',
    )
    return int(sum(ml[name].size * 4 for name in arrays))


def stats(samples):
    return {
        'samples_s': samples,
        'median_s': float(np.median(samples)),
        'min_s': float(np.min(samples)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case', required=True,
        choices=('compact', 'slab', 'large-slab', 'two-patches'),
    )
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    arrays = DENSE_GATE.make_case(args.case)
    hybrid_pa = DENSE_GATE.make_pa('hybrid', arrays)
    dense_pa = DENSE_GATE.make_pa('dense', arrays)
    common = dict(
        dim=3, radius_scale=DENSE_GATE.RADIUS_SCALE,
        h_ref=DENSE_GATE.H_REF, level_ratio=DENSE_GATE.LEVEL_RATIO,
        nlevels=DENSE_GATE.NLEVELS,
    )
    hybrid = MultilevelGridWarpNNPS(
        particles=[hybrid_pa], sparse_cell_ratio=4.0, **common
    )
    dense = MultilevelGridWarpNNPS(
        particles=[dense_pa], sparse_cell_ratio=1.0e30, **common
    )
    hybrid_build = timed_build(hybrid, args.repeats)
    dense_build = timed_build(dense, args.repeats)
    hybrid_fused, hybrid_hash, hybrid_sum = DENSE_GATE.timed_fused(
        hybrid, 'multilevel', args.repeats
    )
    dense_fused, dense_hash, dense_sum = DENSE_GATE.timed_fused(
        dense, 'multilevel', args.repeats
    )
    output_names = ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')
    hybrid_outputs = np.column_stack([
        hybrid_pa.gpu.get_device_array(name).get() for name in output_names
    ])
    dense_outputs = np.column_stack([
        dense_pa.gpu.get_device_array(name).get() for name in output_names
    ])
    output_abs = np.abs(hybrid_outputs - dense_outputs)
    output_scale = np.maximum(np.abs(dense_outputs), np.float32(1.0e-6))
    reference_peak = float(np.max(np.abs(dense_outputs)))
    reference_l2 = float(np.linalg.norm(dense_outputs.astype(np.float64)))
    hybrid_ml = hybrid._build_multilevel(0)
    dense_ml = dense._build_multilevel(0)
    cache_h = hybrid.build_neighbor_cache_gpu(0, 0)
    cache_d = dense.build_neighbor_cache_gpu(0, 0)
    diag = hybrid.storage_diagnostics(0)
    device = wp.get_device(hybrid.device)
    result = {
        'case': args.case,
        'precision': 'fp32',
        'particles': int(arrays[0].size),
        'threshold_logical_per_occupied': 4.0,
        'storage_mode': diag['storage_mode'].tolist(),
        'level_cells': hybrid_ml['level_cells_host'].tolist(),
        'occupied_cells_by_level': hybrid_ml['occupied_host'].tolist(),
        'dense_cells_retained': diag['hybrid_dense_cells'],
        'sparse_cells_retained': diag['hybrid_sparse_cells'],
        'hybrid_actual_persistent_bytes': actual_storage_bytes(hybrid_ml),
        'forced_dense_actual_persistent_bytes': actual_storage_bytes(dense_ml),
        'saved_wcsph_state_bytes': int(28 * arrays[0].size),
        'build_hybrid': stats(hybrid_build),
        'build_forced_dense': stats(dense_build),
        'fused_hybrid': stats(hybrid_fused),
        'fused_forced_dense': stats(dense_fused),
        'build_plus_fused_hybrid_median_s': float(
            np.median(hybrid_build) + np.median(hybrid_fused)
        ),
        'build_plus_fused_forced_dense_median_s': float(
            np.median(dense_build) + np.median(dense_fused)
        ),
        'accepted_pairs_hybrid': cache_h['total_neighbors'],
        'accepted_pairs_forced_dense': cache_d['total_neighbors'],
        'fused_hash_hybrid': hybrid_hash,
        'fused_hash_forced_dense': dense_hash,
        'fused_sum_hybrid': hybrid_sum,
        'fused_sum_forced_dense': dense_sum,
        'fused_max_abs_delta': float(np.max(output_abs)),
        'fused_max_rel_delta': float(np.max(output_abs / output_scale)),
        'fused_max_abs_over_reference_peak': (
            float(np.max(output_abs)) / max(reference_peak, 1.0e-30)
        ),
        'fused_relative_l2_delta': (
            float(np.linalg.norm(output_abs.astype(np.float64)))
            / max(reference_l2, 1.0e-30)
        ),
        'device': device.name,
        'device_arch': device.arch,
        'warp_version': wp.__version__,
        'python': platform.python_version(),
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
