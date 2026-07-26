"""Multilevel (adaptive-resolution) GPU NNPS for the Warp backend.

Kept in a SEPARATE module from ``warp_nnps`` on purpose: Warp compiles and
loads an entire Python module's kernels together on the first launch of any of
them. If these multilevel kernels lived in ``warp_nnps`` they would be JIT-ed
onto the device by every consumer of ``UniformGridWarpNNPS`` (e.g. the WCSPH
suite), inflating the process's device module footprint. On WSL2 that extra PTX
tips the in-process PTX-JIT compiler into a hang while loading a later large
generated kernel. Housing them here means they load lazily only when
``MultilevelGridWarpNNPS`` is actually used.
"""

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from pysph.base.warp_nnps import (
    UniformGridWarpNNPS, _copy_i32,
)


if wp is not None:
    # --- Multilevel (adaptive-resolution) cell-list kernels ---------------
    #
    # Each source particle lives in exactly one level. Compact levels use dense
    # flattened grids; sparse levels use sorted logical cell keys. Traversal
    # loops over levels and, per level, converts the query radius
    # max(radius_scale*h_i, support[k]) into a variable cell-index range (not a
    # fixed +/-1 stencil) with a +/-1 guard band, then applies the exact
    # symmetric cutoff. The lengths and fill kernels are structurally identical
    # so their counts can never diverge (unlike the separate brute/grid passes).

    @wp.kernel
    def _ml_assign_reduce_f64(
            h: wp.array(dtype=wp.float64),
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            edges: wp.array(dtype=wp.float64),
            nlevels: wp.int32,
            dim: wp.int32,
            level_of: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
            hmax: wp.array(dtype=wp.float64),
            xmin: wp.array(dtype=wp.float64),
            xmax: wp.array(dtype=wp.float64),
            ymin: wp.array(dtype=wp.float64),
            ymax: wp.array(dtype=wp.float64),
            zmin: wp.array(dtype=wp.float64),
            zmax: wp.array(dtype=wp.float64),
            oob: wp.array(dtype=wp.int32),
    ):
        # Assign each particle to its half-open level and reduce per-level
        # count, max-h and AABB on the device, so x/y/z/h never leave the GPU.
        i = wp.tid()
        hi = h[i]
        k = wp.int32(-1)
        for m in range(nlevels):
            if hi >= edges[m] and hi < edges[m + 1]:
                k = m
        if hi == edges[nlevels]:          # inclusive top edge -> top level
            k = nlevels - wp.int32(1)
        if k < wp.int32(0):               # below finest or above top -> loud
            wp.atomic_add(oob, 0, wp.int32(1))
            level_of[i] = wp.int32(0)
            return
        level_of[i] = k
        wp.atomic_add(counts, k, wp.int32(1))
        wp.atomic_max(hmax, k, hi)
        wp.atomic_min(xmin, k, x[i])
        wp.atomic_max(xmax, k, x[i])
        if dim > 1:
            wp.atomic_min(ymin, k, y[i])
            wp.atomic_max(ymax, k, y[i])
        if dim > 2:
            wp.atomic_min(zmin, k, z[i])
            wp.atomic_max(zmax, k, z[i])


    @wp.kernel
    def _ml_assign_reduce_f32(
            h: wp.array(dtype=wp.float32),
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            edges: wp.array(dtype=wp.float32),
            nlevels: wp.int32,
            dim: wp.int32,
            level_of: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
            hmax: wp.array(dtype=wp.float32),
            xmin: wp.array(dtype=wp.float32),
            xmax: wp.array(dtype=wp.float32),
            ymin: wp.array(dtype=wp.float32),
            ymax: wp.array(dtype=wp.float32),
            zmin: wp.array(dtype=wp.float32),
            zmax: wp.array(dtype=wp.float32),
            oob: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        hi = h[i]
        k = wp.int32(-1)
        for m in range(nlevels):
            if hi >= edges[m] and hi < edges[m + 1]:
                k = m
        if hi == edges[nlevels]:
            k = nlevels - wp.int32(1)
        if k < wp.int32(0):
            wp.atomic_add(oob, 0, wp.int32(1))
            level_of[i] = wp.int32(0)
            return
        level_of[i] = k
        wp.atomic_add(counts, k, wp.int32(1))
        wp.atomic_max(hmax, k, hi)
        wp.atomic_min(xmin, k, x[i])
        wp.atomic_max(xmax, k, x[i])
        if dim > 1:
            wp.atomic_min(ymin, k, y[i])
            wp.atomic_max(ymax, k, y[i])
        if dim > 2:
            wp.atomic_min(zmin, k, z[i])
            wp.atomic_max(zmax, k, z[i])


    @wp.kernel
    def _ml_virtual_cell_keys_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            level_of: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float64),
            origin_y: wp.array(dtype=wp.float64),
            origin_z: wp.array(dtype=wp.float64),
            cell_size: wp.array(dtype=wp.float64),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            dim: wp.int32,
            keys: wp.array(dtype=wp.int32),
            particles: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        k = level_of[i]
        cs = cell_size[k]
        ix = wp.int32(wp.floor((x[i] - origin_x[k]) / cs))
        iy = wp.int32(0)
        iz = wp.int32(0)
        if dim > 1:
            iy = wp.int32(wp.floor((y[i] - origin_y[k]) / cs))
        if dim > 2:
            iz = wp.int32(wp.floor((z[i] - origin_z[k]) / cs))
        ix = wp.clamp(ix, wp.int32(0), nx[k] - wp.int32(1))
        iy = wp.clamp(iy, wp.int32(0), ny[k] - wp.int32(1))
        iz = wp.clamp(iz, wp.int32(0), nz[k] - wp.int32(1))
        keys[i] = (virtual_offset[k] + ix + iy * nx[k]
                   + iz * nx[k] * ny[k])
        particles[i] = i


    @wp.kernel
    def _ml_virtual_cell_keys_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            level_of: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float32),
            origin_y: wp.array(dtype=wp.float32),
            origin_z: wp.array(dtype=wp.float32),
            cell_size: wp.array(dtype=wp.float32),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            dim: wp.int32,
            keys: wp.array(dtype=wp.int32),
            particles: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        k = level_of[i]
        cs = cell_size[k]
        ix = wp.int32(wp.floor((x[i] - origin_x[k]) / cs))
        iy = wp.int32(0)
        iz = wp.int32(0)
        if dim > 1:
            iy = wp.int32(wp.floor((y[i] - origin_y[k]) / cs))
        if dim > 2:
            iz = wp.int32(wp.floor((z[i] - origin_z[k]) / cs))
        ix = wp.clamp(ix, wp.int32(0), nx[k] - wp.int32(1))
        iy = wp.clamp(iy, wp.int32(0), ny[k] - wp.int32(1))
        iz = wp.clamp(iz, wp.int32(0), nz[k] - wp.int32(1))
        keys[i] = (virtual_offset[k] + ix + iy * nx[k]
                   + iz * nx[k] * ny[k])
        particles[i] = i


    @wp.kernel
    def _ml_count_occupied_levels(
            unique_keys: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            level_cells: wp.array(dtype=wp.int32),
            nlevels: wp.int32,
            occupied: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        key = unique_keys[i]
        for k in range(nlevels):
            if (key >= virtual_offset[k]
                    and key < virtual_offset[k] + level_cells[k]):
                wp.atomic_add(occupied, k, wp.int32(1))


    @wp.kernel
    def _ml_mark_sparse_pairs(
            keys: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            level_cells: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            nlevels: wp.int32,
            flags: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        key = keys[i]
        keep = wp.int32(0)
        for k in range(nlevels):
            if (key >= virtual_offset[k]
                    and key < virtual_offset[k] + level_cells[k]):
                keep = storage_mode[k]
        flags[i] = keep


    @wp.kernel
    def _ml_compact_sparse_pairs(
            keys: wp.array(dtype=wp.int32),
            particles: wp.array(dtype=wp.int32),
            flags: wp.array(dtype=wp.int32),
            positions: wp.array(dtype=wp.int32),
            sparse_keys: wp.array(dtype=wp.int32),
            sparse_particles: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        if flags[i] != wp.int32(0):
            out = positions[i]
            sparse_keys[out] = keys[i]
            sparse_particles[out] = wp.uint32(particles[i])


    @wp.kernel
    def _ml_dense_cell_ids_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            level_of: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float64),
            origin_y: wp.array(dtype=wp.float64),
            origin_z: wp.array(dtype=wp.float64),
            cell_size: wp.array(dtype=wp.float64),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            dim: wp.int32,
            cell_ids: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        k = level_of[i]
        if storage_mode[k] == wp.int32(0):
            cs = cell_size[k]
            ix = wp.int32(wp.floor((x[i] - origin_x[k]) / cs))
            iy = wp.int32(0)
            iz = wp.int32(0)
            if dim > 1:
                iy = wp.int32(wp.floor((y[i] - origin_y[k]) / cs))
            if dim > 2:
                iz = wp.int32(wp.floor((z[i] - origin_z[k]) / cs))
            ix = wp.clamp(ix, wp.int32(0), nx[k] - wp.int32(1))
            iy = wp.clamp(iy, wp.int32(0), ny[k] - wp.int32(1))
            iz = wp.clamp(iz, wp.int32(0), nz[k] - wp.int32(1))
            cid = (dense_offset[k] + ix + iy * nx[k]
                   + iz * nx[k] * ny[k])
            cell_ids[i] = cid
            wp.atomic_add(counts, cid, wp.int32(1))
        else:
            cell_ids[i] = wp.int32(-1)


    @wp.kernel
    def _ml_dense_cell_ids_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            level_of: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float32),
            origin_y: wp.array(dtype=wp.float32),
            origin_z: wp.array(dtype=wp.float32),
            cell_size: wp.array(dtype=wp.float32),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            dim: wp.int32,
            cell_ids: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        k = level_of[i]
        if storage_mode[k] == wp.int32(0):
            cs = cell_size[k]
            ix = wp.int32(wp.floor((x[i] - origin_x[k]) / cs))
            iy = wp.int32(0)
            iz = wp.int32(0)
            if dim > 1:
                iy = wp.int32(wp.floor((y[i] - origin_y[k]) / cs))
            if dim > 2:
                iz = wp.int32(wp.floor((z[i] - origin_z[k]) / cs))
            ix = wp.clamp(ix, wp.int32(0), nx[k] - wp.int32(1))
            iy = wp.clamp(iy, wp.int32(0), ny[k] - wp.int32(1))
            iz = wp.clamp(iz, wp.int32(0), nz[k] - wp.int32(1))
            cid = (dense_offset[k] + ix + iy * nx[k]
                   + iz * nx[k] * ny[k])
            cell_ids[i] = cid
            wp.atomic_add(counts, cid, wp.int32(1))
        else:
            cell_ids[i] = wp.int32(-1)


    @wp.kernel
    def _ml_scatter_dense_particles(
            cell_ids: wp.array(dtype=wp.int32),
            cursor: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        cid = cell_ids[i]
        if cid >= wp.int32(0):
            pos = wp.atomic_add(cursor, cid, wp.int32(1))
            cell_particles[pos] = wp.uint32(i)


    @wp.func
    def _ml_cell_range(
            dq: wp.float64, qr: wp.float64, origin: wp.float64,
            cs: wp.float64, n: wp.int32):
        lo = wp.int32(wp.floor((dq - qr - origin) / cs)) - wp.int32(1)
        hi = wp.int32(wp.floor((dq + qr - origin) / cs)) + wp.int32(1)
        lo = wp.clamp(lo, wp.int32(0), n - wp.int32(1))
        hi = wp.clamp(hi, wp.int32(0), n - wp.int32(1))
        return wp.vec2i(lo, hi)


    @wp.func
    def _ml_cell_range_f32(
            dq: wp.float32, qr: wp.float32, origin: wp.float32,
            cs: wp.float32, n: wp.int32):
        lo = wp.int32(wp.floor((dq - qr - origin) / cs)) - wp.int32(1)
        hi = wp.int32(wp.floor((dq + qr - origin) / cs)) + wp.int32(1)
        lo = wp.clamp(lo, wp.int32(0), n - wp.int32(1))
        hi = wp.clamp(hi, wp.int32(0), n - wp.int32(1))
        return wp.vec2i(lo, hi)


    @wp.kernel
    def _multilevel_neighbor_lengths_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            origin_x: wp.array(dtype=wp.float64),
            origin_y: wp.array(dtype=wp.float64),
            origin_z: wp.array(dtype=wp.float64),
            cell_size: wp.array(dtype=wp.float64),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            sparse_keys: wp.array(dtype=wp.int32),
            sparse_starts: wp.array(dtype=wp.int32),
            sparse_counts: wp.array(dtype=wp.int32),
            sparse_particles: wp.array(dtype=wp.uint32),
            sparse_cells: wp.int32,
            support: wp.array(dtype=wp.float64),
            nlevels: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        count = wp.int32(0)
        for k in range(nlevels):
            nxk = nx[k]
            if nxk > 0:
                cs = cell_size[k]
                qr = radius_scale * d_h[i]
                if support[k] > qr:
                    qr = support[k]
                rx = _ml_cell_range(d_x[i], qr, origin_x[k], cs, nxk)
                iylo = wp.int32(0)
                iyhi = wp.int32(0)
                nyk = ny[k]
                if dim > 1:
                    ry = _ml_cell_range(
                        d_y[i], qr, origin_y[k], cs, nyk)
                    iylo = ry[0]
                    iyhi = ry[1]
                izlo = wp.int32(0)
                izhi = wp.int32(0)
                nzk = nz[k]
                if dim > 2:
                    rz = _ml_cell_range(
                        d_z[i], qr, origin_z[k], cs, nzk)
                    izlo = rz[0]
                    izhi = rz[1]
                dense_off = dense_offset[k]
                virtual_off = virtual_offset[k]
                for iz in range(izlo, izhi + 1):
                    for iy in range(iylo, iyhi + 1):
                        for ix in range(rx[0], rx[1] + 1):
                            local = ix + iy * nxk + iz * nxk * nyk
                            start = wp.int32(0)
                            stop = wp.int32(0)
                            sparse = storage_mode[k] != wp.int32(0)
                            if sparse:
                                key = virtual_off + local
                                slot = wp.lower_bound(
                                    sparse_keys, wp.int32(0), sparse_cells,
                                    key)
                                if (slot < sparse_cells
                                        and sparse_keys[slot] == key):
                                    start = sparse_starts[slot]
                                    stop = start + sparse_counts[slot]
                            else:
                                cid = dense_off + local
                                start = cell_starts[cid]
                                stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = wp.int32(0)
                                if sparse:
                                    j = wp.int32(sparse_particles[pos])
                                else:
                                    j = wp.int32(cell_particles[pos])
                                dx = d_x[i] - s_x[j]
                                dy = wp.float64(0.0)
                                dz = wp.float64(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx * dx + dy * dy + dz * dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi * hi or dist2 < hj * hj:
                                    count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _multilevel_neighbor_fill_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            out_starts: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float64),
            origin_y: wp.array(dtype=wp.float64),
            origin_z: wp.array(dtype=wp.float64),
            cell_size: wp.array(dtype=wp.float64),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            sparse_keys: wp.array(dtype=wp.int32),
            sparse_starts: wp.array(dtype=wp.int32),
            sparse_counts: wp.array(dtype=wp.int32),
            sparse_particles: wp.array(dtype=wp.uint32),
            sparse_cells: wp.int32,
            support: wp.array(dtype=wp.float64),
            nlevels: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        out = out_starts[i]
        count = wp.int32(0)
        for k in range(nlevels):
            nxk = nx[k]
            if nxk > 0:
                cs = cell_size[k]
                qr = radius_scale * d_h[i]
                if support[k] > qr:
                    qr = support[k]
                rx = _ml_cell_range(d_x[i], qr, origin_x[k], cs, nxk)
                iylo = wp.int32(0)
                iyhi = wp.int32(0)
                nyk = ny[k]
                if dim > 1:
                    ry = _ml_cell_range(
                        d_y[i], qr, origin_y[k], cs, nyk)
                    iylo = ry[0]
                    iyhi = ry[1]
                izlo = wp.int32(0)
                izhi = wp.int32(0)
                nzk = nz[k]
                if dim > 2:
                    rz = _ml_cell_range(
                        d_z[i], qr, origin_z[k], cs, nzk)
                    izlo = rz[0]
                    izhi = rz[1]
                dense_off = dense_offset[k]
                virtual_off = virtual_offset[k]
                for iz in range(izlo, izhi + 1):
                    for iy in range(iylo, iyhi + 1):
                        for ix in range(rx[0], rx[1] + 1):
                            local = ix + iy * nxk + iz * nxk * nyk
                            start = wp.int32(0)
                            stop = wp.int32(0)
                            sparse = storage_mode[k] != wp.int32(0)
                            if sparse:
                                key = virtual_off + local
                                slot = wp.lower_bound(
                                    sparse_keys, wp.int32(0), sparse_cells,
                                    key)
                                if (slot < sparse_cells
                                        and sparse_keys[slot] == key):
                                    start = sparse_starts[slot]
                                    stop = start + sparse_counts[slot]
                            else:
                                cid = dense_off + local
                                start = cell_starts[cid]
                                stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = wp.int32(0)
                                if sparse:
                                    j = wp.int32(sparse_particles[pos])
                                else:
                                    j = wp.int32(cell_particles[pos])
                                dx = d_x[i] - s_x[j]
                                dy = wp.float64(0.0)
                                dz = wp.float64(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx * dx + dy * dy + dz * dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi * hi or dist2 < hj * hj:
                                    neighbors[out + count] = wp.uint32(j)
                                    count += wp.int32(1)


    @wp.kernel
    def _multilevel_neighbor_lengths_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            origin_x: wp.array(dtype=wp.float32),
            origin_y: wp.array(dtype=wp.float32),
            origin_z: wp.array(dtype=wp.float32),
            cell_size: wp.array(dtype=wp.float32),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            sparse_keys: wp.array(dtype=wp.int32),
            sparse_starts: wp.array(dtype=wp.int32),
            sparse_counts: wp.array(dtype=wp.int32),
            sparse_particles: wp.array(dtype=wp.uint32),
            sparse_cells: wp.int32,
            support: wp.array(dtype=wp.float32),
            nlevels: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        count = wp.int32(0)
        for k in range(nlevels):
            nxk = nx[k]
            if nxk > 0:
                cs = cell_size[k]
                qr = radius_scale * d_h[i]
                if support[k] > qr:
                    qr = support[k]
                rx = _ml_cell_range_f32(
                    d_x[i], qr, origin_x[k], cs, nxk)
                iylo = wp.int32(0)
                iyhi = wp.int32(0)
                nyk = ny[k]
                if dim > 1:
                    ry = _ml_cell_range_f32(
                        d_y[i], qr, origin_y[k], cs, nyk)
                    iylo = ry[0]
                    iyhi = ry[1]
                izlo = wp.int32(0)
                izhi = wp.int32(0)
                nzk = nz[k]
                if dim > 2:
                    rz = _ml_cell_range_f32(
                        d_z[i], qr, origin_z[k], cs, nzk)
                    izlo = rz[0]
                    izhi = rz[1]
                dense_off = dense_offset[k]
                virtual_off = virtual_offset[k]
                for iz in range(izlo, izhi + 1):
                    for iy in range(iylo, iyhi + 1):
                        for ix in range(rx[0], rx[1] + 1):
                            local = ix + iy * nxk + iz * nxk * nyk
                            start = wp.int32(0)
                            stop = wp.int32(0)
                            sparse = storage_mode[k] != wp.int32(0)
                            if sparse:
                                key = virtual_off + local
                                slot = wp.lower_bound(
                                    sparse_keys, wp.int32(0), sparse_cells,
                                    key)
                                if (slot < sparse_cells
                                        and sparse_keys[slot] == key):
                                    start = sparse_starts[slot]
                                    stop = start + sparse_counts[slot]
                            else:
                                cid = dense_off + local
                                start = cell_starts[cid]
                                stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = wp.int32(0)
                                if sparse:
                                    j = wp.int32(sparse_particles[pos])
                                else:
                                    j = wp.int32(cell_particles[pos])
                                dx = d_x[i] - s_x[j]
                                dy = wp.float32(0.0)
                                dz = wp.float32(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx * dx + dy * dy + dz * dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi * hi or dist2 < hj * hj:
                                    count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _multilevel_neighbor_fill_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            out_starts: wp.array(dtype=wp.int32),
            origin_x: wp.array(dtype=wp.float32),
            origin_y: wp.array(dtype=wp.float32),
            origin_z: wp.array(dtype=wp.float32),
            cell_size: wp.array(dtype=wp.float32),
            nx: wp.array(dtype=wp.int32),
            ny: wp.array(dtype=wp.int32),
            nz: wp.array(dtype=wp.int32),
            dense_offset: wp.array(dtype=wp.int32),
            storage_mode: wp.array(dtype=wp.int32),
            virtual_offset: wp.array(dtype=wp.int32),
            sparse_keys: wp.array(dtype=wp.int32),
            sparse_starts: wp.array(dtype=wp.int32),
            sparse_counts: wp.array(dtype=wp.int32),
            sparse_particles: wp.array(dtype=wp.uint32),
            sparse_cells: wp.int32,
            support: wp.array(dtype=wp.float32),
            nlevels: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        out = out_starts[i]
        count = wp.int32(0)
        for k in range(nlevels):
            nxk = nx[k]
            if nxk > 0:
                cs = cell_size[k]
                qr = radius_scale * d_h[i]
                if support[k] > qr:
                    qr = support[k]
                rx = _ml_cell_range_f32(
                    d_x[i], qr, origin_x[k], cs, nxk)
                iylo = wp.int32(0)
                iyhi = wp.int32(0)
                nyk = ny[k]
                if dim > 1:
                    ry = _ml_cell_range_f32(
                        d_y[i], qr, origin_y[k], cs, nyk)
                    iylo = ry[0]
                    iyhi = ry[1]
                izlo = wp.int32(0)
                izhi = wp.int32(0)
                nzk = nz[k]
                if dim > 2:
                    rz = _ml_cell_range_f32(
                        d_z[i], qr, origin_z[k], cs, nzk)
                    izlo = rz[0]
                    izhi = rz[1]
                dense_off = dense_offset[k]
                virtual_off = virtual_offset[k]
                for iz in range(izlo, izhi + 1):
                    for iy in range(iylo, iyhi + 1):
                        for ix in range(rx[0], rx[1] + 1):
                            local = ix + iy * nxk + iz * nxk * nyk
                            start = wp.int32(0)
                            stop = wp.int32(0)
                            sparse = storage_mode[k] != wp.int32(0)
                            if sparse:
                                key = virtual_off + local
                                slot = wp.lower_bound(
                                    sparse_keys, wp.int32(0), sparse_cells,
                                    key)
                                if (slot < sparse_cells
                                        and sparse_keys[slot] == key):
                                    start = sparse_starts[slot]
                                    stop = start + sparse_counts[slot]
                            else:
                                cid = dense_off + local
                                start = cell_starts[cid]
                                stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = wp.int32(0)
                                if sparse:
                                    j = wp.int32(sparse_particles[pos])
                                else:
                                    j = wp.int32(cell_particles[pos])
                                dx = d_x[i] - s_x[j]
                                dy = wp.float32(0.0)
                                dz = wp.float32(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx * dx + dy * dy + dz * dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi * hi or dist2 < hj * hj:
                                    neighbors[out + count] = wp.uint32(j)
                                    count += wp.int32(1)


class MultilevelGridWarpNNPS(UniformGridWarpNNPS):
    """Exact device-built multilevel cell-list NNPS for adaptive resolution.

    Sources are binned into discrete smoothing-length levels (see
    ``assign_particle_levels``). Each populated level gets its own padded
    origin, cell size (its conservative support bound) and dimensions. Compact
    levels use flattened dense grids while sparse levels use sorted cell keys.
    Neighbor traversal loops over levels, converting each level's query radius
    ``max(radius_scale*h_i, support[k])`` into a variable cell-index range (not
    a fixed 3x3x3 stencil), then applies the exact symmetric cutoff. The
    accepted set therefore matches brute force while candidate work drops for
    localized refinement, because a small number of coarse particles no longer
    forces coarse cells over dense fine regions.

    Level assignment plus per-level count, max-h, and AABB reductions run on
    the GPU.  Only O(nlevels) scalar metadata is read back to size the dense
    grids; per-particle coordinates and smoothing lengths remain on device.
    """

    def __init__(self, dim, particles, radius_scale=2.0, h_ref=None,
                 level_ratio=2.0, nlevels=1, ghost_layers=1, domain=None,
                 cache=True, sort_gids=False, backend='warp', device=None,
                 sparse_cell_ratio=4.0):
        if h_ref is None:
            raise ValueError("MultilevelGridWarpNNPS requires an h_ref")
        if nlevels < 1:
            raise ValueError("nlevels must be >= 1; got %r" % (nlevels,))
        if h_ref <= 0.0:
            raise ValueError("h_ref must be > 0; got %r" % (h_ref,))
        if level_ratio <= 1.0:
            raise ValueError("level_ratio must be > 1 so level edges are "
                             "strictly ascending; got %r" % (level_ratio,))
        self.h_ref = h_ref
        self.level_ratio = level_ratio
        self.nlevels = nlevels
        if sparse_cell_ratio < 1.0:
            raise ValueError("sparse_cell_ratio must be >= 1; got %r"
                             % (sparse_cell_ratio,))
        self.sparse_cell_ratio = float(sparse_cell_ratio)
        self._ml = {}
        super(MultilevelGridWarpNNPS, self).__init__(
            dim=dim, particles=particles, radius_scale=radius_scale,
            ghost_layers=ghost_layers, domain=domain, cache=cache,
            sort_gids=sort_gids, backend=backend, device=device
        )

    def update(self, push=True):
        if push:
            for pa in self.particles:
                pa.gpu.push('x', 'y', 'z', 'h')
        self._flags.clear()
        self._cache.clear()
        self._ml.clear()

    def _ml_kernels_for(self, gpu):
        if gpu.x.dtype == np.float32:
            return (
                _multilevel_neighbor_lengths_f32,
                _multilevel_neighbor_fill_f32,
                np.float32(self.radius_scale), wp.float32, np.float32,
                _ml_assign_reduce_f32,
            )
        return (
            _multilevel_neighbor_lengths_f64,
            _multilevel_neighbor_fill_f64,
            np.float64(self.radius_scale), wp.float64, np.float64,
            _ml_assign_reduce_f64,
        )

    def _build_multilevel(self, src_index):
        ml = self._ml.get(src_index)
        if ml is not None:
            return ml

        gpu = self.particles[src_index].gpu
        nsrc = gpu.get_number_of_particles()
        dim = self.dim
        nlevels = self.nlevels
        dev = self.device
        _, _, _, wpf, npf, assign_k = self._ml_kernels_for(gpu)

        # Level edges in the device float precision; an fp32 h sitting exactly
        # on an edge then bins like the edge instead of tripping the guard.
        edges_host = (self.h_ref
                      * self.level_ratio ** np.arange(nlevels + 1)).astype(npf)

        # GPU level assignment + per-level reductions. Only O(nlevels) scalar
        # metadata is read back below -- x/y/z/h never leave the device.
        level_of = wp.zeros(nsrc if nsrc > 0 else 1, dtype=wp.int32, device=dev)
        counts_l = wp.zeros(nlevels, dtype=wp.int32, device=dev)
        hmax_l = wp.zeros(nlevels, dtype=wpf, device=dev)
        xmin_l = wp.array(np.full(nlevels, np.inf, npf), dtype=wpf, device=dev)
        xmax_l = wp.array(np.full(nlevels, -np.inf, npf), dtype=wpf, device=dev)
        ymin_l = wp.array(np.full(nlevels, np.inf, npf), dtype=wpf, device=dev)
        ymax_l = wp.array(np.full(nlevels, -np.inf, npf), dtype=wpf, device=dev)
        zmin_l = wp.array(np.full(nlevels, np.inf, npf), dtype=wpf, device=dev)
        zmax_l = wp.array(np.full(nlevels, -np.inf, npf), dtype=wpf, device=dev)
        oob = wp.zeros(1, dtype=wp.int32, device=dev)
        edges_dev = wp.array(edges_host, dtype=wpf, device=dev)
        if nsrc > 0:
            wp.launch(
                assign_k, dim=nsrc,
                inputs=[
                    gpu.h.dev, gpu.x.dev, gpu.y.dev, gpu.z.dev, edges_dev,
                    np.int32(nlevels), np.int32(dim), level_of, counts_l,
                    hmax_l, xmin_l, xmax_l, ymin_l, ymax_l, zmin_l, zmax_l,
                    oob,
                ],
                device=dev,
            )
            wp.synchronize_device(dev)

        # O(nlevels) metadata readback (permitted; not the coordinate arrays).
        counts = counts_l.numpy()
        hmax = hmax_l.numpy()
        xmn, xmx = xmin_l.numpy(), xmax_l.numpy()
        ymn, ymx = ymin_l.numpy(), ymax_l.numpy()
        zmn, zmx = zmin_l.numpy(), zmax_l.numpy()
        if nsrc > 0 and int(oob.numpy()[0]) > 0:
            raise ValueError(
                "smoothing length outside configured level range "
                "[%g, %g]; particles are not silently clipped"
                % (float(edges_host[0]), float(edges_host[-1]))
            )

        support = np.zeros(nlevels, dtype=np.float64)
        ox = np.zeros(nlevels, dtype=np.float64)
        oy = np.zeros(nlevels, dtype=np.float64)
        oz = np.zeros(nlevels, dtype=np.float64)
        cell_size = np.ones(nlevels, dtype=np.float64)
        nx = np.zeros(nlevels, dtype=np.int32)   # 0 => empty level, no cells
        ny = np.ones(nlevels, dtype=np.int32)
        nz = np.ones(nlevels, dtype=np.int32)
        for k in range(nlevels):
            if counts[k] <= 0:
                continue
            cs = float(self.radius_scale * hmax[k])
            support[k] = cs
            cell_size[k] = cs
            ox[k] = float(xmn[k]) - cs
            nx[k] = max(1, int(np.ceil((float(xmx[k]) + cs - ox[k]) / cs)))
            if dim > 1:
                oy[k] = float(ymn[k]) - cs
                ny[k] = max(1, int(np.ceil((float(ymx[k]) + cs - oy[k]) / cs)))
            if dim > 2:
                oz[k] = float(zmn[k]) - cs
                nz[k] = max(1, int(np.ceil((float(zmx[k]) + cs - oz[k]) / cs)))

        sizes = nx.astype(np.int64) * ny.astype(np.int64) * nz.astype(np.int64)
        if int(sizes.sum()) > np.iinfo(np.int32).max:
            raise ValueError(
                "multilevel logical cell keys exceed the int32 prototype "
                "limit; split the domain or adopt 64-bit run-length keys"
            )
        cell_offset = np.zeros(nlevels, dtype=np.int32)
        if nlevels > 1:
            cell_offset[1:] = np.cumsum(sizes)[:-1].astype(np.int32)
        total_cells = int(sizes.sum())

        ml = {
            'level_of': level_of,   # device, per-particle levels (GPU-computed)
            'origin_x': wp.array(ox.astype(npf), dtype=wpf, device=dev),
            'origin_y': wp.array(oy.astype(npf), dtype=wpf, device=dev),
            'origin_z': wp.array(oz.astype(npf), dtype=wpf, device=dev),
            'cell_size': wp.array(cell_size.astype(npf), dtype=wpf, device=dev),
            'nx': wp.array(nx, dtype=wp.int32, device=dev),
            'ny': wp.array(ny, dtype=wp.int32, device=dev),
            'nz': wp.array(nz, dtype=wp.int32, device=dev),
            'support': wp.array(support.astype(npf), dtype=wpf, device=dev),
            'total_cells': total_cells,
            'nsrc': nsrc,
            'support_host': support,
            # Host copies in the SAME precision the GPU sees (npf), for the
            # device-residency-free diagnostics used by boundary tests.
            'ox_host': ox.astype(npf), 'oy_host': oy.astype(npf),
            'oz_host': oz.astype(npf), 'cs_host': cell_size.astype(npf),
            'nx_host': nx, 'ny_host': ny, 'nz_host': nz,
        }

        # Build the sorted sparse-cell oracle for every occupied level. It
        # supplies exact occupied-cell counts for the hybrid decision without
        # reading particle keys or coordinates back to the host.
        self._build_sparse_oracle(
            ml, gpu, counts, sizes.astype(np.int32), cell_offset
        )

        self._build_hybrid_storage(ml, gpu, counts, sizes.astype(np.int32))

        self._ml[src_index] = ml
        return ml

    def _build_sparse_oracle(self, ml, gpu, level_counts, level_cells,
                             virtual_offset):
        """Build sorted occupied-cell keys and choose storage per level.

        Device radix-sort plus run-length encoding forms the oracle. Only the
        per-level occupied counts are read back; particle keys and indices stay
        on the device. ``storage_mode`` is 0 for dense and 1 for sparse.
        """
        nsrc = ml['nsrc']
        nlevels = self.nlevels
        dev = self.device
        nalloc = max(1, nsrc)
        keys = wp.zeros(max(2, 2 * nsrc), dtype=wp.int32, device=dev)
        particles = wp.zeros(max(2, 2 * nsrc), dtype=wp.int32, device=dev)
        virtual_offset_dev = wp.array(
            virtual_offset, dtype=wp.int32, device=dev
        )
        level_cells_dev = wp.array(level_cells, dtype=wp.int32, device=dev)
        if nsrc > 0:
            key_kernel = (_ml_virtual_cell_keys_f32
                          if gpu.x.dtype == np.float32
                          else _ml_virtual_cell_keys_f64)
            wp.launch(
                key_kernel, dim=nsrc,
                inputs=[
                    gpu.x.dev, gpu.y.dev, gpu.z.dev, ml['level_of'],
                    ml['origin_x'], ml['origin_y'], ml['origin_z'],
                    ml['cell_size'], ml['nx'], ml['ny'], ml['nz'],
                    virtual_offset_dev, np.int32(self.dim), keys, particles,
                ],
                device=dev,
            )
            wp.utils.radix_sort_pairs(keys, particles, nsrc)

        unique_keys = wp.zeros(nalloc, dtype=wp.int32, device=dev)
        unique_counts = wp.zeros(nalloc, dtype=wp.int32, device=dev)
        occupied_total = (
            wp.utils.runlength_encode(
                keys[:nsrc], unique_keys, unique_counts
            ) if nsrc > 0 else 0
        )
        occupied_l = wp.zeros(nlevels, dtype=wp.int32, device=dev)
        if occupied_total > 0:
            wp.launch(
                _ml_count_occupied_levels, dim=occupied_total,
                inputs=[
                    unique_keys, virtual_offset_dev, level_cells_dev,
                    np.int32(nlevels), occupied_l,
                ],
                device=dev,
            )
            wp.synchronize_device(dev)
        occupied = occupied_l.numpy()
        storage_mode = np.zeros(nlevels, dtype=np.int32)
        for k in range(nlevels):
            if occupied[k] > 0 and (
                    float(level_cells[k]) / float(occupied[k])
                    > self.sparse_cell_ratio):
                storage_mode[k] = 1

        ml.update({
            'virtual_offset': virtual_offset_dev,
            'virtual_offset_host': virtual_offset,
            'level_cells': level_cells_dev,
            'level_cells_host': level_cells,
            'occupied_host': occupied,
            'storage_mode': wp.array(
                storage_mode, dtype=wp.int32, device=dev
            ),
            'storage_mode_host': storage_mode,
            'sorted_keys': keys,
            'oracle_keys': unique_keys,
            'oracle_counts': unique_counts,
            'oracle_particles': particles,
        })

    def _build_hybrid_storage(self, ml, gpu, level_counts, level_cells):
        """Materialize dense compact grids and sparse sorted runs per level."""
        dev = self.device
        nsrc = ml['nsrc']
        nlevels = self.nlevels
        mode = ml['storage_mode_host']

        # Stable compaction of the already sorted all-particle key stream keeps
        # only particles belonging to sparse-selected levels.
        nsparse = int(level_counts[mode == 1].sum())
        sparse_flags = wp.zeros(max(1, nsrc), dtype=wp.int32, device=dev)
        sparse_pos = wp.zeros(max(1, nsrc), dtype=wp.int32, device=dev)
        sparse_pair_keys = wp.zeros(max(1, nsparse), dtype=wp.int32,
                                    device=dev)
        sparse_particles = wp.zeros(max(1, nsparse), dtype=wp.uint32,
                                    device=dev)
        # ``oracle_particles`` is the value array after radix sort; the sorted
        # keys remain in the first nsrc entries of a private array. Preserve it
        # explicitly in the oracle record to avoid confusing keys and runs.
        sorted_keys = ml['sorted_keys']
        sorted_particles = ml['oracle_particles']
        if nsrc > 0:
            wp.launch(
                _ml_mark_sparse_pairs, dim=nsrc,
                inputs=[
                    sorted_keys, ml['virtual_offset'], ml['level_cells'],
                    ml['storage_mode'], np.int32(nlevels), sparse_flags,
                ],
                device=dev,
            )
            wp.utils.array_scan(sparse_flags, sparse_pos, inclusive=False)
            if nsparse > 0:
                wp.launch(
                    _ml_compact_sparse_pairs, dim=nsrc,
                    inputs=[
                        sorted_keys, sorted_particles, sparse_flags, sparse_pos,
                        sparse_pair_keys, sparse_particles,
                    ],
                    device=dev,
                )

        sparse_keys = wp.zeros(max(1, nsparse), dtype=wp.int32, device=dev)
        sparse_counts = wp.zeros(max(1, nsparse), dtype=wp.int32, device=dev)
        sparse_cells = (
            wp.utils.runlength_encode(
                sparse_pair_keys[:nsparse], sparse_keys, sparse_counts
            ) if nsparse > 0 else 0
        )
        sparse_starts = wp.zeros(max(1, sparse_cells), dtype=wp.int32,
                                 device=dev)
        if sparse_cells > 0:
            wp.utils.array_scan(
                sparse_counts[:sparse_cells], sparse_starts, inclusive=False
            )
            if sparse_cells < nsparse:
                compact_keys = wp.zeros(
                    sparse_cells, dtype=wp.int32, device=dev
                )
                compact_counts = wp.zeros(
                    sparse_cells, dtype=wp.int32, device=dev
                )
                wp.launch(
                    _copy_i32, dim=sparse_cells,
                    inputs=[sparse_keys, compact_keys], device=dev,
                )
                wp.launch(
                    _copy_i32, dim=sparse_cells,
                    inputs=[sparse_counts, compact_counts], device=dev,
                )
                sparse_keys = compact_keys
                sparse_counts = compact_counts

        # Dense-selected levels receive compact offsets; sparse levels allocate
        # zero dense cells and are addressed by their virtual sorted key.
        dense_offset = np.zeros(nlevels, dtype=np.int32)
        running = 0
        for k in range(nlevels):
            dense_offset[k] = running
            if mode[k] == 0:
                running += int(level_cells[k])
        dense_cells = running
        ndense = nsrc - nsparse
        dense_counts = wp.zeros(max(1, dense_cells), dtype=wp.int32,
                                device=dev)
        dense_starts = wp.zeros(max(1, dense_cells), dtype=wp.int32,
                                device=dev)
        dense_cursor = wp.zeros(max(1, dense_cells), dtype=wp.int32,
                                device=dev)
        dense_particles = wp.zeros(max(1, ndense), dtype=wp.uint32, device=dev)
        if nsrc > 0 and dense_cells > 0:
            cell_ids = wp.zeros(nsrc, dtype=wp.int32, device=dev)
            dense_id_kernel = (_ml_dense_cell_ids_f32
                               if gpu.x.dtype == np.float32
                               else _ml_dense_cell_ids_f64)
            dense_offset_dev = wp.array(
                dense_offset, dtype=wp.int32, device=dev
            )
            wp.launch(
                dense_id_kernel, dim=nsrc,
                inputs=[
                    gpu.x.dev, gpu.y.dev, gpu.z.dev, ml['level_of'],
                    ml['origin_x'], ml['origin_y'], ml['origin_z'],
                    ml['cell_size'], ml['nx'], ml['ny'], ml['nz'],
                    dense_offset_dev, ml['storage_mode'], np.int32(self.dim),
                    cell_ids, dense_counts,
                ],
                device=dev,
            )
            wp.utils.array_scan(dense_counts, dense_starts, inclusive=False)
            wp.launch(
                _copy_i32, dim=dense_cells,
                inputs=[dense_starts, dense_cursor], device=dev,
            )
            wp.launch(
                _ml_scatter_dense_particles, dim=nsrc,
                inputs=[cell_ids, dense_cursor, dense_particles], device=dev,
            )
        else:
            dense_offset_dev = wp.array(
                dense_offset, dtype=wp.int32, device=dev
            )
        wp.synchronize_device(dev)
        ml.update({
            'counts': dense_counts,
            'starts': dense_starts,
            'cell_particles': dense_particles,
            'dense_offset': dense_offset_dev,
            'dense_offset_host': dense_offset,
            'dense_cells': dense_cells,
            'sparse_keys': sparse_keys,
            'sparse_counts': sparse_counts,
            'sparse_starts': sparse_starts,
            'sparse_particles': sparse_particles,
            'sparse_cells': int(sparse_cells),
            'sparse_particle_count': nsparse,
        })
        # The all-level sort/RLE arrays are build scratch. Drop their Python
        # references so the persistent hybrid representation retains only the
        # compact dense particles and sparse-selected runs.
        for name in (
                'sorted_keys', 'oracle_keys', 'oracle_counts',
                'oracle_particles'):
            ml.pop(name, None)

    def build_neighbor_cache_gpu(self, src_index, dst_index):
        """Device-resident multilevel neighbor cache (test/diagnostic oracle).

        Retains only the small lengths/total-size readback used to allocate the
        packed neighbor output; the level cell lists and traversal stay on the
        GPU. Generated equation kernels walk the level cells directly rather
        than materializing this cache.
        """
        ml = self._build_multilevel(src_index)
        src = self.particles[src_index].gpu
        dst = self.particles[dst_index].gpu
        dev = self.device
        ndst = dst.get_number_of_particles()
        lengths_k, fill_k, radius_scale, _, _, _ = \
            self._ml_kernels_for(src)

        lengths = wp.zeros(ndst if ndst > 0 else 1, dtype=wp.int32, device=dev)
        starts = wp.zeros(ndst if ndst > 0 else 1, dtype=wp.int32, device=dev)
        base_inputs = [
            src.x.dev, src.y.dev, src.z.dev, src.h.dev,
            dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
            ml['starts'], ml['counts'], ml['cell_particles'],
            ml['origin_x'], ml['origin_y'], ml['origin_z'], ml['cell_size'],
            ml['nx'], ml['ny'], ml['nz'], ml['dense_offset'],
            ml['storage_mode'], ml['virtual_offset'], ml['sparse_keys'],
            ml['sparse_starts'], ml['sparse_counts'], ml['sparse_particles'],
            np.int32(ml['sparse_cells']), ml['support'],
            np.int32(self.nlevels), np.int32(self.dim), radius_scale,
        ]
        active = ndst > 0 and ml['total_cells'] > 0
        if active:
            wp.launch(lengths_k, dim=ndst, inputs=base_inputs + [lengths],
                      device=dev)
            wp.utils.array_scan(lengths, starts, inclusive=False)
            wp.synchronize_device(dev)

        lengths_cpu = lengths.numpy() if ndst > 0 else np.array([], np.int32)
        total = int(np.sum(lengths_cpu, dtype=np.int64))
        neighbors = wp.zeros(total if total > 0 else 1, dtype=wp.uint32,
                             device=dev)
        if active and total > 0:
            fill_inputs = (
                base_inputs[:11] + [starts] + base_inputs[11:] + [neighbors]
            )
            wp.launch(fill_k, dim=ndst, inputs=fill_inputs, device=dev)
            wp.synchronize_device(dev)

        return {
            'lengths': lengths_cpu,
            'lengths_dev': lengths,
            'starts_dev': starts,
            'neighbors_dev': neighbors,
            'total_neighbors': total,
        }

    def level_grid_info(self, src_index):
        """Per-level grid metadata for a source array (test diagnostic).

        Returns host arrays (in the device float precision) so tests can verify
        the per-level padding directly -- e.g. that a particle on a level's
        far/origin edge floors to a cell index inside ``[0, n)`` *before* the
        binning kernel's clamp, which would otherwise mask a padding defect.
        """
        ml = self._build_multilevel(src_index)
        # Per-particle levels are read back from the device here (diagnostic
        # path only -- NOT on the warm update/traversal path).
        levels = (ml['level_of'].numpy()[:ml['nsrc']] if ml['nsrc'] > 0
                  else np.zeros(0, dtype=np.int32))
        return {
            'levels': levels,
            'support': ml['support_host'],
            'origin_x': ml['ox_host'], 'origin_y': ml['oy_host'],
            'origin_z': ml['oz_host'], 'cell_size': ml['cs_host'],
            'nx': ml['nx_host'], 'ny': ml['ny_host'], 'nz': ml['nz_host'],
            'occupied': ml['occupied_host'],
            'storage_mode': ml['storage_mode_host'],
        }

    def sparse_oracle_info(self, src_index):
        """Host diagnostic for the device-built sorted-cell oracle."""
        ml = self._build_multilevel(src_index)
        nruns = ml['sparse_cells']
        return {
            'keys': ml['sparse_keys'].numpy()[:nruns],
            'counts': ml['sparse_counts'].numpy()[:nruns],
            'sorted_particles': ml['sparse_particles'].numpy()[
                :ml['sparse_particle_count']],
            'occupied': ml['occupied_host'].copy(),
            'storage_mode': ml['storage_mode_host'].copy(),
            'level_cells': ml['level_cells_host'].copy(),
        }

    def storage_diagnostics(self, src_index):
        """Exact/projected persistent bytes for dense, sparse, and hybrid.

        The keyed oracle uses int32 logical cell keys because the existing
        flattened-cell contract is int32. Counts, starts, particle indices, and
        level assignments are also int32. Transient radix-sort scratch is
        intentionally reported separately by the benchmark, not hidden in
        these persistent representation totals.
        """
        ml = self._build_multilevel(src_index)
        n = ml['nsrc']
        cells = ml['level_cells_host'].astype(np.int64)
        occupied = ml['occupied_host'].astype(np.int64)
        mode = ml['storage_mode_host']
        dense_cells = int(cells.sum())
        occupied_cells = int(occupied.sum())
        hybrid_dense = int(cells[mode == 0].sum())
        hybrid_sparse = int(occupied[mode == 1].sum())
        metadata = 12 * 4 * self.nlevels
        common = 8 * n + metadata
        return {
            'particles': n,
            'dense_cells': dense_cells,
            'occupied_cells': occupied_cells,
            'hybrid_dense_cells': hybrid_dense,
            'hybrid_sparse_cells': hybrid_sparse,
            'storage_mode': mode.copy(),
            'dense_persistent_bytes': common + 8 * dense_cells,
            'sparse_persistent_bytes': common + 12 * occupied_cells,
            'hybrid_projected_persistent_bytes': (
                common + 8 * hybrid_dense + 12 * hybrid_sparse
            ),
        }

    def candidate_pairs(self, src_index, dst_index):
        """Total candidate pairs the multilevel traversal distance-tests.

        Host-side diagnostic (benchmark path, NOT the residency-constrained
        query path): mirrors the traversal's per-level variable cell-range scan
        and sums the source-particle counts in every scanned cell. Used to show
        the candidate-work reduction versus the global-hmax uniform grid. The
        accepted set is always a subset of the candidate set.
        """
        ml = self._build_multilevel(src_index)
        dst = self.particles[dst_index].gpu
        nlevels, dim, rs = self.nlevels, self.dim, self.radius_scale
        d_x, d_y, d_z, d_h = (dst.x.get(), dst.y.get(), dst.z.get(),
                              dst.h.get())
        counts = ml['counts'].numpy()
        sparse_keys = ml['sparse_keys'].numpy()[:ml['sparse_cells']]
        sparse_counts = ml['sparse_counts'].numpy()[:ml['sparse_cells']]
        sparse_lookup = {
            int(key): int(value)
            for key, value in zip(sparse_keys, sparse_counts)
        }
        mode = ml['storage_mode_host']
        ox, oy, oz = ml['ox_host'], ml['oy_host'], ml['oz_host']
        cs, sup = ml['cs_host'], ml['support_host']
        nx, ny, nz = ml['nx_host'], ml['ny_host'], ml['nz_host']
        sizes = (nx.astype(np.int64) * ny.astype(np.int64)
                 * nz.astype(np.int64))
        virtual_offset = ml['virtual_offset_host'].astype(np.int64)
        dense_offset = ml['dense_offset_host'].astype(np.int64)

        def _rng(c, o, csk, n):
            lo = int(np.floor((c - qr - o) / csk)) - 1
            hi = int(np.floor((c + qr - o) / csk)) + 1
            return max(0, lo), min(int(n) - 1, hi)

        total = 0
        for i in range(len(d_x)):
            for k in range(nlevels):
                if nx[k] <= 0:
                    continue
                csk = float(cs[k])
                qr = max(rs * float(d_h[i]), float(sup[k]))
                ixlo, ixhi = _rng(float(d_x[i]), float(ox[k]), csk, nx[k])
                iylo, iyhi = ((0, 0) if dim < 2 else
                              _rng(float(d_y[i]), float(oy[k]), csk, ny[k]))
                izlo, izhi = ((0, 0) if dim < 3 else
                              _rng(float(d_z[i]), float(oz[k]), csk, nz[k]))
                for iz in range(izlo, izhi + 1):
                    for iy in range(iylo, iyhi + 1):
                        local = iz * nx[k] * ny[k] + iy * nx[k]
                        if mode[k] == 0:
                            base = dense_offset[k] + local
                            total += int(
                                counts[base + ixlo:base + ixhi + 1].sum()
                            )
                        else:
                            for ix in range(ixlo, ixhi + 1):
                                total += sparse_lookup.get(
                                    int(virtual_offset[k] + local + ix), 0
                                )
        return total
