"""Tests for Verlet caching with CUDA arrays."""

import cupy as cp
import pytest

from vesin import NeighborList


try:
    cp.cuda.Device(0).compute_capability
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA is not available", allow_module_level=True)


def test_cuda_pair_crossing_cutoff_inside_skin_is_reported():
    nl = NeighborList(cutoff=1.0, full_list=False, skin=0.4, algorithm="cell_list")
    box = cp.asarray(10.0 * cp.eye(3), dtype=cp.float64)

    pos1 = cp.asarray([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], dtype=cp.float64)
    i1, j1 = nl.compute(pos1, box, periodic=False, quantities="ij")
    assert list(zip(cp.asnumpy(i1).tolist(), cp.asnumpy(j1).tolist(), strict=True)) == []

    pos2 = cp.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=cp.float64)
    i2, j2, d2 = nl.compute(pos2, box, periodic=False, quantities="ijd")

    assert list(zip(cp.asnumpy(i2).tolist(), cp.asnumpy(j2).tolist(), strict=True)) == [(0, 1)]
    assert cp.asnumpy(d2).tolist() == pytest.approx([0.95])
