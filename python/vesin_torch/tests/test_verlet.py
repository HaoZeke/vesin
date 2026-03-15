"""Tests for VerletNeighborList TorchScript bindings."""

import pytest
import torch

from vesin.torch import NeighborList, VerletNeighborList


class TestVerletLifecycle:
    def test_create_and_destroy(self):
        vl = VerletNeighborList(cutoff=3.0, skin=0.5, full_list=True)
        assert vl.did_rebuild is False

    def test_compute_returns_results(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)
        vl = VerletNeighborList(cutoff=3.0, skin=0.5, full_list=True)
        results = vl.compute(points, box, periodic=True, quantities="ij")
        assert len(results) == 2


class TestVerletRebuild:
    def test_first_call_rebuilds(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)
        vl = VerletNeighborList(cutoff=3.0, skin=0.5, full_list=True)
        vl.compute(points, box, periodic=True, quantities="ij")
        assert vl.did_rebuild is True

    def test_small_movement_no_rebuild(self):
        points1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]],
            dtype=torch.float64,
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)
        vl = VerletNeighborList(cutoff=3.0, skin=1.0, full_list=True)

        vl.compute(points1, box, periodic=True, quantities="ij")
        assert vl.did_rebuild is True

        # Move atoms by less than skin/2 = 0.5
        points2 = torch.tensor(
            [[0.1, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [1.5, 1.4, 0.0]],
            dtype=torch.float64,
        )
        vl.compute(points2, box, periodic=True, quantities="ij")
        assert vl.did_rebuild is False

    def test_large_movement_rebuilds(self):
        points1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)
        vl = VerletNeighborList(cutoff=3.0, skin=0.5, full_list=True)

        vl.compute(points1, box, periodic=True, quantities="ij")
        assert vl.did_rebuild is True

        # Move atom 0 by more than skin/2 = 0.25
        points2 = torch.tensor(
            [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        vl.compute(points2, box, periodic=True, quantities="ij")
        assert vl.did_rebuild is True


class TestVerletCorrectness:
    def test_superset_of_stateless(self):
        """Verlet NL output must be a superset of stateless NL within cutoff."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)

        cutoff = 3.0
        stateless = NeighborList(cutoff=cutoff, full_list=True, sorted=True)
        verlet = VerletNeighborList(
            cutoff=cutoff, skin=0.5, full_list=True, sorted=True
        )

        ref_i, ref_j, ref_S = stateless.compute(points, box, True, "ijS")
        vl_i, vl_j, vl_S = verlet.compute(points, box, True, "ijS")

        ref_pairs = set()
        for k in range(ref_i.shape[0]):
            ref_pairs.add(
                (ref_i[k].item(), ref_j[k].item(), tuple(ref_S[k].tolist()))
            )

        vl_pairs = set()
        for k in range(vl_i.shape[0]):
            vl_pairs.add((vl_i[k].item(), vl_j[k].item(), tuple(vl_S[k].tolist())))

        for p in ref_pairs:
            assert p in vl_pairs, f"Missing pair {p}"


class TestVerletQuantities:
    @pytest.mark.parametrize("quantities", ["ij", "ijS", "D", "d", "ijSDd", "P"])
    def test_quantity_chars(self, quantities):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        )
        box = 4.0 * torch.eye(3, dtype=torch.float64)
        vl = VerletNeighborList(cutoff=3.0, skin=0.5, full_list=True)
        results = vl.compute(points, box, periodic=True, quantities=quantities)
        assert len(results) == len(quantities)


class TestVerletAutograd:
    @pytest.mark.parametrize("full_list", [True, False])
    @pytest.mark.parametrize(
        "requires_grad", [(True, True), (False, True), (True, False)]
    )
    @pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
    def test_autograd(self, full_list, requires_grad, quantities):
        torch.manual_seed(0xDEADBEEF)

        points_fractional = torch.rand((34, 3), dtype=torch.float64)
        box = torch.diag(5 * torch.rand(3, dtype=torch.float64))
        box += torch.rand((3, 3), dtype=torch.float64)

        points = points_fractional @ box

        points.requires_grad_(requires_grad[0])
        box.requires_grad_(requires_grad[1])

        calculator = VerletNeighborList(
            cutoff=7.8, skin=0.5, full_list=full_list, sorted=True
        )

        def compute(points, box):
            results = calculator.compute(
                points, box, periodic=True, quantities=quantities
            )
            return results

        torch.autograd.gradcheck(compute, (points, box), fast_mode=True)


class TestVerletDtype:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        box = torch.eye(3, dtype=dtype) * 3.0
        points = torch.rand((10, 3), dtype=dtype) * 3.0

        vl = VerletNeighborList(cutoff=1.5, skin=0.3, full_list=True)
        i, j, s, D, d = vl.compute(points, box, True, "ijSDd")

        assert i.dtype == torch.int64
        assert j.dtype == torch.int64
        assert s.dtype == torch.int32
        assert D.dtype == dtype
        assert d.dtype == dtype
