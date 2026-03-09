"""Tests for the Verlet (displacement-cached) neighbor list."""

from typing import Dict, List, Optional

import pytest

torch = pytest.importorskip("torch")
metatomic = pytest.importorskip("metatomic")

from metatensor.torch import Labels, TensorMap  # noqa: E402
from metatomic.torch import (  # noqa: E402
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from vesin.metatomic import (  # noqa: E402
    NeighborList,
    VerletNeighborList,
    compute_requested_neighbors_with_skin,
)


def _make_system(positions, cell_scale=4.0, pbc=True):
    """Helper to create a System from positions array."""
    pos = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    cell = (cell_scale * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
    pbc_t = torch.ones(3, dtype=bool) if pbc else torch.zeros(3, dtype=bool)
    types = torch.tensor(list(range(pos.shape[0])), dtype=torch.int32)
    return System(positions=pos, cell=cell, pbc=pbc_t, types=types)


def _collect_pairs(neighbors):
    """Extract (i, j, sa, sb, sc) tuples from a TensorBlock for comparison."""
    samples = neighbors.samples
    result = set()
    for k in range(samples.values.shape[0]):
        row = samples.values[k]
        result.add(tuple(row.tolist()))
    return result


class TestVerletLifecycle:
    def test_create_and_destroy(self):
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)
        assert vl.did_rebuild is False

    def test_strict_raises(self):
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=True)
        with pytest.raises(ValueError, match="strict=False"):
            VerletNeighborList(options, length_unit="A", skin=0.5)

    def test_negative_skin_raises(self):
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        with pytest.raises(ValueError, match="skin must be > 0"):
            VerletNeighborList(options, length_unit="A", skin=-0.1)


class TestVerletRebuild:
    def test_first_call_rebuilds(self):
        system = _make_system([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)

        neighbors = vl.compute(system)
        assert vl.did_rebuild is True
        assert neighbors.values.shape[0] > 0

    def test_small_movement_no_rebuild(self):
        system1 = _make_system(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]]
        )
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=1.0)

        vl.compute(system1)
        assert vl.did_rebuild is True

        # Move atoms by less than skin/2 = 0.5
        system2 = _make_system(
            [[0.1, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [1.5, 1.4, 0.0]]
        )
        vl.compute(system2)
        assert vl.did_rebuild is False

    def test_large_movement_triggers_rebuild(self):
        system1 = _make_system(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)

        vl.compute(system1)
        assert vl.did_rebuild is True

        # Move atom 0 by more than skin/2 = 0.25
        system2 = _make_system(
            [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        vl.compute(system2)
        assert vl.did_rebuild is True

    def test_repeated_same_positions_no_rebuild(self):
        system = _make_system(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)

        vl.compute(system)
        assert vl.did_rebuild is True

        for _ in range(5):
            vl.compute(system)
            assert vl.did_rebuild is False


class TestVerletCorrectness:
    def test_matches_stateless(self):
        """Verlet NL must contain all pairs that stateless NL finds."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        system = _make_system(positions)

        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)

        # Stateless reference
        ref_calc = NeighborList(options, length_unit="A")
        ref_neighbors = ref_calc.compute(system)
        ref_pairs = _collect_pairs(ref_neighbors)

        # Verlet
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)
        vl_neighbors = vl.compute(system)
        vl_pairs = _collect_pairs(vl_neighbors)

        # All reference pairs must be in verlet output
        for p in ref_pairs:
            assert p in vl_pairs, f"Missing pair {p}"

    def test_half_list_matches_stateless(self):
        """Half list Verlet must match stateless half list."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        system = _make_system(positions)
        options = NeighborListOptions(cutoff=3.0, full_list=False, strict=False)

        ref_calc = NeighborList(options, length_unit="A")
        ref_neighbors = ref_calc.compute(system)
        ref_pairs = _collect_pairs(ref_neighbors)

        vl = VerletNeighborList(options, length_unit="A", skin=0.5)
        vl_neighbors = vl.compute(system)
        vl_pairs = _collect_pairs(vl_neighbors)

        for p in ref_pairs:
            assert p in vl_pairs

    def test_md_trajectory_correctness(self):
        """Over a short MD-like trajectory, Verlet output must always be a
        superset of the stateless NL output."""
        import math

        positions = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ]
        options = NeighborListOptions(cutoff=3.0, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)
        ref_calc = NeighborList(options, length_unit="A")

        rebuild_count = 0
        n_steps = 15

        for step in range(n_steps):
            system = _make_system(positions, cell_scale=5.0)
            vl_neighbors = vl.compute(system)
            ref_neighbors = ref_calc.compute(system)

            if vl.did_rebuild:
                rebuild_count += 1

            vl_pairs = _collect_pairs(vl_neighbors)
            ref_pairs = _collect_pairs(ref_neighbors)
            for p in ref_pairs:
                assert p in vl_pairs, f"Step {step}: missing pair {p}"

            # Small perturbation
            dx = 0.03 * math.sin(step * 1.1)
            dy = 0.03 * math.sin(step * 1.3 + 1.0)
            dz = 0.03 * math.sin(step * 1.7 + 2.0)
            for i in range(len(positions)):
                positions[i][0] += dx * (i + 1)
                positions[i][1] += dy * (i + 1)
                positions[i][2] += dz * (i + 1)

        assert rebuild_count < n_steps
        assert rebuild_count >= 1


class TestVerletAutograd:
    def test_gradients_flow(self):
        """Forces from Verlet NL must have nonzero gradients."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
        system = System(
            positions=positions,
            cell=cell,
            pbc=torch.ones(3, dtype=bool),
            types=torch.tensor([6, 8]),
        )

        options = NeighborListOptions(cutoff=3.5, full_list=True, strict=False)
        vl = VerletNeighborList(options, length_unit="A", skin=0.5)
        neighbors = vl.compute(system)

        value = ((neighbors.values) ** 2).sum() * torch.linalg.det(cell)
        value.backward()

        assert positions.grad is not None
        assert cell.grad is not None
        assert torch.linalg.norm(positions.grad) > 0
        assert torch.linalg.norm(cell.grad) > 0


class TestComputeWithSkin:
    """Test the compute_requested_neighbors_with_skin helper."""

    def test_basic_usage(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
        pbc = torch.ones(3, dtype=bool)
        types = torch.tensor([6, 8])

        class SimpleModel(torch.nn.Module):
            def requested_neighbor_lists(self) -> List[NeighborListOptions]:
                return [
                    NeighborListOptions(cutoff=3.5, full_list=True, strict=False)
                ]

            def forward(
                self,
                systems: List[System],
                outputs: Dict[str, ModelOutput],
                selected_atoms: Optional[Labels],
            ) -> Dict[str, TensorMap]:
                return {}

        model = SimpleModel()
        system = System(positions=positions, cell=cell, pbc=pbc, types=types)

        cache = {}
        compute_requested_neighbors_with_skin(
            systems=system,
            system_length_unit="A",
            model=model,
            skin=0.5,
            model_length_unit="A",
            verlet_cache=cache,
        )

        all_options = system.known_neighbor_lists()
        assert len(all_options) == 1
        assert len(cache) == 1

    def test_strict_option_falls_back(self):
        """strict=True options should use stateless NL even with skin > 0."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
        pbc = torch.ones(3, dtype=bool)
        types = torch.tensor([6, 8])

        class StrictModel(torch.nn.Module):
            def requested_neighbor_lists(self) -> List[NeighborListOptions]:
                return [
                    NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
                ]

            def forward(
                self,
                systems: List[System],
                outputs: Dict[str, ModelOutput],
                selected_atoms: Optional[Labels],
            ) -> Dict[str, TensorMap]:
                return {}

        model = StrictModel()
        system = System(positions=positions, cell=cell, pbc=pbc, types=types)

        cache = {}
        compute_requested_neighbors_with_skin(
            systems=system,
            system_length_unit="A",
            model=model,
            skin=0.5,
            model_length_unit="A",
            verlet_cache=cache,
        )

        all_options = system.known_neighbor_lists()
        assert len(all_options) == 1
        # strict options should NOT be cached (they use stateless)
        assert len(cache) == 0
