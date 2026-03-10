import ctypes
from ctypes import POINTER
from typing import Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

import numpy as np

from .._c_api import VesinDevice, VesinNeighborList, VesinOptions, VesinCPU
from .._c_lib import _get_library

try:
    from vesin.torch import VerletNeighborList as VerletNeighborListTorch
except ImportError:
    VerletNeighborListTorch = None


class VerletNeighborList:
    """
    A Verlet (displacement-cached) neighbor list calculator for metatomic systems.

    Caches the pair topology from a spatial search with ``cutoff + skin`` and
    reuses it until any atom moves beyond ``skin / 2``. On reuse, only the
    distance vectors are recomputed from current positions (O(N_pairs) instead of
    O(N) spatial search).

    Requires ``strict=False`` in the ``NeighborListOptions`` (extra pairs from
    the skin buffer are acceptable with smooth cutoffs).

    When the system lives on a CUDA device and ``vesin-torch`` is installed, the
    GPU Verlet path is used automatically (NVRTC displacement check + pair
    recompute kernels). Otherwise falls back to CPU Verlet via ctypes.
    """

    def __init__(
        self,
        options: NeighborListOptions,
        length_unit: str,
        skin: float,
        check_consistency: bool = False,
    ):
        if options.strict:
            raise ValueError(
                "VerletNeighborList requires strict=False "
                "(extra pairs from skin buffer are OK with smooth cutoffs)"
            )

        if skin <= 0:
            raise ValueError("skin must be > 0")

        self.options = options
        self.length_unit = length_unit
        self.skin = skin
        self.check_consistency = check_consistency
        self._cutoff = self.options.engine_cutoff(self.length_unit)

        # CPU path (always available): ctypes-based Verlet
        self._lib = _get_library()
        error_message = ctypes.c_char_p()
        self._handle = self._lib.vesin_verlet_new(
            self._cutoff, skin, options.full_list, error_message
        )
        if self._handle is None:
            raise RuntimeError(error_message.value.decode("utf8"))

        self._neighbors = VesinNeighborList()

        # GPU path (optional): vesin-torch VerletNeighborList
        self._torch_vl: Optional[object] = None

        # cached Labels
        self._components = Labels("xyz", torch.tensor([[0], [1], [2]]))
        self._properties = Labels("distance", torch.tensor([[0]]))

    def _get_torch_vl(self):
        """Lazily create the torch-based Verlet NL (avoids overhead when CPU-only)."""
        if self._torch_vl is None:
            if VerletNeighborListTorch is None:
                raise RuntimeError(
                    "GPU Verlet requires vesin-torch. "
                    "Install it or move the system to CPU."
                )
            self._torch_vl = VerletNeighborListTorch(
                cutoff=self._cutoff,
                skin=self.skin,
                full_list=self.options.full_list,
                sorted=False,
            )
        return self._torch_vl

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_handle") and self._handle:
            self._lib.vesin_verlet_free(self._handle)
        if hasattr(self, "_lib") and hasattr(self, "_neighbors"):
            self._lib.vesin_free(self._neighbors)

    @property
    def did_rebuild(self) -> bool:
        """Whether the last compute() call performed a full rebuild."""
        if self._last_was_gpu:
            return self._get_torch_vl().did_rebuild
        if self._handle is None:
            return False
        return self._lib.vesin_verlet_did_rebuild(self._handle)

    def compute(self, system: System) -> TensorBlock:
        """
        Compute the neighbor list for the given system, using Verlet caching.

        Automatically dispatches to GPU Verlet when the system is on CUDA
        and vesin-torch is available.
        """
        device = system.positions.device
        use_gpu = device.type == "cuda" and VerletNeighborListTorch is not None

        if use_gpu:
            return self._compute_gpu(system, device)
        else:
            return self._compute_cpu(system, device)

    def _compute_gpu(self, system: System, device: torch.device) -> TensorBlock:
        """GPU path: use vesin-torch VerletNeighborList with CUDA tensors."""
        self._last_was_gpu = True
        vl = self._get_torch_vl()

        points = system.positions.detach().to(dtype=torch.float64)
        box = system.cell.detach().to(dtype=torch.float64)

        # vesin-torch VerletNeighborList returns (P, S, D) or subsets
        results = vl.compute(
            points=points, box=box, periodic=system.pbc,
            quantities="PSD", copy=True,
        )
        P, S, D = results

        P = P.to(dtype=torch.int32)
        S = S.to(dtype=torch.int32)

        self._components = self._components.to(device=device)
        self._properties = self._properties.to(device=device)

        neighbors = TensorBlock(
            D.reshape(-1, 3, 1),
            samples=Labels(
                names=[
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                values=torch.hstack([P, S]),
            ),
            components=[self._components],
            properties=self._properties,
        )

        register_autograd_neighbors(
            system, neighbors, check_consistency=self.check_consistency
        )

        return neighbors

    def _compute_cpu(self, system: System, device: torch.device) -> TensorBlock:
        """CPU path: ctypes-based Verlet (original implementation)."""
        self._last_was_gpu = False

        points = system.positions.detach().cpu().numpy().astype(np.float64)
        box = system.cell.detach().cpu().numpy().astype(np.float64)
        pbc = system.pbc.cpu().numpy()

        points = np.ascontiguousarray(points)
        box = np.ascontiguousarray(box)
        periodic = np.ascontiguousarray(pbc.astype(np.bool_))

        points_ptr = points.ctypes.data_as(POINTER(ctypes.c_double))
        box_ptr = box.ctypes.data_as(POINTER(ctypes.c_double))
        periodic_ptr = periodic.ctypes.data_as(POINTER(ctypes.c_bool))

        options = VesinOptions()
        options.cutoff = 0.0  # ignored by verlet_compute
        options.full = False  # ignored by verlet_compute
        options.sorted = False
        options.algorithm = 0  # auto
        options.return_shifts = True
        options.return_distances = False
        options.return_vectors = True

        error_message = ctypes.c_char_p()
        vesin_dev = VesinDevice(VesinCPU, 0)
        status = self._lib.vesin_verlet_compute(
            self._handle,
            points_ptr,
            points.shape[0],
            box_ptr,
            periodic_ptr,
            vesin_dev,
            options,
            ctypes.byref(self._neighbors),
            error_message,
        )

        if status != 0:
            raise RuntimeError(error_message.value.decode("utf8"))

        n_pairs = self._neighbors.length

        if n_pairs == 0:
            P = torch.zeros((0, 2), dtype=torch.int32)
            S = torch.zeros((0, 3), dtype=torch.int32)
            D = torch.zeros((0, 3), dtype=torch.float64)
        else:
            # Convert ctypes pointers to numpy arrays (zero-copy views)
            pairs_np = np.ctypeslib.as_array(
                ctypes.cast(self._neighbors.pairs, POINTER(ctypes.c_size_t)),
                shape=(n_pairs, 2),
            )
            shifts_np = np.ctypeslib.as_array(
                ctypes.cast(self._neighbors.shifts, POINTER(ctypes.c_int32)),
                shape=(n_pairs, 3),
            )
            vectors_np = np.ctypeslib.as_array(
                ctypes.cast(self._neighbors.vectors, POINTER(ctypes.c_double)),
                shape=(n_pairs, 3),
            )

            P = torch.as_tensor(pairs_np.copy(), dtype=torch.int32)
            S = torch.as_tensor(shifts_np.copy(), dtype=torch.int32)
            D = torch.as_tensor(vectors_np.copy())

        P = P.to(device=device)
        S = S.to(device=device)
        D = D.to(device=device)

        self._components = self._components.to(device=device)
        self._properties = self._properties.to(device=device)

        neighbors = TensorBlock(
            D.reshape(-1, 3, 1),
            samples=Labels(
                names=[
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                values=torch.hstack([P, S]),
            ),
            components=[self._components],
            properties=self._properties,
        )

        register_autograd_neighbors(
            system, neighbors, check_consistency=self.check_consistency
        )

        return neighbors
