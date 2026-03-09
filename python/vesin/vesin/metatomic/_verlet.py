import ctypes
from ctypes import POINTER
from typing import Union

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

import numpy as np

from .._c_api import VesinNeighborList, VesinOptions
from .._c_lib import _get_library


class VerletNeighborList:
    """
    A Verlet (displacement-cached) neighbor list calculator for metatomic systems.

    Caches the pair topology from a spatial search with ``cutoff + skin`` and
    reuses it until any atom moves beyond ``skin / 2``. On reuse, only the
    distance vectors are recomputed from current positions (O(N_pairs) instead of
    O(N) spatial search).

    Requires ``strict=False`` in the ``NeighborListOptions`` (extra pairs from
    the skin buffer are acceptable with smooth cutoffs).
    """

    def __init__(
        self,
        options: NeighborListOptions,
        length_unit: str,
        skin: float,
        check_consistency: bool = False,
    ):
        """
        :param options: :py:class:`metatomic.torch.NeighborListOptions` defining the
            parameters of the neighbor list. Must have ``strict=False``.
        :param length_unit: unit of length used for the systems data
        :param skin: extra distance (in the same length unit) added to cutoff for
            the spatial search buffer. Must be > 0.
        :param check_consistency: whether to run additional checks on the neighbor
            list validity
        """
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

        cutoff = self.options.engine_cutoff(self.length_unit)
        self._lib = _get_library()

        error_message = ctypes.c_char_p()
        self._handle = self._lib.vesin_verlet_new(
            cutoff, skin, options.full_list, error_message
        )
        if self._handle is None:
            raise RuntimeError(error_message.value.decode("utf8"))

        self._neighbors = VesinNeighborList()

        # cached Labels
        self._components = Labels("xyz", torch.tensor([[0], [1], [2]]))
        self._properties = Labels("distance", torch.tensor([[0]]))

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_handle") and self._handle:
            self._lib.vesin_verlet_free(self._handle)
        if hasattr(self, "_lib") and hasattr(self, "_neighbors"):
            self._lib.vesin_free(self._neighbors)

    @property
    def did_rebuild(self) -> bool:
        """Whether the last compute() call performed a full rebuild."""
        if self._handle is None:
            return False
        return self._lib.vesin_verlet_did_rebuild(self._handle)

    def compute(self, system: System) -> TensorBlock:
        """
        Compute the neighbor list for the given system, using Verlet caching.

        :param system: a :py:class:`metatomic.torch.System` containing data about a
            single structure.
        """
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
        status = self._lib.vesin_verlet_compute(
            self._handle,
            points_ptr,
            points.shape[0],
            box_ptr,
            periodic_ptr,
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

        device = system.positions.device
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
