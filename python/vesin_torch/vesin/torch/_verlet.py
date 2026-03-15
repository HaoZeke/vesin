from typing import List, Union

import torch


class VerletNeighborList:
    """A Verlet (displacement-cached) neighbor list calculator for TorchScript.

    Caches the pair topology from a spatial search with ``cutoff + skin`` and
    reuses it until any atom moves beyond ``skin / 2``. On reuse, only the
    distance vectors are recomputed from current positions.
    """

    def __init__(
        self,
        cutoff: float,
        skin: float,
        full_list: bool,
        sorted: bool = False,
    ):
        """
        :param cutoff: spherical cutoff radius (without skin)
        :param skin: extra distance added to cutoff for the spatial search buffer
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``)
            or only once
        :param sorted: should vesin sort the returned pairs in lexicographic order
        """
        self._c = torch.classes.vesin._VerletNeighborList(
            cutoff=cutoff,
            skin=skin,
            full_list=full_list,
            sorted=sorted,
        )

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: Union[bool, torch.Tensor],
        quantities: str = "ij",
        copy: bool = True,
    ) -> List[torch.Tensor]:
        """
        Compute the neighbor list using Verlet caching.

        Same interface as :py:meth:`NeighborList.compute`.

        :param points: positions of all points in the system
        :param box: bounding box of the system
        :param periodic: should we use periodic boundary conditions?
        :param quantities: quantities to return, defaults to "ij"
        :param copy: should we copy the returned quantities, defaults to ``True``

        :return: list of :py:class:`torch.Tensor` as indicated by ``quantities``
        """
        if isinstance(periodic, bool):
            periodic = torch.as_tensor(periodic)

        initial_dtype = points.dtype
        if box.dtype != initial_dtype:
            raise RuntimeError(
                "`points` and `box` must have the same dtype, "
                f"got {points.dtype} and {box.dtype}"
            )

        points = points.to(torch.float64)
        box = box.to(torch.float64)

        results = self._c.compute(
            points=points,
            box=box,
            periodic=periodic,
            quantities=quantities,
            copy=copy,
        )

        updated_results = []
        for q, result in zip(quantities, results, strict=True):
            if q in ("d", "D"):
                result = result.to(initial_dtype)
            updated_results.append(result)
        return updated_results

    @property
    def did_rebuild(self) -> bool:
        """Whether the last compute() call performed a full rebuild."""
        return self._c.did_rebuild()
