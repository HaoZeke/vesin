.. _metatomic-api:

Metatomic interface
===================

Vesin offers an interface to compute neighbor lists for `metatomic's
<https://docs.metatensor.org/metatomic/>`_ atomistic machine learning models.

.. autofunction:: vesin.metatomic.compute_requested_neighbors

.. autofunction:: vesin.metatomic.compute_requested_neighbors_from_options

.. autoclass:: vesin.metatomic.NeighborList
    :members:


Verlet interface
----------------

For molecular dynamics simulations, the Verlet interface caches neighbor list
topology and avoids redundant spatial searches across MD steps. Pairs within
``cutoff + skin`` are found once, and reused until any atom moves beyond
``skin / 2``.

When the system lives on a CUDA device and ``vesin-torch`` is installed, GPU
Verlet kernels are used automatically (no device-to-host copies).

.. autofunction:: vesin.metatomic.compute_requested_neighbors_with_skin

.. autoclass:: vesin.metatomic.VerletNeighborList
    :members:
