#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <cstring>
#include <vector>

#include "vesin.h"
#include "math.hpp"

namespace vesin {

/// GPU-side buffers for Verlet caching.
/// Allocated on demand when device is CUDA.
struct VerletGpuBuffers {
    // Reference positions on device [n_points * 3]
    double* d_ref_positions = nullptr;

    // Cached topology on device
    size_t* d_pairs_i = nullptr;    // [n_pairs]
    size_t* d_pairs_j = nullptr;    // [n_pairs]
    int32_t* d_shifts = nullptr;    // [n_pairs * 3]

    // Rebuild flag [1] on device
    int* d_rebuild_flag = nullptr;

    // Capacity tracking
    size_t capacity_points = 0;
    size_t capacity_pairs = 0;

    // Device ID (-1 = not allocated)
    int32_t device_id = -1;
};

/// Internal state for Verlet neighbor list caching.
///
/// Caches the pair topology from a full spatial search (with cutoff + skin)
/// and reuses it until any atom displaces by more than skin/2. On reuse,
/// only the distance vectors are recomputed from current positions and
/// cached topology (O(N_pairs) instead of O(N) spatial search).
struct VerletState {
    // Reference positions copied at last rebuild
    std::vector<double> ref_positions; // [n_points * 3], flat
    double ref_box[3][3];
    size_t n_points;
    bool ref_periodic[3];

    // Cached topology from last rebuild (cutoff + skin)
    std::vector<size_t> pairs_i;    // [n_pairs]
    std::vector<size_t> pairs_j;    // [n_pairs]
    std::vector<int32_t> shifts;    // [n_pairs * 3], flat
    size_t n_pairs;

    // Parameters
    double cutoff;        // model cutoff (without skin)
    double skin;
    double half_skin_sq;  // (skin/2)^2, for displacement check
    bool full_list;

    // Track whether last compute() rebuilt
    bool did_rebuild_flag;

    // Is the cache populated at all?
    bool has_cache;

    // GPU buffers (allocated on demand)
    VerletGpuBuffers gpu;

    // Which device was last used
    VesinDeviceKind last_device;
};

/// Check whether any atom has moved more than skin/2 from its reference
/// position, or the box/periodicity/N changed. Returns true if a rebuild
/// is needed.
bool verlet_needs_rebuild(
    const VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
);

/// GPU version of needs_rebuild: runs displacement kernel on device.
/// points must be a device pointer.
bool verlet_needs_rebuild_gpu(
    const VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
);

/// Rebuild the cached topology by running a full spatial search with
/// cutoff + skin. Stores reference positions and pair topology.
void verlet_rebuild(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinDevice device = {VesinCPU, 0}
);

/// Recompute distance vectors from cached topology and current positions.
/// Output written to the VesinNeighborList (caller-owned).
/// Only pairs within the model cutoff are kept.
void verlet_recompute(
    const VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

/// GPU version of recompute: launches kernel to recompute vectors on device.
void verlet_recompute_gpu(
    const VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

/// Free GPU buffers in VerletState.
void verlet_free_gpu_buffers(VerletGpuBuffers& gpu);

/// Upload cached topology from CPU vectors to GPU buffers.
void verlet_upload_topology(VerletState& state);

/// GPU rebuild: run stateless GPU NL, copy topology, upload to GPU buffers.
void verlet_rebuild_gpu(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinDevice device
);

/// Allocate GPU output buffers for Verlet recompute.
void verlet_ensure_gpu_output(
    VesinNeighborList& neighbors,
    size_t n_points,
    VesinDevice device,
    VesinOptions options
);

} // namespace vesin

#endif
