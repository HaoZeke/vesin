#ifndef VESIN_VERLET_HPP
#define VESIN_VERLET_HPP

#include <cstring>
#include <vector>

#include "vesin.h"
#include "math.hpp"

namespace vesin {

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

    // Cached topology from last rebuild (outer list: cutoff + skin)
    std::vector<size_t> pairs_i;    // [n_pairs]
    std::vector<size_t> pairs_j;    // [n_pairs]
    std::vector<int32_t> shifts;    // [n_pairs * 3], flat
    size_t n_pairs;

    // Inner list: indices into outer arrays for pairs within model cutoff.
    // Built by prune_full after rebuild, updated by rolling prune on reuse.
    std::vector<size_t> inner_indices;  // compact list of outer-list indices
    std::vector<uint8_t> inner_mask;    // [n_pairs] 1 if pair is in inner list
    size_t n_inner;
    bool inner_dirty;                   // rolling prune changed mask, needs recompact

    // Rolling pruning state
    size_t prune_cursor;       // position in outer list for rolling scan
    size_t prune_chunk_size;   // outer pairs checked per rolling prune step
    int32_t steps_since_rebuild;
    int32_t prune_interval;    // re-prune every N reuse steps (default 4)

    // Parameters
    double cutoff;        // model cutoff (without skin)
    double skin;
    double half_skin_sq;  // (skin/2)^2, for displacement check
    bool full_list;

    // Track whether last compute() rebuilt
    bool did_rebuild_flag;

    // Is the cache populated at all?
    bool has_cache;
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

/// Rebuild the cached topology by running a full spatial search with
/// cutoff + skin. Stores reference positions and pair topology.
void verlet_rebuild(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
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

/// Build the inner list by pruning the outer list: keep pairs with
/// dist < cutoff. Called once after rebuild. O(N_pairs_outer).
void verlet_prune_full(
    VerletState& state,
    const double (*points)[3],
    const double box[3][3]
);

/// Recompute distance vectors using the inner list only (no distance check).
/// All inner pairs are guaranteed within cutoff by the skin margin.
void verlet_recompute_inner(
    const VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
);

/// Rolling prune: periodically scan a chunk of the outer list to update
/// the inner list (add drifted-in pairs, remove drifted-out pairs).
void verlet_prune_rolling(
    VerletState& state,
    const double (*points)[3],
    const double box[3][3]
);

} // namespace vesin

#endif
