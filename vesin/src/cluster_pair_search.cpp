#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "cluster.hpp"
#include "cpu_cell_list.hpp"

using namespace vesin;

/// Maximal number of cells (same as cell list)
#define MAX_NUMBER_OF_CELLS 1e5

/// divmod with Python semantics (positive remainder)
static std::tuple<int32_t, int32_t> divmod(int32_t a, int32_t b) {
    auto quotient = a / b;
    auto remainder = a % b;
    if (remainder < 0) {
        remainder += b;
        quotient -= 1;
    }
    return std::make_tuple(quotient, remainder);
}

ClusterGrid vesin::build_cluster_grid(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    double cutoff
) {
    ClusterGrid grid;

    auto distances_between_faces = box.distances_between_faces();

    // Compute grid cell dimensions
    auto n_cells_f = Vector{
        std::clamp(std::trunc(distances_between_faces[0] / cutoff), 1.0, HUGE_VAL),
        std::clamp(std::trunc(distances_between_faces[1] / cutoff), 1.0, HUGE_VAL),
        std::clamp(std::trunc(distances_between_faces[2] / cutoff), 1.0, HUGE_VAL),
    };

    // Limit memory (same as cell list)
    auto n_cells_total = n_cells_f[0] * n_cells_f[1] * n_cells_f[2];
    if (n_cells_total > MAX_NUMBER_OF_CELLS) {
        auto ratio_x_y = n_cells_f[0] / n_cells_f[1];
        auto ratio_y_z = n_cells_f[1] / n_cells_f[2];
        n_cells_f[2] = std::trunc(std::cbrt(MAX_NUMBER_OF_CELLS / (ratio_x_y * ratio_y_z * ratio_y_z)));
        n_cells_f[1] = std::trunc(ratio_y_z * n_cells_f[2]);
        n_cells_f[0] = std::trunc(ratio_x_y * n_cells_f[1]);
    }

    grid.n_cells = {
        static_cast<int32_t>(n_cells_f[0]),
        static_cast<int32_t>(n_cells_f[1]),
        static_cast<int32_t>(n_cells_f[2]),
    };

    // Clamp to at least 1
    for (int d = 0; d < 3; d++) {
        if (grid.n_cells[d] < 1) grid.n_cells[d] = 1;
    }

    int32_t total_cells = grid.n_cells[0] * grid.n_cells[1] * grid.n_cells[2];

    // Assign atoms to cells
    struct AtomCell {
        size_t atom_index;
        int32_t cell_linear;
        CellShift wrap_shift;
        float z_frac;  // fractional z for sorting within cell
    };

    std::vector<AtomCell> assignments(n_points);
    for (size_t i = 0; i < n_points; i++) {
        auto fractional = box.cartesian_to_fractional(points[i]);

        auto cell_idx = std::array<int32_t, 3>{
            static_cast<int32_t>(std::floor(fractional[0] * static_cast<double>(grid.n_cells[0]))),
            static_cast<int32_t>(std::floor(fractional[1] * static_cast<double>(grid.n_cells[1]))),
            static_cast<int32_t>(std::floor(fractional[2] * static_cast<double>(grid.n_cells[2]))),
        };

        CellShift shift{};
        for (int d = 0; d < 3; d++) {
            if (box.periodic(d)) {
                auto [q, r] = divmod(cell_idx[d], grid.n_cells[d]);
                shift[d] = q;
                cell_idx[d] = r;
            } else {
                shift[d] = 0;
                cell_idx[d] = std::clamp(cell_idx[d], 0, grid.n_cells[d] - 1);
            }
        }

        int32_t linear = (grid.n_cells[0] * grid.n_cells[1] * cell_idx[2])
                       + (grid.n_cells[0] * cell_idx[1])
                       + cell_idx[0];

        assignments[i] = {i, linear, shift, static_cast<float>(fractional[2])};
    }

    // Count atoms per cell
    std::vector<int32_t> cell_counts(total_cells, 0);
    for (auto& a : assignments) {
        cell_counts[a.cell_linear]++;
    }

    // Sort assignments by cell, then by z within each cell
    std::sort(assignments.begin(), assignments.end(),
        [](const AtomCell& a, const AtomCell& b) {
            if (a.cell_linear != b.cell_linear) return a.cell_linear < b.cell_linear;
            return a.z_frac < b.z_frac;
        }
    );

    // Build clusters: group atoms within each cell into groups of CLUSTER_SIZE_CPU
    grid.cell_offsets.resize(total_cells + 1, 0);
    grid.clusters.clear();

    size_t atom_cursor = 0;
    for (int32_t cell = 0; cell < total_cells; cell++) {
        grid.cell_offsets[cell] = static_cast<int32_t>(grid.clusters.size());

        int32_t count = cell_counts[cell];
        size_t cell_start = atom_cursor;

        // Group into clusters
        for (int32_t offset = 0; offset < count; offset += CLUSTER_SIZE_CPU) {
            Cluster cl{};
            cl.n_atoms = std::min(CLUSTER_SIZE_CPU, count - offset);

            // Initialize BB to inverted extremes
            for (int d = 0; d < 3; d++) {
                cl.bb_lower[d] = std::numeric_limits<float>::max();
                cl.bb_upper[d] = -std::numeric_limits<float>::max();
            }

            for (int32_t k = 0; k < cl.n_atoms; k++) {
                size_t idx = cell_start + offset + k;
                cl.atom_indices[k] = static_cast<int32_t>(assignments[idx].atom_index);

                // Update bounding box from atom positions
                auto& pos = points[assignments[idx].atom_index];
                for (int d = 0; d < 3; d++) {
                    float p = static_cast<float>(pos[d]);
                    cl.bb_lower[d] = std::min(cl.bb_lower[d], p);
                    cl.bb_upper[d] = std::max(cl.bb_upper[d], p);
                }
            }

            // Pad unused indices with -1
            for (int32_t k = cl.n_atoms; k < CLUSTER_SIZE_CPU; k++) {
                cl.atom_indices[k] = -1;
            }

            grid.clusters.push_back(cl);
        }

        atom_cursor += count;
    }
    grid.cell_offsets[total_cells] = static_cast<int32_t>(grid.clusters.size());

    return grid;
}

void vesin::cpu::cluster_pair_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox cell,
    VesinOptions options,
    VesinNeighborList& raw_neighbors
) {
    auto grid = build_cluster_grid(points, n_points, cell, options.cutoff);

    auto cell_matrix = cell.matrix();
    auto cutoff2 = options.cutoff * options.cutoff;
    float cutoff2_f = static_cast<float>(cutoff2);

    auto neighbors = GrowableNeighborList{raw_neighbors, raw_neighbors.length, options};
    neighbors.reset();

    auto distances_between_faces = cell.distances_between_faces();

    // Number of cells to search in each direction
    auto n_search = std::array<int32_t, 3>{
        static_cast<int32_t>(std::ceil(options.cutoff * grid.n_cells[0] / distances_between_faces[0])),
        static_cast<int32_t>(std::ceil(options.cutoff * grid.n_cells[1] / distances_between_faces[1])),
        static_cast<int32_t>(std::ceil(options.cutoff * grid.n_cells[2] / distances_between_faces[2])),
    };

    for (int d = 0; d < 3; d++) {
        if (n_search[d] < 1) n_search[d] = 1;
        if (grid.n_cells[d] == 1 && !cell.periodic(d)) n_search[d] = 0;
    }

    // Iterate over all cells
    for (int32_t cz = 0; cz < grid.n_cells[2]; cz++) {
    for (int32_t cy = 0; cy < grid.n_cells[1]; cy++) {
    for (int32_t cx = 0; cx < grid.n_cells[0]; cx++) {

        int32_t cell_i_linear = (grid.n_cells[0] * grid.n_cells[1] * cz)
                              + (grid.n_cells[0] * cy) + cx;

        int32_t ci_start = grid.cell_offsets[cell_i_linear];
        int32_t ci_end = grid.cell_offsets[cell_i_linear + 1];

        // Search neighboring cells
        for (int32_t dz = -n_search[2]; dz <= n_search[2]; dz++) {
        for (int32_t dy = -n_search[1]; dy <= n_search[1]; dy++) {
        for (int32_t dx = -n_search[0]; dx <= n_search[0]; dx++) {

            int32_t nx = cx + dx, ny = cy + dy, nz = cz + dz;

            // Wrap neighbor cell and compute cell shift
            auto [sx, rx] = divmod(nx, grid.n_cells[0]);
            auto [sy, ry] = divmod(ny, grid.n_cells[1]);
            auto [sz, rz] = divmod(nz, grid.n_cells[2]);

            // Skip non-periodic wrapping
            if ((sx != 0 && !cell.periodic(0)) ||
                (sy != 0 && !cell.periodic(1)) ||
                (sz != 0 && !cell.periodic(2))) {
                continue;
            }

            int32_t cell_j_linear = (grid.n_cells[0] * grid.n_cells[1] * rz)
                                  + (grid.n_cells[0] * ry) + rx;

            int32_t cj_start = grid.cell_offsets[cell_j_linear];
            int32_t cj_end = grid.cell_offsets[cell_j_linear + 1];

            auto cell_shift_base = CellShift{{sx, sy, sz}};

            // Iterate over cluster pairs between these two cells
            for (int32_t ci = ci_start; ci < ci_end; ci++) {
                const auto& cluster_i = grid.clusters[ci];
                for (int32_t cj = cj_start; cj < cj_end; cj++) {
                    const auto& cluster_j = grid.clusters[cj];

                    // BB distance test (fast reject)
                    // Note: BB test uses Cartesian positions which don't
                    // account for the cell shift, so we skip it when there
                    // is a non-zero cell shift (periodic images). The BB test
                    // is still beneficial for same-image cluster pairs.
                    if (sx == 0 && sy == 0 && sz == 0) {
                        float bb_dist = bb_distance_sq(cluster_i, cluster_j);
                        if (bb_dist > cutoff2_f) {
                            continue;
                        }
                    }

                    // Expand to atom pairs
                    for (int32_t ai = 0; ai < cluster_i.n_atoms; ai++) {
                        size_t idx_i = cluster_i.atom_indices[ai];
                        for (int32_t aj = 0; aj < cluster_j.n_atoms; aj++) {
                            size_t idx_j = cluster_j.atom_indices[aj];

                            auto shift = cell_shift_base;
                            bool shift_is_zero = shift[0] == 0 && shift[1] == 0 && shift[2] == 0;

                            if (idx_i == idx_j && shift_is_zero) {
                                continue;
                            }

                            if (!options.full) {
                                if (idx_i > idx_j) continue;
                                if (idx_i == idx_j) {
                                    if (shift[0] + shift[1] + shift[2] < 0) continue;
                                    if ((shift[0] + shift[1] + shift[2] == 0) &&
                                        (shift[2] < 0 || (shift[2] == 0 && shift[1] < 0))) {
                                        continue;
                                    }
                                }
                            }

                            auto vector = points[idx_j] - points[idx_i] + shift.cartesian(cell_matrix);
                            auto distance2 = vector.dot(vector);

                            if (distance2 < cutoff2) {
                                auto index = neighbors.length();
                                neighbors.set_pair(index, idx_i, idx_j);
                                if (options.return_shifts) {
                                    neighbors.set_shift(index, shift);
                                }
                                if (options.return_distances) {
                                    neighbors.set_distance(index, std::sqrt(distance2));
                                }
                                if (options.return_vectors) {
                                    neighbors.set_vector(index, vector);
                                }
                                neighbors.increment_length();
                            }
                        }
                    }
                }
            }
        }}}
    }}}

    if (options.sorted) {
        neighbors.sort();
    }
}
