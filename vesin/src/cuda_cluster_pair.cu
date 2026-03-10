// Cluster-pair neighbor search on GPU.
//
// Atoms are grouped into clusters of CLUSTER_SIZE (8 for GPU). Each cluster
// has a bounding box (AABB). The search iterates over cluster pairs between
// neighboring cells, applies a BB distance test for fast rejection, then
// expands surviving cluster pairs to atom-level distance checks.
//
// The cluster grid is built on CPU and copied to GPU.
// Output format: per-atom pairs with optional shifts, distances, vectors.

#define CLUSTER_SIZE_GPU 8

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

/// Minimum squared distance between two AABBs.
__device__ inline float bb_distance_sq_gpu(
    const float* __restrict__ bb_lower_i,
    const float* __restrict__ bb_upper_i,
    const float* __restrict__ bb_lower_j,
    const float* __restrict__ bb_upper_j
) {
    float dist_sq = 0.0f;
    for (int d = 0; d < 3; d++) {
        float gap = 0.0f;
        if (bb_lower_i[d] > bb_upper_j[d]) {
            gap = bb_lower_i[d] - bb_upper_j[d];
        } else if (bb_lower_j[d] > bb_upper_i[d]) {
            gap = bb_lower_j[d] - bb_upper_i[d];
        }
        dist_sq += gap * gap;
    }
    return dist_sq;
}

/// Cluster-pair neighbor search kernel.
///
/// Each thread handles one i-cluster. It iterates over neighboring cells and
/// j-clusters, applies the BB distance test, and expands surviving cluster
/// pairs to atom-level checks.
///
/// Arguments:
///   positions:       [n_points][3] atom positions (double, on GPU)
///   cluster_atom_indices: [n_clusters * CLUSTER_SIZE] original atom indices (-1 = padding)
///   cluster_n_atoms: [n_clusters] actual atom count per cluster
///   cluster_bb_lower: [n_clusters * 3] AABB lower bound (float)
///   cluster_bb_upper: [n_clusters * 3] AABB upper bound (float)
///   cluster_cell_ids: [n_clusters] which cell each cluster belongs to
///   cell_offsets:    [n_cells_total + 1] CSR offsets into cluster arrays
///   n_clusters:      total number of clusters
///   box:             [3][3] cell matrix (row-major, double)
///   periodic:        [3] periodicity flags
///   n_cells:         [3] grid dimensions
///   n_search:        [3] search range per dimension
///   cutoff:          cutoff distance (double)
///   is_full_list:    true = emit both (i,j) and (j,i)
///   pair_counter:    atomic counter for output pairs
///   pair_indices:    [max_pairs * 2] output pair indices (size_t)
///   shifts:          [max_pairs * 3] output shifts (int32_t), or nullptr
///   distances:       [max_pairs] output distances (double), or nullptr
///   vectors:         [max_pairs * 3] output vectors (double), or nullptr
///   return_shifts, return_distances, return_vectors: flags
///   max_pairs:       capacity of output arrays
///   overflow_flag:   set to 1 if capacity exceeded
extern "C" __global__ void cluster_pair_search(
    const double* __restrict__ positions,
    const int* __restrict__ cluster_atom_indices,
    const int* __restrict__ cluster_atom_shifts,
    const int* __restrict__ cluster_n_atoms,
    const float* __restrict__ cluster_bb_lower,
    const float* __restrict__ cluster_bb_upper,
    const int* __restrict__ cell_offsets,
    int n_clusters_total,
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    const int* __restrict__ n_search,
    int n_cells_total,
    double cutoff,
    bool is_full_list,
    size_t* __restrict__ pair_counter,
    size_t* __restrict__ pair_indices,
    int* __restrict__ shifts,
    double* __restrict__ distances,
    double* __restrict__ vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* __restrict__ overflow_flag
) {
    int ci_global = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci_global >= n_clusters_total) return;

    double cutoff2 = cutoff * cutoff;
    float cutoff2_f = static_cast<float>(cutoff2);

    int nc_x = n_cells[0], nc_y = n_cells[1], nc_z = n_cells[2];
    int ns_x = n_search[0], ns_y = n_search[1], ns_z = n_search[2];

    // Load i-cluster data
    int ni_atoms = cluster_n_atoms[ci_global];
    int ci_atom_idx[CLUSTER_SIZE_GPU];
    double ci_pos[CLUSTER_SIZE_GPU][3];
    int ci_shift[CLUSTER_SIZE_GPU][3];
    for (int k = 0; k < ni_atoms; k++) {
        ci_atom_idx[k] = cluster_atom_indices[ci_global * CLUSTER_SIZE_GPU + k];
        ci_pos[k][0] = positions[ci_atom_idx[k] * 3 + 0];
        ci_pos[k][1] = positions[ci_atom_idx[k] * 3 + 1];
        ci_pos[k][2] = positions[ci_atom_idx[k] * 3 + 2];
        ci_shift[k][0] = cluster_atom_shifts[(ci_global * CLUSTER_SIZE_GPU + k) * 3 + 0];
        ci_shift[k][1] = cluster_atom_shifts[(ci_global * CLUSTER_SIZE_GPU + k) * 3 + 1];
        ci_shift[k][2] = cluster_atom_shifts[(ci_global * CLUSTER_SIZE_GPU + k) * 3 + 2];
    }

    float bbi_lower[3], bbi_upper[3];
    bbi_lower[0] = cluster_bb_lower[ci_global * 3 + 0];
    bbi_lower[1] = cluster_bb_lower[ci_global * 3 + 1];
    bbi_lower[2] = cluster_bb_lower[ci_global * 3 + 2];
    bbi_upper[0] = cluster_bb_upper[ci_global * 3 + 0];
    bbi_upper[1] = cluster_bb_upper[ci_global * 3 + 1];
    bbi_upper[2] = cluster_bb_upper[ci_global * 3 + 2];

    // Determine which cell this cluster belongs to (via binary search on cell_offsets)
    int cell_i = 0;
    for (int c = 0; c < n_cells_total; c++) {
        if (ci_global >= cell_offsets[c] && ci_global < cell_offsets[c + 1]) {
            cell_i = c;
            break;
        }
    }

    // Decompose cell_i into (cx, cy, cz)
    int cx = cell_i % nc_x;
    int cy = (cell_i / nc_x) % nc_y;
    int cz = cell_i / (nc_x * nc_y);

    // Output buffer (reduces atomic contention)
    const int MAX_BUFFERED = 8;
    size_t buf_i[MAX_BUFFERED], buf_j[MAX_BUFFERED];
    int buf_sx[MAX_BUFFERED], buf_sy[MAX_BUFFERED], buf_sz[MAX_BUFFERED];
    double buf_dist[MAX_BUFFERED];
    double buf_vx[MAX_BUFFERED], buf_vy[MAX_BUFFERED], buf_vz[MAX_BUFFERED];
    int buf_count = 0;

    // Iterate over neighboring cells
    for (int dz = -ns_z; dz <= ns_z; dz++) {
    for (int dy = -ns_y; dy <= ns_y; dy++) {
    for (int dx = -ns_x; dx <= ns_x; dx++) {

        int nx_raw = cx + dx;
        int ny_raw = cy + dy;
        int nz_raw = cz + dz;

        // Wrap and compute cell shift
        int sx = 0, sy = 0, sz = 0;
        int rx = nx_raw, ry = ny_raw, rz = nz_raw;

        // divmod with positive remainder
        if (nc_x > 0) {
            sx = (rx >= 0) ? rx / nc_x : (rx - nc_x + 1) / nc_x;
            rx = rx - sx * nc_x;
            if (rx < 0) { rx += nc_x; sx -= 1; }
        }
        if (nc_y > 0) {
            sy = (ry >= 0) ? ry / nc_y : (ry - nc_y + 1) / nc_y;
            ry = ry - sy * nc_y;
            if (ry < 0) { ry += nc_y; sy -= 1; }
        }
        if (nc_z > 0) {
            sz = (rz >= 0) ? rz / nc_z : (rz - nc_z + 1) / nc_z;
            rz = rz - sz * nc_z;
            if (rz < 0) { rz += nc_z; sz -= 1; }
        }

        // Skip if non-periodic and shift is nonzero
        if ((sx != 0 && !periodic[0]) ||
            (sy != 0 && !periodic[1]) ||
            (sz != 0 && !periodic[2])) {
            continue;
        }

        int cell_j = rz * nc_x * nc_y + ry * nc_x + rx;
        if (cell_j < 0 || cell_j >= n_cells_total) continue;

        int cj_start = cell_offsets[cell_j];
        int cj_end = cell_offsets[cell_j + 1];

        bool cell_shift_is_zero = (sx == 0 && sy == 0 && sz == 0);

        for (int cj = cj_start; cj < cj_end; cj++) {
            // BB distance test (only for same-cell-image pairs)
            if (cell_shift_is_zero) {
                float bbj_lower[3], bbj_upper[3];
                bbj_lower[0] = cluster_bb_lower[cj * 3 + 0];
                bbj_lower[1] = cluster_bb_lower[cj * 3 + 1];
                bbj_lower[2] = cluster_bb_lower[cj * 3 + 2];
                bbj_upper[0] = cluster_bb_upper[cj * 3 + 0];
                bbj_upper[1] = cluster_bb_upper[cj * 3 + 1];
                bbj_upper[2] = cluster_bb_upper[cj * 3 + 2];

                float bb_dist = bb_distance_sq_gpu(
                    bbi_lower, bbi_upper, bbj_lower, bbj_upper
                );
                if (bb_dist > cutoff2_f) continue;
            }

            // Expand cluster pair to atom-level checks
            int nj_atoms = cluster_n_atoms[cj];
            for (int ai = 0; ai < ni_atoms; ai++) {
                int idx_i = ci_atom_idx[ai];
                int si_x = ci_shift[ai][0];
                int si_y = ci_shift[ai][1];
                int si_z = ci_shift[ai][2];

                for (int aj = 0; aj < nj_atoms; aj++) {
                    int idx_j = cluster_atom_indices[cj * CLUSTER_SIZE_GPU + aj];

                    // Per-atom wrapping shifts for j
                    int sj_x = cluster_atom_shifts[(cj * CLUSTER_SIZE_GPU + aj) * 3 + 0];
                    int sj_y = cluster_atom_shifts[(cj * CLUSTER_SIZE_GPU + aj) * 3 + 1];
                    int sj_z = cluster_atom_shifts[(cj * CLUSTER_SIZE_GPU + aj) * 3 + 2];

                    // Total shift = shift_i - shift_j + cell_shift
                    int total_sx = si_x - sj_x + sx;
                    int total_sy = si_y - sj_y + sy;
                    int total_sz = si_z - sj_z + sz;

                    bool total_shift_is_zero = (total_sx == 0 && total_sy == 0 && total_sz == 0);

                    // Self-pair exclusion
                    if (idx_i == idx_j && total_shift_is_zero) continue;

                    // Half-list filtering
                    if (!is_full_list) {
                        if (idx_i > idx_j) continue;
                        if (idx_i == idx_j) {
                            int s_sum = total_sx + total_sy + total_sz;
                            if (s_sum < 0) continue;
                            if (s_sum == 0 && (total_sz < 0 || (total_sz == 0 && total_sy < 0))) continue;
                        }
                    }

                    double pj_x = positions[idx_j * 3 + 0];
                    double pj_y = positions[idx_j * 3 + 1];
                    double pj_z = positions[idx_j * 3 + 2];

                    // Distance vector using total shift
                    double total_cart_x = total_sx * box[0] + total_sy * box[3] + total_sz * box[6];
                    double total_cart_y = total_sx * box[1] + total_sy * box[4] + total_sz * box[7];
                    double total_cart_z = total_sx * box[2] + total_sy * box[5] + total_sz * box[8];

                    double vx = pj_x - ci_pos[ai][0] + total_cart_x;
                    double vy = pj_y - ci_pos[ai][1] + total_cart_y;
                    double vz = pj_z - ci_pos[ai][2] + total_cart_z;

                    double dist2 = vx * vx + vy * vy + vz * vz;

                    if (dist2 < cutoff2) {
                        // Buffer the pair
                        buf_i[buf_count] = static_cast<size_t>(idx_i);
                        buf_j[buf_count] = static_cast<size_t>(idx_j);
                        buf_sx[buf_count] = total_sx;
                        buf_sy[buf_count] = total_sy;
                        buf_sz[buf_count] = total_sz;
                        buf_dist[buf_count] = sqrt(dist2);
                        buf_vx[buf_count] = vx;
                        buf_vy[buf_count] = vy;
                        buf_vz[buf_count] = vz;
                        buf_count++;

                        if (buf_count == MAX_BUFFERED) {
                            // Flush buffer
                            size_t base = atomicAdd_size_t(pair_counter, MAX_BUFFERED);
                            if (base + MAX_BUFFERED > max_pairs) {
                                atomicExch(overflow_flag, 1);
                                return;
                            }
                            for (int b = 0; b < MAX_BUFFERED; b++) {
                                size_t idx = base + b;
                                pair_indices[idx * 2 + 0] = buf_i[b];
                                pair_indices[idx * 2 + 1] = buf_j[b];
                                if (return_shifts) {
                                    shifts[idx * 3 + 0] = buf_sx[b];
                                    shifts[idx * 3 + 1] = buf_sy[b];
                                    shifts[idx * 3 + 2] = buf_sz[b];
                                }
                                if (return_distances) {
                                    distances[idx] = buf_dist[b];
                                }
                                if (return_vectors) {
                                    vectors[idx * 3 + 0] = buf_vx[b];
                                    vectors[idx * 3 + 1] = buf_vy[b];
                                    vectors[idx * 3 + 2] = buf_vz[b];
                                }
                            }
                            buf_count = 0;
                        }
                    }
                }
            }
        }
    }}}

    // Flush remaining buffer
    if (buf_count > 0) {
        size_t base = atomicAdd_size_t(pair_counter, buf_count);
        if (base + buf_count > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }
        for (int b = 0; b < buf_count; b++) {
            size_t idx = base + b;
            pair_indices[idx * 2 + 0] = buf_i[b];
            pair_indices[idx * 2 + 1] = buf_j[b];
            if (return_shifts) {
                shifts[idx * 3 + 0] = buf_sx[b];
                shifts[idx * 3 + 1] = buf_sy[b];
                shifts[idx * 3 + 2] = buf_sz[b];
            }
            if (return_distances) {
                distances[idx] = buf_dist[b];
            }
            if (return_vectors) {
                vectors[idx * 3 + 0] = buf_vx[b];
                vectors[idx * 3 + 1] = buf_vy[b];
                vectors[idx * 3 + 2] = buf_vz[b];
            }
        }
    }
}
