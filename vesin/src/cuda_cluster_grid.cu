// GPU-native cluster grid builder for cluster-pair neighbor search.
//
// Replaces CPU-side grid construction + H2D/D2H transfers with on-device
// kernels. Mirrors the cuda_cell_list.cu pattern for cell assignment,
// counting, prefix sum, and scattering, then adds a cluster building step.

#define CLUSTER_SIZE_GPU 8

__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

// Kernel 1: Compute grid parameters from box matrix.
// Single thread. Computes inv_box, n_cells, n_search, n_cells_total.
// Same logic as compute_cell_grid_params in cuda_cell_list.cu but with
// cluster-pair specific cell count limits.
__global__ void compute_cluster_grid_params(
    const double* __restrict__ box,
    const bool* __restrict__ periodic,
    double cutoff,
    int max_cells,
    size_t n_points,
    double* __restrict__ inv_box,
    int* __restrict__ n_cells,
    int* __restrict__ n_search,
    int* __restrict__ n_cells_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Box matrix elements
    double a = box[0], b = box[1], c = box[2];
    double d = box[3], e = box[4], f = box[5];
    double g = box[6], h = box[7], i = box[8];

    // Determinant
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = 1.0 / det;

    // Inverse box matrix
    inv_box[0] = (e * i - f * h) * invdet;
    inv_box[1] = (c * h - b * i) * invdet;
    inv_box[2] = (b * f - c * e) * invdet;
    inv_box[3] = (f * g - d * i) * invdet;
    inv_box[4] = (a * i - c * g) * invdet;
    inv_box[5] = (c * d - a * f) * invdet;
    inv_box[6] = (d * h - e * g) * invdet;
    inv_box[7] = (b * g - a * h) * invdet;
    inv_box[8] = (a * e - b * d) * invdet;

    // Box vectors
    double va[3] = {box[0], box[1], box[2]};
    double vb[3] = {box[3], box[4], box[5]};
    double vc[3] = {box[6], box[7], box[8]};

    // Cross products for face normals
    double bc[3] = {vb[1] * vc[2] - vb[2] * vc[1], vb[2] * vc[0] - vb[0] * vc[2], vb[0] * vc[1] - vb[1] * vc[0]};
    double ca[3] = {vc[1] * va[2] - vc[2] * va[1], vc[2] * va[0] - vc[0] * va[2], vc[0] * va[1] - vc[1] * va[0]};
    double ab[3] = {va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2], va[0] * vb[1] - va[1] * vb[0]};

    double bc_norm = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    double ca_norm = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
    double ab_norm = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);

    // Distances between opposite faces
    double dist_a = fabs(va[0] * bc[0] + va[1] * bc[1] + va[2] * bc[2]) / bc_norm;
    double dist_b = fabs(vb[0] * ca[0] + vb[1] * ca[1] + vb[2] * ca[2]) / ca_norm;
    double dist_c = fabs(vc[0] * ab[0] + vc[1] * ab[1] + vc[2] * ab[2]) / ab_norm;
    double distances[3] = {dist_a, dist_b, dist_c};

    // Compute number of cells based on cutoff
    n_cells[0] = max(1, (int)floor(distances[0] / cutoff));
    n_cells[1] = max(1, (int)floor(distances[1] / cutoff));
    n_cells[2] = max(1, (int)floor(distances[2] / cutoff));

    int total = n_cells[0] * n_cells[1] * n_cells[2];

    // Limit total cells
    if (total > max_cells) {
        double ratio = cbrt((double)max_cells / total);
        n_cells[0] = max(1, (int)(n_cells[0] * ratio));
        n_cells[1] = max(1, (int)(n_cells[1] * ratio));
        n_cells[2] = max(1, (int)(n_cells[2] * ratio));
        total = n_cells[0] * n_cells[1] * n_cells[2];
    }
    n_cells_total[0] = total;

    // Compute search range
    for (int dim = 0; dim < 3; dim++) {
        double cell_size = distances[dim] / n_cells[dim];
        n_search[dim] = max(1, (int)ceil(cutoff / cell_size));
        if (!periodic[dim] && n_cells[dim] == 1) {
            n_search[dim] = 0;
        }
    }
}

// Kernel 2: Assign atoms to cells via fractional coordinates.
// Also stores fractional z for within-cell sorting.
__global__ void assign_cell_indices_cluster(
    const double* __restrict__ positions,
    const double* __restrict__ inv_box,
    const bool* __restrict__ periodic,
    const int* __restrict__ n_cells,
    size_t n_points,
    int* __restrict__ cell_indices,
    float* __restrict__ atom_frac_z
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) {
        return;
    }

    const double3* pos3 = reinterpret_cast<const double3*>(positions);
    const double3 pos = pos3[idx];

    // frac = pos @ inv_box
    double frac[3];
    frac[0] = pos.x * inv_box[0] + pos.y * inv_box[3] + pos.z * inv_box[6];
    frac[1] = pos.x * inv_box[1] + pos.y * inv_box[4] + pos.z * inv_box[7];
    frac[2] = pos.x * inv_box[2] + pos.y * inv_box[5] + pos.z * inv_box[8];

    int cell_idx[3];

    for (int d = 0; d < 3; d++) {
        if (periodic[d]) {
            int shift = (int)floor(frac[d]);
            frac[d] -= shift;
        } else {
            frac[d] = fmax(0.0, fmin(frac[d], 1.0 - 1e-10));
        }
        cell_idx[d] = (int)(frac[d] * n_cells[d]);
        cell_idx[d] = max(0, min(n_cells[d] - 1, cell_idx[d]));
    }

    cell_indices[idx] = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1];
    atom_frac_z[idx] = (float)frac[2];
}

// Kernel 3a: Count atoms per cell (histogram via atomicAdd)
__global__ void count_atoms_per_cell(
    const int* __restrict__ cell_indices,
    size_t n_points,
    int* __restrict__ cell_counts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }
    atomicAdd(&cell_counts[cell_indices[i]], 1);
}

// Kernel 3b: Exclusive prefix sum of cell counts -> cell_starts
// Single block, uses shared memory. Same as cuda_cell_list.cu.
__global__ void prefix_sum_cluster_cells(
    int* __restrict__ cell_counts,
    int* __restrict__ cell_starts,
    const int* __restrict__ n_cells_total_ptr
) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int n_cells_total = n_cells_total_ptr[0];

    int chunk_size = (n_cells_total + nthreads - 1) / nthreads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, n_cells_total);

    int local_sum = 0;
    for (int i = start; i < end; i++) {
        int val = cell_counts[i];
        cell_starts[i] = local_sum;
        local_sum += val;
    }

    shared[tid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < nthreads; i++) {
            int val = shared[i];
            shared[i] = sum;
            sum += val;
        }
    }
    __syncthreads();

    int offset = shared[tid];
    for (int i = start; i < end; i++) {
        cell_starts[i] += offset;
    }
}

// Kernel 4: Scatter atoms into sorted order by cell.
// Each atom atomically claims a slot in its cell's range.
__global__ void scatter_atoms_by_cell(
    const double* __restrict__ positions,
    const int* __restrict__ cell_indices,
    const float* __restrict__ atom_frac_z,
    int* __restrict__ cell_offsets,
    size_t n_points,
    double* __restrict__ sorted_positions,
    int* __restrict__ sorted_atom_indices,
    float* __restrict__ sorted_frac_z
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    int cell = cell_indices[i];
    int slot = atomicAdd(&cell_offsets[cell], 1);

    // Copy position
    const double3* pos_in = reinterpret_cast<const double3*>(positions);
    double3* pos_out = reinterpret_cast<double3*>(sorted_positions);
    pos_out[slot] = pos_in[i];

    sorted_atom_indices[slot] = static_cast<int>(i);
    sorted_frac_z[slot] = atom_frac_z[i];
}

// Kernel 5: Build clusters from sorted atoms and compute bounding boxes.
// One thread per cell. Groups atoms within the cell into clusters of
// CLUSTER_SIZE_GPU, sorts by frac_z (insertion sort for small groups),
// and computes AABB bounding boxes.
__global__ void build_clusters_and_bboxes(
    const double* __restrict__ sorted_positions,
    const int* __restrict__ sorted_atom_indices,
    const float* __restrict__ sorted_frac_z,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const int* __restrict__ n_cells_total_ptr,
    int* __restrict__ cluster_atom_indices,
    int* __restrict__ cluster_n_atoms,
    float* __restrict__ cluster_bb_lower,
    float* __restrict__ cluster_bb_upper,
    int* __restrict__ cell_cluster_offsets,
    int* __restrict__ n_clusters_total_out
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells_total = n_cells_total_ptr[0];
    if (cell >= n_cells_total) {
        return;
    }

    int start = cell_starts[cell];
    int count = cell_counts[cell];

    // Sort atoms in this cell by frac_z using insertion sort
    // (count is typically small: 8-64 atoms per cell)
    // We sort indices locally, not the global arrays
    int local_order[256]; // max atoms per cell
    int actual_count = min(count, 256);
    for (int i = 0; i < actual_count; i++) {
        local_order[i] = i;
    }
    for (int i = 1; i < actual_count; i++) {
        int key = local_order[i];
        float key_z = sorted_frac_z[start + key];
        int j = i - 1;
        while (j >= 0 && sorted_frac_z[start + local_order[j]] > key_z) {
            local_order[j + 1] = local_order[j];
            j--;
        }
        local_order[j + 1] = key;
    }

    // Compute cluster offset for this cell (use atomicAdd on total counter)
    int n_clusters_this_cell = (actual_count + CLUSTER_SIZE_GPU - 1) / CLUSTER_SIZE_GPU;
    int cluster_base = atomicAdd(n_clusters_total_out, n_clusters_this_cell);

    // Store cell cluster offset
    cell_cluster_offsets[cell] = cluster_base;

    // Build clusters
    for (int ci = 0; ci < n_clusters_this_cell; ci++) {
        int cluster_idx = cluster_base + ci;
        int n_in_cluster = min(CLUSTER_SIZE_GPU, actual_count - ci * CLUSTER_SIZE_GPU);

        // Bounding box init
        float bb_lo[3] = {1e30f, 1e30f, 1e30f};
        float bb_hi[3] = {-1e30f, -1e30f, -1e30f};

        for (int k = 0; k < CLUSTER_SIZE_GPU; k++) {
            if (k < n_in_cluster) {
                int sorted_idx = start + local_order[ci * CLUSTER_SIZE_GPU + k];
                int atom_idx = sorted_atom_indices[sorted_idx];
                cluster_atom_indices[cluster_idx * CLUSTER_SIZE_GPU + k] = atom_idx;

                // Update bounding box from original positions
                float px = (float)sorted_positions[sorted_idx * 3 + 0];
                float py = (float)sorted_positions[sorted_idx * 3 + 1];
                float pz = (float)sorted_positions[sorted_idx * 3 + 2];

                bb_lo[0] = fminf(bb_lo[0], px);
                bb_lo[1] = fminf(bb_lo[1], py);
                bb_lo[2] = fminf(bb_lo[2], pz);
                bb_hi[0] = fmaxf(bb_hi[0], px);
                bb_hi[1] = fmaxf(bb_hi[1], py);
                bb_hi[2] = fmaxf(bb_hi[2], pz);
            } else {
                cluster_atom_indices[cluster_idx * CLUSTER_SIZE_GPU + k] = -1; // padding
            }
        }

        cluster_n_atoms[cluster_idx] = n_in_cluster;
        cluster_bb_lower[cluster_idx * 3 + 0] = bb_lo[0];
        cluster_bb_lower[cluster_idx * 3 + 1] = bb_lo[1];
        cluster_bb_lower[cluster_idx * 3 + 2] = bb_lo[2];
        cluster_bb_upper[cluster_idx * 3 + 0] = bb_hi[0];
        cluster_bb_upper[cluster_idx * 3 + 1] = bb_hi[1];
        cluster_bb_upper[cluster_idx * 3 + 2] = bb_hi[2];
    }
}

// Kernel 6: Finalize cell_cluster_offsets by writing the sentinel value.
// Must run after build_clusters_and_bboxes completes.
// Single thread: writes cell_cluster_offsets[n_cells_total] = n_clusters_total.
__global__ void finalize_cluster_offsets(
    int* __restrict__ cell_cluster_offsets,
    const int* __restrict__ n_cells_total_ptr,
    const int* __restrict__ n_clusters_total_ptr
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    cell_cluster_offsets[n_cells_total_ptr[0]] = n_clusters_total_ptr[0];
}
