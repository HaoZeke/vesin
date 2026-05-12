__device__ inline size_t atomicAdd_size_t(size_t* address, size_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    return static_cast<size_t>(atomicAdd(address_as_ull, static_cast<unsigned long long>(val)));
}

__global__ void check_verlet_displacements(
    const double* __restrict__ positions,
    const double* __restrict__ ref_positions,
    size_t n_points,
    double half_skin_sq,
    int* rebuild_flag
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    double dx = positions[i * 3 + 0] - ref_positions[i * 3 + 0];
    double dy = positions[i * 3 + 1] - ref_positions[i * 3 + 1];
    double dz = positions[i * 3 + 2] - ref_positions[i * 3 + 2];
    double disp_sq = dx * dx + dy * dy + dz * dz;

    if (disp_sq > half_skin_sq) {
        atomicExch(rebuild_flag, 1);
    }
}

__global__ void filter_verlet_candidates(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const size_t* __restrict__ candidate_pairs,
    const int* __restrict__ candidate_shifts,
    size_t candidate_length,
    double cutoff,
    size_t* length,
    size_t* pair_indices,
    int* shifts_out,
    double* distances,
    double* vectors,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* overflow_flag
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = idx < candidate_length;
    double cutoff2 = cutoff * cutoff;

    size_t i = 0;
    size_t j = 0;
    int sx = 0;
    int sy = 0;
    int sz = 0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double dist_sq = 0.0;

    if (valid) {
        i = candidate_pairs[idx * 2 + 0];
        j = candidate_pairs[idx * 2 + 1];

        sx = candidate_shifts[idx * 3 + 0];
        sy = candidate_shifts[idx * 3 + 1];
        sz = candidate_shifts[idx * 3 + 2];

        const double* ri = &positions[i * 3];
        const double* rj = &positions[j * 3];

        double shift_x = sx * box[0] + sy * box[3] + sz * box[6];
        double shift_y = sx * box[1] + sy * box[4] + sz * box[7];
        double shift_z = sx * box[2] + sy * box[5] + sz * box[8];

        vx = rj[0] - ri[0] + shift_x;
        vy = rj[1] - ri[1] + shift_y;
        vz = rj[2] - ri[2] + shift_z;
        dist_sq = vx * vx + vy * vy + vz * vz;
    }

    bool keep = valid && dist_sq < cutoff2;

    extern __shared__ unsigned int shared_counts[];
    unsigned int local_count = keep ? 1u : 0u;
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int value = 0;
        if (threadIdx.x >= offset) {
            value = shared_counts[threadIdx.x - offset];
        }
        __syncthreads();
        shared_counts[threadIdx.x] += value;
        __syncthreads();
    }

    __shared__ size_t block_base;
    unsigned int block_count = shared_counts[blockDim.x - 1];
    if (threadIdx.x == 0) {
        block_base = block_count == 0 ? 0 : atomicAdd_size_t(length, static_cast<size_t>(block_count));
    }
    __syncthreads();

    if (keep) {
        size_t out = block_base + static_cast<size_t>(shared_counts[threadIdx.x] - 1u);
        if (out >= max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }

        pair_indices[out * 2 + 0] = i;
        pair_indices[out * 2 + 1] = j;

        if (return_shifts) {
            shifts_out[out * 3 + 0] = sx;
            shifts_out[out * 3 + 1] = sy;
            shifts_out[out * 3 + 2] = sz;
        }

        if (return_distances) {
            distances[out] = sqrt(dist_sq);
        }

        if (return_vectors) {
            vectors[out * 3 + 0] = vx;
            vectors[out * 3 + 1] = vy;
            vectors[out * 3 + 2] = vz;
        }
    }
}
