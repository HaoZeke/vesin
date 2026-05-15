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

__device__ inline int unpack_verlet_shift(int packed, int offset) {
    int value = (packed >> offset) & 1023;
    return value >= 512 ? value - 1024 : value;
}

__global__ void pack_verlet_candidates(
    const size_t* __restrict__ candidate_pairs,
    const int* __restrict__ candidate_shifts,
    size_t candidate_length,
    unsigned int* compact_pairs,
    int* compact_shifts,
    int* overflow_flag
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_length) {
        return;
    }

    size_t i = candidate_pairs[idx * 2 + 0];
    size_t j = candidate_pairs[idx * 2 + 1];
    int sx = candidate_shifts[idx * 3 + 0];
    int sy = candidate_shifts[idx * 3 + 1];
    int sz = candidate_shifts[idx * 3 + 2];

    if (i > 4294967295ULL || j > 4294967295ULL ||
        sx < -512 || sx > 511 || sy < -512 || sy > 511 || sz < -512 || sz > 511) {
        atomicExch(overflow_flag, 1);
        return;
    }

    compact_pairs[idx * 2 + 0] = static_cast<unsigned int>(i);
    compact_pairs[idx * 2 + 1] = static_cast<unsigned int>(j);
    compact_shifts[idx] = (sx & 1023) | ((sy & 1023) << 10) | ((sz & 1023) << 20);
}

__global__ void filter_verlet_compact_candidates(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const unsigned int* __restrict__ candidate_pairs,
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
    if (idx >= candidate_length) {
        return;
    }

    double cutoff2 = cutoff * cutoff;
    size_t i = static_cast<size_t>(candidate_pairs[idx * 2 + 0]);
    size_t j = static_cast<size_t>(candidate_pairs[idx * 2 + 1]);

    int packed_shift = candidate_shifts[idx];
    int sx = unpack_verlet_shift(packed_shift, 0);
    int sy = unpack_verlet_shift(packed_shift, 10);
    int sz = unpack_verlet_shift(packed_shift, 20);

    const double* ri = &positions[i * 3];
    const double* rj = &positions[j * 3];

    double vx = rj[0] - ri[0];
    double vy = rj[1] - ri[1];
    double vz = rj[2] - ri[2];
    if (sx != 0 || sy != 0 || sz != 0) {
        vx += sx * box[0] + sy * box[3] + sz * box[6];
        vy += sx * box[1] + sy * box[4] + sz * box[7];
        vz += sx * box[2] + sy * box[5] + sz * box[8];
    }
    double dist_sq = vx * vx + vy * vy + vz * vz;

    if (dist_sq < cutoff2) {
        size_t out = atomicAdd_size_t(length, 1);
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

__global__ void filter_verlet_compact_candidates_block(
    const double* __restrict__ positions,
    const double* __restrict__ box,
    const unsigned int* __restrict__ candidate_pairs,
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
    const double cutoff2 = cutoff * cutoff;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < candidate_length;
         idx += blockDim.x * gridDim.x) {

        size_t i = static_cast<size_t>(candidate_pairs[idx * 2 + 0]);
        size_t j = static_cast<size_t>(candidate_pairs[idx * 2 + 1]);

        int packed_shift = candidate_shifts[idx];
        int sx = unpack_verlet_shift(packed_shift, 0);
        int sy = unpack_verlet_shift(packed_shift, 10);
        int sz = unpack_verlet_shift(packed_shift, 20);

        const double* ri = &positions[i * 3];
        const double* rj = &positions[j * 3];

        double vx = rj[0] - ri[0];
        double vy = rj[1] - ri[1];
        double vz = rj[2] - ri[2];
        if (sx != 0 || sy != 0 || sz != 0) {
            vx += sx * box[0] + sy * box[3] + sz * box[6];
            vy += sx * box[1] + sy * box[4] + sz * box[7];
            vz += sx * box[2] + sy * box[5] + sz * box[8];
        }
        double dist_sq = vx * vx + vy * vy + vz * vz;

        bool keep = dist_sq < cutoff2;
    unsigned active_mask = __activemask();
    unsigned passing_mask = __ballot_sync(active_mask, keep);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_rank = __popc(passing_mask & ((1u << lane) - 1u));
    unsigned warp_count = __popc(passing_mask);

    extern __shared__ unsigned int warp_offsets[];
    int warp_count_slots = (blockDim.x + 31) / 32;
    if (lane == 0) {
        warp_offsets[warp] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned running = 0;
        for (int slot = 0; slot < warp_count_slots; slot++) {
            unsigned count = warp_offsets[slot];
            warp_offsets[slot] = running;
            running += count;
        }
        warp_offsets[warp_count_slots] = running;
    }
    __syncthreads();

    __shared__ size_t block_base;
    if (threadIdx.x == 0) {
        block_base = warp_offsets[warp_count_slots] == 0 ? 0 : atomicAdd_size_t(length, warp_offsets[warp_count_slots]);
    }
    __syncthreads();

    if (keep) {
        size_t out = block_base + static_cast<size_t>(warp_offsets[warp] + lane_rank);
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
    if (idx >= candidate_length) {
        return;
    }

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

    if (dist_sq < cutoff2) {
        size_t out = atomicAdd_size_t(length, 1);
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
