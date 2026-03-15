// GPU kernels for Verlet neighbor list caching.
//
// Two kernels:
// 1. verlet_max_displacement: O(N) parallel check if any atom moved > skin/2
// 2. verlet_recompute_pairs: O(N_pairs) parallel recompute of vectors/distances
//    from cached topology and current positions

__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double dot3(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 frac_to_cart_v(const double3& frac, const double3 box[3]) {
    return make_double3(
        frac.x * box[0].x + frac.y * box[1].x + frac.z * box[2].x,
        frac.x * box[0].y + frac.y * box[1].y + frac.z * box[2].y,
        frac.x * box[0].z + frac.y * box[1].z + frac.z * box[2].z
    );
}

// Kernel 1: Check maximum displacement across all atoms.
// One thread per atom. If displacement^2 > half_skin_sq, atomically set flag.
__global__ void verlet_max_displacement(
    const double* __restrict__ current_positions,  // [n_points * 3]
    const double* __restrict__ ref_positions,       // [n_points * 3]
    size_t n_points,
    double half_skin_sq,
    int* __restrict__ rebuild_flag                  // [1], set to 1 if rebuild needed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    double dx = current_positions[idx * 3 + 0] - ref_positions[idx * 3 + 0];
    double dy = current_positions[idx * 3 + 1] - ref_positions[idx * 3 + 1];
    double dz = current_positions[idx * 3 + 2] - ref_positions[idx * 3 + 2];
    double disp_sq = dx * dx + dy * dy + dz * dz;

    if (disp_sq > half_skin_sq) {
        atomicExch(rebuild_flag, 1);
    }
}

// Kernel 2: Recompute distance vectors from cached pair topology.
// One thread per cached pair. Computes vector from current positions + shift,
// filters by cutoff^2, and writes to output arrays using atomic counter.
__global__ void verlet_recompute_pairs(
    const double* __restrict__ positions,      // [n_points * 3]
    const double* __restrict__ box,            // [9] row-major 3x3
    const size_t* __restrict__ cached_pairs_i, // [n_cached_pairs]
    const size_t* __restrict__ cached_pairs_j, // [n_cached_pairs]
    const int* __restrict__ cached_shifts,     // [n_cached_pairs * 3]
    size_t n_cached_pairs,
    double cutoff_sq,
    size_t* __restrict__ output_length,        // [1] atomic counter
    size_t* __restrict__ output_pairs,         // [max_pairs * 2]
    int* __restrict__ output_shifts,           // [max_pairs * 3] (or nullptr)
    double* __restrict__ output_distances,     // [max_pairs] (or nullptr)
    double* __restrict__ output_vectors,       // [max_pairs * 3] (or nullptr)
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    size_t max_pairs,
    int* __restrict__ overflow_flag            // [1]
) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_cached_pairs) return;

    size_t i = cached_pairs_i[k];
    size_t j = cached_pairs_j[k];
    int sa = cached_shifts[k * 3 + 0];
    int sb = cached_shifts[k * 3 + 1];
    int sc = cached_shifts[k * 3 + 2];

    // Load box rows
    double3 box_rows[3];
    box_rows[0] = make_double3(box[0], box[1], box[2]);
    box_rows[1] = make_double3(box[3], box[4], box[5]);
    box_rows[2] = make_double3(box[6], box[7], box[8]);

    // vector = positions[j] - positions[i] + shift @ box
    double3 pi = make_double3(
        positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]
    );
    double3 pj = make_double3(
        positions[j * 3 + 0], positions[j * 3 + 1], positions[j * 3 + 2]
    );

    double3 shift_frac = make_double3((double)sa, (double)sb, (double)sc);
    double3 shift_cart = frac_to_cart_v(shift_frac, box_rows);

    double3 vec = make_double3(
        pj.x - pi.x + shift_cart.x,
        pj.y - pi.y + shift_cart.y,
        pj.z - pi.z + shift_cart.z
    );

    double dist_sq = dot3(vec, vec);

    if (dist_sq < cutoff_sq) {
        // Use unsigned long long for atomicAdd on size_t
        size_t idx = static_cast<size_t>(atomicAdd(
            reinterpret_cast<unsigned long long*>(output_length),
            static_cast<unsigned long long>(1)
        ));

        if (idx + 1 > max_pairs) {
            atomicExch(overflow_flag, 1);
            return;
        }

        output_pairs[idx * 2 + 0] = i;
        output_pairs[idx * 2 + 1] = j;

        if (return_shifts) {
            output_shifts[idx * 3 + 0] = sa;
            output_shifts[idx * 3 + 1] = sb;
            output_shifts[idx * 3 + 2] = sc;
        }

        if (return_vectors) {
            output_vectors[idx * 3 + 0] = vec.x;
            output_vectors[idx * 3 + 1] = vec.y;
            output_vectors[idx * 3 + 2] = vec.z;
        }

        if (return_distances) {
            output_distances[idx] = sqrt(dist_sq);
        }
    }
}
