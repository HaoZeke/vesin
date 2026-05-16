#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

#define NOMINMAX
#include <gpulite/gpulite.hpp>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

// NVTX for profiling (optional, enabled if available)
#ifdef VESIN_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) \
    do {                \
    } while (0)
#define NVTX_POP() \
    do {           \
    } while (0)
#endif

const unsigned char CUDA_BRUTEFORCE_CODE[] = {
#include "generated/cuda_bruteforce.cu.inc"
};

const unsigned char CUDA_CELL_LIST_CODE[] = {
#include "generated/cuda_cell_list.cu.inc"
};

const unsigned char CUDA_SORT_PAIRS_CODE[] = {
#include "generated/cuda_sort_pairs.cu.inc"
};

const unsigned char CUDA_VERLET_CODE[] = {
#include "generated/cuda_verlet.cu.inc"
};

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t DEFAULT_MAX_CELLS = 8192;
// Minimum particles per cell target for good GPU utilization.
// Lower values create more cells and reduce per-cell neighbor work, which is
// beneficial on larger systems where more coarse grids become too dense.
static constexpr size_t MIN_PARTICLES_PER_CELL = 8;
static constexpr size_t CUDA_VERLET_COMPACT_MIN_POINTS = 1024;  // lowered so the efficient block-compact + sorted path is used even in medium-size profiling runs (real hotspot data)

// Helper functions for CPU-side vector math
static inline double cpu_dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double cpu_norm3(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static inline void cpu_cross3(const double* a, const double* b, double* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void cpu_invert_matrix(const double* m, double* inv) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

    double inv_det = 1.0 / det;

    inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}

/// CPU-side box check that avoids GPU kernel launch overhead
/// Returns: {is_valid, is_orthogonal}
/// Also fills box_diag_out[3] and inv_box_out[9] if provided
static std::pair<bool, bool> cpu_box_check(
    const double h_box[9],
    const bool h_periodic[3],
    double cutoff,
    double* box_diag_out, // [3] output, can be nullptr
    double* inv_box_out   // [9] output, can be nullptr
) {
    const double* a = &h_box[0];
    const double* b = &h_box[3];
    const double* c = &h_box[6];

    double a_norm = cpu_norm3(a);
    double b_norm = cpu_norm3(b);
    double c_norm = cpu_norm3(c);

    // Count periodic directions
    size_t n_periodic = 0;
    if (h_periodic[0]) {
        n_periodic++;
    }
    if (h_periodic[1]) {
        n_periodic++;
    }
    if (h_periodic[2]) {
        n_periodic++;
    }

    double ab_dot = cpu_dot3(a, b);
    double ac_dot = cpu_dot3(a, c);
    double bc_dot = cpu_dot3(b, c);

    double tol = 1e-6;
    // Treat fully non-periodic systems as orthogonal
    // Also treat systems with zero-norm vectors as orthogonal (degenerate case)
    bool is_orthogonal = (n_periodic == 0) ||
                         (a_norm < tol || b_norm < tol || c_norm < tol) ||
                         ((std::fabs(ab_dot) < tol * a_norm * b_norm) &&
                          (std::fabs(ac_dot) < tol * a_norm * c_norm) &&
                          (std::fabs(bc_dot) < tol * b_norm * c_norm));

    // Output box diagonal (lengths)
    if (box_diag_out != nullptr) {
        box_diag_out[0] = a_norm;
        box_diag_out[1] = b_norm;
        box_diag_out[2] = c_norm;
    }

    // Compute and output inverse box (needed for general PBC)
    if ((inv_box_out != nullptr) && !is_orthogonal) {
        cpu_invert_matrix(h_box, inv_box_out);
    }

    // Compute minimum dimension for cutoff check
    double min_dim = 1e30;
    if (is_orthogonal) {
        if (h_periodic[0]) {
            min_dim = a_norm;
        }
        if (h_periodic[1]) {
            min_dim = std::fmin(min_dim, b_norm);
        }
        if (h_periodic[2]) {
            min_dim = std::fmin(min_dim, c_norm);
        }
    } else {
        // General case: compute perpendicular distances
        double bc_cross[3];
        double ac_cross[3];
        double ab_cross[3];
        cpu_cross3(b, c, bc_cross);
        cpu_cross3(a, c, ac_cross);
        cpu_cross3(a, b, ab_cross);

        double bc_norm = cpu_norm3(bc_cross);
        double ac_norm = cpu_norm3(ac_cross);
        double ab_norm = cpu_norm3(ab_cross);

        double V = std::fabs(cpu_dot3(a, bc_cross));

        double d_a = V / bc_norm;
        double d_b = V / ac_norm;
        double d_c = V / ab_norm;

        if (h_periodic[0]) {
            min_dim = d_a;
        }
        if (h_periodic[1]) {
            min_dim = std::fmin(min_dim, d_b);
        }
        if (h_periodic[2]) {
            min_dim = std::fmin(min_dim, d_c);
        }
    }

    bool is_valid = (cutoff * 2.0 <= min_dim);
    return {is_valid, is_orthogonal};
}

static std::optional<cudaPointerAttributes> getPtrAttributes(const void* ptr) {
    if (ptr == nullptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        GPULITE_CUDART_CALL(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    } catch (const std::runtime_error& e) {
        return std::nullopt;
    }
}

static bool is_device_ptr(const std::optional<cudaPointerAttributes>& maybe_attr, const char* name) {
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        return (attr.type == cudaMemoryTypeDevice);
    } else {
        throw std::runtime_error(
            "failed to resolve attributes for pointer: " + std::string(name)
        );
    }
}

static int32_t get_device_id(const void* ptr) {
    if (ptr == nullptr) {
        return -1;
    }

    auto maybe_attr = getPtrAttributes(ptr);
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        if (attr.type != cudaMemoryTypeDevice) {
            return -1;
        }
        return attr.device;
    }
    return -1;
}

static void free_cell_list_buffers(CellListBuffers& cl) {
    GPULITE_CUDART_CALL(cudaFree(cl.cell_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.particle_shifts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_counts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_starts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_offsets));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_positions));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_shifts));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_cell_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.inv_box));
    GPULITE_CUDART_CALL(cudaFree(cl.n_cells));
    GPULITE_CUDART_CALL(cudaFree(cl.n_search));
    GPULITE_CUDART_CALL(cudaFree(cl.n_cells_total));
    GPULITE_CUDART_CALL(cudaFree(cl.bounding_min));
    GPULITE_CUDART_CALL(cudaFree(cl.bounding_max));

    cl = CellListBuffers();
}

static void free_sort_buffers(CudaNeighborListExtras& extras) {
    GPULITE_CUDART_CALL(cudaFree(extras.sort_pairs_tmp));
    GPULITE_CUDART_CALL(cudaFree(extras.sort_shifts_tmp));
    GPULITE_CUDART_CALL(cudaFree(extras.sort_distances_tmp));
    GPULITE_CUDART_CALL(cudaFree(extras.sort_vectors_tmp));

    extras.sort_pairs_tmp = nullptr;
    extras.sort_shifts_tmp = nullptr;
    extras.sort_distances_tmp = nullptr;
    extras.sort_vectors_tmp = nullptr;
    extras.sort_capacity = 0;
}

static size_t next_power_of_two(size_t value) {
    if (value <= 1) {
        return 1;
    }

    size_t power = 1;
    while (power < value) {
        power <<= 1;
    }

    return power;
}

static void ensure_sort_buffers(
    CudaNeighborListExtras& extras,
    size_t sort_capacity,
    bool return_shifts,
    bool return_distances,
    bool return_vectors
) {
    if (extras.sort_capacity < sort_capacity) {
        free_sort_buffers(extras);
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.sort_pairs_tmp, sizeof(size_t) * sort_capacity * 2));
        extras.sort_capacity = sort_capacity;
    }

    if (return_shifts && extras.sort_shifts_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.sort_shifts_tmp, sizeof(int32_t) * sort_capacity * 3));
    }

    if (return_distances && extras.sort_distances_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.sort_distances_tmp, sizeof(double) * sort_capacity));
    }

    if (return_vectors && extras.sort_vectors_tmp == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.sort_vectors_tmp, sizeof(double) * sort_capacity * 3));
    }
}

static void free_verlet_compact_buffers(CudaNeighborListExtras& extras) {
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_compact_candidate_pairs));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_compact_candidate_shifts));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_compact_overflow_flag));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_pairs_alt));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_shifts_alt));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_histogram));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_cursor));

    extras.verlet_compact_candidate_pairs = nullptr;
    extras.verlet_compact_candidate_shifts = nullptr;
    extras.verlet_compact_overflow_flag = nullptr;
    extras.verlet_radix_pairs_alt = nullptr;
    extras.verlet_radix_shifts_alt = nullptr;
    extras.verlet_radix_histogram = nullptr;
    extras.verlet_radix_cursor = nullptr;
    extras.verlet_radix_alt_capacity = 0;
    extras.verlet_candidate_length = 0;
    extras.verlet_compact_candidate_capacity = 0;
    extras.verlet_has_compact_candidates = false;
}

static void free_verlet_buffers(CudaNeighborListExtras& extras) {
    if (extras.verlet_candidates.device.type == VesinCUDA) {
        free_neighbors(extras.verlet_candidates);
    }

    extras.verlet_candidates = VesinNeighborList();
    free_verlet_compact_buffers(extras);
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_ref_positions));
    GPULITE_CUDART_CALL(cudaFree(extras.verlet_rebuild_flag));

    extras.verlet_ref_positions = nullptr;
    extras.verlet_rebuild_flag = nullptr;
    extras.verlet_ref_capacity = 0;
    extras.verlet_n_points = 0;
    extras.verlet_options = VesinOptions();
    extras.verlet_half_skin_sq = 0.0;
    extras.verlet_has_cache = false;
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    if (this->length_ptr != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->cell_check_ptr);
    }
    if (this->overflow_flag != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->overflow_flag);
    }
    if (this->box_diag != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->box_diag);
    }
    if (this->inv_box_brute != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->inv_box_brute);
    }
    try {
        free_cell_list_buffers(this->cell_list);
        free_sort_buffers(*this);
        free_verlet_buffers(*this);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing CUDA buffers: " << e.what() << std::endl;
    }
}

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (neighbors->opaque == nullptr) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        auto* test = static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

static void free_output_buffers(VesinNeighborList& neighbors, CudaNeighborListExtras& extras) {
    if ((neighbors.pairs != nullptr) && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.pairs));
    }
    if ((neighbors.shifts != nullptr) && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.shifts));
    }
    if ((neighbors.distances != nullptr) && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.distances));
    }
    if ((neighbors.vectors != nullptr) && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras.length_ptr));
    extras.length_ptr = nullptr;

    if (extras.pinned_length_ptr != nullptr) {
        GPULITE_CUDART_CALL(cudaFreeHost(extras.pinned_length_ptr));
        extras.pinned_length_ptr = nullptr;
    }

    GPULITE_CUDART_CALL(cudaFree(extras.cell_check_ptr));
    extras.cell_check_ptr = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras.box_diag));
    extras.box_diag = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras.inv_box_brute));
    extras.inv_box_brute = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras.overflow_flag));
    extras.overflow_flag = nullptr;

    free_cell_list_buffers(extras.cell_list);
    free_sort_buffers(extras);

    extras.capacity = 0;
    extras.max_pairs = 0;
}

static void reset(VesinNeighborList& neighbors) {
    auto* extras = vesin::cuda::get_cuda_extras(&neighbors);

    free_output_buffers(neighbors, *extras);
    free_verlet_buffers(*extras);

    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    int32_t curr_device = -1;
    int32_t device_id = -1;

    if (neighbors.pairs != nullptr) {
        GPULITE_CUDART_CALL(cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if ((device_id != -1) && curr_device != device_id) {
            GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if ((device_id != -1) && curr_device != device_id) {
        GPULITE_CUDART_CALL(cudaSetDevice(curr_device));
    }

    delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

void checkCuda() {
    std::string cuda_libname;
    std::string cudart_libname;
    std::string nvrtc_libname;
    std::string suggestion;
#if defined(__linux__)
    cuda_libname = "libcuda.so";
    cudart_libname = "libcudart.so(.*)";
    nvrtc_libname = "libnvrtc.so(.*)";
    suggestion = ("Try appending the directory containing this library to "
                  "your $LD_LIBRARY_PATH environment variable.");

#elif defined(_WIN32)
    cuda_libname = "nvcuda.dll";
    cudart_libname = "cudart64_*.dll";
    nvrtc_libname = "nvrtc64_*.dll";
    suggestion = ("Try adding the directory containing this library to your "
                  "system PATH, or making sure that CUDA_PATH is properly set "
                  "to your CUDA installation directory.");
#else
    cuda_libname = "cuda";
    cudart_libname = "cudart";
    nvrtc_libname = "nvrtc";
    suggestion = "Unsupported platform: unable to load CUDA libraries.";
#endif
    if (!gpulite::CUDADriver::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cuda_libname + ". " + suggestion
        );
    }

    if (!gpulite::CUDART::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cudart_libname + ". " + suggestion
        );
    }

    if (!gpulite::NVRTC::loaded()) {
        throw std::runtime_error(
            "Failed to load " + nvrtc_libname + ". " + suggestion
        );
    }
}

// Ensure cell list buffers are allocated with sufficient capacity
static void realloc_buffers_if_needed(
    CellListBuffers& cl,
    size_t n_points,
    size_t n_cells
) {
    if (cl.max_points < n_points) {
        // Free old point-related buffers
        GPULITE_CUDART_CALL(cudaFree(cl.cell_indices));
        GPULITE_CUDART_CALL(cudaFree(cl.particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_positions));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_cell_indices));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.particle_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_positions, sizeof(double) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_cell_indices, sizeof(int32_t) * n_points));
        cl.max_points = n_points;
    }

    if (cl.max_cells < n_cells) {
        // Free old cell-related buffers
        GPULITE_CUDART_CALL(cudaFree(cl.cell_counts));
        GPULITE_CUDART_CALL(cudaFree(cl.cell_starts));
        GPULITE_CUDART_CALL(cudaFree(cl.cell_offsets));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_counts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_starts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_offsets, sizeof(int32_t) * n_cells));
        cl.max_cells = n_cells;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (cl.inv_box == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.inv_box, sizeof(double) * 9));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_cells, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_search, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_cells_total, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.bounding_min, sizeof(double) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.bounding_max, sizeof(double) * 3));
    }
}

static void sort_pairs_if_needed(
    CudaNeighborListExtras& extras,
    gpulite::KernelFactory& factory,
    const std::string& cuda_sort_pairs_code,
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t* d_pair_indices,
    int32_t* d_shifts,
    double* d_distances,
    double* d_vectors
) {
    if (!options.sorted || neighbors.length <= 1) {
        return;
    }

    NVTX_PUSH("sort_pairs");
    size_t sort_capacity = next_power_of_two(neighbors.length);
    ensure_sort_buffers(
        extras,
        sort_capacity,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    auto* fill_kernel = factory.create(
        "sort_pairs_fill_buffers",
        cuda_sort_pairs_code,
        "cuda_sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    auto* bitonic_kernel = factory.create(
        "sort_pairs_bitonic_step",
        cuda_sort_pairs_code,
        "cuda_sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    auto* copy_back_kernel = factory.create(
        "sort_pairs_copy_back",
        cuda_sort_pairs_code,
        "cuda_sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    auto* d_pairs_tmp = extras.sort_pairs_tmp;
    auto* d_shifts_tmp = extras.sort_shifts_tmp;
    auto* d_distances_tmp = extras.sort_distances_tmp;
    auto* d_vectors_tmp = extras.sort_vectors_tmp;

    const size_t sort_threads = 256;
    const size_t sort_fill_blocks = (sort_capacity + sort_threads - 1) / sort_threads;

    std::vector<void*> fill_args = {
        static_cast<void*>(&d_pair_indices),
        static_cast<void*>(&d_shifts),
        static_cast<void*>(&d_distances),
        static_cast<void*>(&d_vectors),
        static_cast<void*>(&d_pairs_tmp),
        static_cast<void*>(&d_shifts_tmp),
        static_cast<void*>(&d_distances_tmp),
        static_cast<void*>(&d_vectors_tmp),
        static_cast<void*>(&neighbors.length),
        static_cast<void*>(&sort_capacity),
        static_cast<void*>(&options.return_shifts),
        static_cast<void*>(&options.return_distances),
        static_cast<void*>(&options.return_vectors),
    };
    fill_kernel->launch(
        dim3(std::max(sort_fill_blocks, static_cast<size_t>(1))),
        dim3(sort_threads),
        0,
        nullptr,
        fill_args,
        false
    );

    for (size_t k = 2; k <= sort_capacity; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            std::vector<void*> bitonic_args = {
                static_cast<void*>(&d_pairs_tmp),
                static_cast<void*>(&d_shifts_tmp),
                static_cast<void*>(&d_distances_tmp),
                static_cast<void*>(&d_vectors_tmp),
                static_cast<void*>(&sort_capacity),
                static_cast<void*>(&j),
                static_cast<void*>(&k),
                static_cast<void*>(&options.return_shifts),
                static_cast<void*>(&options.return_distances),
                static_cast<void*>(&options.return_vectors),
            };
            bitonic_kernel->launch(
                dim3(std::max(sort_fill_blocks, static_cast<size_t>(1))),
                dim3(sort_threads),
                0,
                nullptr,
                bitonic_args,
                false
            );
        }
    }

    const size_t copy_blocks = (neighbors.length + sort_threads - 1) / sort_threads;
    std::vector<void*> copy_back_args = {
        static_cast<void*>(&d_pair_indices),
        static_cast<void*>(&d_shifts),
        static_cast<void*>(&d_distances),
        static_cast<void*>(&d_vectors),
        static_cast<void*>(&d_pairs_tmp),
        static_cast<void*>(&d_shifts_tmp),
        static_cast<void*>(&d_distances_tmp),
        static_cast<void*>(&d_vectors_tmp),
        static_cast<void*>(&neighbors.length),
        static_cast<void*>(&options.return_shifts),
        static_cast<void*>(&options.return_distances),
        static_cast<void*>(&options.return_vectors),
    };
    copy_back_kernel->launch(
        dim3(std::max(copy_blocks, static_cast<size_t>(1))),
        dim3(sort_threads),
        0,
        nullptr,
        copy_back_args,
        false
    );

    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    NVTX_POP();
}

static bool verlet_options_changed(const CudaNeighborListExtras& extras, VesinOptions options) {
    return extras.verlet_options.cutoff != options.cutoff ||
           extras.verlet_options.skin != options.skin ||
           extras.verlet_options.full != options.full;
}

static bool verlet_box_changed(
    const CudaNeighborListExtras& extras,
    const double h_box[9],
    const bool h_periodic[3]
) {
    for (size_t i = 0; i < 9; i++) {
        if (std::abs(extras.verlet_ref_box[i] - h_box[i]) > 1e-12) {
            return true;
        }
    }

    for (size_t i = 0; i < 3; i++) {
        if (extras.verlet_ref_periodic[i] != h_periodic[i]) {
            return true;
        }
    }

    return false;
}

static void ensure_verlet_ref_buffers(CudaNeighborListExtras& extras, size_t n_points) {
    if (extras.verlet_ref_capacity < n_points) {
        GPULITE_CUDART_CALL(cudaFree(extras.verlet_ref_positions));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_ref_positions, sizeof(double) * n_points * 3));
        extras.verlet_ref_capacity = n_points;
    }

    if (extras.verlet_rebuild_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_rebuild_flag, sizeof(int32_t)));
    }
}

static void ensure_verlet_compact_buffers(CudaNeighborListExtras& extras, size_t candidate_length) {
    // Bitonic sort needs the buffer padded out to the next power of two so
    // every compare-exchange has a valid partner index. Allocate up front.
    auto needed = std::max(next_power_of_two(candidate_length), static_cast<size_t>(1));
    if (extras.verlet_compact_candidate_capacity < needed) {
        free_verlet_compact_buffers(extras);
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_compact_candidate_pairs, sizeof(uint32_t) * needed * 2));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_compact_candidate_shifts, sizeof(int32_t) * needed));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_compact_overflow_flag, sizeof(int32_t)));
        extras.verlet_compact_candidate_capacity = needed;
    }

    if (extras.verlet_compact_overflow_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_compact_overflow_flag, sizeof(int32_t)));
    }
}

static void compact_verlet_candidate_cache(
    CudaNeighborListExtras& extras,
    gpulite::KernelFactory& factory,
    const std::string& cuda_verlet_code,
    size_t n_points
) {
    auto candidate_length = extras.verlet_candidates.length;
    extras.verlet_candidate_length = candidate_length;
    extras.verlet_has_compact_candidates = false;

    if (candidate_length == 0 || n_points < CUDA_VERLET_COMPACT_MIN_POINTS ||
        n_points > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return;
    }

    ensure_verlet_compact_buffers(extras, candidate_length);
    extras.verlet_candidate_length = candidate_length;
    GPULITE_CUDART_CALL(cudaMemset(extras.verlet_compact_overflow_flag, 0, sizeof(int32_t)));

    auto* d_candidate_pairs = reinterpret_cast<size_t*>(extras.verlet_candidates.pairs);
    auto* d_candidate_shifts = reinterpret_cast<int32_t*>(extras.verlet_candidates.shifts);
    auto* d_compact_pairs = extras.verlet_compact_candidate_pairs;
    auto* d_compact_shifts = extras.verlet_compact_candidate_shifts;
    auto* d_overflow_flag = extras.verlet_compact_overflow_flag;

    auto* kernel = factory.create(
        "pack_verlet_candidates",
        cuda_verlet_code,
        "cuda_verlet.cu",
        {"-std=c++17", "-default-device"}
    );

    size_t threads = 256;
    size_t blocks = (candidate_length + threads - 1) / threads;
    std::vector<void*> args = {
        static_cast<void*>(&d_candidate_pairs),
        static_cast<void*>(&d_candidate_shifts),
        static_cast<void*>(&candidate_length),
        static_cast<void*>(&d_compact_pairs),
        static_cast<void*>(&d_compact_shifts),
        static_cast<void*>(&d_overflow_flag),
    };
    kernel->launch(
        dim3(std::max(blocks, static_cast<size_t>(1))),
        dim3(threads),
        0,
        nullptr,
        args,
        false
    );

    int32_t h_overflow = 0;
    GPULITE_CUDART_CALL(cudaMemcpy(
        &h_overflow,
        d_overflow_flag,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost
    ));

    if (h_overflow != 0) {
        return;
    }

    // Sort the compact Verlet candidates by the first particle index (i) so
    // the subsequent filter kernel gets good temporal locality on positions
    // gathers and the ri-reuse hits in
    // filter_verlet_compact_candidates_block actually fire. The earlier
    // host-side std::sort with two cudaMemcpy round trips regressed rebuild
    // ~10x at N=32k (see vissue vesin-3okr); replace with an in-place
    // bitonic sort that runs entirely on device.
    //
    // Compile-time toggle for profiling: pass -DVESIN_DISABLE_COMPACT_SORT
    // to skip the sort entirely. Empirically (see vissue vesin-3okr data)
    // disabling the sort keeps reuse perf within ~5% while saving the sort
    // launch chain on rebuild, but the gpu sort is also cheap enough that
    // the default keeps it on.
#ifndef VESIN_DISABLE_COMPACT_SORT
    if (candidate_length > 1) {
        // 4-pass 8-bit radix sort. Replaces the previous bitonic sort
        // chain (~1880 launches / rebuild at N=8192) with 12 launches
        // (3 per pass x 4 passes). Sort is NOT stable; the downstream
        // filter only requires i-grouping for ri reuse, not a specific
        // order among pairs sharing an i.
        //
        // The ping-pong alt buffer must be at least as big as the compact
        // buffer (verlet_compact_candidate_capacity). If a previous call
        // sized it smaller (because candidate_length grew on this system
        // or after a system-size change), reallocate.
        size_t alt_cap = extras.verlet_compact_candidate_capacity;
        if (extras.verlet_radix_pairs_alt == nullptr ||
            extras.verlet_radix_alt_capacity < alt_cap) {
            if (extras.verlet_radix_pairs_alt != nullptr) {
                GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_pairs_alt));
                extras.verlet_radix_pairs_alt = nullptr;
            }
            if (extras.verlet_radix_shifts_alt != nullptr) {
                GPULITE_CUDART_CALL(cudaFree(extras.verlet_radix_shifts_alt));
                extras.verlet_radix_shifts_alt = nullptr;
            }
            GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_radix_pairs_alt,
                                           sizeof(uint32_t) * alt_cap * 2));
            GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_radix_shifts_alt,
                                           sizeof(int32_t) * alt_cap));
            extras.verlet_radix_alt_capacity = alt_cap;
        }
        if (extras.verlet_radix_histogram == nullptr) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_radix_histogram,
                                           sizeof(int32_t) * 256));
        }
        if (extras.verlet_radix_cursor == nullptr) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.verlet_radix_cursor,
                                           sizeof(int32_t) * 256));
        }

        auto* hist_kernel = factory.create(
            "radix_histogram",
            cuda_verlet_code,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );
        auto* prefix_kernel = factory.create(
            "radix_prefix_sum_256",
            cuda_verlet_code,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );
        auto* scatter_kernel = factory.create(
            "radix_scatter",
            cuda_verlet_code,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );

        const size_t threads = 256;
        const size_t blocks = std::max(
            (candidate_length + threads - 1) / threads,
            static_cast<size_t>(1)
        );

        uint32_t* src_pairs = d_compact_pairs;
        int32_t* src_shifts = d_compact_shifts;
        uint32_t* dst_pairs = extras.verlet_radix_pairs_alt;
        int32_t* dst_shifts = extras.verlet_radix_shifts_alt;
        int32_t* d_histogram = extras.verlet_radix_histogram;
        int32_t* d_cursor = extras.verlet_radix_cursor;

        for (unsigned int pass = 0; pass < 4; ++pass) {
            unsigned int shift_bits = pass * 8u;

            GPULITE_CUDART_CALL(cudaMemset(d_histogram, 0, sizeof(int32_t) * 256));
            GPULITE_CUDART_CALL(cudaMemset(d_cursor, 0, sizeof(int32_t) * 256));

            std::vector<void*> hist_args = {
                static_cast<void*>(&src_pairs),
                static_cast<void*>(&candidate_length),
                static_cast<void*>(&shift_bits),
                static_cast<void*>(&d_histogram),
            };
            hist_kernel->launch(
                dim3(blocks), dim3(threads), 0, nullptr, hist_args, false
            );

            std::vector<void*> prefix_args = {
                static_cast<void*>(&d_histogram),
            };
            prefix_kernel->launch(
                dim3(1), dim3(256), 0, nullptr, prefix_args, false
            );

            std::vector<void*> scatter_args = {
                static_cast<void*>(&src_pairs),
                static_cast<void*>(&src_shifts),
                static_cast<void*>(&dst_pairs),
                static_cast<void*>(&dst_shifts),
                static_cast<void*>(&candidate_length),
                static_cast<void*>(&shift_bits),
                static_cast<void*>(&d_histogram),
                static_cast<void*>(&d_cursor),
            };
            scatter_kernel->launch(
                dim3(blocks), dim3(threads), 0, nullptr, scatter_args, false
            );

            // Ping-pong: next pass reads from the just-written dst.
            std::swap(src_pairs, dst_pairs);
            std::swap(src_shifts, dst_shifts);
        }

        // After 4 passes the final sorted result is in src_pairs / src_shifts
        // (the buffers were swapped after each pass, so what was the source
        // of the LAST pass now holds the output of the LAST pass after the
        // swap). If that is not the original d_compact_pairs, copy the
        // sorted data back so the filter kernel reads from the expected
        // location.
        if (src_pairs != d_compact_pairs) {
            GPULITE_CUDART_CALL(cudaMemcpy(
                d_compact_pairs, src_pairs,
                sizeof(uint32_t) * candidate_length * 2,
                cudaMemcpyDeviceToDevice
            ));
            GPULITE_CUDART_CALL(cudaMemcpy(
                d_compact_shifts, src_shifts,
                sizeof(int32_t) * candidate_length,
                cudaMemcpyDeviceToDevice
            ));
        }
    }
#endif // VESIN_DISABLE_COMPACT_SORT

    extras.verlet_has_compact_candidates = true;
    if (extras.verlet_candidates.device.type == VesinCUDA) {
        free_neighbors(extras.verlet_candidates);
        extras.verlet_candidates = VesinNeighborList();
    }
}

static bool verlet_needs_rebuild(
    CudaNeighborListExtras& extras,
    gpulite::KernelFactory& factory,
    const std::string& cuda_verlet_code,
    const double* d_positions,
    size_t n_points,
    const double h_box[9],
    const bool h_periodic[3]
) {
    if (!extras.verlet_has_cache) {
        return true;
    }

    if (extras.verlet_n_points != n_points) {
        return true;
    }

    if (verlet_box_changed(extras, h_box, h_periodic)) {
        return true;
    }

    GPULITE_CUDART_CALL(cudaMemset(extras.verlet_rebuild_flag, 0, sizeof(int32_t)));

    auto* kernel = factory.create(
        "check_verlet_displacements",
        cuda_verlet_code,
        "cuda_verlet.cu",
        {"-std=c++17", "-default-device"}
    );

    // check_verlet_displacements fires on every reuse call. ncu showed
    // it at ~14% achieved occupancy with threads=256 (grid size 32 at
    // N=8192) -- too few blocks to cover all 66 SMs on the RTX 4070 Ti
    // SUPER. Dropping threads to 64 quadrupled the grid but achieved
    // occupancy actually fell to ~7% (the kernel is hardware-limited at
    // 24 blocks/SM x 2 warps = 48 warps/SM, so the SMs that DO get
    // blocks are already saturated and the new blocks don't help).
    // Kernel duration is 3 us either way -- not worth optimising
    // further at this size. Keeping the original 256.
    size_t threads = 256;
    size_t blocks = (n_points + threads - 1) / threads;
    auto* d_ref_positions = extras.verlet_ref_positions;
    auto* d_rebuild_flag = extras.verlet_rebuild_flag;
    std::vector<void*> args = {
        static_cast<void*>(&d_positions),
        static_cast<void*>(&d_ref_positions),
        static_cast<void*>(&n_points),
        static_cast<void*>(&extras.verlet_half_skin_sq),
        static_cast<void*>(&d_rebuild_flag),
    };
    kernel->launch(
        dim3(std::max(blocks, static_cast<size_t>(1))),
        dim3(threads),
        0,
        nullptr,
        args,
        false
    );

    int32_t h_rebuild = 0;
    GPULITE_CUDART_CALL(cudaMemcpy(&h_rebuild, d_rebuild_flag, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return h_rebuild != 0;
}

static void rebuild_verlet_cache(
    CudaNeighborListExtras& extras,
    gpulite::KernelFactory& factory,
    const std::string& cuda_verlet_code,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    int32_t device_id,
    VesinOptions options,
    const double h_box[9],
    const bool h_periodic[3]
) {
    if (extras.verlet_candidates.device.type == VesinCUDA) {
        free_neighbors(extras.verlet_candidates);
    }

    extras.verlet_candidates = VesinNeighborList();
    extras.verlet_candidates.device = {VesinCUDA, device_id};

    auto build_options = VesinOptions();
    build_options.cutoff = options.cutoff + options.skin;
    build_options.full = options.full;
    build_options.sorted = false;
    build_options.algorithm = VesinCellList;
    build_options.return_shifts = true;
    build_options.return_distances = false;
    build_options.return_vectors = false;
    build_options.skin = 0.0;

    vesin::cuda::neighbors(
        points,
        n_points,
        box,
        periodic,
        build_options,
        extras.verlet_candidates
    );
    compact_verlet_candidate_cache(extras, factory, cuda_verlet_code, n_points);

    ensure_verlet_ref_buffers(extras, n_points);
    GPULITE_CUDART_CALL(cudaMemcpy(
        extras.verlet_ref_positions,
        points,
        sizeof(double) * n_points * 3,
        cudaMemcpyDeviceToDevice
    ));

    std::memcpy(extras.verlet_ref_box, h_box, sizeof(double) * 9);
    std::memcpy(extras.verlet_ref_periodic, h_periodic, sizeof(bool) * 3);
    extras.verlet_n_points = n_points;
    extras.verlet_options = options;
    extras.verlet_half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
    extras.verlet_has_cache = true;
}

static void ensure_verlet_output_buffers(
    VesinNeighborList& neighbors,
    CudaNeighborListExtras& extras,
    size_t n_points,
    size_t max_pairs,
    VesinOptions options
) {
    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras.max_pairs < max_pairs || extras.length_ptr == nullptr || extras.overflow_flag == nullptr ||
        extras.pinned_length_ptr == nullptr || extras.cell_check_ptr == nullptr || missing_requested_output) {
        free_output_buffers(neighbors, extras);

        extras.max_pairs = std::max(max_pairs, static_cast<size_t>(1));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * extras.max_pairs * 2));

        if (options.return_shifts) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * extras.max_pairs * 3));
        }

        if (options.return_distances) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * extras.max_pairs));
        }

        if (options.return_vectors) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * extras.max_pairs * 3));
        }

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.length_ptr, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras.pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.cell_check_ptr, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras.overflow_flag, sizeof(int32_t)));
        extras.capacity = n_points;
    }

    GPULITE_CUDART_CALL(cudaMemset(extras.length_ptr, 0, sizeof(size_t)));
    GPULITE_CUDART_CALL(cudaMemset(extras.cell_check_ptr, 0, sizeof(int32_t)));
    GPULITE_CUDART_CALL(cudaMemset(extras.overflow_flag, 0, sizeof(int32_t)));
}

static void recompute_verlet_neighbors(
    CudaNeighborListExtras& extras,
    gpulite::KernelFactory& factory,
    const std::string& cuda_verlet_code,
    const std::string& cuda_sort_pairs_code,
    const double* d_positions,
    const double* d_box,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    size_t candidate_length = extras.verlet_candidate_length;
    ensure_verlet_output_buffers(neighbors, extras, extras.verlet_n_points, candidate_length, options);

    auto* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    auto* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    auto* d_distances = neighbors.distances;
    auto* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    auto* d_pair_counter = extras.length_ptr;
    auto* d_overflow_flag = extras.overflow_flag;
    size_t max_pairs = extras.max_pairs;

    size_t threads = 256;
    size_t blocks = (candidate_length + threads - 1) / threads;
    if (extras.verlet_has_compact_candidates) {
        auto* d_candidate_pairs = extras.verlet_compact_candidate_pairs;
        auto* d_candidate_shifts = extras.verlet_compact_candidate_shifts;

        auto* kernel = factory.create(
            "filter_verlet_compact_candidates_block",
            cuda_verlet_code,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );
        size_t shared_mem = sizeof(uint32_t) * ((threads + 31) / 32 + 1);

        std::vector<void*> args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_candidate_pairs),
            static_cast<void*>(&d_candidate_shifts),
            static_cast<void*>(&candidate_length),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&d_pair_counter),
            static_cast<void*>(&d_pair_indices),
            static_cast<void*>(&d_shifts),
            static_cast<void*>(&d_distances),
            static_cast<void*>(&d_vectors),
            static_cast<void*>(&options.return_shifts),
            static_cast<void*>(&options.return_distances),
            static_cast<void*>(&options.return_vectors),
            static_cast<void*>(&max_pairs),
            static_cast<void*>(&d_overflow_flag),
        };

        kernel->launch(
            dim3(std::max(blocks, static_cast<size_t>(1))),
            dim3(threads),
            shared_mem,
            nullptr,
            args,
            false
        );
    } else {
        auto* d_candidate_pairs = reinterpret_cast<size_t*>(extras.verlet_candidates.pairs);
        auto* d_candidate_shifts = reinterpret_cast<int32_t*>(extras.verlet_candidates.shifts);

        auto* kernel = factory.create(
            "filter_verlet_candidates",
            cuda_verlet_code,
            "cuda_verlet.cu",
            {"-std=c++17", "-default-device"}
        );

        std::vector<void*> args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_candidate_pairs),
            static_cast<void*>(&d_candidate_shifts),
            static_cast<void*>(&candidate_length),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&d_pair_counter),
            static_cast<void*>(&d_pair_indices),
            static_cast<void*>(&d_shifts),
            static_cast<void*>(&d_distances),
            static_cast<void*>(&d_vectors),
            static_cast<void*>(&options.return_shifts),
            static_cast<void*>(&options.return_distances),
            static_cast<void*>(&options.return_vectors),
            static_cast<void*>(&max_pairs),
            static_cast<void*>(&d_overflow_flag),
        };

        kernel->launch(
            dim3(std::max(blocks, static_cast<size_t>(1))),
            dim3(threads),
            0,
            nullptr,
            args,
            false
        );
    }

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        extras.pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    int h_overflow_flag = 0;
    GPULITE_CUDART_CALL(cudaMemcpy(
        &h_overflow_flag,
        d_overflow_flag,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (h_overflow_flag != 0) {
        throw std::runtime_error("The CUDA Verlet output exceeds the cached candidate capacity");
    }

    neighbors.length = *extras.pinned_length_ptr;
    sort_pairs_if_needed(
        extras,
        factory,
        cuda_sort_pairs_code,
        options,
        neighbors,
        d_pair_indices,
        d_shifts,
        d_distances,
        d_vectors
    );
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(neighbors.device.type == VesinCUDA);

    // Check if CUDA is available
    checkCuda();

    // check that all pointers are are device pointers
    if (!is_device_ptr(getPtrAttributes(points), "points")) {
        throw std::runtime_error("`points` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(getPtrAttributes(box), "box")) {
        throw std::runtime_error("`box` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(getPtrAttributes(periodic), "periodic")) {
        throw std::runtime_error("`periodic` pointer is not allocated on a CUDA device");
    }

    auto device_id = get_device_id(points);
    if (device_id != get_device_id(box)) {
        throw std::runtime_error("`points` and `box` do not exist on the same device");
    }

    if (device_id != get_device_id(periodic)) {
        throw std::runtime_error("`points` and `periodic` do not exist on the same device");
    }

    if (device_id != neighbors.device.device_id) {
        throw std::runtime_error("`points`, `box` and `periodic` device differs from input neighbors device_id");
    }

    auto* extras = vesin::cuda::get_cuda_extras(&neighbors);
    size_t max_pairs_per_point = std::max(
        static_cast<size_t>(VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT),
        static_cast<size_t>(std::ceil(std::pow(options.cutoff, 3)))
    );

    double h_box[9];
    bool h_periodic[3];
    GPULITE_CUDART_CALL(cudaMemcpy(h_box, box, sizeof(double) * 9, cudaMemcpyDeviceToHost));
    GPULITE_CUDART_CALL(cudaMemcpy(h_periodic, periodic, sizeof(bool) * 3, cudaMemcpyDeviceToHost));

    if (extras->allocated_device_id != device_id) {
        // first switch to previous device
        if (extras->allocated_device_id >= 0) {
            GPULITE_CUDART_CALL(cudaSetDevice(extras->allocated_device_id));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        extras->allocated_device_id = device_id;
    }

    auto& factory = gpulite::KernelFactory::instance(device_id);

    auto cuda_bruteforce_code = std::string(reinterpret_cast<const char*>(CUDA_BRUTEFORCE_CODE), sizeof(CUDA_BRUTEFORCE_CODE));
    auto cuda_cell_list_code = std::string(reinterpret_cast<const char*>(CUDA_CELL_LIST_CODE), sizeof(CUDA_CELL_LIST_CODE));
    auto cuda_sort_pairs_code = std::string(reinterpret_cast<const char*>(CUDA_SORT_PAIRS_CODE), sizeof(CUDA_SORT_PAIRS_CODE));
    auto cuda_verlet_code = std::string(reinterpret_cast<const char*>(CUDA_VERLET_CODE), sizeof(CUDA_VERLET_CODE));

    if (options.skin > 0.0) {
        if (verlet_options_changed(*extras, options)) {
            free_verlet_buffers(*extras);
        }

        if (verlet_needs_rebuild(
                *extras,
                factory,
                cuda_verlet_code,
                reinterpret_cast<const double*>(points),
                n_points,
                h_box,
                h_periodic
            )) {
            rebuild_verlet_cache(
                *extras,
                factory,
                cuda_verlet_code,
                points,
                n_points,
                box,
                periodic,
                device_id,
                options,
                h_box,
                h_periodic
            );
        }

        recompute_verlet_neighbors(
            *extras,
            factory,
            cuda_verlet_code,
            cuda_sort_pairs_code,
            reinterpret_cast<const double*>(points),
            reinterpret_cast<const double*>(box),
            options,
            neighbors
        );
        return;
    }

    free_verlet_buffers(*extras);

    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras->capacity >= n_points && (extras->length_ptr != nullptr) && (extras->cell_check_ptr != nullptr) &&
        (extras->overflow_flag != nullptr) && !missing_requested_output) {
        GPULITE_CUDART_CALL(cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));
    } else {
        auto saved_device = extras->allocated_device_id;
        reset(neighbors);
        extras->allocated_device_id = saved_device;

        auto* env_max_pairs = std::getenv("VESIN_CUDA_MAX_PAIRS_PER_POINT");
        if (env_max_pairs != nullptr) {
            auto length = std::strlen(env_max_pairs);
            char* end = nullptr;
            errno = 0;
            auto parsed_max_pairs_per_point = std::strtoll(env_max_pairs, &end, 10);
            if (errno != 0 || end != env_max_pairs + length || parsed_max_pairs_per_point <= 0) {
                throw std::runtime_error(
                    "Invalid value for VESIN_CUDA_MAX_PAIRS_PER_POINT: '" +
                    std::string(env_max_pairs) + "'"
                );
            }
            max_pairs_per_point = static_cast<size_t>(parsed_max_pairs_per_point);
        }

        extras->max_pairs = n_points * max_pairs_per_point;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * extras->max_pairs * 2));

        if (options.return_shifts) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * extras->max_pairs * 3));
        }

        if (options.return_distances) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * extras->max_pairs));
        }

        if (options.return_vectors) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * extras->max_pairs * 3));
        }

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        // Pinned host memory for async D2H copy
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->overflow_flag, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));

        extras->capacity = static_cast<size_t>(1.2 * static_cast<double>(n_points));
    }

    const auto* d_positions = reinterpret_cast<const double*>(points);
    const auto* d_box = reinterpret_cast<const double*>(box);
    const auto* d_periodic = periodic;

    auto* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    auto* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    auto* d_distances = neighbors.distances;
    auto* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    auto* d_pair_counter = extras->length_ptr;
    auto* d_cell_check = extras->cell_check_ptr;
    auto* d_overflow_flag = extras->overflow_flag;
    size_t max_pairs = extras->max_pairs;

    if (extras->box_diag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->box_diag, sizeof(double) * 3));
    }
    if (extras->inv_box_brute == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->inv_box_brute, sizeof(double) * 9));
    }

    auto* box_check_kernel = factory.create(
        "mic_box_check",
        cuda_bruteforce_code,
        "cuda_bruteforce.cu",
        {"-std=c++17", "-default-device"}
    );

    double* d_box_diag = extras->box_diag;
    double* d_inv_box_brute = extras->inv_box_brute;
    std::vector<void*> box_check_args = {
        static_cast<void*>(&d_box),
        static_cast<void*>(&d_periodic),
        static_cast<void*>(&options.cutoff),
        static_cast<void*>(&d_cell_check),
        static_cast<void*>(&d_box_diag),
        static_cast<void*>(&d_inv_box_brute),
    };

    box_check_kernel->launch(dim3(1), dim3(32), 0, nullptr, box_check_args, false);

    int32_t h_cell_check = 1;
    GPULITE_CUDART_CALL(cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool box_check_error = (h_cell_check & 1) != 0;
    bool is_orthogonal = (h_cell_check & 2) != 0;

    // Get box dimensions for auto algorithm selection
    double h_box_diag[3];
    GPULITE_CUDART_CALL(cudaMemcpy(h_box_diag, d_box_diag, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    double min_box_dim = std::min({h_box_diag[0], h_box_diag[1], h_box_diag[2]});
    bool cutoff_requires_cell_list = options.cutoff > min_box_dim / 2.0;

    bool use_cell_list;
    switch (options.algorithm) {
    case VesinBruteForce:
        if (box_check_error) {
            throw std::runtime_error("Invalid cutoff: too large for box dimensions");
        }
        use_cell_list = false;
        break;
    case VesinCellList:
        use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        // Use cell list if cutoff > half box size, or for large/non-orthogonal systems
        use_cell_list = cutoff_requires_cell_list || !is_orthogonal || n_points >= 5000;
        break;
    }

    if (use_cell_list) {
        NVTX_PUSH("cell_list_total");

        // Compute effective max cells based on minimum particles per cell target
        // This ensures we have enough work per cell for good GPU utilization
        size_t max_cells_from_particles = std::max(n_points / MIN_PARTICLES_PER_CELL, static_cast<size_t>(1));
        size_t max_cells = std::min(DEFAULT_MAX_CELLS, max_cells_from_particles);

        NVTX_PUSH("ensure_buffers");
        realloc_buffers_if_needed(extras->cell_list, n_points, max_cells);
        NVTX_POP();
        auto& cl = extras->cell_list;

        size_t THREADS_PER_BLOCK = 256;
        size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        NVTX_PUSH("kernel0_bounding_box");
        auto* bounding_kernel = factory.create(
            "compute_bounding_box",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> bounding_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        // the 256 here must match the size of shared memory allocated inside the code,
        // if you update one please update the other.
        bounding_kernel->launch(dim3(1), dim3(256), 0, nullptr, bounding_args, false);
        NVTX_POP();

        NVTX_PUSH("kernel1_grid_params");
        auto* grid_kernel = factory.create(
            "compute_cell_grid_params",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> grid_args = {
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&max_cells),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&cl.n_search),
            static_cast<void*>(&cl.n_cells_total),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        grid_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_args, false);
        NVTX_POP();

        NVTX_PUSH("memset_cell_counts_starts");
        GPULITE_CUDART_CALL(cudaMemset(cl.cell_counts, 0, sizeof(int32_t) * max_cells));
        GPULITE_CUDART_CALL(cudaMemset(cl.cell_starts, 0, sizeof(int32_t) * max_cells));
        NVTX_POP();

        NVTX_PUSH("kernel1_assign_cells");
        auto* assign_kernel = factory.create(
            "assign_cell_indices",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> assign_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&cl.particle_shifts),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        assign_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, assign_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel2_count_particles");
        auto* count_kernel = factory.create(
            "count_particles_per_cell",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> count_args = {
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.cell_counts),
        };
        count_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, count_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel3_prefix_sum");
        auto* prefix_kernel = factory.create(
            "prefix_sum_cells",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> prefix_args = {
            static_cast<void*>(&cl.cell_counts),
            static_cast<void*>(&cl.cell_starts),
            static_cast<void*>(&cl.n_cells_total),
        };
        size_t prefix_threads = 256;
        size_t shared_mem = sizeof(int32_t) * prefix_threads;
        prefix_kernel->launch(
            dim3(1), dim3(prefix_threads), shared_mem, nullptr, prefix_args, false
        );
        NVTX_POP();

        NVTX_PUSH("memcpy_cell_offsets");
        GPULITE_CUDART_CALL(cudaMemcpy(
            cl.cell_offsets, cl.cell_starts, sizeof(int32_t) * max_cells, cudaMemcpyDeviceToDevice
        ));
        NVTX_POP();

        NVTX_PUSH("kernel4_scatter");
        auto* scatter_kernel = factory.create(
            "scatter_particles",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> scatter_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&cl.particle_shifts),
            static_cast<void*>(&cl.cell_offsets),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.sorted_positions),
            static_cast<void*>(&cl.sorted_indices),
            static_cast<void*>(&cl.sorted_shifts),
            static_cast<void*>(&cl.sorted_cell_indices),
        };
        scatter_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, scatter_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel5_find_neighbors");
        auto* find_kernel = factory.create(
            "find_neighbors_optimized",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> find_args = {
            static_cast<void*>(&cl.sorted_positions),
            static_cast<void*>(&cl.sorted_indices),
            static_cast<void*>(&cl.sorted_shifts),
            static_cast<void*>(&cl.sorted_cell_indices),
            static_cast<void*>(&cl.cell_starts),
            static_cast<void*>(&cl.cell_counts),
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&cl.n_search),
            static_cast<void*>(&n_points),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&options.full),
            static_cast<void*>(&d_pair_counter),
            static_cast<void*>(&d_pair_indices),
            static_cast<void*>(&d_shifts),
            static_cast<void*>(&d_distances),
            static_cast<void*>(&d_vectors),
            static_cast<void*>(&options.return_shifts),
            static_cast<void*>(&options.return_distances),
            static_cast<void*>(&options.return_vectors),
            static_cast<void*>(&max_pairs),
            static_cast<void*>(&d_overflow_flag)
        };
        size_t THREADS_PER_PARTICLE = 8;
        size_t particles_per_block = THREADS_PER_BLOCK / THREADS_PER_PARTICLE;
        size_t num_blocks_find = (n_points + particles_per_block - 1) / particles_per_block;
        find_kernel->launch(
            dim3(num_blocks_find), dim3(THREADS_PER_BLOCK), 0, nullptr, find_args, false
        );
        NVTX_POP();

        NVTX_POP(); // cell_list_total
    }

    if (!use_cell_list) {
        NVTX_PUSH("brute_force_total");

        size_t THREADS_PER_BLOCK = 128;
        double cutoff2 = options.cutoff * options.cutoff;

        size_t num_half_pairs = n_points * (n_points - 1) / 2;

        if (is_orthogonal) {
            if (options.full) {
                NVTX_PUSH("brute_force_full_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_full_orthogonal",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box_diag),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_half_orthogonal",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box_diag),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            }
        } else {
            if (options.full) {
                NVTX_PUSH("brute_force_full_general");
                auto* kernel = factory.create(
                    "brute_force_full_general",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box),
                    static_cast<void*>(&d_inv_box_brute),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_general");
                auto* kernel = factory.create(
                    "brute_force_half_general",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box),
                    static_cast<void*>(&d_inv_box_brute),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            }
        }

        NVTX_POP(); // brute_force_total
    }

    NVTX_PUSH("async_copy_and_sync");

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));

    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    // Check for overflow
    int h_overflow_flag = 0;
    GPULITE_CUDART_CALL(cudaMemcpy(
        &h_overflow_flag,
        d_overflow_flag,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (h_overflow_flag != 0) {
        throw std::runtime_error(
            "The number of neighbor pairs exceeds the maximum capacity of " +
            std::to_string(max_pairs) + " (max_pairs_per_point=" +
            std::to_string(max_pairs_per_point) + "; n_points=" +
            std::to_string(n_points) + "). " +
            "Consider reducing the cutoff distance, or explicitly setting " +
            "VESIN_CUDA_MAX_PAIRS_PER_POINT as an environment variable."
        );
    }

    neighbors.length = *extras->pinned_length_ptr;

    NVTX_POP();

    sort_pairs_if_needed(
        *extras,
        factory,
        cuda_sort_pairs_code,
        options,
        neighbors,
        d_pair_indices,
        d_shifts,
        d_distances,
        d_vectors
    );
}
