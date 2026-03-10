#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
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

static const char* CUDA_BRUTEFORCE_CODE =
#include "generated/cuda_bruteforce.cu"
    ;

static const char* CUDA_CELL_LIST_CODE =
#include "generated/cuda_cell_list.cu"
    ;

static const char* CUDA_CLUSTER_PAIR_CODE =
#include "generated/cuda_cluster_pair.cu"
    ;

static const char* CUDA_CLUSTER_GRID_CODE =
#include "generated/cuda_cluster_grid.cu"
    ;

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t MAX_CELLS = 8192;

/// GPU cluster size (maps to warp sublanes: 4 clusters of 8 = 32 = 1 warp)
static constexpr int32_t CLUSTER_SIZE_GPU = 8;

/// Threshold for switching to cluster-pair on GPU (same as CPU threshold)
static constexpr size_t GPU_CLUSTER_PAIR_THRESHOLD = 256;
// Minimum particles per cell target for good GPU utilization
// Higher values = fewer cells = more work per block but larger search range
static constexpr size_t MIN_PARTICLES_PER_CELL = 128;

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
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaPointerGetAttributes(&attr, ptr));
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
    CUDART_INSTANCE.cudaFree(cl.cell_indices);
    CUDART_INSTANCE.cudaFree(cl.particle_shifts);
    CUDART_INSTANCE.cudaFree(cl.cell_counts);
    CUDART_INSTANCE.cudaFree(cl.cell_starts);
    CUDART_INSTANCE.cudaFree(cl.cell_offsets);
    CUDART_INSTANCE.cudaFree(cl.sorted_positions);
    CUDART_INSTANCE.cudaFree(cl.sorted_indices);
    CUDART_INSTANCE.cudaFree(cl.sorted_shifts);
    CUDART_INSTANCE.cudaFree(cl.sorted_cell_indices);
    CUDART_INSTANCE.cudaFree(cl.inv_box);
    CUDART_INSTANCE.cudaFree(cl.n_cells);
    CUDART_INSTANCE.cudaFree(cl.n_search);
    CUDART_INSTANCE.cudaFree(cl.n_cells_total);

    cl = CellListBuffers();
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    if (this->length_ptr != nullptr) {
        CUDART_INSTANCE.cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cell_check_ptr);
    }
    if (this->overflow_flag != nullptr) {
        CUDART_INSTANCE.cudaFree(this->overflow_flag);
    }
    if (this->box_diag != nullptr) {
        CUDART_INSTANCE.cudaFree(this->box_diag);
    }
    if (this->inv_box_brute != nullptr) {
        CUDART_INSTANCE.cudaFree(this->inv_box_brute);
    }
    free_cell_list_buffers(this->cell_list);

    // Free cluster-pair buffers
    if (this->cluster_pair.cluster_atom_indices != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cluster_pair.cluster_atom_indices);
    }
    if (this->cluster_pair.cluster_n_atoms != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cluster_pair.cluster_n_atoms);
    }
    if (this->cluster_pair.cluster_bb_lower != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cluster_pair.cluster_bb_lower);
    }
    if (this->cluster_pair.cluster_bb_upper != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cluster_pair.cluster_bb_upper);
    }
    if (this->cluster_pair.cell_offsets != nullptr) {
        CUDART_INSTANCE.cudaFree(this->cluster_pair.cell_offsets);
    }

    // Free GPU-native grid builder buffers
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_cell_indices);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_atom_frac_z);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_cell_atom_counts);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_cell_atom_starts);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_cell_scatter_offsets);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_sorted_positions);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_sorted_atom_indices);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_sorted_frac_z);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_n_clusters_total);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_grid_inv_box);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_grid_n_cells);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_grid_n_search);
    CUDART_INSTANCE.cudaFree(this->cluster_pair.d_grid_n_cells_total);
}

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (neighbors->opaque == nullptr) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        auto* test = static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

static void reset(VesinNeighborList& neighbors) {
    auto* extras = vesin::cuda::get_cuda_extras(&neighbors);

    if ((neighbors.pairs != nullptr) && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.pairs));
    }
    if ((neighbors.shifts != nullptr) && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.shifts));
    }
    if ((neighbors.distances != nullptr) && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.distances));
    }
    if ((neighbors.vectors != nullptr) && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    extras->length_ptr = nullptr;

    // Free pinned memory if allocated
    if (extras->pinned_length_ptr != nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFreeHost(extras->pinned_length_ptr));
        extras->pinned_length_ptr = nullptr;
    }

    // Free brute force buffers
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(extras->box_diag));
    extras->box_diag = nullptr;

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(extras->inv_box_brute));
    extras->inv_box_brute = nullptr;

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(extras->overflow_flag));
    extras->overflow_flag = nullptr;

    free_cell_list_buffers(extras->cell_list);

    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    int32_t curr_device = -1;
    int32_t device_id = -1;

    if (neighbors.pairs != nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if ((device_id != -1) && curr_device != device_id) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if ((device_id != -1) && curr_device != device_id) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(curr_device));
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
    if (!CUDA_DRIVER_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load " + cuda_libname + ". " + suggestion
        );
    }

    if (!CUDART_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load " + cudart_libname + ". " + suggestion
        );
    }

    if (!NVRTC_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load " + nvrtc_libname + ". " + suggestion
        );
    }
}

// Ensure cell list buffers are allocated with sufficient capacity
static void ensure_cell_list_buffers(
    CellListBuffers& cl,
    size_t n_points,
    size_t n_cells_total
) {
    bool need_realloc_points = (cl.max_points < n_points);
    bool need_realloc_cells = (cl.max_cells < n_cells_total);

    if (need_realloc_points) {
        // Free old point-related buffers
        CUDART_INSTANCE.cudaFree(cl.cell_indices);
        CUDART_INSTANCE.cudaFree(cl.particle_shifts);
        CUDART_INSTANCE.cudaFree(cl.sorted_positions);
        CUDART_INSTANCE.cudaFree(cl.sorted_indices);
        CUDART_INSTANCE.cudaFree(cl.sorted_shifts);
        CUDART_INSTANCE.cudaFree(cl.sorted_cell_indices);

        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_points));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_indices, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.particle_shifts, sizeof(int32_t) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_positions, sizeof(double) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_indices, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_shifts, sizeof(int32_t) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_cell_indices, sizeof(int32_t) * new_max));
        cl.max_points = new_max;
    }

    if (need_realloc_cells) {
        // Free old cell-related buffers
        CUDART_INSTANCE.cudaFree(cl.cell_counts);
        CUDART_INSTANCE.cudaFree(cl.cell_starts);
        CUDART_INSTANCE.cudaFree(cl.cell_offsets);

        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_cells_total));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_counts, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_starts, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_offsets, sizeof(int32_t) * new_max));
        cl.max_cells = new_max;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (cl.inv_box == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.inv_box, sizeof(double) * 9));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_cells, sizeof(int32_t) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_search, sizeof(int32_t) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_cells_total, sizeof(int32_t)));
    }
}

/// Ensure cluster grid builder GPU buffers are allocated with sufficient capacity.
static void ensure_cluster_grid_buffers(
    CudaNeighborListExtras::ClusterPairBuffers& cp,
    size_t n_points,
    size_t n_cells_total
) {
    bool need_realloc_points = (cp.grid_max_points < n_points);
    bool need_realloc_cells = (cp.grid_max_cells < n_cells_total);

    if (need_realloc_points) {
        CUDART_INSTANCE.cudaFree(cp.d_cell_indices);
        CUDART_INSTANCE.cudaFree(cp.d_atom_frac_z);
        CUDART_INSTANCE.cudaFree(cp.d_sorted_positions);
        CUDART_INSTANCE.cudaFree(cp.d_sorted_atom_indices);
        CUDART_INSTANCE.cudaFree(cp.d_sorted_frac_z);

        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_points)) + 16;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_cell_indices, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_atom_frac_z, sizeof(float) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_sorted_positions, sizeof(double) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_sorted_atom_indices, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_sorted_frac_z, sizeof(float) * new_max));
        cp.grid_max_points = new_max;
    }

    if (need_realloc_cells) {
        CUDART_INSTANCE.cudaFree(cp.d_cell_atom_counts);
        CUDART_INSTANCE.cudaFree(cp.d_cell_atom_starts);
        CUDART_INSTANCE.cudaFree(cp.d_cell_scatter_offsets);

        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_cells_total)) + 16;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_cell_atom_counts, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_cell_atom_starts, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_cell_scatter_offsets, sizeof(int32_t) * new_max));
        cp.grid_max_cells = new_max;
    }

    // Fixed-size buffers (only allocate once)
    if (cp.d_n_clusters_total == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_n_clusters_total, sizeof(int32_t)));
    }
    if (cp.d_grid_inv_box == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_grid_inv_box, sizeof(double) * 9));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_grid_n_cells, sizeof(int32_t) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_grid_n_search, sizeof(int32_t) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.d_grid_n_cells_total, sizeof(int32_t)));
    }
}

/// Build a cluster grid on CPU from host-side positions, then upload to GPU.
///
/// The CPU builds the grid using the same cell-based binning as the CPU
/// cluster-pair path, but with CLUSTER_SIZE_GPU = 8 atoms per cluster.
/// The resulting cluster data is copied to pre-allocated GPU buffers.
///
/// Returns the total number of clusters and the grid parameters.
struct GpuClusterGridInfo {
    int32_t n_clusters_total;
    int32_t n_cells_total;
    int32_t n_cells[3];
    int32_t n_search[3];
};

static GpuClusterGridInfo build_and_upload_cluster_grid(
    const double* h_positions, // host [n_points * 3]
    size_t n_points,
    const double h_box[3][3],
    const bool h_periodic[3],
    double cutoff,
    CudaNeighborListExtras::ClusterPairBuffers& gpu_bufs
) {
    GpuClusterGridInfo info{};

    // Compute distances between faces (same as BoundingBox)
    // For simplicity, use the box diagonal for orthogonal cells,
    // or cross products for general cells
    double a[3] = {h_box[0][0], h_box[0][1], h_box[0][2]};
    double b[3] = {h_box[1][0], h_box[1][1], h_box[1][2]};
    double c[3] = {h_box[2][0], h_box[2][1], h_box[2][2]};

    // Cross products for face normals
    double bxc[3] = {b[1]*c[2] - b[2]*c[1], b[2]*c[0] - b[0]*c[2], b[0]*c[1] - b[1]*c[0]};
    double cxa[3] = {c[1]*a[2] - c[2]*a[1], c[2]*a[0] - c[0]*a[2], c[0]*a[1] - c[1]*a[0]};
    double axb[3] = {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};

    double vol = std::abs(a[0]*bxc[0] + a[1]*bxc[1] + a[2]*bxc[2]);
    double norm_bxc = std::sqrt(bxc[0]*bxc[0] + bxc[1]*bxc[1] + bxc[2]*bxc[2]);
    double norm_cxa = std::sqrt(cxa[0]*cxa[0] + cxa[1]*cxa[1] + cxa[2]*cxa[2]);
    double norm_axb = std::sqrt(axb[0]*axb[0] + axb[1]*axb[1] + axb[2]*axb[2]);

    double dist_faces[3];
    dist_faces[0] = (norm_bxc > 0) ? vol / norm_bxc : 1e30;
    dist_faces[1] = (norm_cxa > 0) ? vol / norm_cxa : 1e30;
    dist_faces[2] = (norm_axb > 0) ? vol / norm_axb : 1e30;

    // Grid dimensions
    for (int d = 0; d < 3; d++) {
        double nc = std::trunc(dist_faces[d] / cutoff);
        if (nc < 1) nc = 1;
        info.n_cells[d] = static_cast<int32_t>(nc);
    }

    // Limit total cells
    double total = (double)info.n_cells[0] * info.n_cells[1] * info.n_cells[2];
    if (total > 1e5) {
        double rx = (double)info.n_cells[0] / info.n_cells[1];
        double ry = (double)info.n_cells[1] / info.n_cells[2];
        info.n_cells[2] = std::max(1, (int32_t)std::trunc(std::cbrt(1e5 / (rx * ry * ry))));
        info.n_cells[1] = std::max(1, (int32_t)std::trunc(ry * info.n_cells[2]));
        info.n_cells[0] = std::max(1, (int32_t)std::trunc(rx * info.n_cells[1]));
    }

    info.n_cells_total = info.n_cells[0] * info.n_cells[1] * info.n_cells[2];

    // Compute search range
    for (int d = 0; d < 3; d++) {
        info.n_search[d] = static_cast<int32_t>(std::ceil(cutoff * info.n_cells[d] / dist_faces[d]));
        if (info.n_search[d] < 1) info.n_search[d] = 1;
        if (info.n_cells[d] == 1 && !h_periodic[d]) info.n_search[d] = 0;
    }

    // Compute inverse box for fractional coordinates
    double det = a[0]*(b[1]*c[2] - b[2]*c[1])
               - a[1]*(b[0]*c[2] - b[2]*c[0])
               + a[2]*(b[0]*c[1] - b[1]*c[0]);
    double inv_det = 1.0 / det;
    double inv_box[9] = {
        (b[1]*c[2] - b[2]*c[1]) * inv_det,
        (a[2]*c[1] - a[1]*c[2]) * inv_det,
        (a[1]*b[2] - a[2]*b[1]) * inv_det,
        (b[2]*c[0] - b[0]*c[2]) * inv_det,
        (a[0]*c[2] - a[2]*c[0]) * inv_det,
        (a[2]*b[0] - a[0]*b[2]) * inv_det,
        (b[0]*c[1] - b[1]*c[0]) * inv_det,
        (a[1]*c[0] - a[0]*c[1]) * inv_det,
        (a[0]*b[1] - a[1]*b[0]) * inv_det,
    };

    // Assign atoms to cells
    struct AtomAssignment {
        size_t atom_index;
        int32_t cell_linear;
        float z_frac;
    };
    std::vector<AtomAssignment> assignments(n_points);

    for (size_t i = 0; i < n_points; i++) {
        double px = h_positions[i * 3 + 0];
        double py = h_positions[i * 3 + 1];
        double pz = h_positions[i * 3 + 2];

        // Fractional coordinates: frac = inv_box^T * pos (column-major convention)
        double fx = inv_box[0]*px + inv_box[3]*py + inv_box[6]*pz;
        double fy = inv_box[1]*px + inv_box[4]*py + inv_box[7]*pz;
        double fz = inv_box[2]*px + inv_box[5]*py + inv_box[8]*pz;

        int32_t cx = static_cast<int32_t>(std::floor(fx * info.n_cells[0]));
        int32_t cy = static_cast<int32_t>(std::floor(fy * info.n_cells[1]));
        int32_t cz = static_cast<int32_t>(std::floor(fz * info.n_cells[2]));

        // Wrap into cell range
        for (int d = 0; d < 3; d++) {
            int32_t* c_ptr = (d == 0) ? &cx : (d == 1) ? &cy : &cz;
            int32_t nc = info.n_cells[d];
            if (h_periodic[d]) {
                *c_ptr = ((*c_ptr % nc) + nc) % nc;
            } else {
                *c_ptr = std::clamp(*c_ptr, 0, nc - 1);
            }
        }

        int32_t linear = cz * info.n_cells[0] * info.n_cells[1]
                       + cy * info.n_cells[0] + cx;
        assignments[i] = {i, linear, static_cast<float>(fz)};
    }

    // Sort by cell, then by z within each cell
    std::sort(assignments.begin(), assignments.end(),
        [](const AtomAssignment& a, const AtomAssignment& b) {
            if (a.cell_linear != b.cell_linear) return a.cell_linear < b.cell_linear;
            return a.z_frac < b.z_frac;
        }
    );

    // Count atoms per cell
    std::vector<int32_t> cell_counts(info.n_cells_total, 0);
    for (auto& a : assignments) {
        cell_counts[a.cell_linear]++;
    }

    // Build clusters (groups of CLUSTER_SIZE_GPU)
    std::vector<int32_t> h_cell_offsets(info.n_cells_total + 1, 0);
    std::vector<int32_t> h_cluster_atom_indices;
    std::vector<int32_t> h_cluster_n_atoms;
    std::vector<float> h_cluster_bb_lower;
    std::vector<float> h_cluster_bb_upper;

    size_t atom_cursor = 0;
    for (int32_t cell = 0; cell < info.n_cells_total; cell++) {
        h_cell_offsets[cell] = static_cast<int32_t>(h_cluster_n_atoms.size());

        int32_t count = cell_counts[cell];
        for (int32_t off = 0; off < count; off += CLUSTER_SIZE_GPU) {
            int32_t n_in_cluster = std::min(CLUSTER_SIZE_GPU, count - off);

            // Atom indices
            for (int32_t k = 0; k < CLUSTER_SIZE_GPU; k++) {
                if (k < n_in_cluster) {
                    h_cluster_atom_indices.push_back(
                        static_cast<int32_t>(assignments[atom_cursor + off + k].atom_index)
                    );
                } else {
                    h_cluster_atom_indices.push_back(-1); // padding
                }
            }

            // Bounding box
            float bb_lo[3] = {1e30f, 1e30f, 1e30f};
            float bb_hi[3] = {-1e30f, -1e30f, -1e30f};
            for (int32_t k = 0; k < n_in_cluster; k++) {
                size_t ai = assignments[atom_cursor + off + k].atom_index;
                for (int d = 0; d < 3; d++) {
                    float p = static_cast<float>(h_positions[ai * 3 + d]);
                    bb_lo[d] = std::min(bb_lo[d], p);
                    bb_hi[d] = std::max(bb_hi[d], p);
                }
            }
            for (int d = 0; d < 3; d++) {
                h_cluster_bb_lower.push_back(bb_lo[d]);
                h_cluster_bb_upper.push_back(bb_hi[d]);
            }

            h_cluster_n_atoms.push_back(n_in_cluster);
        }

        atom_cursor += count;
    }
    h_cell_offsets[info.n_cells_total] = static_cast<int32_t>(h_cluster_n_atoms.size());
    info.n_clusters_total = static_cast<int32_t>(h_cluster_n_atoms.size());

    // Upload to GPU
    size_t n_cl = static_cast<size_t>(info.n_clusters_total);
    size_t n_cells_p1 = static_cast<size_t>(info.n_cells_total + 1);

    // Reallocate if needed
    if (n_cl > gpu_bufs.max_clusters) {
        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_cl)) + 16;
        if (gpu_bufs.cluster_atom_indices) CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(gpu_bufs.cluster_atom_indices));
        if (gpu_bufs.cluster_n_atoms) CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(gpu_bufs.cluster_n_atoms));
        if (gpu_bufs.cluster_bb_lower) CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(gpu_bufs.cluster_bb_lower));
        if (gpu_bufs.cluster_bb_upper) CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(gpu_bufs.cluster_bb_upper));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&gpu_bufs.cluster_atom_indices, sizeof(int32_t) * new_max * CLUSTER_SIZE_GPU));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&gpu_bufs.cluster_n_atoms, sizeof(int32_t) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&gpu_bufs.cluster_bb_lower, sizeof(float) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&gpu_bufs.cluster_bb_upper, sizeof(float) * new_max * 3));
        gpu_bufs.max_clusters = new_max;
    }
    if (n_cells_p1 > gpu_bufs.max_cells) {
        auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_cells_p1)) + 16;
        if (gpu_bufs.cell_offsets) CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(gpu_bufs.cell_offsets));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&gpu_bufs.cell_offsets, sizeof(int32_t) * new_max));
        gpu_bufs.max_cells = new_max;
    }

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(gpu_bufs.cluster_atom_indices, h_cluster_atom_indices.data(), sizeof(int32_t) * n_cl * CLUSTER_SIZE_GPU, cudaMemcpyHostToDevice));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(gpu_bufs.cluster_n_atoms, h_cluster_n_atoms.data(), sizeof(int32_t) * n_cl, cudaMemcpyHostToDevice));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(gpu_bufs.cluster_bb_lower, h_cluster_bb_lower.data(), sizeof(float) * n_cl * 3, cudaMemcpyHostToDevice));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(gpu_bufs.cluster_bb_upper, h_cluster_bb_upper.data(), sizeof(float) * n_cl * 3, cudaMemcpyHostToDevice));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(gpu_bufs.cell_offsets, h_cell_offsets.data(), sizeof(int32_t) * n_cells_p1, cudaMemcpyHostToDevice));

    return info;
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
    if (options.sorted) {
        throw std::runtime_error("CUDA implemented does not support sorted output yet");
    }

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
    size_t max_pairs_per_point = VESIN_DEFAULT_CUDA_MAX_PAIRS_PER_POINT;

    if (extras->allocated_device_id != device_id) {
        // first switch to previous device
        if (extras->allocated_device_id >= 0) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(extras->allocated_device_id));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device_id));
        extras->allocated_device_id = device_id;
    }

    if (extras->capacity >= n_points && (extras->length_ptr != nullptr)) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));
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

        auto max_pairs = static_cast<size_t>(1.2 * static_cast<double>(n_points * max_pairs_per_point));
        extras->max_pairs = max_pairs;

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * max_pairs * 2));

        if (options.return_shifts) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3)
            );
        }

        if (options.return_distances) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.distances, sizeof(double) * max_pairs)
            );
        }

        if (options.return_vectors) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.vectors, sizeof(double) * max_pairs * 3)
            );
        }

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        // Pinned host memory for async D2H copy
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaHostAlloc(
            (void**)&extras->pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int32_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->overflow_flag, sizeof(int32_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));

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

    auto& factory = KernelFactory::instance(device_id);

    if (extras->box_diag == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->box_diag, sizeof(double) * 3));
    }
    if (extras->inv_box_brute == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->inv_box_brute, sizeof(double) * 9));
    }

    auto* box_check_kernel = factory.create(
        "mic_box_check",
        CUDA_BRUTEFORCE_CODE,
        "cuda_bruteforce.cu",
        {"-std=c++17"}
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
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool box_check_error = (h_cell_check & 1) != 0;
    bool is_orthogonal = (h_cell_check & 2) != 0;

    // Get box dimensions for auto algorithm selection
    double h_box_diag[3];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(h_box_diag, d_box_diag, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    double min_box_dim = std::min({h_box_diag[0], h_box_diag[1], h_box_diag[2]});
    bool cutoff_requires_cell_list = options.cutoff > min_box_dim / 2.0;

    bool use_cell_list = false;
    bool use_cluster_pair = false;
    switch (options.algorithm) {
    case VesinBruteForce:
        if (box_check_error) {
            throw std::runtime_error("Invalid cutoff: too large for box dimensions");
        }
        break;
    case VesinCellList:
        use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        if (n_points >= GPU_CLUSTER_PAIR_THRESHOLD) {
            // Cluster-pair for large systems
            use_cluster_pair = true;
        } else if (cutoff_requires_cell_list || !is_orthogonal || n_points >= 5000) {
            use_cell_list = true;
        }
        break;
    }

    if (use_cluster_pair) {
        NVTX_PUSH("cluster_pair_total");

        auto& cp = extras->cluster_pair;
        size_t THREADS_PER_BLOCK_CG = 256;
        size_t num_blocks_points_cg = (n_points + THREADS_PER_BLOCK_CG - 1) / THREADS_PER_BLOCK_CG;

        // --- GPU-native cluster grid builder ---
        // All grid construction happens on-device, no D2H/H2D transfers
        // except a single 4-byte D2H for n_clusters_total.

        NVTX_PUSH("build_cluster_grid_gpu");

        // Ensure grid builder buffers are allocated
        ensure_cluster_grid_buffers(cp, n_points, MAX_CELLS);

        // Kernel 1: Compute grid parameters (inv_box, n_cells, n_search)
        int32_t max_cells_cp = 100000; // cluster-pair cell limit
        auto* grid_params_kernel = factory.create(
            "compute_cluster_grid_params",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> grid_params_args = {
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&max_cells_cp),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cp.d_grid_inv_box),
            static_cast<void*>(&cp.d_grid_n_cells),
            static_cast<void*>(&cp.d_grid_n_search),
            static_cast<void*>(&cp.d_grid_n_cells_total),
        };
        grid_params_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_params_args, false);

        // Kernel 2: Assign atoms to cells
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cp.d_cell_atom_counts, 0, sizeof(int32_t) * MAX_CELLS));
        auto* assign_kernel = factory.create(
            "assign_cell_indices_cluster",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> assign_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cp.d_grid_inv_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cp.d_grid_n_cells),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cp.d_cell_indices),
            static_cast<void*>(&cp.d_atom_frac_z),
        };
        assign_kernel->launch(
            dim3(num_blocks_points_cg), dim3(THREADS_PER_BLOCK_CG),
            0, nullptr, assign_args, false
        );

        // Kernel 3a: Count atoms per cell
        auto* count_kernel = factory.create(
            "count_atoms_per_cell",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> count_args = {
            static_cast<void*>(&cp.d_cell_indices),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cp.d_cell_atom_counts),
        };
        count_kernel->launch(
            dim3(num_blocks_points_cg), dim3(THREADS_PER_BLOCK_CG),
            0, nullptr, count_args, false
        );

        // Kernel 3b: Prefix sum for cell starts
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cp.d_cell_atom_starts, 0, sizeof(int32_t) * MAX_CELLS));
        auto* prefix_kernel = factory.create(
            "prefix_sum_cluster_cells",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> prefix_args = {
            static_cast<void*>(&cp.d_cell_atom_counts),
            static_cast<void*>(&cp.d_cell_atom_starts),
            static_cast<void*>(&cp.d_grid_n_cells_total),
        };
        size_t prefix_threads = 256;
        size_t shared_mem_cg = sizeof(int32_t) * prefix_threads;
        prefix_kernel->launch(
            dim3(1), dim3(prefix_threads), shared_mem_cg, nullptr, prefix_args, false
        );

        // Copy cell_starts -> scatter offsets (working copy for atomicAdd scatter)
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            cp.d_cell_scatter_offsets, cp.d_cell_atom_starts,
            sizeof(int32_t) * MAX_CELLS, cudaMemcpyDeviceToDevice
        ));

        // Kernel 4: Scatter atoms into sorted order
        auto* scatter_kernel = factory.create(
            "scatter_atoms_by_cell",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> scatter_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cp.d_cell_indices),
            static_cast<void*>(&cp.d_atom_frac_z),
            static_cast<void*>(&cp.d_cell_scatter_offsets),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cp.d_sorted_positions),
            static_cast<void*>(&cp.d_sorted_atom_indices),
            static_cast<void*>(&cp.d_sorted_frac_z),
        };
        scatter_kernel->launch(
            dim3(num_blocks_points_cg), dim3(THREADS_PER_BLOCK_CG),
            0, nullptr, scatter_args, false
        );

        // Reset n_clusters_total counter
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cp.d_n_clusters_total, 0, sizeof(int32_t)));

        // Ensure cluster output buffers are large enough.
        // Estimate max clusters: n_points / CLUSTER_SIZE_GPU + n_cells_total
        size_t est_clusters = n_points / CLUSTER_SIZE_GPU + MAX_CELLS + 16;
        if (est_clusters > cp.max_clusters) {
            auto new_max = static_cast<size_t>(1.2 * static_cast<double>(est_clusters)) + 16;
            CUDART_INSTANCE.cudaFree(cp.cluster_atom_indices);
            CUDART_INSTANCE.cudaFree(cp.cluster_n_atoms);
            CUDART_INSTANCE.cudaFree(cp.cluster_bb_lower);
            CUDART_INSTANCE.cudaFree(cp.cluster_bb_upper);
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.cluster_atom_indices, sizeof(int32_t) * new_max * CLUSTER_SIZE_GPU));
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.cluster_n_atoms, sizeof(int32_t) * new_max));
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.cluster_bb_lower, sizeof(float) * new_max * 3));
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.cluster_bb_upper, sizeof(float) * new_max * 3));
            cp.max_clusters = new_max;
        }
        size_t n_cells_plus_1 = MAX_CELLS + 1;
        if (n_cells_plus_1 > cp.max_cells) {
            auto new_max = static_cast<size_t>(1.2 * static_cast<double>(n_cells_plus_1)) + 16;
            CUDART_INSTANCE.cudaFree(cp.cell_offsets);
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cp.cell_offsets, sizeof(int32_t) * new_max));
            cp.max_cells = new_max;
        }

        // D2H: read n_cells_total to size kernel 5 launch
        int32_t h_n_cells_total = 0;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            &h_n_cells_total, cp.d_grid_n_cells_total,
            sizeof(int32_t), cudaMemcpyDeviceToHost
        ));

        // Kernel 5: Build clusters and bounding boxes (one thread per cell)
        auto* build_kernel = factory.create(
            "build_clusters_and_bboxes",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        size_t num_blocks_cells = (h_n_cells_total + THREADS_PER_BLOCK_CG - 1) / THREADS_PER_BLOCK_CG;
        std::vector<void*> build_args = {
            static_cast<void*>(&cp.d_sorted_positions),
            static_cast<void*>(&cp.d_sorted_atom_indices),
            static_cast<void*>(&cp.d_sorted_frac_z),
            static_cast<void*>(&cp.d_cell_atom_starts),
            static_cast<void*>(&cp.d_cell_atom_counts),
            static_cast<void*>(&cp.d_grid_n_cells_total),
            static_cast<void*>(&cp.cluster_atom_indices),
            static_cast<void*>(&cp.cluster_n_atoms),
            static_cast<void*>(&cp.cluster_bb_lower),
            static_cast<void*>(&cp.cluster_bb_upper),
            static_cast<void*>(&cp.cell_offsets),
            static_cast<void*>(&cp.d_n_clusters_total),
        };
        build_kernel->launch(
            dim3(std::max(num_blocks_cells, static_cast<size_t>(1))),
            dim3(THREADS_PER_BLOCK_CG),
            0, nullptr, build_args, false
        );

        // Kernel 6: Finalize cell_cluster_offsets sentinel
        auto* finalize_kernel = factory.create(
            "finalize_cluster_offsets",
            CUDA_CLUSTER_GRID_CODE,
            "cuda_cluster_grid.cu",
            {"-std=c++17"}
        );
        std::vector<void*> finalize_args = {
            static_cast<void*>(&cp.cell_offsets),
            static_cast<void*>(&cp.d_grid_n_cells_total),
            static_cast<void*>(&cp.d_n_clusters_total),
        };
        finalize_kernel->launch(dim3(1), dim3(1), 0, nullptr, finalize_args, false);

        // D2H: read n_clusters_total for kernel launch sizing
        int32_t h_n_clusters_total = 0;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            &h_n_clusters_total, cp.d_n_clusters_total,
            sizeof(int32_t), cudaMemcpyDeviceToHost
        ));

        NVTX_POP(); // build_cluster_grid_gpu

        // --- Launch cluster-pair search kernel ---
        NVTX_PUSH("cluster_pair_kernel");
        auto* kernel = factory.create(
            "cluster_pair_search",
            CUDA_CLUSTER_PAIR_CODE,
            "cuda_cluster_pair.cu",
            {"-std=c++17"}
        );

        int n_clusters_total = h_n_clusters_total;
        int n_cells_total_int = h_n_cells_total;
        std::vector<void*> args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cp.cluster_atom_indices),
            static_cast<void*>(&cp.cluster_n_atoms),
            static_cast<void*>(&cp.cluster_bb_lower),
            static_cast<void*>(&cp.cluster_bb_upper),
            static_cast<void*>(&cp.cell_offsets),
            static_cast<void*>(&n_clusters_total),
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cp.d_grid_n_cells),
            static_cast<void*>(&cp.d_grid_n_search),
            static_cast<void*>(&n_cells_total_int),
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
            static_cast<void*>(&d_overflow_flag),
        };

        size_t THREADS_PER_BLOCK_CP = 128;
        size_t num_blocks_cp = (n_clusters_total + THREADS_PER_BLOCK_CP - 1) / THREADS_PER_BLOCK_CP;
        kernel->launch(
            dim3(std::max(num_blocks_cp, static_cast<size_t>(1))),
            dim3(THREADS_PER_BLOCK_CP),
            0, nullptr, args, false
        );
        NVTX_POP();

        NVTX_POP(); // cluster_pair_total
    } else if (use_cell_list) {
        NVTX_PUSH("cell_list_total");

        NVTX_PUSH("ensure_buffers");
        ensure_cell_list_buffers(extras->cell_list, n_points, MAX_CELLS);
        NVTX_POP();
        auto& cl = extras->cell_list;

        int32_t max_cells_int = static_cast<int32_t>(MAX_CELLS);
        int32_t min_particles_per_cell = MIN_PARTICLES_PER_CELL;

        size_t THREADS_PER_BLOCK = 256;
        size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        NVTX_PUSH("kernel0_grid_params");
        auto* grid_kernel = factory.create(
            "compute_cell_grid_params",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> grid_args = {
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&max_cells_int),
            static_cast<void*>(&n_points),
            static_cast<void*>(&min_particles_per_cell),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&cl.n_search),
            static_cast<void*>(&cl.n_cells_total),
        };
        grid_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_args, false);
        NVTX_POP();

        NVTX_PUSH("memset_cell_counts");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cl.cell_counts, 0, sizeof(int32_t) * MAX_CELLS));
        NVTX_POP();

        NVTX_PUSH("memset_cell_starts");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cl.cell_starts, 0, sizeof(int32_t) * MAX_CELLS));
        NVTX_POP();

        NVTX_PUSH("kernel1_assign_cells");
        auto* assign_kernel = factory.create(
            "assign_cell_indices",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> assign_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&cl.particle_shifts),
        };
        assign_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, assign_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel2_count_particles");
        auto* count_kernel = factory.create(
            "count_particles_per_cell",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
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
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
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
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            cl.cell_offsets, cl.cell_starts, sizeof(int32_t) * MAX_CELLS, cudaMemcpyDeviceToDevice
        ));
        NVTX_POP();

        NVTX_PUSH("kernel4_scatter");
        auto* scatter_kernel = factory.create(
            "scatter_particles",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
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
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
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
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
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
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
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
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
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
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
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

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize());

    // Check for overflow
    int h_overflow_flag = 0;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
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
}
