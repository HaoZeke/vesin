// GPU Verlet implementation: displacement check and pair recompute kernels.
// Separated from verlet.cpp because this file requires gpulite headers.

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>

#define NOMINMAX
#include <gpulite/gpulite.hpp>

#include "vesin.h"
#include "vesin_cuda.hpp"
#include "verlet.hpp"

using namespace vesin;

static const char* CUDA_VERLET_CODE =
#include "generated/cuda_verlet.cu"
    ;

void vesin::verlet_free_gpu_buffers(VerletGpuBuffers& gpu) {
    if (gpu.device_id < 0) return;

    if (gpu.d_ref_positions != nullptr) {
        CUDART_INSTANCE.cudaFree(gpu.d_ref_positions);
        gpu.d_ref_positions = nullptr;
    }
    if (gpu.d_pairs_i != nullptr) {
        CUDART_INSTANCE.cudaFree(gpu.d_pairs_i);
        gpu.d_pairs_i = nullptr;
    }
    if (gpu.d_pairs_j != nullptr) {
        CUDART_INSTANCE.cudaFree(gpu.d_pairs_j);
        gpu.d_pairs_j = nullptr;
    }
    if (gpu.d_shifts != nullptr) {
        CUDART_INSTANCE.cudaFree(gpu.d_shifts);
        gpu.d_shifts = nullptr;
    }
    if (gpu.d_rebuild_flag != nullptr) {
        CUDART_INSTANCE.cudaFree(gpu.d_rebuild_flag);
        gpu.d_rebuild_flag = nullptr;
    }
    gpu.capacity_points = 0;
    gpu.capacity_pairs = 0;
    gpu.device_id = -1;
}

static void ensure_gpu_buffers(
    VerletGpuBuffers& gpu,
    size_t n_points,
    size_t n_pairs,
    int32_t device_id
) {
    if (gpu.device_id != device_id) {
        vesin::verlet_free_gpu_buffers(gpu);
        gpu.device_id = device_id;
    }

    if (gpu.capacity_points < n_points) {
        CUDART_INSTANCE.cudaFree(gpu.d_ref_positions);
        auto cap = static_cast<size_t>(1.2 * static_cast<double>(n_points));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&gpu.d_ref_positions, sizeof(double) * cap * 3
        ));
        gpu.capacity_points = cap;
    }

    if (gpu.d_rebuild_flag == nullptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&gpu.d_rebuild_flag, sizeof(int)
        ));
    }

    if (gpu.capacity_pairs < n_pairs) {
        CUDART_INSTANCE.cudaFree(gpu.d_pairs_i);
        CUDART_INSTANCE.cudaFree(gpu.d_pairs_j);
        CUDART_INSTANCE.cudaFree(gpu.d_shifts);
        auto cap = static_cast<size_t>(1.2 * static_cast<double>(n_pairs));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&gpu.d_pairs_i, sizeof(size_t) * cap
        ));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&gpu.d_pairs_j, sizeof(size_t) * cap
        ));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&gpu.d_shifts, sizeof(int32_t) * cap * 3
        ));
        gpu.capacity_pairs = cap;
    }
}

void vesin::verlet_upload_topology(VerletState& state) {
    auto& gpu = state.gpu;
    auto n_pairs = state.n_pairs;
    auto n_points = state.n_points;

    ensure_gpu_buffers(gpu, n_points, n_pairs, gpu.device_id);

    // Upload reference positions
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        gpu.d_ref_positions,
        state.ref_positions.data(),
        sizeof(double) * n_points * 3,
        cudaMemcpyHostToDevice
    ));

    // Upload cached topology
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        gpu.d_pairs_i,
        state.pairs_i.data(),
        sizeof(size_t) * n_pairs,
        cudaMemcpyHostToDevice
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        gpu.d_pairs_j,
        state.pairs_j.data(),
        sizeof(size_t) * n_pairs,
        cudaMemcpyHostToDevice
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        gpu.d_shifts,
        state.shifts.data(),
        sizeof(int32_t) * n_pairs * 3,
        cudaMemcpyHostToDevice
    ));
}

bool vesin::verlet_needs_rebuild_gpu(
    const VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
) {
    if (!state.has_cache) return true;
    if (n_points != state.n_points) return true;

    // Check periodicity (D2H copy of 3 bools)
    bool h_periodic[3];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_periodic, periodic, sizeof(bool) * 3, cudaMemcpyDeviceToHost
    ));
    for (int d = 0; d < 3; d++) {
        if (h_periodic[d] != state.ref_periodic[d]) return true;
    }

    // Check box (D2H copy of 72 bytes)
    double h_box[9];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_box, box, sizeof(double) * 9, cudaMemcpyDeviceToHost
    ));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(h_box[i * 3 + j] - state.ref_box[i][j]) > 1e-12) {
                return true;
            }
        }
    }

    // GPU displacement check
    auto& gpu = state.gpu;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(gpu.d_rebuild_flag, 0, sizeof(int)));

    auto& factory = KernelFactory::instance(gpu.device_id);
    auto* kernel = factory.create(
        "verlet_max_displacement",
        CUDA_VERLET_CODE,
        "cuda_verlet.cu",
        {"-std=c++17"}
    );

    auto d_positions = reinterpret_cast<const double*>(points);
    auto d_ref = gpu.d_ref_positions;
    auto half_skin_sq = state.half_skin_sq;
    auto* d_flag = gpu.d_rebuild_flag;

    std::vector<void*> args = {
        const_cast<void*>(static_cast<const void*>(&d_positions)),
        static_cast<void*>(&d_ref),
        static_cast<void*>(const_cast<size_t*>(&n_points)),
        static_cast<void*>(const_cast<double*>(&half_skin_sq)),
        static_cast<void*>(&d_flag),
    };

    size_t THREADS = 256;
    size_t blocks = (n_points + THREADS - 1) / THREADS;
    kernel->launch(
        dim3(std::max(blocks, static_cast<size_t>(1))),
        dim3(THREADS),
        0, nullptr, args, false
    );

    int h_flag = 0;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        &h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost
    ));

    return h_flag != 0;
}

void vesin::verlet_recompute_gpu(
    const VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto& gpu = state.gpu;
    auto n_cached = state.n_pairs;

    auto* extras = cuda::get_cuda_extras(&neighbors);

    // Reset the length counter
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->overflow_flag, 0, sizeof(int)));

    auto& factory = KernelFactory::instance(gpu.device_id);
    auto* kernel = factory.create(
        "verlet_recompute_pairs",
        CUDA_VERLET_CODE,
        "cuda_verlet.cu",
        {"-std=c++17"}
    );

    auto d_positions = reinterpret_cast<const double*>(points);
    auto d_box = reinterpret_cast<const double*>(box);
    auto cutoff_sq = state.cutoff * state.cutoff;
    auto* d_pairs_i = gpu.d_pairs_i;
    auto* d_pairs_j = gpu.d_pairs_j;
    auto* d_shifts = gpu.d_shifts;
    auto* d_output_len = extras->length_ptr;
    auto* d_output_pairs = reinterpret_cast<size_t*>(neighbors.pairs);
    auto* d_output_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    auto* d_output_distances = neighbors.distances;
    auto* d_output_vectors = reinterpret_cast<double*>(neighbors.vectors);
    auto max_pairs = extras->max_pairs;
    auto* d_overflow = extras->overflow_flag;

    std::vector<void*> args = {
        const_cast<void*>(static_cast<const void*>(&d_positions)),
        const_cast<void*>(static_cast<const void*>(&d_box)),
        static_cast<void*>(&d_pairs_i),
        static_cast<void*>(&d_pairs_j),
        static_cast<void*>(&d_shifts),
        static_cast<void*>(const_cast<size_t*>(&n_cached)),
        static_cast<void*>(const_cast<double*>(&cutoff_sq)),
        static_cast<void*>(&d_output_len),
        static_cast<void*>(&d_output_pairs),
        static_cast<void*>(&d_output_shifts),
        static_cast<void*>(&d_output_distances),
        static_cast<void*>(&d_output_vectors),
        static_cast<void*>(const_cast<bool*>(&options.return_shifts)),
        static_cast<void*>(const_cast<bool*>(&options.return_distances)),
        static_cast<void*>(const_cast<bool*>(&options.return_vectors)),
        static_cast<void*>(const_cast<size_t*>(&max_pairs)),
        static_cast<void*>(&d_overflow),
    };

    size_t THREADS = 256;
    size_t blocks = (n_cached + THREADS - 1) / THREADS;
    kernel->launch(
        dim3(std::max(blocks, static_cast<size_t>(1))),
        dim3(THREADS),
        0, nullptr, args, false
    );

    // Read back length
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_output_len,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize());

    // Check overflow
    int h_overflow = 0;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        &h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost
    ));
    if (h_overflow != 0) {
        throw std::runtime_error(
            "Verlet recompute: pair count exceeds GPU buffer capacity (" +
            std::to_string(max_pairs) + ")"
        );
    }

    neighbors.length = *extras->pinned_length_ptr;
}

// GPU rebuild: run stateless GPU NL, copy topology to CPU + upload to GPU
void vesin::verlet_rebuild_gpu(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinDevice device
) {
    double expanded_cutoff = state.cutoff + state.skin;

    auto options = VesinOptions();
    options.cutoff = expanded_cutoff;
    options.full = state.full_list;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;
    options.skin = 0.0; // stateless search for the rebuild

    VesinNeighborList tmp_neighbors;
    tmp_neighbors.device = device;

    const char* rebuild_error = nullptr;
    int status = vesin_neighbors(
        points,
        n_points,
        box,
        periodic,
        device,
        options,
        &tmp_neighbors,
        &rebuild_error
    );
    if (status != EXIT_SUCCESS) {
        std::string msg = "verlet_rebuild (GPU): ";
        if (rebuild_error) msg += rebuild_error;
        throw std::runtime_error(msg);
    }

    // Copy topology from GPU to CPU vectors
    state.n_pairs = tmp_neighbors.length;
    state.pairs_i.resize(state.n_pairs);
    state.pairs_j.resize(state.n_pairs);
    state.shifts.resize(state.n_pairs * 3);

    std::vector<size_t> h_pairs(state.n_pairs * 2);
    std::vector<int32_t> h_shifts(state.n_pairs * 3);

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_pairs.data(), tmp_neighbors.pairs,
        sizeof(size_t) * state.n_pairs * 2,
        cudaMemcpyDeviceToHost
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_shifts.data(), tmp_neighbors.shifts,
        sizeof(int32_t) * state.n_pairs * 3,
        cudaMemcpyDeviceToHost
    ));

    for (size_t k = 0; k < state.n_pairs; k++) {
        state.pairs_i[k] = h_pairs[k * 2 + 0];
        state.pairs_j[k] = h_pairs[k * 2 + 1];
        state.shifts[k * 3 + 0] = h_shifts[k * 3 + 0];
        state.shifts[k * 3 + 1] = h_shifts[k * 3 + 1];
        state.shifts[k * 3 + 2] = h_shifts[k * 3 + 2];
    }

    // Store reference state from device
    state.n_points = n_points;
    state.ref_positions.resize(n_points * 3);
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        state.ref_positions.data(), points,
        sizeof(double) * n_points * 3,
        cudaMemcpyDeviceToHost
    ));

    double h_box[9];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_box, box, sizeof(double) * 9, cudaMemcpyDeviceToHost
    ));
    std::memcpy(state.ref_box, h_box, 9 * sizeof(double));

    bool h_periodic[3];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
        h_periodic, periodic, sizeof(bool) * 3, cudaMemcpyDeviceToHost
    ));
    for (int d = 0; d < 3; d++) {
        state.ref_periodic[d] = h_periodic[d];
    }

    // Upload topology to GPU buffers
    state.gpu.device_id = device.device_id;
    ensure_gpu_buffers(state.gpu, n_points, state.n_pairs, device.device_id);
    verlet_upload_topology(state);

    vesin_free(&tmp_neighbors);

    state.has_cache = true;
    state.did_rebuild_flag = true;
}

// Allocate GPU output buffers for Verlet recompute
void vesin::verlet_ensure_gpu_output(
    VesinNeighborList& neighbors,
    size_t n_points,
    VesinDevice device,
    VesinOptions options
) {
    auto* extras = cuda::get_cuda_extras(&neighbors);

    if (extras->capacity >= n_points && extras->length_ptr != nullptr) {
        return; // Already allocated with enough capacity
    }

    auto saved_device_id = extras->allocated_device_id;

    if (saved_device_id >= 0 && saved_device_id != device.device_id) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(saved_device_id));
        cuda::free_neighbors(neighbors);
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device.device_id));
        extras = cuda::get_cuda_extras(&neighbors);
    }

    extras->allocated_device_id = device.device_id;

    size_t max_pairs_per_point = VESIN_DEFAULT_CUDA_MAX_PAIRS_PER_POINT;
    auto max_pairs = static_cast<size_t>(
        1.2 * static_cast<double>(n_points * max_pairs_per_point)
    );
    extras->max_pairs = max_pairs;

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
        (void**)&neighbors.pairs, sizeof(size_t) * max_pairs * 2
    ));

    if (options.return_shifts) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3
        ));
    }
    if (options.return_distances) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&neighbors.distances, sizeof(double) * max_pairs
        ));
    }
    if (options.return_vectors) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
            (void**)&neighbors.vectors, sizeof(double) * max_pairs * 3
        ));
    }

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
        (void**)&extras->length_ptr, sizeof(size_t)
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaHostAlloc(
        (void**)&extras->pinned_length_ptr, sizeof(size_t),
        cudaHostAllocDefault
    ));
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc(
        (void**)&extras->overflow_flag, sizeof(int32_t)
    ));

    extras->capacity = static_cast<size_t>(
        1.2 * static_cast<double>(n_points)
    );
}
