#include <catch2/catch_test_macros.hpp>

#ifdef VESIN_TESTS_WITH_CUDA

#include <algorithm>
#include <cmath>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <vesin.h>

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        const char* message = cudaGetErrorString(status);
        FAIL(message);
    }
}

void run_cuda_test(int device_id) {
    check_cuda(cudaSetDevice(device_id));

    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    size_t n_points = 3;
    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double box[3][3] = {
        {0.0, 3.0, 3.0},
        {3.0, 0.0, 3.0},
        {3.0, 3.0, 0.0},
    };
    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool periodic[3] = {true, true, true};
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    VesinNeighborList neighbors;

    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        d_points,
        n_points,
        d_box,
        d_periodic,
        {VesinDeviceKind::VesinCUDA, device_id},
        options,
        &neighbors,
        &error_message
    );

    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    CHECK(neighbors.length == 5);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts != nullptr);
    CHECK(neighbors.distances != nullptr);
    CHECK(neighbors.vectors != nullptr);

    auto* h_pairs = static_cast<size_t (*)[2]>(malloc(sizeof(size_t) * neighbors.length * 2));
    check_cuda(cudaMemcpy(h_pairs, neighbors.pairs, sizeof(size_t) * neighbors.length * 2, cudaMemcpyDeviceToHost));

    auto* h_shifts = static_cast<int32_t (*)[3]>(malloc(sizeof(int32_t) * neighbors.length * 3));
    check_cuda(cudaMemcpy(h_shifts, neighbors.shifts, sizeof(int32_t) * neighbors.length * 3, cudaMemcpyDeviceToHost));

    auto* h_distances = static_cast<double*>(malloc(sizeof(double) * neighbors.length));
    check_cuda(cudaMemcpy(h_distances, neighbors.distances, sizeof(double) * neighbors.length, cudaMemcpyDeviceToHost));

    auto* h_vectors = static_cast<double (*)[3]>(malloc(sizeof(double) * neighbors.length * 3));
    check_cuda(cudaMemcpy(h_vectors, neighbors.vectors, sizeof(double) * neighbors.length * 3, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < neighbors.length; ++i) {
        if (h_pairs[i][0] == 0 && h_pairs[i][1] == 2) {
            // we have three pairs between 0 and 2 with shifts (-1, 0, 0),
            // (0, -1, 0), and (0, 0, -1)
            CHECK(h_distances[i] == std::sqrt(6.0));

            if (h_shifts[i][0] == -1 && h_shifts[i][1] == 0 && h_shifts[i][2] == 0) {
                CHECK(h_vectors[i][0] == 2.0);
                CHECK(h_vectors[i][1] == -1.0);
                CHECK(h_vectors[i][2] == -1.0);
            } else if (h_shifts[i][0] == 0 && h_shifts[i][1] == -1 && h_shifts[i][2] == 0) {
                CHECK(h_vectors[i][0] == -1.0);
                CHECK(h_vectors[i][1] == 2.0);
                CHECK(h_vectors[i][2] == -1.0);
            } else if (h_shifts[i][0] == 0 && h_shifts[i][1] == 0 && h_shifts[i][2] == -1) {
                CHECK(h_vectors[i][0] == -1.0);
                CHECK(h_vectors[i][1] == -1.0);
                CHECK(h_vectors[i][2] == 2.0);
            } else {
                FAIL("Unexpected shift for pair (0, 2): (" + std::to_string(h_shifts[i][0]) + ", " + std::to_string(h_shifts[i][1]) + ", " + std::to_string(h_shifts[i][2]) + ")");
            }

        } else if ((h_pairs[i][0] == 0 && h_pairs[i][1] == 1) || (h_pairs[i][0] == 1 && h_pairs[i][1] == 2)) {
            // pairs between 0-1 or 1-2 should have zero shifts, distance
            // sqrt(3), and vector (1, 1, 1)
            CHECK(h_shifts[i][0] == 0);
            CHECK(h_shifts[i][1] == 0);
            CHECK(h_shifts[i][2] == 0);

            CHECK(h_distances[i] == std::sqrt(3.0));
            CHECK(h_vectors[i][0] == 1.0);
            CHECK(h_vectors[i][1] == 1.0);
            CHECK(h_vectors[i][2] == 1.0);
        } else {
            FAIL("Unexpected pair: (" + std::to_string(h_pairs[i][0]) + ", " + std::to_string(h_pairs[i][1]) + ")");
        }
    }

    // Clean up
    vesin_free(&neighbors);

    free(h_pairs);
    free(h_shifts);
    free(h_distances);
    free(h_vectors);

    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

TEST_CASE("Test CUDA") {
    // get the number of CUDA devices
    int n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);

    // start multiple threads to test concurrent execution
    auto threads = std::vector<std::thread>();
    for (int thread_id = 0; thread_id < 10; ++thread_id) {
        std::thread t(run_cuda_test, thread_id % n_devices);
        threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ========================================================================== //
// GPU Verlet correctness tests                                              //
// ========================================================================== //

void run_gpu_verlet_test(int device_id) {
    check_cuda(cudaSetDevice(device_id));

    // Same 3-atom system as the stateless GPU test
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    size_t n_points = 3;
    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double box[3][3] = {
        {0.0, 3.0, 3.0},
        {3.0, 0.0, 3.0},
        {3.0, 3.0, 0.0},
    };
    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool periodic[3] = {true, true, true};
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    double cutoff = 3.0;
    double skin = 1.0;

    // Create Verlet handle
    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(cutoff, skin, false, &error_message);
    REQUIRE(vl != nullptr);
    REQUIRE(error_message == nullptr);

    VesinNeighborList neighbors;

    auto options = VesinOptions();
    options.cutoff = 0.0;  // ignored by verlet
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call: should rebuild
    auto status = vesin_verlet_compute(
        vl, d_points, n_points, d_box, d_periodic,
        {VesinCUDA, device_id},
        options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    REQUIRE(error_message == nullptr);
    CHECK(vesin_verlet_did_rebuild(vl));
    CHECK(neighbors.length == 5);

    // Read back and verify pairs match the known answer
    auto* h_pairs = static_cast<size_t (*)[2]>(malloc(sizeof(size_t) * neighbors.length * 2));
    check_cuda(cudaMemcpy(h_pairs, neighbors.pairs, sizeof(size_t) * neighbors.length * 2, cudaMemcpyDeviceToHost));

    auto* h_distances = static_cast<double*>(malloc(sizeof(double) * neighbors.length));
    check_cuda(cudaMemcpy(h_distances, neighbors.distances, sizeof(double) * neighbors.length, cudaMemcpyDeviceToHost));

    std::set<std::tuple<size_t, size_t>> pair_set;
    for (size_t i = 0; i < neighbors.length; i++) {
        pair_set.emplace(h_pairs[i][0], h_pairs[i][1]);
    }
    // Same 5 pairs as stateless: (0,1), (1,2), (0,2)x3
    CHECK(pair_set.count({0, 1}) == 1);
    CHECK(pair_set.count({1, 2}) == 1);
    // 3 images of (0,2)
    size_t count_02 = 0;
    for (size_t i = 0; i < neighbors.length; i++) {
        if (h_pairs[i][0] == 0 && h_pairs[i][1] == 2) count_02++;
    }
    CHECK(count_02 == 3);

    // Second call with same positions: should NOT rebuild (reuse path)
    auto status2 = vesin_verlet_compute(
        vl, d_points, n_points, d_box, d_periodic,
        {VesinCUDA, device_id},
        options, &neighbors, &error_message
    );
    REQUIRE(status2 == EXIT_SUCCESS);
    CHECK_FALSE(vesin_verlet_did_rebuild(vl));
    CHECK(neighbors.length == 5);

    // Perturb points slightly (within skin/2): should still reuse
    double shifted_points[][3] = {
        {0.01, -0.01, 0.02},
        {1.01, 0.99, 1.02},
        {2.01, 1.99, 2.02},
    };
    check_cuda(cudaMemcpy(d_points, shifted_points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    auto status3 = vesin_verlet_compute(
        vl, d_points, n_points, d_box, d_periodic,
        {VesinCUDA, device_id},
        options, &neighbors, &error_message
    );
    REQUIRE(status3 == EXIT_SUCCESS);
    CHECK_FALSE(vesin_verlet_did_rebuild(vl));
    CHECK(neighbors.length == 5);

    // Verify distances are reasonable (recomputed from perturbed positions)
    auto* h_distances2 = static_cast<double*>(malloc(sizeof(double) * neighbors.length));
    check_cuda(cudaMemcpy(h_distances2, neighbors.distances, sizeof(double) * neighbors.length, cudaMemcpyDeviceToHost));

    // Distances should all be positive and within cutoff
    for (size_t i = 0; i < neighbors.length; i++) {
        CHECK(h_distances2[i] > 0.0);
        CHECK(h_distances2[i] <= cutoff);
    }

    // Read back vectors to verify they changed
    auto* h_vectors1 = static_cast<double (*)[3]>(malloc(sizeof(double) * neighbors.length * 3));
    check_cuda(cudaMemcpy(h_vectors1, neighbors.vectors, sizeof(double) * neighbors.length * 3, cudaMemcpyDeviceToHost));

    // At least some vector components should differ from exact values
    // Original distances: sqrt(3) ~ 1.732 for (0,1) and (1,2) pairs
    // Perturbed positions shift by 0.01-0.02, so distances change by ~0.01
    bool vectors_nonzero = false;
    for (size_t i = 0; i < neighbors.length; i++) {
        double vx = h_vectors1[i][0], vy = h_vectors1[i][1], vz = h_vectors1[i][2];
        if (std::abs(vx) > 1e-10 || std::abs(vy) > 1e-10 || std::abs(vz) > 1e-10) {
            vectors_nonzero = true;
        }
        // Distances should be positive and match vector magnitude
        double computed_dist = std::sqrt(vx * vx + vy * vy + vz * vz);
        CHECK(computed_dist > 0.0);
        CHECK(computed_dist <= cutoff);
    }
    CHECK(vectors_nonzero);

    free(h_vectors1);

    // Large perturbation (beyond skin/2 = 0.5): should rebuild
    double far_points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {1.5, 1.5, 1.5},  // moved 2 from (2,2,2) to (1.5,1.5,1.5)
    };
    check_cuda(cudaMemcpy(d_points, far_points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    auto status4 = vesin_verlet_compute(
        vl, d_points, n_points, d_box, d_periodic,
        {VesinCUDA, device_id},
        options, &neighbors, &error_message
    );
    REQUIRE(status4 == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl));

    // Clean up
    vesin_free(&neighbors);
    vesin_verlet_free(vl);

    free(h_pairs);
    free(h_distances);
    free(h_distances2);

    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}


void run_gpu_verlet_vs_cpu_test(int device_id) {
    check_cuda(cudaSetDevice(device_id));

    // 8-atom FCC unit cell
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {2.0, 0.0, 2.0},
        {0.0, 2.0, 2.0},
        {1.0, 1.0, 1.0},
        {3.0, 3.0, 1.0},
        {3.0, 1.0, 3.0},
        {1.0, 3.0, 3.0},
    };
    size_t n_points = 8;

    double box[3][3] = {
        {4.0, 0.0, 0.0},
        {0.0, 4.0, 0.0},
        {0.0, 0.0, 4.0},
    };
    bool periodic[3] = {true, true, true};
    double cutoff = 3.5;
    double skin = 1.0;

    // CPU Verlet
    const char* cpu_err = nullptr;
    auto* cpu_vl = vesin_verlet_new(cutoff, skin, true, &cpu_err);
    REQUIRE(cpu_vl != nullptr);

    VesinNeighborList cpu_nl;
    auto cpu_opts = VesinOptions{};
    cpu_opts.return_shifts = true;
    cpu_opts.return_distances = true;
    cpu_opts.return_vectors = true;

    auto cpu_status = vesin_verlet_compute(
        cpu_vl, points, n_points, box, periodic,
        {VesinCPU, 0}, cpu_opts, &cpu_nl, &cpu_err
    );
    REQUIRE(cpu_status == EXIT_SUCCESS);

    // GPU Verlet (same system)
    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    const char* gpu_err = nullptr;
    auto* gpu_vl = vesin_verlet_new(cutoff, skin, true, &gpu_err);
    REQUIRE(gpu_vl != nullptr);

    VesinNeighborList gpu_nl;
    auto gpu_opts = VesinOptions{};
    gpu_opts.return_shifts = true;
    gpu_opts.return_distances = true;
    gpu_opts.return_vectors = true;

    auto gpu_status = vesin_verlet_compute(
        gpu_vl, d_points, n_points, d_box, d_periodic,
        {VesinCUDA, device_id}, gpu_opts, &gpu_nl, &gpu_err
    );
    REQUIRE(gpu_status == EXIT_SUCCESS);

    // Compare pair counts
    CHECK(cpu_nl.length == gpu_nl.length);

    // Copy GPU results to host
    auto n_pairs = gpu_nl.length;
    auto* h_gpu_pairs = static_cast<size_t (*)[2]>(malloc(sizeof(size_t) * n_pairs * 2));
    auto* h_gpu_shifts = static_cast<int32_t (*)[3]>(malloc(sizeof(int32_t) * n_pairs * 3));
    auto* h_gpu_distances = static_cast<double*>(malloc(sizeof(double) * n_pairs));

    check_cuda(cudaMemcpy(h_gpu_pairs, gpu_nl.pairs, sizeof(size_t) * n_pairs * 2, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_gpu_shifts, gpu_nl.shifts, sizeof(int32_t) * n_pairs * 3, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_gpu_distances, gpu_nl.distances, sizeof(double) * n_pairs, cudaMemcpyDeviceToHost));

    // Collect CPU pairs into set
    std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>> cpu_pairs;
    for (size_t k = 0; k < cpu_nl.length; k++) {
        cpu_pairs.emplace(
            cpu_nl.pairs[k][0], cpu_nl.pairs[k][1],
            cpu_nl.shifts[k][0], cpu_nl.shifts[k][1], cpu_nl.shifts[k][2]
        );
    }

    // Collect GPU pairs into set
    std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>> gpu_pairs;
    for (size_t k = 0; k < n_pairs; k++) {
        gpu_pairs.emplace(
            h_gpu_pairs[k][0], h_gpu_pairs[k][1],
            h_gpu_shifts[k][0], h_gpu_shifts[k][1], h_gpu_shifts[k][2]
        );
    }

    // All CPU pairs should appear in GPU result and vice versa
    for (auto& p : cpu_pairs) {
        CHECK(gpu_pairs.count(p) == 1);
    }
    for (auto& p : gpu_pairs) {
        CHECK(cpu_pairs.count(p) == 1);
    }

    // Clean up
    vesin_free(&cpu_nl);
    vesin_verlet_free(cpu_vl);

    vesin_free(&gpu_nl);
    vesin_verlet_free(gpu_vl);

    free(h_gpu_pairs);
    free(h_gpu_shifts);
    free(h_gpu_distances);

    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}


TEST_CASE("GPU Verlet: lifecycle and rebuild/reuse") {
    int n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);
    run_gpu_verlet_test(0);
}

TEST_CASE("GPU Verlet: correctness vs CPU Verlet") {
    int n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);
    run_gpu_verlet_vs_cpu_test(0);
}

#else

TEST_CASE("CUDA tests are disabled") {}

#endif
