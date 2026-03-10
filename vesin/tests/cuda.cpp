#include <catch2/catch_test_macros.hpp>

#ifdef VESIN_TESTS_WITH_CUDA

#include <cmath>
#include <thread>

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

// Helper: run a GPU NL and a CPU NL, compare pair sets
static void compare_gpu_cpu(
    int device_id,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    bool full_list
) {
    check_cuda(cudaSetDevice(device_id));

    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    // GPU NL
    VesinNeighborList gpu_nl;
    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = full_list;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = false;

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        d_points, n_points, d_box, d_periodic,
        {VesinDeviceKind::VesinCUDA, device_id},
        options, &gpu_nl, &error_message
    );
    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    // Copy GPU results to host
    auto* h_pairs = static_cast<size_t*>(malloc(sizeof(size_t) * gpu_nl.length * 2));
    check_cuda(cudaMemcpy(h_pairs, gpu_nl.pairs, sizeof(size_t) * gpu_nl.length * 2, cudaMemcpyDeviceToHost));

    auto* h_shifts = static_cast<int32_t*>(malloc(sizeof(int32_t) * gpu_nl.length * 3));
    check_cuda(cudaMemcpy(h_shifts, gpu_nl.shifts, sizeof(int32_t) * gpu_nl.length * 3, cudaMemcpyDeviceToHost));

    // CPU NL
    VesinNeighborList cpu_nl;
    auto cpu_options = options;
    status = vesin_neighbors(
        points, n_points, box, periodic,
        {VesinDeviceKind::VesinCPU, 0},
        cpu_options, &cpu_nl, &error_message
    );
    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    // Build pair sets for comparison (ignoring order)
    std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>> gpu_set, cpu_set;
    for (size_t i = 0; i < gpu_nl.length; i++) {
        gpu_set.emplace(
            h_pairs[i * 2], h_pairs[i * 2 + 1],
            h_shifts[i * 3], h_shifts[i * 3 + 1], h_shifts[i * 3 + 2]
        );
    }
    for (size_t i = 0; i < cpu_nl.length; i++) {
        cpu_set.emplace(
            cpu_nl.pairs[i][0], cpu_nl.pairs[i][1],
            cpu_nl.shifts[i][0], cpu_nl.shifts[i][1], cpu_nl.shifts[i][2]
        );
    }

    CHECK(gpu_set == cpu_set);

    free(h_pairs);
    free(h_shifts);
    vesin_free(&gpu_nl);
    vesin_free(&cpu_nl);
    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

TEST_CASE("GPU cluster-pair: correctness vs CPU (orthogonal box)") {
    // N=512 triggers cluster-pair path (threshold=256)
    const size_t n = 512;
    double points[n][3];
    // Simple cubic lattice in a 10x10x10 box
    size_t idx = 0;
    for (int iz = 0; iz < 8; iz++) {
        for (int iy = 0; iy < 8; iy++) {
            for (int ix = 0; ix < 8; ix++) {
                points[idx][0] = ix * 1.25;
                points[idx][1] = iy * 1.25;
                points[idx][2] = iz * 1.25;
                idx++;
            }
        }
    }
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    compare_gpu_cpu(0, points, n, box, periodic, 3.0, true);
}

TEST_CASE("GPU cluster-pair: triclinic box") {
    const size_t n = 512;
    double points[n][3];
    size_t idx = 0;
    for (int iz = 0; iz < 8; iz++) {
        for (int iy = 0; iy < 8; iy++) {
            for (int ix = 0; ix < 8; ix++) {
                points[idx][0] = ix * 1.0 + iy * 0.3;
                points[idx][1] = iy * 1.0 + iz * 0.2;
                points[idx][2] = iz * 1.0;
                idx++;
            }
        }
    }
    double box[3][3] = {{8.0, 0.0, 0.0}, {2.4, 8.0, 0.0}, {1.6, 1.6, 8.0}};
    bool periodic[3] = {true, true, true};

    compare_gpu_cpu(0, points, n, box, periodic, 2.5, false);
}

TEST_CASE("GPU cluster-pair: repeated calls reuse buffers") {
    const size_t n = 512;
    double points[n][3];
    size_t idx = 0;
    for (int iz = 0; iz < 8; iz++) {
        for (int iy = 0; iy < 8; iy++) {
            for (int ix = 0; ix < 8; ix++) {
                points[idx][0] = ix * 1.25;
                points[idx][1] = iy * 1.25;
                points[idx][2] = iz * 1.25;
                idx++;
            }
        }
    }
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    check_cuda(cudaSetDevice(0));

    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n * 3, cudaMemcpyHostToDevice));

    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    VesinNeighborList gpu_nl;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = false;

    const char* error_message = nullptr;

    // Call 3 times, verify same result each time
    size_t first_length = 0;
    for (int call = 0; call < 3; call++) {
        auto status = vesin_neighbors(
            d_points, n, d_box, d_periodic,
            {VesinDeviceKind::VesinCUDA, 0},
            options, &gpu_nl, &error_message
        );
        REQUIRE(error_message == nullptr);
        REQUIRE(status == EXIT_SUCCESS);

        if (call == 0) {
            first_length = gpu_nl.length;
        } else {
            CHECK(gpu_nl.length == first_length);
        }
    }

    vesin_free(&gpu_nl);
    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

#else

TEST_CASE("CUDA tests are disabled") {}

#endif
