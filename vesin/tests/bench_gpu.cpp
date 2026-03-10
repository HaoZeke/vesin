// GPU micro-benchmark: measures steady-state neighbor list construction
// time with buffer reuse (no allocation overhead after first call).
//
// Reports both fresh-allocation and steady-state timings to quantify
// the cudaMalloc overhead vs actual compute time.

#include <catch2/catch_test_macros.hpp>

#ifdef VESIN_TESTS_WITH_CUDA

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <vesin.h>

static void check_cuda_bench(cudaError_t status) {
    if (status != cudaSuccess) {
        FAIL(cudaGetErrorString(status));
    }
}

static void generate_fcc(
    int n_repeat, double a,
    std::vector<double>& pos, double box[3][3]
) {
    double basis[4][3] = {
        {0,0,0}, {0.5,0.5,0}, {0.5,0,0.5}, {0,0.5,0.5}
    };
    int n = 4 * n_repeat * n_repeat * n_repeat;
    pos.resize(n * 3);
    int idx = 0;
    for (int ix = 0; ix < n_repeat; ix++)
    for (int iy = 0; iy < n_repeat; iy++)
    for (int iz = 0; iz < n_repeat; iz++)
    for (int b = 0; b < 4; b++) {
        pos[idx*3+0] = (ix + basis[b][0]) * a;
        pos[idx*3+1] = (iy + basis[b][1]) * a;
        pos[idx*3+2] = (iz + basis[b][2]) * a;
        idx++;
    }
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        box[i][j] = (i == j) ? n_repeat * a : 0.0;
}

using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

TEST_CASE("GPU steady-state benchmark", "[.benchmark]") {
    int repeats[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20};
    int n_sizes = sizeof(repeats) / sizeof(repeats[0]);
    double a = 5.43, cutoff = 5.0;
    bool periodic[3] = {true, true, true};
    int N_WARMUP = 5, N_ITER = 20;

    printf("\n%-8s  %-12s  %-12s  %-12s  %-8s\n",
           "N_atoms", "Fresh(ms)", "Steady(ms)", "CPU(ms)", "Speedup");
    printf("%-8s  %-12s  %-12s  %-12s  %-8s\n",
           "-------", "---------", "----------", "-------", "-------");

    for (int si = 0; si < n_sizes; si++) {
        std::vector<double> h_pos;
        double box[3][3];
        generate_fcc(repeats[si], a, h_pos, box);
        int n = (int)h_pos.size() / 3;

        VesinOptions opts{};
        opts.cutoff = cutoff;
        opts.full = true;
        opts.return_shifts = true;
        opts.return_distances = true;
        opts.return_vectors = true;
        opts.algorithm = VesinAutoAlgorithm;

        // CPU benchmark (reuses buffers)
        VesinNeighborList cpu_nl{};
        cpu_nl.device.type = VesinCPU;
        const char* err = nullptr;
        for (int i = 0; i < N_WARMUP; i++)
            vesin_neighbors((const double(*)[3])h_pos.data(), n,
                            box, periodic, {VesinCPU, 0}, opts, &cpu_nl, &err);
        double cpu_total = 0;
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = Clock::now();
            vesin_neighbors((const double(*)[3])h_pos.data(), n,
                            box, periodic, {VesinCPU, 0}, opts, &cpu_nl, &err);
            auto t1 = Clock::now();
            cpu_total += ms(t0, t1);
        }
        double cpu_ms = cpu_total / N_ITER;

        // GPU setup
        double *d_pos = nullptr, *d_box = nullptr;
        bool *d_per = nullptr;
        check_cuda_bench(cudaMalloc(&d_pos, sizeof(double) * n * 3));
        check_cuda_bench(cudaMalloc(&d_box, sizeof(double) * 9));
        check_cuda_bench(cudaMalloc(&d_per, sizeof(bool) * 3));
        check_cuda_bench(cudaMemcpy(d_pos, h_pos.data(), sizeof(double)*n*3, cudaMemcpyHostToDevice));
        check_cuda_bench(cudaMemcpy(d_box, box, sizeof(double)*9, cudaMemcpyHostToDevice));
        check_cuda_bench(cudaMemcpy(d_per, periodic, sizeof(bool)*3, cudaMemcpyHostToDevice));

        // Fresh allocation (new VesinNeighborList each call)
        double fresh_total = 0;
        for (int i = 0; i < N_WARMUP + 3; i++) {
            VesinNeighborList gpu_nl{};
            auto t0 = Clock::now();
            vesin_neighbors((const double(*)[3])d_pos, n,
                            (const double(*)[3])d_box, d_per,
                            {VesinCUDA, 0}, opts, &gpu_nl, &err);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            if (i >= N_WARMUP)
                fresh_total += ms(t0, t1);
            vesin_free(&gpu_nl);
        }
        double fresh_ms_val = fresh_total / 3.0;

        // Steady-state (reuse buffers)
        VesinNeighborList gpu_nl{};
        for (int i = 0; i < N_WARMUP; i++) {
            vesin_neighbors((const double(*)[3])d_pos, n,
                            (const double(*)[3])d_box, d_per,
                            {VesinCUDA, 0}, opts, &gpu_nl, &err);
            cudaDeviceSynchronize();
        }
        double steady_total = 0;
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = Clock::now();
            vesin_neighbors((const double(*)[3])d_pos, n,
                            (const double(*)[3])d_box, d_per,
                            {VesinCUDA, 0}, opts, &gpu_nl, &err);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            steady_total += ms(t0, t1);
        }
        double steady_ms_val = steady_total / N_ITER;

        printf("%-8d  %-12.3f  %-12.3f  %-12.3f  %-8.2f\n",
               n, fresh_ms_val, steady_ms_val, cpu_ms, cpu_ms / steady_ms_val);

        vesin_free(&gpu_nl);
        vesin_free(&cpu_nl);
        cudaFree(d_pos);
        cudaFree(d_box);
        cudaFree(d_per);
    }
}

#else
TEST_CASE("GPU benchmark (no CUDA)", "[.benchmark]") {
    SKIP("CUDA not available");
}
#endif
