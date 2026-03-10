/// Standalone C++ example demonstrating vesin Verlet neighbor list.
///
/// Simulates a short MD-like loop: perturbs positions each step, computes the
/// neighbor list, and reports how many steps reused the cached topology vs
/// performing a full rebuild.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vesin.h>

int main() {
    // Simple 4-atom system in a periodic box
    const size_t N = 4;
    double points[N][3] = {
        {0.0, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {0.0, 1.5, 0.0},
        {0.0, 0.0, 1.5},
    };
    double box[3][3] = {{5.0, 0, 0}, {0, 5.0, 0}, {0, 0, 5.0}};
    bool periodic[3] = {true, true, true};

    double cutoff = 3.0;
    double skin = 0.5;

    const char* error_message = nullptr;
    VesinVerletList* vl = vesin_verlet_new(cutoff, skin, true, &error_message);
    if (vl == nullptr) {
        fprintf(stderr, "Failed to create Verlet list: %s\n", error_message);
        return 1;
    }

    VesinNeighborList neighbors = {};

    VesinOptions options = {};
    options.cutoff = cutoff;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = false;

    int n_steps = 50;
    int rebuild_count = 0;
    int reuse_count = 0;

    printf("Running %d MD steps with cutoff=%.1f, skin=%.1f\n", n_steps, cutoff, skin);

    for (int step = 0; step < n_steps; step++) {
        int status = vesin_verlet_compute(
            vl, points, N, box, periodic,
            {VesinCPU, 0}, options, &neighbors, &error_message
        );
        if (status != 0) {
            fprintf(stderr, "Step %d: error: %s\n", step, error_message);
            vesin_verlet_free(vl);
            return 1;
        }

        if (vesin_verlet_did_rebuild(vl)) {
            rebuild_count++;
        } else {
            reuse_count++;
        }

        // Small perturbation
        for (size_t i = 0; i < N; i++) {
            double dx = 0.02 * sin(step * 1.1 + i * 0.3);
            double dy = 0.02 * sin(step * 1.3 + i * 0.7 + 1.0);
            double dz = 0.02 * sin(step * 1.7 + i * 1.1 + 2.0);
            points[i][0] += dx;
            points[i][1] += dy;
            points[i][2] += dz;
        }
    }

    printf("Completed: %d rebuilds, %d reuses (%.0f%% cache hit rate)\n",
           rebuild_count, reuse_count,
           100.0 * reuse_count / n_steps);
    printf("Final neighbor list: %zu pairs\n", neighbors.length);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);

    return 0;
}
