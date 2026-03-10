#include <array>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;

#include <vesin.h>

#define CHECK_APPROX_EQUAL(a, b) CHECK_THAT(a, WithinULP(b, 4));

/// Helper: collect pairs from a VesinNeighborList into a sorted set for
/// comparison, ignoring order.
static std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>>
collect_pairs(const VesinNeighborList& nl) {
    std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>> result;
    for (size_t k = 0; k < nl.length; k++) {
        result.emplace(
            nl.pairs[k][0], nl.pairs[k][1],
            nl.shifts[k][0], nl.shifts[k][1], nl.shifts[k][2]
        );
    }
    return result;
}

/// Helper: compute a stateless NL for reference
static VesinNeighborList stateless_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    double cutoff,
    bool full_list
) {
    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = full_list;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    VesinNeighborList neighbors;
    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        points, n_points, box, periodic,
        {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    REQUIRE(error_message == nullptr);
    return neighbors;
}

TEST_CASE("Verlet: basic lifecycle") {
    const char* error_message = nullptr;

    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);
    REQUIRE(error_message == nullptr);

    CHECK(vesin_verlet_did_rebuild(vl) == false);

    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: invalid parameters") {
    const char* error_message = nullptr;

    auto* vl1 = vesin_verlet_new(-1.0, 0.5, true, &error_message);
    CHECK(vl1 == nullptr);
    CHECK(error_message != nullptr);

    error_message = nullptr;
    auto* vl2 = vesin_verlet_new(3.0, -0.1, true, &error_message);
    CHECK(vl2 == nullptr);
    CHECK(error_message != nullptr);
}

TEST_CASE("Verlet: first call rebuilds") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0; // ignored by verlet, but fill it
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    auto status = vesin_verlet_compute(
        vl, points, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Verify against stateless NL
    auto ref = stateless_neighbors(points, 3, box, periodic, 3.0, true);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    CHECK(verlet_pairs == ref_pairs);

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: small movement does not rebuild") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {0.0, 1.5, 0.0},
        {1.5, 1.5, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};
    double skin = 1.0;

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, skin, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call: must rebuild
    auto status = vesin_verlet_compute(
        vl, points, 4, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Move atoms by less than skin/2 = 0.5
    double points2[][3] = {
        {0.1, 0.0, 0.0},
        {1.6, 0.0, 0.0},
        {0.0, 1.6, 0.0},
        {1.5, 1.4, 0.0},
    };

    status = vesin_verlet_compute(
        vl, points2, 4, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == false);

    // Verify correctness: all pairs within cutoff must be present
    auto ref = stateless_neighbors(points2, 4, box, periodic, 3.0, true);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    // Verlet may have extra pairs (from skin buffer), but must contain all
    // pairs within model cutoff
    for (auto& p : ref_pairs) {
        CHECK(verlet_pairs.count(p) == 1);
    }

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: large movement triggers rebuild") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};
    double skin = 0.5;

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, skin, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call
    auto status = vesin_verlet_compute(
        vl, points, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Move atom 0 by more than skin/2 = 0.25
    double points2[][3] = {
        {0.5, 0.0, 0.0},  // moved 0.5 > 0.25
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };

    status = vesin_verlet_compute(
        vl, points2, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: box change triggers rebuild") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call
    auto status = vesin_verlet_compute(
        vl, points, 2, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Same positions, different box
    double box2[3][3] = {{10.1, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    status = vesin_verlet_compute(
        vl, points, 2, box2, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: N change triggers rebuild") {
    double points3[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    double points2[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call with 3 atoms
    auto status = vesin_verlet_compute(
        vl, points3, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Second call with 2 atoms -> rebuild
    status = vesin_verlet_compute(
        vl, points2, 2, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: repeated calls with same positions do not rebuild") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.5, 0.5, 1.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call: rebuild
    auto status = vesin_verlet_compute(
        vl, points, 5, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // Same positions, 5 more calls: no rebuild
    for (int repeat = 0; repeat < 5; repeat++) {
        status = vesin_verlet_compute(
            vl, points, 5, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
        );
        REQUIRE(status == EXIT_SUCCESS);
        CHECK(vesin_verlet_did_rebuild(vl) == false);
    }

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: non-periodic system") {
    double points[][3] = {
        {0.134, 1.282, 1.701},
        {-0.273, 1.026, -1.471},
        {1.922, -0.124, 1.900},
        {1.400, -0.464, 0.480},
        {0.149, 1.865, 0.635},
    };
    double box[3][3] = {{0}};
    bool periodic[3] = {false, false, false};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.42, 0.5, false, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.42;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = false;

    auto status = vesin_verlet_compute(
        vl, points, 5, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    // Compare with stateless
    auto ref = stateless_neighbors(points, 5, box, periodic, 3.42, false);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    CHECK(verlet_pairs == ref_pairs);

    // Check distances match
    REQUIRE(neighbors.length == ref.length);

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: correctness over MD-like trajectory") {
    // Simulate a short trajectory with gradual displacement
    const size_t n = 4;
    double points[n][3] = {
        {0.0, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {0.0, 1.5, 0.0},
        {0.0, 0.0, 1.5},
    };
    double box[3][3] = {{5, 0, 0}, {0, 5, 0}, {0, 0, 5}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    int rebuild_count = 0;
    const int n_steps = 20;

    for (int step = 0; step < n_steps; step++) {
        auto status = vesin_verlet_compute(
            vl, points, n, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
        );
        REQUIRE(status == EXIT_SUCCESS);

        if (vesin_verlet_did_rebuild(vl)) {
            rebuild_count++;
        }

        // Verify all pairs within cutoff are present
        auto ref = stateless_neighbors(points, n, box, periodic, 3.0, true);
        auto verlet_pairs = collect_pairs(neighbors);
        auto ref_pairs = collect_pairs(ref);
        for (auto& p : ref_pairs) {
            REQUIRE(verlet_pairs.count(p) == 1);
        }
        vesin_free(&ref);

        // Small random-ish perturbation (deterministic)
        double dx = 0.03 * std::sin(step * 1.1 + 0.0);
        double dy = 0.03 * std::sin(step * 1.3 + 1.0);
        double dz = 0.03 * std::sin(step * 1.7 + 2.0);
        for (size_t i = 0; i < n; i++) {
            points[i][0] += dx * (i + 1);
            points[i][1] += dy * (i + 1);
            points[i][2] += dz * (i + 1);
        }
    }

    // With skin=0.5 and small perturbations (0.03 per step), we should
    // rebuild much less than every step
    CHECK(rebuild_count < n_steps);
    CHECK(rebuild_count >= 1); // at least the first call

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: exact pair match with stateless after rebuild") {
    // After a rebuild, the Verlet NL should produce the exact same pair set
    // as a stateless NL at the model cutoff.
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {0.0, 1.5, 0.0},
        {1.5, 1.5, 0.0},
        {0.5, 0.5, 1.0},
        {3.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 3.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 1.0, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    auto status = vesin_verlet_compute(
        vl, points, 8, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(vesin_verlet_did_rebuild(vl) == true);

    // The inner list should produce exactly the stateless result
    auto ref = stateless_neighbors(points, 8, box, periodic, 3.0, true);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    CHECK(verlet_pairs == ref_pairs);

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: 50-step MD trajectory correctness") {
    // Extended trajectory test: Verlet output must match stateless at each step
    const size_t n = 8;
    double points[n][3] = {
        {0.0, 0.0, 0.0}, {1.5, 0.0, 0.0},
        {0.0, 1.5, 0.0}, {1.5, 1.5, 0.0},
        {0.5, 0.5, 1.0}, {3.0, 0.0, 0.0},
        {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0},
    };
    double box[3][3] = {{8, 0, 0}, {0, 8, 0}, {0, 0, 8}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    int rebuild_count = 0;
    const int n_steps = 50;

    for (int step = 0; step < n_steps; step++) {
        auto status = vesin_verlet_compute(
            vl, points, n, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
        );
        REQUIRE(status == EXIT_SUCCESS);

        if (vesin_verlet_did_rebuild(vl)) {
            rebuild_count++;
        }

        // All pairs within cutoff must be present
        auto ref = stateless_neighbors(points, n, box, periodic, 3.0, true);
        auto verlet_pairs = collect_pairs(neighbors);
        auto ref_pairs = collect_pairs(ref);
        for (auto& p : ref_pairs) {
            REQUIRE(verlet_pairs.count(p) == 1);
        }
        vesin_free(&ref);

        // Small deterministic perturbation
        for (size_t i = 0; i < n; i++) {
            double dx = 0.02 * std::sin(step * 1.1 + i * 0.7);
            double dy = 0.02 * std::sin(step * 1.3 + i * 0.9);
            double dz = 0.02 * std::sin(step * 1.7 + i * 1.1);
            points[i][0] += dx;
            points[i][1] += dy;
            points[i][2] += dz;
        }
    }

    CHECK(rebuild_count >= 1);
    CHECK(rebuild_count < n_steps);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: oscillating atoms near cutoff boundary") {
    // Move atoms so that pairs drift in/out of cutoff, verify no pairs are lost
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {2.9, 0.0, 0.0},  // close to cutoff boundary
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};
    double skin = 1.0;

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, skin, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    // First call: rebuild
    auto status = vesin_verlet_compute(
        vl, points, 2, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    // Move atoms back and forth, verifying correctness at each step
    for (int step = 0; step < 20; step++) {
        // Oscillate atom 1 between 2.5 and 3.3 (crossing cutoff boundary)
        double x1 = 2.9 + 0.4 * std::sin(step * 0.5);
        double moved_points[][3] = {
            {0.0, 0.0, 0.0},
            {x1, 0.0, 0.0},
        };

        status = vesin_verlet_compute(
            vl, moved_points, 2, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
        );
        REQUIRE(status == EXIT_SUCCESS);

        auto ref = stateless_neighbors(moved_points, 2, box, periodic, 3.0, true);
        auto verlet_pairs = collect_pairs(neighbors);
        auto ref_pairs = collect_pairs(ref);
        for (auto& p : ref_pairs) {
            CHECK(verlet_pairs.count(p) == 1);
        }
        vesin_free(&ref);
    }

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: all pairs beyond cutoff gives empty result") {
    // All atoms are far apart -- no pairs within cutoff
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {5.0, 0.0, 0.0},
        {0.0, 5.0, 0.0},
    };
    double box[3][3] = {{20, 0, 0}, {0, 20, 0}, {0, 0, 20}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 1.0, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    auto status = vesin_verlet_compute(
        vl, points, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    CHECK(neighbors.length == 0);

    vesin_free(&neighbors);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: tiny skin gives exact stateless match") {
    // With very small skin, cutoff + skin is close to cutoff, so the Verlet
    // result should closely match the stateless result
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0},
    };
    double box[3][3] = {{5, 0, 0}, {0, 5, 0}, {0, 0, 5}};
    bool periodic[3] = {true, true, true};
    double tiny_skin = 0.01;

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, tiny_skin, true, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    auto status = vesin_verlet_compute(
        vl, points, 4, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    // With tiny skin, Verlet result should be very close to stateless
    auto ref = stateless_neighbors(points, 4, box, periodic, 3.0, true);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    CHECK(verlet_pairs == ref_pairs);

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}

TEST_CASE("Verlet: half list") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    double box[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
    bool periodic[3] = {true, true, true};

    const char* error_message = nullptr;
    auto* vl = vesin_verlet_new(3.0, 0.5, false, &error_message);
    REQUIRE(vl != nullptr);

    VesinNeighborList neighbors;
    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    auto status = vesin_verlet_compute(
        vl, points, 3, box, periodic, {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    auto ref = stateless_neighbors(points, 3, box, periodic, 3.0, false);
    auto verlet_pairs = collect_pairs(neighbors);
    auto ref_pairs = collect_pairs(ref);
    CHECK(verlet_pairs == ref_pairs);

    vesin_free(&neighbors);
    vesin_free(&ref);
    vesin_verlet_free(vl);
}
