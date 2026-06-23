#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

using namespace std;

struct SimilarityResult {
    int   document_id;
    float score;
};

bool compare_results(const SimilarityResult &a, const SimilarityResult &b) {
    return a.score > b.score;
}

int main(int argc, char **argv) {
    int num_documents = (argc > 1) ? atoi(argv[1]) : 200000;
    int DIMENSIONS    = (argc > 2) ? atoi(argv[2]) : 4096;
    int TOP_K         = (argc > 3) ? atoi(argv[3]) : 5;

    int num_threads = omp_get_max_threads();

    cout << "======================================================\n";
    cout << "  OPENMP EXACT SEARCH\n";
    cout << "======================================================\n";
    cout << "Documents : " << num_documents << "\n";
    cout << "Dims      : " << DIMENSIONS    << "\n";
    cout << "Top-K     : " << TOP_K         << "\n";
    cout << "Threads   : " << num_threads   << "\n";

    vector<float> database(num_documents * DIMENSIONS);
    vector<float> query_vector(DIMENSIONS);
    vector<SimilarityResult> results(num_documents);

    srand(42);
    for (int d = 0; d < DIMENSIONS; ++d)
        query_vector[d] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < num_documents * DIMENSIONS; ++i)
        database[i] = static_cast<float>(rand()) / RAND_MAX;

    int plant_idx = (num_documents * 3) / 4;
    for (int d = 0; d < DIMENSIONS; ++d)
        database[plant_idx * DIMENSIONS + d] = query_vector[d];

    auto start_time = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_documents; ++i) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        int offset = i * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; ++d) {
            float a = query_vector[d];
            float b = database[offset + d];
            dot += a * b;
            na  += a * a;
            nb  += b * b;
        }
        results[i] = {i, (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (sqrtf(na) * sqrtf(nb))};
    }

    partial_sort(results.begin(), results.begin() + TOP_K, results.end(), compare_results);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    cout << "TOP MATCH: doc " << results[0].document_id
         << "  score=" << results[0].score << "\n";

    bool found = false;
    for (int i = 0; i < TOP_K; ++i) {
        if (results[i].document_id == plant_idx) {
            cout << "Correctness PASSED: planted doc " << plant_idx
                 << " at rank " << (i + 1) << "\n";
            found = true;
            break;
        }
    }
    if (!found)
        cout << "Correctness FAILED: planted doc not in top-" << TOP_K << "\n";

    cout << "\n--- CSV BENCHMARK DATA ---\n";
    cout << "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime\n";
    cout << "OpenMP,1," << num_threads << "," << num_documents << ","
         << DIMENSIONS << ",0.00000,"
         << fixed << setprecision(5) << elapsed.count() << ","
         << elapsed.count() << "\n";
    cout << "======================================================\n";

    return 0;
}
