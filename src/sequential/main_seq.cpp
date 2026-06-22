#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>

using namespace std;

struct SimilarityResult {
    int document_id;
    float score;
};

bool compare_results(const SimilarityResult &a, const SimilarityResult &b) {
    return a.score > b.score;
}

int main(int argc, char **argv) {
    int num_documents = (argc > 1) ? atoi(argv[1]) : 200000;
    int DIMENSIONS    = (argc > 2) ? atoi(argv[2]) : 4096;
    int TOP_K         = (argc > 3) ? atoi(argv[3]) : 5;

    cout << "======================================================\n";
    cout << "  SEQUENTIAL EXACT SEARCH\n";
    cout << "======================================================\n";
    cout << "Documents : " << num_documents << "\n";
    cout << "Dims      : " << DIMENSIONS << "\n";
    cout << "Top-K     : " << TOP_K << "\n";

    vector<float> database(num_documents * DIMENSIONS);
    vector<float> query_vector(DIMENSIONS);
    vector<SimilarityResult> results(num_documents);

    srand(42);
    for (int d = 0; d < DIMENSIONS; d++)
        query_vector[d] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < num_documents * DIMENSIONS; i++)
        database[i] = static_cast<float>(rand()) / RAND_MAX;

    // plant a known exact match so we can verify correctness
    int plant_idx = (num_documents * 3) / 4;
    for (int d = 0; d < DIMENSIONS; d++)
        database[plant_idx * DIMENSIONS + d] = query_vector[d];

    cout << "Running search...\n";
    auto t0 = chrono::high_resolution_clock::now();

    for (int i = 0; i < num_documents; i++) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        int off = i * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; d++) {
            float a = query_vector[d];
            float b = database[off + d];
            dot += a * b;
            na  += a * a;
            nb  += b * b;
        }
        float score = (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (sqrt(na) * sqrt(nb));
        results[i] = {i, score};
    }

    partial_sort(results.begin(), results.begin() + TOP_K, results.end(), compare_results);

    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = t1 - t0;

    cout << "TOP MATCH: doc " << results[0].document_id
         << "  score=" << results[0].score << "\n";

    cout << "\n--- CSV BENCHMARK DATA ---\n";
    cout << "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime\n";
    cout << "Sequential,1,1," << num_documents << "," << DIMENSIONS << ",0.000,"
         << fixed << setprecision(5) << elapsed.count() << ","
         << elapsed.count() << "\n";
    cout << "======================================================\n";

    return 0;
}
