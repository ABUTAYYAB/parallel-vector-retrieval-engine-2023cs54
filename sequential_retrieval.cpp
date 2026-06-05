#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>

using namespace std;

struct SimilarityResult
{
    int document_id;
    float score;
};

bool compare_results(const SimilarityResult &a, const SimilarityResult &b)
{
    return a.score > b.score;
}

int main(int argc, char **argv)
{

    int num_documents = (argc > 1) ? atoi(argv[1]) : 200000;
    int DIMENSIONS = (argc > 2) ? atoi(argv[2]) : 4096;
    int TOP_K = (argc > 3) ? atoi(argv[3]) : 5;

    cout << "======================================================\n";
    cout << "  SEQUENTIAL EXACT SEARCH (k-NN) BASELINE\n";
    cout << "======================================================\n";
    cout << "Total Documents : " << num_documents << "\n";
    cout << "Vector Dims     : " << DIMENSIONS << " (Modern LLM Size)\n";
    cout << "Retrieving Top  : " << TOP_K << " matches\n";
    cout << "Allocating RAM... This may take a moment.\n";

    // 1D Flattened memory for maximum CPU cache alignment

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
    {
        database[plant_idx * DIMENSIONS + d] = query_vector[d];
    }


    
    cout << "Starting Sequential Search...\n";
    auto start_time = chrono::high_resolution_clock::now();

    // Core Math Loop
    for (int i = 0; i < num_documents; ++i)
    {
        float dot_product = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        int offset = i * DIMENSIONS;

        for (int d = 0; d < DIMENSIONS; ++d)
        {
            float val_a = query_vector[d];
            float val_b = database[offset + d];
            dot_product += val_a * val_b;
            norm_a += val_a * val_a;
            norm_b += val_b * val_b;
        }

        float score = (norm_a == 0.0f || norm_b == 0.0f) ? 0.0f : (dot_product / (sqrt(norm_a) * sqrt(norm_b)));
        results[i] = {i, score};
    }

    partial_sort(results.begin(), results.begin() + TOP_K, results.end(), compare_results);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> execution_time = end_time - start_time;

    cout << "\n🏆 TOP MATCH: Doc ID " << results[0].document_id << " | Score: " << results[0].score << "\n";

    cout << "\n--- CSV BENCHMARK DATA ---\n";
    cout << "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime\n";
    cout << "Sequential,1,1," << num_documents << "," << DIMENSIONS << ",0.000,"
         << fixed << setprecision(5) << execution_time.count() << ","
         << execution_time.count() << "\n";
    cout << "======================================================\n";

    return 0;
}