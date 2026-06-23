#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <pthread.h>

using namespace std;

struct SimilarityResult {
    int   document_id;
    float score;
};

bool compare_results(const SimilarityResult &a, const SimilarityResult &b) {
    return a.score > b.score;
}

struct ThreadArgs {
    int thread_id;
    int num_threads;
    int num_documents;
    int dimensions;
    const float *database;
    const float *query_vector;
    SimilarityResult *results;
};

void *compute_similarity(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    int chunk = args->num_documents / args->num_threads;
    int start = args->thread_id * chunk;
    int end   = (args->thread_id == args->num_threads - 1)
                    ? args->num_documents
                    : start + chunk;

    int D = args->dimensions;

    for (int i = start; i < end; i++) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        int off = i * D;
        for (int d = 0; d < D; d++) {
            float a = args->query_vector[d];
            float b = args->database[off + d];
            dot += a * b;
            na  += a * a;
            nb  += b * b;
        }
        float score = (na == 0.0f || nb == 0.0f)
                          ? 0.0f
                          : dot / (sqrtf(na) * sqrtf(nb));
        // each thread writes to its own slice of results — no lock needed
        args->results[i] = {i, score};
    }
    return nullptr;
}

int main(int argc, char **argv) {
    int num_documents = (argc > 1) ? atoi(argv[1]) : 200000;
    int DIMENSIONS    = (argc > 2) ? atoi(argv[2]) : 4096;
    int TOP_K         = (argc > 3) ? atoi(argv[3]) : 5;
    int NUM_THREADS   = (argc > 4) ? atoi(argv[4]) : 4;

    cout << "======================================================\n";
    cout << "  PTHREADS EXACT SEARCH\n";
    cout << "======================================================\n";
    cout << "Documents : " << num_documents << "\n";
    cout << "Dims      : " << DIMENSIONS    << "\n";
    cout << "Top-K     : " << TOP_K         << "\n";
    cout << "Threads   : " << NUM_THREADS   << "\n";

    vector<float> database(num_documents * DIMENSIONS);
    vector<float> query_vector(DIMENSIONS);
    vector<SimilarityResult> results(num_documents);

    srand(42);
    for (int d = 0; d < DIMENSIONS; d++)
        query_vector[d] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < num_documents * DIMENSIONS; i++)
        database[i] = static_cast<float>(rand()) / RAND_MAX;

    int plant_idx = (num_documents * 3) / 4;
    for (int d = 0; d < DIMENSIONS; d++)
        database[plant_idx * DIMENSIONS + d] = query_vector[d];

    vector<pthread_t>  threads(NUM_THREADS);
    vector<ThreadArgs> targs(NUM_THREADS);

    auto t0 = chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_THREADS; t++) {
        targs[t] = {t, NUM_THREADS, num_documents, DIMENSIONS,
                    database.data(), query_vector.data(), results.data()};
        if (pthread_create(&threads[t], nullptr, compute_similarity, &targs[t]) != 0) {
            cerr << "pthread_create failed for thread " << t << "\n";
            return 1;
        }
    }

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], nullptr);

    partial_sort(results.begin(), results.begin() + TOP_K, results.end(), compare_results);

    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = t1 - t0;

    cout << "TOP MATCH: doc " << results[0].document_id
         << "  score=" << results[0].score << "\n";

    bool found = false;
    for (int i = 0; i < TOP_K; i++) {
        if (results[i].document_id == plant_idx) {
            cout << "Correctness PASSED: planted doc " << plant_idx
                 << " at rank " << (i+1) << "\n";
            found = true;
            break;
        }
    }
    if (!found)
        cout << "Correctness FAILED: planted doc not in top-" << TOP_K << "\n";

    cout << "\n--- CSV BENCHMARK DATA ---\n";
    cout << "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime\n";
    cout << "Pthreads,1," << NUM_THREADS << "," << num_documents << ","
         << DIMENSIONS << ",0.00000,"
         << fixed << setprecision(5) << elapsed.count() << ","
         << elapsed.count() << "\n";
    cout << "======================================================\n";

    return 0;
}
