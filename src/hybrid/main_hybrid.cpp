#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

using namespace std;

struct SimilarityResult {
    int   document_id;
    float score;
};

bool compare_results(const SimilarityResult &a, const SimilarityResult &b) {
    return a.score > b.score;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int total_documents = (argc > 1) ? atoi(argv[1]) : 200000;
    int DIMENSIONS      = (argc > 2) ? atoi(argv[2]) : 4096;
    int TOP_K           = (argc > 3) ? atoi(argv[3]) : 5;

    int local_documents = total_documents / world_size;

    vector<float> global_database;
    vector<float> local_database(local_documents * DIMENSIONS);
    vector<float> query_vector(DIMENSIONS);

    if (world_rank == 0) {
        cout << "======================================================\n";
        cout << "  HYBRID MPI+OPENMP EXACT SEARCH\n";
        cout << "======================================================\n";
        cout << "Documents : " << total_documents << "\n";
        cout << "Dims      : " << DIMENSIONS      << "\n";
        cout << "Top-K     : " << TOP_K           << "\n";
        cout << "Ranks     : " << world_size       << "\n";
        cout << "Docs/rank : " << local_documents  << "\n";

        global_database.resize(total_documents * DIMENSIONS);
        srand(42);

        for (int d = 0; d < DIMENSIONS; ++d)
            query_vector[d] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < total_documents * DIMENSIONS; ++i)
            global_database[i] = static_cast<float>(rand()) / RAND_MAX;

        int plant_idx = (total_documents * 3) / 4;
        for (int d = 0; d < DIMENSIONS; ++d)
            global_database[plant_idx * DIMENSIONS + d] = query_vector[d];
    }

    MPI_Scatter(global_database.data(), local_documents * DIMENSIONS, MPI_FLOAT,
                local_database.data(),  local_documents * DIMENSIONS, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    double t_start_total = MPI_Wtime();

    double t_comm_start = MPI_Wtime();
    MPI_Bcast(query_vector.data(), DIMENSIONS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double t_comm_end = MPI_Wtime();
    double comm_time  = t_comm_end - t_comm_start;

    vector<SimilarityResult> local_results(local_documents);

    double t_comp_start = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_documents; ++i) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        int offset = i * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; ++d) {
            float a = query_vector[d];
            float b = local_database[offset + d];
            dot += a * b;
            na  += a * a;
            nb  += b * b;
        }
        float score = (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (sqrtf(na) * sqrtf(nb));
        local_results[i] = {(world_rank * local_documents) + i, score};
    }
    double t_comp_end = MPI_Wtime();
    double comp_time  = t_comp_end - t_comp_start;

    partial_sort(local_results.begin(), local_results.begin() + TOP_K,
                 local_results.end(), compare_results);

    vector<int>   local_top_ids(TOP_K);
    vector<float> local_top_scores(TOP_K);
    for (int i = 0; i < TOP_K; ++i) {
        local_top_ids[i]    = local_results[i].document_id;
        local_top_scores[i] = local_results[i].score;
    }

    t_comm_start = MPI_Wtime();
    vector<int>   gathered_ids;
    vector<float> gathered_scores;
    if (world_rank == 0) {
        gathered_ids.resize(world_size * TOP_K);
        gathered_scores.resize(world_size * TOP_K);
    }

    MPI_Gather(local_top_ids.data(),    TOP_K, MPI_INT,
               gathered_ids.data(),     TOP_K, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Gather(local_top_scores.data(), TOP_K, MPI_FLOAT,
               gathered_scores.data(),  TOP_K, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm_end = MPI_Wtime();
    comm_time += (t_comm_end - t_comm_start);

    double t_end_total = MPI_Wtime();

    if (world_rank == 0) {
        vector<SimilarityResult> global_candidates(world_size * TOP_K);
        for (int i = 0; i < world_size * TOP_K; ++i)
            global_candidates[i] = {gathered_ids[i], gathered_scores[i]};

        partial_sort(global_candidates.begin(), global_candidates.begin() + TOP_K,
                     global_candidates.end(), compare_results);

        cout << "TOP MATCH: doc " << global_candidates[0].document_id
             << "  score=" << global_candidates[0].score << "\n";

        int plant_idx = (total_documents * 3) / 4;
        bool found = false;
        for (int i = 0; i < TOP_K; ++i) {
            if (global_candidates[i].document_id == plant_idx) {
                cout << "Correctness PASSED: planted doc " << plant_idx
                     << " at rank " << (i + 1) << "\n";
                found = true;
                break;
            }
        }
        if (!found)
            cout << "Correctness FAILED: planted doc not in top-" << TOP_K << "\n";

        int num_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            num_threads = omp_get_num_threads();
        }

        cout << "\n--- CSV BENCHMARK DATA ---\n";
        cout << "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime\n";
        cout << "Hybrid," << world_size << "," << num_threads << "," << total_documents << ","
             << DIMENSIONS << "," << fixed << setprecision(5) << comm_time << ","
             << comp_time << "," << (t_end_total - t_start_total) << "\n";
        cout << "======================================================\n";
    }

    MPI_Finalize();
    return 0;
}
