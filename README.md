# Parallel Exact Top-*k* Vector Retrieval Engine

**Course:** Parallel and Distributed Algorithms (Pthreads, OpenMP, MPI)  
**Student:** Abu Tayyab · 2023CS54  
**Instructor:** Waqas Ali · University of Engineering and Technology, Lahore  


---

## Overview

Exact top-*k* cosine-similarity search over dense vector embeddings is the core retrieval primitive in retrieval-augmented generation (RAG) systems. This project implements and benchmarks **five versions** of the same kernel:

| Version | Source file | Parallelism |
|---------|-------------|-------------|
| Sequential baseline | `src/sequential/main_seq.cpp` | None — reference implementation |
| Pthreads | `src/pthreads/main_pth.cpp` | POSIX threads, static partition |
| OpenMP | `src/openmp/main_omp.cpp` | `#pragma omp parallel for schedule(static)` |
| MPI-only | `src/mpi/main_mpi.cpp` | MPI Scatter / Bcast / Gather, single-threaded ranks |
| Hybrid MPI+OpenMP | `src/hybrid/main_hybrid.cpp` | MPI across ranks + OpenMP within each rank |

**Best result:** Pthreads 8 threads → **5.71× speedup** over sequential  
(T = 0.098 s vs T_seq = 0.558 s, N = 200,000 docs, D = 2,048 dims, k = 5)

---

## Dependencies

| Tool | Required version | Notes |
|------|-----------------|-------|
| `g++` | ≥ 9.0 | Tested with GCC 13.3.0 (Ubuntu 24.04) |
| OpenMPI | ≥ 4.0 | Tested with OpenMPI 4.1.6 |
| `mpicxx` | ≥ 4.0 | Wrapper included with OpenMPI |
| Python 3 | ≥ 3.8 | For plotting only; tested with 3.12.3 |
| matplotlib | ≥ 3.5 | In `src/venv/` — no system install needed |
| pandas | ≥ 1.3 | In `src/venv/` |

### Install dependencies (Ubuntu / Debian)

```bash
sudo apt-get install g++ openmpi-bin libopenmpi-dev
```

---

## Repository Structure

```
.
├── src/
│   ├── sequential/
│   │   └── main_seq.cpp          # Sequential baseline
│   ├── pthreads/
│   │   └── main_pth.cpp          # Pthreads version
│   ├── openmp/
│   │   └── main_omp.cpp          # OpenMP version
│   ├── mpi/
│   │   └── main_mpi.cpp          # MPI-only version
│   └── hybrid/
│       └── main_hybrid.cpp       # Hybrid MPI+OpenMP version
│
├── scripts/
│   ├── run_experiments.sh        # Compiles all 5 + runs full benchmark matrix
│   └── plot_results.py           # Generates all 5 figures from CSV
│
├── docs/
│   ├── paper.tex                 # IEEE IEEEtran paper (Overleaf-ready)
│   └── appendices.pdf            # Gantt Chart (App. A) + Logbook (App. B)
│
├── results/
│   ├── benchmark_results.csv     # Combined CSV — all 66 benchmark rows
│   ├── raw_outputs/              # One .txt file per run (66 files)
│   └── figures/                  # All 5 generated figures (300 DPI PNG)
│
└── README.md
```

---

## Compilation

All commands are run from the **project root**.

### Sequential

```bash
g++ -O3 src/sequential/main_seq.cpp -o src/sequential/main_seq -lm
```

### Pthreads

```bash
g++ -O3 -pthread src/pthreads/main_pth.cpp -o src/pthreads/main_pth -lm
```

### OpenMP

```bash
g++ -O3 -fopenmp src/openmp/main_omp.cpp -o src/openmp/main_omp -lm
```

### MPI-only

```bash
mpicxx -O3 src/mpi/main_mpi.cpp -o src/mpi/main_mpi
```

### Hybrid MPI+OpenMP

```bash
mpicxx -O3 -fopenmp src/hybrid/main_hybrid.cpp -o src/hybrid/main_hybrid
```

---

## Running Experiments

All binaries accept the same positional arguments:

```
<binary> [documents] [dimensions] [top_k] [threads]   # threads: Pthreads only
```

Defaults: `documents=200000  dimensions=4096  top_k=5  threads=4`

### Sequential

```bash
src/sequential/main_seq 200000 2048 5
```

### Pthreads

```bash
src/pthreads/main_pth 200000 2048 5 4    # 4 threads
src/pthreads/main_pth 200000 2048 5 8    # 8 threads
```

### OpenMP

```bash
OMP_NUM_THREADS=4 src/openmp/main_omp 200000 2048 5
OMP_NUM_THREADS=8 src/openmp/main_omp 200000 2048 5
```

### MPI-only

```bash
mpirun -np 2 src/mpi/main_mpi 200000 2048 5
mpirun -np 4 src/mpi/main_mpi 200000 2048 5
mpirun --oversubscribe -np 8 src/mpi/main_mpi 200000 2048 5   # >4 ranks on 4-core machine
```

### Hybrid MPI+OpenMP

```bash
OMP_NUM_THREADS=2 mpirun -np 2 src/hybrid/main_hybrid 200000 2048 5   # 2r × 2t
OMP_NUM_THREADS=2 mpirun -np 4 src/hybrid/main_hybrid 200000 2048 5   # 4r × 2t
```

---

## Full Benchmark Suite (Reproducibility)

The script compiles all five binaries and runs all three benchmark matrices documented in the paper and logbook:

```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

| Matrix | Fixed | Varied | Runs |
|--------|-------|--------|------|
| A — Dimensionality scaling | N = 100k, k = 5 | D ∈ {128, 512, 768, 1536, 3072, 4096} | 30 |
| B — Strong scaling | N = 200k, D = 2048, k = 5 | threads/ranks ∈ {1, 2, 4, 8} | 22 |
| C — Weak scaling | D = 2048, k = 5, 50k docs/unit | N ∈ {50k, 100k, 200k, 400k} | 12 |

Output is written to:
- `results/raw_outputs/<label>.txt` — full stdout per run
- `results/benchmark_results.csv` — combined CSV, one row per run

> **Crucial:** The TA can verify results by running `./scripts/run_experiments.sh` and comparing `results/benchmark_results.csv` against the tables and logbook in `docs/appendices.pdf`. Trends will match; individual timings vary ±10% with system load.

---

## Generate Figures

```bash
src/venv/bin/python3 scripts/plot_results.py
```

Reads `results/benchmark_results.csv` and writes five 300-DPI PNGs to `results/figures/`:

| Figure | Shows |
|--------|-------|
| `fig1_dimensionality_speedup.png` | Speedup vs. D — memory-bandwidth plateau above D = 1,536 |
| `fig2_strong_scaling_speedup.png` | Speedup vs. threads/ranks — Pthreads best at 5.71× |
| `fig3_strong_scaling_efficiency.png` | Efficiency (%) — MPI collapses below 4 ranks on single node |
| `fig4_weak_scaling.png` | Runtime vs. N — Pthreads near-ideal to 4 units |
| `fig5_execution_time_bar.png` | All 17 configurations at a glance |

See `results/figures/FIGURES.md` for detailed per-figure analysis.

---

## Hardware & Software Environment

| Item | Detail |
|------|--------|
| CPU | Intel Core i5-1135G7 — 4 physical cores / 8 logical (Hyper-Threading) |
| Clock | 2.40 GHz base / 4.20 GHz boost |
| RAM | 8 GB DDR4-3200 |
| Cache | 6 MB L3, 128 KB L1-D per core |
| OS | Ubuntu Linux 22.04 LTS |
| Compiler | g++ 13.3.0 · flags: `-O3` |
| MPI | OpenMPI 4.1.6 · compiler wrapper: `mpicxx -O3 [-fopenmp]` |
| Python | 3.12.3 with matplotlib 3.10.9, pandas 3.0.3, numpy 2.4.6 |

> Runs with more than 4 MPI ranks use `mpirun --oversubscribe` because the machine has only 4 physical cores. This is noted in the paper's Experimental Setup section.

---

## Key Results Summary

### Strong Scaling (N = 200,000, D = 2,048, k = 5)

| Configuration | Time (s) | Speedup | Efficiency |
|---------------|----------|---------|------------|
| Sequential (baseline) | 0.558 | 1.00× | 100.0% |
| Pthreads 4 threads | 0.271 | 2.06× | 51.5% |
| **Pthreads 8 threads** | **0.098** | **5.71×** | **71.4%** |
| OpenMP 4 threads | 0.147 | 3.80× | 95.0% |
| OpenMP 8 threads | 0.105 | 5.30× | 66.3% |
| MPI 2 ranks | 0.374 | 1.49× | 74.7% |
| MPI 4 ranks | 0.257 | 2.17× | 54.3% |
| Hybrid 2r × 2t | 0.182 | 3.07× | 76.7% |
| Hybrid 4r × 1t | 0.275 | 2.03× | 50.7% |

### Dimensionality Scaling (N = 100,000, 4 threads/ranks, k = 5)

Speedup rises from ~3.2× at D = 128 to ~3.8× at D = 4,096, then plateaus — consistent with memory-bandwidth saturation (Roofline model).

---

## Correctness Verification

Every parallel version plants a known exact match at position `⌊3N/4⌋` using the same fixed seed (42) as the sequential baseline. Each binary prints a correctness check line on every run:

```
✅ Correctness Check PASSED: Planted doc (ID=150000) found at rank 1 with score 1.000000
```

All five versions produce identical top-*k* results within floating-point tolerance.

---

## Known Limitations

- Integer division `N / world_size` silently discards remainder documents when `N % world_size ≠ 0`. Use corpus sizes divisible by the rank count.
- Query norm is recomputed inside the inner loop (one redundant `sqrt` per document). Precomputing it once would reduce arithmetic by ~30%.
- The final top-*k* merge on MPI rank 0 is serial; a distributed reduction would improve large-rank scaling.
- All MPI runs are on a single physical node. Efficiency results for MPI will improve significantly on a true multi-node cluster.

---

## Academic Integrity

All parallel code and the paper were written by the project authors. No code was copied from external repositories. The `rand()` / `srand(42)` data generator, the cosine-similarity formula, and the `partial_sort` top-*k* approach follow standard C++ idioms documented in the cited textbooks and papers. Any reference to external sources is cited in `docs/paper.tex`.
