#!/bin/bash
# =============================================================================
#  run_experiments.sh
#  Compiles all five versions (Sequential, Pthreads, OpenMP, MPI, Hybrid) and
#  runs the full benchmark matrix documented in the paper:
#
#    Matrix A — Dimensionality scaling  (N=100k, D varies, fixed parallelism)
#    Matrix B — Strong scaling          (N=200k, D=2048, parallelism varies)
#    Matrix C — Weak scaling            (D=2048, 50k docs per thread/rank)
#
#  Usage:
#    chmod +x scripts/run_experiments.sh
#    ./scripts/run_experiments.sh
#
#  Output:
#    results/raw_outputs/          — one .txt file per run
#    results/benchmark_results.csv — combined CSV for all runs
# =============================================================================

set -e

# ─── Paths (all relative — no hardcoded paths) ────────────────────────────────
SEQ_SRC="src/sequential/main_seq.cpp"
PTH_SRC="src/pthreads/main_pth.cpp"
OMP_SRC="src/openmp/main_omp.cpp"
MPI_SRC="src/mpi/main_mpi.cpp"
HYB_SRC="src/hybrid/main_hybrid.cpp"

SEQ_BIN="src/sequential/main_seq"
PTH_BIN="src/pthreads/main_pth"
OMP_BIN="src/openmp/main_omp"
MPI_BIN="src/mpi/main_mpi"
HYB_BIN="src/hybrid/main_hybrid"

RAW_DIR="results/raw_outputs"
CSV="results/benchmark_results.csv"

# ─── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$RAW_DIR"

echo "Algorithm,Nodes,Threads,Documents,Dimensions,CommTime,CompTime,TotalTime" > "$CSV"

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ─── Helper: run one binary, capture output, append CSV line ──────────────────
run_and_log() {
    local label="$1"
    local cmd="$2"
    local outfile="$RAW_DIR/${label}.txt"

    echo "  Running: $cmd"
    eval "$cmd" > "$outfile" 2>&1

    local csv_line
    csv_line=$(grep -A1 "Algorithm,Nodes" "$outfile" | tail -n1)

    if [ -n "$csv_line" ]; then
        echo "$csv_line" >> "$CSV"
        echo "  ✅ Logged: $csv_line"
    else
        echo "  ⚠️  WARNING: No CSV line found in output of $label"
    fi
}

# =============================================================================
#  STEP 1: Compile all five versions
# =============================================================================
banner "COMPILING ALL VERSIONS"

echo "[1/5] Compiling Sequential..."
g++ -O3 "$SEQ_SRC" -o "$SEQ_BIN" -lm
echo "      ✅ $SEQ_BIN"

echo "[2/5] Compiling Pthreads..."
g++ -O3 -pthread "$PTH_SRC" -o "$PTH_BIN" -lm
echo "      ✅ $PTH_BIN"

echo "[3/5] Compiling OpenMP..."
g++ -O3 -fopenmp "$OMP_SRC" -o "$OMP_BIN" -lm
echo "      ✅ $OMP_BIN"

echo "[4/5] Compiling MPI-only..."
mpicxx -O3 "$MPI_SRC" -o "$MPI_BIN"
echo "      ✅ $MPI_BIN"

echo "[5/5] Compiling Hybrid MPI+OpenMP..."
mpicxx -O3 -fopenmp "$HYB_SRC" -o "$HYB_BIN"
echo "      ✅ $HYB_BIN"

echo ""
echo "All five binaries compiled successfully."

# =============================================================================
#  MATRIX A: Dimensionality Scaling
#  N=100,000 | D in {128,512,768,1536,3072,4096} | k=5
#  Fixed parallelism: Pthreads 4t | OpenMP 4t | MPI-only 2r & 4r | Hybrid 2x2
# =============================================================================
banner "MATRIX A: DIMENSIONALITY SCALING (N=100000)"

DIMS=(128 512 768 1536 3072 4096)

echo ""
echo "--- Sequential ---"
for D in "${DIMS[@]}"; do
    run_and_log "seq_N100000_D${D}" \
        "$SEQ_BIN 100000 $D 5"
done

echo ""
echo "--- Pthreads 4 threads ---"
for D in "${DIMS[@]}"; do
    run_and_log "pth_4t_N100000_D${D}" \
        "$PTH_BIN 100000 $D 5 4"
done

echo ""
echo "--- OpenMP 4 threads ---"
for D in "${DIMS[@]}"; do
    run_and_log "omp_4t_N100000_D${D}" \
        "OMP_NUM_THREADS=4 $OMP_BIN 100000 $D 5"
done

echo ""
echo "--- MPI-only 2 ranks ---"
for D in "${DIMS[@]}"; do
    run_and_log "mpi_2r_N100000_D${D}" \
        "mpirun -np 2 $MPI_BIN 100000 $D 5"
done

echo ""
echo "--- MPI-only 4 ranks ---"
for D in "${DIMS[@]}"; do
    run_and_log "mpi_4r_N100000_D${D}" \
        "mpirun -np 4 $MPI_BIN 100000 $D 5"
done

echo ""
echo "--- Hybrid 2 ranks x 2 threads ---"
for D in "${DIMS[@]}"; do
    run_and_log "hybrid_2x2_N100000_D${D}" \
        "OMP_NUM_THREADS=2 mpirun -np 2 $HYB_BIN 100000 $D 5"
done

# =============================================================================
#  MATRIX B: Strong Scaling
#  N=200,000 | D=2,048 | k=5 — vary threads/ranks 1→8 for all models
# =============================================================================
banner "MATRIX B: STRONG SCALING (N=200000, D=2048)"

echo ""
echo "--- Sequential baseline ---"
run_and_log "seq_N200000_D2048" \
    "$SEQ_BIN 200000 2048 5"

echo ""
echo "--- Pthreads: 1, 2, 4, 8 threads ---"
for T in 1 2 4 8; do
    run_and_log "pth_${T}t_N200000_D2048" \
        "$PTH_BIN 200000 2048 5 $T"
done

echo ""
echo "--- OpenMP: 1, 2, 4, 8 threads ---"
for T in 1 2 4 8; do
    run_and_log "omp_${T}t_N200000_D2048" \
        "OMP_NUM_THREADS=$T $OMP_BIN 200000 2048 5"
done

echo ""
echo "--- MPI-only: 1, 2, 4 ranks ---"
for P in 1 2 4; do
    run_and_log "mpi_${P}r_N200000_D2048" \
        "mpirun -np $P $MPI_BIN 200000 2048 5"
done

echo ""
echo "--- MPI-only: 8 ranks (oversubscribed) ---"
run_and_log "mpi_8r_N200000_D2048" \
    "mpirun --oversubscribe -np 8 $MPI_BIN 200000 2048 5"

echo ""
echo "--- Hybrid configurations ---"

# 2 ranks x 1 thread  (total 2 units — baseline for hybrid)
run_and_log "hybrid_2x1_N200000_D2048" \
    "OMP_NUM_THREADS=1 mpirun -np 2 $HYB_BIN 200000 2048 5"

# 2 ranks x 2 threads  (total 4 units)
run_and_log "hybrid_2x2_N200000_D2048" \
    "OMP_NUM_THREADS=2 mpirun -np 2 $HYB_BIN 200000 2048 5"

# 2 ranks x 4 threads  (total 8 units)
run_and_log "hybrid_2x4_N200000_D2048" \
    "OMP_NUM_THREADS=4 mpirun -np 2 $HYB_BIN 200000 2048 5"

# 4 ranks x 1 thread  (total 4 units)
run_and_log "hybrid_4x1_N200000_D2048" \
    "OMP_NUM_THREADS=1 mpirun -np 4 $HYB_BIN 200000 2048 5"

# 4 ranks x 2 threads  (total 8 units — best hybrid config in paper)
run_and_log "hybrid_4x2_N200000_D2048" \
    "OMP_NUM_THREADS=2 mpirun -np 4 $HYB_BIN 200000 2048 5"

# =============================================================================
#  MATRIX C: Weak Scaling
#  D=2,048 | k=5 | 50,000 documents per thread/rank
#  N scales with parallelism: 1→50k, 2→100k, 4→200k, 8→400k
# =============================================================================
banner "MATRIX C: WEAK SCALING (D=2048, 50k docs per unit)"

echo ""
echo "--- Pthreads weak scaling ---"
run_and_log "weak_pth_1t"  "$PTH_BIN  50000 2048 5 1"
run_and_log "weak_pth_2t"  "$PTH_BIN 100000 2048 5 2"
run_and_log "weak_pth_4t"  "$PTH_BIN 200000 2048 5 4"
run_and_log "weak_pth_8t"  "$PTH_BIN 400000 2048 5 8"

echo ""
echo "--- OpenMP weak scaling ---"
run_and_log "weak_omp_1t"  "OMP_NUM_THREADS=1 $OMP_BIN  50000 2048 5"
run_and_log "weak_omp_2t"  "OMP_NUM_THREADS=2 $OMP_BIN 100000 2048 5"
run_and_log "weak_omp_4t"  "OMP_NUM_THREADS=4 $OMP_BIN 200000 2048 5"
run_and_log "weak_omp_8t"  "OMP_NUM_THREADS=8 $OMP_BIN 400000 2048 5"

echo ""
echo "--- MPI-only weak scaling ---"
run_and_log "weak_mpi_1r"  "mpirun -np 1 $MPI_BIN  50000 2048 5"
run_and_log "weak_mpi_2r"  "mpirun -np 2 $MPI_BIN 100000 2048 5"
run_and_log "weak_mpi_4r"  "mpirun -np 4 $MPI_BIN 200000 2048 5"
run_and_log "weak_mpi_8r"  "mpirun --oversubscribe -np 8 $MPI_BIN 400000 2048 5"

# =============================================================================
#  STEP 4: Summary
# =============================================================================
banner "BENCHMARK COMPLETE"

echo "Raw output files  : $RAW_DIR/"
echo "Combined CSV      : $CSV"
echo ""
echo "CSV Preview (first 10 lines):"
echo "----------------------------------------------"
head -10 "$CSV"
echo "----------------------------------------------"
echo ""
echo "Total runs logged : $(( $(wc -l < "$CSV") - 1 ))"
echo ""
echo "✅ Done. Include results/ in your submission ZIP."
