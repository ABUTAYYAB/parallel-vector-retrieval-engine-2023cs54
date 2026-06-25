#!/usr/bin/env python3
"""
plot_results.py
Generates all benchmark figures for the Parallel Vector Retrieval Engine paper.

Run from the project root:
    src/venv/bin/python3 scripts/plot_results.py

Output: results/figures/
    fig1_dimensionality_speedup.png   — Speedup vs. embedding dimension (4 models)
    fig2_strong_scaling_speedup.png   — Speedup vs. threads/ranks (strong scaling)
    fig3_strong_scaling_efficiency.png — Parallel efficiency vs. threads/ranks
    fig4_weak_scaling.png             — Runtime vs. N (weak scaling)
    fig5_execution_time_bar.png       — Execution time bar chart, all configs
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
CSV_PATH   = "results/benchmark_results.csv"
FIGURE_DIR = "results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ─── Publication-style rc settings ───────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   11,
    "axes.titleweight": "bold",
    "legend.fontsize":  9,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   ":",
    "lines.linewidth":  2.0,
    "lines.markersize": 7,
})

# ─── Consistent per-model style ──────────────────────────────────────────────
STYLE = {
    "Pthreads": dict(color="#1a6fb3", marker="o", ls="-"),
    "OpenMP":   dict(color="#27ae60", marker="s", ls="-"),
    "MPI":      dict(color="#c0392b", marker="^", ls="-"),
    "Hybrid":   dict(color="#e67e22", marker="D", ls="-"),
    "Ideal":    dict(color="#999999", marker="",  ls="--", lw=1.4),
}

COLOR_MAP = {
    "Sequential": "#777777",
    "Pthreads":   "#1a6fb3",
    "OpenMP":     "#27ae60",
    "MPI":        "#c0392b",
    "Parallel":   "#e67e22",
}

# ─── Load & deduplicate data ──────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# The script appends strong-scaling rows first, then weak-scaling rows.
# drop_duplicates(keep="first") retains the strong-scaling measurement
# when a (algo, nodes, threads, N, D) combination appears in both matrices.
df = df.drop_duplicates(
    subset=["Algorithm", "Nodes", "Threads", "Documents", "Dimensions"],
    keep="first"
).reset_index(drop=True)

# Sequential reference point for all speedup calculations
SEQ_200K = df[
    (df.Algorithm == "Sequential") &
    (df.Documents == 200000) &
    (df.Dimensions == 2048)
]["TotalTime"].iloc[0]

print(f"Sequential baseline (N=200k, D=2048): {SEQ_200K:.5f} s")
print(f"Loaded {len(df)} unique benchmark rows.\n")

DIMS  = [128, 512, 768, 1536, 3072, 4096]
UNITS = [1, 2, 4, 8]

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 1 — Dimensionality Scaling: Speedup vs. Embedding Dimension
#  N = 100,000 | fixed 4 threads or 2 ranks | k = 5
# ─────────────────────────────────────────────────────────────────────────────
def dim_speedup(algo, **kwargs):
    """Return speedup Series indexed by Dimensions for N=100k rows."""
    mask = (df.Algorithm == algo) & (df.Documents == 100000)
    for k, v in kwargs.items():
        mask &= (df[k] == v)
    sub = df[mask].sort_values("Dimensions").set_index("Dimensions")["TotalTime"]
    seq = df[(df.Algorithm == "Sequential") & (df.Documents == 100000)] \
              .sort_values("Dimensions").set_index("Dimensions")["TotalTime"]
    return seq / sub

pth_d = dim_speedup("Pthreads", Threads=4)
omp_d = dim_speedup("OpenMP",   Threads=4)
mpi_d = dim_speedup("MPI",      Nodes=2,   Threads=1)
hyb_d = dim_speedup("Parallel", Nodes=2,   Threads=2)

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(DIMS, pth_d.reindex(DIMS), label="Pthreads (4t)",    **STYLE["Pthreads"])
ax.plot(DIMS, omp_d.reindex(DIMS), label="OpenMP (4t)",      **STYLE["OpenMP"])
ax.plot(DIMS, mpi_d.reindex(DIMS), label="MPI-only (2r)",    **STYLE["MPI"])
ax.plot(DIMS, hyb_d.reindex(DIMS), label="Hybrid (2r × 2t)", **STYLE["Hybrid"])
ax.axhline(4.0, label="Ideal (4×)", **STYLE["Ideal"])

ax.set_xlabel("Embedding Dimension  D")
ax.set_ylabel("Speedup  (×)")
ax.set_title("Fig. 1 — Speedup vs. Embedding Dimension\n"
             "N = 100,000 docs | 4 threads/ranks | k = 5")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.set_ylim(0, 5.5)
ax.legend(loc="lower right")
fig.savefig(f"{FIGURE_DIR}/fig1_dimensionality_speedup.png")
plt.close(fig)
print("✅  fig1_dimensionality_speedup.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 2 — Strong Scaling: Speedup vs. Parallel Units
#  N = 200,000 | D = 2,048 | k = 5
# ─────────────────────────────────────────────────────────────────────────────
def strong_speedup(algo, unit_col):
    mask = (
        (df.Algorithm == algo) &
        (df.Documents  == 200000) &
        (df.Dimensions == 2048)
    )
    sub = df[mask].copy()
    sub["units"] = sub[unit_col]
    return (SEQ_200K / sub.sort_values("units").set_index("units")["TotalTime"])

pth_ss = strong_speedup("Pthreads", "Threads")
omp_ss = strong_speedup("OpenMP",   "Threads")
mpi_ss = strong_speedup("MPI",      "Nodes")

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(UNITS, pth_ss.reindex(UNITS), label="Pthreads", **STYLE["Pthreads"])
ax.plot(UNITS, omp_ss.reindex(UNITS), label="OpenMP",   **STYLE["OpenMP"])
ax.plot(UNITS, mpi_ss.reindex(UNITS), label="MPI-only", **STYLE["MPI"])
ax.plot(UNITS, UNITS,                 label="Ideal",     **STYLE["Ideal"])

ax.set_xlabel("Number of Threads / Ranks")
ax.set_ylabel("Speedup  (×)")
ax.set_title("Fig. 2 — Strong Scaling Speedup\n"
             "N = 200,000 docs | D = 2,048 | k = 5")
ax.set_xticks(UNITS)
ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_ylim(0, 9.5)
ax.legend()
fig.savefig(f"{FIGURE_DIR}/fig2_strong_scaling_speedup.png")
plt.close(fig)
print("✅  fig2_strong_scaling_speedup.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 3 — Strong Scaling: Parallel Efficiency (%)
#  Same workload as Fig 2
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))

for label, ss in [("Pthreads", pth_ss), ("OpenMP", omp_ss), ("MPI", mpi_ss)]:
    eff = [ss.get(u, np.nan) / u * 100 for u in UNITS]
    ax.plot(UNITS, eff, label=label, **STYLE[label])

ax.axhline(100, label="Ideal (100 %)", **STYLE["Ideal"])

ax.set_xlabel("Number of Threads / Ranks")
ax.set_ylabel("Parallel Efficiency  (%)")
ax.set_title("Fig. 3 — Parallel Efficiency vs. Parallel Units\n"
             "N = 200,000 docs | D = 2,048 | k = 5")
ax.set_xticks(UNITS)
ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_ylim(0, 120)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.legend()
fig.savefig(f"{FIGURE_DIR}/fig3_strong_scaling_efficiency.png")
plt.close(fig)
print("✅  fig3_strong_scaling_efficiency.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 4 — Weak Scaling: Runtime vs. Document Count
#  D = 2,048 | k = 5 | 50,000 docs per thread / rank
# ─────────────────────────────────────────────────────────────────────────────
WEAK_N = [50000, 100000, 200000, 400000]

def weak_runtime(algo, unit_col):
    vals = []
    for n, u in zip(WEAK_N, UNITS):
        row = df[
            (df.Algorithm  == algo) &
            (df.Documents  == n) &
            (df.Dimensions == 2048) &
            (df[unit_col]  == u)
        ]
        vals.append(row["TotalTime"].iloc[0] if len(row) else np.nan)
    return vals

pth_w = weak_runtime("Pthreads", "Threads")
omp_w = weak_runtime("OpenMP",   "Threads")
mpi_w = weak_runtime("MPI",      "Nodes")

x_pos    = range(4)
x_labels = ["50k\n(1 unit)", "100k\n(2 units)", "200k\n(4 units)", "400k\n(8 units)"]
baseline = pth_w[0]   # 1-thread runtime at N = 50k

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(x_pos, pth_w, label="Pthreads", **STYLE["Pthreads"])
ax.plot(x_pos, omp_w, label="OpenMP",   **STYLE["OpenMP"])
ax.plot(x_pos, mpi_w, label="MPI-only", **STYLE["MPI"])
ax.axhline(baseline, label="Ideal (constant)", **STYLE["Ideal"])

ax.set_xlabel("Document Count  N  (documents per unit in parentheses)")
ax.set_ylabel("Wall-clock Time  (s)")
ax.set_title("Fig. 4 — Weak Scaling: Runtime vs. Document Count\n"
             "D = 2,048 | k = 5 | 50,000 docs per thread/rank")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(x_labels)
ax.legend()
fig.savefig(f"{FIGURE_DIR}/fig4_weak_scaling.png")
plt.close(fig)
print("✅  fig4_weak_scaling.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 5 — Execution Time Bar Chart: All Configurations
#  N = 200,000 | D = 2,048 | k = 5
# ─────────────────────────────────────────────────────────────────────────────
sub = df[
    (df.Documents  == 200000) &
    (df.Dimensions == 2048)
].copy().sort_values(["Algorithm", "Nodes", "Threads"])

labels, times, bar_colors = [], [], []

def _config_label(row):
    alg = row["Algorithm"]
    if alg == "Sequential":
        return "Seq\n1×1"
    if alg in ("Pthreads", "OpenMP"):
        tag = {"Pthreads": "Pth", "OpenMP": "OMP"}[alg]
        return f"{tag}\n{int(row.Threads)}t"
    if alg == "MPI":
        return f"MPI\n{int(row.Nodes)}r"
    # Hybrid (Parallel)
    return f"Hyb\n{int(row.Nodes)}r×{int(row.Threads)}t"

for _, row in sub.iterrows():
    labels.append(_config_label(row))
    times.append(row["TotalTime"])
    bar_colors.append(COLOR_MAP[row["Algorithm"]])

fig, ax = plt.subplots(figsize=(14, 4.5))
x = range(len(labels))
bars = ax.bar(x, times, color=bar_colors, width=0.65,
              edgecolor="white", linewidth=0.4)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{t:.3f}s",
        ha="center", va="bottom", fontsize=7, fontweight="bold"
    )

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Total Execution Time  (s)")
ax.set_title("Fig. 5 — Execution Time Across All Configurations\n"
             "N = 200,000 docs | D = 2,048 | k = 5")
ax.axhline(SEQ_200K, color="#777777", ls="--", lw=1.2, alpha=0.7,
           label=f"Sequential baseline  ({SEQ_200K:.3f} s)")

legend_patches = [
    Patch(facecolor="#777777", label="Sequential"),
    Patch(facecolor="#1a6fb3", label="Pthreads"),
    Patch(facecolor="#27ae60", label="OpenMP"),
    Patch(facecolor="#c0392b", label="MPI-only"),
    Patch(facecolor="#e67e22", label="Hybrid MPI+OpenMP"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
fig.savefig(f"{FIGURE_DIR}/fig5_execution_time_bar.png")
plt.close(fig)
print("✅  fig5_execution_time_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nAll 5 figures saved to  {FIGURE_DIR}/")
