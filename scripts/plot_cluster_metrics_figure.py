"""
Re-render figure crystal_cluster_metrics.png from existing
cluster_optimization_metrics.csv WITHOUT recomputing KMeans.

Гарантирует, что значения на графике 1:1 соответствуют таблице
sweep в отчёте (results_crystal.tex). Layout: 2+2+1 (5 графиков),
figsize=15x16, dpi=200.

Использование:
  python scripts/plot_cluster_metrics_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = _ROOT / "data" / "crystal" / "analysis" / "cluster_optimization_metrics.csv"
OUT_PATH = _ROOT.parent / "text.diploma" / "spbu_diploma" / "images" / "crystal_cluster_metrics.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH).sort_values("k").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"k range: {df['k'].min()}..{df['k'].max()}")

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(df["k"], df["silhouette"], "o-", color="#2196F3", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score (больше — лучше)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(df["k"], df["calinski_harabasz"], "o-", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Индекс Calinski-Harabasz")
    ax.set_title("Индекс Calinski-Harabasz (больше — лучше)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(df["k"], df["davies_bouldin"], "o-", color="#FF9800", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Индекс Davies-Bouldin")
    ax.set_title("Индекс Davies-Bouldin (меньше — лучше)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df["k"], df["inertia"], "o-", color="#9C27B0", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Инерция")
    ax.set_title("Метод локтя (инерция)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(df["k"], df["ami_miller"], "o-", color="#F44336", linewidth=2, label="AMI")
    ax.plot(df["k"], df["nmi_miller"], "s--", color="#E91E63", linewidth=2, label="NMI")
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Значение метрики")
    ax.set_title("Согласованность с индексами Миллера (AMI / NMI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Оптимизация числа кластеров для SimCLR-эмбеддингов",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
