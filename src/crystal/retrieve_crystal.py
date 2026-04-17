"""
Обратный поиск (retrieval) по кристаллическим эмбеддингам.

Для каждого query-патча ищем K ближайших соседей в пространстве SimCLR-эмбеддингов
и оцениваем:
  1. cluster_coherence@K  — доля соседей из того же кластера (DIAGNOSTIC,
     см. примечание ниже).
  2. precision@K (miller)  — доля соседей из того же семейства Миллера
     (ВНЕШНЯЯ МЕТРИКА — основной показатель качества retrieval).
  3. Mean cosine similarity  — средняя близость к найденным соседям.

ВАЖНО: cluster_coherence@K — это круговая (circular) метрика: кластеры получены
из тех же эмбеддингов, по которым проводится поиск. Она характеризует
компактность кластеризации в латентном пространстве, а НЕ внешнее качество
retrieval. Содержательной мерой полезности является precision@K_miller,
которая использует независимые физические метки (семейства Миллера).

Это связывает Crystal-ветку с темой «обратный поиск» из названия диплома
и является аналогом cross_scale_retrieval.py для SEM-ветки.

Использование:
  python src/crystal/retrieve_crystal.py
  python src/crystal/retrieve_crystal.py --K 5 --n_queries 1000
  python src/crystal/retrieve_crystal.py --query_idx 42  # retrieve для конкретного патча
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.crystal.miller_utils import assign_miller_labels, FAMILY_NAMES


# ---------------------------------------------------------------------------
# FAISS index (cosine similarity через inner product на L2-нормированных)
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """Строит FAISS cosine-similarity индекс.

    Нормализует эмбеддинги к единичной L2-норме и использует IndexFlatIP
    (inner product = cosine similarity для unit vectors).

    Возвращает:
        index: faiss.IndexFlatIP
        normalized: np.ndarray (float32) — нормализованные эмбеддинги.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu required for retrieval. Install: pip install faiss-cpu"
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (embeddings / norms).astype("float32")
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


# ---------------------------------------------------------------------------
# Retrieval test
# ---------------------------------------------------------------------------

def run_crystal_retrieval(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    K: int = 10,
    n_queries: int = 0,
    query_indices: list | None = None,
    cluster_col: str = "cluster_8",
    seed: int = 42,
):
    """Проводит retrieval-тест на Crystal-эмбеддингах.

    Аргументы:
        embeddings: (N, D) матрица эмбеддингов.
        df: DataFrame с метаданными (должен содержать X, Y, Z и cluster_col).
        K: число ближайших соседей.
        n_queries: если > 0, случайная выборка query-патчей (0 = все).
        query_indices: конкретные индексы для query (приоритет над n_queries).
        cluster_col: столбец с метками кластеров.
        seed: для воспроизводимости выборки.

    Возвращает:
        pd.DataFrame — результаты: per-query precision + similarity.
    """
    # Подготовка Miller-меток
    xyz = df[["X", "Y", "Z"]].values
    miller_labels, miller_names = assign_miller_labels(xyz, tol_deg=6.0)
    df = df.copy()
    df["miller_label"] = miller_labels
    df["miller_family"] = [miller_names[i] for i in miller_labels]

    # Проверка кластерного столбца
    if cluster_col not in df.columns:
        raise ValueError(
            f"Column '{cluster_col}' not found. Available: {[c for c in df.columns if c.startswith('cluster_')]}"
        )

    # Build index
    index, normed = build_faiss_index(embeddings)

    # Определяем query set
    if query_indices is not None:
        queries = query_indices
    elif n_queries > 0 and n_queries < len(df):
        rng = np.random.default_rng(seed)
        queries = rng.choice(len(df), size=n_queries, replace=False)
    else:
        queries = np.arange(len(df))

    print(f"Running retrieval: {len(queries)} queries, K={K}")

    # Batch FAISS search (гораздо быстрее, чем поштучно)
    query_embs = normed[queries]
    sims_all, indices_all = index.search(query_embs, K + 1)  # +1 для self

    results = []
    for i, q_idx in enumerate(queries):
        q_cluster = df.iloc[q_idx][cluster_col]
        q_miller = df.iloc[q_idx]["miller_label"]

        # Убираем self из результатов
        ret_sims = []
        ret_same_cluster = []
        ret_same_miller = []
        count = 0

        for sim, ret_idx in zip(sims_all[i], indices_all[i]):
            if ret_idx == q_idx:
                continue
            ret_sims.append(float(sim))
            ret_same_cluster.append(int(df.iloc[ret_idx][cluster_col] == q_cluster))
            ret_same_miller.append(int(df.iloc[ret_idx]["miller_label"] == q_miller))
            count += 1
            if count >= K:
                break

        if count < K:
            continue

        results.append({
            "query_idx": int(q_idx),
            "cluster": int(q_cluster),
            "miller_family": df.iloc[q_idx]["miller_family"],
            f"cluster_coherence@{K}": sum(ret_same_cluster) / K,
            f"precision@{K}_miller": sum(ret_same_miller) / K,
            f"mean_similarity@{K}": np.mean(ret_sims),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Вывод и визуализация
# ---------------------------------------------------------------------------

def print_summary(results_df: pd.DataFrame, K: int):
    """Выводит сводку retrieval-результатов."""
    pc = f"cluster_coherence@{K}"
    pm = f"precision@{K}_miller"
    ms = f"mean_similarity@{K}"

    print(f"\n{'='*60}")
    print(f"  Crystal Retrieval Results (K={K})")
    print(f"{'='*60}")
    print(f"  Queries: {len(results_df)}")
    print(f"  Cluster coherence@{K} (diagnostic): {results_df[pc].mean():.4f}")
    print(f"  Precision@{K} (same Miller):        {results_df[pm].mean():.4f}")
    print(f"  Mean similarity@{K}:                {results_df[ms].mean():.4f}")
    print(f"\n  NOTE: cluster_coherence is a circular metric (clusters derived from")
    print(f"        the same embeddings). Use precision@K_miller as the primary metric.")

    print(f"\n  Per-cluster breakdown:")
    print(f"  {'Cluster':>8} {'Coherence':>10} {'P@K Miller':>11} {'MeanSim':>8} {'Count':>6}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*6}")

    for cl in sorted(results_df["cluster"].unique()):
        sub = results_df[results_df["cluster"] == cl]
        print(
            f"  {cl:>8} {sub[pc].mean():>10.4f} {sub[pm].mean():>11.4f} "
            f"{sub[ms].mean():>8.4f} {len(sub):>6}"
        )

    print(f"\n  Per-Miller-family breakdown:")
    print(f"  {'Family':<16} {'Coherence':>10} {'P@K Miller':>11} {'Count':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*11} {'-'*6}")

    for fam in FAMILY_NAMES:
        sub = results_df[results_df["miller_family"] == fam]
        if len(sub) == 0:
            continue
        print(f"  {fam:<16} {sub[pc].mean():>12.4f} {sub[pm].mean():>11.4f} {len(sub):>6}")


def plot_retrieval_results(results_df: pd.DataFrame, K: int, output_path: Path):
    """Строит визуализацию retrieval-результатов."""
    pc = f"cluster_coherence@{K}"
    pm = f"precision@{K}_miller"
    ms = f"mean_similarity@{K}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Per-cluster Precision
    per_cl = results_df.groupby("cluster")[[pc, pm]].mean().sort_index()
    x = per_cl.index
    width = 0.35
    axes[0].bar(x - width / 2, per_cl[pc], width, label="Cluster coherence (diag.)", color="#2196F3", alpha=0.8)
    axes[0].bar(x + width / 2, per_cl[pm], width, label="Precision@K Miller", color="#FF9800", alpha=0.8)
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel(f"Score@{K}")
    axes[0].set_title(f"Coherence & Precision@{K} по кластерам")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # 2. Per-Miller-family Precision
    per_fam = results_df.groupby("miller_family")[[pc, pm]].mean()
    per_fam = per_fam.reindex([f for f in FAMILY_NAMES if f in per_fam.index])
    y = range(len(per_fam))
    axes[1].barh(y, per_fam[pm], color="#4CAF50", alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(per_fam.index, fontsize=9)
    axes[1].set_xlabel(f"Precision@{K} (same Miller)")
    axes[1].set_title(f"Precision@{K} по семействам Миллера")
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # 3. Similarity distribution
    axes[2].hist(results_df[ms], bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel(f"Mean cosine similarity@{K}")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Распределение средней похожести (K={K})")
    axes[2].axvline(results_df[ms].mean(), color="red", linestyle="--", label=f"mean={results_df[ms].mean():.3f}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Crystal Surface Motif Retrieval (K={K}, N={len(results_df)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    _root = Path(__file__).resolve().parents[2]

    # 1. Загрузка
    emb_path = Path(args.embeddings_dir) / "crystal_embeddings.npy"
    meta_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"

    print(f"Loading embeddings: {emb_path}")
    embeddings = np.load(emb_path)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading metadata: {meta_path}")
    df = pd.read_csv(meta_path)
    print(f"  Rows: {len(df)}")

    assert len(embeddings) == len(df), (
        f"Mismatch: embeddings ({len(embeddings)}) != metadata ({len(df)})"
    )

    cluster_col = f"cluster_{args.n_clusters}"

    # 2. Retrieval
    query_indices = [args.query_idx] if args.query_idx >= 0 else None
    results = run_crystal_retrieval(
        embeddings, df,
        K=args.K,
        n_queries=args.n_queries,
        query_indices=query_indices,
        cluster_col=cluster_col,
        seed=42,
    )

    # 3. Сводка
    print_summary(results, args.K)

    # 4. Сохранение
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"crystal_retrieval_K{args.K}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    plot_path = output_dir / f"crystal_retrieval_K{args.K}.png"
    plot_retrieval_results(results, args.K, plot_path)


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Crystal Surface Motif Retrieval")
    parser.add_argument(
        "--embeddings_dir", type=str,
        default=str(_root / "data" / "crystal" / "embeddings"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(_root / "data" / "crystal" / "analysis"),
    )
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--K", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--n_queries", type=int, default=5000,
                        help="Number of random query patches (0 = all, but slow for 150K)")
    parser.add_argument("--query_idx", type=int, default=-1,
                        help="Specific patch index to query (-1 = use n_queries)")
    args = parser.parse_args()
    main(args)
