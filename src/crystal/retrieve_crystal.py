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
from src.utils.repro import set_global_seed
from src.utils.stats import bootstrap_metric_ci


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

    # Делаем нормализацию in-place, чтобы не удваивать пик памяти при
    # больших размерностях (flatten 5120-D × 150k = 3 GB; копия → OOM).
    if embeddings.dtype != np.float32:
        normalized = embeddings.astype("float32", copy=True)
    else:
        normalized = np.array(embeddings, copy=True)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized /= norms
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
    spatial_split_deg: float = 0.0,
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
        spatial_split_deg: при > 0 исключать соседей с угловым расстоянием
            < этого порога от query (по (X,Y,Z) на полусфере). Это убирает
            тривиальную геометрическую близость: Miller-семейство задаётся
            углом к крист. осям с допуском 6°, поэтому соседи в радиусе
            < 6° по построению попадают в то же семейство, и P@K_miller
            на них не отражает работу обученного пространства. Разумный
            порог 12-15°: вдвое больше Miller-tolerance.

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
        queries = np.asarray(query_indices)
    elif n_queries > 0 and n_queries < len(df):
        rng = np.random.default_rng(seed)
        queries = rng.choice(len(df), size=n_queries, replace=False)
    else:
        queries = np.arange(len(df))

    # При spatial split нормируем (X,Y,Z) и считаем cos(theta) через dot.
    # fetch_K увеличиваем на размер сферической шапки + запас, чтобы после
    # фильтра оставалось ≥ K кандидатов даже при сильной геом. близости.
    use_spatial = spatial_split_deg > 0
    if use_spatial:
        xyz_norm = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        cos_thresh = float(np.cos(np.deg2rad(spatial_split_deg)))
        # Доля поверхности полусферы внутри шапки = (1 - cos(theta)) / 2;
        # верхняя граница на число патчей в шапке любой точки.
        cap_fraction = (1.0 - cos_thresh) / 2.0
        cap_size_est = int(np.ceil(cap_fraction * len(df) * 2))  # запас x2
        fetch_K = min(K + cap_size_est + 64, len(df))
    else:
        fetch_K = K + 1  # +1 на self

    print(f"Running retrieval: {len(queries)} queries, K={K}, "
          f"spatial_split={spatial_split_deg}°, fetch_K={fetch_K}")

    # Batch FAISS search
    query_embs = normed[queries]
    sims_all, indices_all = index.search(query_embs, fetch_K)

    # numpy-кеш колонок для скорости
    cluster_arr = df[cluster_col].to_numpy()
    miller_arr = df["miller_label"].to_numpy()
    fam_arr = df["miller_family"].to_numpy()

    results = []
    skipped = 0
    for i, q_idx in enumerate(queries):
        q_cluster = cluster_arr[q_idx]
        q_miller = miller_arr[q_idx]

        cand_ids = indices_all[i]
        cand_sims = sims_all[i]

        # self-фильтр
        mask = cand_ids != q_idx

        # spatial фильтр: исключаем кандидатов в угловом радиусе < threshold
        if use_spatial:
            q_dir = xyz_norm[q_idx]
            cand_dirs = xyz_norm[cand_ids]
            cos_to_q = cand_dirs @ q_dir
            mask = mask & (cos_to_q < cos_thresh)

        kept_ids = cand_ids[mask][:K]
        if len(kept_ids) < K:
            skipped += 1
            continue
        kept_sims = cand_sims[mask][:K]

        ret_same_cluster = (cluster_arr[kept_ids] == q_cluster).astype(int)
        ret_same_miller = (miller_arr[kept_ids] == q_miller).astype(int)

        results.append({
            "query_idx": int(q_idx),
            "cluster": int(q_cluster),
            "miller_family": fam_arr[q_idx],
            f"cluster_coherence@{K}": float(ret_same_cluster.sum()) / K,
            f"precision@{K}_miller": float(ret_same_miller.sum()) / K,
            f"mean_similarity@{K}": float(kept_sims.mean()),
        })

    if skipped:
        print(f"WARN: skipped {skipped} queries (insufficient candidates after filter)")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
# `bootstrap_metric_ci` вынесен в src/utils/stats.py (используется также
# в cross_scale_retrieval, linear_probe, knn_probe). Здесь только wrapper.


def compute_bootstrap_summary(
    results_df: pd.DataFrame,
    K: int,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Применяет bootstrap CI ко всем ключевым метрикам retrieval."""
    metric_cols = [
        f"cluster_coherence@{K}",
        f"precision@{K}_miller",
        f"mean_similarity@{K}",
    ]
    rows = []
    for col in metric_cols:
        vals = results_df[col].to_numpy(dtype=np.float64)
        mean, lo, hi = bootstrap_metric_ci(vals, n_bootstrap, confidence, seed)
        rows.append({
            "metric": col,
            "mean": round(mean, 4),
            f"ci_lo ({int(confidence*100)}%)": round(lo, 4),
            f"ci_hi ({int(confidence*100)}%)": round(hi, 4),
            "ci_half_width": round((hi - lo) / 2.0, 4),
            "n_queries": int(len(vals)),
        })
    return pd.DataFrame(rows)


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
    """Строит визуализацию retrieval-результатов.

    Под отчёт по практике/ВКР: увеличен figsize, шрифты подписей и
    легенд, dpi=200; логика расчётов не тронута.
    """
    pc = f"cluster_coherence@{K}"
    pm = f"precision@{K}_miller"
    ms = f"mean_similarity@{K}"

    title_fs = 16
    label_fs = 14
    tick_fs = 12
    legend_fs = 13
    suptitle_fs = 18

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # 1. Per-cluster Precision
    per_cl = results_df.groupby("cluster")[[pc, pm]].mean().sort_index()
    x = per_cl.index
    width = 0.35
    axes[0].bar(x - width / 2, per_cl[pc], width,
                label="Согласованность кластера (диаг.)", color="#2196F3", alpha=0.8)
    axes[0].bar(x + width / 2, per_cl[pm], width,
                label=f"Precision@{K} (Миллер)", color="#FF9800", alpha=0.8)
    axes[0].set_xlabel("Номер кластера", fontsize=label_fs)
    axes[0].set_ylabel(f"Значение метрики @ K={K}", fontsize=label_fs)
    axes[0].set_title(f"Согласованность и Precision@{K} по кластерам", fontsize=title_fs)
    axes[0].legend(fontsize=legend_fs)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="both", labelsize=tick_fs)
    axes[0].grid(True, alpha=0.3)

    # 2. Per-Miller-family Precision
    per_fam = results_df.groupby("miller_family")[[pc, pm]].mean()
    per_fam = per_fam.reindex([f for f in FAMILY_NAMES if f in per_fam.index])
    y = range(len(per_fam))
    axes[1].barh(y, per_fam[pm], color="#4CAF50", alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(per_fam.index, fontsize=tick_fs)
    axes[1].set_xlabel(f"Precision@{K} (то же семейство Миллера)", fontsize=label_fs)
    axes[1].set_title(f"Precision@{K} по семействам Миллера", fontsize=title_fs)
    axes[1].set_xlim(0, 1)
    axes[1].tick_params(axis="x", labelsize=tick_fs)
    axes[1].grid(True, alpha=0.3)

    # 3. Similarity distribution
    axes[2].hist(results_df[ms], bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel(f"Среднее косинусное сходство @ K={K}", fontsize=label_fs)
    axes[2].set_ylabel("Количество", fontsize=label_fs)
    axes[2].set_title(f"Распределение средней похожести (K={K})", fontsize=title_fs)
    axes[2].axvline(results_df[ms].mean(), color="red", linestyle="--",
                    label=f"среднее={results_df[ms].mean():.3f}")
    axes[2].legend(fontsize=legend_fs)
    axes[2].tick_params(axis="both", labelsize=tick_fs)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Обратный поиск по поверхностным мотивам кристалла "
        f"(K={K}, N={len(results_df)})",
        fontsize=suptitle_fs, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    used_seed = set_global_seed(args.seed, deterministic_torch=False)
    print(f"[repro] Global seed fixed: {used_seed}")

    _root = Path(__file__).resolve().parents[2]

    # 1. Загрузка. Если emb_path не задан явно — берём дефолт из embeddings_dir.
    emb_path = Path(args.emb_path) if args.emb_path else Path(args.embeddings_dir) / "crystal_embeddings.npy"
    meta_path = Path(args.meta_path) if args.meta_path else Path(args.embeddings_dir) / "embeddings_metadata.csv"

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

    # 2. Retrieval. Если задан --restrict_query_split, ограничиваем
    # множество query-патчей нужным split'ом из split_csv. Это нужно
    # для region-holdout: query берём только из test-секторов, а
    # candidate-pool остаётся полным (модель пытается найти соседа из
    # train-сектора, который никогда не видела как query во время
    # обучения). Сами эмбеддинги извлечены на всех 150k патчах.
    query_indices = [args.query_idx] if args.query_idx >= 0 else None
    if args.restrict_query_split and args.split_csv:
        split_df = pd.read_csv(args.split_csv)
        keep = split_df.loc[
            split_df["split"] == args.restrict_query_split, "patch_idx"
        ].to_numpy()
        if query_indices is not None:
            query_indices = [q for q in query_indices if q in set(keep)]
        else:
            # Сначала случайно выбираем n_queries по seed=42 как раньше,
            # затем фильтруем по split, чтобы порядок остался
            # воспроизводимым между запусками.
            rng = np.random.default_rng(42)
            if args.n_queries > 0:
                base = rng.choice(len(df), size=min(args.n_queries * 4, len(df)),
                                  replace=False)
            else:
                base = np.arange(len(df))
            query_indices = [int(i) for i in base if i in set(keep)]
            if args.n_queries > 0:
                query_indices = query_indices[:args.n_queries]
        print(f"  restrict_query_split={args.restrict_query_split}: "
              f"{len(query_indices)} queries")

    results = run_crystal_retrieval(
        embeddings, df,
        K=args.K,
        n_queries=args.n_queries,
        query_indices=query_indices,
        cluster_col=cluster_col,
        seed=42,
        spatial_split_deg=args.spatial_split_deg,
    )

    # 3. Сводка
    print_summary(results, args.K)

    # 4. Сохранение. Если задан --model_name, добавляем суффикс к именам
    # артефактов, чтобы запуски разных baseline'ов не перетирали друг друга.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.model_name}" if args.model_name else ""
    if args.spatial_split_deg > 0:
        suffix += f"_split{int(args.spatial_split_deg)}deg"

    csv_path = output_dir / f"crystal_retrieval_K{args.K}{suffix}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    plot_path = output_dir / f"crystal_retrieval_K{args.K}{suffix}.png"
    plot_retrieval_results(results, args.K, plot_path)

    # 5. Bootstrap CI (по умолчанию включён, можно отключить --no-bootstrap)
    if args.bootstrap:
        boot_df = compute_bootstrap_summary(
            results,
            K=args.K,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
        print(f"\n{'='*60}")
        print(f"  Bootstrap {int(args.confidence*100)}% CI "
              f"({args.n_bootstrap} iterations, seed={args.seed})")
        print(f"{'='*60}")
        print(boot_df.to_string(index=False))
        boot_path = output_dir / f"crystal_retrieval_K{args.K}{suffix}_bootstrap.csv"
        boot_df.to_csv(boot_path, index=False)
        print(f"\nBootstrap summary saved: {boot_path}")
        print(
            "\n  NOTE: Overlapping CI for precision@K_miller between two k values\n"
            "        means the difference is NOT statistically significant — use\n"
            "        this when comparing k=35 vs k=50 rather than raw means."
        )


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
    parser.add_argument("--bootstrap", action="store_true", default=True,
                        help="Compute bootstrap 95% CI for retrieval metrics (default: True)")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false",
                        help="Disable bootstrap CI computation")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="CI confidence level (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global + bootstrap seed for reproducibility")
    parser.add_argument("--emb_path", type=str, default=None,
                        help="Override path to embeddings .npy "
                             "(default: <embeddings_dir>/crystal_embeddings.npy)")
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Override path to metadata CSV "
                             "(default: <embeddings_dir>/embeddings_metadata.csv)")
    parser.add_argument("--model_name", type=str, default="",
                        help="Suffix added to output filenames (e.g. 'simclr', 'flatten')")
    parser.add_argument("--spatial_split_deg", type=float, default=0.0,
                        help="Angular distance threshold (deg) for spatial split. "
                             "Neighbors within this radius from query are excluded. "
                             "Recommended: 12-15° (twice the Miller tolerance of 6°). "
                             "0 = disabled (default).")
    parser.add_argument("--split_csv", type=str, default="",
                        help="Path to split CSV with patch_idx+split columns. "
                             "Used together with --restrict_query_split for "
                             "region-holdout retrieval evaluation.")
    parser.add_argument("--restrict_query_split", type=str, default="",
                        help="If set (e.g. 'test'), only patches with this "
                             "split label are used as queries. The candidate "
                             "pool stays full. Requires --split_csv.")
    args = parser.parse_args()
    main(args)
