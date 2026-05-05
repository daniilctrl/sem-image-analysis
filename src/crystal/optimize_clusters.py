"""
Перебор числа кластеров (k) для SimCLR-эмбеддингов с оценкой качества.

Метрики:
  1. Silhouette Score — чёткость разделения кластеров (макс → лучше)
  2. Calinski-Harabasz Index — межкластерная/внутрикластерная дисперсия (макс → лучше)
  3. Davies-Bouldin Index — средняя «похожесть» кластеров (мин → лучше)
  4. Inertia — сумма квадратов расстояний (Elbow Method)
  5. AMI / NMI — согласованность с аналитической классификацией по Миллеру

Использование:
  python src/crystal/optimize_clusters.py \
      --embeddings_dir data/crystal/embeddings \
      --output_dir data/crystal/analysis \
      --k_range 3 4 5 6 7 8 10 12 15 20 25 30 40 50
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
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.repro import set_global_seed  # noqa: E402
from src.crystal.miller_utils import assign_miller_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _minmax_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Нормализация метрики в [0, 1] для наглядного сравнения."""
    mn = series.min()
    mx = series.max()
    if mx <= mn:
        return pd.Series(np.ones(len(series)), index=series.index)

    normalized = (series - mn) / (mx - mn)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def build_recommendation_table(metrics_df: pd.DataFrame, near_best_ratio: float) -> tuple[pd.DataFrame, int, int]:
    """Строит нормализованную таблицу и возвращает coarse/fine рекомендации.

    Вместо одного "лучшего" k используем near-best диапазон:
      - coarse_k: минимальный k среди вариантов, близких к лучшему combined-score
      - fine_k:   максимальный k среди вариантов, близких к лучшему combined-score

    Это снижает риск автоматически предпочесть большие k только потому,
    что несколько метрик монотонно улучшаются с ростом детализации.
    """
    norm_df = metrics_df.copy()
    norm_df["sil_norm"] = _minmax_normalize(norm_df["silhouette"])
    norm_df["ch_norm"] = _minmax_normalize(norm_df["calinski_harabasz"])
    norm_df["db_norm"] = _minmax_normalize(norm_df["davies_bouldin"], invert=True)
    norm_df["ami_norm"] = _minmax_normalize(norm_df["ami_miller"])
    norm_df["nmi_norm"] = _minmax_normalize(norm_df["nmi_miller"])

    norm_df["combined"] = (
        norm_df["sil_norm"]
        + norm_df["ch_norm"]
        + norm_df["db_norm"]
        + norm_df["ami_norm"]
        + norm_df["nmi_norm"]
    ) / 5.0

    best_combined = norm_df["combined"].max()
    near_best_mask = norm_df["combined"] >= near_best_ratio * best_combined
    near_best_df = norm_df.loc[near_best_mask].sort_values("k")

    coarse_k = int(near_best_df["k"].min())
    fine_k = int(near_best_df["k"].max())

    return norm_df, coarse_k, fine_k


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------


def main(args):
    used_seed = set_global_seed(args.seed, deterministic_torch=False)
    print(f"[repro] Global seed fixed: {used_seed}")

    embeddings_path = Path(args.embeddings_dir) / "crystal_embeddings.npy"
    metadata_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка эмбеддингов: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Размерность: {embeddings.shape}")

    print(f"Загрузка метаданных: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Валидация: embeddings и metadata должны быть 1-к-1
    if len(embeddings) != len(df):
        raise ValueError(
            f"CRITICAL: embeddings ({len(embeddings)}) != metadata ({len(df)}). "
            f"Run fix_embeddings_metadata.py or regenerate."
        )

    # Аналитические метки Миллера
    xyz = df[["X", "Y", "Z"]].values
    miller_labels, miller_names = assign_miller_labels(xyz, tol_deg=args.tol)
    n_miller_classes = len(set(miller_labels))
    print(f"Аналитическая классификация: {n_miller_classes} классов (включая Vicinal/Mixed)")
    for i, name in enumerate(miller_names):
        count = (miller_labels == i).sum()
        if count > 0:
            print(f"  {name}: {count} ({100 * count / len(df):.1f}%)")

    # Перебор k
    k_values = sorted(args.k_range)
    print(f"\nПеребор кластеров: k = {k_values}")

    rows = []
    cluster_assignments = {}

    for k in k_values:
        print(f"\n--- k = {k} ---")
        km = MiniBatchKMeans(
            n_clusters=k,
            n_init=5,
            random_state=args.seed,
            max_iter=200,
            batch_size=4096,
        )
        labels = km.fit_predict(embeddings)

        sil = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
            sample_size=min(10_000, len(embeddings)),
            random_state=args.seed,
        )
        ch = calinski_harabasz_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)
        inertia = km.inertia_

        # Согласованность с аналитической классификацией
        ami = adjusted_mutual_info_score(miller_labels, labels)
        nmi = normalized_mutual_info_score(miller_labels, labels)

        # Sanity: размеры кластеров. При росте k мелкие кластеры делают
        # Silhouette per-cluster шумным, а per-cluster статистику vs Miller —
        # ненадёжной. Помечаем k как unreliable, если min size < threshold.
        cluster_sizes = np.bincount(labels, minlength=k)
        min_size = int(cluster_sizes.min())
        median_size = int(np.median(cluster_sizes))
        max_size = int(cluster_sizes.max())
        n_small = int((cluster_sizes < args.min_size_threshold).sum())
        reliable = min_size >= args.min_size_threshold

        print(f"  Silhouette = {sil:.4f}")
        print(f"  Calinski-Harabasz = {ch:.1f}")
        print(f"  Davies-Bouldin = {db:.4f}")
        print(f"  Inertia = {inertia:.0f}")
        print(f"  AMI (vs Miller) = {ami:.4f}")
        print(f"  NMI (vs Miller) = {nmi:.4f}")
        size_suffix = (
            f"  [WARNING: {n_small} cluster(s) < {args.min_size_threshold}]"
            if not reliable
            else ""
        )
        print(
            f"  Cluster sizes: min={min_size}, median={median_size}, max={max_size}"
            + size_suffix
        )

        rows.append({
            "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 4),
            "inertia": round(inertia, 0),
            "ami_miller": round(ami, 4),
            "nmi_miller": round(nmi, 4),
            "min_cluster_size": min_size,
            "median_cluster_size": median_size,
            "max_cluster_size": max_size,
            "n_clusters_below_threshold": n_small,
            "reliable": reliable,
        })

        cluster_assignments[k] = labels

    # Сохранение таблицы метрик
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "cluster_optimization_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nМетрики сохранены: {metrics_path}")

    # Сохранение кластерных меток для лучших k (обновление metadata)
    # Добавляем все варианты k в metadata
    for k, labels in cluster_assignments.items():
        df[f"cluster_{k}"] = labels

    updated_meta_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"
    df.to_csv(updated_meta_path, index=False)
    print(f"Метаданные обновлены (добавлены кластеры для k={k_values}): {updated_meta_path}")

    # ---------------------------------------------------------------------------
    # Визуализация
    # ---------------------------------------------------------------------------
    # Под отчёт по практике/ВКР: 3-уровневая сетка через GridSpec —
    #   ряд 1: Silhouette / Calinski–Harabasz
    #   ряд 2: Davies–Bouldin / Метод локтя
    #   ряд 3: AMI vs Miller (на всю ширину — ключевая внешняя метрика)
    # Сводная нормализованная таблица сохраняется отдельно в CSV
    # (cluster_optimization_normalized.csv) и переносится в LaTeX-отчёт
    # как самостоятельная таблица. Логика расчётов не тронута.
    import matplotlib.gridspec as gridspec

    title_fs = 18
    label_fs = 16
    tick_fs = 13
    legend_fs = 14
    suptitle_fs = 22

    fig = plt.figure(figsize=(15, 16))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
    )

    # 1. Silhouette
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(metrics_df["k"], metrics_df["silhouette"], "o-", color="#2196F3", linewidth=2.5)
    ax.set_xlabel("Число кластеров k", fontsize=label_fs)
    ax.set_ylabel("Silhouette Score", fontsize=label_fs)
    ax.set_title("Silhouette Score (больше — лучше)", fontsize=title_fs, pad=12)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    # 2. Calinski-Harabasz
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(metrics_df["k"], metrics_df["calinski_harabasz"], "o-", color="#4CAF50", linewidth=2.5)
    ax.set_xlabel("Число кластеров k", fontsize=label_fs)
    ax.set_ylabel("Индекс Calinski–Harabasz", fontsize=label_fs)
    ax.set_title("Индекс Calinski–Harabasz (больше — лучше)", fontsize=title_fs, pad=12)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    # 3. Davies-Bouldin
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(metrics_df["k"], metrics_df["davies_bouldin"], "o-", color="#FF9800", linewidth=2.5)
    ax.set_xlabel("Число кластеров k", fontsize=label_fs)
    ax.set_ylabel("Индекс Davies–Bouldin", fontsize=label_fs)
    ax.set_title("Индекс Davies–Bouldin (меньше — лучше)", fontsize=title_fs, pad=12)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    # 4. Elbow (Inertia)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(metrics_df["k"], metrics_df["inertia"], "o-", color="#9C27B0", linewidth=2.5)
    ax.set_xlabel("Число кластеров k", fontsize=label_fs)
    ax.set_ylabel("Инерция", fontsize=label_fs)
    ax.set_title("Метод локтя (инерция)", fontsize=title_fs, pad=12)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    # 5. AMI vs Miller — на всю ширину
    ax = fig.add_subplot(gs[2, :])
    ax.plot(metrics_df["k"], metrics_df["ami_miller"], "o-", color="#F44336", linewidth=2.5, label="AMI")
    ax.plot(metrics_df["k"], metrics_df["nmi_miller"], "s--", color="#E91E63", linewidth=2.5, label="NMI")
    ax.set_xlabel("Число кластеров k", fontsize=label_fs)
    ax.set_ylabel("Значение метрики", fontsize=label_fs)
    ax.set_title("Согласованность с индексами Миллера", fontsize=title_fs, pad=12)
    ax.legend(fontsize=legend_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    # Подготовка нормализованной таблицы (сохраняется в CSV отдельно,
    # для использования в LaTeX-отчёте как самостоятельная таблица).
    norm_df, coarse_k, fine_k = build_recommendation_table(metrics_df, args.near_best_ratio)

    plt.suptitle("Оптимизация числа кластеров для SimCLR-эмбеддингов",
                 fontsize=suptitle_fs, fontweight="bold", y=0.995)
    # rect=[left, bottom, right, top] оставляет место под suptitle сверху;
    # h_pad / w_pad дают воздух между рядами и колонками.
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0, w_pad=3.0)

    plot_path = output_dir / "cluster_optimization_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {plot_path}")

    # Сводная таблица — отдельным CSV-файлом (для LaTeX).
    norm_export = norm_df[
        ["k", "sil_norm", "ch_norm", "db_norm", "ami_norm", "combined"]
    ].round(3)
    norm_export.columns = ["k", "Sil_norm", "CH_norm", "DB_inv_norm", "AMI_norm", "Combined"]
    norm_csv = output_dir / "cluster_optimization_normalized.csv"
    norm_export.to_csv(norm_csv, index=False)
    print(f"Нормализованные метрики (для LaTeX): {norm_csv}")
    print(f"  coarse={coarse_k}, fine={fine_k}, "
          f"near-best≥{args.near_best_ratio:.0%}")

    # Итоговая рекомендация
    best_idx = norm_df["combined"].idxmax()
    best_k = int(norm_df.loc[best_idx, "k"])
    near_best_df = norm_df.loc[
        norm_df["combined"] >= args.near_best_ratio * norm_df["combined"].max()
    ].sort_values("k")

    # near-best в пределах reliable-подмножества (min_cluster_size ≥ threshold)
    reliable_mask = metrics_df["reliable"].to_numpy()
    reliable_norm = norm_df.loc[reliable_mask]
    unreliable_ks = metrics_df.loc[~reliable_mask, "k"].astype(int).tolist()

    if len(reliable_norm) > 0:
        rel_best = reliable_norm["combined"].max()
        rel_mask = reliable_norm["combined"] >= args.near_best_ratio * rel_best
        rel_near_best = reliable_norm.loc[rel_mask].sort_values("k")
        reliable_coarse = int(rel_near_best["k"].min())
        reliable_fine = int(rel_near_best["k"].max())
    else:
        reliable_coarse = reliable_fine = None

    print(f"\n{'='*70}")
    print("РЕКОМЕНДАЦИЯ: не использовать один 'оптимальный' k как абсолютную истину")
    print(f"  Формальный максимум combined-score: k = {best_k}  "
          f"(score={norm_df.loc[best_idx, 'combined']:.3f})")
    print(
        f"  Near-best диапазон (>= {args.near_best_ratio:.0%} от max combined): "
        f"k = {int(near_best_df['k'].min())}..{int(near_best_df['k'].max())}"
    )
    print(f"  Coarse candidate: k = {coarse_k}")
    print(f"  Fine candidate:   k = {fine_k}")
    if unreliable_ks:
        print(
            f"  Unreliable k (min cluster < {args.min_size_threshold}): {unreliable_ks}"
        )
        if reliable_coarse is not None:
            print(
                f"  Near-best среди reliable: k = {reliable_coarse}..{reliable_fine}"
            )
    print("  Интерпретация:")
    print("    - coarse candidate: минимальная разумная детализация без лишнего дробления")
    print("    - fine candidate:   более физически детализированная сегментация поверхности")
    print(
        "    - для сравнения k внутри near-best используйте bootstrap CI\n"
        "      (retrieve_crystal.py --bootstrap): пересечение CI по precision@K_miller\n"
        "      означает, что разница не статистически значима."
    )
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оптимизация числа кластеров для SimCLR")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "crystal" / "embeddings"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "crystal" / "analysis"),
    )
    parser.add_argument(
        "--k_range",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50],
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=6.0,
        help="Допуск для классификации Миллера (градусы)",
    )
    parser.add_argument(
        "--near_best_ratio",
        type=float,
        default=0.95,
        help="Порог near-best диапазона относительно максимального combined-score",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducibility (MiniBatchKMeans, silhouette sampling)",
    )
    parser.add_argument(
        "--min_size_threshold",
        type=int,
        default=50,
        help="Minimal cluster size below which per-cluster metrics are marked unreliable",
    )

    args = parser.parse_args()
    main(args)