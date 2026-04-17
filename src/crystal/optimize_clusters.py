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

# ---------------------------------------------------------------------------
# Аналитическая классификация Миллера (единый модуль miller_utils.py)
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.crystal.miller_utils import assign_miller_labels


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
        km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42, max_iter=200, batch_size=4096)
        labels = km.fit_predict(embeddings)

        sil = silhouette_score(embeddings, labels, metric="cosine", sample_size=min(10_000, len(embeddings)), random_state=42)
        ch = calinski_harabasz_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)
        inertia = km.inertia_

        # Согласованность с аналитической классификацией
        ami = adjusted_mutual_info_score(miller_labels, labels)
        nmi = normalized_mutual_info_score(miller_labels, labels)

        print(f"  Silhouette = {sil:.4f}")
        print(f"  Calinski-Harabasz = {ch:.1f}")
        print(f"  Davies-Bouldin = {db:.4f}")
        print(f"  Inertia = {inertia:.0f}")
        print(f"  AMI (vs Miller) = {ami:.4f}")
        print(f"  NMI (vs Miller) = {nmi:.4f}")

        rows.append({
            "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 4),
            "inertia": round(inertia, 0),
            "ami_miller": round(ami, 4),
            "nmi_miller": round(nmi, 4),
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Silhouette
    ax = axes[0, 0]
    ax.plot(metrics_df["k"], metrics_df["silhouette"], "o-", color="#2196F3", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score (макс → лучше)")
    ax.grid(True, alpha=0.3)

    # 2. Calinski-Harabasz
    ax = axes[0, 1]
    ax.plot(metrics_df["k"], metrics_df["calinski_harabasz"], "o-", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Calinski-Harabasz Index")
    ax.set_title("Calinski-Harabasz (макс → лучше)")
    ax.grid(True, alpha=0.3)

    # 3. Davies-Bouldin
    ax = axes[0, 2]
    ax.plot(metrics_df["k"], metrics_df["davies_bouldin"], "o-", color="#FF9800", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Davies-Bouldin (мин → лучше)")
    ax.grid(True, alpha=0.3)

    # 4. Elbow (Inertia)
    ax = axes[1, 0]
    ax.plot(metrics_df["k"], metrics_df["inertia"], "o-", color="#9C27B0", linewidth=2)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method (инерция)")
    ax.grid(True, alpha=0.3)

    # 5. AMI vs Miller
    ax = axes[1, 1]
    ax.plot(metrics_df["k"], metrics_df["ami_miller"], "o-", color="#F44336", linewidth=2, label="AMI")
    ax.plot(metrics_df["k"], metrics_df["nmi_miller"], "s--", color="#E91E63", linewidth=2, label="NMI")
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Score")
    ax.set_title("Согласованность с индексами Миллера")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Сводная нормализованная таблица
    ax = axes[1, 2]
    ax.axis("off")
    norm_df, coarse_k, fine_k = build_recommendation_table(metrics_df, args.near_best_ratio)

    table_data = norm_df[["k", "sil_norm", "ch_norm", "db_norm", "ami_norm", "combined"]].round(3)
    table_data.columns = ["k", "Sil↑", "CH↑", "DB↓(inv)", "AMI↑", "Combined"]

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(table_data.columns)))

    ax.set_title(
        f"Нормализованные метрики\n(coarse={coarse_k}, fine={fine_k}, near-best≥{args.near_best_ratio:.0%})",
        fontsize=12,
        fontweight="bold",
    )

    plt.suptitle("Оптимизация числа кластеров для SimCLR-эмбеддингов", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / "cluster_optimization_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {plot_path}")

    # Итоговая рекомендация
    best_idx = norm_df["combined"].idxmax()
    best_k = int(norm_df.loc[best_idx, "k"])
    near_best_df = norm_df.loc[norm_df["combined"] >= args.near_best_ratio * norm_df["combined"].max()].sort_values("k")

    print(f"\n{'='*60}")
    print("РЕКОМЕНДАЦИЯ: не использовать один 'оптимальный' k как абсолютную истину")
    print(f"  Формальный максимум combined-score: k = {best_k}  (score={norm_df.loc[best_idx, 'combined']:.3f})")
    print(
        f"  Near-best диапазон (>= {args.near_best_ratio:.0%} от max combined): "
        f"k = {int(near_best_df['k'].min())}..{int(near_best_df['k'].max())}"
    )
    print(f"  Coarse candidate: k = {coarse_k}")
    print(f"  Fine candidate:   k = {fine_k}")
    print("  Интерпретация:")
    print("    - coarse candidate: минимальная разумная детализация без лишнего дробления")
    print("    - fine candidate:   более физически детализированная сегментация поверхности")
    print(f"{'='*60}")


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

    args = parser.parse_args()
    main(args)