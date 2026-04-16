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
      --k_range 3 4 5 6 7 8 10 12 15 20
"""

import argparse
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
# Аналитическая классификация Миллера (дублируем логику для автономности)
# ---------------------------------------------------------------------------
import itertools


def _get_symmetry_vectors(indices):
    h, k, l = indices
    perms = set(itertools.permutations([h, k, l]))
    vecs = []
    for p in perms:
        for signs in itertools.product([1, -1], repeat=3):
            vec = np.array([p[0] * signs[0], p[1] * signs[1], p[2] * signs[2]], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vecs.append(tuple(vec / norm))
    return np.unique(vecs, axis=0)


_FAMILIES = {
    "{100}": _get_symmetry_vectors((1, 0, 0)),
    "{110}": _get_symmetry_vectors((1, 1, 0)),
    "{111}": _get_symmetry_vectors((1, 1, 1)),
    "{210}": _get_symmetry_vectors((2, 1, 0)),
    "{211}": _get_symmetry_vectors((2, 1, 1)),
    "{221}": _get_symmetry_vectors((2, 2, 1)),
    "{310}": _get_symmetry_vectors((3, 1, 0)),
    "{321}": _get_symmetry_vectors((3, 2, 1)),
    "{411}": _get_symmetry_vectors((4, 1, 1)),
}


def assign_miller_labels(xyz: np.ndarray, tol_deg: float = 6.0) -> np.ndarray:
    """Возвращает целочисленные метки граней (0..N_families, N_families = Vicinal)."""
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    valid = norms.flatten() > 0
    unit = np.zeros_like(xyz)
    unit[valid] = xyz[valid] / norms[valid]

    family_names = list(_FAMILIES.keys())
    n_families = len(family_names)

    labels = np.full(len(xyz), n_families, dtype=np.int32)  # по умолчанию = Vicinal
    best_angles = np.full(len(xyz), np.inf)

    for fi, (fname, ref_vecs) in enumerate(_FAMILIES.items()):
        dots = np.abs(np.dot(unit, ref_vecs.T))
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0)) * (180.0 / np.pi)

        better = (angles < best_angles) & (angles <= tol_deg)
        labels[better] = fi
        best_angles[better] = angles[better]

    return labels, family_names + ["Vicinal/Mixed"]


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
    # Нормализуем метрики для наглядного сравнения
    norm_df = metrics_df.copy()
    for col in ["silhouette", "calinski_harabasz", "ami_miller", "nmi_miller"]:
        mn, mx = norm_df[col].min(), norm_df[col].max()
        if mx > mn:
            norm_df[col] = (norm_df[col] - mn) / (mx - mn)
    for col in ["davies_bouldin"]:
        mn, mx = norm_df[col].min(), norm_df[col].max()
        if mx > mn:
            norm_df[col] = 1.0 - (norm_df[col] - mn) / (mx - mn)  # инвертируем

    # Комбинированный скор
    norm_df["combined"] = (
        norm_df["silhouette"]
        + norm_df["calinski_harabasz"]
        + norm_df["davies_bouldin"]
        + norm_df["ami_miller"]
        + norm_df["nmi_miller"]
    ) / 5.0

    table_data = norm_df[["k", "silhouette", "calinski_harabasz", "davies_bouldin", "ami_miller", "combined"]].round(3)
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

    # Выделяем лучший k
    best_idx = norm_df["combined"].idxmax()
    best_k = int(norm_df.loc[best_idx, "k"])
    ax.set_title(f"Нормализованные метрики\n(лучший k = {best_k})", fontsize=12, fontweight="bold")

    plt.suptitle("Оптимизация числа кластеров для SimCLR-эмбеддингов", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / "cluster_optimization_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {plot_path}")

    # Итоговая рекомендация
    print(f"\n{'='*60}")
    print(f"РЕКОМЕНДАЦИЯ: оптимальное число кластеров k = {best_k}")
    print(f"  Комбинированный нормализованный скор: {norm_df.loc[best_idx, 'combined']:.3f}")
    print(f"  Silhouette = {metrics_df.loc[best_idx, 'silhouette']:.4f}")
    print(f"  AMI (vs Miller) = {metrics_df.loc[best_idx, 'ami_miller']:.4f}")
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
        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=6.0,
        help="Допуск для классификации Миллера (градусы)",
    )

    args = parser.parse_args()
    main(args)