"""
Визуализация кластеров кристаллической поверхности.

  1. UMAP проекция эмбеддингов (2D)
  2. Интерактивная 3D-карта кластеров (Plotly → HTML)
  3. Проекция Мольвейде (развёрнутая карта поверхности)
  4. Статические 3D-виды (matplotlib)
  5. Распределение кластеров

Использование:
  python src/crystal/visualize_crystal.py \\
    --embeddings_dir data/crystal/embeddings \\
    --n_clusters 8
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import umap


# ─── Палитра кластеров ───────────────────────────────────────────────
CLUSTER_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def get_colors(n_clusters: int):
    """Возвращает список цветов для n кластеров."""
    return CLUSTER_PALETTE[:n_clusters]


# ─── UMAP ─────────────────────────────────────────────────────────────
def plot_umap(embeddings, labels, n_clusters, output_dir):
    """UMAP 2D-проекция эмбеддингов, раскрашенная по кластерам."""
    print("Построение UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
    umap_2d = reducer.fit_transform(embeddings)

    colors = get_colors(n_clusters)
    fig, ax = plt.subplots(figsize=(12, 10))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            umap_2d[mask, 0], umap_2d[mask, 1],
            c=colors[cluster_id],
            label=f"Кластер {cluster_id} ({mask.sum():,})",
            s=8, alpha=0.7, edgecolors="none",
        )

    ax.set_title(f"UMAP эмбеддингов кристалла — {n_clusters} кластеров", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3, fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    path = output_dir / f"umap_{n_clusters}кластеров.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")

    return umap_2d


# ─── Интерактивный 3D (Plotly) ────────────────────────────────────────
def plot_interactive_3d(meta_df, labels, n_clusters, output_dir):
    """
    Интерактивная 3D-карта кластеров на полусфере.
    Открывается в браузере, можно вращать мышью.
    """
    import plotly.graph_objects as go

    colors = get_colors(n_clusters)
    color_array = [colors[l] for l in labels]

    # Текст при наведении
    hover_text = [
        f"Атом #{int(row.get('atom_idx', i))}<br>"
        f"X={row['X']:.1f}, Y={row['Y']:.1f}, Z={row['Z']:.1f}<br>"
        f"Кластер: {labels[i]}<br>"
        f"Grayscale: {row.get('grayscale', 0):.3f}"
        for i, (_, row) in enumerate(meta_df.iterrows())
    ]

    fig = go.Figure()

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        idx = np.where(mask)[0]
        fig.add_trace(go.Scatter3d(
            x=meta_df.iloc[idx]["X"],
            y=meta_df.iloc[idx]["Y"],
            z=meta_df.iloc[idx]["Z"],
            mode="markers",
            marker=dict(
                size=2.5,
                color=colors[cluster_id],
                opacity=0.8,
            ),
            text=[hover_text[j] for j in idx],
            hoverinfo="text",
            name=f"Кластер {cluster_id} ({mask.sum():,})",
        ))

    fig.update_layout(
        title=dict(
            text=f"Кристаллическая поверхность — {n_clusters} кластеров",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(
            font=dict(size=11),
            itemsizing="constant",
        ),
        width=1200,
        height=900,
    )

    path = output_dir / f"surface_interactive_{n_clusters}кластеров.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  Сохранено (интерактивный): {path}")


# ─── Проекция Мольвейде ──────────────────────────────────────────────
def plot_mollweide(meta_df, labels, n_clusters, output_dir):
    """
    Развёрнутая карта поверхности полусферы (проекция Мольвейде).
    Каждый атом — точка, раскрашенная по кластеру.
    Координаты: θ (долгота) = atan2(Y, X), φ (широта) = asin(Z/R).
    """
    X = meta_df["X"].values
    Y = meta_df["Y"].values
    Z = meta_df["Z"].values

    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arctan2(Y, X)       # долгота [-π, π]
    phi = np.arcsin(np.clip(Z / R, -1, 1))  # широта [-π/2, π/2]

    colors = get_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": "mollweide"})

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            theta[mask], phi[mask],
            c=colors[cluster_id],
            label=f"Кластер {cluster_id} ({mask.sum():,})",
            s=1.5,
            alpha=0.8,
            edgecolors="none",
        )

    ax.set_title(
        f"Карта поверхности (проекция Мольвейде) — {n_clusters} кластеров",
        fontsize=14, pad=20,
    )
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        markerscale=5, fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"mollweide_{n_clusters}кластеров.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")

    # Дополнительно: тот же график, но окрашенный по grayscale (для сравнения)
    if "grayscale" in meta_df.columns:
        fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": "mollweide"})
        sc = ax.scatter(
            theta, phi,
            c=meta_df["grayscale"].values,
            cmap="gray_r",
            s=1.5,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title("Карта поверхности — полутоновая (grayscale)", fontsize=14, pad=20)
        plt.colorbar(sc, ax=ax, shrink=0.6, label="grayscale")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = output_dir / "mollweide_grayscale.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Сохранено: {path}")


# ─── Статические 3D виды (matplotlib) ─────────────────────────────────
def plot_3d_static(meta_df, labels, n_clusters, output_dir):
    """Статические 3D-виды с крупными точками."""
    colors = get_colors(n_clusters)
    atom_colors = [colors[l] for l in labels]

    for elev, azim, name in [(30, 45, "перспектива"), (90, 0, "сверху"), (0, 0, "сбоку")]:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            meta_df["X"], meta_df["Y"], meta_df["Z"],
            c=atom_colors,
            s=2.0,
            alpha=0.8,
            depthshade=False,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Поверхность кристалла — {n_clusters} кластеров ({name})", fontsize=13)
        ax.view_init(elev=elev, azim=azim)

        path = output_dir / f"surface_{n_clusters}кл_{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Сохранено: {path}")


# ─── Распределение кластеров ──────────────────────────────────────────
def plot_cluster_distribution(meta_df, labels, n_clusters, output_dir):
    """Гистограмма: размер кластеров + средний grayscale."""
    colors = get_colors(n_clusters)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    axes[0].bar(range(n_clusters), cluster_sizes, color=colors)
    axes[0].set_xlabel("Кластер")
    axes[0].set_ylabel("Количество атомов")
    axes[0].set_title("Размеры кластеров")

    if "grayscale" in meta_df.columns:
        gs_means = [meta_df.loc[labels == c, "grayscale"].mean() for c in range(n_clusters)]
        gs_stds = [meta_df.loc[labels == c, "grayscale"].std() for c in range(n_clusters)]
        axes[1].bar(range(n_clusters), gs_means, yerr=gs_stds, color=colors, capsize=3)
        axes[1].set_xlabel("Кластер")
        axes[1].set_ylabel("Средний grayscale")
        axes[1].set_title("Grayscale (выше = ближе к поверхности)")

    plt.suptitle(f"Распределение — {n_clusters} кластеров", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / f"распределение_{n_clusters}кл.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")


# ─── main ─────────────────────────────────────────────────────────────
def main(args):
    emb_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(emb_dir / "crystal_embeddings.npy")
    meta_df = pd.read_csv(emb_dir / "embeddings_metadata.csv")
    print(f"Загружено {len(embeddings)} эмбеддингов, {meta_df.shape[1]} столбцов метаданных")

    n_clusters = args.n_clusters
    cluster_col = f"cluster_{n_clusters}"

    if cluster_col not in meta_df.columns:
        print(f"Ошибка: {cluster_col} не найден. Доступные: {list(meta_df.columns)}")
        return

    labels = meta_df[cluster_col].values

    print("\n=== UMAP ===")
    umap_2d = plot_umap(embeddings, labels, n_clusters, output_dir)

    if "X" in meta_df.columns:
        print("\n=== Интерактивная 3D-карта (Plotly) ===")
        plot_interactive_3d(meta_df, labels, n_clusters, output_dir)

        print("\n=== Проекция Мольвейде ===")
        plot_mollweide(meta_df, labels, n_clusters, output_dir)

        print("\n=== Статические 3D-виды ===")
        plot_3d_static(meta_df, labels, n_clusters, output_dir)

    print("\n=== Распределение кластеров ===")
    plot_cluster_distribution(meta_df, labels, n_clusters, output_dir)

    print(f"\nГотово! Визуализации в {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация кластеров кристалла")
    parser.add_argument("--embeddings_dir", type=str, default=r"c:\projects\diploma\data\crystal\embeddings")
    parser.add_argument("--output_dir", type=str, default=r"c:\projects\diploma\data\crystal\visualizations\clusters")
    parser.add_argument("--n_clusters", type=int, default=8)
    args = parser.parse_args()
    main(args)
