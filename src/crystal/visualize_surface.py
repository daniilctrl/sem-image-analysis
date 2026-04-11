"""
3D-визуализация полусферической поверхности кристалла.

Воспроизводит визуализации из презентации руководителя (слайды 9–16):
  1. RGB-кодирование: R=n1/8, G=n2/6, B=n3/12
  2. Полутоновое кодирование: grayscale = 1 - n1*n2*n3*n4*n5 / 110592
  3. Виды с разных углов для проверки «скользящего окна»
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_atoms(parquet_path: str) -> pd.DataFrame:
    """Загрузка предобработанных данных."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} atoms from {parquet_path}")
    return df


def plot_rgb(df: pd.DataFrame, output_dir: Path, elev: float = 30, azim: float = 45):
    """
    Слайд 9: RGB-кодирование соседей 1–3 порядка.
    R = n1/8, G = n2/6, B = n3/12.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.stack([df["rgb_r"], df["rgb_g"], df["rgb_b"]], axis=1)
    colors = np.clip(colors, 0, 1)

    ax.scatter(
        df["X"], df["Y"], df["Z"],
        c=colors,
        s=0.3,
        alpha=0.6,
        depthshade=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("BCC Hemisphere — RGB Encoding (R=n1/8, G=n2/6, B=n3/12)", fontsize=13)
    ax.view_init(elev=elev, azim=azim)

    path = output_dir / f"surface_rgb_elev{int(elev)}_azim{int(azim)}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_grayscale(df: pd.DataFrame, output_dir: Path, elev: float = 30, azim: float = 45):
    """
    Слайды 11-16: полутоновое кодирование.
    grayscale = 1 - n1*n2*n3*n4*n5 / 110592 (инвертированное).
    Тёмные атомы = глубинные (больше соседей), светлые = поверхностные.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    gray = df["grayscale"].values
    colors = np.stack([gray, gray, gray], axis=1)
    colors = np.clip(colors, 0, 1)

    ax.scatter(
        df["X"], df["Y"], df["Z"],
        c=colors,
        s=0.3,
        alpha=0.6,
        depthshade=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("BCC Hemisphere — Grayscale (1 − n1·n2·n3·n4·n5 / 110592)", fontsize=13)
    ax.view_init(elev=elev, azim=azim)

    path = output_dir / f"surface_gray_elev{int(elev)}_azim{int(azim)}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sliding_window(
    df: pd.DataFrame,
    output_dir: Path,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 50,
    window_radius: float = 10,
):
    """
    Слайды 11-16: визуализация «скользящего окна» — вырезаем участок
    поверхности вокруг заданной точки и показываем его под двумя углами.
    """
    # Расстояние от каждого атома до центра окна
    dist = np.sqrt(
        (df["X"] - center_x) ** 2
        + (df["Y"] - center_y) ** 2
        + (df["Z"] - center_z) ** 2
    )
    mask = dist <= window_radius
    sub = df[mask]

    if len(sub) < 10:
        print(f"  Warning: only {len(sub)} atoms in window around ({center_x}, {center_y}, {center_z}), skipping.")
        return

    print(f"  Window ({center_x}, {center_y}, {center_z}), R={window_radius}: {len(sub)} atoms")

    gray = sub["grayscale"].values
    colors = np.stack([gray, gray, gray], axis=1)
    colors = np.clip(colors, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={"projection": "3d"})

    for i, (elev, azim) in enumerate([(30, 45), (75, 135)]):
        ax = axes[i]
        ax.scatter(
            sub["X"], sub["Y"], sub["Z"],
            c=colors,
            s=5,
            alpha=0.8,
            depthshade=False,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"elev={elev}°, azim={azim}°", fontsize=11)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(
        f"Sliding Window — center=({center_x}, {center_y}, {center_z}), R={window_radius}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    path = output_dir / f"window_x{center_x}_y{center_y}_z{center_z}_R{window_radius}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_atoms(args.parquet_path)

    print("\n=== Full Surface: RGB ===")
    for elev, azim in [(30, 45), (90, 0), (0, 0)]:
        plot_rgb(df, output_dir, elev=elev, azim=azim)

    print("\n=== Full Surface: Grayscale ===")
    for elev, azim in [(30, 45), (90, 0), (0, 0)]:
        plot_grayscale(df, output_dir, elev=elev, azim=azim)

    print("\n=== Sliding Windows ===")
    # Координаты в единицах полурешётки (радиус ≈ 160 для R=50 параметров BCC)
    R = df["radius"].median()
    print(f"  Median radius: {R:.1f}")
    windows = [
        (0, 0, R, 20),          # вершина полусферы (полюс)
        (R * 0.7, 0, R * 0.7, 20),   # бок (45°)
        (0, R * 0.7, R * 0.7, 20),   # другой бок
        (R * 0.9, 0, R * 0.4, 20),   # ближе к экватору
    ]

    for cx, cy, cz, r in windows:
        plot_sliding_window(df, output_dir, center_x=cx, center_y=cy, center_z=cz, window_radius=r)

    print("\nDone! All visualizations saved to", output_dir)


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualize BCC crystal hemisphere surface")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=str(_root / "data" / "crystal" / "atoms.parquet"),
        help="Path to atoms.parquet (output of load_data.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_root / "data" / "crystal" / "visualizations"),
        help="Directory to save visualization images",
    )
    args = parser.parse_args()
    main(args)
