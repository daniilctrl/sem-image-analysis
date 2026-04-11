"""
Генерация 2D-патчей из 3D-координат атомов кристаллической поверхности.

Алгоритм:
  1. Строим KDTree по координатам [X, Y, Z]
  2. Для каждого атома-якоря (anchor) находим соседей в радиусе R
  3. Проецируем соседей на локальную касательную плоскость:
     - Нормаль к поверхности ≈ нормализованный вектор [X, Y, Z] (полусфера)
     - Строим ортонормальный базис (e1, e2) на касательной плоскости
     - Получаем 2D-координаты (u, v) каждого соседа
  4. Растеризуем (u, v) на сетку resolution×resolution
     - Каждый пиксель хранит 5 каналов: [n1_norm, n2_norm, n3_norm, n4_norm, n5_norm]
  5. Сохраняем как .npy файлы с метаданными

Использование:
  python src/crystal/patch_generator.py --num_atoms 0 --output_dir data/crystal/patches
  (--num_atoms 0 = все атомы, иначе случайная выборка)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm


# Порядок каналов и нормировочные множители
NEIGHBOR_COLS = ["n1", "n2", "n3", "n4", "n5"]
MAX_NEIGHBORS = np.array([8, 6, 12, 24, 8], dtype=np.float32)


def build_local_basis(normal: np.ndarray):
    """
    Строим ортонормальный базис (e1, e2) на касательной плоскости
    по нормали к поверхности.
    
    Используем метод: выбираем вспомогательный вектор, не параллельный normal,
    и вычисляем e1 = normal × aux, e2 = normal × e1.
    """
    normal = normal / np.linalg.norm(normal)
    
    # Вспомогательный вектор (выбираем тот, что менее параллелен normal)
    if abs(normal[0]) < 0.9:
        aux = np.array([1.0, 0.0, 0.0])
    else:
        aux = np.array([0.0, 1.0, 0.0])
    
    e1 = np.cross(normal, aux)
    e1 = e1 / np.linalg.norm(e1)
    
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    
    return e1, e2


def atoms_to_patch(
    anchor_xyz: np.ndarray,
    neighbor_xyz: np.ndarray,
    neighbor_features: np.ndarray,
    resolution: int = 32,
    window_radius: float = 10.0,
) -> np.ndarray:
    """
    Проецирует 3D-соседей на 2D-патч через касательную плоскость.
    
    Args:
        anchor_xyz: координаты атома-якоря (3,)
        neighbor_xyz: координаты соседних атомов (N, 3)
        neighbor_features: нормированные числа соседей (N, 5)
        resolution: размер выходного патча (resolution × resolution)
        window_radius: радиус окна (для масштабирования)
    
    Returns:
        patch: массив (5, resolution, resolution) — 5-канальное «изображение»
    """
    # Нормаль к полусфере = нормализованный вектор от центра к атому
    normal = anchor_xyz / np.linalg.norm(anchor_xyz)
    e1, e2 = build_local_basis(normal)
    
    # Смещения соседей относительно якоря
    deltas = neighbor_xyz - anchor_xyz  # (N, 3)
    
    # Проекция на касательную плоскость
    u = deltas @ e1  # (N,)
    v = deltas @ e2  # (N,)
    
    # Масштабирование: [-window_radius, +window_radius] → [0, resolution-1]
    half = window_radius
    u_px = ((u + half) / (2 * half) * (resolution - 1)).astype(int)
    v_px = ((v + half) / (2 * half) * (resolution - 1)).astype(int)
    
    # Фильтрация: оставляем только атомы, попавшие в квадрат патча
    mask = (u_px >= 0) & (u_px < resolution) & (v_px >= 0) & (v_px < resolution)
    u_px = u_px[mask]
    v_px = v_px[mask]
    features = neighbor_features[mask]
    
    # Заполнение патча через np.add.at — в 5–20× быстрее Python-цикла.
    # Если несколько атомов попадают в один пиксель — берём среднее.
    patch = np.zeros((5, resolution, resolution), dtype=np.float32)
    counts = np.zeros((resolution, resolution), dtype=np.float32)

    np.add.at(patch, (slice(None), v_px, u_px), features.T)  # (5, N) → накопление
    np.add.at(counts, (v_px, u_px), 1.0)

    # Нормировка пикселей с несколькими атомами
    nonzero = counts > 0
    patch[:, nonzero] /= counts[nonzero]
    
    return patch


def generate_patches(
    df: pd.DataFrame,
    num_atoms: int = 0,
    resolution: int = 32,
    window_radius: float = 10.0,
    search_radius: float = 15.0,
    output_dir: str = "data/crystal/patches",
    seed: int = 42,
):
    """
    Основной pipeline генерации патчей.
    
    Args:
        df: DataFrame с атомами (X, Y, Z, n1..n5, n1_norm..n5_norm)
        num_atoms: сколько атомов использовать (0 = все)
        resolution: размер патча
        window_radius: радиус 2D-проекции
        search_radius: радиус поиска соседей в 3D (должен быть ≥ window_radius)
        output_dir: куда сохранять
        seed: random seed для воспроизводимости
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    coords = df[["X", "Y", "Z"]].values.astype(np.float64)
    features = df[["n1_norm", "n2_norm", "n3_norm", "n4_norm", "n5_norm"]].values.astype(np.float32)
    
    print(f"Building KDTree for {len(coords):,} atoms...")
    tree = KDTree(coords)
    
    # Выборка якорей
    if num_atoms > 0 and num_atoms < len(df):
        rng = np.random.default_rng(seed)
        anchor_indices = rng.choice(len(df), size=num_atoms, replace=False)
        anchor_indices = np.sort(anchor_indices)
        print(f"Selected {num_atoms:,} anchor atoms (random subset)")
    else:
        # Для полного набора берём только «поверхностные» атомы (grayscale > порог)
        # Это отсекает глубинные атомы, которые просто дублируют одинаковый паттерн
        if "grayscale" in df.columns:
            surface_mask = df["grayscale"] > 0.01  # хотя бы 1% отклонение от полного набора соседей
            anchor_indices = np.where(surface_mask)[0]
            print(f"Using {len(anchor_indices):,} surface atoms as anchors (grayscale > 0.01)")
        else:
            anchor_indices = np.arange(len(df))
            print(f"Using all {len(anchor_indices):,} atoms as anchors")
    
    # Генерация патчей
    metadata_rows = []
    all_patches = []
    
    min_neighbors = float("inf")
    max_neighbors = 0
    
    for i, idx in enumerate(tqdm(anchor_indices, desc="Generating patches")):
        anchor_xyz = coords[idx]
        
        # Поиск соседей в радиусе search_radius
        neighbor_ids = tree.query_ball_point(anchor_xyz, r=search_radius)
        
        if len(neighbor_ids) < 5:
            continue
        
        neighbor_ids = np.array(neighbor_ids)
        neighbor_xyz = coords[neighbor_ids]
        neighbor_features = features[neighbor_ids]
        
        min_neighbors = min(min_neighbors, len(neighbor_ids))
        max_neighbors = max(max_neighbors, len(neighbor_ids))
        
        # Генерация патча
        patch = atoms_to_patch(
            anchor_xyz,
            neighbor_xyz,
            neighbor_features,
            resolution=resolution,
            window_radius=window_radius,
        )
        
        all_patches.append(patch)
        metadata_rows.append({
            "patch_idx": len(all_patches) - 1,
            "atom_idx": idx,
            "X": float(anchor_xyz[0]),
            "Y": float(anchor_xyz[1]),
            "Z": float(anchor_xyz[2]),
            "n_neighbors_3d": len(neighbor_ids),
            "n1": int(df.iloc[idx]["n1"]),
            "n2": int(df.iloc[idx]["n2"]),
            "n3": int(df.iloc[idx]["n3"]),
            "n4": int(df.iloc[idx]["n4"]),
            "n5": int(df.iloc[idx]["n5"]),
            "grayscale": float(df.iloc[idx].get("grayscale", 0)),
        })
    
    # Сохранение
    patches_array = np.stack(all_patches, axis=0)  # (N, 5, 32, 32)
    patches_path = output_path / "patches.npy"
    np.save(patches_path, patches_array)
    
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = output_path / "patches_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    
    print(f"\n--- Patch Generation Summary ---")
    print(f"  Total patches: {len(all_patches):,}")
    print(f"  Patch shape: {patches_array.shape} (N, C, H, W)")
    print(f"  3D neighbors per atom: [{min_neighbors}, {max_neighbors}]")
    print(f"  Non-zero pixels per patch: "
          f"mean={np.mean(np.sum(patches_array[:, 0] > 0, axis=(1, 2))):.1f}, "
          f"max={np.max(np.sum(patches_array[:, 0] > 0, axis=(1, 2)))}")
    print(f"  Saved patches: {patches_path} ({patches_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Saved metadata: {meta_path}")
    
    return patches_array, meta_df


def visualize_sample_patches(
    patches: np.ndarray, meta_df: pd.DataFrame, output_dir: str, n_samples: int = 16
):
    """Визуализация нескольких примеров патчей для проверки."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(patches), size=min(n_samples, len(patches)), replace=False)
    
    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    # Визуализация: среднее по каналам (как «серый» снимок)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        patch = patches[idx]  # (5, 32, 32)
        # Среднее по 5 каналам для визуализации
        gray = patch.mean(axis=0)
        
        row = meta_df.iloc[idx]
        ax = axes[i]
        ax.imshow(gray, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(
            f"Atom #{int(row['atom_idx'])}\n"
            f"({row['X']:.0f}, {row['Y']:.0f}, {row['Z']:.0f})\n"
            f"gs={row['grayscale']:.3f}",
            fontsize=8,
        )
        ax.axis("off")
    
    # Скрыть пустые
    for i in range(len(indices), len(axes)):
        axes[i].axis("off")
    
    plt.suptitle("Sample Crystal Patches (mean of 5 channels)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    path = output_path / "sample_patches.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved sample patches: {path}")
    
    # 5-канальная визуализация одного патча
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    idx = indices[0]
    patch = patches[idx]
    channel_names = ["n1/8", "n2/6", "n3/12", "n4/24", "n5/8"]
    
    for ch in range(5):
        axes[ch].imshow(patch[ch], cmap="viridis", vmin=0, vmax=1)
        axes[ch].set_title(channel_names[ch], fontsize=11)
        axes[ch].axis("off")
    
    # Композит RGB (первые 3 канала)
    rgb = np.stack([patch[0], patch[1], patch[2]], axis=2)
    axes[5].imshow(np.clip(rgb, 0, 1))
    axes[5].set_title("RGB (n1, n2, n3)", fontsize=11)
    axes[5].axis("off")
    
    row = meta_df.iloc[idx]
    plt.suptitle(
        f"Patch channels — Atom #{int(row['atom_idx'])} at ({row['X']:.0f}, {row['Y']:.0f}, {row['Z']:.0f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    
    path = output_path / "patch_channels.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved channel visualization: {path}")


def main(args):
    print("Loading data...")
    df = pd.read_parquet(args.parquet_path)
    print(f"  Loaded {len(df):,} atoms")
    
    patches, meta_df = generate_patches(
        df,
        num_atoms=args.num_atoms,
        resolution=args.resolution,
        window_radius=args.window_radius,
        search_radius=args.search_radius,
        output_dir=args.output_dir,
    )
    
    print("\nGenerating sample visualizations...")
    visualize_sample_patches(patches, meta_df, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D patches from 3D crystal surface data")
    _root = Path(__file__).resolve().parents[2]
    parser.add_argument("--parquet_path", type=str, default=str(_root / "data" / "crystal" / "atoms.parquet"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "data" / "crystal" / "patches"))
    parser.add_argument("--num_atoms", type=int, default=0, help="Number of anchor atoms (0 = all surface atoms)")
    parser.add_argument("--resolution", type=int, default=32, help="Patch resolution (32x32)")
    parser.add_argument("--window_radius", type=float, default=10.0, help="2D projection window radius")
    parser.add_argument("--search_radius", type=float, default=15.0, help="3D neighbor search radius (>= window_radius)")
    args = parser.parse_args()
    main(args)
