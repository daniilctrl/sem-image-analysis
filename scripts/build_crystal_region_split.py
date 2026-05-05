"""
Генерирует region-holdout split для crystal-патчей.

Полусфера разбивается на N азимутальных секторов вокруг оси, проходящей
через центроид облака патчей; group_id ∈ [0..N-1]. Из них K соседних
секторов помечаются как `test`, остальные — как `train`. Это даёт
геометрически связную тестовую область, которую SimCLR никогда не
видит при обучении (в отличие от текущего сетапа, где тренинг идёт по
всем 150k патчам).

Сохраняет CSV: `patch_idx,region_id,split` (split ∈ {train, test}).

Запуск:
  python scripts/build_crystal_region_split.py
  python scripts/build_crystal_region_split.py --n_sectors 8 --test_sectors 5 6
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def assign_regions(xyz: np.ndarray, n_sectors: int) -> np.ndarray:
    """Делит полусферу на n_sectors азимутальных секторов.

    Полярная ось — направление от центроида облака к началу координат.
    Этот выбор естественен: точки лежат на сфере R≈160 вокруг origin,
    но половина (X≥0, Z≥0) занята только частью сферы; полярная ось
    через центроид и origin даёт «вертикальную» ось такой шапки.
    Азимут φ ∈ [-π, π) на ортогональной плоскости, разбивается на
    равные сегменты по 360°/n_sectors.
    """
    c = xyz.mean(axis=0)
    polar = c / np.linalg.norm(c)            # ось через центроид

    # Базис ортогональной плоскости: два произвольных перпендикулярных
    # к polar единичных вектора (Gram-Schmidt от стандартных осей).
    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, polar)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])
    e1 = aux - polar * np.dot(aux, polar)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(polar, e1)

    # Проекция точек на (e1, e2) -> азимут
    rel = xyz - c
    a = rel @ e1
    b = rel @ e2
    phi = np.arctan2(b, a)                   # [-π, π)

    sector_size = 2 * np.pi / n_sectors
    region_id = np.floor((phi + np.pi) / sector_size).astype(int)
    region_id = np.clip(region_id, 0, n_sectors - 1)
    return region_id


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=Path,
                        default=root / "data" / "crystal" / "embeddings" / "embeddings_metadata.csv")
    parser.add_argument("--out", type=Path,
                        default=root / "data" / "crystal" / "splits" / "region_holdout_v1.csv")
    parser.add_argument("--n_sectors", type=int, default=6,
                        help="Количество азимутальных секторов")
    parser.add_argument("--test_sectors", type=int, nargs="+", default=[4, 5],
                        help="ID секторов, идущих в test (остальные — train)")
    args = parser.parse_args()

    df = pd.read_csv(args.meta_path)
    print(f"Loaded {len(df)} patches from {args.meta_path}")

    xyz = df[["X", "Y", "Z"]].values
    region_id = assign_regions(xyz, args.n_sectors)

    test_set = set(args.test_sectors)
    split = np.where(np.isin(region_id, list(test_set)), "test", "train")

    out_df = pd.DataFrame({
        "patch_idx": df["patch_idx"].values,
        "region_id": region_id,
        "split": split,
    })

    print("\nRegion sizes:")
    counts = out_df["region_id"].value_counts().sort_index()
    for r, n in counts.items():
        marker = " (TEST)" if r in test_set else ""
        print(f"  region {r}: {n:>7d}{marker}")
    print(f"\nSplit totals:")
    print(out_df["split"].value_counts().to_string())

    train_frac = (out_df["split"] == "train").mean()
    print(f"\ntrain fraction: {train_frac:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
