"""
Flatten-baseline для crystal-патчей.

Каждый патч (5, H, W) опционально даунсемплится до (5, side, side) и
разворачивается в (5*side*side)-D вектор, нормируется L2. Используется
как нижний предел для retrieval: если SimCLR не превосходит flatten в
P@K_miller, его метрики объясняются попиксельной похожестью соседних
патчей, а не семантически богатым представлением.

side=16 (по умолчанию) даёт 1280-D и фитится в 8 GB RAM при 150k
патчей. side=32 — без даунсемплинга, 5120-D, требует ~6 GB на индекс.

Запуск:
  python -m src.crystal.extract_flatten_baseline             # 5x16x16
  python -m src.crystal.extract_flatten_baseline --side 32   # full
"""
import argparse
from pathlib import Path

import numpy as np


def _downsample(patches: np.ndarray, side: int) -> np.ndarray:
    """Box-mean downsampling 32x32 -> side x side; side должен делить 32."""
    n, c, h, w = patches.shape
    assert h == w and h % side == 0, f"32 not divisible by side={side}"
    f = h // side
    # (N, C, side, f, side, f) -> mean over (3, 5)
    reshaped = patches.reshape(n, c, side, f, side, f)
    return reshaped.mean(axis=(3, 5))


def extract(patches_path: Path, output_dir: Path, side: int) -> None:
    print(f"Loading patches: {patches_path}")
    patches = np.load(patches_path, mmap_mode="r")
    print(f"  Shape: {patches.shape}, dtype: {patches.dtype}")

    if side != patches.shape[-1]:
        print(f"Downsampling 32x32 -> {side}x{side}")
        patches = _downsample(np.asarray(patches), side)

    flat = patches.reshape(len(patches), -1).astype(np.float32, copy=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    flat /= norms

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"flatten{side}_embeddings.npy"
    np.save(out_path, flat)
    print(f"Saved {flat.shape} -> {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_path", type=Path,
                        default=root / "data" / "crystal" / "patches" / "patches.npy")
    parser.add_argument("--output_dir", type=Path,
                        default=root / "data" / "crystal" / "embeddings_flatten")
    parser.add_argument("--side", type=int, default=16, choices=[8, 16, 32],
                        help="Spatial side after downsampling (default: 16 -> 1280-D)")
    args = parser.parse_args()
    extract(args.patches_path, args.output_dir, args.side)
