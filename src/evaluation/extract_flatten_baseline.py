"""
Flatten-baseline эмбеддинги: сырая картинка как «обученное» пространство.

Используется как наивный нижний предел для cross-scale retrieval. Если
SimCLR не превосходит этот baseline, его метрики объясняются тривиальной
попиксельной похожестью, а не семантически богатым представлением.

Pipeline для каждого тайла:
  PNG (256x256, grayscale) -> resize(32x32) -> flatten -> L2-norm -> 1024-D

Запуск:
  python -m src.evaluation.extract_flatten_baseline
  python -m src.evaluation.extract_flatten_baseline --side 16  # 256-D вариант
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def extract(meta_path: Path, data_dir: Path, output_dir: Path, side: int) -> None:
    df = pd.read_csv(meta_path)
    print(f"Loaded {len(df)} tile names from {meta_path}")

    dim = side * side
    embeddings = np.empty((len(df), dim), dtype=np.float32)

    for i, name in enumerate(tqdm(df["tile_name"].to_numpy(), desc=f"flatten {side}x{side}")):
        img = Image.open(data_dir / name).convert("L")
        img = img.resize((side, side), Image.BILINEAR)
        vec = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        embeddings[i] = vec

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"flatten{side}_embeddings.npy"
    names_path = output_dir / f"flatten{side}_names.csv"
    np.save(emb_path, embeddings)
    df[["tile_name"]].to_csv(names_path, index=False)
    print(f"Saved {embeddings.shape} -> {emb_path}")
    print(f"Saved names    -> {names_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Flatten-baseline embedding extractor")
    parser.add_argument("--meta_path", type=Path,
                        default=root / "data" / "processed" / "tiles_metadata.csv")
    parser.add_argument("--data_dir", type=Path,
                        default=root / "data" / "processed")
    parser.add_argument("--output_dir", type=Path,
                        default=root / "data" / "embeddings")
    parser.add_argument("--side", type=int, default=32,
                        help="Side of the downsampled tile (default: 32 -> 1024-D)")
    args = parser.parse_args()
    extract(args.meta_path, args.data_dir, args.output_dir, args.side)
