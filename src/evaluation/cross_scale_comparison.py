"""
Cross-Scale Retrieval Comparison: Baseline vs SimCLR vs BYOL.

For each tile, find K nearest neighbors excluding same-source tiles.
Measure:
  - Precision@K (material): fraction of neighbors from the same material
  - Precision@K (cross-scale): fraction from same material but DIFFERENT magnification

Usage:
  python src/evaluation/cross_scale_comparison.py
  python src/evaluation/cross_scale_comparison.py --K 5 --output_dir data/results
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.evaluation.eval_utils import (
    _ROOT, build_faiss_index, extract_material,
    load_model_embeddings, load_metadata, load_aligned_data,
    DEFAULT_META_PATH, DEFAULT_EMB_DIR, DEFAULT_OUTPUT_DIR, MODEL_CONFIGS,
)


def run_retrieval(embeddings, df, K=10):
    """Векторизованная Cross-Scale Retrieval.

    Inner loop использует numpy-массивы; batch FAISS search вместо
    поштучного — ускорение ~10-50x vs предыдущая версия на 22k тайлов.
    """
    index, normed = build_faiss_index(embeddings)

    df = df.reset_index(drop=True).copy()
    df['material'] = extract_material(df['source_image'])

    materials = df['material'].to_numpy()
    sources = df['source_image'].to_numpy()
    mags = df['mag'].to_numpy()
    mag_is_nan = pd.isna(mags)

    # Only keep materials that have >1 unique magnification (for cross-scale)
    mat_mag = df.dropna(subset=['mag']).groupby('material')['mag'].nunique()
    valid_mats = set(mat_mag[mat_mag > 1].index.tolist())
    test_mask = np.array([m in valid_mats for m in materials]) & ~mag_is_nan
    test_indices = np.flatnonzero(test_mask)

    print(f"  Test tiles: {len(test_indices)} across {len(valid_mats)} materials")

    query_embs = normed[test_indices]
    sims_all, ids_all = index.search(query_embs, K * 5)

    precision_material = []
    precision_cross = []

    for i, q_idx in enumerate(tqdm(test_indices, desc=f"  Retrieval K={K}",
                                    leave=False)):
        q_material = materials[q_idx]
        q_mag = mags[q_idx]
        q_source = sources[q_idx]

        ret_ids = ids_all[i]
        ret_sources = sources[ret_ids]
        ret_materials = materials[ret_ids]
        ret_mags = mags[ret_ids]
        ret_mag_valid = ~pd.isna(ret_mags)

        keep = (ret_ids != q_idx) & (ret_sources != q_source)
        kept_materials = ret_materials[keep][:K]
        kept_mags = ret_mags[keep][:K]
        kept_mag_valid = ret_mag_valid[keep][:K]
        if len(kept_materials) < K:
            continue

        is_same_material = (kept_materials == q_material)
        is_cross_scale = is_same_material & kept_mag_valid & (kept_mags != q_mag)

        precision_material.append(float(is_same_material.sum()) / K)
        precision_cross.append(float(is_cross_scale.sum()) / K)

    return np.mean(precision_material), np.mean(precision_cross)


def main(args):
    print("Loading and aligning data...\n")

    results = []

    for name in MODEL_CONFIGS:
        try:
            emb, df = load_aligned_data(name, args.emb_dir, args.meta_path)
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
            continue

        print(f"Running {name}...")
        p_mat, p_cross = run_retrieval(emb, df, K=args.K)
        results.append({
            'Model': name,
            f'P@{args.K} Material': f'{p_mat:.4f}',
            f'P@{args.K} Cross-Scale': f'{p_cross:.4f}',
        })
        print(f"  -> P@{args.K} Material={p_mat:.4f}, Cross-Scale={p_cross:.4f}\n")

    if not results:
        print("ERROR: No model embeddings loaded. Exiting.")
        return

    print("=" * 60)
    print(f"CROSS-SCALE RETRIEVAL (K={args.K})")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    print("P@K Material:    % of neighbors from the same material")
    print("P@K Cross-Scale: % from same material at DIFFERENT magnification")
    print("                 (higher = better scale invariance)")
    print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cross_scale_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-Scale Retrieval Comparison")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--K", type=int, default=10)
    args = parser.parse_args()
    main(args)
