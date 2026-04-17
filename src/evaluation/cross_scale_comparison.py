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
    index, normed = build_faiss_index(embeddings)

    df = df.copy()
    df['material'] = extract_material(df['source_image'])

    # Only keep materials that have >1 unique magnification (for cross-scale test)
    mat_mag = df.dropna(subset=['mag']).groupby('material')['mag'].nunique()
    valid_mats = mat_mag[mat_mag > 1].index
    test_mask = df['material'].isin(valid_mats) & df['mag'].notna()
    test_indices = df[test_mask].index.tolist()

    print(f"  Test tiles: {len(test_indices)} across {len(valid_mats)} materials")

    precision_material = []
    precision_cross = []

    for idx in tqdm(test_indices, desc=f"  Retrieval K={K}", leave=False):
        query = normed[idx].reshape(1, -1)
        q_material = df.loc[idx, 'material']
        q_mag = df.loc[idx, 'mag']
        q_source = df.loc[idx, 'source_image']

        sims, ret_ids = index.search(query, K * 5)

        mat_hits = 0
        cross_hits = 0
        count = 0

        for ret_idx in ret_ids[0]:
            if ret_idx == idx:
                continue
            if df.loc[ret_idx, 'source_image'] == q_source:
                continue

            r_material = df.loc[ret_idx, 'source_image'].split('__')[0]
            r_mag = df.loc[ret_idx, 'mag']

            if r_material == q_material:
                mat_hits += 1
                if pd.notna(r_mag) and r_mag != q_mag:
                    cross_hits += 1

            count += 1
            if count >= K:
                break

        if count >= K:
            precision_material.append(mat_hits / K)
            precision_cross.append(cross_hits / K)

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
