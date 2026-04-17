"""
Единый entrypoint для всех SEM-метрик.

Запускает по порядку:
  1. Cross-Scale Retrieval (P@K material, P@K cross-scale)
  2. Scale Invariance Metrics (Entropy, Cramér's V, AMI, MagVarRatio)

Все результаты сохраняются в output_dir как CSV + TXT.

Usage:
  python src/evaluation/run_sem_evaluation.py
  python src/evaluation/run_sem_evaluation.py --K 5 --n_clusters 6
  python src/evaluation/run_sem_evaluation.py --no-normalize  # raw embeddings for KMeans
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.evaluation.eval_utils import (
    load_metadata, load_model_embeddings, load_aligned_data, l2_normalize,
    DEFAULT_META_PATH, DEFAULT_EMB_DIR, DEFAULT_OUTPUT_DIR, MODEL_CONFIGS,
)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_lines = []
    report_lines.append(f"SEM Evaluation Report — {timestamp}")
    report_lines.append(f"K={args.K}, n_clusters={args.n_clusters}, normalize={args.normalize}")
    report_lines.append("=" * 70)

    # Load aligned data for each model (embedding_names.csv → metadata merge)
    aligned = {}
    for name in MODEL_CONFIGS:
        try:
            emb, df = load_aligned_data(name, args.emb_dir, args.meta_path)
            aligned[name] = (emb, df)
            print(f"  Loaded {name}: {emb.shape}")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")

    if not aligned:
        print("ERROR: No model embeddings found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 1. Cross-Scale Retrieval (canonical implementation)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Cross-Scale Retrieval")
    print(f"{'='*60}\n")

    try:
        from src.evaluation.cross_scale_retrieval import run_retrieval_test
        import pandas as pd

        retrieval_results = []
        for name, (emb, df) in aligned.items():
            print(f"Running {name}...")
            results_df = run_retrieval_test(emb, df, K=args.K)
            pm_col = f'precision@{args.K}_material'
            pcs_col = f'precision@{args.K}_cross_scale'
            p_mat = results_df[pm_col].mean() if len(results_df) > 0 else 0.0
            p_cross = results_df[pcs_col].mean() if len(results_df) > 0 else 0.0
            retrieval_results.append({
                'Model': name,
                f'P@{args.K} Material': round(p_mat, 4),
                f'P@{args.K} Cross-Scale': round(p_cross, 4),
                'N tiles evaluated': len(results_df),
            })
            print(f"  -> P@{args.K} Material={p_mat:.4f}, Cross-Scale={p_cross:.4f} "
                  f"({len(results_df)} tiles)\n")

        ret_df = pd.DataFrame(retrieval_results)
        print(ret_df.to_string(index=False))
        ret_df.to_csv(output_dir / "cross_scale_comparison.csv", index=False)
        report_lines.append("\nCross-Scale Retrieval:")
        report_lines.append(ret_df.to_string(index=False))
    except Exception as e:
        print(f"  Cross-scale retrieval failed: {e}")
        import traceback; traceback.print_exc()
        report_lines.append(f"\nCross-Scale Retrieval: FAILED ({e})")

    # ------------------------------------------------------------------
    # 2. Scale Invariance Metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Scale Invariance Metrics")
    print(f"{'='*60}\n")

    try:
        from src.evaluation.scale_invariance_metrics import (
            magnification_entropy, cramers_v, mag_variance_ratio,
            MAG_BINS, MAG_BIN_LABELS,
        )
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_mutual_info_score
        import numpy as np
        import pandas as pd

        max_ent = np.log2(len(MAG_BIN_LABELS))
        scale_results = []

        for name, (emb, df) in aligned.items():
            print(f"Processing {name}...")

            mags = df['mag'].values
            log_mags = np.log10(mags)
            mag_binned = pd.cut(mags, bins=MAG_BINS, labels=MAG_BIN_LABELS)

            emb_proc = l2_normalize(emb) if args.normalize else emb

            kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb_proc)

            _, weighted_avg_ent, _ = magnification_entropy(
                labels, mags, args.n_clusters, MAG_BINS, MAG_BIN_LABELS
            )
            cv = cramers_v(labels, mag_binned)
            ami = adjusted_mutual_info_score(labels, mag_binned)
            mvr = mag_variance_ratio(labels, log_mags, args.n_clusters)

            scale_results.append({
                'Model': name,
                'Entropy': f"{weighted_avg_ent:.3f} ({weighted_avg_ent/max_ent:.0%})",
                "Cramer's V": round(cv, 4),
                'AMI': round(ami, 4),
                'MagVarRatio': round(mvr, 4),
            })

        scale_df = pd.DataFrame(scale_results)
        print(f"\n{'='*70}")
        print("SCALE INVARIANCE METRICS (lower = better for scale invariance)")
        print(f"{'='*70}")
        print(scale_df.to_string(index=False))
        scale_df.to_csv(output_dir / "scale_invariance_metrics.csv", index=False)
        report_lines.append(f"\nScale Invariance Metrics (normalize={args.normalize}):")
        report_lines.append(scale_df.to_string(index=False))
    except Exception as e:
        print(f"  Scale invariance metrics failed: {e}")
        import traceback; traceback.print_exc()
        report_lines.append(f"\nScale Invariance Metrics: FAILED ({e})")

    # ------------------------------------------------------------------
    # Save combined report
    # ------------------------------------------------------------------
    report_path = output_dir / f"sem_eval_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nCombined report saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SEM Evaluation Pipeline")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--K", type=int, default=10, help="K for retrieval")
    parser.add_argument("--n_clusters", type=int, default=4, help="K for KMeans clustering")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize embeddings before KMeans (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    args = parser.parse_args()
    main(args)
