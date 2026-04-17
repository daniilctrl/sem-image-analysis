"""Unified entrypoint для всех SEM-метрик.

Запускает по порядку:
  1. Cross-Scale Retrieval (P@K material, P@K cross-scale, bootstrap CI)
  2. Scale Invariance Metrics (Entropy, Cramér's V, AMI, MagVarRatio)
  3. Linear Probe (accuracy, macro-F1 с CI; stratified K-fold на sft_annotations)
  4. k-NN Probe (K=1/5/20, accuracy, macro-F1 с CI)

Все результаты сохраняются в единый markdown-отчёт с datestamp в названии.

Зачем unified:
  Раньше было 4 отдельных запуска и 4 отдельных CSV/PNG. Для защиты
  и сравнения конфигураций обучения удобнее один файл.

Usage:
  python src/evaluation/run_sem_evaluation.py
  python src/evaluation/run_sem_evaluation.py --K 5 --n_clusters 6
  python src/evaluation/run_sem_evaluation.py --no-normalize
  python src/evaluation/run_sem_evaluation.py --skip_knn_probe  # speed
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.evaluation.eval_utils import (  # noqa: E402
    DEFAULT_META_PATH, DEFAULT_EMB_DIR, DEFAULT_OUTPUT_DIR, MODEL_CONFIGS,
    load_aligned_data, l2_normalize,
)
from src.utils.repro import set_global_seed  # noqa: E402
from src.utils.stats import bootstrap_metric_ci, format_mean_ci  # noqa: E402


def _safe_df_to_markdown(df) -> str:
    """Рендерит DataFrame в markdown-таблицу без зависимости от tabulate."""
    if len(df) == 0:
        return "_(empty)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_cross_scale(aligned: dict, args, report: list[str]) -> None:
    print(f"\n{'='*60}")
    print("STEP 1: Cross-Scale Retrieval")
    print(f"{'='*60}\n")

    from src.evaluation.cross_scale_retrieval import run_retrieval_test
    import pandas as pd

    rows = []
    for name, (emb, df) in aligned.items():
        print(f"Running {name}...")
        res = run_retrieval_test(emb, df, K=args.K)
        pm = f"precision@{args.K}_material"
        pcs = f"precision@{args.K}_cross_scale"

        if len(res) == 0:
            print(f"  -> no eligible tiles")
            rows.append({
                "Model": name,
                f"P@{args.K} Material": "n/a",
                f"P@{args.K} Cross-Scale": "n/a",
                "N tiles": 0,
            })
            continue

        pm_p, pm_lo, pm_hi = bootstrap_metric_ci(
            res[pm].to_numpy(), n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        pcs_p, pcs_lo, pcs_hi = bootstrap_metric_ci(
            res[pcs].to_numpy(), n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        rows.append({
            "Model": name,
            f"P@{args.K} Material": format_mean_ci(pm_p, pm_lo, pm_hi),
            f"P@{args.K} Cross-Scale": format_mean_ci(pcs_p, pcs_lo, pcs_hi),
            "N tiles": int(len(res)),
        })
        print(f"  -> P@{args.K} Material={format_mean_ci(pm_p, pm_lo, pm_hi)}, "
              f"Cross-Scale={format_mean_ci(pcs_p, pcs_lo, pcs_hi)}")

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    report.append("## 1. Cross-Scale Retrieval\n")
    report.append(_safe_df_to_markdown(summary))
    report.append("")


def run_scale_invariance(aligned: dict, args, report: list[str]) -> None:
    print(f"\n{'='*60}")
    print("STEP 2: Scale Invariance Metrics")
    print(f"{'='*60}\n")

    from src.evaluation.scale_invariance_metrics import (
        magnification_entropy, cramers_v, mag_variance_ratio,
        MAG_BINS, MAG_BIN_LABELS, compute_adaptive_mag_bins,
    )
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    import numpy as np
    import pandas as pd

    # Единый набор бинов на весь запуск, из первой доступной модели.
    first_mags = next(iter(aligned.values()))[1]["mag"].values
    if args.adaptive_bins:
        bins, bin_labels = compute_adaptive_mag_bins(
            first_mags, n_bins=args.n_bins, method="log_quantile",
        )
    else:
        bins, bin_labels = MAG_BINS, MAG_BIN_LABELS
    max_ent = np.log2(len(bin_labels))
    print(f"  Using mag bins: {bin_labels}")

    rows = []
    for name, (emb, df) in aligned.items():
        mags = df["mag"].values
        log_mags = np.log10(mags)
        mag_binned = pd.cut(mags, bins=bins, labels=bin_labels)

        emb_proc = l2_normalize(emb) if args.normalize else emb
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        labels = kmeans.fit_predict(emb_proc)

        _, weighted_avg_ent, _ = magnification_entropy(
            labels, mags, args.n_clusters, bins, bin_labels,
        )
        cv = cramers_v(labels, mag_binned)
        ami = adjusted_mutual_info_score(labels, mag_binned)
        mvr = mag_variance_ratio(labels, log_mags, args.n_clusters)

        rows.append({
            "Model": name,
            "Entropy": f"{weighted_avg_ent:.3f} ({weighted_avg_ent/max_ent:.0%})",
            "Cramér's V (low=good)": round(cv, 4),
            "AMI (low=good)": round(ami, 4),
            "MagVarRatio (low=good)": round(mvr, 4),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    report.append("## 2. Scale Invariance Metrics\n")
    report.append(f"(Adaptive mag bins: `{bin_labels}`; "
                  f"K-Means with `n_clusters={args.n_clusters}`; "
                  f"L2-normalized={args.normalize})\n")
    report.append(_safe_df_to_markdown(summary))
    report.append("")
    report.append("_Lower = better for scale invariance. "
                  "Cramér's V and AMI measure cluster<->magnification association; "
                  "MagVarRatio is fraction of mag variance explained by clusters._")
    report.append("")


def run_linear_probe(args, report: list[str]) -> None:
    print(f"\n{'='*60}")
    print("STEP 3: Linear Probe")
    print(f"{'='*60}\n")

    from src.evaluation.linear_probe import run_for_model as lp_run
    import pandas as pd

    probe_args = argparse.Namespace(
        emb_dir=args.emb_dir, meta_path=args.meta_path,
        exclude_classes=["trash"], C=10.0, max_iter=2000,
        normalize=args.normalize, seed=args.seed, n_folds=args.n_folds,
        annotations_path=str(args.annotations_path), output_dir=str(args.output_dir),
    )

    rows = []
    for name in MODEL_CONFIGS:
        print(f"--- {name} ---")
        res = lp_run(name, probe_args, args.annotations_path, Path(args.output_dir))
        if res is not None:
            rows.append(res)

    if not rows:
        report.append("## 3. Linear Probe\n_(no models evaluated)_\n")
        return

    summary = pd.DataFrame(rows)[[
        "model", "n_labeled", "n_classes",
        "mean_accuracy", "ci_lo_accuracy", "ci_hi_accuracy",
        "mean_macro_f1", "ci_lo_macro_f1", "ci_hi_macro_f1",
    ]]
    summary["accuracy (95% CI)"] = summary.apply(
        lambda r: format_mean_ci(r["mean_accuracy"], r["ci_lo_accuracy"],
                                 r["ci_hi_accuracy"]),
        axis=1,
    )
    summary["macro_f1 (95% CI)"] = summary.apply(
        lambda r: format_mean_ci(r["mean_macro_f1"], r["ci_lo_macro_f1"],
                                 r["ci_hi_macro_f1"]),
        axis=1,
    )
    report_df = summary[["model", "n_labeled", "n_classes",
                         "accuracy (95% CI)", "macro_f1 (95% CI)"]]
    print(report_df.to_string(index=False))

    report.append("## 3. Linear Probe (logistic regression, stratified K-fold)\n")
    report.append(f"(K={args.n_folds}, C=10, L2-normalized={args.normalize})\n")
    report.append(_safe_df_to_markdown(report_df))
    report.append("")


def run_knn_probe(args, report: list[str]) -> None:
    print(f"\n{'='*60}")
    print("STEP 4: k-NN Probe")
    print(f"{'='*60}\n")

    from src.evaluation.knn_probe import run_for_model as knn_run
    import pandas as pd

    probe_args = argparse.Namespace(
        emb_dir=args.emb_dir, meta_path=args.meta_path,
        exclude_classes=["trash"],
        normalize=args.normalize, seed=args.seed, n_folds=args.n_folds,
        knn_k=args.knn_k,
        annotations_path=str(args.annotations_path), output_dir=str(args.output_dir),
    )

    rows = []
    for name in MODEL_CONFIGS:
        print(f"--- {name} ---")
        res = knn_run(name, probe_args, args.annotations_path, Path(args.output_dir))
        if res is not None:
            rows.extend(res)

    if not rows:
        report.append("## 4. k-NN Probe\n_(no models evaluated)_\n")
        return

    summary = pd.DataFrame(rows)
    summary["accuracy (95% CI)"] = summary.apply(
        lambda r: format_mean_ci(r["mean_accuracy"], r["ci_lo_accuracy"],
                                 r["ci_hi_accuracy"]),
        axis=1,
    )
    report_df = summary[["model", "knn_k", "n_labeled",
                         "accuracy (95% CI)", "mean_macro_f1"]]
    print(report_df.to_string(index=False))

    report.append("## 4. k-NN Probe (weighted cosine, stratified K-fold)\n")
    report.append(f"(K-folds={args.n_folds}, knn_k={args.knn_k}, "
                  f"L2-normalized={args.normalize})\n")
    report.append(_safe_df_to_markdown(report_df))
    report.append("")


def main(args):
    used_seed = set_global_seed(args.seed, deterministic_torch=False)
    print(f"[repro] Global seed fixed: {used_seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report: list[str] = []
    report.append(f"# SEM Evaluation Report — {timestamp}\n")
    report.append(
        f"Configuration: K={args.K}, n_clusters={args.n_clusters}, "
        f"n_folds={args.n_folds}, knn_k={args.knn_k}, "
        f"normalize={args.normalize}, adaptive_bins={args.adaptive_bins}"
    )
    report.append("")

    # Load aligned data once — reused by cross_scale and scale_invariance.
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
        return 1

    if not args.skip_cross_scale:
        try:
            run_cross_scale(aligned, args, report)
        except Exception as e:
            print(f"  Cross-scale FAILED: {e}")
            import traceback; traceback.print_exc()
            report.append(f"## 1. Cross-Scale Retrieval\n_FAILED: {e}_\n")

    if not args.skip_scale_invariance:
        try:
            run_scale_invariance(aligned, args, report)
        except Exception as e:
            print(f"  Scale invariance FAILED: {e}")
            import traceback; traceback.print_exc()
            report.append(f"## 2. Scale Invariance Metrics\n_FAILED: {e}_\n")

    if not args.skip_linear_probe and args.annotations_path.exists():
        try:
            run_linear_probe(args, report)
        except Exception as e:
            print(f"  Linear probe FAILED: {e}")
            import traceback; traceback.print_exc()
            report.append(f"## 3. Linear Probe\n_FAILED: {e}_\n")
    elif not args.annotations_path.exists():
        report.append("## 3. Linear Probe\n"
                      f"_SKIPPED: annotations file not found: {args.annotations_path}_\n")

    if not args.skip_knn_probe and args.annotations_path.exists():
        try:
            run_knn_probe(args, report)
        except Exception as e:
            print(f"  k-NN probe FAILED: {e}")
            import traceback; traceback.print_exc()
            report.append(f"## 4. k-NN Probe\n_FAILED: {e}_\n")
    elif not args.annotations_path.exists():
        report.append("## 4. k-NN Probe\n"
                      f"_SKIPPED: annotations file not found: {args.annotations_path}_\n")

    report_path = output_dir / f"sem_eval_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nCombined report saved: {report_path}")
    return 0


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Unified SEM Evaluation Pipeline")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--annotations_path", type=Path,
                        default=_root / "data" / "sft_annotations.csv")
    parser.add_argument("--K", type=int, default=10, help="K for retrieval")
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="K for KMeans in scale invariance")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="K for stratified CV in linear/k-NN probes")
    parser.add_argument("--knn_k", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize embeddings before KMeans / probes")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--adaptive_bins", action="store_true", default=True,
                        help="Compute mag bins from data distribution")
    parser.add_argument("--no-adaptive-bins", dest="adaptive_bins", action="store_false")
    parser.add_argument("--n_bins", type=int, default=5,
                        help="Number of adaptive mag bins")
    parser.add_argument("--skip_cross_scale", action="store_true")
    parser.add_argument("--skip_scale_invariance", action="store_true")
    parser.add_argument("--skip_linear_probe", action="store_true")
    parser.add_argument("--skip_knn_probe", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args))
