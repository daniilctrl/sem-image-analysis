"""k-NN classification probe на SSL-эмбеддингах.

Дополняет linear_probe.py: linear probe меряет «разделяется ли distribution
классов линейно», k-NN probe меряет «локальная геометрия embedding space
согласована с классами?». Это два независимых signals о качестве SSL.

k-NN probe — стандарт оценки SSL (DINO/MoCo/SwAV benchmark). Ключевые
свойства:
  - **Parameter-free** (нет обучения classifier, только distance lookup).
  - **Scale-invariant** после L2-normalization.
  - Непосредственно измеряет то, что использует retrieval: cosine similarity
    между соседями в embedding space. Если k-NN probe низкий, retrieval
    тоже будет плохой.

Протокол:
  - Stratified K-fold CV (по умолчанию K=5).
  - Weighted k-NN: голоса соседей взвешены по cosine similarity.
  - Перебор нескольких K-окон: {1, 5, 20} — стандарт из DINO paper.
  - Метрики: accuracy, balanced_accuracy, macro-F1.
  - Bootstrap 95% CI для mean accuracy через src.utils.stats.bootstrap_metric_ci.

Использование:
  python3 src/evaluation/knn_probe.py
  python3 src/evaluation/knn_probe.py --n_folds 10 --knn_k 1 5 20 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.evaluation.eval_utils import (  # noqa: E402
    MODEL_CONFIGS, DEFAULT_META_PATH, DEFAULT_EMB_DIR, DEFAULT_OUTPUT_DIR,
    load_aligned_data, l2_normalize,
)
from src.utils.repro import set_global_seed  # noqa: E402
from src.utils.stats import bootstrap_metric_ci  # noqa: E402


def load_sft_annotations(
    annotations_path: Path,
    df_aligned: pd.DataFrame,
    exclude_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Загружает SFT разметку и сшивает с aligned-метаданными.

    Дубликаты tile_name -> keep first. Inner join с df_aligned.
    """
    annot = pd.read_csv(annotations_path).rename(columns={"filename": "tile_name"})
    annot = annot.drop_duplicates(subset=["tile_name"], keep="first")
    if exclude_classes:
        before = len(annot)
        annot = annot[~annot["cluster"].isin(exclude_classes)]
        print(f"  Excluded classes {exclude_classes}: {before} -> {len(annot)} tiles")
    return df_aligned.merge(annot, on="tile_name", how="inner")


def run_knn_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Stratified K-fold CV с weighted k-NN на cosine distance.

    Для каждого K-окна возвращает per-fold и aggregate метрики.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Per-k folds: {k: DataFrame(fold, acc, bal_acc, macro_f1)}
    per_k_folds: dict[int, list[dict]] = {k: [] for k in k_values}
    per_k_y_true: dict[int, list] = {k: [] for k in k_values}
    per_k_y_pred: dict[int, list] = {k: [] for k in k_values}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # L2 уже сделана снаружи, но на всякий — sklearn KNN не нормализует сам.
        # metric='cosine' использует 1 - cos_similarity, weights='distance'
        # даёт взвешенное голосование.
        for k in k_values:
            k_eff = min(k, len(X_tr))
            clf = KNeighborsClassifier(
                n_neighbors=k_eff,
                weights="distance",
                metric="cosine",
                algorithm="brute",  # cosine не поддерживается kd/ball tree
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            bal_acc = balanced_accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

            per_k_folds[k].append({
                "fold": fold_i,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "k_eff": int(k_eff),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "macro_f1": round(f1, 4),
            })
            per_k_y_true[k].extend(y_te.tolist())
            per_k_y_pred[k].extend(y_pred.tolist())

    return {
        "per_k_folds": {k: pd.DataFrame(v) for k, v in per_k_folds.items()},
        "per_k_y_true": {k: np.array(v) for k, v in per_k_y_true.items()},
        "per_k_y_pred": {k: np.array(v) for k, v in per_k_y_pred.items()},
    }


def run_for_model(
    model_name: str,
    args,
    annotations_path: Path,
    output_dir: Path,
) -> list[dict] | None:
    """k-NN probe для одной модели по всем K-окнам. Возвращает summary rows."""
    try:
        emb, df_aligned = load_aligned_data(model_name, args.emb_dir, args.meta_path)
    except FileNotFoundError as e:
        print(f"  SKIP {model_name}: {e}")
        return None

    # Добавляем индекс строки эмбеддинга ДО merge с аннотациями,
    # чтобы после inner join можно было точно извлечь подмножество строк.
    df_aligned = df_aligned.copy()
    df_aligned["_emb_row"] = np.arange(len(df_aligned))

    merged = load_sft_annotations(annotations_path, df_aligned,
                                   exclude_classes=args.exclude_classes)
    if len(merged) < args.n_folds * 2:
        print(f"  SKIP {model_name}: only {len(merged)} annotated tiles")
        return None

    # Извлекаем эмбеддинги для подмножества — через _emb_row,
    # гарантируя len(X) == len(y).
    X = emb[merged["_emb_row"].values]
    if args.normalize:
        X = l2_normalize(X)
    y = merged["cluster"].to_numpy()

    class_counts = pd.Series(y).value_counts()
    print(f"  {model_name}: {len(y)} labeled tiles, {len(class_counts)} classes")

    rare = class_counts[class_counts < args.n_folds]
    if len(rare) > 0:
        print(f"    WARNING: classes below n_folds={args.n_folds}: {dict(rare)}")

    result = run_knn_probe_cv(X, y, args.knn_k, n_folds=args.n_folds, seed=args.seed)

    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    summary_rows = []
    for k in args.knn_k:
        fold_df = result["per_k_folds"][k]
        fold_path = output_dir / f"knn_probe_{safe_name}_k{k}_folds.csv"
        fold_df.to_csv(fold_path, index=False)

        # Bootstrap CI для accuracy (по per-fold значениям — точечно, но
        # с минимум 5 фолдами оценка CI грубая; для полноты фиксируем)
        acc_vals = fold_df["accuracy"].to_numpy()
        _, lo, hi = bootstrap_metric_ci(acc_vals, n_bootstrap=1000, seed=args.seed)

        labels_sorted = sorted(set(y))
        cm = confusion_matrix(
            result["per_k_y_true"][k], result["per_k_y_pred"][k],
            labels=labels_sorted,
        )
        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
        cm_df.to_csv(output_dir / f"knn_probe_{safe_name}_k{k}_confusion.csv")

        agg = {
            "model": model_name,
            "knn_k": k,
            "n_labeled": int(len(y)),
            "n_classes": int(len(class_counts)),
            "mean_accuracy": round(float(fold_df["accuracy"].mean()), 4),
            "std_accuracy": round(float(fold_df["accuracy"].std(ddof=1)), 4),
            "ci_lo_accuracy": round(lo, 4),
            "ci_hi_accuracy": round(hi, 4),
            "mean_balanced_accuracy": round(float(fold_df["balanced_accuracy"].mean()), 4),
            "mean_macro_f1": round(float(fold_df["macro_f1"].mean()), 4),
        }
        summary_rows.append(agg)

        print(
            f"    k={k}: acc = {agg['mean_accuracy']:.4f} "
            f"[{agg['ci_lo_accuracy']:.4f}, {agg['ci_hi_accuracy']:.4f}], "
            f"macro-F1 = {agg['mean_macro_f1']:.4f}"
        )

    return summary_rows


def main() -> int:
    _root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="k-NN classification probe on SSL embeddings",
    )
    parser.add_argument("--annotations_path", type=str,
                        default=str(_root / "data" / "sft_annotations.csv"))
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--knn_k", type=int, nargs="+", default=[1, 5, 20],
                        help="K values for k-NN probe (DINO uses 1, 5, 20)")
    parser.add_argument("--exclude_classes", type=str, nargs="*",
                        default=["trash"])
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize embeddings before k-NN (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    used_seed = set_global_seed(args.seed, deterministic_torch=False)
    print(f"[repro] Global seed fixed: {used_seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = Path(args.annotations_path)
    if not annotations_path.exists():
        print(f"ERROR: annotations not found: {annotations_path}", file=sys.stderr)
        return 1

    print(f"Loading annotations from {annotations_path}...")

    summary = []
    for model_name in MODEL_CONFIGS:
        print(f"\n--- {model_name} ---")
        rows = run_for_model(model_name, args, annotations_path, output_dir)
        if rows is not None:
            summary.extend(rows)

    if not summary:
        print("ERROR: no models evaluated.")
        return 2

    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "knn_probe_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*90}")
    print(f"k-NN probe summary (n_folds={args.n_folds}, K values={args.knn_k}, "
          f"L2-normalized={args.normalize})")
    print(f"{'='*90}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
