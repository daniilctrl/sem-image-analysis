"""Linear probe: supervised evaluation of frozen SSL embeddings.

Linear evaluation protocol (Chen et al. 2020) — де-факто стандарт для
оценки качества SSL-представлений. Принцип:

  1. Берём замороженный encoder (SSL-обученный или Baseline).
  2. Обучаем простой классификатор (логистическая регрессия / 1 слой MLP)
     поверх эмбеддингов на размеченном подмножестве.
  3. Метрика качества — accuracy / macro-F1 на hold-out fold.

Зачем это здесь
---------------

Без linear probe сравнение Baseline / SimCLR / BYOL опирается только
на intrinsic clustering metrics (Silhouette, CH, DB) и на cross-scale
precision. Intrinsic-метрики не отвечают на ключевой вопрос: «различает ли
модель релевантные с точки зрения эксперта структурные классы?»

В проекте уже есть ручная разметка `data/sft_annotations.csv` (187 меток,
8 морфологических кластеров). Linear probe на этой разметке закрывает
пробел и усиливает аргументацию «SimCLR/BYOL лучше Baseline» (или
показывает, что нет).

Протокол
--------

- Аугментации отключены (эмбеддинги извлекаются на фиксированных трансформах).
- Эмбеддинги L2-нормализованы (как во всех остальных pipeline-скриптах).
- Стратифицированный K-fold (по умолчанию K=5, stratified по классу).
- Классификатор: scikit-learn LogisticRegression (multinomial, lbfgs,
  C=10 — стандартный режим linear probe).
- Метрики per fold: accuracy, macro-F1, balanced accuracy.

Использование
-------------

  python3 src/evaluation/linear_probe.py
  python3 src/evaluation/linear_probe.py --n_folds 10
  python3 src/evaluation/linear_probe.py \\
      --annotations_path data/sft_annotations.csv \\
      --emb_dir data/embeddings \\
      --output_dir data/results \\
      --exclude_classes trash
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    """Загружает ручную разметку и выравнивает с (embedding-aligned) метаданными.

    При дублировании метки для одного tile_name берётся первая запись
    (стабильно по порядку в CSV).
    """
    annot = pd.read_csv(annotations_path)
    annot = annot.rename(columns={"filename": "tile_name"})
    annot = annot.drop_duplicates(subset=["tile_name"], keep="first")

    if exclude_classes:
        before = len(annot)
        annot = annot[~annot["cluster"].isin(exclude_classes)]
        print(f"  Excluded classes {exclude_classes}: "
              f"{before} -> {len(annot)} annotated tiles")

    # Inner join с aligned metadata: оставляем только те разметки,
    # у которых есть эмбеддинг
    merged = df_aligned.merge(annot, on="tile_name", how="inner")
    return merged


def run_linear_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    C: float = 10.0,
    max_iter: int = 2000,
) -> dict:
    """Стратифицированный K-fold linear probe (логистическая регрессия).

    Возвращает словарь с per-fold и агрегированными метриками.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold = []
    all_y_true = []
    all_y_pred = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Standardize (after L2-norm это близко к no-op, но не вредит)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # sklearn >= 1.5: multi_class устарел, lbfgs уже использует multinomial
        # автоматически при наличии >= 3 классов.
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

        per_fold.append({
            "fold": fold_i,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(f1, 4),
        })

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Aggregated stats + bootstrap CI по fold-accuracies (грубая оценка:
    # K фолдов — маленькая выборка, но даёт интервал). Более строгий
    # вариант — bootstrap per-sample prediction accuracy, но для защиты
    # и summary-таблицы достаточно per-fold.
    fold_df = pd.DataFrame(per_fold)
    _, acc_lo, acc_hi = bootstrap_metric_ci(
        fold_df["accuracy"].to_numpy(), n_bootstrap=1000, seed=seed,
    )
    _, f1_lo, f1_hi = bootstrap_metric_ci(
        fold_df["macro_f1"].to_numpy(), n_bootstrap=1000, seed=seed,
    )
    aggregate = {
        "mean_accuracy": round(float(fold_df["accuracy"].mean()), 4),
        "std_accuracy": round(float(fold_df["accuracy"].std(ddof=1)), 4),
        "ci_lo_accuracy": round(acc_lo, 4),
        "ci_hi_accuracy": round(acc_hi, 4),
        "mean_balanced_accuracy": round(float(fold_df["balanced_accuracy"].mean()), 4),
        "std_balanced_accuracy": round(float(fold_df["balanced_accuracy"].std(ddof=1)), 4),
        "mean_macro_f1": round(float(fold_df["macro_f1"].mean()), 4),
        "std_macro_f1": round(float(fold_df["macro_f1"].std(ddof=1)), 4),
        "ci_lo_macro_f1": round(f1_lo, 4),
        "ci_hi_macro_f1": round(f1_hi, 4),
    }

    return {
        "per_fold": fold_df,
        "aggregate": aggregate,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
    }


def run_for_model(
    model_name: str,
    args,
    annotations_path: Path,
    output_dir: Path,
) -> dict | None:
    """Запускает linear probe для одной модели. Возвращает результат или None."""
    try:
        emb, df_aligned = load_aligned_data(model_name, args.emb_dir, args.meta_path)
    except FileNotFoundError as e:
        print(f"  SKIP {model_name}: {e}")
        return None

    merged = load_sft_annotations(
        annotations_path,
        df_aligned,
        exclude_classes=args.exclude_classes,
    )

    if len(merged) < args.n_folds * 2:
        print(f"  SKIP {model_name}: only {len(merged)} annotated tiles; "
              f"need >= {args.n_folds * 2}")
        return None

    # Извлекаем эмбеддинги для подмножества
    idx_in_aligned = df_aligned.reset_index().merge(
        merged[["tile_name"]], on="tile_name", how="inner"
    )["index"].values
    X = emb[idx_in_aligned]
    if args.normalize:
        X = l2_normalize(X)
    y = merged["cluster"].to_numpy()

    class_counts = pd.Series(y).value_counts()
    print(f"  {model_name}: {len(y)} labeled tiles, {len(class_counts)} classes")
    print(f"    class distribution: {dict(class_counts)}")

    # Проверка: все классы должны иметь хотя бы n_folds примеров
    rare = class_counts[class_counts < args.n_folds]
    if len(rare) > 0:
        print(f"    WARNING: classes with fewer than n_folds={args.n_folds} "
              f"samples will skew CV: {dict(rare)}")

    result = run_linear_probe_cv(
        X, y, n_folds=args.n_folds, seed=args.seed,
        C=args.C, max_iter=args.max_iter,
    )

    agg = result["aggregate"]
    print(f"    accuracy = {agg['mean_accuracy']:.4f} "
          f"[{agg['ci_lo_accuracy']:.4f}, {agg['ci_hi_accuracy']:.4f}] (95% CI)")
    print(f"    balanced = {agg['mean_balanced_accuracy']:.4f} "
          f"+/- {agg['std_balanced_accuracy']:.4f}")
    print(f"    macro-F1 = {agg['mean_macro_f1']:.4f} "
          f"[{agg['ci_lo_macro_f1']:.4f}, {agg['ci_hi_macro_f1']:.4f}] (95% CI)")

    # Save per-fold table
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    per_fold_path = output_dir / f"linear_probe_{safe_name}_folds.csv"
    result["per_fold"].to_csv(per_fold_path, index=False)

    # Confusion matrix
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_path = output_dir / f"linear_probe_{safe_name}_confusion.csv"
    cm_df.to_csv(cm_path)

    return {
        "model": model_name,
        "n_labeled": int(len(y)),
        "n_classes": int(len(class_counts)),
        **result["aggregate"],
    }


def main() -> int:
    _root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Linear probe on SSL embeddings")
    parser.add_argument("--annotations_path", type=str,
                        default=str(_root / "data" / "sft_annotations.csv"))
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--exclude_classes", type=str, nargs="*",
                        default=["trash"],
                        help="Classes to exclude from evaluation (default: trash)")
    parser.add_argument("--C", type=float, default=10.0,
                        help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize embeddings before linear probe (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    used_seed = set_global_seed(args.seed, deterministic_torch=False)
    print(f"[repro] Global seed fixed: {used_seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = Path(args.annotations_path)
    if not annotations_path.exists():
        print(f"ERROR: annotations file not found: {annotations_path}",
              file=sys.stderr)
        return 1

    print(f"Loading annotations from {annotations_path}...")

    summary = []
    for model_name in MODEL_CONFIGS:
        print(f"\n--- {model_name} ---")
        res = run_for_model(model_name, args, annotations_path, output_dir)
        if res is not None:
            summary.append(res)

    if not summary:
        print("ERROR: no models evaluated.")
        return 2

    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "linear_probe_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print(f"Linear probe summary (n_folds={args.n_folds}, "
          f"L2-normalized={args.normalize})")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
