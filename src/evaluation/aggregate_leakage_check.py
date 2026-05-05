"""
Сводит cross_scale_ci_*.csv в одну таблицу для сравнения baselines.

Используется как финальный артефакт для главы про data leakage:
показывает, что SimCLR превосходит наивный flatten-baseline и
ImageNet-инициализированный ResNet50 с непересекающимися 95% CI.

Запуск:
  python -m src.evaluation.aggregate_leakage_check
  python -m src.evaluation.aggregate_leakage_check --models flatten32 imagenet_resnet50 simclr_finetuned
"""
import argparse
from pathlib import Path

import pandas as pd


def aggregate(results_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        ci_path = results_dir / f"cross_scale_ci_{m}.csv"
        if not ci_path.exists():
            print(f"WARN: missing {ci_path}")
            continue
        df = pd.read_csv(ci_path)
        df["model"] = m
        rows.append(df)
    if not rows:
        raise SystemExit("No CI files found")
    return pd.concat(rows, ignore_index=True)


def format_overall(merged: pd.DataFrame) -> pd.DataFrame:
    overall = merged[merged["scope"] == "overall"].copy()
    overall["ci"] = overall.apply(
        lambda r: f"{r['mean']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", axis=1
    )
    pivot = overall.pivot(index="model", columns="metric", values="ci")
    pivot.columns = [c.replace("precision@K_", "P@10 ") for c in pivot.columns]
    return pivot


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path,
                        default=root / "data" / "results" / "leakage_check")
    parser.add_argument("--models", nargs="+",
                        default=["flatten32", "imagenet_resnet50", "simclr_finetuned"])
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <results_dir>/leakage_summary.csv)")
    args = parser.parse_args()

    merged = aggregate(args.results_dir, args.models)
    out = args.out or (args.results_dir / "leakage_summary.csv")
    merged.to_csv(out, index=False)
    print(f"Combined CI rows -> {out}")

    summary = format_overall(merged)
    print("\nOverall P@10 with 95% bootstrap CI:")
    print(summary.to_string())

    summary_out = args.results_dir / "leakage_summary_overall.csv"
    summary.to_csv(summary_out)
    print(f"\nOverall pivot -> {summary_out}")
