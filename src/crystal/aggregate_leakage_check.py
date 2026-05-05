"""
Сводит crystal_retrieval_*_bootstrap.csv в одну таблицу для сравнения
SimCLR vs flatten, с/без spatial split.

Артефакт для главы про data leakage в crystal-retrieval: показывает,
что SimCLR держит P@K_miller под spatial split, а flatten коллапсирует —
значит, его исходный P@K объяснялся тривиальной угловой близостью
соседних патчей, а SimCLR — нет.

Запуск:
  python -m src.crystal.aggregate_leakage_check
"""
import argparse
from pathlib import Path

import pandas as pd


def load_bootstrap(results_dir: Path, model: str, split_deg: float) -> pd.DataFrame:
    suffix = f"_{model}"
    if split_deg > 0:
        suffix += f"_split{int(split_deg)}deg"
    path = results_dir / f"crystal_retrieval_K10{suffix}_bootstrap.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["model"] = model
    df["spatial_split_deg"] = split_deg
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path,
                        default=root / "data" / "results" / "crystal_leakage_check")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells = [
        ("simclr", 0.0),
        ("simclr", 12.0),
        ("flatten", 0.0),
        ("flatten", 12.0),
    ]
    rows = [load_bootstrap(args.results_dir, m, d) for m, d in cells]
    merged = pd.concat(rows, ignore_index=True)

    out = args.out or (args.results_dir / "leakage_summary.csv")
    merged.to_csv(out, index=False)
    print(f"Combined CI rows -> {out}")

    # Pretty pivot for the writeup: P@10_miller as model × split.
    pm = merged[merged["metric"] == "precision@10_miller"].copy()
    pm["ci"] = pm.apply(
        lambda r: f"{r['mean']:.4f} [{r['ci_lo (95%)']:.4f}, {r['ci_hi (95%)']:.4f}]",
        axis=1,
    )
    pivot = pm.pivot(index="model", columns="spatial_split_deg", values="ci")
    pivot.columns = [f"P@10_miller (split={c}°)" for c in pivot.columns]
    print("\nP@10_miller with 95% bootstrap CI:")
    print(pivot.to_string())

    summary_out = args.results_dir / "leakage_summary_overall.csv"
    pivot.to_csv(summary_out)
    print(f"\nOverall pivot -> {summary_out}")


if __name__ == "__main__":
    main()
