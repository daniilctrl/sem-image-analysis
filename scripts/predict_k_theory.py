"""Теоретическая оценка числа различимых кристаллографических граней k(R).

Реализация опирается на формулу (5) из статьи:

    Никифоров К.А., Егоров Н.В., Шень Ч.-Ч.
    "Реконструкция поверхности полевого электронного эмиттера"
    Поверхность. Рентгеновские, синхротронные и нейтронные исследования,
    2009, №10, с. 100-106.

Формула (5):

    g(hkl) = (2/3) * sqrt(3 * a * r / (h^2 + k^2 + l^2))

где g -- средний радиус грани, a -- параметр решётки, r -- радиус
полусферы эмиттера, (h,k,l) -- индексы Миллера. Зависимость линейная
(в статье) или корневая в зависимости от модели.

В безразмерных единицах (a = 1, r выражается в параметрах решётки):

    g(hkl) = (2/3) * sqrt(3 * r / (h^2 + k^2 + l^2))

Грань считается "различимой" кластеризующей моделью, если её радиус
превышает заданный порог (обычно связан с размером окна проекции
window_radius или с параметром решётки как минимумом атомов в кластере).

Скрипт выдаёт:
  - таблицу g(hkl) для каждого семейства по возрастанию |hkl|^2;
  - число различимых семейств при нескольких порогах g;
  - предсказание k_theory(R) для списка радиусов;
  - график k_theory(R) vs эмпирический k_best (если передан --empirical).

Использование:

    python3 scripts/predict_k_theory.py
    python3 scripts/predict_k_theory.py --radii 50 100 200 316
    python3 scripts/predict_k_theory.py --window_radius 10 --max_hkl2 50
    python3 scripts/predict_k_theory.py --empirical '{"50": [35, 50]}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.crystal.miller_utils import FAMILIES  # noqa: E402


def _parse_family_hkl(name: str) -> tuple[int, int, int]:
    """Парсит строковое имя '{hkl}' в кортеж индексов (h, k, l)."""
    inner = name.strip().strip("{}")
    digits = [int(c) for c in inner if c.isdigit()]
    if len(digits) != 3:
        raise ValueError(f"Cannot parse Miller family name: {name!r}")
    return tuple(digits)  # type: ignore[return-value]


def face_radius(hkl: tuple[int, int, int], r: float) -> float:
    """Средний радиус грани (hkl) на полусфере радиуса r. Формула (5)."""
    h, k, l = hkl
    hkl2 = h * h + k * k + l * l
    if hkl2 == 0:
        return 0.0
    return (2.0 / 3.0) * math.sqrt(3.0 * r / hkl2)


def build_family_table(radii: Iterable[float]) -> pd.DataFrame:
    """Таблица: семейство × радиус эмиттера -> g(hkl, R)."""
    radii = list(radii)
    rows = []
    for name in FAMILIES.keys():
        hkl = _parse_family_hkl(name)
        hkl2 = sum(v * v for v in hkl)
        row = {"family": name, "|hkl|^2": hkl2}
        for r in radii:
            row[f"g(R={int(r)})"] = round(face_radius(hkl, r), 3)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("|hkl|^2").reset_index(drop=True)
    return df


def count_visible_families(
    r: float,
    thresholds: Iterable[float],
) -> dict[float, int]:
    """Для радиуса r возвращает число семейств с g >= threshold."""
    counts: dict[float, int] = {}
    g_values = [face_radius(_parse_family_hkl(n), r) for n in FAMILIES.keys()]
    for t in thresholds:
        counts[t] = int(sum(1 for g in g_values if g >= t))
    return counts


def predict_k_theory(
    radii: Iterable[float],
    window_radius: float,
) -> pd.DataFrame:
    """Предсказывает k_theory(R) при нескольких порогах различимости.

    Пороги соответствуют разной жёсткости:
      - strict: g >= window_radius (грань целиком умещается в патч)
      - medium: g >= window_radius / 3 (треть окна)
      - soft:   g >= 1 (минимум ~10 поверхностных атомов; в параметрах решётки)
      - very_soft: g >= 0.5 (длинный хвост семейств)
    """
    radii = list(radii)
    thresholds = {
        "strict (g >= window)": float(window_radius),
        "medium (g >= window/3)": float(window_radius) / 3.0,
        "soft (g >= 1 lattice param)": 1.0,
        "very_soft (g >= 0.5)": 0.5,
    }
    rows = []
    for r in radii:
        row = {"R (lattice params)": r}
        counts = count_visible_families(r, thresholds.values())
        for label, t in thresholds.items():
            row[label] = counts[t]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_k_theory(
    theory_df: pd.DataFrame,
    empirical: dict[float, tuple[float, float]] | None,
    output_path: Path,
) -> None:
    """График k_theory(R) для всех порогов + эмпирические точки (если есть)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    radii = theory_df["R (lattice params)"].values
    palette = {
        "strict (g >= window)": "#E53935",
        "medium (g >= window/3)": "#FB8C00",
        "soft (g >= 1 lattice param)": "#1E88E5",
        "very_soft (g >= 0.5)": "#8E24AA",
    }
    for label, color in palette.items():
        ax.plot(radii, theory_df[label], "o-", color=color, linewidth=2, label=label)

    if empirical:
        for r, (lo, hi) in empirical.items():
            ax.fill_between([r - 2, r + 2], [lo, lo], [hi, hi],
                            color="#4CAF50", alpha=0.3)
            ax.plot([r], [(lo + hi) / 2], "D", color="#2E7D32",
                    markersize=10, label=f"empirical near-best (R={int(r)})")

    ax.set_xlabel("Radius R (lattice parameters)", fontsize=12)
    ax.set_ylabel("Number of distinguishable Miller families", fontsize=12)
    ax.set_title(
        "Theoretical k(R) from Nikiforov 2009 formula (5)\n"
        f"Based on {len(FAMILIES)} families from miller_utils.py",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Theoretical k(R) prediction from Nikiforov 2009 formula (5)",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[50, 100, 200, 316, 500, 1000],
        help="Hemisphere radii in lattice parameters",
    )
    parser.add_argument(
        "--window_radius",
        type=float,
        default=10.0,
        help="Projection window radius (lattice parameters); "
             "matches patch_generator.py default",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "crystal" / "analysis"),
    )
    parser.add_argument(
        "--empirical",
        type=str,
        default='{"50": [35, 50]}',
        help="JSON map of radius -> [k_lo, k_hi] empirical near-best range",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip matplotlib plot generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Miller families in miller_utils.py: {len(FAMILIES)}")
    print(f"Radii under analysis: {args.radii}")
    print(f"Window radius (lattice params): {args.window_radius}")
    print()

    family_df = build_family_table(args.radii)
    family_path = output_dir / "k_theory_families.csv"
    family_df.to_csv(family_path, index=False)
    print("--- Face radius g(hkl, R) per family ---")
    print(family_df.to_string(index=False))
    print(f"\nSaved: {family_path}\n")

    theory_df = predict_k_theory(args.radii, args.window_radius)
    theory_path = output_dir / "k_theory_prediction.csv"
    theory_df.to_csv(theory_path, index=False)
    print("--- k_theory(R) at several thresholds ---")
    print(theory_df.to_string(index=False))
    print(f"\nSaved: {theory_path}")

    # Warning: если soft threshold достиг потолка len(FAMILIES), нужно
    # расширить FAMILIES для корректной оценки на больших R.
    soft_col = "soft (g >= 1 lattice param)"
    if (theory_df[soft_col] == len(FAMILIES)).any():
        saturated_radii = theory_df.loc[theory_df[soft_col] == len(FAMILIES),
                                        "R (lattice params)"].tolist()
        print(
            f"\nWARNING: soft threshold reached FAMILIES ceiling ({len(FAMILIES)}) "
            f"at R in {saturated_radii}. "
            "Extend miller_utils.FAMILIES to |hkl|^2 > 35 for accurate estimation "
            "on large R (otherwise k_theory_soft is artificially capped)."
        )

    try:
        empirical_raw = json.loads(args.empirical) if args.empirical else {}
        empirical = {float(r): (float(lo), float(hi))
                     for r, (lo, hi) in empirical_raw.items()}
    except (ValueError, TypeError) as e:
        print(f"WARNING: could not parse --empirical ({e}); plotting without overlay")
        empirical = {}

    if not args.no_plot:
        plot_path = output_dir / "k_theory_vs_empirical.png"
        plot_k_theory(theory_df, empirical, plot_path)
        print(f"Saved plot: {plot_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(
        "  soft threshold (g >= 1 lattice param): expected k_theory is most\n"
        "  directly comparable to empirical k_best from SimCLR clustering.\n"
        "  If k_empirical matches 'soft' column within +/-20% across R, H_phys\n"
        "  is supported (model captures Miller face geometry).\n"
        "  If k_empirical stays constant with R, H_inv holds (model invariant\n"
        "  to hemisphere scale; contradicts Nikiforov 2009 formula (5))."
    )


if __name__ == "__main__":
    main()
