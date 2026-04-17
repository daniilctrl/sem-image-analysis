"""Общие статистические утилиты (percentile bootstrap CI).

Используется:
  - src/crystal/retrieve_crystal.py (precision@K_miller и др.)
  - src/evaluation/cross_scale_retrieval.py (precision@K_material, cross-scale)
  - src/evaluation/linear_probe.py, knn_probe.py (accuracy, macro-F1)

Раньше bootstrap_metric_ci жил локально в retrieve_crystal.py; при попытке
переиспользовать в SEM-evaluation возникло бы дублирование. Вынос в
src/utils/stats.py — единый источник истины.
"""

from __future__ import annotations

import numpy as np


def bootstrap_metric_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Percentile bootstrap CI для скалярной статистики выборки (Efron 1979).

    Аргументы:
        values: np.ndarray формы (N,) — per-sample значения метрики
                (например, precision@K каждого query-атома).
        n_bootstrap: число bootstrap-итераций (1000 — минимум для 95% CI).
        confidence: уровень доверия (0.95 по умолчанию).
        seed: RNG seed для воспроизводимости.
        statistic: 'mean' | 'median' — какую статистику bootstrap'ить.

    Возвращает:
        (point_estimate, ci_lo, ci_hi).
    """
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))

    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}. Use 'mean' or 'median'.")

    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = stat_fn(values[idx])

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    point = float(stat_fn(values))
    return (point, lo, hi)


def bootstrap_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """CI для разницы средних двух выборок (paired или unpaired).

    Полезно для «значимо ли SimCLR лучше Baseline»: если 0 НЕ входит в CI
    разницы — разница статистически значима на уровне `confidence`.

    Аргументы:
        values_a, values_b: выборки одинаковой или разной длины.
        n_bootstrap: число bootstrap-итераций.
        confidence: уровень доверия.
        seed: RNG seed.

    Возвращает:
        (diff_mean, ci_lo, ci_hi) где diff = mean(a) - mean(b).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx_a = rng.integers(0, len(a), size=len(a))
        idx_b = rng.integers(0, len(b), size=len(b))
        diffs[i] = a[idx_a].mean() - b[idx_b].mean()

    alpha = 1.0 - confidence
    lo = float(np.percentile(diffs, 100.0 * alpha / 2.0))
    hi = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    point = float(a.mean() - b.mean())
    return (point, lo, hi)


def format_mean_ci(mean: float, lo: float, hi: float, decimals: int = 4) -> str:
    """Форматирует (mean, lo, hi) в строку '0.7234 [0.6800, 0.7700]'."""
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
