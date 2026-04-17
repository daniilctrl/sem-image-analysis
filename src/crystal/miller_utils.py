"""
Единый модуль аналитической классификации по индексам Миллера.

Используется в:
  - optimize_clusters.py (AMI/NMI vs Miller)
  - analyze_miller_indices.py (crosstab + heatmaps)

Ранее логика была продублирована в обоих файлах с небольшими расхождениями.
Теперь — единый источник истины.

Алгоритм:
  1. Для каждого семейства {hkl} генерируем все симметрично-эквивалентные
     единичные вектора (перестановки + знаки, кубическая симметрия).
  2. Для каждого атома вычисляем угол до ближайшего эталонного вектора
     каждого семейства.
  3. Атом назначается ближайшему семейству, если минимальный угол ≤ tolerance.
     Иначе — Vicinal/Mixed.
"""

import itertools
import numpy as np


def get_symmetry_vectors(indices: tuple[int, int, int]) -> np.ndarray:
    """Генерация всех симметрично-эквивалентных единичных векторов для кубической системы.

    Аргументы:
        indices: Кортеж (h, k, l) индексов Миллера.

    Возвращает:
        np.ndarray формы (N, 3) — уникальные нормализованные вектора.
    """
    h, k, l = indices
    perms = set(itertools.permutations([h, k, l]))
    vecs = []
    for p in perms:
        for signs in itertools.product([1, -1], repeat=3):
            vec = np.array([p[0] * signs[0], p[1] * signs[1], p[2] * signs[2]], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vecs.append(tuple(vec / norm))
    return np.unique(vecs, axis=0)


# Полный набор значимых семейств граней для ОЦК-полусферы
# (по рис. 3 из: Никифоров, Егоров, Шен, 2009)
FAMILIES = {
    "{100}": get_symmetry_vectors((1, 0, 0)),
    "{110}": get_symmetry_vectors((1, 1, 0)),
    "{111}": get_symmetry_vectors((1, 1, 1)),
    "{210}": get_symmetry_vectors((2, 1, 0)),
    "{211}": get_symmetry_vectors((2, 1, 1)),
    "{221}": get_symmetry_vectors((2, 2, 1)),
    "{310}": get_symmetry_vectors((3, 1, 0)),
    "{321}": get_symmetry_vectors((3, 2, 1)),
    "{411}": get_symmetry_vectors((4, 1, 1)),
}

FAMILY_NAMES = list(FAMILIES.keys()) + ["Vicinal/Mixed"]
VICINAL_LABEL = len(FAMILIES)  # целочисленная метка для Vicinal/Mixed


def assign_miller_labels(xyz: np.ndarray, tol_deg: float = 6.0) -> tuple[np.ndarray, list[str]]:
    """Векторизованная классификация атомов по семействам индексов Миллера.

    Для каждого атома находится ближайшее семейство {hkl}.
    Если минимальный угол > tol_deg, атом помечается как Vicinal/Mixed.

    Аргументы:
        xyz: np.ndarray формы (N, 3) — координаты атомов.
        tol_deg: Допуск в градусах (по умолчанию 6.0).

    Возвращает:
        labels: np.ndarray формы (N,), dtype=int32 — индексы семейств (0..N_families-1)
                или VICINAL_LABEL для неклассифицированных.
        family_names: list[str] — названия семейств в порядке индексов,
                      последний элемент = "Vicinal/Mixed".
    """
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    valid = norms.flatten() > 0
    unit = np.zeros_like(xyz)
    unit[valid] = xyz[valid] / norms[valid]

    labels = np.full(len(xyz), VICINAL_LABEL, dtype=np.int32)
    best_angles = np.full(len(xyz), np.inf)

    for fi, (fname, ref_vecs) in enumerate(FAMILIES.items()):
        # |cos(angle)| — для эквивалентности встречных плоскостей на полусфере
        dots = np.abs(np.dot(unit, ref_vecs.T))
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0)) * (180.0 / np.pi)

        better = (angles < best_angles) & (angles <= tol_deg)
        labels[better] = fi
        best_angles[better] = angles[better]

    return labels, FAMILY_NAMES.copy()


def assign_miller_family_single(x: float, y: float, z: float,
                                tolerance_deg: float = 6.0) -> str:
    """Классификация одного атома (для совместимости с pandas apply).

    Возвращает строковое название семейства или 'Vicinal/Mixed'.
    """
    vec = np.array([x, y, z])
    norm = np.linalg.norm(vec)
    if norm == 0:
        return "Vicinal/Mixed"
    vec = vec / norm

    best_family_name = "Vicinal/Mixed"
    min_angle = np.inf

    for family_name, ref_vecs in FAMILIES.items():
        dots = np.clip(np.dot(ref_vecs, vec), -1.0, 1.0)
        angles = np.arccos(np.abs(dots)) * (180.0 / np.pi)
        family_min_angle = np.min(angles)
        if family_min_angle < min_angle:
            min_angle = family_min_angle
            best_family_name = family_name

    if min_angle <= tolerance_deg:
        return best_family_name
    else:
        return "Vicinal/Mixed"
