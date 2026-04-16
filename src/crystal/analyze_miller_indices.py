import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_symmetry_vectors(indices):
    """Генерация всех симметрично-эквивалентных векторов для кубической системы."""
    h, k, l = indices
    perms = set(itertools.permutations([h, k, l]))
    vecs = []
    for p in perms:
        for signs in itertools.product([1, -1], repeat=3):
            vec = np.array([p[0]*signs[0], p[1]*signs[1], p[2]*signs[2]], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                vecs.append(tuple(vec))
    # Возвращаем уникальные нормализованные векторы
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

def assign_miller_family(x, y, z, tolerance_deg=6.0):
    vec = np.array([x, y, z])
    norm = np.linalg.norm(vec)
    if norm == 0:
        return "Vicinal/Mixed"
    vec = vec / norm
    
    best_family_name = "Vicinal/Mixed"
    min_angle = np.inf
    
    for family_name, ref_vecs in FAMILIES.items():
        # Скалярное произведение с множеством эквивалентных плоскостей семейства
        dots = np.clip(np.dot(ref_vecs, vec), -1.0, 1.0)
        # На полусфере учитываем только внешний угол, 
        # но для общности берем косинус от абсолютного модуля (эквивалентные встречные плоскости)
        angles = np.arccos(np.abs(dots)) * (180.0 / np.pi) 
        
        family_min_angle = np.min(angles)
        if family_min_angle < min_angle:
            min_angle = family_min_angle
            best_family_name = family_name
            
    if min_angle <= tolerance_deg:
        return best_family_name
    else:
        return "Vicinal/Mixed"

def main(args):
    metadata_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка данных из {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Вычисление индексов Миллера для каждого атома (допуск {args.tol}°)...")
    # Параллельное применение лучше через numpy, но pandas с векторами быстрее через apply?
    # Ускорим: преобразуем X,Y,Z в массивы
    xyz = df[['X', 'Y', 'Z']].values
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    valid_mask = norms.flatten() > 0
    unit_vecs = np.zeros_like(xyz)
    unit_vecs[valid_mask] = xyz[valid_mask] / norms[valid_mask]
    
    # Эксклюзивная классификация: каждому атому — только ближайшая грань
    # (исправлено: ранее атом мог быть перезаписан последним подходящим семейством)
    results = np.full(len(df), "Vicinal/Mixed", dtype=object)
    best_angles = np.full(len(df), np.inf)

    for family_name, ref_vecs in FAMILIES.items():
        dots = np.abs(np.dot(unit_vecs, ref_vecs.T))  # shape (V, N)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0)) * (180.0 / np.pi)

        # Обновляем только если этот угол меньше предыдущего лучшего И в пределах допуска
        better_mask = (angles < best_angles) & (angles <= args.tol)
        results[better_mask] = family_name
        best_angles[better_mask] = angles[better_mask]

    df['miller_family'] = results
    df['miller_angle_deg'] = best_angles  # сохраняем для диагностики
    
    # Кросс-табуляция (Матрица совпадений)
    cluster_col = f"cluster_{args.n_clusters}"
    if cluster_col not in df.columns:
        print(f"Ошибка: столбец {cluster_col} не найден в {metadata_path}")
        return
        
    crosstab = pd.crosstab(df[cluster_col], df['miller_family'])
    crosstab['Всего в кластере'] = crosstab.sum(axis=1)
    
    print("\n--- Матрица совпадений (абс. кол-во атомов) ---")
    print(crosstab.to_string())
    
    crosstab.to_csv(output_dir / "miller_classification_crosstab.csv")
    
    # 1. Степень чистоты кластера (Из чего на N% состоит кластер?)
    crosstab_plot = pd.crosstab(df[cluster_col], df['miller_family'], normalize='index') * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab_plot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% атомов в Кластере'})
    plt.title(f"Степень совпадения: ML-кластеры vs Аналитические индексы (допуск {args.tol}°)")
    plt.ylabel(f"Идентификатор кластера (KMeans)")
    plt.xlabel("Синтетическое семейство по Миллеру")
    plt.tight_layout()
    plt.savefig(output_dir / "miller_crosstab_heatmap.png", dpi=150)
    plt.close()
    
    # 2. Плотность распределения аналитической зоны по кластерам (Где лежат атомы из зоны {110}?)
    crosstab_miller = pd.crosstab(df[cluster_col], df['miller_family'], normalize='columns') * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab_miller, annot=True, fmt=".1f", cmap="OrRd", cbar_kws={'label': '% аналитической грани распределено в кластер'})
    plt.title(f"Распределение эталонных аналитических граней по кластерам")
    plt.ylabel(f"Идентификатор кластера (KMeans)")
    plt.xlabel("Синтетическое семейство по Миллеру")
    plt.tight_layout()
    plt.savefig(output_dir / "miller_composition_heatmap.png", dpi=150)
    plt.close()
    
    print(f"\nГотово. Матрицы сохранены в {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "crystal" / "embeddings"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "crystal" / "analysis"),
    )
    parser.add_argument("--n_clusters", type=int, default=8, help="Идентификатор номера кластеризации")
    parser.add_argument("--tol", type=float, default=6.0, help="Допуск отклонения угла от идеальной плоскости (градусы)")
    args = parser.parse_args()
    main(args)