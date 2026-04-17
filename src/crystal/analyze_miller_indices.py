import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.crystal.miller_utils import assign_miller_labels, FAMILY_NAMES

def main(args):
    metadata_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка данных из {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Вычисление индексов Миллера для каждого атома (допуск {args.tol}°)...")
    xyz = df[['X', 'Y', 'Z']].values
    miller_labels, family_names = assign_miller_labels(xyz, tol_deg=args.tol)

    # Преобразуем целочисленные метки в строковые названия семейств
    df['miller_family'] = [family_names[i] for i in miller_labels]

    for i, name in enumerate(family_names):
        count = (miller_labels == i).sum()
        if count > 0:
            print(f"  {name}: {count} ({100 * count / len(df):.1f}%)")
    
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