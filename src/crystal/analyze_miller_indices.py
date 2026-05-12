import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корень проекта в sys.path при запуске скриптом напрямую
# (`python src/crystal/analyze_miller_indices.py ...`); такой же приём
# используется в соседних скриптах patch_generator.py, optimize_clusters.py,
# retrieve_crystal.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.crystal.miller_utils import assign_miller_labels, FAMILY_NAMES  # noqa: E402

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
    
    # Размер figsize масштабируем по числу строк (кластеров) И столбцов
    # (семейств Миллера), чтобы аннотации в ячейках не выходили за границы.
    n_rows = int(df[cluster_col].nunique())
    crosstab_tmp = pd.crosstab(df[cluster_col], df['miller_family'])
    n_cols = crosstab_tmp.shape[1]
    cell_w, cell_h = 0.55, 0.40  # дюймов на ячейку
    fig_w = max(14.0, n_cols * cell_w + 2.5)
    fig_h = max(8.0, n_rows * cell_h + 2.5)
    annot_fs = 9 if n_rows <= 25 else 8
    label_fs = 14
    tick_fs = 12
    title_fs = 15

    # 1. Степень чистоты кластера (Из чего на N% состоит кластер?)
    crosstab_plot = pd.crosstab(df[cluster_col], df['miller_family'], normalize='index') * 100

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        crosstab_plot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        annot_kws={"size": annot_fs},
        cbar_kws={'label': '% атомов в Кластере'},
    )
    ax.figure.axes[-1].yaxis.label.set_size(label_fs)
    plt.title(
        f"Степень совпадения: ML-кластеры и аналитические индексы Миллера "
        f"(допуск {args.tol}°)",
        fontsize=title_fs,
    )
    plt.ylabel("Идентификатор кластера (KMeans)", fontsize=label_fs)
    plt.xlabel("Синтетическое семейство по Миллеру", fontsize=label_fs)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.tight_layout()
    plt.savefig(output_dir / "miller_crosstab_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Плотность распределения аналитической зоны по кластерам
    crosstab_miller = pd.crosstab(df[cluster_col], df['miller_family'], normalize='columns') * 100
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        crosstab_miller,
        annot=True,
        fmt=".1f",
        cmap="OrRd",
        annot_kws={"size": annot_fs},
        cbar_kws={'label': '% аналитической грани распределено в кластер'},
    )
    ax.figure.axes[-1].yaxis.label.set_size(label_fs)
    plt.title(
        "Распределение эталонных аналитических граней по кластерам",
        fontsize=title_fs,
    )
    plt.ylabel("Идентификатор кластера (KMeans)", fontsize=label_fs)
    plt.xlabel("Синтетическое семейство по Миллеру", fontsize=label_fs)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.tight_layout()
    plt.savefig(output_dir / "miller_composition_heatmap.png", dpi=200, bbox_inches="tight")
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