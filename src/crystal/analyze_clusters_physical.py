"""
Анализ физического смысла кластеров, полученных из SimCLR.
Для учебной практики: вычисляем среднее количество соседей (n1-n5) для каждого кластера
и визуализируем примеры локальных патчей, чтобы связать ML-результат с физической моделью поверхности (TSK).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(args):
    metadata_path = Path(args.embeddings_dir) / "embeddings_metadata.csv"
    patches_path = Path(args.patch_dir) / "patches.npy"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка метаданных: {metadata_path}")
    
    if not metadata_path.exists():
        print(f"Файл {metadata_path} не найден. Сначала запустите скрипт кластеризации.")
        return
        
    df = pd.read_csv(metadata_path)
    
    # 1. Анализ средних значений n1..n5
    print("\n--- Физический состав кластеров (средние значения) ---")
    cols_to_avg = ['n1', 'n2', 'n3', 'n4', 'n5', 'grayscale']
    
    # Берем кластеризацию с n_clusters (по умолчанию 8)
    cluster_col = f"cluster_{args.n_clusters}"
    if cluster_col not in df.columns:
        print(f"Ошибка: столбец {cluster_col} не найден в {metadata_path}")
        return
        
    stats = df.groupby(cluster_col)[cols_to_avg].mean().round(2)
    counts = df.groupby(cluster_col).size().rename("count")
    stats = pd.concat([counts, stats], axis=1)
    
    # Сортируем по n1 от большего к меньшему, чтобы на графике было нагляднее 
    # (от плоских террас к изломам/адатомам)
    stats = stats.sort_values(by="n1", ascending=False)
    
    print(stats.to_string())
    
    stats.to_csv(output_dir / "cluster_physical_stats.csv")
    print(f"\nСтатистика сохранена в: {output_dir / 'cluster_physical_stats.csv'}")
    
    # 2. Визуализация средних значений (Столбчатая диаграмма отклонений)
    # Вычисляем разницу от глобального среднего по всей сфере
    global_means = df[cols_to_avg].mean()
    stats_diff = stats[cols_to_avg] - global_means
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(stats_diff))
    width = 0.15
    
    for i, col in enumerate(['n1', 'n2', 'n3', 'n4', 'n5']):
        plt.bar(x + i*width, stats_diff[col], width, label=col)
        
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel(f"ID Кластера (отсортировано по убыванию координации n1)", fontsize=12)
    plt.ylabel("Отклонение от среднего по сфере (атомов)", fontsize=12)
    plt.title(f"Специфика кластеров: отклонение координации от среднего ({args.n_clusters} кластеров)", fontsize=14)
    # Метки по оси X выставим посередине групп столбцов
    plt.xticks(x + width*2, stats_diff.index)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_neighbors_bar.png", dpi=150)
    plt.close()
    
    # 3. Визуализация примеров патчей для каждого кластера
    if not patches_path.exists():
        print(f"Патчи по пути {patches_path} не найдены. Пропускаю визуализацию примеров.")
        return
        
    print(f"\nЗагрузка патчей для визуализации примеров из: {patches_path}")
    # Используем mmap_mode="r", чтобы не загружать весь файл в память 
    patches = np.load(patches_path, mmap_mode="r")
    
    n_examples = 6
    clusters = stats.index.tolist()
    
    fig, axes = plt.subplots(len(clusters), n_examples, figsize=(n_examples*2.2, len(clusters)*2.2))
    if len(clusters) == 1:
        axes = [axes]
        
    for i, cluster_id in enumerate(clusters):
        cluster_indices = df[df[cluster_col] == cluster_id].index.values
        # Выбираем случайные примеры (патчи) для данного кластера
        np.random.seed(42)  # чтобы примеры были детерминированы
        sampled_idxs = np.random.choice(cluster_indices, min(n_examples, len(cluster_indices)), replace=False)
        
        for j, p_idx in enumerate(sampled_idxs):
            patch = patches[p_idx] # форма патча: (C, H, W)
            # Рисуем изображение первого канала (n1 - ближайшие соседи)
            img = patch[0]
            ax = axes[i][j]
            im = ax.imshow(img, cmap="viridis", vmin=0, vmax=1) # отнормировано было от 0 до 1 
            
            if j == 0:
                ax.set_ylabel(f"Кластер {cluster_id}\n(Размер: {counts[cluster_id]})", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.suptitle("Примеры локальных патчей (канал n1) для физической интерпретации кластеров", fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Добавляем общий колорбар справа
    fig.subplots_adjust(right=0.92, top=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Нормированное число соседей n1')
    
    plt.savefig(output_dir / "cluster_patch_examples.png", dpi=200)
    plt.close()
    print(f"Примеры патчей успешно сохранены в: {output_dir / 'cluster_patch_examples.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подтверждение физического смысла SimCLR кластеров")
    parser.add_argument("--embeddings_dir", type=str, default=r"c:\projects\diploma\data\crystal\embeddings")
    parser.add_argument("--patch_dir", type=str, default=r"c:\projects\diploma\data\crystal\patches")
    parser.add_argument("--output_dir", type=str, default=r"c:\projects\diploma\data\crystal\analysis")
    parser.add_argument("--n_clusters", type=int, default=8, help="Номер кластеризации (какую колонку брать из CSV)")
    
    args = parser.parse_args()
    main(args)
