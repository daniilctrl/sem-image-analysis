import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

def visualize_umap(embeddings_path, metadata_path, output_dir):
    """Снижение размерности UMAP и визуализация эмбеддингов."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    # Загрузка эмбеддингов
    embeddings = np.load(embeddings_path)
    # Загрузка имен тайлов (порядок может отличаться от исходного CSV из-за DataLoader,
    # но мы сохранили порядок в embedding_names.csv!)
    names_df = pd.read_csv(Path(embeddings_path).parent / "embedding_names.csv")
    
    # Загрузка исходных метаданных со шкалами/увеличениями
    meta_df = pd.read_csv(metadata_path)
    meta_df = meta_df.drop_duplicates(subset=['tile_name'])
    
    # Объединение (inner join) для фильтрации метаданных по фактически существующим эмбеддингам
    df = names_df.merge(meta_df, on='tile_name', how='inner')
    
    # Обработка масштабов (могут быть пропуски)
    # Попробуем извлечь увеличение из текста метки 'mag', либо парсим 'scale'
    # Для упрощения демо будем использовать просто имя исходного изображения
    # как "категорию", если метаданные TIFF не парсятся идеально
    
    df['mag'] = df['mag'].fillna('Unknown')
    
    # Извлекаем префикс (например, '1-45degree' из '1-45degree__01') как суррогатный класс/увеличение
    def extract_group(img_name):
        return str(img_name).split('__')[0]
        
    df['group'] = df['source_image'].apply(extract_group)
    
    print(f"Running UMAP on {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    df['umap_x'] = embedding_2d[:, 0]
    df['umap_y'] = embedding_2d[:, 1]
    
    print("Plotting UMAP...")
    plt.figure(figsize=(12, 10))
    
    groups = df['group'].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))
    
    for group, color in zip(groups, colors):
        mask = df['group'] == group
        plt.scatter(
            df.loc[mask, 'umap_x'], 
            df.loc[mask, 'umap_y'], 
            c=[color], 
            label=group, 
            alpha=0.6, 
            s=15
        )
    
    plt.title('UMAP Project of ResNet50 Features for SEM Tiles', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    
    plot_path = output_path / "umap_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем 2D координаты для дальнейшего использования
    df.to_csv(output_path / "umap_coordinates.csv", index=False)
    
    print(f"UMAP visualization saved to {plot_path}")
    print(f"UMAP coordinates saved to {output_path / 'umap_coordinates.csv'}")

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    embeddings_file = str(_root / "data" / "embeddings" / "resnet50_embeddings.npy")
    metadata_file = str(_root / "data" / "processed" / "tiles_metadata.csv")
    output_directory = str(_root / "data" / "embeddings")

    visualize_umap(embeddings_file, metadata_file, output_directory)
