import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_embeddings(embeddings_path, metadata_path, output_dir):
    """Вычисляет базовые метрики качества кластеризации эмбеддингов."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for evaluation...")
    try:
        embeddings = np.load(embeddings_path)
        names_df = pd.read_csv(Path(embeddings_path).parent / "embedding_names.csv")
    except FileNotFoundError:
        print("Embeddings not found. Run feature extraction first.")
        return
        
    meta_df = pd.read_csv(metadata_path)
    meta_df = meta_df.drop_duplicates(subset=['tile_name'])
    
    df = names_df.merge(meta_df, on='tile_name', how='inner')
    
    # Для базовой метрики используем исходное изображение как ground truth класс (текстуру)
    def extract_group(img_name):
        return str(img_name).split('__')[0]
        
    df['group'] = df['source_image'].apply(extract_group)
    labels = df['group'].astype('category').cat.codes.values
    
    if len(np.unique(labels)) < 2:
        print("Not enough diverse groups for clustering metrics.")
        return
        
    print(f"Calculating metrics for {embeddings.shape[0]} samples with {len(np.unique(labels))} groups...")
    
    # 1. Silhouette Score: от -1 (плохо) до 1 (отлично). Учитывает внутрикластерное и межкластерное расстояние.
    sil_score = silhouette_score(embeddings, labels, metric='cosine')
    
    # 2. Calinski-Harabasz Index (Variance Ratio Criterion): чем выше, тем лучше.
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    # 3. Davies-Bouldin Index: чем ниже, тем лучше (означает лучшую сепарацию).
    db_score = davies_bouldin_score(embeddings, labels)
    
    report = f"""
Clustering Evaluation Report (Base ResNet50 Features)
=====================================================
Number of samples: {len(embeddings)}
Number of ground truth groups (textures): {len(np.unique(labels))}

Metrics:
--------
1. Silhouette Score (Cosine): {sil_score:.4f}  (Higher is better, max 1.0)
2. Calinski-Harabasz Index:   {ch_score:.4f}  (Higher is better)
3. Davies-Bouldin Index:      {db_score:.4f}  (Lower is better)

Note: 
These metrics evaluate how well the embeddings naturally group by their source image 'texture' group, 
without explicitly tuning for scale-invariance yet. This serves as our baseline.
"""
    print(report)
    
    report_file = output_path / "evaluation_report_baseline.txt"
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"Report saved to {report_file}")
    
if __name__ == "__main__":
    embeddings_file = r"c:\projects\diploma\data\embeddings\resnet50_embeddings.npy"
    metadata_file = r"c:\projects\diploma\data\processed\tiles_metadata.csv"
    output_directory = r"c:\projects\diploma\data\embeddings"
    
    evaluate_embeddings(embeddings_file, metadata_file, output_directory)
