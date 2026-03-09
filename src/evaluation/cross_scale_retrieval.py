"""
Cross-Scale Retrieval Test — главный эксперимент диплома.

Для каждого тайла ищем K ближайших соседей и проверяем:
1. Precision@K (material) — сколько из K найденных принадлежат тому же материалу?
2. Cross-Scale Precision@K — сколько из K найденных принадлежат тому же материалу, 
   но при ДРУГОМ увеличении? (Это и есть инвариантность к масштабу.)

Запуск:
  python cross_scale_retrieval.py                            # Baseline
  python cross_scale_retrieval.py --emb_file simclr_emb.npy  # SimCLR
"""
import argparse
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def build_faiss_index(embeddings):
    """Строит FAISS косинусный индекс."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (embeddings / norms).astype('float32')
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


def run_retrieval_test(embeddings, df, K=10):
    """
    Проводит Cross-Scale Retrieval тест.
    Возвращает метрики для каждого тайла.
    """
    index, normed = build_faiss_index(embeddings)
    
    # Фильтруем только тайлы с известным увеличением и материалом
    df = df.copy()
    df['material'] = df['source_image'].str.split('__').str[0]
    
    # Оставляем только материалы с >1 увеличением (для cross-scale)
    multi_scale = df.dropna(subset=['mag']).copy()
    mat_mag_counts = multi_scale.groupby('material')['mag'].nunique()
    valid_materials = mat_mag_counts[mat_mag_counts > 1].index
    multi_scale = multi_scale[multi_scale['material'].isin(valid_materials)]
    
    print(f"Cross-scale test: {len(multi_scale)} tiles across {len(valid_materials)} materials")
    print(f"Materials: {sorted(valid_materials.tolist())}")
    
    results = []
    
    for idx in tqdm(multi_scale.index, desc=f"Retrieval (K={K})"):
        query_emb = normed[idx].reshape(1, -1)
        query_material = multi_scale.loc[idx, 'material']
        query_mag = multi_scale.loc[idx, 'mag']
        query_source = multi_scale.loc[idx, 'source_image']
        
        # Ищем K+1 (первый — сам запрос)
        sims, indices = index.search(query_emb, K + 50)  # берём больше, чтобы отфильтровать self
        
        # Убираем самого себя и тайлы из того же source_image (того же снимка)
        retrieved_materials = []
        retrieved_cross_scale = []
        count = 0
        
        for sim, ret_idx in zip(sims[0], indices[0]):
            if ret_idx == idx:
                continue
            ret_source = df.loc[ret_idx, 'source_image']
            # Пропускаем тайлы из того же снимка (они тривиально похожи)
            if ret_source == query_source:
                continue
            
            ret_material = df.loc[ret_idx, 'source_image'].split('__')[0]
            ret_mag = df.loc[ret_idx, 'mag'] if not pd.isna(df.loc[ret_idx, 'mag']) else None
            
            is_same_material = (ret_material == query_material)
            is_cross_scale = is_same_material and ret_mag is not None and ret_mag != query_mag
            
            retrieved_materials.append(is_same_material)
            retrieved_cross_scale.append(is_cross_scale)
            
            count += 1
            if count >= K:
                break
        
        if len(retrieved_materials) < K:
            continue  # Недостаточно результатов
        
        precision_material = sum(retrieved_materials) / K
        precision_cross_scale = sum(retrieved_cross_scale) / K
        
        results.append({
            'tile_idx': idx,
            'material': query_material,
            'mag': query_mag,
            f'precision@{K}_material': precision_material,
            f'precision@{K}_cross_scale': precision_cross_scale,
        })
    
    return pd.DataFrame(results)


def print_summary(results_df, K, model_name):
    """Выводит сводку результатов."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Cross-Scale Retrieval Results (K={K})")
    print(f"{'='*60}")
    
    pm = f'precision@{K}_material'
    pcs = f'precision@{K}_cross_scale'
    
    print(f"\n  Overall Precision@{K} (same material):     {results_df[pm].mean():.4f}")
    print(f"  Overall Precision@{K} (cross-scale):       {results_df[pcs].mean():.4f}")
    
    print(f"\n  Per-material breakdown:")
    print(f"  {'Material':<25} {'P@K Material':>14} {'P@K CrossScale':>16} {'Tiles':>7}")
    print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*7}")
    
    for material in sorted(results_df['material'].unique()):
        sub = results_df[results_df['material'] == material]
        print(f"  {material:<25} {sub[pm].mean():>14.4f} {sub[pcs].mean():>16.4f} {len(sub):>7}")
    
    return results_df[pm].mean(), results_df[pcs].mean()


def plot_results(results_df, K, output_path, model_name):
    """Строит графики результатов."""
    pm = f'precision@{K}_material'
    pcs = f'precision@{K}_cross_scale'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Precision по материалам
    per_mat = results_df.groupby('material')[[pm, pcs]].mean().sort_values(pm, ascending=True)
    
    y_pos = range(len(per_mat))
    axes[0].barh(y_pos, per_mat[pm], color='#4a90d9', alpha=0.8, label=f'Precision@{K} (same material)')
    axes[0].barh(y_pos, per_mat[pcs], color='#e74c3c', alpha=0.8, label=f'Precision@{K} (cross-scale)')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(per_mat.index, fontsize=9)
    axes[0].set_xlabel(f'Precision@{K}')
    axes[0].set_title(f'{model_name}: Precision по материалам', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_xlim(0, 1)
    
    # 2. Precision vs увеличение
    axes[1].scatter(results_df['mag'], results_df[pm], s=3, alpha=0.3, c='#4a90d9', label='Same material')
    axes[1].scatter(results_df['mag'], results_df[pcs], s=3, alpha=0.3, c='#e74c3c', label='Cross-scale')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Увеличение (magnification)', fontsize=11)
    axes[1].set_ylabel(f'Precision@{K}', fontsize=11)
    axes[1].set_title(f'{model_name}: Precision vs Увеличение', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main(args):
    # 1. Загрузка данных
    emb_path = Path(args.emb_dir) / args.emb_file
    names_path = Path(args.emb_dir) / args.names_file
    meta_path = Path(args.meta_path)
    
    print(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    names_df = pd.read_csv(names_path)
    meta_df = pd.read_csv(meta_path)
    
    # Объединяем
    meta_df = meta_df.drop_duplicates(subset=['tile_name'])
    df = names_df.merge(meta_df, on='tile_name', how='inner')
    
    assert len(df) == len(embeddings), f"Mismatch: {len(df)} names vs {len(embeddings)} embeddings"
    print(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D)")
    
    # 2. Запуск теста
    results = run_retrieval_test(embeddings, df, K=args.K)
    
    # 3. Вывод результатов
    avg_pm, avg_pcs = print_summary(results, args.K, args.model_name)
    
    # 4. Сохранение
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_dir / f"cross_scale_results_{args.model_name.lower().replace(' ', '_')}.csv", index=False)
    plot_results(results, args.K, output_dir / f"cross_scale_plot_{args.model_name.lower().replace(' ', '_')}.png", args.model_name)
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Scale Retrieval Test")
    parser.add_argument("--emb_dir", type=str, default=r"c:\projects\diploma\data\embeddings")
    parser.add_argument("--emb_file", type=str, default="resnet50_embeddings.npy")
    parser.add_argument("--names_file", type=str, default="embedding_names.csv")
    parser.add_argument("--meta_path", type=str, default=r"c:\projects\diploma\data\processed\tiles_metadata.csv")
    parser.add_argument("--output_dir", type=str, default=r"c:\projects\diploma\data\results")
    parser.add_argument("--model_name", type=str, default="Baseline")
    parser.add_argument("--K", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
