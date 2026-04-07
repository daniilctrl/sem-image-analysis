"""
Cross-Scale Retrieval Comparison: Baseline vs SimCLR vs BYOL.

For each tile, find K nearest neighbors excluding same-source tiles.
Measure:
  - Precision@K (material): fraction of neighbors from the same material
  - Precision@K (cross-scale): fraction from same material but DIFFERENT magnification
"""
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm

K = 10

def build_faiss_index(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = (embeddings / norms).astype('float32')
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed)
    return index, normed


def run_retrieval(embeddings, df, K=10):
    index, normed = build_faiss_index(embeddings)
    
    df = df.copy()
    df['material'] = df['source_image'].str.split('__').str[0]
    
    # Only keep materials that have >1 unique magnification (for cross-scale test)
    mat_mag = df.dropna(subset=['mag']).groupby('material')['mag'].nunique()
    valid_mats = mat_mag[mat_mag > 1].index
    test_mask = df['material'].isin(valid_mats) & df['mag'].notna()
    test_indices = df[test_mask].index.tolist()
    
    print(f"  Test tiles: {len(test_indices)} across {len(valid_mats)} materials")
    
    precision_material = []
    precision_cross = []
    
    for idx in tqdm(test_indices, desc=f"  Retrieval K={K}", leave=False):
        query = normed[idx].reshape(1, -1)
        q_material = df.loc[idx, 'material']
        q_mag = df.loc[idx, 'mag']
        q_source = df.loc[idx, 'source_image']
        
        sims, ret_ids = index.search(query, K + 50)
        
        mat_hits = 0
        cross_hits = 0
        count = 0
        
        for ret_idx in ret_ids[0]:
            if ret_idx == idx:
                continue
            if df.loc[ret_idx, 'source_image'] == q_source:
                continue
            
            r_material = df.loc[ret_idx, 'source_image'].split('__')[0]
            r_mag = df.loc[ret_idx, 'mag']
            
            if r_material == q_material:
                mat_hits += 1
                if pd.notna(r_mag) and r_mag != q_mag:
                    cross_hits += 1
            
            count += 1
            if count >= K:
                break
        
        if count >= K:
            precision_material.append(mat_hits / K)
            precision_cross.append(cross_hits / K)
    
    return np.mean(precision_material), np.mean(precision_cross)


def main():
    meta = pd.read_csv('data/processed/tiles_metadata.csv')
    print(f"Loaded {len(meta)} tiles\n")
    
    models = {
        'Baseline': np.load('data/embeddings/resnet50_embeddings.npy'),
        'SimCLR':   np.load('data/embeddings/simclr/finetuned_embeddings.npy'),
        'BYOL':     np.load('data/embeddings/byol/finetuned_embeddings.npy'),
    }
    
    results = []
    
    for name, emb in models.items():
        print(f"Running {name}...")
        p_mat, p_cross = run_retrieval(emb, meta, K=K)
        results.append({
            'Model': name,
            f'P@{K} Material': f'{p_mat:.4f}',
            f'P@{K} Cross-Scale': f'{p_cross:.4f}',
        })
        print(f"  -> P@{K} Material={p_mat:.4f}, Cross-Scale={p_cross:.4f}\n")
    
    print("=" * 60)
    print(f"CROSS-SCALE RETRIEVAL (K={K})")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    print("P@K Material:    % of neighbors from the same material")
    print("P@K Cross-Scale: % from same material at DIFFERENT magnification")
    print("                 (higher = better scale invariance)")
    print()


if __name__ == '__main__':
    main()
