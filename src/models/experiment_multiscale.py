import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap
import matplotlib.cm as cm
import time
import cv2

# Import from our existing modules
import sys
_repo = Path(__file__).resolve().parents[2]
sys.path.append(str(_repo))
from src.models.feature_extraction import TileDataset
from src.models.multiscale_feature_extraction import MultiScaleTileDataset

def run_experiment(data_dir, metadata_path, output_dir, samples_per_group=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Multi-Scale Experiment on {device}...")
    
    # 1. Load and sample metadata
    df = pd.read_csv(metadata_path)
    df = df.drop_duplicates(subset=['tile_name'])
    
    # Group by source image prefix
    df['group'] = df['source_image'].apply(lambda x: str(x).split('__')[0])
    
    # Sample N from each group to keep it balanced and small for CPU
    sampled_df = df.sample(frac=1, random_state=42).groupby('group').head(samples_per_group).reset_index(drop=True)
    
    print(f"Sampled {len(sampled_df)} tiles across {sampled_df['group'].nunique()} groups.")
    
    labels = sampled_df['group'].astype('category').cat.codes.values
    
    # 2. Setup Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules).to(device).eval()
    
    # ====== BASELINE EVALUATION ======
    print("\n--- Running Baseline Formulation ---")
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    base_dataset = TileDataset(sampled_df, data_dir, transform=base_transform)
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=64, shuffle=False)
    
    base_features = []
    t0 = time.time()
    with torch.no_grad():
        for images, _ in base_loader:
            feats = model(images.to(device)).squeeze().cpu().numpy()
            base_features.append(feats)
    base_features = np.vstack(base_features)
    print(f"Baseline extraction took {time.time()-t0:.1f}s")
    
    base_sil = silhouette_score(base_features, labels, metric='cosine')
    print(f"Baseline Silhouette Score (Cosine): {base_sil:.4f}")
    
    # ====== MULTI-SCALE EVALUATION ======
    print("\n--- Running Multi-Scale (Max Pooling) Formulation ---")
    ms_dataset = MultiScaleTileDataset(sampled_df, data_dir)
    ms_loader = torch.utils.data.DataLoader(ms_dataset, batch_size=24, shuffle=False)
    
    ms_features = []
    t0 = time.time()
    with torch.no_grad():
        for multi_batch, _ in ms_loader:
            b = multi_batch.size(0)
            flat_batch = multi_batch.view(-1, 3, 224, 224).to(device)
            feats = model(flat_batch).squeeze().cpu().numpy()
            feats = feats.reshape(b, 3, -1)
            # Max Pooling fusion
            fused = np.max(feats, axis=1)
            ms_features.append(fused)
    ms_features = np.vstack(ms_features)
    print(f"Multi-Scale extraction took {time.time()-t0:.1f}s")
    
    ms_sil = silhouette_score(ms_features, labels, metric='cosine')
    print(f"Multi-Scale Silhouette Score (Cosine): {ms_sil:.4f}")
    
    # ====== VISUALIZATION UMAP COMPARED ======
    output_path = Path(output_dir)
    
    print("\nGenerating comparative UMAP plots...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    
    base_umap = reducer.fit_transform(base_features)
    ms_umap = reducer.fit_transform(ms_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    groups = sampled_df['group'].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))
    
    for grp, color in zip(groups, colors):
        mask = sampled_df['group'] == grp
        ax1.scatter(base_umap[mask, 0], base_umap[mask, 1], c=[color], label=grp, alpha=0.7, s=20)
        ax2.scatter(ms_umap[mask, 0], ms_umap[mask, 1], c=[color], label=grp, alpha=0.7, s=20)
        
    ax1.set_title(f"Baseline (Single Scale)\nSilhouette: {base_sil:.3f}")
    ax2.set_title(f"Max-Pool Fusion (0.7x, 1x, 1.5x)\nSilhouette: {ms_sil:.3f}")
    
    # Only show legend if not too many groups
    if len(groups) <= 15:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plot_path = output_path / "scale_invariance_experiment.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nExperiment finished! Comparison graph saved to {plot_path}")

if __name__ == "__main__":
    data_directory = str(_repo / "data" / "processed")
    metadata_file = str(_repo / "data" / "processed" / "tiles_metadata.csv")
    output_directory = str(_repo / "data" / "embeddings")

    # Use 10 samples per group (roughly 550 total samples since we have 55 groups)
    run_experiment(data_directory, metadata_file, output_directory, samples_per_group=10)
