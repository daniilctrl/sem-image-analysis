"""
Извлечение эмбеддингов из обученной CrystalSimCLR модели.

Аналог extract_simclr_embeddings.py для SEM-пайплайна.
Сохраняет 512-мерные эмбеддинги + вычисляет метрики кластеризации.

Использование:
  python src/crystal/extract_crystal_embeddings.py \
    --checkpoint models/crystal/crystal_simclr_best.pth \
    --patch_dir data/crystal/patches
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.crystal.dataset_crystal import CrystalInferenceDataset
from src.crystal.model_crystal import CrystalSimCLR
from src.utils.repro import set_global_seed


def extract_embeddings(model, dataloader, device):
    """Извлечение 512-мерных эмбеддингов (h) из энкодера."""
    model.eval()
    all_features = []
    all_indices = []

    with torch.no_grad():
        for patches, indices in tqdm(dataloader, desc="Extracting embeddings"):
            patches = patches.to(device)
            h, _ = model(patches)
            all_features.append(h.cpu().numpy())
            all_indices.extend(indices.numpy())

            del patches
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.vstack(all_features), np.array(all_indices)


def compute_metrics(embeddings, labels, label_name="CrystalSimCLR"):
    """Вычисление метрик кластеризации."""
    # sample_size обязателен: полный silhouette на 150k × 512D занимает ~10 минут.
    sil = silhouette_score(
        embeddings, labels,
        metric="cosine",
        sample_size=min(10_000, len(embeddings)),
        random_state=42,
    )
    ch = calinski_harabasz_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    print(f"\n--- {label_name} Clustering Metrics ---")
    print(f"  Silhouette Score (Cosine): {sil:.4f}")
    print(f"  Calinski-Harabasz Index:   {ch:.4f}")
    print(f"  Davies-Bouldin Index:      {db:.4f}")
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}


def main(args):
    used_seed = set_global_seed(args.seed)
    print(f"[repro] Global seed fixed: {used_seed}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CrystalSimCLR(in_channels=5, out_dim=128).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', '?')
        loss_info = checkpoint.get('best_loss', checkpoint.get('avg_loss', '?'))
        print(f"Loaded full checkpoint (epoch={epoch_info}, loss={loss_info})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict (legacy format)")
    print("Model loaded successfully!")

    # 2. Load patches
    patches_path = Path(args.patch_dir) / "patches.npy"
    dataset = CrystalInferenceDataset(str(patches_path))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # 3. Extract embeddings
    embeddings, indices = extract_embeddings(model, loader, device)
    print(f"Extracted embeddings: {embeddings.shape}")

    # 4. Save embeddings
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "crystal_embeddings.npy", embeddings)
    np.save(output_path / "crystal_embedding_indices.npy", indices)
    print(f"Saved to {output_path}")

    # 5. KMeans кластеризация + метрики
    meta_path = Path(args.patch_dir) / "patches_metadata.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        print(f"\nLoaded metadata: {len(meta_df)} rows")
        # Валидация: embeddings и metadata должны быть 1-к-1
        if len(embeddings) != len(meta_df):
            raise ValueError(
                f"CRITICAL: embeddings ({len(embeddings)}) != patches_metadata ({len(meta_df)}). "
                f"Regenerate patches or embeddings to fix this mismatch."
            )
    else:
        print(f"\nWARNING: {meta_path} not found, creating stub metadata")
        meta_df = pd.DataFrame({"patch_idx": range(len(embeddings))})

    for n_clusters in args.n_clusters:
        print(f"\n--- KMeans with {n_clusters} clusters ---")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Метрики
        metrics = compute_metrics(embeddings, labels, f"KMeans-{n_clusters}")

        # Сохраняем метки кластеров
        meta_df[f"cluster_{n_clusters}"] = labels
        np.save(output_path / f"cluster_labels_{n_clusters}.npy", labels)

    # Сохраняем обновлённые метаданные
    meta_df.to_csv(output_path / "embeddings_metadata.csv", index=False)
    print(f"\nSaved metadata with cluster labels to {output_path / 'embeddings_metadata.csv'}")

    # Финальная проверка целостности артефактов
    emb_file = output_path / "crystal_embeddings.npy"
    meta_file = output_path / "embeddings_metadata.csv"
    n_emb = np.load(emb_file).shape[0]
    n_meta = len(pd.read_csv(meta_file))
    assert n_emb == n_meta, f"INTEGRITY CHECK FAILED: {emb_file} ({n_emb}) != {meta_file} ({n_meta})"
    print(f"Integrity check passed: {n_emb} embeddings == {n_meta} metadata rows")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract crystal embeddings and cluster")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patch_dir", type=str, default=str(_root / "data" / "crystal" / "patches"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "data" / "crystal" / "embeddings"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, nargs="+", default=[5, 8, 10],
                        help="Number of KMeans clusters to try")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility")
    args = parser.parse_args()
    main(args)
