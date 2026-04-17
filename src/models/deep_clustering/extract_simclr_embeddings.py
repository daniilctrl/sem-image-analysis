"""
Скрипт для извлечения эмбеддингов из обученной SimCLR/BYOL модели
и сравнения их с Baseline (ImageNet ResNet50) через UMAP и метрики.

Использование:
  # Fine-tuned модель:
  python extract_simclr_embeddings.py --checkpoint path/to/simclr_resnet50_best.pth

  # Baseline (ImageNet ResNet50, без дообучения):
  python extract_simclr_embeddings.py --checkpoint IMAGENET --output_dir data/embeddings/baseline
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.models.deep_clustering.model import SimCLR
from src.models.deep_clustering.model_byol import BYOL
from src.utils.repro import set_global_seed


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['tile_name']
        img_path = self.data_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def extract_embeddings(model, dataloader, device, model_type="simclr"):
    """Извлечение эмбеддингов (h - выход энкодера, 2048-мерные)"""
    model.eval()
    all_features = []
    all_names = []

    with torch.no_grad():
        for images, names in tqdm(dataloader, desc=f"Extracting {model_type.upper()} embeddings"):
            images = images.to(device)
            if model_type == "byol":
                h = model.online_encoder(images)
                h = nn.functional.normalize(h, dim=1)
            elif model_type == "baseline":
                h = model(images)
                h = nn.functional.normalize(h, dim=1)
            else:
                h, _ = model(images)
            all_features.append(h.cpu().numpy())
            all_names.extend(names)

            del images
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return np.vstack(all_features), all_names


def _build_baseline_model():
    """ImageNet ResNet50 без финальной FC-слой — выход 2048-dim."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Убираем fc, заменяем на Identity → выход avgpool = (N, 2048)
    backbone.fc = nn.Identity()
    return backbone


def compute_metrics(embeddings, labels, label_name="SimCLR"):
    """Вычислим метрики кластеризации"""
    sil = silhouette_score(embeddings, labels, metric='cosine')
    ch = calinski_harabasz_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    print(f"\n--- {label_name} Clustering Metrics ---")
    print(f"  Silhouette Score (Cosine): {sil:.4f}")
    print(f"  Calinski-Harabasz Index:   {ch:.4f}")
    print(f"  Davies-Bouldin Index:      {db:.4f}")
    return sil, ch, db


def build_comparison_umap(baseline_emb, simclr_emb, labels, output_path):
    """Сравнительная UMAP визуализация Baseline vs SimCLR"""
    print("\nBuilding UMAP for Baseline...")
    reducer1 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_baseline = reducer1.fit_transform(baseline_emb)

    print("Building UMAP for SimCLR...")
    reducer2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_simclr = reducer2.fit_transform(simclr_emb)

    # Кодируем метки цветами
    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    colors = [label_to_int[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc1 = axes[0].scatter(umap_baseline[:, 0], umap_baseline[:, 1], c=colors, cmap='tab20', s=1, alpha=0.5)
    axes[0].set_title('Baseline (ImageNet ResNet50)', fontsize=14)
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')

    sc2 = axes[1].scatter(umap_simclr[:, 0], umap_simclr[:, 1], c=colors, cmap='tab20', s=1, alpha=0.5)
    axes[1].set_title('SimCLR (Fine-tuned on SEM)', fontsize=14)
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')

    plt.suptitle('Сравнение: Baseline vs SimCLR Embeddings', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison UMAP saved to {output_path}")


def main(args):
    used_seed = set_global_seed(args.seed)
    print(f"[repro] Global seed fixed: {used_seed}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load metadata
    df = pd.read_csv(args.metadata_path)
    df['group'] = df['source_image'].apply(lambda x: str(x).split('__')[0])
    print(f"Loaded {len(df)} tiles across {df['group'].nunique()} groups.")

    # 2. Prepare dataloader (без аугментаций! Обычная нормализация)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TileDataset(df, args.data_dir, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # 3. Load model
    is_baseline = args.checkpoint.upper() == "IMAGENET"

    if is_baseline:
        print("\nUsing Baseline ImageNet ResNet50 (no fine-tuning)")
        model = _build_baseline_model().to(device)
        model_type = "baseline"
    else:
        print(f"\nLoading {args.model_type.upper()} checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Поддержка обоих форматов: полный dict (new) и raw state_dict (legacy)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if args.model_type == "byol":
            model = BYOL(base_model="resnet50").to(device)
            model.load_state_dict(state_dict)
        else:
            # Factory: auto-detect v1/v2 projection head by BN keys in state_dict.
            model = SimCLR.from_state_dict(
                state_dict, base_model="resnet50", out_dim=128,
            ).to(device)
            print(f"  Detected projection head version: {model.head_version}")
        model_type = args.model_type
    print("Model loaded successfully!")

    # 4. Extract embeddings
    finetuned_emb, names = extract_embeddings(model, loader, device, model_type=model_type)

    # 5. Save embeddings
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if is_baseline:
        # Baseline сохраняется в формате, который ожидает eval_utils.py
        np.save(output_path / "resnet50_embeddings.npy", finetuned_emb)
        pd.DataFrame({'tile_name': names}).to_csv(output_path / "embedding_names.csv", index=False)
        print(f"\nSaved baseline embeddings: {finetuned_emb.shape}")
    else:
        np.save(output_path / "finetuned_embeddings.npy", finetuned_emb)
        pd.DataFrame({'tile_name': names}).to_csv(output_path / "finetuned_embedding_names.csv", index=False)
        print(f"\nSaved {model_type} embeddings: {finetuned_emb.shape}")

    # 6. Load baseline embeddings for comparison (only for fine-tuned models)
    if not is_baseline:
        # Ищем baseline в родительской директории (data/embeddings/)
        parent_emb = output_path.parent
        baseline_emb_path = parent_emb / "resnet50_embeddings.npy"
        baseline_names_path = parent_emb / "embedding_names.csv"

        if baseline_emb_path.exists():
            print("\nLoading baseline embeddings for comparison...")
            baseline_emb = np.load(baseline_emb_path)
            baseline_names_df = pd.read_csv(baseline_names_path)

            # Синхронизируем порядок (inner join по именам)
            simclr_df = pd.DataFrame({'tile_name': names, 'idx': range(len(names))})
            baseline_df = baseline_names_df.copy()
            baseline_df['idx'] = range(len(baseline_df))

            merged = simclr_df.merge(baseline_df, on='tile_name', suffixes=('_simclr', '_baseline'))

            sim_idx = merged['idx_simclr'].values
            base_idx = merged['idx_baseline'].values
            simclr_aligned = finetuned_emb[sim_idx]
            baseline_aligned = baseline_emb[base_idx]

            # Получаем метки для общего набора
            tile_to_group = dict(zip(df['tile_name'], df['group']))
            labels = [tile_to_group.get(n, 'unknown') for n in merged['tile_name']]

            # 7. Compute metrics
            compute_metrics(baseline_aligned, labels, label_name="Baseline (ImageNet)")
            compute_metrics(simclr_aligned, labels, label_name=f"{model_type.upper()} (Fine-tuned)")

            # 8. UMAP comparison
            build_comparison_umap(
                baseline_aligned, simclr_aligned, labels,
                output_path / f"baseline_vs_{model_type}_umap.png"
            )
        else:
            print(f"\nBaseline embeddings not found at {baseline_emb_path}. Skipping comparison.")
            print("  Hint: run with --checkpoint IMAGENET first to generate baseline.")
            tile_to_group = dict(zip(df['tile_name'], df['group']))
            labels = [tile_to_group.get(n, 'unknown') for n in names]
            compute_metrics(finetuned_emb, labels, label_name=f"{model_type.upper()} (Fine-tuned)")

    print("\nDone!")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Extract embeddings from trained SimCLR/BYOL or Baseline ImageNet")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint, or 'IMAGENET' for baseline")
    parser.add_argument("--data_dir", type=str, default=str(_root / "data" / "processed"))
    parser.add_argument("--metadata_path", type=str, default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "data" / "embeddings"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="simclr", choices=["simclr", "byol"],
                        help="Type of model: simclr or byol (ignored if --checkpoint IMAGENET)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility")

    args = parser.parse_args()
    main(args)
