"""
Скрипт обучения BYOL для СЭМ-изображений.
Использует те же аугментации и датасет, что и SimCLR.

Использование:
  python train_byol.py --epochs 30 --batch_size 64
  python train_byol.py --resume checkpoint.pth --start_epoch 15 --epochs 15
"""
import os
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.models.deep_clustering.augmentations import get_simclr_transforms
from src.models.deep_clustering.dataset import ContrastiveLearningDataset
from src.models.deep_clustering.model_byol import BYOL, byol_loss


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training BYOL on {device}...")

    # 1. Dataset & Dataloader
    df = pd.read_csv(args.metadata_path)
    if args.subset > 0:
        df = df.sample(args.subset, random_state=42)
        print(f"Using subset of {args.subset} images.")

    transforms = get_simclr_transforms(input_size=224)
    dataset = ContrastiveLearningDataset(df, args.data_dir, transforms)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # 2. Model & Optimizer
    model = BYOL(base_model="resnet50").to(device)
    optimizer = optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters()),
        lr=args.learning_rate, weight_decay=1e-4
    )

    # 2.5 Resume
    start_epoch = args.start_epoch
    if args.resume:
        print(f"Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded. Continuing from epoch {start_epoch + 1}.")

    # 3. Training Loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    print(f"Training from epoch {start_epoch + 1} to {total_epochs}.")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for view1, view2 in progress_bar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # Forward: получаем предсказания Online и проекции Target
            online_p1, online_p2, target_z1, target_z2 = model(view1, view2)

            # BYOL Loss: симметричный (предсказание 1→target 2 + предсказание 2→target 1)
            loss = byol_loss(online_p1, target_z2) + byol_loss(online_p2, target_z1)

            loss.backward()
            optimizer.step()

            # EMA обновление Target network
            model.update_target(tau=args.ema_tau)

            total_loss += loss.item()
            progress_bar.set_postfix({'BYOL Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{total_epochs}] Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_name = f"byol_resnet50_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), output_path / checkpoint_name)

    print("BYOL Training Complete!")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="BYOL Training for SEM Analysis")
    parser.add_argument("--data_dir", type=str, default=str(_root / "data" / "processed"))
    parser.add_argument("--metadata_path", type=str, default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "models" / "checkpoints_byol"))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ema_tau", type=float, default=0.996)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)

    args = parser.parse_args()
    train(args)
