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
from src.models.deep_clustering.model import SimCLR
from src.models.deep_clustering.loss import NTXentLoss
from src.utils.repro import set_global_seed, seed_worker, make_generator


def train(args):
    used_seed = set_global_seed(args.seed)
    print(f"[repro] Global seed fixed: {used_seed}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training SimCLR on {device}...")
    
    # 1. Dataset & Dataloader
    df = pd.read_csv(args.metadata_path)
    if args.subset > 0:
        df = df.sample(args.subset, random_state=42)
        print(f"Using a subset of {args.subset} images for fast testing.")
        
    transforms = get_simclr_transforms(input_size=224)
    dataset = ContrastiveLearningDataset(df, args.data_dir, transforms)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    
    # 2. Model, Loss, Optimizer, Scheduler
    model = SimCLR(base_model="resnet50", out_dim=128).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Cosine annealing — согласовано с Crystal-веткой. Стандартная схема
    # для contrastive learning с Adam на 30+ эпохах.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 2.5 Resume from checkpoint (полный формат: модель + optimizer + scheduler).
    # Legacy чекпоинты (только state_dict) загружаются с явным предупреждением.
    start_epoch = args.start_epoch
    best_loss = float("inf")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", args.start_epoch) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(
                f"Restored optimizer + scheduler state. "
                f"Continuing from epoch {start_epoch + 1}."
            )
        else:
            # Legacy: только веса. Оптимизатор и scheduler стартуют с нуля —
            # это означает, что первые ~N шагов Adam восстанавливает моменты.
            model.load_state_dict(checkpoint)
            print(
                "Loaded weights only (legacy state_dict). "
                "Optimizer state lost; first few steps may be noisy."
            )

    # 3. Training Loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    print(f"Training from epoch {start_epoch + 1} to {total_epochs}.")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch_idx, (view1, view2) in enumerate(progress_bar):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # _, z1 — при contrastive обучении используется только z (projection);
            # h (base embedding) игнорируется на forward pass.
            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = criterion(z1, z2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'NT-Xent Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{total_epochs}] Average Loss: {avg_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

        scheduler.step()

        # Periodic checkpoint (полный формат)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == total_epochs:
            ckpt_name = f"simclr_resnet50_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "avg_loss": avg_loss,
            }, output_path / ckpt_name)
            print(f"  Saved checkpoint: {ckpt_name}")

        # Best checkpoint по avg loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }, output_path / "simclr_resnet50_best.pth")

    print(f"Training Complete! Best avg loss: {best_loss:.4f}")

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="SimCLR Training pipeline for SEM Analysis")
    parser.add_argument("--data_dir", type=str, default=str(_root / "data" / "processed"))
    parser.add_argument("--metadata_path", type=str, default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "models" / "checkpoints"))
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10) # В реальности нужно 50-100+
    parser.add_argument("--batch_size", type=int, default=32) # На Colab T4 можно 64-128
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0) # 0 для Win, 2-4 для Colab
    parser.add_argument("--subset", type=int, default=0) # Разрешить брать срез (например 1000 шт для тестов)
    parser.add_argument("--resume", type=str, default="") # Путь к .pth чекпоинту для продолжения обучения
    parser.add_argument("--start_epoch", type=int, default=0) # С какой эпохи продолжать нумерацию
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save periodic checkpoint every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility (torch/numpy/random/cudnn)")

    args = parser.parse_args()
    train(args)
