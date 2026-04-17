"""
Обучение SimCLR на кристаллических патчах.

Аналог train.py для SEM-пайплайна, но адаптированный:
  - Загрузка .npy патчей вместо PIL-изображений
  - 5-канальный вход (ResNet18)
  - Кристаллоспецифичные аугментации

Использование:
  python src/crystal/train_crystal.py --patch_dir data/crystal/patches --epochs 50
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.crystal.dataset_crystal import CrystalPatchDataset
from src.crystal.augmentations_crystal import get_crystal_transforms
from src.crystal.model_crystal import CrystalSimCLR
from src.models.deep_clustering.loss import NTXentLoss
from src.utils.repro import set_global_seed, seed_worker, make_generator


def train(args):
    used_seed = set_global_seed(args.seed)
    print(f"[repro] Global seed fixed: {used_seed}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training CrystalSimCLR on {device}...")

    # 1. Dataset & DataLoader
    transforms = get_crystal_transforms(strong=True)
    patches_path = Path(args.patch_dir) / "patches.npy"
    dataset = CrystalPatchDataset(
        str(patches_path),
        transform=transforms,
        subset=args.subset,
    )

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

    # 2. Model, Loss, Optimizer
    model = CrystalSimCLR(in_channels=5, out_dim=128).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Scheduler: cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2.5 Resume from checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Новый формат: полный checkpoint с optimizer и scheduler
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', args.start_epoch) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Restored optimizer and scheduler state. Continuing from epoch {start_epoch + 1}.")
        else:
            # Старый формат: только веса модели
            model.load_state_dict(checkpoint)
            print(f"Loaded weights only (old checkpoint format). Continuing from epoch {start_epoch + 1}.")

    # 3. Training Loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    best_loss = float("inf")

    print(f"\nTraining from epoch {start_epoch + 1} to {total_epochs}.")
    print(f"Batches per epoch: {len(loader)}")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        for view1, view2 in progress_bar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"NT-Xent": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch + 1}/{total_epochs}] Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        scheduler.step()

        # Save checkpoint (полный формат: модель + optimizer + scheduler)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == total_epochs:
            ckpt_name = f"crystal_simclr_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'avg_loss': avg_loss,
            }, output_path / ckpt_name)
            print(f"  Saved checkpoint: {ckpt_name}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, output_path / "crystal_simclr_best.pth")

    print(f"\nTraining Complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Crystal SimCLR Training")
    parser.add_argument("--patch_dir", type=str, default=str(_root / "data" / "crystal" / "patches"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "models" / "crystal"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility (torch/numpy/random/cudnn)")
    args = parser.parse_args()
    train(args)
