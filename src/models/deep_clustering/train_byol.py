"""
Скрипт обучения BYOL для СЭМ-изображений.
Использует те же аугментации и датасет, что и SimCLR.

Использование:
  python train_byol.py --epochs 30 --batch_size 64
  python train_byol.py --resume checkpoint.pth --start_epoch 15 --epochs 15
"""
import math
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
from src.models.deep_clustering.augmentations import (
    get_simclr_transforms, add_augmentation_args, make_config_from_args,
)
from src.models.deep_clustering.dataset import ContrastiveLearningDataset
from src.models.deep_clustering.model_byol import BYOL, byol_loss
from src.utils.repro import set_global_seed, seed_worker, make_generator


def _make_tb_writer(log_dir):
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        print("WARNING: tensorboard not installed; no TB logs.")
        return None


@torch.no_grad()
def _compute_val_loss(model, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for view1, view2 in val_loader:
        view1, view2 = view1.to(device), view2.to(device)
        online_p1, online_p2, target_z1, target_z2 = model(view1, view2)
        total += (byol_loss(online_p1, target_z2)
                  + byol_loss(online_p2, target_z1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(args):
    used_seed = set_global_seed(args.seed)
    print(f"[repro] Global seed fixed: {used_seed}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training BYOL on {device}...")

    # 1. Dataset & Dataloader
    df = pd.read_csv(args.metadata_path)
    if args.subset > 0:
        df = df.sample(args.subset, random_state=args.seed)
        print(f"Using subset of {args.subset} images.")

    val_loader = None
    if args.val_frac > 0.0:
        n_val = max(int(len(df) * args.val_frac), args.batch_size)
        df_val = df.sample(n=n_val, random_state=args.seed)
        df_train = df.drop(df_val.index).reset_index(drop=True)
        print(f"Train/val split: {len(df_train)} / {len(df_val)} "
              f"(val_frac={args.val_frac})")
    else:
        df_train = df.reset_index(drop=True)
        df_val = None

    aug_config = make_config_from_args(args)
    transforms = get_simclr_transforms(config=aug_config)
    print(f"Augmentation config: {aug_config.to_dict()}")
    dataset = ContrastiveLearningDataset(df_train, args.data_dir, transforms)

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

    if df_val is not None:
        val_dataset = ContrastiveLearningDataset(df_val, args.data_dir, transforms)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + 1),
        )

    # 2. Model, Optimizer, Scheduler
    model = BYOL(base_model="resnet50").to(device)
    optimizer = optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters()),
        lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 2.5 Resume from checkpoint (полный формат)
    start_epoch = args.start_epoch
    best_loss = float("inf")
    if args.resume:
        print(f"Resuming from: {args.resume}")
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
                f"Restored optimizer + scheduler + online/target weights. "
                f"Continuing from epoch {start_epoch + 1}."
            )
        else:
            model.load_state_dict(checkpoint)
            print(
                "Loaded weights only (legacy state_dict). "
                "Optimizer state lost; first few steps may be noisy."
            )

    # 3. Training Loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tb_writer = _make_tb_writer(args.tb_log_dir)
    if tb_writer is not None:
        print(f"TensorBoard logs: {args.tb_log_dir}")
        tb_writer.add_text("config", str(vars(args)), 0)

    metric_for_best = "val_loss" if val_loader is not None else "avg_loss"
    best_metric = float("inf")

    total_epochs = start_epoch + args.epochs
    steps_per_epoch = len(loader)
    total_steps = max(total_epochs * steps_per_epoch, 1)
    print(f"Training from epoch {start_epoch + 1} to {total_epochs} "
          f"({steps_per_epoch} steps/epoch, {total_steps} total steps). "
          f"Best-checkpoint metric: {metric_for_best}")

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for view1, view2 in progress_bar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # Forward: получаем предсказания Online и проекции Target
            online_p1, online_p2, target_z1, target_z2 = model(view1, view2)

            # BYOL Loss: симметричный (предсказание 1 -> target 2 + 2 -> 1)
            loss = byol_loss(online_p1, target_z2) + byol_loss(online_p2, target_z1)

            loss.backward()

            if tb_writer is not None:
                total_norm = 0.0
                for p in model.online_encoder.parameters():
                    if p.grad is not None:
                        total_norm += float(p.grad.norm(2).item()) ** 2
                tb_writer.add_scalar("train/grad_norm_online_encoder",
                                     total_norm ** 0.5, global_step)

            optimizer.step()

            # EMA tau ramp-up по косинусу: от base -> 1.0 за весь прогон
            # (Grill et al. 2020, BYOL appendix B).
            if args.ema_tau_schedule == "cosine":
                tau = 1.0 - (1.0 - args.ema_tau) * (
                    math.cos(math.pi * global_step / total_steps) + 1.0
                ) / 2.0
            else:
                tau = args.ema_tau
            model.update_target(tau=tau)

            total_loss += loss.item()
            progress_bar.set_postfix({
                "BYOL Loss": f"{loss.item():.4f}",
                "tau": f"{tau:.4f}",
            })

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                tb_writer.add_scalar("train/tau", tau, global_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"],
                                     global_step)
            global_step += 1

        avg_loss = total_loss / len(loader)
        lr = optimizer.param_groups[0]["lr"]

        val_loss = None
        if val_loader is not None:
            val_loss = _compute_val_loss(model, val_loader, device)

        msg = (f"Epoch [{epoch+1}/{total_epochs}] train_loss={avg_loss:.4f} | "
               f"LR={lr:.6f} | tau_final={tau:.4f}")
        if val_loss is not None:
            msg += f" | val_loss={val_loss:.4f}"
        print(msg)

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
            tb_writer.add_scalar("train/lr_epoch", lr, epoch + 1)
            tb_writer.add_scalar("train/tau_epoch", tau, epoch + 1)
            if val_loss is not None:
                tb_writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)

        scheduler.step()

        # Periodic checkpoint (полный формат)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == total_epochs:
            checkpoint_name = f"byol_resnet50_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_metric,
                "avg_loss": avg_loss,
                "val_loss": val_loss,
            }, output_path / checkpoint_name)
            print(f"  Saved checkpoint: {checkpoint_name}")

        current_metric = val_loss if val_loss is not None else avg_loss
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_metric,
                "best_metric_name": metric_for_best,
            }, output_path / "byol_resnet50_best.pth")

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    print(f"BYOL Training Complete! Best {metric_for_best}: {best_metric:.4f}")


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
    parser.add_argument("--ema_tau", type=float, default=0.996,
                        help="Base EMA tau (initial value; with cosine schedule "
                             "ramps up to 1.0)")
    parser.add_argument("--ema_tau_schedule", type=str, default="cosine",
                        choices=["cosine", "constant"],
                        help="tau schedule: 'cosine' (Grill 2020, default) or "
                             "'constant' (legacy)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save periodic checkpoint every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility (torch/numpy/random/cudnn)")
    parser.add_argument("--val_frac", type=float, default=0.0,
                        help="Fraction of dataset for hold-out val BYOL loss (default: 0)")
    parser.add_argument("--tb_log_dir", type=str, default="",
                        help="TensorBoard log dir. Empty = no logging.")

    add_augmentation_args(parser)

    args = parser.parse_args()
    train(args)
