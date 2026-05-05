"""
PyTorch Dataset для 2D-патчей кристаллической поверхности.

Загружает предварительно сгенерированные .npy патчи (N, 5, H, W)
и возвращает пару аугментированных версий для контрастного обучения.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_split_indices(split_csv: str, split_filter: str | Iterable[str]) -> np.ndarray:
    """Возвращает упорядоченный массив patch_idx, попавших в нужный split.

    Используется для region-holdout: SimCLR обучается только на patches,
    у которых split == 'train', а извлечение эмбеддингов потом идёт на
    всём датасете. Принимает либо строку, либо iterable строк (например
    ['train', 'val']).
    """
    df = pd.read_csv(split_csv)
    if "patch_idx" not in df.columns or "split" not in df.columns:
        raise ValueError(
            f"split CSV must have columns patch_idx + split, got {df.columns.tolist()}"
        )
    wanted = {split_filter} if isinstance(split_filter, str) else set(split_filter)
    mask = df["split"].isin(wanted)
    indices = df.loc[mask, "patch_idx"].to_numpy()
    indices.sort()
    return indices


class CrystalPatchDataset(Dataset):
    """
    Датасет для Contrastive Learning (SimCLR/BYOL) на кристаллических патчах.

    Для каждого патча возвращает ДВЕ его аугментированные версии.
    """

    def __init__(
        self,
        patches_path: str,
        transform=None,
        subset: int = 0,
        seed: int = 42,
        split_csv: str | None = None,
        split_filter: str | Iterable[str] = "train",
    ):
        """
        Args:
            patches_path: путь к .npy файлу с патчами (N, 5, H, W)
            transform: аугментации (callable, принимает torch.Tensor)
            subset: если > 0, берём случайную подвыборку
            seed: для воспроизводимости подвыборки
            split_csv: путь к CSV с колонками `patch_idx,split`. Если задан,
                из patches.npy остаются только индексы с подходящим split.
                Используется для region-holdout: SimCLR обучается только
                на train-секторах полусферы.
            split_filter: значение(я) split, которые оставлять (default 'train')
        """
        # mmap_mode='r': файл читается лениво — только нужные патчи попадают в RAM.
        # Без этого: 149k × 5 × 32 × 32 × float32 ≈ 3 GB RAM загружается целиком.
        self.patches = np.load(patches_path, mmap_mode='r')  # (N, 5, H, W)

        if split_csv is not None:
            keep = _load_split_indices(split_csv, split_filter)
            n_before = len(self.patches)
            # fancy indexing на mmap создаёт обычный массив в RAM —
            # это нормально, train-сектора занимают ~2 GB при 100k патчей.
            self.patches = self.patches[keep]
            print(
                f"CrystalPatchDataset: split_csv={split_csv} filter={split_filter} "
                f"-> kept {len(self.patches)}/{n_before} patches"
            )

        if subset > 0 and subset < len(self.patches):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.patches), size=subset, replace=False)
            self.patches = self.patches[indices]

        self.transform = transform
        print(f"CrystalPatchDataset: {len(self.patches)} patches, shape={self.patches.shape[1:]}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx].copy())  # (5, H, W)
        
        if self.transform is not None:
            view1 = self.transform(patch)
            view2 = self.transform(patch)
        else:
            view1 = patch
            view2 = patch
        
        return view1, view2


class CrystalInferenceDataset(Dataset):
    """Датасет для инференса (без аугментаций, возвращает один вид)."""
    
    def __init__(self, patches_path: str):
        self.patches = np.load(patches_path)
        print(f"CrystalInferenceDataset: {len(self.patches)} patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx].copy())  # (5, H, W)
        return patch, idx
