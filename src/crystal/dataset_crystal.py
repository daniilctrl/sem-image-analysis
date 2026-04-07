"""
PyTorch Dataset для 2D-патчей кристаллической поверхности.

Загружает предварительно сгенерированные .npy патчи (N, 5, H, W)
и возвращает пару аугментированных версий для контрастного обучения.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class CrystalPatchDataset(Dataset):
    """
    Датасет для Contrastive Learning (SimCLR/BYOL) на кристаллических патчах.
    
    Для каждого патча возвращает ДВЕ его аугментированные версии.
    """
    
    def __init__(self, patches_path: str, transform=None, subset: int = 0, seed: int = 42):
        """
        Args:
            patches_path: путь к .npy файлу с патчами (N, 5, H, W)
            transform: аугментации (callable, принимает torch.Tensor)
            subset: если > 0, берём случайную подвыборку
            seed: для воспроизводимости подвыборки
        """
        self.patches = np.load(patches_path)  # (N, 5, H, W)
        
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
