"""
Аугментации для кристаллических патчей.

В отличие от SEM-аугментаций, здесь:
  - НЕ используем color jitter (каналы — физические величины, не оптические)
  - Используем повороты на 90° (поверхность инвариантна к ориентации)
  - Добавляем гауссов шум (моделирование неопределённости в подсчёте соседей)
  - Случайное обрезание + масштабирование (как в SimCLR для scale invariance)
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T


class RandomRotation90:
    """Случайный поворот на 0°, 90°, 180° или 270°."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, dims=[-2, -1])


class RandomFlip:
    """Случайное отражение по горизонтали и/или вертикали."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])  # horizontal
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-2])  # vertical
        return x


class GaussianNoise:
    """Добавление гауссова шума к значениям каналов."""
    def __init__(self, std: float = 0.05):
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


class RandomCropResize:
    """
    Случайное обрезание участка патча и масштабирование до исходного размера.
    Имитирует разные масштабы наблюдения (аналог RandomResizedCrop в SimCLR).
    """
    def __init__(self, scale_min: float = 0.5, scale_max: float = 1.0):
        self.scale_min = scale_min
        self.scale_max = scale_max
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        cropped = x[:, top:top + crop_h, left:left + crop_w]
        
        # Масштабирование обратно к HxW через bilinear interpolation
        resized = F.interpolate(
            cropped.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        return resized


class ChannelDrop:
    """
    Случайное обнуление одного из каналов (dropout по каналам).
    Заставляет модель не зависеть от одного конкретного порядка соседей.
    """
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = x.clone()  # не мутируем оригинальный тензор (важно при mmap)
            ch = torch.randint(0, x.shape[0], (1,)).item()
            x[ch] = 0.0
        return x


def get_crystal_transforms(strong: bool = True):
    """
    Возвращает pipeline аугментаций для кристаллических патчей.
    
    Args:
        strong: если True — полный набор (для обучения),
                если False — только базовые (для тестирования)
    """
    if strong:
        return T.Compose([
            RandomRotation90(),
            RandomFlip(),
            RandomCropResize(scale_min=0.5, scale_max=1.0),
            GaussianNoise(std=0.03),
            ChannelDrop(p=0.1),
        ])
    else:
        return T.Compose([
            RandomRotation90(),
            RandomFlip(),
        ])
