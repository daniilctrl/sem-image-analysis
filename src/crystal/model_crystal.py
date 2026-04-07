"""
SimCLR модель для кристаллических патчей (5-канальный вход).

Отличия от SEM-версии (`model.py`):
  - Первый Conv2d слой принимает 5 каналов вместо 3
  - Используем ResNet18 (вместо ResNet50) — патчи 32×32 проще натуральных изображений
  - Сохраняем ту же архитектуру Projection Head (MLP)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CrystalSimCLR(nn.Module):
    """
    SimCLR для кристаллических 5-канальных 32×32 патчей.
    
    Архитектура:
      1. ResNet18 (адаптированный под 5 каналов)
      2. Projection Head: Linear(512 → 512) → ReLU → Linear(512 → out_dim)
    """
    
    def __init__(self, in_channels: int = 5, out_dim: int = 128):
        super().__init__()
        
        # Загружаем ResNet18 (без предобученных весов — домен слишком отличается)
        resnet = models.resnet18(weights=None)
        self.enc_dim = resnet.fc.in_features  # 512 для ResNet18
        
        # Заменяем первый Conv2d: 3 → in_channels
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        
        # Для малых 32×32 патчей убираем MaxPool и уменьшаем stride
        # (иначе feature map станет слишком маленьким)
        resnet.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet.maxpool = nn.Identity()
        
        # Удаляем FC-слой
        modules = list(resnet.children())[:-1]  # всё кроме fc
        self.encoder = nn.Sequential(*modules)
        
        # Projection Head (как в SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(self.enc_dim, self.enc_dim),
            nn.ReLU(),
            nn.Linear(self.enc_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 5, 32, 32)
        Returns:
            h: (B, 512) — нормализованный embedding энкодера
            z: (B, out_dim) — нормализованный выход проектора
        """
        h = self.encoder(x)
        h = torch.flatten(h, 1)  # (B, 512)
        
        z = self.projector(h)  # (B, 128)
        
        h = nn.functional.normalize(h, dim=1)
        z = nn.functional.normalize(z, dim=1)
        
        return h, z


if __name__ == "__main__":
    # Быстрый тест формы
    model = CrystalSimCLR(in_channels=5, out_dim=128)
    x = torch.randn(4, 5, 32, 32)
    h, z = model(x)
    print(f"Input:  {x.shape}")
    print(f"h (encoder): {h.shape}")
    print(f"z (projector): {z.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
