"""SimCLR модель для кристаллических патчей (5-канальный вход).

Отличия от SEM-версии (`src/models/deep_clustering/model.py`):
  - Первый Conv2d слой принимает 5 каналов вместо 3
  - Используем ResNet18 (вместо ResNet50): патчи 32x32 проще натуральных изображений
  - Используем тот же унифицированный `SimCLRProjectionHead` из `src.models.heads`
    (v2 по умолчанию), что и SEM — так в обеих ветках одна архитектура head
    и честное сравнение SimCLR методов.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.heads import SimCLRProjectionHead  # noqa: E402


class CrystalSimCLR(nn.Module):
    """SimCLR для кристаллических 5-канальных 32x32 патчей.

    Архитектура:
      1. ResNet18 (адаптированный под 5 каналов и 32x32)
      2. Projection Head v2 (унифицирован с SEM): Linear → BN → ReLU → Linear → BN
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_dim: int = 128,
        head_version: str = "v2",
    ) -> None:
        super().__init__()

        # ResNet18 без предобученных весов — домен сильно отличается от ImageNet
        resnet = models.resnet18(weights=None)
        self.enc_dim = resnet.fc.in_features  # 512

        # Для малых 32x32 патчей заменяем conv1 (7x7 stride=2) на 3x3 stride=1
        # и убираем maxpool. Иначе feature map сжимается до 4x4, что слишком мало.
        resnet.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        resnet.maxpool = nn.Identity()

        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        use_bn = head_version == "v2"
        self.projector = SimCLRProjectionHead(
            in_dim=self.enc_dim,
            out_dim=out_dim,
            use_bn=use_bn,
        )
        self.head_version = head_version

    def forward(self, x: torch.Tensor):
        """Args: x: (B, 5, 32, 32) -> Returns: h (B, 512), z (B, out_dim)."""
        h = self.encoder(x)
        h = torch.flatten(h, 1)

        z = self.projector(h)

        h = nn.functional.normalize(h, dim=1)
        z = nn.functional.normalize(z, dim=1)

        return h, z

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override: адаптирует legacy-чекпоинты (projector.X -> projector.net.X).

        До выделения `SimCLRProjectionHead` projector был inline `nn.Sequential`
        без обёртки, поэтому ключи в старых чекпоинтах имеют форму
        `projector.0.weight`, а не `projector.net.0.weight`. Переименовываем
        на лету — это позволяет загружать все исторические чекпоинты без
        ручных миграций.
        """
        needs_remap = any(
            k.startswith("projector.") and not k.startswith("projector.net.")
            for k in state_dict.keys()
        )
        if needs_remap:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("projector.") and not k.startswith("projector.net."):
                    remapped[k.replace("projector.", "projector.net.", 1)] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    model = CrystalSimCLR(in_channels=5, out_dim=128)
    x = torch.randn(4, 5, 32, 32)
    h, z = model(x)
    print(f"Input:  {x.shape}")
    print(f"h (encoder): {h.shape}")
    print(f"z (projector): {z.shape}")
    print(f"Head version: {model.head_version}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
