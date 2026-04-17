"""SimCLR для SEM-изображений (ResNet50, 3-канальный вход).

Использует унифицированный `SimCLRProjectionHead` из `src.models.heads`,
согласованный с Crystal-веткой. Это убирает переменную «архитектура
projection head» в сравнении SEM SimCLR vs Crystal SimCLR.

По умолчанию используется SimCLR v2 head (BN в projection). Для загрузки
legacy SEM-чекпоинтов без BN можно передать `head_version='v1'` или
использовать `SimCLR.from_state_dict(state_dict, ...)`, которая автоматически
определяет версию head по составу ключей.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.heads import SimCLRProjectionHead, detect_head_version_from_state_dict  # noqa: E402


class SimCLR(nn.Module):
    """SimCLR архитектура для SEM-изображений.

    1. Base Encoder (ResNet50 ImageNet v2) извлекает вектор h (2048).
    2. Projection Head (v2 по умолчанию) проецирует h в z (128).
       Contrastive loss (NT-Xent) применяется к z.
    """

    def __init__(
        self,
        base_model: str = "resnet50",
        out_dim: int = 128,
        head_version: str = "v2",
    ) -> None:
        super().__init__()

        if base_model == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.enc_dim = resnet.fc.in_features
            modules = list(resnet.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown base model {base_model}")

        use_bn = head_version == "v2"
        self.projector = SimCLRProjectionHead(
            in_dim=self.enc_dim,
            out_dim=out_dim,
            use_bn=use_bn,
        )
        self.head_version = head_version

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        h = torch.flatten(h, 1)  # (B, 2048)

        z = self.projector(h)    # (B, out_dim)

        h = nn.functional.normalize(h, dim=1)
        z = nn.functional.normalize(z, dim=1)

        return h, z

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override: адаптирует legacy-чекпоинты с projector.X -> projector.net.X."""
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

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        base_model: str = "resnet50",
        out_dim: int = 128,
    ) -> "SimCLR":
        """Фабрика: создаёт SimCLR с правильной версией head по state_dict.

        Автоматически определяет v1/v2 по наличию BN-ключей
        (`running_mean`/`running_var`) в projector. Это важно для загрузки
        legacy SEM-чекпоинтов (v1, до этого коммита) без ручного указания
        `head_version`.
        """
        # Сначала ищем BN в новой структуре (projector.net.*), затем в legacy (projector.*).
        version = detect_head_version_from_state_dict(state_dict, "projector.net.")
        if version == "v1":
            version = detect_head_version_from_state_dict(state_dict, "projector.")
        model = cls(base_model=base_model, out_dim=out_dim, head_version=version)
        model.load_state_dict(state_dict, strict=False)
        return model
