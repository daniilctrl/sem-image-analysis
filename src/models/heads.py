"""Общие projection / prediction heads для SSL-моделей.

Раньше SEM-ветка и Crystal-ветка использовали разные архитектуры
projection head для SimCLR:

  - SEM (model.py):           Linear -> ReLU -> Linear                 (v1, plain)
  - Crystal (model_crystal.py):Linear -> BN -> ReLU -> Linear -> BN    (v2, BN)

Без унификации сравнение «SimCLR на SEM» vs «SimCLR на Crystal»
некорректно: переменная «архитектура projection head» не зафиксирована.
BatchNorm в projection head — де-факто стандарт после SimCLR v2 (Chen et al. 2020b).

Этот модуль вводит единый `SimCLRProjectionHead` c флагом `use_bn`.
По умолчанию `use_bn=True` (соответствует v2 и Crystal).

Для обратной совместимости с legacy state_dict (v1 без BN) предусмотрена
утилита `detect_head_version_from_state_dict`, которая автоматически
определяет версию checkpoint по составу ключей.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimCLRProjectionHead(nn.Module):
    """Унифицированный projection head для SimCLR/Crystal-SimCLR.

    v2 (use_bn=True, default):
        Linear(in, in, bias=False) -> BN(in) -> ReLU -> Linear(in, out, bias=False) -> BN(out)
    v1 (use_bn=False, legacy):
        Linear(in, in)              -> ReLU          -> Linear(in, out)

    Аргументы:
        in_dim: размерность входного эмбеддинга (например, 2048 для ResNet50,
                512 для ResNet18).
        out_dim: размерность выхода (обычно 128 для SimCLR).
        hidden_dim: размерность hidden слоя (по умолчанию == in_dim).
        use_bn: использовать BatchNorm между Linear слоями.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int | None = None,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.use_bn = use_bn

        if use_bn:
            # SimCLR v2 style: BN стабилизирует contrastive loss.
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.out_dim, bias=False),
                nn.BatchNorm1d(self.out_dim),
            )
        else:
            # SimCLR v1 style: без BN. Оставлен для обратной совместимости
            # с legacy SEM-чекпоинтами.
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def version(self) -> str:
        return "v2" if self.use_bn else "v1"


def detect_head_version_from_state_dict(
    state_dict: dict,
    projector_prefix: str = "projector.",
) -> str:
    """Автоматически определяет версию projection head по ключам state_dict.

    v2 state_dict содержит ключи вида 'projector.1.weight', 'projector.1.bias',
    'projector.4.weight', 'projector.4.bias' для BN слоёв. Если таких ключей
    нет — это v1.

    Используется в extract_*_embeddings.py для автоматической загрузки
    legacy чекпоинтов без ручного указания use_bn.

    Возвращает 'v1' или 'v2'.
    """
    bn_keys = [
        k for k in state_dict.keys()
        if k.startswith(projector_prefix) and (
            k.endswith(".running_mean") or k.endswith(".running_var")
        )
    ]
    return "v2" if bn_keys else "v1"
