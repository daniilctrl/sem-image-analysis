"""Глобальная фиксация seed для воспроизводимости экспериментов.

Используется во всех train/extract/eval скриптах в начале исполнения.
Без этого повторные прогоны на одних и тех же данных дают разные числа
(веса модели, kmeans, subsampling, augmentation), что делает сравнение
разных конфигураций — в том числе multi-radius ablation — невалидным.

Пример использования:

    from src.utils.repro import set_global_seed
    set_global_seed(42)

Покрытые источники недетерминизма:
    - `random` (Python stdlib)
    - `numpy`
    - `torch` CPU + CUDA (включая cuDNN)
    - `PYTHONHASHSEED`

Не покрытые (сознательно):
    - FAISS IndexFlatIP tie-breaking для эквидистантных векторов
    - KDTree neighbor ordering при равных расстояниях (scipy)
    - Некоторые CUDA операции, у которых нет детерминированной реализации
      (с warn_only=True pytorch выведет warning, но не упадёт)
"""

from __future__ import annotations

import os
import random
from typing import Optional


_DEFAULT_SEED = 42


def set_global_seed(seed: int = _DEFAULT_SEED, deterministic_torch: bool = True) -> int:
    """Фиксирует seed для всех основных источников случайности.

    Аргументы:
        seed: начальное значение (по умолчанию 42 — то же, что в manifest).
        deterministic_torch: включить `torch.use_deterministic_algorithms`
            и `cudnn.deterministic = True`. Может замедлить обучение
            на 5–15%, но обеспечивает побитово-воспроизводимые результаты
            на одной и той же версии CUDA/cuDNN.

    Возвращает:
        Использованный seed (для логирования).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed helper для `DataLoader(..., worker_init_fn=seed_worker)`.

    Нужен, чтобы каждый DataLoader-worker имел детерминированный,
    но разный seed. Без этого при num_workers > 0 можно получить
    недетерминированную выборку в батчах.
    """
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = _DEFAULT_SEED) -> Optional["torch.Generator"]:
    """Возвращает `torch.Generator` с зафиксированным seed.

    Нужен для DataLoader(..., generator=g), чтобы shuffle был детерминированным.
    """
    try:
        import torch
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    except ImportError:
        return None
