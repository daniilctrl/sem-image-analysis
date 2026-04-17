"""Contrastive augmentations for SEM images.

Параметры аугментаций вытащены в kwargs, чтобы их можно было менять через
CLI trainers'ов без правки кода. Дефолты соответствуют предыдущей
hardcoded-конфигурации (SimCLR paper adapted для grayscale SEM):

  - RandomResizedCrop(224, scale=(0.2, 1.0))
  - RandomHorizontal/VerticalFlip
  - ColorJitter(brightness=0.4, contrast=0.4), p=0.8 (sat/hue=0 т.к. SEM ЧБ)
  - GaussianBlur(sigma∈[0.1, 2.0]), p=0.5
  - AddGaussianNoise(std=0.04), p=0.2
  - Normalize ImageNet

Dataclass `AugmentationConfig` удобен для передачи из CLI + сериализации
в TensorBoard hparams и в `data/results/sem_eval_report_*.md`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter


class GaussianBlur:
    """Gaussian blur augmentation as described in the SimCLR paper."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class AddGaussianNoise:
    """Add gaussian noise to simulate SEM electronic noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.05) -> None:
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


@dataclass
class AugmentationConfig:
    """All knobs of the SEM contrastive augmentation pipeline.

    Аккуратная структура вместо магических чисел в `get_simclr_transforms`:
    передаётся из CLI, логируется в TensorBoard hparams и в отчёты. Любой
    sweep (PR #9) может сериализовать эту структуру как конфиг эксперимента.
    """

    input_size: int = 224

    # RandomResizedCrop
    crop_scale_min: float = 0.2
    crop_scale_max: float = 1.0

    # ColorJitter
    color_jitter_p: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.0     # 0 для grayscale SEM
    hue: float = 0.0

    # GaussianBlur
    blur_p: float = 0.5
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0

    # AddGaussianNoise
    noise_p: float = 0.2
    noise_std: float = 0.04
    noise_mean: float = 0.0

    # ImageNet normalize
    norm_mean: tuple[float, float, float] = field(default=(0.485, 0.456, 0.406))
    norm_std: tuple[float, float, float] = field(default=(0.229, 0.224, 0.225))

    def to_dict(self) -> dict:
        return asdict(self)


def get_simclr_transforms(
    input_size: int = 224,
    config: AugmentationConfig | None = None,
):
    """Возвращает torchvision-композицию SEM-аугментаций.

    Обратная совместимость: если передан только `input_size`, поведение
    идентично предыдущей hardcoded-версии. Если передан `config`,
    используются его значения.
    """
    cfg = config or AugmentationConfig(input_size=input_size)

    return transforms.Compose([
        # Масштабный кроп — имитация разных увеличений
        transforms.RandomResizedCrop(
            cfg.input_size,
            scale=(cfg.crop_scale_min, cfg.crop_scale_max),
        ),
        # Ориентация текстур незначима
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Параметры электронного пучка (яркость/контраст)
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
            )
        ], p=cfg.color_jitter_p),
        # Дефокус
        transforms.RandomApply(
            [GaussianBlur(sigma_min=cfg.blur_sigma_min, sigma_max=cfg.blur_sigma_max)],
            p=cfg.blur_p,
        ),
        transforms.ToTensor(),
        # Шум детектора
        transforms.RandomApply(
            [AddGaussianNoise(mean=cfg.noise_mean, std=cfg.noise_std)],
            p=cfg.noise_p,
        ),
        transforms.Normalize(mean=list(cfg.norm_mean), std=list(cfg.norm_std)),
    ])


def add_augmentation_args(parser) -> None:
    """Добавляет CLI-флаги для AugmentationConfig в argparse.

    Используется в train.py и train_byol.py — один источник истины.
    """
    group = parser.add_argument_group("augmentations (SEM contrastive)")
    group.add_argument("--input_size", type=int, default=224)
    group.add_argument("--crop_scale_min", type=float, default=0.2,
                       help="RandomResizedCrop min scale (default: 0.2)")
    group.add_argument("--crop_scale_max", type=float, default=1.0)
    group.add_argument("--color_jitter_p", type=float, default=0.8)
    group.add_argument("--brightness", type=float, default=0.4)
    group.add_argument("--contrast", type=float, default=0.4)
    group.add_argument("--blur_p", type=float, default=0.5)
    group.add_argument("--blur_sigma_min", type=float, default=0.1)
    group.add_argument("--blur_sigma_max", type=float, default=2.0)
    group.add_argument("--noise_p", type=float, default=0.2)
    group.add_argument("--noise_std", type=float, default=0.04)


def make_config_from_args(args) -> AugmentationConfig:
    """Собирает AugmentationConfig из argparse Namespace."""
    return AugmentationConfig(
        input_size=getattr(args, "input_size", 224),
        crop_scale_min=getattr(args, "crop_scale_min", 0.2),
        crop_scale_max=getattr(args, "crop_scale_max", 1.0),
        color_jitter_p=getattr(args, "color_jitter_p", 0.8),
        brightness=getattr(args, "brightness", 0.4),
        contrast=getattr(args, "contrast", 0.4),
        blur_p=getattr(args, "blur_p", 0.5),
        blur_sigma_min=getattr(args, "blur_sigma_min", 0.1),
        blur_sigma_max=getattr(args, "blur_sigma_max", 2.0),
        noise_p=getattr(args, "noise_p", 0.2),
        noise_std=getattr(args, "noise_std", 0.04),
    )
