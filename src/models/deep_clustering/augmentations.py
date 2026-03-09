import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter, Image

class GaussianBlur(object):
    """Gaussian blur augmentation as described in the SimCLR paper."""
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class AddGaussianNoise(object):
    """Add gaussian noise to simulate SEM electronic noise"""
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_simclr_transforms(input_size=224):
    """
    Создает пайплайн аугментаций для Contrastive Learning СЭМ-снимков.
    Важно: В микроскопии цвет не играет роли (все ЧБ), поэтому color jitter смещен 
    в сторону яркости и контраста, чтобы учить инвариантность к условиям пучка электронов.
    Сильные случайные кропы (RandomResizedCrop) используются для выучивания инвариантности к масштабу.
    """
    
    # Аугментация для тренировки (дает две разные искаженные версии картинки)
    train_transform = transforms.Compose([
        # 1. Симуляция разного масштаба: отрезаем от 20% до 100% картинки и растягиваем в 224х224
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        # 2. Повороты и отзеркаливания (для текстур ориентация часто неважна)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # 3. Варьируем параметры съемки (яркость и контраст, без hue/saturation для ЧБ)
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)
        ], p=0.8),
        # 4. Расфокус линзы микроскопа
        transforms.RandomApply([GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5),
        transforms.ToTensor(),
        # 5. Электронный шум матрицы (применяется уже к тензору)
        transforms.RandomApply([AddGaussianNoise(0., 0.04)], p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform
