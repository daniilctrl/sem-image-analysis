import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class ContrastiveLearningDataset(Dataset):
    """
    Датасет для Contrastive Learning (SimCLR/BYOL).
    Для каждой картинки-тайла возвращает ДВЕ её аугментированные (искаженные) версии.
    Сеть должна будет научиться получать из них одинаковые эмбеддинги.
    """
    def __init__(self, df, data_dir, transform):
        """
        Args:
            df: DataFrame c метаданными (полем tile_name)
            data_dir: Путь к папке `processed` с тайлами
            transform: Композиция PyTorch Transforms, описанная в augmentations.py
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['tile_name']
        img_path = self.data_dir / img_name
        
        # Конвертация в RGB, т.к. ResNet50 ожидает 3 канала
        image = Image.open(img_path).convert('RGB')
        
        # Главная фишка Contrastive Learning: 
        # Мы пропускаем ОДНУ и ТУ ЖЕ картинку через случайные аугментации ДВА РАЗА.
        # Получаем (x_i, x_j) - positive pair (якорь и позитивный пример).
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return view1, view2
