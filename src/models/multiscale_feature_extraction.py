import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class MultiScaleTileDataset(torch.utils.data.Dataset):
    """
    Датасет, который для каждого тайла генерирует 3 версии (масштаба):
    - Оригинал
    - Уменьшенный (0.7x, что симулирует более мелкий масштаб)
    - Увеличенный (1.5x, что симулирует более крупный масштаб/кроп)
    """
    def __init__(self, df, data_dir, base_transform=None):
        self.df = df
        self.data_dir = Path(data_dir)
        self.base_transform = base_transform
        
        # Для Scale 0.7x: нам нужно сделать padding или resize, чтобы "отдалить" объект
        # В контексте 224x224 входа для ResNet, это эквивалентно сжатию до 156x156 и паддингу
        self.transform_0_7x = transforms.Compose([
            transforms.Resize((int(224 * 0.7), int(224 * 0.7))),
            transforms.Pad(padding=int((224 - int(224 * 0.7)) / 2), fill=0),
            transforms.CenterCrop((224, 224)), # Гарантируем точный размер
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Для Scale 1.5x: берем центральный кроп (приближение) и растягиваем до 224x224
        self.transform_1_5x = transforms.Compose([
            transforms.CenterCrop((int(256 / 1.5), int(256 / 1.5))), # Берем кроп из оригинального размера 256
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Стандартный 1.0x (наш base_transform)
        if base_transform is None:
            self.transform_1_0x = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_1_0x = base_transform
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['tile_name']
        img_path = self.data_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        # Создаем три масштаба
        img_07x = self.transform_0_7x(image)
        img_10x = self.transform_1_0x(image)
        img_15x = self.transform_1_5x(image)
        
        # Объединяем их в один "супер-батч" тензор
        # Shape: (3, Channels, H, W)
        multi_scale_images = torch.stack([img_07x, img_10x, img_15x])
        
        return multi_scale_images, img_name

def extract_multiscale_features(data_dir, metadata_path, output_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Multi-Scale Extraction")
    
    # 1. Загружаем предобученную модель
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1] 
    model = torch.nn.Sequential(*modules)
    model = model.to(device)
    model.eval()
    
    # 2. Подготовка данных
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(df)} tiles.")
    
    dataset = MultiScaleTileDataset(df, data_dir)
    # Batch size меньше, так как на каждый элемент мы генерируем 3 картинки
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_features_fused = []
    all_names = []
    
    # 3. Извлечение признаков
    print("Extracting multi-scale features...")
    with torch.no_grad():
        for multi_batch, names in tqdm(dataloader):
            # multi_batch shape: (B, 3, 3, 224, 224)
            b = multi_batch.size(0)
            
            # Решейпим в плоский батч (B*3, 3, 224, 224) для отправки в сеть
            flat_batch = multi_batch.view(-1, 3, 224, 224).to(device)
            
            # Извлекаем признаки
            features = model(flat_batch)
            features = features.squeeze().cpu().numpy() # shape (B*3, 2048)
            
            # Решейпим обратно к (B, 3, 2048) 
            features = features.reshape(b, 3, -1)
            
            # Стратегия объединения (Fusion Strategy)
            # Мы используем Max Pooling вдоль оси масштабов (взято как лучший baseline)
            # Это дает вектору признаков способность выбирать наиболее выраженные паттерны из любого масштаба
            fused_features = np.max(features, axis=1) # shape (B, 2048)
            
            all_features_fused.append(fused_features)
            all_names.extend(names)
            
            del multi_batch, flat_batch, features, fused_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    print("Concatenating fused features...")
    all_features_fused = np.vstack(all_features_fused)
    
    # 4. Сохраняем результаты
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    emb_file = output_path / "resnet50_multiscale_embeddings.npy"
    names_file = output_path / "embedding_names_multiscale.csv" # Должен совпадать с оригинальным
    
    np.save(emb_file, all_features_fused)
    pd.DataFrame({'tile_name': all_names}).to_csv(names_file, index=False)
    
    print(f"Saved {all_features_fused.shape[0]} Multi-Scale embeddings of dimension {all_features_fused.shape[1]}")
    print(f"Embeddings saved to {emb_file}")

if __name__ == "__main__":
    data_directory = r"c:\projects\diploma\data\processed"
    metadata_file = r"c:\projects\diploma\data\processed\tiles_metadata.csv"
    output_directory = r"c:\projects\diploma\data\embeddings"
    
    extract_multiscale_features(data_directory, metadata_file, output_directory, batch_size=32)
