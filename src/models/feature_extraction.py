import os
import gc
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class TileDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = Path(data_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['tile_name']
        img_path = self.data_dir / img_name
        
        # Читаем как RGB, т.к. ResNet50 ожидает 3 канала
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

def extract_features(data_dir, metadata_path, output_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Загружаем предобученную модель
    # Используем ResNet50, убираем последний полносвязный слой (fc)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1] # Убираем klassifier
    model = torch.nn.Sequential(*modules)
    model = model.to(device)
    model.eval()
    
    # 2. Подготовка данных
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(df)} tiles.")
    
    # Стандартные трансформации для ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TileDataset(df, data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_names = []
    
    # 3. Извлечение признаков
    print("Extracting features...")
    with torch.no_grad():
        for images, names in tqdm(dataloader):
            images = images.to(device)
            # Извлекаем фичи, форма будет (batch_size, 2048, 1, 1)
            features = model(images)
            # Делаем squeeze до (batch_size, 2048)
            features = features.squeeze().cpu().numpy()
            
            all_features.append(features)
            all_names.extend(names)
            
            # Очистка памяти
            del images
            del features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Объединяем списки батчей
    print("Concatenating features...")
    all_features = np.vstack(all_features)
    
    # 4. Сохраняем результаты
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    emb_file = output_path / "resnet50_embeddings.npy"
    names_file = output_path / "embedding_names.csv"
    
    np.save(emb_file, all_features)
    pd.DataFrame({'tile_name': all_names}).to_csv(names_file, index=False)
    
    print(f"Saved {all_features.shape[0]} embeddings of dimension {all_features.shape[1]}")
    print(f"Embeddings saved to {emb_file}")

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    data_directory = str(_root / "data" / "processed")
    metadata_file = str(_root / "data" / "processed" / "tiles_metadata.csv")
    output_directory = str(_root / "data" / "embeddings")

    extract_features(data_directory, metadata_file, output_directory, batch_size=128)
