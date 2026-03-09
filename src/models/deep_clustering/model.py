import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    """
    SimCLR архитектура:
    1. Base Encoder: (здесь ResNet50) извлекает вектор представления h.
    2. Projection Head: MLP (здесь 1 скрытый слой), который проецирует h в пространство меньшей размерности,
       в котором применяется функция потерь Contrastive Loss.
    """
    def __init__(self, base_model="resnet50", out_dim=128):
        super(SimCLR, self).__init__()
        
        # 1. Загружаем энкодер
        if base_model == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.enc_dim = resnet.fc.in_features
            
            # Удаляем финальный FC слой (classifier)
            modules = list(resnet.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown base model {base_model}")
            
        # 2. Создаем Projection Head (g(·)) как в статье SimCLR (нелинейный MLP)
        self.projector = nn.Sequential(
            nn.Linear(self.enc_dim, self.enc_dim),
            nn.ReLU(),
            nn.Linear(self.enc_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.squeeze() # (Batch, 2048)
        
        z = self.projector(h) # (Batch, 128)
        
        # В статье SimCLR рекомендуется нормализовать выход
        h = nn.functional.normalize(h, dim=1)
        z = nn.functional.normalize(z, dim=1)
        
        return h, z
