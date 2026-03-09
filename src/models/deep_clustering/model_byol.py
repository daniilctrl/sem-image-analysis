"""
BYOL (Bootstrap Your Own Latent) — архитектура для Self-Supervised Learning.

Ключевые отличия от SimCLR:
 - НЕ использует негативные пары (нет NT-Xent Loss).
 - НЕ зависит от размера батча (можно учить на batch=32 без потери качества).
 - Использует асимметричную архитектуру: Online network (ученик) и Target network (учитель).
 - Target network обновляется через экспоненциальное скользящее среднее (EMA) весов Online network.
 - Loss = MSE между нормализованными предсказаниями Online и проекциями Target.

Ссылка: Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020.
"""
import copy
import torch
import torch.nn as nn
import torchvision.models as models


class MLP(nn.Module):
    """Projection / Prediction head (2-layer MLP)."""
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL architecture.
    Online network:  encoder -> projector -> predictor
    Target network:  encoder -> projector  (NO predictor! Это создаёт асимметрию.)
    """
    def __init__(self, base_model="resnet50", proj_dim=256, pred_dim=256, hidden_dim=4096):
        super().__init__()

        # ===== Online Network (ученик) =====
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.enc_dim = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()  # Убираем classifier
        self.online_encoder = resnet

        self.online_projector = MLP(self.enc_dim, hidden_dim, proj_dim)
        self.online_predictor = MLP(proj_dim, hidden_dim, pred_dim)  # <-- Только у Online!

        # ===== Target Network (учитель) — копия Online БЕЗ predictor =====
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Замораживаем target: его веса обновляются только через EMA
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau=0.996):
        """
        Exponential Moving Average обновление Target network.
        target = tau * target + (1 - tau) * online
        tau = 0.996 означает, что target медленно «догоняет» online.
        """
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    def forward(self, view1, view2):
        """
        Прогоняем две аугментации через Online и Target.
        Возвращаем предсказания Online и проекции Target для подсчета Loss.
        """
        # Online network: encoder -> projector -> predictor
        online_z1 = self.online_projector(self.online_encoder(view1))
        online_z2 = self.online_projector(self.online_encoder(view2))
        online_p1 = self.online_predictor(online_z1)
        online_p2 = self.online_predictor(online_z2)

        # Target network: encoder -> projector (NO predictor)
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(view1))
            target_z2 = self.target_projector(self.target_encoder(view2))

        return online_p1, online_p2, target_z1.detach(), target_z2.detach()

    def get_encoder(self):
        """Возвращает обученный энкодер для извлечения эмбеддингов."""
        return self.online_encoder


def byol_loss(p, z):
    """
    BYOL Loss: MSE между нормализованными предсказаниями и проекциями.
    L = 2 - 2 * cosine_similarity(p, z)  (эквивалент MSE на единичной сфере)
    """
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()
