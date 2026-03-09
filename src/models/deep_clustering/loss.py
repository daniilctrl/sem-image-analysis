import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    Функция потерь из статьи SimCLR. Приближает (притягивает) позитивные пары 
    и расталкивает (отдаляет) все остальные негативные пары в батче.
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z1, z2):
        """
        z1: вектора первой половины аугментаций (Batch, Dim)
        z2: вектора второй половины аугментаций (Batch, Dim)
        """
        device = z1.device
        batch_size = z1.size(0)

        # 1. Объединяем вектора [z1_1, z1_2... z1_N, z2_1, z2_2... z2_N]
        # Всего 2 * Batch элементов
        z = torch.cat((z1, z2), dim=0)

        # 2. Вычисляем косинусное сходство между всеми парами векторов
        # Результат - матрица (2N) x (2N)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # 3. Убираем самоподобие (диагональ), заменяя ее на большое отрицательное число, 
        # чтобы exp(-inf) = 0 в Softmax.
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim_matrix.masked_fill_(mask, -9e15)

        # 4. Формируем метки
        # Истинный класс для z1[i] это z2[i], то есть индекс (i + batch_size)
        # Истинный класс для z2[i] это z1[i], то есть индекс (i)
        target1 = torch.arange(batch_size, 2 * batch_size).to(device)
        target2 = torch.arange(batch_size).to(device)
        targets = torch.cat((target1, target2), dim=0)

        # 5. Считаем Loss
        loss = self.criterion(sim_matrix, targets)

        return loss / (2 * batch_size)
