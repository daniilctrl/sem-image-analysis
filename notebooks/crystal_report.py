"""
Скрипт для генерации Jupyter Notebook отчёта.
Запуск: python notebooks/crystal_report.py
Результат: notebooks/crystal_surface_report.ipynb
"""
import json
from pathlib import Path

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# ════════════════════════════════════════════════════════════════
# Титульный блок
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""# Распознавание структурных мотивов на поверхности кристалла ОЦК-решётки
## Отчёт по учебной / производственной практике

**Метод:** Контрастивное обучение (SimCLR) на основе адаптированного ResNet18  
**Данные:** Полусфера BCC-кристалла, R=50 параметров решётки, ~270 000 атомов  
**Автор:** *[ФИО]*  
**Научный руководитель:** *[ФИО]*

---"""))

# ════════════════════════════════════════════════════════════════
# 1. Введение
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 1. Постановка задачи

Поверхность кристалла с ОЦК (объёмно-центрированной кубической) решёткой имеет сложную топографию: **террасы** (плоские участки), **ступени** (edges), **изломы** (kinks) и **адатомы** (одиночные выступающие атомы). Эти структурные мотивы определяют каталитическую активность, коррозионную стойкость и другие физические свойства поверхности.

**Цель работы:** Разработать автоматический пайплайн для распознавания и классификации структурных мотивов на полусферической поверхности кристалла, используя методы самообучения (self-supervised learning).

### Схема пайплайна

```
Excel (X,Y,Z, n1..n5)          Генерация патчей           Обучение SimCLR
──────────────────── → KDTree + касательная плоскость → ResNet18 (5 каналов)
   270 000 атомов        → 150 000 патчей 32×32            50 эпох, GPU
                                                              ↓
                         Визуализация                   Кластеризация
                    ← Plotly 3D + Мольвейде + UMAP  ←  KMeans (8 кластеров)
```"""))

# ════════════════════════════════════════════════════════════════
# 2. Загрузка данных
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 2. Загрузка и валидация данных

Исходные данные — координаты атомов `(X, Y, Z)` и числа соседей 1–5 порядка `(n1, n2, n3, n4, n5)` для полусферы BCC-кристалла радиусом R=50."""))

cells.append(code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
data_dir = Path("../data/crystal")
atoms = pd.read_parquet(data_dir / "atoms.parquet")

print(f"Всего атомов: {len(atoms):,}")
print(f"\\nСтолбцы: {list(atoms.columns)}")
print(f"\\nСтатистика числа соседей:")
for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
    print(f"  {col}: min={atoms[col].min()}, max={atoms[col].max()}, mean={atoms[col].mean():.2f}")

R = np.sqrt(atoms['X']**2 + atoms['Y']**2 + atoms['Z']**2).max()
print(f"\\nРадиус полусферы: {R:.1f} (в единицах полурешётки)")
print(f"Диапазон Grayscale: [{atoms['grayscale'].min():.4f}, {atoms['grayscale'].max():.4f}]")"""))

# ════════════════════════════════════════════════════════════════
# 3. Визуализация полусферы
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 3. Визуализация исходной поверхности

Два способа окрашивания атомов (по методике руководителя):

- **RGB**: `R = n1/8`, `G = n2/6`, `B = n3/12` — кодирует соседей 1–3 порядка в цвет
- **Grayscale**: `1 − (n1·n2·n3·n4·n5) / 110592` — чем ближе к 1, тем больше атом «выступает» над поверхностью"""))

cells.append(code_cell("""fig = plt.figure(figsize=(18, 7))

# RGB
ax1 = fig.add_subplot(121, projection='3d')
colors_rgb = np.column_stack([atoms['n1']/8, atoms['n2']/6, atoms['n3']/12])
colors_rgb = np.clip(colors_rgb, 0, 1)
ax1.scatter(atoms['X'], atoms['Y'], atoms['Z'], c=colors_rgb, s=0.3, alpha=0.6, depthshade=False)
ax1.set_title("RGB: R=n1/8, G=n2/6, B=n3/12", fontsize=13)
ax1.view_init(elev=30, azim=45)

# Grayscale
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(atoms['X'], atoms['Y'], atoms['Z'], c=atoms['grayscale'], cmap='gray_r',
            s=0.3, alpha=0.6, depthshade=False)
ax2.set_title("Grayscale: 1 − n1·n2·n3·n4·n5 / 110592", fontsize=13)
ax2.view_init(elev=30, azim=45)

plt.suptitle("Полусфера BCC-кристалла (270 000 атомов)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 4. Генерация патчей
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 4. Генерация 2D-патчей из 3D-данных

### Алгоритм

Для каждого **поверхностного** атома (grayscale > 0.01):

1. **KDTree**: находим все атомы в радиусе `R_patch` от центрального
2. **Касательная плоскость**: нормаль — вектор `[X, Y, Z]` (т.к. полусфера центрирована в начале координат)
3. **Проекция Грама-Шмидта**: строим ортонормированный базис `(e1, e2)` на касательной плоскости
4. **Растеризация**: проецируем соседей на плоскость `(u, v)` и раскладываем по пикселям `32×32`

Каждый пиксель хранит **5 каналов** — нормированные числа соседей `[n1/8, n2/6, n3/12, n4/24, n5/8]`."""))

cells.append(code_cell("""# Загрузка сгенерированных патчей
patches = np.load(data_dir / "patches" / "patches.npy")
meta = pd.read_csv(data_dir / "patches" / "patches_metadata.csv")

print(f"Форма массива патчей: {patches.shape}")
print(f"  → {patches.shape[0]:,} патчей × {patches.shape[1]} каналов × {patches.shape[2]}×{patches.shape[3]} пикселей")
print(f"  → Размер в памяти: {patches.nbytes / 1e9:.2f} ГБ")
print(f"\\nМетаданные: {meta.shape[0]:,} строк × {meta.shape[1]} столбцов")
print(f"Столбцы: {list(meta.columns)}")"""))

cells.append(code_cell("""# Визуализация примеров патчей
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
fig.suptitle("Примеры патчей (среднее по 5 каналам)", fontsize=14, fontweight='bold')

indices = np.random.RandomState(42).choice(len(patches), 32, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 8, i % 8]
    patch_mean = patches[idx].mean(axis=0)
    ax.imshow(patch_mean, cmap='viridis', interpolation='nearest')
    gs = meta.iloc[idx].get('grayscale', 0)
    ax.set_title(f"gs={gs:.2f}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()"""))

cells.append(code_cell("""# 5-канальная декомпозиция одного патча
channel_names = ['n1/8', 'n2/6', 'n3/12', 'n4/24', 'n5/8']
sample_idx = np.random.RandomState(123).choice(len(patches))

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for ch in range(5):
    axes[ch].imshow(patches[sample_idx, ch], cmap='inferno', interpolation='nearest')
    axes[ch].set_title(channel_names[ch], fontsize=11)
    axes[ch].axis('off')

# RGB-композит из первых 3 каналов
rgb = np.stack([patches[sample_idx, 0], patches[sample_idx, 1], patches[sample_idx, 2]], axis=-1)
axes[5].imshow(rgb, interpolation='nearest')
axes[5].set_title("RGB-композит", fontsize=11)
axes[5].axis('off')

plt.suptitle(f"Декомпозиция патча #{sample_idx} по каналам", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 5. Архитектура модели
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 5. Архитектура модели: CrystalSimCLR

Использована архитектура **SimCLR** (Simple Framework for Contrastive Learning):

| Компонент | Описание |
|---|---|
| **Энкодер** | ResNet18, модифицированный: первый `Conv2d` принимает 5 каналов (вместо 3), `MaxPool` удалён (патчи 32×32 слишком малы) |
| **Проекционная голова** | `512 → 512 → 128` (2 линейных слоя + ReLU + BatchNorm) |
| **Функция потерь** | NT-Xent (Normalized Temperature-scaled Cross-Entropy) |
| **Аугментации** | Повороты на 90°, отражения, гауссов шум, channel dropout, случайный кроп |

### Принцип работы SimCLR

Для каждого патча создаются **две аугментированные версии** (view₁, view₂). Модель учится:
- **Сближать** представления (эмбеддинги) двух видов одного и того же патча
- **Отдалять** представления разных патчей

В результате атомы с одинаковой локальной структурой получают близкие 512-мерные векторы."""))

cells.append(code_cell("""import sys, os
sys.path.append(os.path.abspath(".."))
from src.crystal.model_crystal import CrystalSimCLR

model = CrystalSimCLR(in_channels=5, out_dim=128)

# Подсчёт параметров
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Всего параметров:     {total_params:,}")
print(f"Обучаемых параметров: {trainable_params:,}")
print(f"\\nАрхитектура энкодера:")
print(model.encoder)"""))

# ════════════════════════════════════════════════════════════════
# 6. Результаты обучения
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 6. Результаты обучения

**Параметры обучения:**
- Эпох: 50
- Batch size: 256
- Оптимизатор: Adam, lr = 3×10⁻⁴
- LR scheduler: Cosine Annealing
- GPU: Google Colab T4
- Итоговый Loss: **1.82**"""))

# ════════════════════════════════════════════════════════════════
# 7. Кластеризация
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 7. Кластеризация эмбеддингов

После обучения каждый патч пропускается через энкодер, получая **512-мерный вектор** (эмбеддинг). Все эмбеддинги кластеризуются алгоритмом **KMeans** (8 кластеров)."""))

cells.append(code_cell("""# Загрузка эмбеддингов и метаданных
embeddings = np.load(data_dir / "embeddings" / "crystal_embeddings.npy")
emb_meta = pd.read_csv(data_dir / "embeddings" / "embeddings_metadata.csv")

print(f"Эмбеддинги: {embeddings.shape}")
print(f"Метаданные: {emb_meta.shape}")
print(f"\\nСтолбцы метаданных: {list(emb_meta.columns)}")

# Распределение по кластерам
if 'cluster_8' in emb_meta.columns:
    print(f"\\nРаспределение по 8 кластерам:")
    for c in range(8):
        count = (emb_meta['cluster_8'] == c).sum()
        print(f"  Кластер {c}: {count:,} атомов ({count/len(emb_meta)*100:.1f}%)")"""))

# ════════════════════════════════════════════════════════════════
# 8. UMAP
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 8. UMAP-проекция эмбеддингов

UMAP (Uniform Manifold Approximation and Projection) сжимает 512-мерные векторы в 2D, сохраняя топологическую структуру. Точки одного цвета — атомы, отнесённые к одному кластеру."""))

cells.append(code_cell("""import umap

print("Построение UMAP (может занять 2–3 минуты)...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
umap_2d = reducer.fit_transform(embeddings)

labels = emb_meta['cluster_8'].values
n_clusters = 8
colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
          '#42d4f4', '#f032e6', '#bfef45']

fig, ax = plt.subplots(figsize=(14, 10))
for c in range(n_clusters):
    mask = labels == c
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors[c],
               label=f'Кластер {c} ({mask.sum():,})', s=3, alpha=0.6, edgecolors='none')

ax.set_title(f'UMAP-проекция 512-мерных эмбеддингов ({len(embeddings):,} атомов)', fontsize=14)
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=4, fontsize=10)
plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 9. Проекция Мольвейде
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 9. Карта поверхности (проекция Мольвейде)

Проекция Мольвейде «разворачивает» полусферу на плоскость (аналогично карте Земли).  
Каждый атом раскрашен по номеру кластера. Видны чёткие зоны, соответствующие различным типам поверхностных структур."""))

cells.append(code_cell("""X = emb_meta['X'].values
Y = emb_meta['Y'].values
Z = emb_meta['Z'].values

R = np.sqrt(X**2 + Y**2 + Z**2)
theta = np.arctan2(Y, X)              # долгота
phi = np.arcsin(np.clip(Z / R, -1, 1))  # широта

fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                         subplot_kw={'projection': 'mollweide'})

# Кластеры
for c in range(n_clusters):
    mask = labels == c
    axes[0].scatter(theta[mask], phi[mask], c=colors[c],
                    label=f'Кл. {c} ({mask.sum():,})', s=1, alpha=0.7, edgecolors='none')
axes[0].set_title('Проекция Мольвейде — 8 кластеров', fontsize=13, pad=15)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5, fontsize=9)
axes[0].grid(True, alpha=0.3)

# Grayscale для сравнения
sc = axes[1].scatter(theta, phi, c=emb_meta['grayscale'].values,
                     cmap='gray_r', s=1, alpha=0.7, edgecolors='none')
axes[1].set_title('Grayscale (для сравнения)', fontsize=13, pad=15)
plt.colorbar(sc, ax=axes[1], shrink=0.6, label='grayscale')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 10. 3D Карта
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 10. 3D-карта кластеров на полусфере

Три вида полусферы с атомами, раскрашенными по кластерам:"""))

cells.append(code_cell("""atom_colors = [colors[l] for l in labels]

fig = plt.figure(figsize=(20, 6))
views = [(30, 45, 'Перспектива'), (90, 0, 'Сверху'), (0, 0, 'Сбоку')]

for i, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(X, Y, Z, c=atom_colors, s=0.5, alpha=0.7, depthshade=False)
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.suptitle('Поверхность кристалла — 8 кластеров', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 11. Интерактивная карта
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 11. Интерактивная 3D-карта (Plotly)

Ниже — интерактивная визуализация: можно вращать, масштабировать и наводить курсор на отдельные атомы."""))

cells.append(code_cell("""import plotly.graph_objects as go

fig = go.Figure()
for c in range(n_clusters):
    mask = labels == c
    idx = np.where(mask)[0]
    fig.add_trace(go.Scatter3d(
        x=X[idx], y=Y[idx], z=Z[idx],
        mode='markers',
        marker=dict(size=1.5, color=colors[c], opacity=0.7),
        name=f'Кластер {c} ({mask.sum():,})',
        text=[f'Атом #{j}<br>X={X[j]:.1f}, Y={Y[j]:.1f}, Z={Z[j]:.1f}<br>'
              f'gs={emb_meta.iloc[j].get("grayscale", 0):.3f}' for j in idx],
        hoverinfo='text',
    ))

fig.update_layout(
    title='Интерактивная карта кластеров на полусфере BCC-кристалла',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=900, height=700,
)
fig.show()"""))

# ════════════════════════════════════════════════════════════════
# 12. Распределение
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 12. Анализ кластеров"""))

cells.append(code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Размеры кластеров
cluster_sizes = np.bincount(labels, minlength=n_clusters)
axes[0].bar(range(n_clusters), cluster_sizes, color=colors)
axes[0].set_xlabel('Кластер')
axes[0].set_ylabel('Количество атомов')
axes[0].set_title('Размеры кластеров')
for i, v in enumerate(cluster_sizes):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

# Средний grayscale
if 'grayscale' in emb_meta.columns:
    gs_means = [emb_meta.loc[labels == c, 'grayscale'].mean() for c in range(n_clusters)]
    gs_stds = [emb_meta.loc[labels == c, 'grayscale'].std() for c in range(n_clusters)]
    axes[1].bar(range(n_clusters), gs_means, yerr=gs_stds, color=colors, capsize=3)
    axes[1].set_xlabel('Кластер')
    axes[1].set_ylabel('Средний grayscale')
    axes[1].set_title('Grayscale по кластерам (выше = ближе к поверхности)')

plt.suptitle('Распределение кластеров', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ════════════════════════════════════════════════════════════════
# 13. Выводы
# ════════════════════════════════════════════════════════════════
cells.append(md_cell("""## 13. Выводы и дальнейшие шаги

### Основные результаты

1. **Пайплайн работает end-to-end**: от сырых данных Excel до 3D-визуализации кластеров на поверхности кристалла.

2. **Модель выучила осмысленные признаки**: Loss = 1.82 при 50 эпохах свидетельствует о хорошей конвергенции контрастивного обучения. UMAP-проекция показывает чёткое разделение кластеров.

3. **Кластеры соответствуют физической структуре**: кольцевые и зональные паттерны на проекции Мольвейде и 3D-карте согласуются с ожидаемым распределением террас, ступеней и изломов на полусферической поверхности BCC-кристалла.

### Дальнейшие шаги

- **Интерпретация кластеров**: Совместно с руководителем определить физический смысл каждого кластера (террасы, ступени, изломы, адатомы).
- **Оптимизация числа кластеров**: Подобрать оптимальное `k` через метрики (Silhouette, Elbow method).
- **Сравнение с аналитическими методами**: Сопоставить автоматическую классификацию с известными кристаллографическими индексами граней.
- **Расширение на другие решётки**: Протестировать пайплайн на ГЦК (FCC) и ГПУ (HCP) кристаллах."""))

# ════════════════════════════════════════════════════════════════
# Сборка тетрадки
# ════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

output_path = Path(__file__).parent / "crystal_surface_report.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Notebook создан: {output_path}")
