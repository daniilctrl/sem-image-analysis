# Распознавание структурных мотивов на поверхности BCC-кристалла

## Описание проекта

Автоматический пайплайн для распознавания и классификации локальных паттернов укладки атомов
(террасы, ступени, изломы, адатомы) на полусферической поверхности кристалла с ОЦК-решёткой.

**Метод:** Контрастивное обучение (SimCLR), ResNet18, 5-канальные патчи 32×32.

> Обновлено: 2026-04-17.

---

## Пайплайн

| Этап | Содержание | Файлы |
|------|-----------|-------|
| 1. Загрузка данных | Парсинг координат, валидация, сохранение | `src/crystal/load_data.py` |
| 2. Генерация патчей | KDTree → касательная плоскость → Грам-Шмидт → растеризация 32×32 | `src/crystal/patch_generator.py` |
| 3. Обучение SimCLR | NT-Xent loss, ResNet18 (5ch), cosine annealing, 50 эпох | `src/crystal/train_crystal.py`, `model_crystal.py` |
| 4. Аугментации | Rotation90, Flip, CropResize, GaussianNoise, ChannelDrop | `src/crystal/augmentations_crystal.py` |
| 5. Эмбеддинги + кластеры | 512-мерные эмбеддинги, KMeans (k=5,8,10) | `src/crystal/extract_crystal_embeddings.py` |
| 6. Оптимизация k | Silhouette, CH, DB, Elbow, AMI для k=3..20 | `src/crystal/optimize_clusters.py` |
| 7. Miller-классификация | 9 BCC-семейств + Vicinal, кросс-табуляция vs кластеры | `src/crystal/miller_utils.py`, `analyze_miller_indices.py` |
| 8. Retrieval | FAISS nearest-neighbor, precision@K_miller, cluster_coherence@K | `src/crystal/retrieve_crystal.py` |
| 9. Визуализация | UMAP, Plotly 3D, проекция Мольвейде, HTML-отчёт | `src/crystal/visualize_crystal.py` |

---

## Ключевые результаты

- **149 947** поверхностных патчей (5 каналов, 32×32)
- **NT-Xent Loss = 1.82** после 50 эпох обучения
- **k = 8** кластеров — компромиссный выбор (обоснование: `data/crystal/analysis/k_selection_justification.md`); физически обоснованная оценка ~20 зон
- **9 семейств Миллера** + Vicinal/Mixed — реализованы в `miller_utils.py`
- **Retrieval**: FAISS cosine similarity, precision@K_miller как внешняя метрика

---

## Как запустить

```bash
# 1. Генерация патчей (~15 мин)
python src/crystal/patch_generator.py

# 2. Обучение (GPU, Colab)
python src/crystal/train_crystal.py --epochs 50 --device cuda

# 3. Эмбеддинги + кластеры
python src/crystal/extract_crystal_embeddings.py \
    --checkpoint models/crystal/crystal_simclr_best.pth

# 4. Оптимизация k
python src/crystal/optimize_clusters.py

# 5. Miller-классификация
python src/crystal/analyze_miller_indices.py

# 6. Retrieval
python src/crystal/retrieve_crystal.py --K 10

# 7. Визуализация кластеров
python src/crystal/visualize_crystal.py --n_clusters 8

# 8. Smoke-тесты
python tests/test_smoke.py
```

---

## Артефакты

| Файл | Описание |
|------|----------|
| `data/crystal/patches/patches.npy` | (N, 5, 32, 32) — патчи |
| `data/crystal/patches/patches_metadata.csv` | X, Y, Z + neighbor counts |
| `data/crystal/embeddings/crystal_embeddings.npy` | (N, 512) — эмбеддинги |
| `data/crystal/embeddings/embeddings_metadata.csv` | Метаданные + cluster labels |
| `data/crystal/analysis/cluster_optimization_metrics.csv` | Метрики для k=3..20 |
| `data/crystal/analysis/k_selection_justification.md` | Обоснование выбора k |
| `data/crystal/analysis/miller_classification_crosstab.csv` | Кластеры vs Miller |
| `models/crystal/crystal_simclr_best.pth` | Лучший чекпоинт |
