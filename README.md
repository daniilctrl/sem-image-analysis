# Diploma Project: SEM + Crystal

Репозиторий объединяет две параллельные исследовательские ветки диплома:

- **SEM / SiC (scale invariance)** - продолжается активная разработка и частичная разметка.
- **Crystal / BCC surface motifs** - завершена первая end-to-end итерация продукта.

## Статус направлений

| Направление | Статус | Основная цель |
|---|---|---|
| SEM (SiC) | In progress | Инвариантность к масштабу, retrieval, кластеризация SEM-структур |
| Crystal (BCC) | Iteration 1 complete | Классификация локальных мотивов поверхности (патчи, эмбеддинги, кластеры) |

## Навигация по коду

### SEM (основной R&D поток)

- `src/data/` - подготовка данных, разметка, метаданные масштаба
- `src/models/` - SimCLR/BYOL и извлечение признаков
- `src/evaluation/` - метрики, кросс-масштабное сравнение/retrieval
- `src/visualization/` - визуализация и вспомогательный UI
- `data/processed/`, `data/embeddings/`, `data/results/` - основные артефакты экспериментов

### Crystal (завершенная 1-я итерация)

- `src/crystal/` - полный pipeline (загрузка данных -> патчи -> обучение -> эмбеддинги -> кластеры)
- `docs/crystal_project_overview.md` - обзор реализованного решения
- `docs/next_steps.md` - план развития после первой итерации
- `data/crystal/` - данные и артефакты по направлению
- `notebooks/crystal_surface_report.*` - итоговые отчетные материалы

## Быстрый старт

```bash
pip install -r requirements.txt
```

Примеры запуска:

```bash
# SEM: пример метрик инвариантности к масштабу
python src/evaluation/scale_invariance_metrics.py

# Crystal: пример построения эмбеддингов и кластеризации
python src/crystal/extract_crystal_embeddings.py
```

## Политика версионирования

Правила, что хранится в Git, а что остается внешними артефактами, описаны в `docs/versioning_policy.md`.
