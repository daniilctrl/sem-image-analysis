# Reproducibility Guide

Воспроизведение числовых результатов выпускной квалификационной работы
«Методы глубокого обучения для обратного поиска и кластеризации текстур
наноструктурированных поверхностей по микроскопическим изображениям»
(Даниил Родионов, СПбГУ).

Состояние кода и артефактов на момент защиты зафиксировано git-тегом
[`vkr-submission`](https://github.com/daniilctrl/sem-image-analysis/releases/tag/vkr-submission).

## Три уровня воспроизведения

| Уровень | Что воспроизводится | Размер | GPU |
|---|---|---|---|
| **1** | Все таблицы Главы 2-3 из готовых эмбеддингов | ~440 МБ | не нужен |
| **2** | + извлечение эмбеддингов из чекпоинтов | +2.2 ГБ | нужен |
| **3** | + полная цепочка обучения с нуля | + raw data | нужен |

## Установка окружения

```bash
git clone https://github.com/daniilctrl/sem-image-analysis.git
cd sem-image-analysis
git checkout vkr-submission   # фиксирует версию кода и артефактов

pip install -e .              # core dependencies (PyTorch, FAISS, scikit-learn)
# Опциональные extras:
# pip install -e ".[viz]"      # UMAP, seaborn, plotly, gradio
# pip install -e ".[dataprep]" # opencv, tifffile, openpyxl, h5py

# Установить GitHub CLI для скачивания артефактов:
#   winget install --id GitHub.cli   (Windows)
#   brew install gh                  (macOS)
#   sudo apt install gh              (Debian/Ubuntu)
gh auth login
```

## Уровень 1: воспроизведение таблиц без GPU

```bash
python scripts/fetch_artifacts.py --level 1
```

Скачивает (через GitHub Release):
- `tier1_sem_embeddings.tar.gz` — 4 SEM-эмбеддинга (Baseline ImageNet, SimCLR τ=0.2, SimCLR τ=0.5, BYOL 100 ep) + names + flatten32 baseline + тайминги
- `tier1_crystal_embeddings.tar.gz` — Crystal-эмбеддинги + metadata

После распаковки запустить:

| Таблица в ВКР | Команда |
|---|---|
| Глава 3, табл. retrieval/scale/probe (per-model) | `python src/evaluation/run_sem_evaluation.py --emb_dir data/embeddings --meta_path data/processed/tiles_metadata.csv --output_dir data/results/repro_l1 --K 10 --n_clusters 4` |
| Глава 3, табл. 3.7 (unified, N=18 859) | то же — числа в отчёте совпадут с табл. `tab:sem_unified` |
| Глава 3, k-sweep по метрикам инвариантности | `python src/evaluation/scale_invariance_metrics.py --emb_dir data/embeddings --ks 4 8 15 20` |
| Глава 2, k-sweep кластеризации Crystal | `python src/crystal/optimize_clusters.py --emb_path data/crystal/embeddings/crystal_embeddings.npy` |
| Глава 2, crosstab кластер × Миллер | `python src/crystal/analyze_miller_indices.py --k 8` (также `--k 35`) |
| Глава 2.5, табл. leakage check | `python src/crystal/retrieve_crystal.py --K 10 --bootstrap --spatial_split 12` |

Все таблицы воспроизводятся за единицы минут на CPU.

## Уровень 2: воспроизведение от чекпоинтов

```bash
python scripts/fetch_artifacts.py --level 2
```

Дополнительно скачивает:
- `tier2_sem_checkpoints.tar.gz` — 3 SEM-чекпоинта (~321/321/528 МБ, см. [Colab results/ATTRIBUTION.md](https://github.com/daniilctrl/sem-image-analysis/releases/download/vkr-submission/tier2_sem_checkpoints.tar.gz) внутри архива для деталей атрибуции τ)
- `tier2_crystal_checkpoint.tar.gz` — Crystal SimCLR best
- `tier2_sem_tiles.zip` — 22 791 PNG-тайл $224 \times 224$ + `tiles_metadata.csv`

Регенерация эмбеддингов:

```bash
# SEM (3 SSL модели)
python src/models/deep_clustering/extract_simclr_embeddings.py \
    --checkpoint models/april/simclr_tau0.2_best.pth \
    --output_dir data/embeddings/simclr_t02 --model_type simclr
python src/models/deep_clustering/extract_simclr_embeddings.py \
    --checkpoint models/april/simclr_tau0.5_best.pth \
    --output_dir data/embeddings/simclr --model_type simclr
python src/models/deep_clustering/extract_simclr_embeddings.py \
    --checkpoint models/april/byol_resnet50_best.pth \
    --output_dir data/embeddings/byol --model_type byol

# Baseline ImageNet (одна команда)
python src/models/deep_clustering/extract_simclr_embeddings.py \
    --checkpoint IMAGENET --output_dir data/embeddings/baseline

# Crystal
python src/crystal/extract_crystal_embeddings.py \
    --checkpoint models/crystal/crystal_simclr_best.pth \
    --patch_dir data/crystal/patches \
    --output_dir data/crystal/embeddings
```

Полученные `.npy` бит-в-бит совпадут с tier 1 при одинаковом seed (42).

## Уровень 3: полная цепочка с нуля

SEM training остаётся в Colab (нужны 22 791 тайл + ImageNet pretrained
ResNet50, что превышает локальный SSD среднего разработчика). Notebook:
`notebooks/SimCLR_Colab.ipynb` — содержит ячейки обучения SimCLR и BYOL
с сохранением чекпоинтов и логов.

Crystal patches регенерируются локально:

```bash
python scripts/fetch_artifacts.py --level 3   # bcc_data.csv
python src/crystal/patch_generator.py \
    --bcc_path data/crystal/bcc_data.csv \
    --output_dir data/crystal/patches \
    --window_radius 10.0
# Результат: patches.npy (149 947 × 5 × 32 × 32) + patches_metadata.csv

python src/crystal/train_crystal.py --epochs 50 --batch_size 64 \
    --output_dir models/crystal_retrain
```

## Что НЕ публикуется

- **Сырые SEM TIFF-снимки (172 файла, SiC лаборатории)** — не подлежат
  публикации. Все воспроизведения опираются на пред-нарезанные тайлы
  $224 \times 224$ (уровень 2).
- **`crystal_data_v1.zip` MetalDAM** — лежит локально, но к работе не относится
  (рассматривался как public alternative и отброшен).

## Атрибуция SimCLR τ

Локально и в чекпоинтах апрельского прогона исходные имена файлов
не содержали значение τ. Атрибуция выполнена по `best_loss` чекпоинта
(NT-Xent loss при сходимости ∝ exp(−Δ/τ), что даёт устойчивое разделение)
и **верифицирована end-to-end** сверкой probe-метрик с эталонным отчётом
`sem_eval_report_2026-04-17_2122.md`. Детали в `ATTRIBUTION.md` внутри
архива tier2.

## Глобальные seeds

```python
from src.utils.repro import set_global_seed
set_global_seed(42)   # torch, numpy, random, cudnn
```

Известные источники недетерминированности:
- CUDA-операции (cuDNN не полностью детерминирован)
- KDTree neighbor ordering при равных расстояниях (Crystal patch_generator)
- FAISS IndexFlatIP tie-breaking

При фиксированном seed=42 числа в таблицах ВКР воспроизводятся
до 3-4 знаков после запятой; финальная агрегация (median) и бутстреп-CI
стабильны.

## Связанные файлы

- `experiment_manifest.yaml` — полные гиперпараметры, пути, версии пакетов
- `scripts/fetch_artifacts.py` — скрипт скачивания с трёхуровневой логикой
- `data/results/colab_repro_v2/sem_eval_report_2026-05-12_1846.md` — финальный
  отчёт оценки, на который опираются числа Главы 3 ВКР
- `data/results/crystal_leakage_check/leakage_summary_overall.csv` — числа
  для табл. `tab:crystal_leakage` Главы 2.5
- `Colab results/ATTRIBUTION.md` (в архиве tier2) — лог атрибуции τ
