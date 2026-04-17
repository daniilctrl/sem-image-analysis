# SEM-ветка: self-supervised representation learning для СЭМ-изображений SiC

## Описание задачи

Контрастивное обучение (SimCLR, BYOL) для получения **масштабно-инвариантных** представлений микрофотографий SiC-поверхностей, снятых на СЭМ с разными увеличениями. Цель — обратный поиск похожих текстур и кластеризация по типу морфологии, не по увеличению.

> Обновлено: 2026-04-17 (после PR #1–#7).

---

## Пайплайн

| Этап | Содержание | Файлы |
|------|-----------|-------|
| 1. Подготовка данных | Нарезка TIFF → тайлы 256×256, парсинг метаданных | `src/data/data_prep.py`, `src/data/update_magnification.py` |
| 2. Каталог для разметки | Фильтрация мусора, метаданные для аннотации | `src/data/generate_sft_labels.py` |
| 3. Ручная разметка | Web-based annotation tool (8 морфологических классов) | `src/data/annotate_tiles.py` |
| 4. Обучение SimCLR | ResNet50 (ImageNet V2 init) + v2 projection head | `src/models/deep_clustering/train.py` |
| 5. Обучение BYOL | ResNet50 + online/target + EMA tau ramp-up | `src/models/deep_clustering/train_byol.py` |
| 6. Извлечение эмбеддингов | Frozen encoder → 2048-D, L2-norm для FAISS | `src/models/deep_clustering/extract_simclr_embeddings.py` |
| 7. Cross-scale retrieval | FAISS cosine, P@K по материалу и кросс-scale | `src/evaluation/cross_scale_retrieval.py` |
| 8. Scale invariance metrics | Cramér's V, AMI, MagVarRatio, Entropy per cluster | `src/evaluation/scale_invariance_metrics.py` |
| 9. Linear probe | Stratified K-fold + LogisticRegression на SFT | `src/evaluation/linear_probe.py` |
| 10. k-NN probe | Weighted cosine k-NN, K∈{1,5,20} | `src/evaluation/knn_probe.py` |
| 11. Unified eval | Все 4 оценки + markdown-отчёт | `src/evaluation/run_sem_evaluation.py` |
| 12. Interactive retrieval | Gradio demo | `src/visualization/app.py` |

---

## Ключевые архитектурные решения

### Projection head — унифицирован с Crystal (v2 с BN)

Раньше SEM SimCLR использовал v1 (`Linear → ReLU → Linear`), а Crystal v2 (`Linear → BN → ReLU → Linear → BN`). Это делало сравнение SEM/Crystal SimCLR нечестным. После PR #3 обе ветки используют **единый модуль `src/models/heads.SimCLRProjectionHead`** с `use_bn=True` по умолчанию. Legacy SEM-чекпоинты (v1) остаются загружаемыми через `SimCLR.from_state_dict()` с auto-detect.

### BYOL с cosine tau ramp-up

Константный `tau=0.996` — это **base** значение из статьи, но оригинальная схема Grill et al. 2020 ramp'ит tau до 1.0 по косинусу. Константный tau тормозит обучение в конце и дестабилизирует в начале. Сейчас default — `--ema_tau_schedule cosine`.

### Checkpoint format — полный dict

Было: сохранялся только `model.state_dict()`. На resume оптимизатор Adam пересоздавался с нуля — первые сотни шагов шли без momentum. Теперь сохраняется полный dict:

```python
{
    "epoch", "model_state_dict",
    "optimizer_state_dict", "scheduler_state_dict",
    "best_loss", "avg_loss", "val_loss",
}
```

Cosine-annealing LR scheduler добавлен в оба SEM-трейнера (в Crystal уже был).

### L2-нормализация везде консистентно

- FAISS retrieval — `IndexFlatIP` на L2-нормализованных эмбеддингах (= cosine).
- KMeans в scale invariance — `l2_normalize` перед кластеризацией.
- Linear probe и k-NN probe — L2-норма до classifier (`--normalize` default True).

### Adaptive mag bins

Было: `MAG_BINS = [0, 100, 1000, 10000, 100000, 1e7]` — хардкод, предполагающий конкретный диапазон (W-эмиттер). На SiC-датасете реальное распределение `mag` может быть другим. Теперь `compute_adaptive_mag_bins()` вычисляет log-квантили реального распределения (`--adaptive_bins` default True, `--n_bins 5`).

### Data alignment и `load_aligned_data`

Каждый `.npy` сопровождается `{model}_embedding_names.csv`, связывающим row-index → `tile_name`. `eval_utils.load_aligned_data()` делает inner-join с `tiles_metadata.csv` по `tile_name` и возвращает выровненные `(embeddings, df)`. При расхождении размеров — **hard-fail** (раньше был opaque fallback на index-alignment, маскирующий баги).

Регенерация names-файла: `python3 scripts/regenerate_embedding_names.py --all`.

---

## Ключевые результаты оценки

Для сравнения трёх моделей (Baseline ResNet50 ImageNet, SimCLR, BYOL) используется **единый entrypoint** `src/evaluation/run_sem_evaluation.py`. Он выдаёт markdown-отчёт с 4 таблицами и bootstrap 95% CI везде, где применимо.

**Secondary-метрики** (intrinsic clustering):
- `Silhouette`, `Calinski-Harabasz`, `Davies-Bouldin`.

**Primary-метрики**:
- `precision@K_material` (P@10 same material) — retrieval качество.
- `precision@K_cross_scale` (P@10 same material, different mag) — scale invariance.
- `linear_probe / knn_probe accuracy` на `sft_annotations.csv` — external supervised eval.

**Secondary, диагностические**:
- `Cramér's V`, `AMI`, `MagVarRatio` — связь кластеров с увеличением (хотим **низкие** значения = кластеры не диктуются масштабом).

---

## Воспроизводимость

- Глобальный seed через `src.utils.repro.set_global_seed(42)` в начале всех train/extract/eval скриптов.
- Фиксируются: `PYTHONHASHSEED`, `random`, `numpy`, `torch` (CPU+CUDA), `cudnn.deterministic`, `torch.use_deterministic_algorithms`.
- DataLoader: `worker_init_fn=seed_worker`, `generator=make_generator(seed)`.
- Bootstrap CI: все bootstrap-функции принимают `seed` — одинаковые CI при повторе.

Не покрывается детерминизмом:
- FAISS IndexFlatIP tie-breaking на equidistant vectors.
- Некоторые CUDA ops при `warn_only=True`.

---

## Colab-специфика

Обучение происходит на Colab Pro (A100/V100). Учитываем:

1. **Checkpoints на Google Drive** — `--output_dir /content/drive/MyDrive/diploma_checkpoints/...`. Disconnect сессии → чекпоинты сохранены.
2. **TensorBoard inline** — `--tb_log_dir /content/drive/MyDrive/diploma_logs/<run_name>`, в ноутбуке `%load_ext tensorboard; %tensorboard --logdir ...`.
3. **Resume должен работать** с drop-in чекпоинтом из Drive: `--resume <path> --start_epoch <N>`. Полный формат чекпоинта гарантирует сохранение optimizer+scheduler state.
4. **Notebook `notebooks/SimCLR_Colab.ipynb`** — точка входа для Colab. Синхронизирован с текущим master (PR #8 в планах).

---

## Открытые вопросы и ограничения

### 1. Размер SFT-разметки (0.8%)

`data/sft_annotations.csv` содержит **184 уникальных тайла** из **23 076 good tiles** (<1% покрытие). Это:

- **Недостаточно для стабильного linear probe/k-NN probe** по 8 классам. CI на таком размере широкие (~±0.10 по accuracy). Результаты носят **indicative** характер.
- **Планируется расширение через active learning + label propagation** (PR #10–#12 в roadmap). Цель — довести до ≥ 500–1000 меток перед финальным сравнением моделей.

### 2. Unified projection head и переобучение

После унификации SEM SimCLR получил **v2 с BN** вместо v1. Legacy SEM-чекпоинты загружаются через adapter, но **новые прогоны в Colab должны быть сделаны с v2 head** (default), чтобы сравнение SEM vs Crystal было на одной архитектуре.

### 3. Пересечение задач с Crystal-веткой

SEM и Crystal используют один `NTXentLoss` (`src/models/deep_clustering/loss.py`) и теперь один `SimCLRProjectionHead`. Любое изменение этих компонентов затрагивает обе ветки — надо учитывать при future changes.

---

## Артефакты

| Файл/папка | Описание |
|------|----------|
| `data/processed/*.png` | Тайлы 256×256 (gitignored) |
| `data/processed/tiles_metadata.csv` | Таблица: tile_name, source_image, mag, x, y |
| `data/sft_catalog.csv` | Каталог всех тайлов + флаг `is_trash` |
| `data/sft_annotations.csv` | Ручная разметка 184 тайлов по 8 классам |
| `data/embeddings/resnet50_embeddings.npy` | Baseline ImageNet ResNet50 (N, 2048) |
| `data/embeddings/simclr/finetuned_embeddings.npy` | SimCLR fine-tuned (N, 2048) |
| `data/embeddings/byol/finetuned_embeddings.npy` | BYOL fine-tuned (N, 2048) |
| `data/embeddings/{model}_embedding_names.csv` | row_idx → tile_name mapping |
| `data/results/sem_eval_report_<ts>.md` | Unified markdown-отчёт после run_sem_evaluation |
| `data/results/cross_scale_ci_{model}.csv` | Per-material bootstrap CI |
| `data/results/linear_probe_summary.csv` | Linear probe cross-model |
| `data/results/knn_probe_summary.csv` | k-NN probe cross-model |
| `models/checkpoints/*.pth` | SimCLR checkpoints (gitignored) |
| `models/checkpoints_byol/*.pth` | BYOL checkpoints (gitignored) |

---

## Как запустить

```bash
# Установка
pip install -e ".[viz,dataprep,dev]"

# Подготовка (одноразово, на локальной машине):
python3 src/data/data_prep.py
python3 src/data/generate_sft_labels.py
python3 src/data/annotate_tiles.py --port 8765  # web UI

# Обучение (Colab Pro, GPU):
python3 src/models/deep_clustering/train.py \
    --epochs 30 --batch_size 64 --temperature 0.5 \
    --val_frac 0.1 --tb_log_dir /content/drive/MyDrive/diploma_logs/simclr_v2_t05

python3 src/models/deep_clustering/train_byol.py \
    --epochs 30 --batch_size 64 --ema_tau_schedule cosine \
    --val_frac 0.1 --tb_log_dir /content/drive/MyDrive/diploma_logs/byol_cosine

# Эмбеддинги:
python3 src/models/deep_clustering/extract_simclr_embeddings.py \
    --checkpoint models/checkpoints/simclr_resnet50_best.pth \
    --output_dir data/embeddings/simclr

# Unified evaluation:
python3 src/evaluation/run_sem_evaluation.py --K 10 --n_clusters 4 --n_folds 5
```
