# Region-holdout эксперимент для crystal-retrieval

Закрывает остаточную training-side утечку: текущий `crystal_simclr_best.pth`
обучен на всех 149,947 патчах, поэтому каждый query в retrieval ранее
участвовал в SimCLR-обучении. Region-holdout исключает 2 азимутальных
сектора полусферы из тренинга и оценивает retrieval **только** на этих
секторах. Удержание P@K_miller под этим режимом — сильное доказательство,
что обученное пространство переносит признаки, а не запоминает выборку.

## Артефакты, готовые в репо

- `data/crystal/splits/region_holdout_v1.csv` — split (region 4,5 → test;
  ~99k train / ~50k test).
- `scripts/build_crystal_region_split.py` — генератор сплита, можно
  переразбить (`--n_sectors`, `--test_sectors`).
- `--split_csv` / `--split_filter` в `src/crystal/train_crystal.py` —
  фильтрация патчей при тренинге.
- `--split_csv` / `--restrict_query_split` в `src/crystal/retrieve_crystal.py` —
  ограничение query-множества по split при retrieval.
- `notebooks/prepare_colab_zip.py --mode crystal` — собирает
  `crystal_data_v1.zip` со сплитами и `patches_metadata.csv` (без
  тяжёлого `patches.npy`).

## Шаг 0. Подготовка zip локально

```bash
python notebooks/prepare_colab_zip.py --mode crystal
```

Получится `crystal_data_v1.zip` (~2.5 МБ). Залить его в Drive:
`MyDrive/diploma_data/crystal_data_v1.zip`.

`patches.npy` (~2.9 ГБ) предположительно уже в Drive с прошлых
crystal-runs — путь обычно `MyDrive/diploma_data/crystal/patches.npy`
или `MyDrive/diploma_data/crystal_patches.npy`. Если нет — собрать zip
с `--include-patches` и залить отдельно.

## Шаг 1. Colab cell — подготовка окружения

Эта ячейка повторяет существующий pattern из `simclr_pipeline.ipynb`,
но для crystal-данных. Подставить под свои пути в Drive.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess
PROJ = '/content/diploma_project'

# 1. git clone / pull (чтобы код был свежий)
if os.path.exists(f'{PROJ}/.git'):
    !cd {PROJ} && git pull
else:
    !git clone https://github.com/daniilctrl/sem-image-analysis.git {PROJ}

# 2. Распаковка crystal_data_v1.zip (метаданные + split)
!cp /content/drive/MyDrive/diploma_data/crystal_data_v1.zip /content/
!unzip -qo /content/crystal_data_v1.zip -d {PROJ}/
!rm /content/crystal_data_v1.zip

# 3. patches.npy — НЕ копируем (2.9 GB), а симлинком из Drive
PATCHES_DRIVE = '/content/drive/MyDrive/diploma_data/crystal/patches.npy'   # <-- проверь свой путь
PATCHES_LOCAL = f'{PROJ}/data/crystal/patches/patches.npy'
assert os.path.exists(PATCHES_DRIVE), f"patches.npy not found at {PATCHES_DRIVE} — fix the path"
os.makedirs(os.path.dirname(PATCHES_LOCAL), exist_ok=True)
if not os.path.exists(PATCHES_LOCAL):
    os.symlink(PATCHES_DRIVE, PATCHES_LOCAL)

# 4. Sanity check: все три файла на месте
for f in [
    'data/crystal/patches/patches.npy',
    'data/crystal/patches/patches_metadata.csv',
    'data/crystal/splits/region_holdout_v1.csv',
]:
    p = f'{PROJ}/{f}'
    assert os.path.exists(p), f"missing: {p}"
print("All three crystal artifacts in place.")
```

## Шаг 2. Тренинг SimCLR на train-секторах

```python
RUN_NAME = "crystal_simclr_holdout_v1"

!cd {PROJ} && python -m src.crystal.train_crystal \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --temperature 0.2 \
    --workers 4 \
    --output_dir /content/drive/MyDrive/diploma_checkpoints/{RUN_NAME} \
    --split_csv data/crystal/splits/region_holdout_v1.csv \
    --split_filter train \
    --seed 42
```

`CrystalPatchDataset` при таких флагах оставит ~99,571 патчей вместо
149,947. Время на T4 — примерно 60-70% от full-train run.

## Шаг 3. Извлечение эмбеддингов на ВСЕХ патчах

`extract_crystal_embeddings.py` не имеет `--split_csv` и работает на
полном датасете — это правильно: candidate-pool в retrieval должен
содержать и train-, и test-патчи.

```python
!cd {PROJ} && python -m src.crystal.extract_crystal_embeddings \
    --checkpoint /content/drive/MyDrive/diploma_checkpoints/{RUN_NAME}/crystal_simclr_best.pth \
    --output_dir /content/drive/MyDrive/diploma_embeddings/crystal_holdout \
    --batch_size 256 \
    --workers 4 \
    --seed 42
```

После этой ячейки в Drive лежат:
- `crystal_holdout/crystal_embeddings.npy` (149947, 512)
- `crystal_holdout/embeddings_metadata.csv`
- `crystal_holdout/cluster_labels_*.npy` для k=5,8,10

## Шаг 4. Retrieval (локально)

Скопировать папку `crystal_holdout/` из Drive на локальную машину
в `data/crystal/embeddings_holdout/` (~600 МБ), затем:

```bash
# (a) Query=test, без spatial split (чистый holdout-эффект)
python -m src.crystal.retrieve_crystal \
    --emb_path data/crystal/embeddings_holdout/crystal_embeddings.npy \
    --meta_path data/crystal/embeddings_holdout/embeddings_metadata.csv \
    --split_csv data/crystal/splits/region_holdout_v1.csv \
    --restrict_query_split test \
    --model_name simclr_holdout \
    --output_dir data/results/crystal_leakage_check

# (b) Query=test, spatial split 12° (combo: training-side + eval-side защита)
python -m src.crystal.retrieve_crystal \
    --emb_path data/crystal/embeddings_holdout/crystal_embeddings.npy \
    --meta_path data/crystal/embeddings_holdout/embeddings_metadata.csv \
    --split_csv data/crystal/splits/region_holdout_v1.csv \
    --restrict_query_split test \
    --spatial_split_deg 12 \
    --model_name simclr_holdout \
    --output_dir data/results/crystal_leakage_check
```

После этих двух ран можно расширить `aggregate_leakage_check.py`,
чтобы он автоматически собрал 6 точек: SimCLR-fulltrain × {0°, 12°} +
SimCLR-holdout × {0°, 12°} + flatten × {0°, 12°}.

## Ожидаемая интерпретация

- Если `simclr_holdout` ≈ `simclr_fulltrain` под 12° split → переноса
  признаков достаточно, training-side leakage не существенен — в
  диссертации это финальный аргумент.
- Если `simclr_holdout` ≪ `simclr_fulltrain` → признаки region-specific,
  и часть P@K в текущей таблице действительно объясняется memorization;
  тогда честно репортим обе цифры.

## Заметки

- Сплит детерминирован: те же patch_idx всегда попадают в те же сектора
  (центроид и базис в `assign_regions` — функция от данных, без RNG).
  Можно безопасно регенерировать через `scripts/build_crystal_region_split.py`.
- Если хочется более строгого варианта: `--n_sectors 8 --test_sectors 5 6 7`
  даёт 5/8 train, что ещё агрессивнее (но меньше train data для SimCLR).
- На локальной машине шаг 4 может упереть в RAM (видели это с
  `crystal_embeddings.npy` 512-D × 149947 × 4 = 300 МБ + FAISS-копия).
  Если упёрлось — закрой другие приложения или прогоняй retrieval тоже
  в Colab, импортируя FAISS-cpu.
