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

## Запуск в Colab

GPU нужен только на шаге 1 и 2. Шаги 3-4 локальные.

### 1. Тренинг SimCLR на train-секторах

```bash
python -m src.crystal.train_crystal \
    --epochs 50 \
    --batch_size 64 \
    --output_dir models/crystal_holdout \
    --split_csv data/crystal/splits/region_holdout_v1.csv \
    --split_filter train
```

`CrystalPatchDataset` оставит ~99k патчей вместо 149k. Время на T4 —
сравнимо с обычным crystal-SimCLR run (примерно 60-70% от full-train,
т.к. меньше итераций на эпоху).

### 2. Извлечение эмбеддингов на ВСЕХ патчах

```bash
python -m src.crystal.extract_crystal_embeddings \
    --checkpoint models/crystal_holdout/crystal_simclr_best.pth \
    --output_dir data/crystal/embeddings_holdout
```

`extract_crystal_embeddings.py` без `--split_csv` извлекает на всех 150k
патчах — это правильно: candidate-pool в retrieval должен быть полный,
ограничение касается только query-множества.

### 3. Retrieval (локально)

```bash
# (a) Query=test против полного pool, без spatial split
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

### 4. Сравнение

В таблицу `leakage_summary.csv` добавятся 2 новые строки. Можно расширить
`src/crystal/aggregate_leakage_check.py`, чтобы он автоматически собрал
6 точек: SimCLR-fulltrain × {0°, 12°}, SimCLR-holdout × {0°, 12°},
flatten × {0°, 12°}.

Ожидаемая интерпретация:
- Если `simclr_holdout` ≈ `simclr_fulltrain` под 12° split → переноса
  признаков достаточно, training-side leakage не существенен.
- Если `simclr_holdout` ≪ `simclr_fulltrain` → признаки region-specific,
  и часть P@K в текущей таблице действительно объясняется memorization.

## Заметки

- Сплит детерминирован: те же patch_idx всегда попадают в те же сектора
  (центроид и базис в `assign_regions` — функция от данных, без RNG).
  Можно безопасно регенерировать через `scripts/build_crystal_region_split.py`.
- Если хочется более строгого варианта: `--n_sectors 8 --test_sectors 5 6 7`
  даёт 5/8 train, что ещё агрессивнее (но меньше train data для SimCLR).
