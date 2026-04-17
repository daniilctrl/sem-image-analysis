# Diploma Project: Deep Learning for Inverse Search and Clustering of Nanostructured Surface Textures

Методы глубокого обучения для обратного поиска и кластеризации текстур наноструктурированных поверхностей по микроскопическим изображениям.

Автор: Даниил Родионов, СПбГУ (бакалавриат).

## Обзор

Проект объединяет две ветки:

**SEM / SiC** — контрастивное обучение (SimCLR, BYOL) + ImageNet baseline на СЭМ-изображениях SiC-поверхностей. Цель: масштабно-инвариантные представления для обратного поиска и кластеризации текстур при разных увеличениях.

**Crystal / BCC** — SimCLR на 5-канальных 32x32 патчах, построенных проецированием атомов BCC-полусферы на локальные касательные плоскости. Цель: выделение зон поверхности (Miller-семейства) без прямых меток.

## Статус

| Ветка | Обучение | Эмбеддинги | Кластеризация | Retrieval | Evaluation |
|---|---|---|---|---|---|
| SEM (Baseline) | ImageNet pretrained | resnet50_embeddings.npy | KMeans | FAISS cosine | cross-scale + scale invariance |
| SEM (SimCLR) | 30 ep (Colab) | simclr/finetuned_embeddings.npy | KMeans | FAISS cosine | cross-scale + scale invariance |
| SEM (BYOL) | 30 ep (Colab) | byol/finetuned_embeddings.npy | KMeans | FAISS cosine | cross-scale + scale invariance |
| Crystal | 50 ep | crystal_embeddings.npy | KMeans (k=8) | FAISS cosine | Miller AMI, Precision@K |

## Структура проекта

```
src/
  crystal/                    # Crystal/BCC pipeline
    patch_generator.py        # KDTree + tangent-plane projection -> 5-ch patches
    dataset_crystal.py        # Dataset for 5-channel .npy patches
    augmentations_crystal.py  # Rotation90, Flip, CropResize, Noise, ChannelDrop
    model_crystal.py          # CrystalSimCLR (ResNet18, 5ch, 32x32)
    train_crystal.py          # Training loop (NT-Xent, CosineAnnealing)
    extract_crystal_embeddings.py  # Embeddings + metadata extraction
    optimize_clusters.py      # KMeans k-selection (Silhouette, DB, Elbow, AMI)
    analyze_miller_indices.py # Miller family vs cluster cross-tabulation
    miller_utils.py           # Single source of truth for Miller classification
    retrieve_crystal.py       # FAISS nearest-neighbor retrieval + Precision@K

  models/deep_clustering/     # SEM/SiC pipeline
    model.py                  # SimCLR (ResNet50, ImageNet pretrained)
    model_byol.py             # BYOL (Online + Target networks, EMA)
    train.py                  # SimCLR training
    train_byol.py             # BYOL training
    loss.py                   # NT-Xent loss
    augmentations.py          # SEM-specific augmentations
    extract_simclr_embeddings.py  # Embedding extraction

  evaluation/                 # SEM evaluation
    eval_utils.py             # Shared utils (FAISS, model configs, normalization)
    cross_scale_comparison.py # Cross-scale retrieval (Precision@K material/scale)
    scale_invariance_metrics.py   # Cramer's V, AMI, MagVarRatio, MagEntropy
    evaluate_sic_clustering.py    # Clustering quality metrics
    run_sem_evaluation.py     # Unified entrypoint for all SEM evaluation

data/
  crystal/patches/            # patches.npy (N,5,32,32) + patches_metadata.csv
  crystal/analysis/           # Embeddings, cluster metrics, Miller crosstab
  processed/                  # SEM tiles + tiles_metadata.csv
  embeddings/                 # Baseline, SimCLR, BYOL embeddings

notebooks/
  SimCLR_Colab.ipynb          # Colab notebook: training + evaluation

tests/
  test_smoke.py               # Data integrity, dimension checks, syntax validation

experiment_manifest.yaml      # Full hyperparameters, seeds, paths, versions
```

## Быстрый старт

```bash
pip install -r requirements.txt

# Smoke-check тесты
python tests/test_smoke.py

# Crystal pipeline
python src/crystal/patch_generator.py --output_dir data/crystal/patches
python src/crystal/train_crystal.py --epochs 50
python src/crystal/extract_crystal_embeddings.py
python src/crystal/optimize_clusters.py
python src/crystal/retrieve_crystal.py --K 10

# SEM evaluation (после обучения в Colab)
python src/evaluation/run_sem_evaluation.py \
    --meta_path data/processed/tiles_metadata.csv \
    --emb_dir data/embeddings \
    --output_dir data/results/ \
    --K 10 --n_clusters 4
```

## Обучение SEM моделей (Colab)

Откройте `notebooks/SimCLR_Colab.ipynb` в Google Colab. Notebook содержит ячейки для SimCLR и BYOL обучения с чекпоинтами на Google Drive, извлечения эмбеддингов, и запуска полного evaluation pipeline.

## Ключевые решения

**k=8 для Crystal кластеров**: обоснование по 4 критериям в `data/crystal/analysis/k_selection_justification.md` (Elbow 2nd derivative, Davies-Bouldin minimum, AMI caveat, physical argument).

**L2-нормализация перед KMeans**: все embedding-based операции (FAISS retrieval, KMeans) используют L2-нормализованные эмбеддинги для консистентности.

**Miller classification tolerance**: 6 deg angular tolerance для 9 BCC-семейств, реализована в единственном модуле `miller_utils.py`.
