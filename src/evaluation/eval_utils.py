"""
Shared utilities for SEM evaluation pipeline.

Centralized code that was previously duplicated across:
  - cross_scale_retrieval.py
  - cross_scale_comparison.py
  - scale_invariance_metrics.py
  - evaluate_sic_clustering.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Project root (two levels up from src/evaluation/)
_ROOT = Path(__file__).resolve().parents[2]

# Default paths
DEFAULT_META_PATH = _ROOT / "data" / "processed" / "tiles_metadata.csv"
DEFAULT_EMB_DIR = _ROOT / "data" / "embeddings"
DEFAULT_OUTPUT_DIR = _ROOT / "data" / "results"

# Standard model embedding files and their corresponding names files.
# names_file maps embedding row index → tile_name for alignment with metadata.
MODEL_CONFIGS = {
    "Baseline (ImageNet)": "resnet50_embeddings.npy",
    "SimCLR (τ=0.5)": "simclr/finetuned_embeddings.npy",
    "SimCLR (τ=0.2)": "simclr_t02/finetuned_embeddings.npy",
    "BYOL (100 ep)": "byol/finetuned_embeddings.npy",
}

MODEL_NAMES_FILES = {
    "Baseline (ImageNet)": "embedding_names.csv",
    "SimCLR (τ=0.5)": "simclr/finetuned_embedding_names.csv",
    "SimCLR (τ=0.2)": "simclr_t02/finetuned_embedding_names.csv",
    "BYOL (100 ep)": "byol/finetuned_embedding_names.csv",
}


def build_faiss_index(embeddings: np.ndarray):
    """Строит FAISS cosine-similarity индекс (L2-normalized + inner product).

    Аргументы:
        embeddings: (N, D) матрица эмбеддингов.

    Возвращает:
        index: faiss.IndexFlatIP
        normalized: (N, D) np.ndarray, float32 — нормализованные эмбеддинги.
    """
    import faiss

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (embeddings / norms).astype("float32")
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


def extract_material(source_image: pd.Series) -> pd.Series:
    """Извлекает название материала из source_image.

    Формат source_image: '{material}__{image_id}'
    """
    return source_image.str.split("__").str[0]


def load_model_embeddings(
    model_name: str,
    emb_dir: Path | str | None = None,
) -> np.ndarray:
    """Загружает эмбеддинги модели по имени.

    Аргументы:
        model_name: Ключ из MODEL_CONFIGS или прямой путь к .npy файлу.
        emb_dir: Директория с эмбеддингами (по умолчанию DEFAULT_EMB_DIR).

    Возвращает:
        np.ndarray — (N, D) матрица эмбеддингов.
    """
    if emb_dir is None:
        emb_dir = DEFAULT_EMB_DIR
    emb_dir = Path(emb_dir)

    if model_name in MODEL_CONFIGS:
        path = emb_dir / MODEL_CONFIGS[model_name]
    else:
        path = emb_dir / model_name

    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")

    return np.load(path)


def load_metadata(meta_path: Path | str | None = None) -> pd.DataFrame:
    """Загружает метаданные тайлов.

    Аргументы:
        meta_path: Путь к tiles_metadata.csv (по умолчанию DEFAULT_META_PATH).

    Возвращает:
        pd.DataFrame с валидированными столбцами.
    """
    if meta_path is None:
        meta_path = DEFAULT_META_PATH
    meta_path = Path(meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    df = pd.read_csv(meta_path)

    required_cols = {"tile_name", "source_image", "mag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    return df


def load_aligned_data(
    model_name: str,
    emb_dir: Path | str | None = None,
    meta_path: Path | str | None = None,
    names_file: str | None = None,
    allow_fallback: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Загружает эмбеддинги и метаданные с гарантированным выравниванием.

    Порядок строк в .npy файле не совпадает с tiles_metadata.csv. Для каждой
    модели используется свой names-файл (bridge):
      - Baseline: embedding_names.csv
      - SimCLR:   simclr/finetuned_embedding_names.csv
      - BYOL:     byol/finetuned_embedding_names.csv

    Алгоритм:
      1. Загрузить model-specific names file (row_idx -> tile_name)
      2. Загрузить tiles_metadata.csv
      3. Проверить строгое соответствие len(names) == len(embeddings)
      4. Inner merge по tile_name
      5. Извлечь aligned подмножество строк из embeddings

    Раньше здесь был fallback «index-aligned raw metadata» на случай, когда
    `embedding_names.csv` устарел. Он опасен: две несовместимые выборки
    могут случайно совпасть по размеру и дать молча неверные метки. Теперь
    такой fallback требует явного `allow_fallback=True` — используется только
    в переходный период; для production следует регенерировать names-файл
    (см. scripts/regenerate_embedding_names.py).

    Аргументы:
        model_name: Ключ из MODEL_CONFIGS или прямой путь к .npy файлу.
        emb_dir: Директория с эмбеддингами.
        meta_path: Путь к tiles_metadata.csv.
        names_file: Явный путь к names-файлу (override).
        allow_fallback: Если True, разрешает index-alignment при устаревшем
            names-файле, но только при len(meta_raw) == len(embeddings).
            Выводит громкое предупреждение.

    Возвращает:
        (embeddings_aligned, df_aligned) — оба имеют одинаковое число строк,
        строго выровненных по tile_name.
    """
    if emb_dir is None:
        emb_dir = DEFAULT_EMB_DIR
    emb_dir = Path(emb_dir)

    # 1. Load embeddings
    embeddings = load_model_embeddings(model_name, emb_dir)

    # 2. Resolve model-specific names file
    if names_file is None:
        names_file = MODEL_NAMES_FILES.get(model_name, "embedding_names.csv")
    names_path = emb_dir / names_file
    if not names_path.exists():
        fallback_path = emb_dir / "embedding_names.csv"
        if fallback_path.exists() and fallback_path != names_path:
            import warnings
            warnings.warn(
                f"Model-specific names file {names_path} not found. "
                f"Falling back to {fallback_path}.",
                UserWarning, stacklevel=2,
            )
            names_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Names file not found: {names_path}. "
                f"Cannot verify embedding <-> metadata alignment. "
                f"Run scripts/regenerate_embedding_names.py."
            )
    names_df = pd.read_csv(names_path)

    meta_raw = load_metadata(meta_path)
    meta_dedup = meta_raw.drop_duplicates(subset=["tile_name"])

    if len(names_df) == len(embeddings):
        # Канонический путь: embedding_names -> inner merge with metadata.
        names_df["_emb_idx"] = range(len(names_df))
        merged = names_df.merge(meta_dedup, on="tile_name", how="inner")

        if len(merged) == 0:
            raise ValueError(
                "No matching tile_names between embedding_names.csv and metadata. "
                "Check that both files refer to the same dataset."
            )

        aligned_indices = merged["_emb_idx"].values
        embeddings_aligned = embeddings[aligned_indices]
        df_aligned = merged.drop(columns=["_emb_idx"]).reset_index(drop=True)

        coverage = len(merged) / len(names_df) * 100
        print(
            f"  Aligned via embedding_names.csv: {len(merged)} tiles "
            f"({coverage:.1f}% of {len(names_df)} embeddings matched metadata)"
        )
        return embeddings_aligned, df_aligned

    if allow_fallback and len(meta_raw) == len(embeddings):
        import warnings
        warnings.warn(
            f"embedding_names.csv ({len(names_df)}) != embeddings ({len(embeddings)}), "
            f"but len(meta_raw)={len(meta_raw)} matches. Assuming index alignment "
            f"because allow_fallback=True. This is OPAQUE and may silently produce "
            f"wrong labels if row order of embeddings differs from metadata. "
            f"Regenerate names file via scripts/regenerate_embedding_names.py.",
            UserWarning,
            stacklevel=2,
        )
        print(
            f"  WARNING: using OPAQUE index-alignment fallback "
            f"(names stale: {len(names_df)} vs {len(embeddings)})."
        )
        return embeddings, meta_raw.reset_index(drop=True)

    raise ValueError(
        f"Alignment failed: len(names)={len(names_df)} != len(embeddings)={len(embeddings)}. "
        f"meta_raw={len(meta_raw)}, meta_dedup={len(meta_dedup)}. "
        f"Fix options:\n"
        f"  1. Regenerate names file: python scripts/regenerate_embedding_names.py\n"
        f"  2. Re-extract embeddings: python src/models/deep_clustering/extract_simclr_embeddings.py\n"
        f"  3. If you KNOW row order matches, pass allow_fallback=True (dangerous)."
    )


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-нормализация эмбеддингов (для KMeans в cosine-пространстве).

    Аргументы:
        embeddings: (N, D) матрица.

    Возвращает:
        (N, D) np.ndarray — нормализованные эмбеддинги.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms
