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
    "SimCLR (30 ep)": "simclr/finetuned_embeddings.npy",
    "BYOL (30 ep)": "byol/finetuned_embeddings.npy",
}

MODEL_NAMES_FILES = {
    "Baseline (ImageNet)": "embedding_names.csv",
    "SimCLR (30 ep)": "simclr/finetuned_embedding_names.csv",
    "BYOL (30 ep)": "byol/finetuned_embedding_names.csv",
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
) -> tuple[np.ndarray, pd.DataFrame]:
    """Загружает эмбеддинги и метаданные с гарантированным выравниванием.

    Порядок строк в .npy файле может не совпадать с tiles_metadata.csv.
    Для каждой модели используется свой names-файл (bridge):
      - Baseline: embedding_names.csv
      - SimCLR:   simclr/finetuned_embedding_names.csv
      - BYOL:     byol/finetuned_embedding_names.csv

    Алгоритм:
      1. Загрузить model-specific names file (row_idx → tile_name)
      2. Загрузить tiles_metadata.csv
      3. Inner merge по tile_name
      4. Извлечь aligned подмножество строк из embeddings

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
        # Fallback: try root embedding_names.csv
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
                f"Cannot verify embedding ↔ metadata alignment."
            )
    names_df = pd.read_csv(names_path)

    # 3. Load metadata (raw — without dedup, since embeddings may correspond
    #    to all rows including duplicates)
    meta_raw = load_metadata(meta_path)
    meta_dedup = meta_raw.drop_duplicates(subset=["tile_name"])

    if len(names_df) == len(embeddings):
        # CASE A: embedding_names is up-to-date — use it as the bridge.
        # Merge with deduped metadata (each tile_name appears once).
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
    elif len(meta_raw) == len(embeddings):
        # CASE B: embedding_names is stale, but raw metadata count matches
        # embeddings. Assume row-order alignment (embeddings[i] ↔ meta.iloc[i]).
        import warnings
        warnings.warn(
            f"embedding_names.csv ({len(names_df)}) != embeddings ({len(embeddings)}). "
            f"Raw metadata count ({len(meta_raw)}) matches embeddings — assuming "
            f"index alignment. Regenerate embedding_names.csv to restore verified "
            f"alignment.",
            UserWarning,
            stacklevel=2,
        )
        embeddings_aligned = embeddings
        df_aligned = meta_raw.reset_index(drop=True)
        print(
            f"  WARNING: embedding_names.csv stale ({len(names_df)} rows vs "
            f"{len(embeddings)} embeddings). Using index-aligned metadata "
            f"({len(meta_raw)} rows)."
        )
    else:
        raise ValueError(
            f"Cannot align data: embedding_names ({len(names_df)}), "
            f"embeddings ({len(embeddings)}), metadata_raw ({len(meta_raw)}), "
            f"metadata_dedup ({len(meta_dedup)}) — no pair matches. "
            f"Regenerate embeddings or metadata."
        )

    return embeddings_aligned, df_aligned


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
