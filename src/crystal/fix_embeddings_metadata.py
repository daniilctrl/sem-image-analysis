"""
Починка несоответствия embeddings_metadata.csv и crystal_embeddings.npy.

Проблема:
  - crystal_embeddings.npy: 149947 строк (актуальный)
  - embeddings_metadata.csv: 113922 строк (старый, от предыдущего запуска extract_crystal_embeddings.py)
  - patches_metadata.csv: 149947 строк (правильный источник координат)

Решение:
  1. Берём patches_metadata.csv как базу (149947 строк, правильный 1-к-1 с embeddings.npy)
  2. Перезапускаем KMeans для всех k из списка
  3. Сохраняем как новый embeddings_metadata.csv

Запуск:
  python src/crystal/fix_embeddings_metadata.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

_root = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = _root / "data" / "crystal" / "embeddings" / "crystal_embeddings.npy"
PATCHES_META_PATH = _root / "data" / "crystal" / "patches" / "patches_metadata.csv"
OUTPUT_PATH = _root / "data" / "crystal" / "embeddings" / "embeddings_metadata.csv"

# Те же k, что в optimize_clusters.py / extract_crystal_embeddings.py
K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]


def main():
    print("=== fix_embeddings_metadata.py ===\n")

    # 1. Загрузка
    print(f"Загрузка эмбеддингов: {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode='r')
    print(f"  Форма: {embeddings.shape}")

    print(f"\nЗагрузка метаданных патчей: {PATCHES_META_PATH}")
    meta_df = pd.read_csv(PATCHES_META_PATH)
    print(f"  Строк: {len(meta_df)}")

    # Проверка соответствия
    assert len(embeddings) == len(meta_df), (
        f"КРИТИЧЕСКАЯ ОШИБКА: embeddings {len(embeddings)} != patches_metadata {len(meta_df)}\n"
        f"Требуется перегенерация патчей или эмбеддингов."
    )
    print(f"\nOK: количество строк совпадает: {len(embeddings)}")

    # 2. KMeans для каждого k
    # MiniBatchKMeans быстрее на 150k строк × 512D
    print(f"\nЗапуск KMeans для k = {K_VALUES}...")
    print("  (используется MiniBatchKMeans для скорости)\n")

    # Материализуем в RAM для sklearn (mmap не поддерживается всеми операциями)
    emb_array = np.array(embeddings, dtype=np.float32)

    for k in K_VALUES:
        print(f"  k={k}...", end=" ", flush=True)
        km = MiniBatchKMeans(
            n_clusters=k,
            n_init=5,
            random_state=42,
            batch_size=4096,
        )
        labels = km.fit_predict(emb_array)
        meta_df[f"cluster_{k}"] = labels

        # Быстрый silhouette на сэмпле
        sil = silhouette_score(
            emb_array, labels,
            metric="cosine",
            sample_size=min(5000, len(emb_array)),
            random_state=42,
        )
        print(f"done  (silhouette={sil:.4f})")

    # 3. Сохранение
    meta_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOK: сохранено: {OUTPUT_PATH}")
    print(f"  Строк: {len(meta_df)}, столбцов: {meta_df.shape[1]}")
    print(f"  Столбцы: {list(meta_df.columns)}")

    # Отдельно сохраняем labels для удобства
    for k in K_VALUES:
        labels_path = OUTPUT_PATH.parent / f"cluster_labels_{k}.npy"
        np.save(labels_path, meta_df[f"cluster_{k}"].values)
    print(f"\nOK: cluster_labels_*.npy сохранены в {OUTPUT_PATH.parent}")

    print("\n=== Готово. Теперь можно запускать visualize_crystal.py и analyze_miller_indices.py ===")


if __name__ == "__main__":
    main()
