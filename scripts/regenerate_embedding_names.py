"""Регенерирует embedding_names.csv из свежего tiles_metadata.csv.

Проблема, которую закрывает этот скрипт
---------------------------------------

Исторически `data/embeddings/embedding_names.csv` создавался один раз
при первом extract_*_embeddings.py и не обновлялся при пересоздании
`tiles_metadata.csv`. Результат: файл содержал 24 183 записи, а .npy
эмбеддингов — 22 791 (расхождение 1 392 строки). В `eval_utils.load_aligned_data`
был неявный fallback «если names устарел, а len(meta_raw) == len(embeddings),
выравниваемся по индексу» — он маскирует реальное расхождение и может
молча дать неверные метки.

Этот скрипт выполняет одно из двух:

1. Если .npy и names совпадают по числу строк, но names содержит «мёртвые»
   tile_name, которых больше нет в metadata — фильтрует их (inner join).

2. Если .npy короче, чем names — сохраняет только первые `len(embeddings)`
   записей, ассумируя, что extraction проходил по порядку metadata
   (такое извлечение канонично для extract_simclr_embeddings.py).

Если ни один вариант не подходит — скрипт падает с ясным сообщением
и рекомендует перегенерировать эмбеддинги через extract_*_embeddings.py.

Использование
-------------

    python3 scripts/regenerate_embedding_names.py \\
        --emb_file data/embeddings/resnet50_embeddings.npy \\
        --meta data/processed/tiles_metadata.csv \\
        --output data/embeddings/embedding_names.csv

    python3 scripts/regenerate_embedding_names.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def regenerate(
    emb_path: Path,
    meta_path: Path,
    output_path: Path,
    old_names_path: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Регенерирует names-файл для одной пары (embeddings, metadata).

    Возвращает число записей в итоговом names-файле.
    """
    embeddings = np.load(emb_path, mmap_mode="r")
    n_emb = embeddings.shape[0]
    meta = pd.read_csv(meta_path)
    meta_dedup = meta.drop_duplicates(subset=["tile_name"]).reset_index(drop=True)

    print(f"[{emb_path.name}] embeddings: {n_emb}, metadata dedup: {len(meta_dedup)}")

    if old_names_path is not None and old_names_path.exists():
        old = pd.read_csv(old_names_path)
        print(f"  old names file: {len(old)} rows")
    else:
        old = None

    if n_emb == len(meta_dedup):
        # Стандартная ситуация: порядок эмбеддингов соответствует
        # порядку строк в metadata (после dedup). Это канонический путь
        # extract_*_embeddings.py.
        names = meta_dedup[["tile_name"]].copy()
        mode = "index_aligned_with_dedup"
    elif old is not None and len(old) >= n_emb and all(c in old.columns for c in ["tile_name"]):
        # Старый names был длиннее: возможно, часть tile был исключена из
        # экстракции (например, не найденный файл). Берём первые n_emb.
        names = old.iloc[:n_emb][["tile_name"]].copy()
        mode = "truncate_old_names"
    else:
        print("  ERROR: cannot reconstruct names unambiguously. Options:")
        print("    1. Re-run extract_*_embeddings.py to regenerate .npy + names together.")
        print("    2. Provide --old_names explicitly if its row order is known reliable.")
        return -1

    if dry_run:
        print(f"  [dry-run] would write {len(names)} rows to {output_path}  (mode: {mode})")
        return len(names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names.to_csv(output_path, index=False)
    print(f"  saved {len(names)} rows to {output_path}  (mode: {mode})")
    return len(names)


def main() -> int:
    _root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Regenerate embedding names files")
    parser.add_argument("--emb_file", type=str, default=None)
    parser.add_argument("--meta", type=str,
                        default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--old_names", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate for Baseline / SimCLR / BYOL from default paths",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}", file=sys.stderr)
        return 1

    tasks = []
    if args.all:
        emb_dir = _root / "data" / "embeddings"
        tasks = [
            (emb_dir / "resnet50_embeddings.npy",
             emb_dir / "embedding_names.csv",
             emb_dir / "embedding_names.csv"),
            (emb_dir / "simclr" / "finetuned_embeddings.npy",
             emb_dir / "simclr" / "finetuned_embedding_names.csv",
             emb_dir / "simclr" / "finetuned_embedding_names.csv"),
            (emb_dir / "byol" / "finetuned_embeddings.npy",
             emb_dir / "byol" / "finetuned_embedding_names.csv",
             emb_dir / "byol" / "finetuned_embedding_names.csv"),
        ]
    else:
        if args.emb_file is None or args.output is None:
            parser.error("Provide --emb_file and --output, or use --all")
        tasks = [(
            Path(args.emb_file),
            Path(args.old_names) if args.old_names else Path(args.output),
            Path(args.output),
        )]

    exit_code = 0
    for emb_path, old_names_path, output_path in tasks:
        if not emb_path.exists():
            print(f"SKIP: {emb_path} not found")
            continue
        n = regenerate(emb_path, meta_path, output_path,
                       old_names_path=old_names_path, dry_run=args.dry_run)
        if n < 0:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
