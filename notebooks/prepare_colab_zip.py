"""
Упаковщик данных для Google Colab.

Режимы:
  sem      — src/ + data/processed/ + requirements.txt (текущее поведение).
             Артефакт: diploma_colab.zip.
  crystal  — data/crystal/splits/ + data/crystal/patches/patches_metadata.csv.
             По умолчанию НЕ включает patches.npy (~2.9 GB) — он
             заливается в Drive один раз и переиспользуется. Флаг
             --include-patches добавляет его, если zip нужен «с нуля».
             Артефакт: crystal_data_v1.zip.
  both     — оба архива.

Notebook (simclr_pipeline.ipynb / crystal-cell): `src/` подтягивается
через git clone, поэтому в crystal-zip он не нужен. SEM-режим оставлен
обратно совместимым: `src/` всё ещё попадает в diploma_colab.zip.

Использование:
  python notebooks/prepare_colab_zip.py                    # sem (default)
  python notebooks/prepare_colab_zip.py --mode crystal
  python notebooks/prepare_colab_zip.py --mode crystal --include-patches
  python notebooks/prepare_colab_zip.py --mode both
"""
import argparse
import os
import zipfile
from pathlib import Path

from tqdm import tqdm


def _count_files(paths):
    total = 0
    for p in paths:
        if p.is_file():
            total += 1
        elif p.is_dir():
            for _, _, files in os.walk(p):
                total += len(files)
    return total


def _add_path(zipf, path: Path, root_dir: Path, pbar):
    if path.is_file():
        zipf.write(path, path.relative_to(root_dir))
        pbar.update(1)
        return
    for r, _dirs, files in os.walk(path):
        for f in files:
            if "__pycache__" in r or f.endswith(".pyc"):
                pbar.update(1)
                continue
            fp = Path(r) / f
            zipf.write(fp, fp.relative_to(root_dir))
            pbar.update(1)


def zip_paths(paths, output_path: Path, root_dir: Path, label: str):
    paths = [p for p in paths if p.exists()]
    total = _count_files(paths)
    if total == 0:
        print(f"[{label}] no files to archive — skipping {output_path.name}")
        return False

    print(f"[{label}] archiving {total} files into {output_path.name}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total, desc=f"Zipping {label}") as pbar:
            for p in paths:
                _add_path(zipf, p, root_dir, pbar)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[{label}] done: {output_path} ({size_mb:.1f} MB)")
    return True


def build_sem(root_dir: Path) -> bool:
    return zip_paths(
        [
            root_dir / "src",
            root_dir / "data" / "processed",
            root_dir / "requirements.txt",
        ],
        root_dir / "diploma_colab.zip",
        root_dir,
        "SEM",
    )


def build_crystal(root_dir: Path, include_patches: bool) -> bool:
    paths = [
        root_dir / "data" / "crystal" / "splits",
        root_dir / "data" / "crystal" / "patches" / "patches_metadata.csv",
    ]
    if include_patches:
        paths.append(root_dir / "data" / "crystal" / "patches" / "patches.npy")
    else:
        npy = root_dir / "data" / "crystal" / "patches" / "patches.npy"
        if npy.exists():
            print(
                f"[crystal] excluding {npy.relative_to(root_dir)} "
                f"({npy.stat().st_size / 1e9:.1f} GB) — pass --include-patches "
                "to add it"
            )
    return zip_paths(
        paths,
        root_dir / "crystal_data_v1.zip",
        root_dir,
        "crystal",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sem", "crystal", "both"], default="sem")
    parser.add_argument("--include-patches", action="store_true",
                        help="In crystal mode, also pack patches.npy (~2.9 GB).")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]

    if args.mode in ("sem", "both"):
        build_sem(root_dir)
    if args.mode in ("crystal", "both"):
        build_crystal(root_dir, include_patches=args.include_patches)

    print(
        "\nUpload the resulting .zip to your Google Drive (e.g. "
        "MyDrive/diploma_data/) and unzip into the project root in Colab."
    )


if __name__ == "__main__":
    main()
