#!/usr/bin/env python
"""
Копирует свежесгенерированные картинки Crystal в text.diploma/spbu_diploma/images/
под именами, на которые ссылаются main.tex ВКР и оба отчёта по практикам.

Структура исходников: для k-зависимых картинок результаты лежат
в подкаталогах analysis/k{N}/, как принято в проекте.

Карта соответствий
------------------
data/crystal/patches/patch_channels.png          → crystal_patch_channels.png
data/crystal/analysis/cluster_optimization_plot.png
                                                 → crystal_cluster_metrics.png
data/crystal/analysis/k{N}/miller_crosstab_heatmap.png
                                                 → crystal_miller_heatmap_k{N}.png
data/crystal/analysis/k{N}/crystal_retrieval_K{K}.png
                                                 → crystal_retrieval_k{N}.png
data/crystal/visualizations/clusters/umap_clusters_k{N}.png
                                                 → crystal_umap_k{N}.png
data/crystal/analysis/k{N}/cluster_patch_examples.png
                                                 → crystal_patch_examples_k{N}.png
data/crystal/visualizations/clusters/surface_clusters_k{N}_perspective.png
                                                 → crystal_clusters_3d_k{N}.png

(N — число кластеров KMeans, K — число ближайших соседей retrieval)

Запуск
------
    cd C:\\projects\\diploma
    python scripts/copy_crystal_images.py                     # n_clusters=35, K=10
    python scripts/copy_crystal_images.py --dry-run           # посмотреть, ничего не трогая
    python scripts/copy_crystal_images.py --n_clusters 50     # если k-sweep дал k=50
    python scripts/copy_crystal_images.py --flat              # если файлы лежат прямо в analysis/

Бекап старых картинок (с расширением .bak.png) делается автоматически.
Отключить бекап: --no-backup.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DIPLOMA_ROOT = Path(__file__).resolve().parent.parent       # C:\projects\diploma
THESIS_IMAGES = Path(r"C:\projects\text.diploma\spbu_diploma\images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_clusters", type=int, default=35,
                        help="Число кластеров KMeans (для имени heatmap и retrieval)")
    parser.add_argument("--K", type=int, default=10,
                        help="Число соседей retrieval (для имени исходного файла)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать, что будет скопировано")
    parser.add_argument("--no-backup", action="store_true",
                        help="Не создавать .bak.png бекапы существующих картинок")
    parser.add_argument("--thesis-images-dir", type=Path, default=THESIS_IMAGES,
                        help="Путь к папке images/ с картинками ВКР")
    parser.add_argument("--flat", action="store_true",
                        help="Брать heatmap и retrieval из data/crystal/analysis/, "
                             "а не из подкаталога k{N}/ (используется если в "
                             "проекте нет k-разделённой структуры)")
    args = parser.parse_args()

    src_root = DIPLOMA_ROOT / "data" / "crystal"
    dst_root = args.thesis_images_dir

    if args.flat:
        analysis_k_dir = src_root / "analysis"
    else:
        analysis_k_dir = src_root / "analysis" / f"k{args.n_clusters}"

    # visualize_crystal.py сохраняет UMAP и статические 3D-виды
    # в data/crystal/visualizations/clusters/
    viz_clusters_dir = src_root / "visualizations" / "clusters"

    mapping: list[tuple[Path, Path]] = [
        (src_root / "patches"  / "patch_channels.png",
         dst_root / "crystal_patch_channels.png"),
        (src_root / "analysis" / "cluster_optimization_plot.png",
         dst_root / "crystal_cluster_metrics.png"),
        (analysis_k_dir / "miller_crosstab_heatmap.png",
         dst_root / f"crystal_miller_heatmap_k{args.n_clusters}.png"),
        (analysis_k_dir / f"crystal_retrieval_K{args.K}.png",
         dst_root / f"crystal_retrieval_k{args.n_clusters}.png"),
        # UMAP-проекция эмбеддингов с раскраской по кластерам
        (viz_clusters_dir / f"umap_clusters_k{args.n_clusters}.png",
         dst_root / f"crystal_umap_k{args.n_clusters}.png"),
        # Примеры патчей по кластерам (визуальный контроль)
        (analysis_k_dir / "cluster_patch_examples.png",
         dst_root / f"crystal_patch_examples_k{args.n_clusters}.png"),
        # 3D-визуализация полусферы с раскраской по кластерам KMeans
        # (после обучения) — берём ракурс «перспектива».
        (viz_clusters_dir / f"surface_clusters_k{args.n_clusters}_perspective.png",
         dst_root / f"crystal_clusters_3d_k{args.n_clusters}.png"),
    ]

    print(f"Diploma root:  {DIPLOMA_ROOT}")
    print(f"Thesis images: {dst_root}")
    print(f"n_clusters={args.n_clusters}, K={args.K}")
    if args.dry_run:
        print("DRY-RUN: ни один файл не будет изменён\n")
    else:
        print()

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    for src, dst in mapping:
        if not src.exists():
            print(f"[SKIP] исходник не найден: {src}")
            skipped += 1
            continue

        if dst.exists() and not args.no_backup:
            backup = dst.with_name(dst.stem + ".bak.png")
            if args.dry_run:
                print(f"[DRY-BAK] {dst.name} → {backup.name}")
            else:
                shutil.copy2(dst, backup)
                print(f"[BACKUP] {dst.name} → {backup.name}")

        if args.dry_run:
            print(f"[DRY-CP ] {src} → {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"[COPY  ] {src.name} → {dst.name}")
        ok += 1

    print()
    print(f"Готово: скопировано {ok}, пропущено {skipped} из {len(mapping)}")


if __name__ == "__main__":
    main()
