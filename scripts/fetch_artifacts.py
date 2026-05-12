"""Fetch reproducibility artifacts from the GitHub Release.

Three tiers, увеличивающие глубину воспроизведения:

  --level 1  Embeddings only (~700 MB). Достаточно, чтобы воспроизвести
             все таблицы Главы 2-3 ВКР без GPU за минуты.

  --level 2  Embeddings + checkpoints + SEM tiles (~2 GB). Позволяет
             перегенерировать эмбеддинги из чекпоинтов на своём GPU и
             получить идентичные числа.

  --level 3  Всё вышеуказанное + исходные координаты BCC-атомов
             (Crystal patches пересчитываются из них скриптом
             patch_generator.py).

Сырые SEM TIFF не публикуются (данные лаборатории, NDA).

Использование:
    python scripts/fetch_artifacts.py --level 1
    python scripts/fetch_artifacts.py --level 2 --tag vkr-submission
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = "daniilctrl/sem-image-analysis"
DEFAULT_TAG = "vkr-submission"

# (asset_name_in_release, extract_into_repo_path)
ARTIFACTS: dict[int, list[tuple[str, str]]] = {
    1: [
        ("tier1_sem_embeddings.tar.gz", "data/embeddings"),
        ("tier1_crystal_embeddings.tar.gz", "data/crystal/embeddings"),
    ],
    2: [
        ("tier2_sem_checkpoints.tar.gz", "models/checkpoints"),
        ("tier2_crystal_checkpoint.tar.gz", "models/crystal"),
        ("tier2_sem_tiles.tar.gz", "data/processed"),
    ],
    3: [
        ("tier3_crystal_atoms.tar.gz", "data/crystal"),
    ],
}


def _ensure_gh() -> None:
    if shutil.which("gh") is None:
        sys.exit(
            "ERROR: GitHub CLI ('gh') is required.\n"
            "Install: https://cli.github.com/  (or 'winget install GitHub.cli')\n"
            "Then run:  gh auth login"
        )


def _download(tag: str, asset: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / asset
    if archive.exists():
        print(f"  [skip] {asset} already downloaded")
        return archive
    print(f"  [get]  {asset}")
    subprocess.run(
        ["gh", "release", "download", tag, "--repo", REPO,
         "--pattern", asset, "--dir", str(dest_dir), "--clobber"],
        check=True,
    )
    return archive


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  [tar]  {archive.name} -> {target.relative_to(Path.cwd())}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--cache-dir", default=".cache/artifacts")
    p.add_argument("--keep-archives", action="store_true",
                   help="Don't delete .tar.gz after extraction (default: delete)")
    args = p.parse_args()

    _ensure_gh()
    repo_root = Path(__file__).resolve().parents[1]
    cache = repo_root / args.cache_dir

    tiers = [t for t in (1, 2, 3) if t <= args.level]
    print(f"Fetching tiers {tiers} from {REPO}@{args.tag}\n")

    for tier in tiers:
        print(f"--- tier {tier} ---")
        for asset, target_rel in ARTIFACTS[tier]:
            archive = _download(args.tag, asset, cache)
            _extract(archive, repo_root / target_rel)
            if not args.keep_archives:
                archive.unlink()

    print("\nDone. Verify with:  python tests/test_smoke.py")


if __name__ == "__main__":
    main()
