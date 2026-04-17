"""Run a series of SEM training experiments defined in a YAML sweep config.

Motivation
----------
Запускать серию runs руками в Colab неудобно: каждый требует своего
RUN_NAME, output_dir, tb_log_dir. А когда прогонов 7 (как в
configs/sweeps/sem_default.yaml), это добавляет шанс ошибки и потери
артефактов.

Этот скрипт берёт YAML-конфиг sweep и последовательно запускает каждый
эксперимент через subprocess, вызывая src/models/deep_clustering/train.py
или train_byol.py с нужными CLI-флагами.

Все артефакты (checkpoints, TB logs, извлечённые эмбеддинги, evaluation
отчёт) сохраняются в отдельные папки под `results_root/sweep_{timestamp}/
{experiment_name}/` так, чтобы один прогон не затирал другой.

Features
--------
- `--dry_run` печатает команды без выполнения.
- `--only <name>` запускает один эксперимент из sweep.
- `--skip_extract`, `--skip_evaluate` позволяют разделить stages.
- Resume: если в папке эксперимента есть чекпоинт с номером >= epochs,
  эксперимент пропускается (продолжение после disconnect).

Обычный поток
-------------
В Colab Pro:

    python3 scripts/run_sweep.py \
        --config configs/sweeps/sem_default.yaml \
        --checkpoints_root /content/drive/MyDrive/sweep_ckpts \
        --logs_root /content/drive/MyDrive/sweep_logs \
        --results_root /content/drive/MyDrive/sweep_results \
        --only simclr_t05_default byol_cosine_30ep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SIMCLR = _ROOT / "src" / "models" / "deep_clustering" / "train.py"
TRAIN_BYOL = _ROOT / "src" / "models" / "deep_clustering" / "train_byol.py"
EXTRACT = _ROOT / "src" / "models" / "deep_clustering" / "extract_simclr_embeddings.py"
EVAL = _ROOT / "src" / "evaluation" / "run_sem_evaluation.py"


@dataclass
class SweepContext:
    config_path: Path
    sweep_id: str            # e.g. 'sweep_2026-04-17_1230'
    checkpoints_root: Path
    logs_root: Path
    results_root: Path
    dry_run: bool
    skip_extract: bool
    skip_evaluate: bool


def _resolve_param(exp: dict, defaults: dict, common: dict, key: str, fallback=None):
    if key in exp:
        return exp[key]
    if key in defaults:
        return defaults[key]
    if key in common:
        return common[key]
    return fallback


def build_train_cmd(
    exp: dict,
    defaults: dict,
    common: dict,
    ctx: SweepContext,
) -> list[str]:
    """Собирает команду для train.py или train_byol.py."""
    exp_name = exp["name"]
    exp_type = exp["type"]

    script = TRAIN_SIMCLR if exp_type == "simclr" else TRAIN_BYOL
    output_dir = ctx.checkpoints_root / ctx.sweep_id / exp_name
    tb_log_dir = ctx.logs_root / ctx.sweep_id / exp_name

    def p(key, fallback=None):
        return _resolve_param(exp, defaults, common, key, fallback)

    cmd = [
        sys.executable, str(script),
        "--data_dir", str(p("data_dir")),
        "--metadata_path", str(p("metadata_path")),
        "--output_dir", str(output_dir),
        "--epochs", str(p("epochs")),
        "--batch_size", str(p("batch_size")),
        "--learning_rate", str(p("learning_rate")),
        "--workers", str(p("workers")),
        "--val_frac", str(p("val_frac")),
        "--save_every", str(p("save_every")),
        "--seed", str(p("seed")),
        "--tb_log_dir", str(tb_log_dir),
        # augmentation CLI
        "--crop_scale_min", str(p("crop_scale_min")),
        "--crop_scale_max", str(p("crop_scale_max")),
        "--color_jitter_p", str(p("color_jitter_p")),
        "--brightness", str(p("brightness")),
        "--contrast", str(p("contrast")),
        "--blur_p", str(p("blur_p")),
        "--blur_sigma_min", str(p("blur_sigma_min")),
        "--blur_sigma_max", str(p("blur_sigma_max")),
        "--noise_p", str(p("noise_p")),
        "--noise_std", str(p("noise_std")),
    ]
    if exp_type == "simclr":
        cmd += ["--temperature", str(p("temperature"))]
    else:
        cmd += [
            "--ema_tau", str(p("ema_tau")),
            "--ema_tau_schedule", str(p("ema_tau_schedule")),
        ]
    return cmd


def build_extract_cmd(exp: dict, common: dict, ctx: SweepContext) -> list[str]:
    exp_name = exp["name"]
    exp_type = exp["type"]
    ckpt_dir = ctx.checkpoints_root / ctx.sweep_id / exp_name
    best_ckpt = ckpt_dir / (
        "simclr_resnet50_best.pth" if exp_type == "simclr"
        else "byol_resnet50_best.pth"
    )
    emb_dir = ctx.results_root / ctx.sweep_id / exp_name / "embeddings"
    return [
        sys.executable, str(EXTRACT),
        "--checkpoint", str(best_ckpt),
        "--model_type", exp_type,
        "--data_dir", str(common.get("data_dir", "data/processed")),
        "--metadata_path", str(common.get("metadata_path",
                                          "data/processed/tiles_metadata.csv")),
        "--output_dir", str(emb_dir),
        "--seed", str(common.get("seed", 42)),
    ]


def already_trained(exp: dict, defaults: dict, common: dict, ctx: SweepContext) -> bool:
    exp_name = exp["name"]
    exp_type = exp["type"]
    epochs = int(_resolve_param(exp, defaults, common, "epochs"))
    ckpt_dir = ctx.checkpoints_root / ctx.sweep_id / exp_name
    prefix = "simclr_resnet50" if exp_type == "simclr" else "byol_resnet50"
    last = ckpt_dir / f"{prefix}_epoch_{epochs}.pth"
    return last.exists()


def run_subprocess(cmd: list[str], dry_run: bool) -> int:
    cmd_str = " ".join(cmd)
    print(f"\n$ {cmd_str}\n")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run an SSL sweep from YAML config")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML sweep config")
    parser.add_argument("--checkpoints_root", type=str, default=None,
                        help="Override checkpoints_root from config")
    parser.add_argument("--logs_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Sweep directory name (default: sweep_<timestamp>)")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Only run experiments with these names")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip embedding extraction stage")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Skip run_sem_evaluation stage")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if last checkpoint exists")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    common = cfg.get("common", {})
    defaults = cfg.get("defaults", {})
    experiments = cfg.get("experiments", [])

    if args.only:
        experiments = [e for e in experiments if e["name"] in set(args.only)]
        if not experiments:
            print(f"ERROR: no experiments matched --only {args.only}", file=sys.stderr)
            return 1

    ctx = SweepContext(
        config_path=Path(args.config),
        sweep_id=args.sweep_id or f"sweep_{datetime.now().strftime('%Y-%m-%d_%H%M')}",
        checkpoints_root=Path(args.checkpoints_root or common.get("checkpoints_root",
                                                                   "models/checkpoints")),
        logs_root=Path(args.logs_root or common.get("logs_root", "logs")),
        results_root=Path(args.results_root or common.get("results_root", "data/results")),
        dry_run=args.dry_run,
        skip_extract=args.skip_extract,
        skip_evaluate=args.skip_evaluate,
    )

    print(f"Sweep ID: {ctx.sweep_id}")
    print(f"Checkpoints: {ctx.checkpoints_root}/{ctx.sweep_id}/<exp>/")
    print(f"Logs: {ctx.logs_root}/{ctx.sweep_id}/<exp>/")
    print(f"Results: {ctx.results_root}/{ctx.sweep_id}/<exp>/")
    print(f"Experiments to run ({len(experiments)}): "
          f"{[e['name'] for e in experiments]}")
    if args.dry_run:
        print("\n[DRY RUN — commands будут напечатаны, не выполнены]")

    # Save sweep metadata snapshot
    sweep_meta_dir = ctx.results_root / ctx.sweep_id
    if not args.dry_run:
        sweep_meta_dir.mkdir(parents=True, exist_ok=True)
        (sweep_meta_dir / "sweep_config.yaml").write_text(Path(args.config).read_text())

    # 1. Training phase
    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*70}\n[TRAIN] {name}: {exp.get('description', '')}\n{'='*70}")
        if already_trained(exp, defaults, common, ctx) and not args.force:
            print(f"  SKIP (last checkpoint already exists; use --force to rerun)")
            continue
        cmd = build_train_cmd(exp, defaults, common, ctx)
        rc = run_subprocess(cmd, args.dry_run)
        if rc != 0 and not args.dry_run:
            print(f"  ERROR: training failed (rc={rc}) — continuing with next exp")

    # 2. Extract phase
    if not args.skip_extract:
        for exp in experiments:
            name = exp["name"]
            print(f"\n{'='*70}\n[EXTRACT] {name}\n{'='*70}")
            cmd = build_extract_cmd(exp, common, ctx)
            rc = run_subprocess(cmd, args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"  ERROR: extract failed (rc={rc})")

    # 3. Evaluation phase (one unified eval per experiment — uses the
    # extracted embeddings from stage 2)
    if not args.skip_evaluate:
        for exp in experiments:
            name = exp["name"]
            emb_dir = ctx.results_root / ctx.sweep_id / name / "embeddings"
            out_dir = ctx.results_root / ctx.sweep_id / name / "eval"
            cmd = [
                sys.executable, str(EVAL),
                "--meta_path", str(common.get("metadata_path",
                                              "data/processed/tiles_metadata.csv")),
                "--emb_dir", str(emb_dir),
                "--output_dir", str(out_dir),
                "--annotations_path",
                str(_ROOT / "data" / "sft_annotations.csv"),
                "--seed", str(common.get("seed", 42)),
            ]
            print(f"\n{'='*70}\n[EVAL] {name}\n{'='*70}")
            rc = run_subprocess(cmd, args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"  ERROR: evaluation failed (rc={rc})")

    # Aggregate sweep summary: walk through each experiment's eval output
    # and collect the latest markdown report. Produces one master CSV.
    if not args.dry_run and not args.skip_evaluate:
        summary_rows = []
        for exp in experiments:
            name = exp["name"]
            eval_dir = ctx.results_root / ctx.sweep_id / name / "eval"
            linear_csv = eval_dir / "linear_probe_summary.csv"
            knn_csv = eval_dir / "knn_probe_summary.csv"
            entry = {"experiment": name, "type": exp["type"]}
            if linear_csv.exists():
                import pandas as pd
                lp = pd.read_csv(linear_csv)
                if len(lp) > 0:
                    entry["linear_best_acc"] = float(lp["mean_accuracy"].max())
            if knn_csv.exists():
                import pandas as pd
                knn = pd.read_csv(knn_csv)
                if len(knn) > 0:
                    entry["knn_best_acc"] = float(knn["mean_accuracy"].max())
            summary_rows.append(entry)
        if summary_rows:
            import pandas as pd
            summary_df = pd.DataFrame(summary_rows)
            summary_path = sweep_meta_dir / "sweep_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSweep summary written: {summary_path}")
            print(summary_df.to_string(index=False))

    print(f"\n{'='*70}\nSweep {ctx.sweep_id} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
