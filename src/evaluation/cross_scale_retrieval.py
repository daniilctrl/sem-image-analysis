"""
Cross-Scale Retrieval Test — главный эксперимент диплома.

Для каждого тайла ищем K ближайших соседей и проверяем:
1. Precision@K (material) — сколько из K найденных принадлежат тому же материалу?
2. Cross-Scale Precision@K — сколько из K найденных принадлежат тому же материалу, 
   но при ДРУГОМ увеличении? (Это и есть инвариантность к масштабу.)

Запуск:
  python cross_scale_retrieval.py                            # Baseline
  python cross_scale_retrieval.py --emb_file simclr_emb.npy  # SimCLR
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.stats import bootstrap_metric_ci  # noqa: E402


def build_faiss_index(embeddings):
    """Строит FAISS косинусный индекс."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (embeddings / norms).astype('float32')
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


def run_retrieval_test(embeddings, df, K=10):
    """Cross-Scale Retrieval тест на всей выборке.

    Векторизованная реализация: для скорости все `.loc`/`.iloc` обращения
    в inner loop заменены на numpy-массивы заранее. На 22k тайлов это даёт
    ускорение ~10-50x по сравнению с предыдущей pandas-версией.
    """
    index, normed = build_faiss_index(embeddings)

    # Кеширование колонок в numpy (reset_index необходим, чтобы positional
    # индексы normed совпадали с позициями в df).
    df = df.reset_index(drop=True).copy()
    df['material'] = df['source_image'].str.split('__').str[0]

    materials = df['material'].to_numpy()
    sources = df['source_image'].to_numpy()
    mags = df['mag'].to_numpy()
    mag_is_nan = pd.isna(mags)

    # Только материалы с > 1 увеличением (иначе cross-scale бессмыслен)
    valid_mask = ~mag_is_nan
    mat_mag_counts = df.loc[valid_mask].groupby('material')['mag'].nunique()
    valid_materials = set(mat_mag_counts[mat_mag_counts > 1].index.tolist())
    test_mask = np.array([m in valid_materials for m in materials]) & valid_mask
    test_indices = np.flatnonzero(test_mask)

    print(f"Cross-scale test: {len(test_indices)} tiles across "
          f"{len(valid_materials)} materials")
    print(f"Materials: {sorted(valid_materials)}")

    # Batch FAISS search (один вызов вместо цикла)
    query_embs = normed[test_indices]
    # Берём K*5, чтобы отфильтровать self и same-source кандидатов
    fetch_K = K * 5
    sims_all, ids_all = index.search(query_embs, fetch_K)

    results = []
    for i, q_idx in enumerate(tqdm(test_indices, desc=f"Retrieval (K={K})")):
        q_material = materials[q_idx]
        q_mag = mags[q_idx]
        q_source = sources[q_idx]

        ret_ids = ids_all[i]

        # numpy-векторизованная фильтрация self и same-source
        ret_sources = sources[ret_ids]
        ret_materials = materials[ret_ids]
        ret_mags = mags[ret_ids]
        ret_mag_valid = ~pd.isna(ret_mags)

        keep = (ret_ids != q_idx) & (ret_sources != q_source)
        kept_ids = ret_ids[keep][:K]
        if len(kept_ids) < K:
            continue

        kept_materials = ret_materials[keep][:K]
        kept_mags = ret_mags[keep][:K]
        kept_mag_valid = ret_mag_valid[keep][:K]

        is_same_material = (kept_materials == q_material)
        # cross-scale = same material AND mag known AND mag != query mag
        is_cross_scale = is_same_material & kept_mag_valid & (kept_mags != q_mag)

        results.append({
            'tile_idx': int(q_idx),
            'material': q_material,
            'mag': q_mag,
            f'precision@{K}_material': float(is_same_material.sum()) / K,
            f'precision@{K}_cross_scale': float(is_cross_scale.sum()) / K,
        })

    return pd.DataFrame(results)


def print_summary(results_df, K, model_name,
                  bootstrap: bool = True, n_bootstrap: int = 1000, seed: int = 42):
    """Выводит сводку результатов с опциональным bootstrap 95% CI.

    CI считается по per-tile precision@K (каждый query — одна выборка).
    Ширина CI = стандартная ошибка среднего при данном N тайлов.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Cross-Scale Retrieval Results (K={K})")
    print(f"{'='*60}")

    pm = f'precision@{K}_material'
    pcs = f'precision@{K}_cross_scale'

    if bootstrap and len(results_df) > 0:
        pm_point, pm_lo, pm_hi = bootstrap_metric_ci(
            results_df[pm].to_numpy(), n_bootstrap=n_bootstrap, seed=seed,
        )
        pcs_point, pcs_lo, pcs_hi = bootstrap_metric_ci(
            results_df[pcs].to_numpy(), n_bootstrap=n_bootstrap, seed=seed,
        )
        print(f"\n  Overall P@{K} (same material):   "
              f"{pm_point:.4f} [{pm_lo:.4f}, {pm_hi:.4f}] (95% bootstrap CI)")
        print(f"  Overall P@{K} (cross-scale):     "
              f"{pcs_point:.4f} [{pcs_lo:.4f}, {pcs_hi:.4f}]")
    else:
        print(f"\n  Overall Precision@{K} (same material):     {results_df[pm].mean():.4f}")
        print(f"  Overall Precision@{K} (cross-scale):       {results_df[pcs].mean():.4f}")

    print(f"\n  Per-material breakdown:")
    print(f"  {'Material':<25} {'P@K Material':>14} {'P@K CrossScale':>16} {'Tiles':>7}")
    print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*7}")

    for material in sorted(results_df['material'].unique()):
        sub = results_df[results_df['material'] == material]
        print(f"  {material:<25} {sub[pm].mean():>14.4f} {sub[pcs].mean():>16.4f} {len(sub):>7}")

    return results_df[pm].mean(), results_df[pcs].mean()


def plot_results(results_df, K, output_path, model_name):
    """Строит графики результатов."""
    pm = f'precision@{K}_material'
    pcs = f'precision@{K}_cross_scale'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Precision по материалам
    per_mat = results_df.groupby('material')[[pm, pcs]].mean().sort_values(pm, ascending=True)
    
    y_pos = range(len(per_mat))
    axes[0].barh(y_pos, per_mat[pm], color='#4a90d9', alpha=0.8, label=f'Precision@{K} (same material)')
    axes[0].barh(y_pos, per_mat[pcs], color='#e74c3c', alpha=0.8, label=f'Precision@{K} (cross-scale)')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(per_mat.index, fontsize=9)
    axes[0].set_xlabel(f'Precision@{K}')
    axes[0].set_title(f'{model_name}: Precision по материалам', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_xlim(0, 1)
    
    # 2. Precision vs увеличение
    axes[1].scatter(results_df['mag'], results_df[pm], s=3, alpha=0.3, c='#4a90d9', label='Same material')
    axes[1].scatter(results_df['mag'], results_df[pcs], s=3, alpha=0.3, c='#e74c3c', label='Cross-scale')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Увеличение (magnification)', fontsize=11)
    axes[1].set_ylabel(f'Precision@{K}', fontsize=11)
    axes[1].set_title(f'{model_name}: Precision vs Увеличение', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main(args):
    # 1. Загрузка данных
    emb_path = Path(args.emb_dir) / args.emb_file
    names_path = Path(args.emb_dir) / args.names_file
    meta_path = Path(args.meta_path)
    
    print(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    names_df = pd.read_csv(names_path)
    meta_df = pd.read_csv(meta_path)
    
    # Объединяем
    meta_df = meta_df.drop_duplicates(subset=['tile_name'])
    df = names_df.merge(meta_df, on='tile_name', how='inner')
    
    assert len(df) == len(embeddings), f"Mismatch: {len(df)} names vs {len(embeddings)} embeddings"
    print(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D)")
    
    # 2. Запуск теста
    results = run_retrieval_test(embeddings, df, K=args.K)
    
    # 3. Вывод результатов
    avg_pm, avg_pcs = print_summary(
        results, args.K, args.model_name,
        bootstrap=args.bootstrap, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    # Save per-material CI summary
    if args.bootstrap and len(results) > 0:
        from src.utils.stats import bootstrap_metric_ci as _boot
        pm_col = f'precision@{args.K}_material'
        pcs_col = f'precision@{args.K}_cross_scale'
        ci_rows = []
        # Overall
        pm_p, pm_lo, pm_hi = _boot(results[pm_col].to_numpy(),
                                    n_bootstrap=args.n_bootstrap, seed=args.seed)
        pcs_p, pcs_lo, pcs_hi = _boot(results[pcs_col].to_numpy(),
                                       n_bootstrap=args.n_bootstrap, seed=args.seed)
        ci_rows.append({
            'scope': 'overall', 'metric': 'precision@K_material',
            'mean': round(pm_p, 4),
            'ci_lo': round(pm_lo, 4), 'ci_hi': round(pm_hi, 4),
            'n_tiles': len(results),
        })
        ci_rows.append({
            'scope': 'overall', 'metric': 'precision@K_cross_scale',
            'mean': round(pcs_p, 4),
            'ci_lo': round(pcs_lo, 4), 'ci_hi': round(pcs_hi, 4),
            'n_tiles': len(results),
        })
        # Per-material
        for material in sorted(results['material'].unique()):
            sub = results[results['material'] == material]
            if len(sub) < 10:
                continue
            pm_p, pm_lo, pm_hi = _boot(sub[pm_col].to_numpy(),
                                        n_bootstrap=args.n_bootstrap, seed=args.seed)
            pcs_p, pcs_lo, pcs_hi = _boot(sub[pcs_col].to_numpy(),
                                           n_bootstrap=args.n_bootstrap, seed=args.seed)
            ci_rows.append({
                'scope': material, 'metric': 'precision@K_material',
                'mean': round(pm_p, 4),
                'ci_lo': round(pm_lo, 4), 'ci_hi': round(pm_hi, 4),
                'n_tiles': len(sub),
            })
            ci_rows.append({
                'scope': material, 'metric': 'precision@K_cross_scale',
                'mean': round(pcs_p, 4),
                'ci_lo': round(pcs_lo, 4), 'ci_hi': round(pcs_hi, 4),
                'n_tiles': len(sub),
            })
        ci_df = pd.DataFrame(ci_rows)
        ci_path = output_dir / (
            f"cross_scale_ci_{args.model_name.lower().replace(' ', '_')}.csv"
        )
        ci_df.to_csv(ci_path, index=False)
        print(f"Bootstrap CI saved: {ci_path}")
    
    # 4. Сохранение
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_dir / f"cross_scale_results_{args.model_name.lower().replace(' ', '_')}.csv", index=False)
    plot_results(results, args.K, output_dir / f"cross_scale_plot_{args.model_name.lower().replace(' ', '_')}.png", args.model_name)
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Cross-Scale Retrieval Test")
    parser.add_argument("--emb_dir", type=str, default=str(_root / "data" / "embeddings"))
    parser.add_argument("--emb_file", type=str, default="resnet50_embeddings.npy")
    parser.add_argument("--names_file", type=str, default="embedding_names.csv")
    parser.add_argument("--meta_path", type=str, default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "data" / "results"))
    parser.add_argument("--model_name", type=str, default="Baseline")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--bootstrap", action="store_true", default=True,
                        help="Compute bootstrap 95%% CI per-tile and per-material "
                             "(default: True). Adds cross_scale_ci_*.csv artifact.")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for bootstrap RNG (reproducible CI)")

    args = parser.parse_args()
    main(args)
