"""
Scale Invariance Metrics for SEM Image Clustering.

Metric 1: Magnification Entropy per Cluster
- Bins magnifications into log-scale ranges
- Computes Shannon entropy of the bin distribution per cluster
- Higher entropy = more magnification diversity = better scale invariance

Metric 2: Cramér's V (cluster vs magnification association)
Metric 3: Adjusted Mutual Information
Metric 4: Magnitude Variance Ratio

Reports per-cluster and weighted-average entropy for each model.

Usage:
  python src/evaluation/scale_invariance_metrics.py
  python src/evaluation/scale_invariance_metrics.py --n_clusters 6 --normalize
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import entropy
from pathlib import Path

from src.evaluation.eval_utils import (
    load_model_embeddings, load_metadata, load_aligned_data, l2_normalize,
    DEFAULT_META_PATH, DEFAULT_EMB_DIR, DEFAULT_OUTPUT_DIR, MODEL_CONFIGS,
)

# Default (legacy hardcoded) log-scale bins for magnification.
# Покрывают 6 порядков величины: 10^1 to 10^6. Подходят для вольфрама W,
# но могут не совпадать с реальным распределением mag в tiles_metadata.csv.
# Предпочтительный путь — compute_adaptive_mag_bins().
MAG_BINS = [0, 100, 1000, 10000, 100000, 1e7]
MAG_BIN_LABELS = ['<100x', '100–1K', '1K–10K', '10K–100K', '>100K']


def compute_adaptive_mag_bins(
    mag_values: np.ndarray,
    n_bins: int = 5,
    method: str = "log_quantile",
) -> tuple[list[float], list[str]]:
    """Вычисляет бины увеличения адаптивно к реальному распределению.

    Хардкод `MAG_BINS = [0, 100, 1000, 10000, 100000, 1e7]` предполагает
    конкретный диапазон. На реальных данных распределение может быть
    смещено (например, все снимки при mag ∈ [500, 50000]). Адаптивные бины
    гарантируют, что в каждом бине есть примерно равное число тайлов
    (log-quantile), либо что границы совпадают с «природными» breakpoints
    (для log_quantile это и есть квантили).

    Аргументы:
        mag_values: все значения mag из dataset (может содержать NaN).
        n_bins: число бинов (default 5).
        method: 'log_quantile' (квантили в log10-пространстве) или
                'log_linear' (равные интервалы в log10).

    Возвращает:
        (bins, labels) — как у pd.cut.
    """
    finite = np.asarray(mag_values, dtype=np.float64)
    finite = finite[~np.isnan(finite)]
    finite = finite[finite > 0]
    if len(finite) == 0:
        return MAG_BINS, MAG_BIN_LABELS

    log_mags = np.log10(finite)

    if method == "log_quantile":
        edges = np.percentile(log_mags, np.linspace(0, 100, n_bins + 1))
    elif method == "log_linear":
        edges = np.linspace(log_mags.min(), log_mags.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    bins = [float(10 ** e) for e in edges]
    # Слегка расширим нижнюю и верхнюю границы, чтобы pd.cut не терял краевые точки.
    bins[0] = max(0.0, bins[0] * 0.99)
    bins[-1] = bins[-1] * 1.01

    labels = []
    for i in range(len(bins) - 1):
        lo = int(round(bins[i]))
        hi = int(round(bins[i + 1]))
        labels.append(f"{lo}-{hi}" if lo > 0 else f"≤{hi}")
    return bins, labels


def magnification_entropy(labels, magnifications, n_clusters, bins, bin_labels):
    """
    Compute magnification entropy per cluster.

    Returns:
        per_cluster: dict of {cluster_id: (entropy, n_tiles, dominant_bin, bin_counts)}
        weighted_avg: float - weighted average entropy across clusters
        max_entropy: float - theoretical maximum (uniform across all bins)
    """
    max_ent = np.log2(len(bin_labels))  # uniform distribution

    mag_binned = pd.cut(magnifications, bins=bins, labels=bin_labels)

    results = {}
    total_tiles = 0
    weighted_sum = 0.0

    for cid in range(n_clusters):
        mask = labels == cid
        cluster_bins = mag_binned[mask]
        n_tiles = mask.sum()

        # Count per bin
        counts = cluster_bins.value_counts().reindex(bin_labels, fill_value=0)
        probs = counts.values / counts.values.sum()

        # Shannon entropy (log2)
        ent = entropy(probs, base=2)
        dominant = counts.idxmax()

        results[cid] = {
            'entropy': ent,
            'n_tiles': n_tiles,
            'dominant_bin': dominant,
            'bin_distribution': dict(zip(bin_labels, counts.values)),
            'normalized_entropy': ent / max_ent if max_ent > 0 else 0,
        }

        weighted_sum += ent * n_tiles
        total_tiles += n_tiles

    weighted_avg = weighted_sum / total_tiles

    return results, weighted_avg, max_ent


def cramers_v(labels, mag_bins):
    """Cramér's V: association between cluster and magnification bin.
    0 = no association, 1 = perfect association.
    Lower = more scale-invariant."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(labels, mag_bins)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def mag_variance_ratio(labels, log_mags, n_clusters):
    """Ratio of between-cluster mag variance to total mag variance.
    1.0 = clusters perfectly separate magnifications.
    0.0 = magnification is random w.r.t. clusters.
    Lower = more scale-invariant."""
    total_var = np.var(log_mags)
    if total_var == 0:
        return 0.0

    grand_mean = np.mean(log_mags)
    between_var = 0
    n_total = len(log_mags)
    for cid in range(n_clusters):
        mask = labels == cid
        n_k = mask.sum()
        if n_k == 0:
            continue
        cluster_mean = np.mean(log_mags[mask])
        between_var += n_k * (cluster_mean - grand_mean) ** 2
    between_var /= n_total

    return between_var / total_var


def main(args):
    print("Loading and aligning data...\n")

    # Adaptive mag bins: один набор на весь запуск (чтобы сравнивать модели
    # на одних и тех же bin-границах). Вычисляется по первой доступной модели.
    first_mags = None
    for model_name in MODEL_CONFIGS:
        try:
            _, meta_first = load_aligned_data(model_name, args.emb_dir, args.meta_path)
            first_mags = meta_first["mag"].values
            break
        except FileNotFoundError:
            continue

    if args.adaptive_bins and first_mags is not None:
        bins, bin_labels = compute_adaptive_mag_bins(
            first_mags, n_bins=args.n_bins, method=args.bin_method,
        )
        print(f"Adaptive bins ({args.bin_method}, n_bins={args.n_bins}): {bin_labels}")
    else:
        bins, bin_labels = MAG_BINS, MAG_BIN_LABELS
        print(f"Using hardcoded bins: {bin_labels}")

    max_ent = np.log2(len(bin_labels))
    summary_rows = []

    for model_name in MODEL_CONFIGS:
        try:
            emb, meta = load_aligned_data(model_name, args.emb_dir, args.meta_path)
        except FileNotFoundError as e:
            print(f"  SKIP {model_name}: {e}")
            continue

        print(f"Processing {model_name}...")

        mags = meta['mag'].values
        log_mags = np.log10(mags)
        mag_binned = pd.cut(mags, bins=bins, labels=bin_labels)

        # L2-normalize if requested (default: True for consistency with evaluate_sic_clustering)
        if args.normalize:
            emb = l2_normalize(emb)

        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)

        # Metric 1: Entropy
        _, weighted_avg_ent, _ = magnification_entropy(
            labels, mags, args.n_clusters, bins, bin_labels
        )

        # Metric 2: Cramér's V
        cv = cramers_v(labels, mag_binned)

        # Metric 3: Adjusted Mutual Information
        ami = adjusted_mutual_info_score(labels, mag_binned)

        # Metric 4: Magnitude Variance Ratio
        mvr = mag_variance_ratio(labels, log_mags, args.n_clusters)

        summary_rows.append({
            'Model': model_name,
            'Entropy': f"{weighted_avg_ent:.3f} ({weighted_avg_ent/max_ent:.0%})",
            "Cramer's V (low=good)": f"{cv:.4f}",
            'AMI (low=good)': f"{ami:.4f}",
            'MagVarRatio (low=good)': f"{mvr:.4f}",
        })

    print(f"\n{'=' * 80}")
    print("SCALE INVARIANCE METRICS  (lower = better for scale invariance)")
    print(f"{'=' * 80}")
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    print()
    print("Cramer's V:    Cluster <-> Magnification association (0=none, 1=perfect)")
    print("AMI:           Adjusted Mutual Information (0=random, 1=identical grouping)")
    print("MagVarRatio:   % of magnification variance explained by clusters (0=none, 1=all)")
    norm_str = "L2-normalized" if args.normalize else "raw (unnormalized)"
    print(f"\nKMeans: {args.n_clusters} clusters, embeddings: {norm_str}")
    print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scale_invariance_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scale Invariance Metrics")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--emb_dir", type=str, default=str(DEFAULT_EMB_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize embeddings before KMeans (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Use raw embeddings for KMeans")
    parser.add_argument("--adaptive_bins", action="store_true", default=True,
                        help="Compute mag bins adaptively from data distribution "
                             "instead of hardcoded MAG_BINS (default: True)")
    parser.add_argument("--no-adaptive-bins", dest="adaptive_bins", action="store_false")
    parser.add_argument("--n_bins", type=int, default=5,
                        help="Number of adaptive bins (default: 5)")
    parser.add_argument("--bin_method", type=str, default="log_quantile",
                        choices=["log_quantile", "log_linear"])
    args = parser.parse_args()
    main(args)
