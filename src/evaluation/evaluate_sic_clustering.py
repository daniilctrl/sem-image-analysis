"""
SiC Structure Clustering Evaluation
====================================
Кластеризация тайлов SiC по типу морфологической структуры
и генерация HTML-отчёта с визуализациями и метриками.

Целевые кластеры:
  A — Pedestals & Tips (отдельные пьедесталы с остриями)
  B — Defective Tips (сглаженные конусы без острия)
  C — High-Density Arrays (массивы микроэмиттеров)
  D — Empty/Base Material (голый субстрат SiC)

Запуск:
  python evaluate_sic_clustering.py
  python evaluate_sic_clustering.py --K 6 --model_name "SimCLR v2"
"""
import argparse
import os
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from PIL import Image
from tqdm import tqdm


# ─── Data Loading ────────────────────────────────────────────
def load_data(emb_dir, meta_path):
    """Загружает эмбеддинги и метаданные, объединяет.
    
    Эмбеддинги и метаданные могут иметь разное количество строк
    (например, если метаданные были обновлены после генерации эмбеддингов).
    Используем inner join и фильтруем эмбеддинги по совпавшим индексам.
    """
    emb_path = Path(emb_dir)
    embeddings = np.load(emb_path / "resnet50_embeddings.npy")
    names_df = pd.read_csv(emb_path / "embedding_names.csv")
    meta_df = pd.read_csv(meta_path).drop_duplicates(subset=['tile_name'])

    # Добавляем индекс эмбеддинга для последующей фильтрации
    names_df['emb_idx'] = range(len(names_df))

    df = names_df.merge(meta_df, on='tile_name', how='inner')
    
    # Фильтруем эмбеддинги по совпавшим строкам
    matched_indices = df['emb_idx'].values
    embeddings = embeddings[matched_indices]
    df = df.drop(columns=['emb_idx']).reset_index(drop=True)
    
    print(f"   Matched {len(df)} tiles (from {len(names_df)} embeddings, {len(meta_df)} metadata)")

    df['material'] = df['source_image'].str.split('__').str[0]  # see also eval_utils.extract_material
    return embeddings, df


# ─── Clustering ──────────────────────────────────────────────
def run_clustering(embeddings, K):
    """K-Means на L2-нормализованных эмбеддингах."""
    normed = normalize(embeddings, norm='l2')
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normed)
    return labels, kmeans, normed


def compute_metrics(embeddings_normed, labels):
    """Вычисляет метрики кластеризации."""
    sil = silhouette_score(embeddings_normed, labels, metric='cosine', sample_size=5000, random_state=42)
    ch = calinski_harabasz_score(embeddings_normed, labels)
    db = davies_bouldin_score(embeddings_normed, labels)
    return {'silhouette': sil, 'calinski_harabasz': ch, 'davies_bouldin': db}


def find_optimal_k(embeddings_normed, k_range=range(2, 9)):
    """Ищет оптимальное K по Silhouette Score."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings_normed)
        sil = silhouette_score(embeddings_normed, labels, metric='cosine', sample_size=5000, random_state=42)
        results.append({'K': k, 'silhouette': sil, 'inertia': km.inertia_})
        print(f"  K={k}: Silhouette={sil:.4f}, Inertia={km.inertia_:.0f}")
    return pd.DataFrame(results)


# ─── UMAP Visualization ─────────────────────────────────────
def compute_umap(embeddings_normed):
    """Снижение размерности UMAP."""
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(embeddings_normed)


def plot_umap_by_cluster(umap_2d, labels, K, output_path, model_name):
    """UMAP раскрашенный по кластерам K-Means."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.Set1(np.linspace(0, 1, K))

    for c in range(K):
        mask = labels == c
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[colors[c]], s=3, alpha=0.4, label=f'Cluster {c+1} ({mask.sum()} tiles)')

    ax.set_title(f'{model_name}: UMAP — кластеры K-Means (K={K})', fontsize=14)
    ax.legend(markerscale=4, fontsize=9)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_umap_by_mag(umap_2d, mag_values, output_path, model_name):
    """UMAP раскрашенный по log(magnification)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    log_mag = np.log10(mag_values.clip(lower=1))
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                    c=log_mag, cmap='viridis', s=3, alpha=0.4)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(Magnification)')
    ax.set_title(f'{model_name}: UMAP — увеличение (magnification)', fontsize=14)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_elbow(k_df, output_path, model_name):
    """Elbow + Silhouette plot for choosing K."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_df['K'], k_df['inertia'], 'bo-')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title(f'{model_name}: Elbow Method')

    axes[1].plot(k_df['K'], k_df['silhouette'], 'rs-')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f'{model_name}: Silhouette vs K')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Cluster–Magnification Analysis ─────────────────────────
def analyze_cluster_mag(df, labels, K):
    """Анализ связи кластеров с увеличением."""
    df = df.copy()
    df['cluster'] = labels

    rows = []
    for c in range(K):
        sub = df[df['cluster'] == c]
        mags = sub['mag'].dropna()
        if len(mags) == 0:
            continue
        rows.append({
            'cluster': c + 1,
            'tiles': len(sub),
            'mag_min': mags.min(),
            'mag_median': mags.median(),
            'mag_max': mags.max(),
            'mag_std': mags.std(),
            'unique_mags': mags.nunique(),
            'unique_sources': sub['source_image'].nunique(),
        })
    return pd.DataFrame(rows)


# ─── Thumbnails ──────────────────────────────────────────────
def get_cluster_thumbnails(df, labels, K, data_dir, n_per_cluster=8):
    """Отбирает тайлы-представители из каждого кластера."""
    df = df.copy()
    df['cluster'] = labels
    thumbs = {}

    for c in range(K):
        sub = df[df['cluster'] == c]
        # Берём случайные, но из разных увеличений
        sample = sub.sample(n=min(n_per_cluster, len(sub)), random_state=42)
        images_b64 = []
        for _, row in sample.iterrows():
            img_path = Path(data_dir) / row['tile_name']
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((128, 128))
                buf = BytesIO()
                img.save(buf, format='JPEG', quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                images_b64.append({
                    'b64': b64,
                    'name': row['tile_name'],
                    'mag': row.get('mag', 'N/A'),
                })
            except Exception:
                continue
        thumbs[c] = images_b64
    return thumbs


# ─── HTML Report ─────────────────────────────────────────────
def generate_html_report(model_name, K, metrics, cluster_mag_df, thumbs,
                         umap_cluster_path, umap_mag_path, elbow_path, output_path):
    """Генерирует полный HTML-отчёт."""

    def img_to_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    umap_c_b64 = img_to_b64(umap_cluster_path)
    umap_m_b64 = img_to_b64(umap_mag_path)
    elbow_b64 = img_to_b64(elbow_path)

    # Cluster mag table
    mag_rows = ""
    for _, r in cluster_mag_df.iterrows():
        mag_rows += f"""<tr>
            <td><strong>Cluster {int(r['cluster'])}</strong></td>
            <td>{int(r['tiles'])}</td>
            <td>{r['mag_min']:.0f}x</td>
            <td>{r['mag_median']:.0f}x</td>
            <td>{r['mag_max']:.0f}x</td>
            <td>{int(r['unique_mags'])}</td>
            <td>{int(r['unique_sources'])}</td>
        </tr>"""

    # Thumbnails per cluster
    cluster_colors = ['#4A6FA5', '#B5594E', '#6B8F71', '#9B7EBD', '#C4985A', '#5B8E9B', '#A0718A', '#7A9E4F']
    thumb_sections = ""
    for c in range(K):
        color = cluster_colors[c % len(cluster_colors)]
        imgs_html = ""
        tile_count = len(thumbs.get(c, []))
        for t in thumbs.get(c, []):
            mag_str = f"{t['mag']:.0f}x" if isinstance(t['mag'], (int, float)) else str(t['mag'])
            imgs_html += f"""<div class="thumb-item">
                <img src="data:image/jpeg;base64,{t['b64']}"/>
                <div class="thumb-label">{mag_str}</div>
            </div>"""
        thumb_sections += f"""<div class="cluster-section">
            <div class="cluster-header">
                <div class="cluster-dot" style="background:{color};"></div>
                <h3>Cluster {c+1}</h3>
            </div>
            <div class="thumbs-grid">{imgs_html}</div>
        </div>"""




    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SiC Clustering Report — {model_name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    *, *::before, *::after {{ box-sizing: border-box; }}

    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #F5F3F0;
        color: #2D2D2D;
        margin: 0;
        padding: 0;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
    }}

    .container {{
        max-width: 960px;
        margin: 0 auto;
        padding: 60px 32px 80px;
    }}

    /* ── Header ────────────────────────────────── */
    .report-header {{
        margin-bottom: 56px;
    }}
    .report-header h1 {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #1A1A1A;
        margin: 0 0 8px;
        letter-spacing: -0.02em;
    }}
    .report-header .subtitle {{
        font-size: 1.05rem;
        font-weight: 400;
        color: #6B6B6B;
        margin: 0;
    }}
    .report-header .divider {{
        width: 48px;
        height: 3px;
        background: #4A6FA5;
        border-radius: 2px;
        margin-top: 24px;
    }}

    /* ── Sections ──────────────────────────────── */
    .section {{
        margin-bottom: 56px;
    }}
    .section-title {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9A9A9A;
        margin: 0 0 20px;
    }}
    .section h2 {{
        font-size: 1.5rem;
        font-weight: 600;
        color: #1A1A1A;
        margin: 0 0 24px;
        letter-spacing: -0.01em;
    }}

    /* ── Metric Cards ─────────────────────────── */
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}
    .metric-card {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #1A1A1A;
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B6B6B;
        margin-top: 8px;
    }}
    .metric-hint {{
        font-size: 0.7rem;
        color: #A0A0A0;
        margin-top: 4px;
    }}

    /* ── Interpretation blocks ────────────────── */
    .callout {{
        background: #FFFFFF;
        border-left: 3px solid #4A6FA5;
        border-radius: 0 8px 8px 0;
        padding: 20px 24px;
        margin: 24px 0;
        font-size: 0.9rem;
        color: #4A4A4A;
        line-height: 1.7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .callout strong {{
        color: #2D2D2D;
    }}

    /* ── Plot containers ──────────────────────── */
    .plot-container {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }}
    .plot-container img {{
        max-width: 100%;
        border-radius: 8px;
    }}
    .plot-caption {{
        font-size: 0.8rem;
        color: #9A9A9A;
        margin-top: 12px;
    }}

    /* ── Tables ────────────────────────────────── */
    .table-wrap {{
        background: #FFFFFF;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th {{
        background: #FAFAFA;
        color: #6B6B6B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 14px 16px;
        text-align: left;
        border-bottom: 1px solid #EBEBEB;
    }}
    td {{
        padding: 12px 16px;
        border-bottom: 1px solid #F5F5F5;
        font-size: 0.9rem;
        color: #3D3D3D;
    }}
    tr:last-child td {{
        border-bottom: none;
    }}
    tr:hover td {{
        background: #FAFBFC;
    }}

    /* ── Cluster thumbnails ───────────────────── */
    .cluster-section {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }}
    .cluster-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }}
    .cluster-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .cluster-header h3 {{
        font-size: 1rem;
        font-weight: 600;
        color: #2D2D2D;
        margin: 0;
    }}
    .cluster-header .tile-count {{
        font-size: 0.8rem;
        color: #9A9A9A;
        margin-left: auto;
    }}
    .thumbs-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .thumb-item {{
        text-align: center;
    }}
    .thumb-item img {{
        border-radius: 6px;
        border: 1px solid #EBEBEB;
        display: block;
    }}
    .thumb-item .thumb-label {{
        font-size: 0.65rem;
        color: #A0A0A0;
        margin-top: 4px;
    }}

    /* ── Footer ────────────────────────────────── */
    .report-footer {{
        margin-top: 64px;
        padding-top: 24px;
        border-top: 1px solid #E0E0E0;
        font-size: 0.75rem;
        color: #B0B0B0;
        text-align: center;
    }}

    /* ── Responsive ────────────────────────────── */
    @media (max-width: 640px) {{
        .metrics-grid {{ grid-template-columns: 1fr; }}
        .container {{ padding: 32px 16px 48px; }}
        .report-header h1 {{ font-size: 1.75rem; }}
    }}
</style></head>
<body><div class="container">

<div class="report-header">
    <h1>SiC Clustering Report</h1>
    <p class="subtitle">{model_name} · K = {K} кластеров · Анализ морфологии микроэмиттеров</p>
    <div class="divider"></div>
</div>

<!-- ── Metrics ────────────────────────────────── -->
<div class="section">
    <div class="section-title">Метрики кластеризации</div>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{metrics['silhouette']:.4f}</div>
            <div class="metric-label">Silhouette Score</div>
            <div class="metric-hint">-1 … +1, выше = лучше</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['calinski_harabasz']:.0f}</div>
            <div class="metric-label">Calinski-Harabasz Index</div>
            <div class="metric-hint">выше = лучше</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['davies_bouldin']:.4f}</div>
            <div class="metric-label">Davies-Bouldin Index</div>
            <div class="metric-hint">ниже = лучше</div>
        </div>
    </div>
    <div class="callout">
        <strong>Интерпретация.</strong>
        Silhouette Score показывает, насколько чётко разделены кластеры. Значение &gt;0.25 указывает на
        наличие структуры в данных. Calinski-Harabasz и Davies-Bouldin служат для сравнения моделей
        (Baseline vs SimCLR / BYOL) при одинаковом K.
    </div>
</div>

<!-- ── Optimal K ──────────────────────────────── -->
<div class="section">
    <div class="section-title">Выбор оптимального K</div>
    <div class="plot-container">
        <img src="data:image/png;base64,{elbow_b64}" alt="Elbow + Silhouette plot"/>
        <div class="plot-caption">Elbow Method (inertia) и Silhouette Score для K ∈ [2, 8]</div>
    </div>
</div>

<!-- ── UMAP Clusters ──────────────────────────── -->
<div class="section">
    <div class="section-title">UMAP — Кластеры K-Means</div>
    <div class="plot-container">
        <img src="data:image/png;base64,{umap_c_b64}" alt="UMAP by cluster"/>
        <div class="plot-caption">2D-проекция UMAP, раскраска по кластерам K-Means (K={K})</div>
    </div>
</div>

<!-- ── UMAP Magnification ─────────────────────── -->
<div class="section">
    <div class="section-title">UMAP — Увеличение</div>
    <div class="plot-container">
        <img src="data:image/png;base64,{umap_m_b64}" alt="UMAP by magnification"/>
        <div class="plot-caption">2D-проекция UMAP, раскраска по log₁₀(Magnification)</div>
    </div>
    <div class="callout">
        <strong>Ключевой вопрос.</strong>
        Если UMAP-кластеры совпадают с цветовой картой увеличений — модель группирует тайлы
        по масштабу, а не по морфологии. Это указывает на отсутствие масштабной инвариантности
        и необходимость дообучения (SimCLR / BYOL).
    </div>
</div>

<!-- ── Cluster × Magnification ────────────────── -->
<div class="section">
    <div class="section-title">Кластеры × Увеличение</div>
    <div class="table-wrap">
    <table>
        <tr><th>Кластер</th><th>Тайлов</th><th>Min Mag</th><th>Median Mag</th><th>Max Mag</th><th>Unique Mags</th><th>Sources</th></tr>
        {mag_rows}
    </table>
    </div>
</div>

<!-- ── Thumbnails ─────────────────────────────── -->
<div class="section">
    <div class="section-title">Примеры тайлов из кластеров</div>
    {thumb_sections}
</div>

<div class="report-footer">
    SEM Image Analysis Pipeline · {model_name} · K={K}
</div>

</div></body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ─── Main ────────────────────────────────────────────────────
def main(args):
    print("=" * 60)
    print(f"  SiC Clustering Evaluation — {args.model_name}")
    print("=" * 60)

    # 1. Load
    print("\n1. Loading data...")
    embeddings, df = load_data(args.emb_dir, args.meta_path)
    print(f"   {len(embeddings)} tiles, {embeddings.shape[1]} dimensions")

    # 2. Normalize
    normed = normalize(embeddings, norm='l2')

    # 3. Find optimal K
    print("\n2. Finding optimal K...")
    k_df = find_optimal_k(normed, k_range=range(2, 9))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elbow_path = output_dir / f"sic_elbow_{args.model_name.lower().replace(' ', '_')}.png"
    plot_elbow(k_df, elbow_path, args.model_name)

    # 4. K-Means
    K = args.K
    print(f"\n3. Running K-Means (K={K})...")
    labels, kmeans, _ = run_clustering(embeddings, K)

    # 5. Metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(normed, labels)
    print(f"   Silhouette:        {metrics['silhouette']:.4f}")
    print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.0f}")
    print(f"   Davies-Bouldin:    {metrics['davies_bouldin']:.4f}")

    # 6. Cluster-Magnification analysis
    print("\n5. Analyzing cluster-magnification relationship...")
    cluster_mag_df = analyze_cluster_mag(df, labels, K)
    print(cluster_mag_df.to_string(index=False))

    # 7. UMAP
    print("\n6. Computing UMAP...")
    umap_2d = compute_umap(normed)

    umap_cluster_path = output_dir / f"sic_umap_clusters_{args.model_name.lower().replace(' ', '_')}.png"
    umap_mag_path = output_dir / f"sic_umap_mag_{args.model_name.lower().replace(' ', '_')}.png"

    plot_umap_by_cluster(umap_2d, labels, K, umap_cluster_path, args.model_name)
    plot_umap_by_mag(umap_2d, df['mag'], umap_mag_path, args.model_name)

    # 8. Thumbnails
    print("\n7. Generating thumbnails...")
    thumbs = get_cluster_thumbnails(df, labels, K, args.data_dir, n_per_cluster=10)

    # 9. HTML Report
    print("\n8. Generating HTML report...")
    report_path = output_dir / f"sic_clustering_report_{args.model_name.lower().replace(' ', '_')}.html"
    generate_html_report(
        model_name=args.model_name,
        K=K,
        metrics=metrics,
        cluster_mag_df=cluster_mag_df,
        thumbs=thumbs,
        umap_cluster_path=umap_cluster_path,
        umap_mag_path=umap_mag_path,
        elbow_path=elbow_path,
        output_path=report_path,
    )

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="SiC Structure Clustering Evaluation")
    parser.add_argument("--emb_dir", type=str, default=str(_root / "data" / "embeddings"))
    parser.add_argument("--meta_path", type=str, default=str(_root / "data" / "processed" / "tiles_metadata.csv"))
    parser.add_argument("--data_dir", type=str, default=str(_root / "data" / "processed"))
    parser.add_argument("--output_dir", type=str, default=str(_root / "data" / "results"))
    parser.add_argument("--model_name", type=str, default="Baseline ResNet50")
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()
    main(args)
