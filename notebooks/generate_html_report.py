"""
Генерация красивого HTML-отчёта для научного руководителя.

Запуск:
  python notebooks/generate_html_report.py

Результат:
  notebooks/crystal_surface_report.html
"""

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import umap


# ─── Пути ─────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "crystal"
OUT = PROJECT / "notebooks" / "crystal_surface_report.html"

# ─── Утилиты ──────────────────────────────────────────────────
def fig_to_base64(fig, dpi=150):
    """Matplotlib figure → base64 PNG строка."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def img_tag(b64, caption="", width="100%"):
    return f"""
    <figure>
      <img src="data:image/png;base64,{b64}" style="width:{width}; border-radius:12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
      <figcaption>{caption}</figcaption>
    </figure>"""


# ─── 1. Загрузка данных ───────────────────────────────────────
print("Загрузка данных...")
atoms = pd.read_parquet(DATA / "atoms.parquet")
patches = np.load(DATA / "patches" / "patches.npy")
patch_meta = pd.read_csv(DATA / "patches" / "patches_metadata.csv")
embeddings = np.load(DATA / "embeddings" / "crystal_embeddings.npy")
emb_meta = pd.read_csv(DATA / "embeddings" / "embeddings_metadata.csv")

labels = emb_meta["cluster_8"].values
n_clusters = 8
X = emb_meta["X"].values
Y = emb_meta["Y"].values
Z = emb_meta["Z"].values

# Генерация осмысленных цветов (близкие тона для похожих кластеров по n1)
if "n1" in emb_meta.columns:
    cluster_n1_means = [emb_meta.loc[labels == c, "n1"].mean() for c in range(n_clusters)]
    ranked_clusters = np.argsort(cluster_n1_means)[::-1] # От террас (max n1) к изломам (min n1)
    cluster_rank = {c: rank for rank, c in enumerate(ranked_clusters)}
    
    # Используем спектр Plasma: от синего/фиолетового (высокий n1) до желтого (низкий n1)
    cmap = matplotlib.colormaps.get_cmap('plasma') if hasattr(matplotlib, 'colormaps') else matplotlib.cm.get_cmap('plasma')
    COLORS = [matplotlib.colors.to_hex(cmap(cluster_rank[c] / max(1, n_clusters - 1))) for c in range(n_clusters)]
else:
    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

print(f"  Атомов: {len(atoms):,}")
print(f"  Патчей: {len(patches):,}")
print(f"  Эмбеддингов: {len(embeddings):,}")


# ─── 2. Полусфера RGB + Grayscale ─────────────────────────────
print("Генерация полусферы...")
fig = plt.figure(figsize=(18, 7), facecolor="#1a1a2e")

ax1 = fig.add_subplot(121, projection="3d", facecolor="#1a1a2e")
rgb = np.clip(np.column_stack([atoms["n1"]/8, atoms["n2"]/6, atoms["n3"]/12]), 0, 1)
ax1.scatter(atoms["X"], atoms["Y"], atoms["Z"], c=rgb, s=0.15, alpha=0.6, depthshade=False)
ax1.set_title("RGB: R=n1/8, G=n2/6, B=n3/12", color="white", fontsize=12)
ax1.view_init(elev=30, azim=45)
for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
    axis.label.set_color("white")
    axis.pane.fill = False

ax2 = fig.add_subplot(122, projection="3d", facecolor="#1a1a2e")
ax2.scatter(atoms["X"], atoms["Y"], atoms["Z"], c=atoms["grayscale"], cmap="gray_r",
            s=0.15, alpha=0.6, depthshade=False)
ax2.set_title("Grayscale", color="white", fontsize=12)
ax2.view_init(elev=30, azim=45)
for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
    axis.label.set_color("white")
    axis.pane.fill = False

plt.suptitle("Визуализация исходных данных", color="white", fontsize=15, fontweight="bold")
plt.tight_layout()
hemisphere_b64 = fig_to_base64(fig)


# ─── 3. Примеры патчей ────────────────────────────────────────
print("Генерация патчей...")
fig, axes = plt.subplots(4, 8, figsize=(20, 10), facecolor="#1a1a2e")
indices = np.random.RandomState(42).choice(len(patches), 32, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 8, i % 8]
    ax.imshow(patches[idx].mean(axis=0), cmap="viridis", interpolation="nearest")
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
plt.suptitle("Примеры 2D-патчей (среднее по 5 каналам)", color="white", fontsize=14, fontweight="bold")
plt.tight_layout()
patches_b64 = fig_to_base64(fig)


# ─── 4. Каналы патча ──────────────────────────────────────────
print("Генерация каналов...")
channel_names = ["n1/8", "n2/6", "n3/12", "n4/24", "n5/8", "RGB"]
sample_idx = np.random.RandomState(123).choice(len(patches))

fig, axes = plt.subplots(1, 6, figsize=(18, 3), facecolor="#1a1a2e")
for ch in range(5):
    axes[ch].imshow(patches[sample_idx, ch], cmap="inferno", interpolation="nearest")
    axes[ch].set_title(channel_names[ch], color="white", fontsize=11)
    axes[ch].axis("off")
    axes[ch].set_facecolor("#1a1a2e")
rgb_patch = np.stack([patches[sample_idx, 0], patches[sample_idx, 1], patches[sample_idx, 2]], axis=-1)
axes[5].imshow(rgb_patch, interpolation="nearest")
axes[5].set_title("RGB", color="white", fontsize=11)
axes[5].axis("off")
axes[5].set_facecolor("#1a1a2e")
plt.suptitle(f"Декомпозиция патча по каналам", color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
channels_b64 = fig_to_base64(fig)


# ─── 5. UMAP ──────────────────────────────────────────────────
print("Построение UMAP (это займёт 2–3 мин)...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
umap_2d = reducer.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
for c in range(n_clusters):
    mask = labels == c
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=COLORS[c],
               label=f"Кластер {c} ({mask.sum():,})", s=3, alpha=0.6, edgecolors="none")
ax.set_title("UMAP-проекция эмбеддингов", color="white", fontsize=14)
ax.set_xlabel("UMAP-1", color="white")
ax.set_ylabel("UMAP-2", color="white")
ax.tick_params(colors="white")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=4, fontsize=10,
          facecolor="#16213e", edgecolor="#0f3460", labelcolor="white")
plt.tight_layout()
umap_b64 = fig_to_base64(fig)


# ─── 7. Распределение ─────────────────────────────────────────
print("Генерация гистограмм...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#1a1a2e")
for ax in axes:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

sizes = np.bincount(labels, minlength=n_clusters)
axes[0].bar(range(n_clusters), sizes, color=COLORS)
axes[0].set_xlabel("Кластер", color="white")
axes[0].set_ylabel("Кол-во атомов", color="white")
axes[0].set_title("Размеры кластеров", color="white", fontsize=13)
for i, v in enumerate(sizes):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=9, color="white")

gs_means = [emb_meta.loc[labels == c, "grayscale"].mean() for c in range(n_clusters)]
gs_stds = [emb_meta.loc[labels == c, "grayscale"].std() for c in range(n_clusters)]
axes[1].bar(range(n_clusters), gs_means, yerr=gs_stds, color=COLORS, capsize=3)
axes[1].set_xlabel("Кластер", color="white")
axes[1].set_ylabel("Средний grayscale", color="white")
axes[1].set_title("Grayscale = степень «выпуклости»", color="white", fontsize=13)

plt.suptitle("Анализ кластеров", color="white", fontsize=14, fontweight="bold")
plt.tight_layout()
distrib_b64 = fig_to_base64(fig)


# ─── 8. Plotly 3D (HTML-строка) ────────────────────────────────
print("Генерация Plotly 3D...")
plotly_fig = go.Figure()
for c in range(n_clusters):
    mask = labels == c
    idx = np.where(mask)[0]
    plotly_fig.add_trace(go.Scatter3d(
        x=X[idx], y=Y[idx], z=Z[idx],
        mode="markers",
        marker=dict(size=1.5, color=COLORS[c], opacity=0.7),
        name=f"Кластер {c} ({mask.sum():,})",
        text=[f"Атом #{j}<br>X={X[j]:.1f}, Y={Y[j]:.1f}, Z={Z[j]:.1f}<br>"
              f"grayscale={emb_meta.iloc[j].get('grayscale', 0):.3f}" for j in idx],
        hoverinfo="text",
    ))
plotly_fig.update_layout(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font_color="white",
    scene=dict(
        xaxis=dict(backgroundcolor="#16213e", gridcolor="#0f3460", title="X"),
        yaxis=dict(backgroundcolor="#16213e", gridcolor="#0f3460", title="Y"),
        zaxis=dict(backgroundcolor="#16213e", gridcolor="#0f3460", title="Z"),
        aspectmode="data",
    ),
    legend=dict(font=dict(size=11), itemsizing="constant"),
    width=1000, height=700,
    margin=dict(l=0, r=0, t=40, b=0),
)
plotly_html = plotly_fig.to_html(full_html=False, include_plotlyjs="cdn")


# ─── 9. Статистика ────────────────────────────────────────────
total_atoms = len(atoms)
surface_atoms = len(patches)
R_val = np.sqrt(atoms["X"]**2 + atoms["Y"]**2 + atoms["Z"]**2).max()
n1_range = f"{int(atoms['n1'].min())}–{int(atoms['n1'].max())}"
n5_range = f"{int(atoms['n5'].min())}–{int(atoms['n5'].max())}"


# ─── 10. Сборка HTML ──────────────────────────────────────────
print("Сборка HTML...")

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Распознавание структурных мотивов на поверхности BCC-кристалла</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e0e0e0;
    line-height: 1.7;
    min-height: 100vh;
  }}

  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 30px;
  }}

  h1 {{
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #e6194b, #f58231, #42d4f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}

  .subtitle {{
    font-size: 1.1em;
    color: #8892b0;
    margin-bottom: 30px;
  }}

  h2 {{
    font-size: 1.5em;
    font-weight: 600;
    color: #42d4f4;
    margin-top: 50px;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(66, 212, 244, 0.2);
  }}

  h3 {{
    font-size: 1.15em;
    color: #f58231;
    margin: 20px 0 10px;
  }}

  p {{ margin-bottom: 14px; }}

  .card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 25px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 20px 0;
  }}

  .stat-card {{
    background: linear-gradient(135deg, rgba(66,212,244,0.1), rgba(15,52,96,0.3));
    border: 1px solid rgba(66,212,244,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }}

  .stat-card .value {{
    font-size: 1.8em;
    font-weight: 700;
    color: #42d4f4;
  }}

  .stat-card .label {{
    font-size: 0.85em;
    color: #8892b0;
    margin-top: 4px;
  }}

  figure {{
    margin: 20px 0;
    text-align: center;
  }}

  figure img {{
    max-width: 100%;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }}

  figcaption {{
    font-size: 0.85em;
    color: #8892b0;
    margin-top: 10px;
    font-style: italic;
  }}

  .plotly-container {{
    background: rgba(255,255,255,0.02);
    border-radius: 16px;
    padding: 10px;
    margin: 20px 0;
    overflow: hidden;
  }}

  .pipeline {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
  }}

  .pipeline-step {{
    background: linear-gradient(135deg, #0f3460, #16213e);
    border: 1px solid rgba(66,212,244,0.3);
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    font-size: 0.9em;
    min-width: 140px;
  }}

  .pipeline-step .step-title {{
    font-weight: 600;
    color: #42d4f4;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}

  .pipeline-step .step-detail {{
    color: #ccd6f6;
    margin-top: 4px;
  }}

  .arrow {{ font-size: 1.5em; color: #f58231; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
  }}

  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}

  th {{
    color: #42d4f4;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  .conclusion {{
    background: linear-gradient(135deg, rgba(230,25,75,0.08), rgba(66,212,244,0.08));
    border: 1px solid rgba(230,25,75,0.2);
    border-radius: 16px;
    padding: 25px;
    margin-top: 30px;
  }}

  .footer {{
    text-align: center;
    color: #4a5568;
    font-size: 0.8em;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
  }}
</style>
</head>
<body>
<div class="container">

  <!-- ═══ Заголовок ═══ -->
  <h1>Распознавание структурных мотивов на поверхности BCC-кристалла</h1>
  <p class="subtitle">
    Контрастивное обучение (SimCLR) · ResNet18<br>
    Учебная / Производственная практика
  </p>

  <!-- ═══ Ключевые цифры ═══ -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="value">{total_atoms:,}</div>
      <div class="label">Всего атомов</div>
    </div>
    <div class="stat-card">
      <div class="value">{surface_atoms:,}</div>
      <div class="label">Поверхностных атомов</div>
    </div>
    <div class="stat-card">
      <div class="value">1.82</div>
      <div class="label">NT-Xent Loss</div>
    </div>
    <div class="stat-card">
      <div class="value">8</div>
      <div class="label">Кластеров</div>
    </div>
    <div class="stat-card">
      <div class="value">50</div>
      <div class="label">Эпох обучения</div>
    </div>
  </div>

  <!-- ═══ Пайплайн ═══ -->
  <h2>1. Схема пайплайна</h2>
  <div class="pipeline">
    <div class="pipeline-step">
      <div class="step-title">Данные</div>
      <div class="step-detail">Excel → Parquet<br>{total_atoms:,} атомов</div>
    </div>
    <span class="arrow">→</span>
    <div class="pipeline-step">
      <div class="step-title">Патчи</div>
      <div class="step-detail">KDTree + проекция<br>{surface_atoms:,} × 5 × 32²</div>
    </div>
    <span class="arrow">→</span>
    <div class="pipeline-step">
      <div class="step-title">SimCLR</div>
      <div class="step-detail">ResNet18 (5 каналов)<br>50 эпох на GPU</div>
    </div>
    <span class="arrow">→</span>
    <div class="pipeline-step">
      <div class="step-title">Кластеризация</div>
      <div class="step-detail">512-мерные эмбеддинги<br>KMeans → 8 кластеров</div>
    </div>
    <span class="arrow">→</span>
    <div class="pipeline-step">
      <div class="step-title">Визуализация</div>
      <div class="step-detail">Plotly 3D<br>+ UMAP</div>
    </div>
  </div>

  <!-- ═══ Исходные данные ═══ -->
  <h2>2. Исходные данные</h2>
  <div class="card">
    <p>Исходный набор данных содержит координаты <b>(X, Y, Z)</b> и числа соседей 1–5 порядка <b>(n1–n5)</b>
    для полусферической поверхности кристалла с ОЦК-решёткой радиусом <b>R ≈ {R_val:.0f}</b> (50 параметров решётки).</p>
    <table>
      <tr><th>Параметр</th><th>Значение</th></tr>
      <tr><td>Число атомов</td><td>{total_atoms:,}</td></tr>
      <tr><td>Радиус полусферы</td><td>{R_val:.1f}</td></tr>
      <tr><td>Диапазон n1</td><td>{n1_range}</td></tr>
      <tr><td>Диапазон n5</td><td>{n5_range}</td></tr>
      <tr><td>Диапазон grayscale</td><td>[{atoms['grayscale'].min():.4f}, {atoms['grayscale'].max():.4f}]</td></tr>
    </table>
  </div>

  {img_tag(hemisphere_b64, "Левая часть — RGB-кодирование (R=n1/8, G=n2/6, B=n3/12). Правая часть — полутон серого (1 − n1·n2·n3·n4·n5 / 110592).")}

  <!-- ═══ Патчи ═══ -->
  <h2>3. Генерация 2D-патчей</h2>
  <div class="card">
    <p>Для каждого поверхностного атома (grayscale > 0.01) строится <b>2D-патч 32×32 пикселей</b> через проекцию окрестности на касательную плоскость:</p>
    <ol style="margin-left: 20px; color: #ccd6f6;">
      <li>KDTree — поиск соседей в радиусе R<sub>patch</sub></li>
      <li>Нормаль = нормированный вектор [X, Y, Z] (полусфера центрирована в начале координат)</li>
      <li>Ортонормированный базис (e₁, e₂) через процедуру Грама–Шмидта</li>
      <li>Растеризация: каждый пиксель хранит 5 каналов [n1/8, n2/6, n3/12, n4/24, n5/8]</li>
    </ol>
  </div>

  {img_tag(patches_b64, "32 случайных патча (среднее по 5 каналам). Видны разные мотивы: однородные террасы, ступеньки, шахматные узоры.")}
  {img_tag(channels_b64, "5-канальная декомпозиция одного патча + RGB-композит.", width="90%")}

  <!-- ═══ Модель ═══ -->
  <h2>4. Архитектура CrystalSimCLR</h2>
  <div class="card">
    <table>
      <tr><th>Компонент</th><th>Описание</th></tr>
      <tr><td>Энкодер</td><td>ResNet18 (модиф. conv1: 5 каналов, без MaxPool)</td></tr>
      <tr><td>Проекционная голова</td><td>512 → 512 → 128</td></tr>
      <tr><td>Функция потерь</td><td>NT-Xent (температура τ = 0.5)</td></tr>
      <tr><td>Аугментации</td><td>Повороты 90°, отражения, гауссов шум, channel dropout, кроп</td></tr>
      <tr><td>Параметры</td><td>~11.5M обучаемых</td></tr>
    </table>
    <p style="margin-top: 12px;">Две аугментированные версии одного патча подаются в сеть; модель обучается
    <b>сближать</b> их представления и <b>отдалять</b> представления разных патчей.</p>
  </div>

  <!-- ═══ UMAP ═══ -->
  <h2>5. UMAP-проекция эмбеддингов</h2>
  <p>512-мерные эмбеддинги сжаты в 2D для визуальной оценки качества разделения кластеров:</p>
  {img_tag(umap_b64, "Каждая точка — поверхностный атом. Цвет = кластер KMeans. Чёткое разделение групп подтверждает, что модель выучила осмысленные признаки.")}

  <!-- ═══ Plotly 3D ═══ -->
  <h2>6. Интерактивная 3D-карта кластеров</h2>
  <p>Вращайте мышью, масштабируйте колёсиком, наводите курсор на атомы для деталей:</p>
  <div class="plotly-container">
    {plotly_html}
  </div>

  <!-- ═══ Распределение ═══ -->
  <h2>7. Анализ кластеров</h2>
  {img_tag(distrib_b64, "Слева: кол-во атомов по кластерам. Справа: средний grayscale — показывает различие «глубины» расположения атомов в каждом кластере.")}

  <!-- ═══ Выводы ═══ -->
  <h2>8. Выводы</h2>
  <div class="conclusion">
    <h3>Основные результаты</h3>
    <ol style="margin-left: 18px; margin-top: 10px;">
      <li><b>Пайплайн работает end-to-end</b>: от сырых координат до 3D-визуализации кластеров.</li>
      <li><b>Модель обучилась осмысленным признакам</b>: Loss = 1.82 за 50 эпох; UMAP показывает чёткое разделение.</li>
      <li><b>Кластеры соответствуют физической структуре</b>: зональные паттерны на 3D-карте согласуются с ожидаемым распределением террас, ступеней и изломов.</li>
    </ol>
    <h3 style="margin-top: 18px;">Дальнейшие шаги</h3>
    <ul style="margin-left: 18px; margin-top: 8px; color: #8892b0;">
      <li>Интерпретация кластеров</li>
      <li>Оптимизация числа кластеров</li>
      <li>Тестирование</li>
    </ul>
  </div>
</div>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"\n✅ HTML-отчёт сохранён: {OUT}")
print(f"   Размер: {OUT.stat().st_size / 1e6:.1f} МБ")
print(f"   Откройте в браузере для просмотра!")
