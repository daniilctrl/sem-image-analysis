"""
Web-based SFT Annotation Tool for SEM tiles.

Reads sft_catalog.csv, shows tiles one by one, and lets you assign
a morphological cluster by clicking a button or pressing a hotkey.

Usage:
    python src/data/annotate_tiles.py [--port 8765]

Then open http://localhost:8765 in your browser.
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.sft_sampler import SftSampler  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CATALOG_CSV = BASE_DIR / "data" / "sft_catalog.csv"
ANNOTATIONS_CSV = BASE_DIR / "data" / "sft_annotations.csv"

CLUSTERS = [
    {"id": "periodic_small", "label": "Периодич. массив (мелкий)", "key": "1", "color": "#4CAF50"},
    {"id": "periodic_large", "label": "Периодич. массив (крупный)", "key": "2", "color": "#2196F3"},
    {"id": "tips",           "label": "Острия / tips",              "key": "3", "color": "#FF9800"},
    {"id": "pedestals",      "label": "Пьедесталы",                 "key": "4", "color": "#9C27B0"},
    {"id": "flat_surface",   "label": "Плоская поверхность",        "key": "5", "color": "#607D8B"},
    {"id": "boundary",       "label": "Граница / переход",          "key": "6", "color": "#F44336"},
    {"id": "surface_texture","label": "Текстура поверхности",       "key": "7", "color": "#795548"},
    {"id": "trash",          "label": "Мусор / артефакт",           "key": "8", "color": "#9E9E9E"},
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>SEM Tile Annotator</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  h1 { color: #00d4ff; margin-bottom: 8px; font-size: 1.4em; }
  .stats {
    background: #16213e; padding: 8px 20px; border-radius: 8px;
    margin-bottom: 16px; font-size: 0.9em; color: #a0a0a0;
  }
  .stats span { color: #00d4ff; font-weight: bold; }
  .main { display: flex; gap: 24px; align-items: flex-start; }
  .tile-container {
    background: #0f3460; border-radius: 12px; padding: 16px;
    display: flex; flex-direction: column; align-items: center;
  }
  .annotate-label {
    font-size: 1.1em; font-weight: bold; color: #00d4ff;
    margin-bottom: 8px; letter-spacing: 1px;
    animation: pulse-text 1.5s ease-in-out infinite;
  }
  @keyframes pulse-text {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  .tile-container img.main-tile {
    width: 384px; height: 384px; image-rendering: pixelated;
    border-radius: 8px;
    border: 3px solid #00d4ff;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4), 0 0 40px rgba(0, 212, 255, 0.15);
  }
  .tile-info {
    margin-top: 8px; font-size: 0.8em; color: #888;
    text-align: center; max-width: 384px; word-break: break-all;
  }
  .context-label {
    margin-top: 16px; font-size: 0.75em; color: #555;
    text-transform: uppercase; letter-spacing: 1px;
  }
  .buttons {
    display: flex; flex-direction: column; gap: 8px;
  }
  .cluster-btn {
    padding: 12px 24px; border: 2px solid transparent; border-radius: 8px;
    font-size: 1em; cursor: pointer; transition: all 0.15s;
    display: flex; align-items: center; gap: 10px;
    background: #16213e; color: #e0e0e0; min-width: 280px;
    text-align: left;
  }
  .cluster-btn:hover { transform: translateX(4px); border-color: #00d4ff; }
  .cluster-btn:active { transform: scale(0.97); }
  .cluster-btn .key {
    background: #333; padding: 2px 8px; border-radius: 4px;
    font-family: monospace; font-size: 0.9em; color: #00d4ff;
  }
  .cluster-btn .dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
  }
  .nav-btns {
    display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; justify-content: center;
  }
  .nav-btn {
    padding: 8px 20px; border: 1px solid #444; border-radius: 6px;
    background: #16213e; color: #ccc; cursor: pointer; font-size: 0.9em;
  }
  .nav-btn:hover { border-color: #00d4ff; color: #fff; }
  .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .per-class-bar {
    display: flex; gap: 6px; justify-content: center; flex-wrap: wrap;
    margin: 10px 0; font-size: 0.72em;
  }
  .per-class-chip {
    background: #16213e; padding: 4px 10px; border-radius: 12px;
    display: flex; align-items: center; gap: 6px; position: relative;
    overflow: hidden; min-width: 110px;
  }
  .per-class-chip .chip-bg {
    position: absolute; top: 0; left: 0; bottom: 0;
    background: rgba(0, 212, 255, 0.15); transition: width 0.3s;
    z-index: 0;
  }
  .per-class-chip > * { position: relative; z-index: 1; }
  .per-class-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }
  .confidence-toggle {
    display: flex; gap: 8px; align-items: center;
    margin: 8px 0; font-size: 0.8em; color: #a0a0a0;
  }
  .confidence-toggle .toggle {
    padding: 4px 10px; border-radius: 4px; cursor: pointer;
    border: 1px solid #444; background: #16213e;
  }
  .confidence-toggle .toggle.active { border-color: #00d4ff; color: #00d4ff; }
  .progress-bar {
    width: 100%; height: 4px; background: #333; border-radius: 2px;
    margin: 12px 0; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, #00d4ff, #4CAF50);
    transition: width 0.3s;
  }
  .context-row {
    display: flex; gap: 4px; margin-top: 12px; flex-wrap: wrap;
    max-width: 500px; justify-content: center;
  }
  .context-tile {
    width: 64px; height: 64px; image-rendering: pixelated;
    border-radius: 4px; border: 1px solid #333; opacity: 0.6;
    cursor: pointer;
  }
  .context-tile:hover { opacity: 1; border-color: #00d4ff; }
  .context-tile.current { border: 2px solid #00d4ff; opacity: 1; }
</style>
</head>
<body>
  <h1>🔬 SEM Tile Annotator</h1>
  <div class="stats">
    Размечено: <span id="annotated">0</span> / <span id="total">0</span>
    (<span id="progress-pct">0%</span>)
    &nbsp;|&nbsp; Осталось: <span id="remaining">0</span>
    &nbsp;|&nbsp; Материал: <span id="material">-</span>
  </div>
  <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
  <div class="stats" style="margin-top:4px; font-size:0.75em;">
    <span id="strategy-label">strategy: loading...</span>
  </div>

  <!-- Per-class прогресс: видно, какие классы под-представлены -->
  <div class="per-class-bar" id="per-class-bar"></div>

  <!-- Confidence toggle: low-confidence метки отдельно помечены -->
  <div class="confidence-toggle">
    Confidence:
    <span class="toggle active" id="conf-high" onclick="setConfidence('high')">
      high (по умолчанию)
    </span>
    <span class="toggle" id="conf-low" onclick="setConfidence('low')">
      low (под сомнением)
    </span>
  </div>

  <div class="main">
    <div class="tile-container">
      <div class="annotate-label">▼ ОЦЕНИТЕ ЭТОТ ТАЙЛ ▼</div>
      <img id="tile-img" class="main-tile" src="" alt="tile">
      <div class="tile-info" id="tile-info"></div>
      <div class="context-label">контекст (соседние тайлы из снимка):</div>
      <div class="context-row" id="context-row"></div>
    </div>

    <div class="buttons" id="buttons"></div>
  </div>

  <div class="nav-btns">
    <button class="nav-btn" onclick="prevTile()">← Назад (Z)</button>
    <button class="nav-btn" onclick="skipTile()">Пропустить (S)</button>
    <button class="nav-btn" onclick="nextUnannotated()">Следующий без метки (N)</button>
    <button class="nav-btn" id="propagate-btn" onclick="togglePropagation()">
      🧠 Label propagation (P)
    </button>
    <button class="nav-btn" onclick="undoLast()" title="Отменить последнюю разметку (U)">
      ⎌ Undo (U)
    </button>
    <button class="nav-btn" onclick="doExport()"
            title="Сохранить snapshot в data/annotations_exports/">
      💾 Export
    </button>
  </div>
  <div id="propagation-panel" style="display:none; margin-top:12px; padding:10px;
       background:#0f3460; border-radius:8px; max-width:700px; text-align:center;">
    <div style="font-size:0.8em; color:#a0a0a0; margin-bottom:6px;">
      Propagation: <span id="prop-info">—</span>
    </div>
    <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
      <button class="nav-btn" style="background:#1a6b3a; border-color:#4CAF50;"
              onclick="acceptPropagation()">✓ Принять (A)</button>
      <button class="nav-btn" style="background:#6b1a1a; border-color:#f44336;"
              onclick="rejectPropagation()">✗ Пропустить (R)</button>
      <button class="nav-btn" onclick="togglePropagation()">Выйти (Esc)</button>
    </div>
    <div id="prop-remaining" style="font-size:0.75em; color:#777; margin-top:6px;"></div>
  </div>

<script>
const CLUSTERS = CLUSTERS_JSON;
const BATCH_SIZE = 50;            // сколько тайлов тянем за раз
const PREFETCH_THRESHOLD = 10;    // подгружать новую партию когда осталось <= N

let tiles = [];                    // текущий упорядоченный ranked-список
let currentIdx = 0;
let annotations = {};
let history = [];                  // для prev-кнопки
let serverStats = null;            // {total_good, total_annotated, has_embeddings, ...}
let fetching = false;              // guard от double-fetch

async function fetchNextBatch() {
  // Берём следующий ranked-батч от сэмплера.
  // Уже размеченные тайлы сервер отфильтрует сам (sampler._unannotated_good).
  if (fetching) return [];
  fetching = true;
  try {
    const resp = await fetch(
      `/api/next_tiles?batch_size=${BATCH_SIZE}&strategy=hybrid`
    );
    const data = await resp.json();
    const meta = data.meta || {};
    if (meta.strategy) {
      document.getElementById('strategy-label').textContent =
        `strategy: ${meta.strategy}` +
        (meta.has_embeddings ? ' (embeddings: yes)' : ' (no embeddings)');
    }
    return data.tiles || [];
  } finally {
    fetching = false;
  }
}

// Confidence state — применяется к следующей annotation.
// Сохраняется в localStorage между reload'ами.
let currentConfidence = localStorage.getItem('sft_confidence') || 'high';

function setConfidence(level) {
  currentConfidence = level;
  localStorage.setItem('sft_confidence', level);
  document.getElementById('conf-high').classList.toggle('active', level === 'high');
  document.getElementById('conf-low').classList.toggle('active', level === 'low');
}

async function refreshStats() {
  try {
    const [statsResp, classResp] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/progress_by_class?target=50'),
    ]);
    serverStats = await statsResp.json();
    const classData = await classResp.json();
    renderStats();
    renderPerClass(classData);
  } catch (_) { /* ignore */ }
}

function renderStats() {
  if (!serverStats) return;
  document.getElementById('total').textContent = serverStats.total_good;
  document.getElementById('annotated').textContent = serverStats.total_annotated;
  document.getElementById('remaining').textContent =
    serverStats.total_good - serverStats.total_annotated;
  const pct = serverStats.total_good > 0
    ? (serverStats.total_annotated / serverStats.total_good * 100).toFixed(2)
    : 0;
  document.getElementById('progress').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
}

function renderPerClass(data) {
  const container = document.getElementById('per-class-bar');
  container.innerHTML = '';
  data.per_class.forEach(r => {
    const chip = document.createElement('div');
    chip.className = 'per-class-chip';
    const widthPct = (r.pct * 100).toFixed(0);
    chip.innerHTML = `
      <div class="chip-bg" style="width:${widthPct}%;
           background: ${r.color}33;"></div>
      <span class="per-class-dot" style="background:${r.color}"></span>
      <span>${r.label}</span>
      <span style="color:#888;">${r.count}/${r.target}</span>
    `;
    container.appendChild(chip);
  });
}

async function undoLast() {
  if (!confirm('Отменить последнюю разметку?')) return;
  const resp = await fetch('/api/undo', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({}),
  });
  const data = await resp.json();
  if (!data.removed || data.removed.length === 0) {
    alert('Нет разметки для отмены.');
    return;
  }
  const r = data.removed[0];
  delete annotations[r.filename];
  await refreshStats();
  // Вернуть этот тайл в начало очереди для немедленной переразметки.
  // Достаточно достать ещё одну partию — сервер его теперь увидит unlabeled.
  const more = await fetchNextBatch();
  tiles = more.concat(tiles.filter(t => !annotations[t.filename]));
  currentIdx = 0;
  showTile();
  alert(`Отменена разметка: ${r.filename} → ${r.cluster}`);
}

async function doExport() {
  const resp = await fetch('/api/export', {method: 'POST'});
  const data = await resp.json();
  if (data.path) {
    alert(`Snapshot сохранён:\n${data.path}\nСтрок: ${data.rows}`);
  } else {
    alert(`Export failed: ${data.reason || 'unknown'}`);
  }
}

async function init() {
  // Инициализация confidence из localStorage
  setConfidence(currentConfidence);

  // Существующие разметки
  const annResp = await fetch('/api/annotations');
  const annData = await annResp.json();
  annotations = annData.annotations || {};

  // Первая партия ranked тайлов
  tiles = await fetchNextBatch();

  // Кнопки кластеров
  const btnContainer = document.getElementById('buttons');
  CLUSTERS.forEach(c => {
    const btn = document.createElement('button');
    btn.className = 'cluster-btn';
    btn.innerHTML = `<span class="dot" style="background:${c.color}"></span>
                     <span class="key">${c.key}</span> ${c.label}`;
    btn.onclick = () => annotate(c.id);
    btnContainer.appendChild(btn);
  });

  await refreshStats();
  showTile();
}

function showTile() {
  if (tiles.length === 0) {
    document.getElementById('tile-info').textContent = 'Нет тайлов для разметки.';
    document.getElementById('tile-img').src = '';
    return;
  }
  if (currentIdx < 0) currentIdx = 0;
  if (currentIdx >= tiles.length) currentIdx = tiles.length - 1;

  const t = tiles[currentIdx];
  document.getElementById('tile-img').src = '/tiles/' + t.filename;
  document.getElementById('tile-info').textContent =
    `${t.filename} | std=${t.std} | mat=${t.material}` +
    (annotations[t.filename] ? ` | ✓ ${annotations[t.filename]}` : '');
  document.getElementById('material').textContent = t.material;

  // Соседи из того же снимка — для контекста
  const ctx = document.getElementById('context-row');
  ctx.innerHTML = '';
  const neighbors = tiles.filter(x => x.source === t.source).slice(0, 8);
  neighbors.forEach(n => {
    const img = document.createElement('img');
    img.className = 'context-tile' + (n.filename === t.filename ? ' current' : '');
    img.src = '/tiles/' + n.filename;
    img.title = n.filename;
    img.onclick = () => { currentIdx = tiles.indexOf(n); showTile(); };
    ctx.appendChild(img);
  });
}

async function annotate(clusterId) {
  const t = tiles[currentIdx];
  annotations[t.filename] = clusterId;
  history.push(currentIdx);

  // Сохранить на сервере (с confidence из toggle).
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      filename: t.filename,
      cluster: clusterId,
      confidence: currentConfidence,
    }),
  });

  // Убрать размеченный тайл из текущего ranked-батча.
  tiles.splice(currentIdx, 1);
  if (currentIdx >= tiles.length) currentIdx = tiles.length - 1;

  // Если батч почти пуст — подгрузить следующий.
  // Ranked-порядок может измениться (uncertainty пересчитывается), поэтому
  // запрашиваем у сервера свежий ranked set.
  if (tiles.length <= PREFETCH_THRESHOLD) {
    const more = await fetchNextBatch();
    // dedup по filename
    const seen = new Set(tiles.map(x => x.filename));
    for (const m of more) {
      if (!seen.has(m.filename) && !annotations[m.filename]) {
        tiles.push(m);
      }
    }
  }

  await refreshStats();
  showTile();
}

async function prevTile() {
  // Откат: возвращаемся к последнему размеченному и показываем его.
  // ВНИМАНИЕ: сервер уже записал аннотацию; чтобы реально откатить —
  // перерозметка через annotate(newCluster) перепишет запись.
  if (history.length === 0) return;
  const lastIdx = history.pop();
  if (lastIdx < tiles.length) {
    currentIdx = lastIdx;
    showTile();
  }
}

function skipTile() {
  if (currentIdx < tiles.length - 1) currentIdx++;
  showTile();
}

async function nextUnannotated() {
  // С ranked-батчем это просто currentIdx++; сервер уже подсовывает
  // только unannotated. Но на всякий случай пропускаем уже размеченные.
  for (let i = currentIdx + 1; i < tiles.length; i++) {
    if (!annotations[tiles[i].filename]) {
      currentIdx = i; showTile(); return;
    }
  }
  // В конце батча — попросим у сервера следующий.
  const more = await fetchNextBatch();
  const seen = new Set(tiles.map(x => x.filename));
  for (const m of more) {
    if (!seen.has(m.filename) && !annotations[m.filename]) {
      tiles.push(m);
    }
  }
  if (currentIdx + 1 < tiles.length) {
    currentIdx++;
    showTile();
  } else {
    alert('Все тайлы размечены! 🎉');
  }
}

// ─── Label propagation mode ─────────────────────────────────────────
// После накопления ~10-30 меток можем «размножить» их через kNN в baseline
// embedding space: для похожих unlabeled тайлов модель предлагает predicted
// label, пользователь лишь accept/reject.
let propagationActive = false;
let propagationQueue = [];       // очередь предложений от /api/propagate
let propagationCurrent = null;

async function togglePropagation() {
  if (propagationActive) {
    propagationActive = false;
    document.getElementById('propagation-panel').style.display = 'none';
    // Возвращаем обычные кнопки классов.
    document.getElementById('buttons').style.pointerEvents = 'auto';
    return;
  }
  // Fetch proposals.
  const resp = await fetch(
    '/api/propagate?batch_size=30&knn_k=5&min_similarity=0.85&min_agreement=0.6'
  );
  const data = await resp.json();
  if (!data.meta.has_embeddings) {
    alert('Label propagation недоступен: baseline embeddings отсутствуют. ' +
          'Сначала извлеките их через extract_simclr_embeddings.py.');
    return;
  }
  if (!data.proposals || data.proposals.length === 0) {
    alert('Нет подходящих предложений. Нужно больше разметки или снижение порогов.');
    return;
  }
  propagationQueue = data.proposals;
  propagationActive = true;
  document.getElementById('propagation-panel').style.display = 'block';
  document.getElementById('buttons').style.pointerEvents = 'none';
  showPropagationProposal();
}

function showPropagationProposal() {
  if (propagationQueue.length === 0) {
    propagationCurrent = null;
    document.getElementById('prop-info').textContent =
      'Очередь пуста. Нажмите P, чтобы обновить предложения.';
    document.getElementById('prop-remaining').textContent = '';
    return;
  }
  propagationCurrent = propagationQueue.shift();
  const p = propagationCurrent;
  const cluster = CLUSTERS.find(c => c.id === p.predicted_cluster);
  const clusterLabel = cluster ? cluster.label : p.predicted_cluster;
  const color = cluster ? cluster.color : '#888';

  document.getElementById('tile-img').src = '/tiles/' + p.tile.filename;
  document.getElementById('tile-info').textContent =
    `${p.tile.filename} | mat=${p.tile.material}`;
  document.getElementById('material').textContent = p.tile.material;
  document.getElementById('prop-info').innerHTML =
    `<span style="color:${color}; font-weight:bold;">${clusterLabel}</span>` +
    ` · sim=${p.similarity.toFixed(3)} · agreement=${(p.agreement*100).toFixed(0)}%`;
  document.getElementById('prop-remaining').textContent =
    `Осталось в очереди: ${propagationQueue.length}`;
  // Обновляем контекст соседей если source известен
  const ctx = document.getElementById('context-row');
  ctx.innerHTML = '';
}

async function acceptPropagation() {
  if (!propagationCurrent) return;
  const p = propagationCurrent;
  annotations[p.tile.filename] = p.predicted_cluster;
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      filename: p.tile.filename,
      cluster: p.predicted_cluster,
    }),
  });
  await refreshStats();
  showPropagationProposal();
}

function rejectPropagation() {
  // Просто пропускаем без записи.
  showPropagationProposal();
}

document.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (key === 'escape') {
    if (propagationActive) togglePropagation();
    return;
  }
  if (propagationActive) {
    if (key === 'a' || key === 'ф') { acceptPropagation(); return; }
    if (key === 'r' || key === 'к') { rejectPropagation(); return; }
    if (key === 'p' || key === 'з') { togglePropagation(); return; }
    return; // в propagation режиме остальные клавиши игнорируем
  }

  if (key === 'z' || key === 'я') { prevTile(); return; }
  if (key === 's' || key === 'ы') { skipTile(); return; }
  if (key === 'n' || key === 'т') { nextUnannotated(); return; }
  if (key === 'p' || key === 'з') { togglePropagation(); return; }
  if (key === 'u' || key === 'г') { undoLast(); return; }
  // Toggle confidence low/high на Shift для быстрого переключения «я сомневаюсь».
  if (e.shiftKey && (key === 'c' || key === 'с')) {
    setConfidence(currentConfidence === 'high' ? 'low' : 'high');
    return;
  }
  const cluster = CLUSTERS.find(c => c.key === key);
  if (cluster) annotate(cluster.id);
});

init();
</script>
</body>
</html>"""


class AnnotationHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the annotation tool."""

    def __init__(self, *args, catalog=None, annotations_path=None,
                 sampler=None, default_strategy="hybrid", **kwargs):
        self.catalog = catalog
        self.annotations_path = annotations_path
        self.sampler = sampler
        self.default_strategy = default_strategy
        self._annotations = self._load_annotations()
        super().__init__(*args, **kwargs)

    def _load_annotations(self):
        """Load existing annotations from CSV.

        Backward compatible: старый формат (filename, cluster) читается;
        новый формат (filename, cluster, confidence, timestamp) — тоже.
        """
        anns = {}
        self._confidence: dict[str, str] = {}
        self._timestamps: dict[str, str] = {}
        if self.annotations_path and self.annotations_path.exists():
            with open(self.annotations_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row['filename']
                    anns[fname] = row['cluster']
                    if row.get('confidence'):
                        self._confidence[fname] = row['confidence']
                    if row.get('timestamp'):
                        self._timestamps[fname] = row['timestamp']
        return anns

    def _save_annotations(self):
        """Persist annotations in the extended format.

        Columns: filename, cluster, confidence, timestamp.
        Legacy readers that only read (filename, cluster) keep working
        because DictReader скипает неизвестные колонки.
        """
        with open(self.annotations_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['filename', 'cluster', 'confidence', 'timestamp'],
            )
            writer.writeheader()
            for fname, cluster in sorted(self._annotations.items()):
                writer.writerow({
                    'filename': fname,
                    'cluster': cluster,
                    'confidence': self._confidence.get(fname, ''),
                    'timestamp': self._timestamps.get(fname, ''),
                })

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            # Serve the HTML page
            html = HTML_TEMPLATE.replace('CLUSTERS_JSON', json.dumps(CLUSTERS))
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif parsed.path == '/api/catalog':
            # Legacy endpoint: возвращает весь каталог shuffled.
            # Сохранён для обратной совместимости; UI предпочитает
            # /api/next_tiles (smart ordering).
            tiles = [r for r in self.catalog if not r.get('is_trash', False)]
            random.shuffle(tiles)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'tiles': tiles}).encode())

        elif parsed.path == '/api/next_tiles':
            # Smart ordering endpoint.
            # Query params: ?batch_size=50&strategy=hybrid
            q = parse_qs(parsed.query or "")
            batch_size = int((q.get("batch_size") or ["50"])[0])
            strategy = (q.get("strategy") or [self.default_strategy])[0]

            if self.sampler is None:
                # Fallback (sampler failed to init): просто shuffle.
                tiles = [r for r in self.catalog if not r.get('is_trash', False)]
                random.shuffle(tiles)
                tiles = tiles[:batch_size]
                meta = {"strategy": "random_fallback", "has_embeddings": False}
            else:
                tiles = self.sampler.next_batch(
                    self._annotations, batch_size=batch_size, strategy=strategy,
                )
                meta = {
                    "strategy": strategy,
                    "has_embeddings": self.sampler._normed is not None,
                    "n_annotated": len(self._annotations),
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"tiles": tiles, "meta": meta}).encode())

        elif parsed.path == '/api/stats':
            if self.sampler is not None:
                stats = self.sampler.stats(self._annotations)
            else:
                stats = {
                    "total_good": sum(1 for r in self.catalog
                                      if not r.get("is_trash", False)),
                    "total_annotated": len(self._annotations),
                    "by_class": {},
                    "by_material": [],
                    "has_embeddings": False,
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())

        elif parsed.path == '/api/propagate':
            # Query params:
            # ?batch_size=30&knn_k=5&min_similarity=0.85&min_agreement=0.6
            q = parse_qs(parsed.query or "")
            batch_size = int((q.get("batch_size") or ["30"])[0])
            knn_k = int((q.get("knn_k") or ["5"])[0])
            min_sim = float((q.get("min_similarity") or ["0.85"])[0])
            min_agr = float((q.get("min_agreement") or ["0.6"])[0])

            if self.sampler is None or self.sampler._normed is None:
                payload = {
                    "proposals": [],
                    "meta": {
                        "has_embeddings": False,
                        "reason": "no embeddings — label propagation unavailable",
                    },
                }
            else:
                proposals = self.sampler.propose_propagation(
                    self._annotations,
                    batch_size=batch_size,
                    knn_k=knn_k,
                    min_similarity=min_sim,
                    min_agreement=min_agr,
                )
                payload = {
                    "proposals": proposals,
                    "meta": {
                        "has_embeddings": True,
                        "n_annotated": len(self._annotations),
                        "min_similarity": min_sim,
                        "min_agreement": min_agr,
                        "knn_k": knn_k,
                    },
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        elif parsed.path == '/api/progress_by_class':
            # Возвращает: для каждого CLUSTERS класса — count и percent от target.
            q = parse_qs(parsed.query or "")
            target = int((q.get("target") or ["50"])[0])
            from collections import Counter
            counts = Counter(self._annotations.values())
            rows = [
                {
                    "cluster_id": c["id"],
                    "label": c["label"],
                    "color": c["color"],
                    "count": int(counts.get(c["id"], 0)),
                    "target": target,
                    "pct": round(min(counts.get(c["id"], 0) / target, 1.0), 4),
                }
                for c in CLUSTERS
            ]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "per_class": rows,
                "target_per_class": target,
                "total": len(self._annotations),
            }).encode())

        elif parsed.path == '/api/annotations':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'annotations': self._annotations}).encode())

        elif parsed.path.startswith('/tiles/'):
            # Serve tile image
            fname = parsed.path[7:]  # Remove '/tiles/'
            fpath = PROCESSED_DIR / fname
            if fpath.exists() and fpath.suffix == '.png':
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/annotate':
            content_len = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_len))
            filename = body['filename']
            cluster = body['cluster']
            confidence = body.get('confidence', 'high')  # high|low
            from datetime import datetime, timezone
            # microseconds нужны, чтобы undo (который берёт max по timestamp)
            # корректно определял «последнюю» запись даже при быстрой разметке.
            timestamp = datetime.now(timezone.utc).isoformat(timespec='microseconds')

            self._annotations[filename] = cluster
            self._confidence[filename] = confidence
            self._timestamps[filename] = timestamp
            self._save_annotations()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'timestamp': timestamp}).encode())

        elif self.path == '/api/undo':
            # Удаляет последнюю N записей (по timestamp) или конкретный filename.
            content_len = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            filename = body.get('filename')

            removed = []
            if filename:
                if filename in self._annotations:
                    removed.append({
                        "filename": filename,
                        "cluster": self._annotations.pop(filename),
                        "confidence": self._confidence.pop(filename, ''),
                        "timestamp": self._timestamps.pop(filename, ''),
                    })
            else:
                # Найти самую последнюю запись по timestamp.
                if self._timestamps:
                    latest = max(self._timestamps.items(), key=lambda kv: kv[1])
                    fname = latest[0]
                    removed.append({
                        "filename": fname,
                        "cluster": self._annotations.pop(fname),
                        "confidence": self._confidence.pop(fname, ''),
                        "timestamp": self._timestamps.pop(fname, ''),
                    })
            if removed:
                self._save_annotations()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'removed': removed}).encode())

        elif self.path == '/api/export':
            # Snapshot-копия sft_annotations.csv с timestamp в имени.
            # Полезно перед большими изменениями (batch propagation, undo N).
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            exports_dir = self.annotations_path.parent / 'annotations_exports'
            exports_dir.mkdir(parents=True, exist_ok=True)
            out_path = exports_dir / f'sft_annotations_{ts}.csv'
            import shutil
            if self.annotations_path.exists():
                shutil.copy2(self.annotations_path, out_path)
                payload = {'path': str(out_path), 'rows': len(self._annotations)}
            else:
                payload = {'path': None, 'rows': 0,
                           'reason': 'annotations file missing'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging to keep console clean."""
        pass


def load_catalog():
    """Load catalog from CSV."""
    if not CATALOG_CSV.exists():
        print(f"ERROR: Catalog not found at {CATALOG_CSV}")
        print(f"Run 'python src/data/generate_sft_labels.py' first!")
        exit(1)

    rows = []
    with open(CATALOG_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['is_trash'] = row.get('is_trash', 'False') == 'True'
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='SEM Tile Annotation Tool')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument(
        '--strategy', type=str, default='hybrid',
        choices=['random', 'stratified_mat', 'diversity', 'uncertainty', 'hybrid'],
        help='Default ordering strategy for /api/next_tiles (UI can override).',
    )
    parser.add_argument(
        '--disable_smart_ordering', action='store_true',
        help='Force random shuffle (disable sampler entirely).',
    )
    args = parser.parse_args()

    catalog = load_catalog()
    good_tiles = [r for r in catalog if not r['is_trash']]
    trash_tiles = [r for r in catalog if r['is_trash']]

    print(f"Loaded {len(catalog)} tiles ({len(good_tiles)} good, {len(trash_tiles)} trash)")

    sampler = None
    if not args.disable_smart_ordering:
        try:
            sampler = SftSampler.from_defaults()
            has_emb = sampler._normed is not None
            print(
                f"Smart ordering sampler ready "
                f"(strategy={args.strategy!r}, embeddings={'YES' if has_emb else 'no'})"
            )
            if not has_emb:
                print(
                    "  NOTE: baseline embeddings not found — "
                    "will fallback to stratified_mat (material round-robin). "
                    "For full diversity/uncertainty sampling, extract baseline "
                    "embeddings first."
                )
        except Exception as e:
            print(f"WARNING: sampler init failed ({e}); falling back to random shuffle.")
            sampler = None

    print(f"\nStarting annotation server at http://localhost:{args.port}")
    print(f"Annotations will be saved to {ANNOTATIONS_CSV}")
    print(f"Press Ctrl+C to stop.\n")

    def handler_factory(*handler_args, **kwargs):
        return AnnotationHandler(
            *handler_args,
            catalog=catalog,
            annotations_path=ANNOTATIONS_CSV,
            sampler=sampler,
            default_strategy=args.strategy,
            **kwargs
        )

    server = HTTPServer(('localhost', args.port), handler_factory)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        if ANNOTATIONS_CSV.exists():
            with open(ANNOTATIONS_CSV, 'r') as f:
                count = sum(1 for _ in f) - 1
            print(f"Total annotations saved: {count}")


if __name__ == '__main__':
    main()
