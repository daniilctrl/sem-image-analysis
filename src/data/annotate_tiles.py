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
    display: flex; gap: 12px; margin-top: 16px;
  }
  .nav-btn {
    padding: 8px 20px; border: 1px solid #444; border-radius: 6px;
    background: #16213e; color: #ccc; cursor: pointer; font-size: 0.9em;
  }
  .nav-btn:hover { border-color: #00d4ff; color: #fff; }
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

async function refreshStats() {
  try {
    const resp = await fetch('/api/stats');
    serverStats = await resp.json();
    renderStats();
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

async function init() {
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

  // Сохранить на сервере.
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: t.filename, cluster: clusterId})
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

document.addEventListener('keydown', (e) => {
  const key = e.key;
  if (key === 'z' || key === 'я') { prevTile(); return; }
  if (key === 's' || key === 'ы') { skipTile(); return; }
  if (key === 'n' || key === 'т') { nextUnannotated(); return; }
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
        """Load existing annotations from CSV."""
        anns = {}
        if self.annotations_path and self.annotations_path.exists():
            with open(self.annotations_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    anns[row['filename']] = row['cluster']
        return anns

    def _save_annotations(self):
        """Persist annotations as unique filename -> cluster mapping."""
        with open(self.annotations_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'cluster'])
            writer.writeheader()
            for fname, cluster in sorted(self._annotations.items()):
                writer.writerow({'filename': fname, 'cluster': cluster})

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

            self._annotations[filename] = cluster
            self._save_annotations()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())
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
