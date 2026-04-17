"""Unit-тесты для CSV-формата аннотаций и undo/export логики.

Тестируем только серверную часть: загрузку/сохранение `AnnotationHandler`
и эндпоинты /api/undo, /api/export, /api/progress_by_class, /api/annotate
(с confidence). Фронтенд покрывается end-to-end smoke-скриптом в PR #12.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import annotate_tiles as at  # noqa: E402
from src.data.sft_sampler import SftSampler  # noqa: E402


def _start_server(catalog, ann_csv):
    """Поднимает AnnotationHandler на свободном порту, возвращает (port, srv)."""
    sampler = SftSampler(catalog=catalog)

    def handler_factory(*args, **kwargs):
        return at.AnnotationHandler(
            *args, catalog=catalog, annotations_path=ann_csv,
            sampler=sampler, default_strategy="stratified_mat", **kwargs,
        )

    srv = HTTPServer(("127.0.0.1", 0), handler_factory)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return port, srv


def _fake_catalog(n: int = 10) -> list[dict]:
    return [
        {
            "filename": f"test_{i:03d}.png",
            "material": ["A", "B", "C"][i % 3],
            "img_id": str(i), "source": f"src_{i}",
            "x": 0, "y": 0, "std": 50.0, "mean": 100.0, "bottom_mean": 50.0,
            "is_trash": False, "cluster": "",
        }
        for i in range(n)
    ]


def _get(port, path):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}").read())


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())


def test_annotation_csv_has_extended_columns():
    """После /api/annotate в CSV должны быть filename, cluster, confidence, timestamp."""
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            _post(port, "/api/annotate",
                  {"filename": "test_000.png", "cluster": "tips",
                   "confidence": "high"})
            rows = list(csv.DictReader(open(ann_csv)))
            assert len(rows) == 1
            assert rows[0]["filename"] == "test_000.png"
            assert rows[0]["cluster"] == "tips"
            assert rows[0]["confidence"] == "high"
            assert rows[0]["timestamp"]  # non-empty ISO
            assert "T" in rows[0]["timestamp"]  # ISO format
        finally:
            srv.shutdown()


def test_annotation_legacy_csv_backward_compat():
    """Старый CSV формат (filename, cluster без extra колонок) читается."""
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        # Пишем legacy-формат
        with open(ann_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "cluster"])
            w.writeheader()
            w.writerow({"filename": "test_000.png", "cluster": "tips"})
            w.writerow({"filename": "test_001.png", "cluster": "pedestals"})

        port, srv = _start_server(catalog, ann_csv)
        try:
            # Stats должен показать 2 уже размеченных
            stats = _get(port, "/api/stats")
            assert stats["total_annotated"] == 2
            # Новая разметка должна пройти и дописать confidence+timestamp
            _post(port, "/api/annotate",
                  {"filename": "test_002.png", "cluster": "tips",
                   "confidence": "low"})
            rows = list(csv.DictReader(open(ann_csv)))
            assert len(rows) == 3
            # Legacy записи имеют пустой confidence, новая — 'low'
            new_row = next(r for r in rows if r["filename"] == "test_002.png")
            assert new_row["confidence"] == "low"
        finally:
            srv.shutdown()


def test_undo_last_removes_most_recent():
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            _post(port, "/api/annotate", {"filename": "test_000.png", "cluster": "tips"})
            time.sleep(0.01)
            _post(port, "/api/annotate", {"filename": "test_001.png", "cluster": "pedestals"})
            time.sleep(0.01)
            _post(port, "/api/annotate", {"filename": "test_002.png", "cluster": "boundary"})

            undo = _post(port, "/api/undo", {})
            assert len(undo["removed"]) == 1
            assert undo["removed"][0]["filename"] == "test_002.png", (
                f"Должен удалить последний по timestamp, "
                f"получил {undo['removed'][0]['filename']}"
            )

            stats = _get(port, "/api/stats")
            assert stats["total_annotated"] == 2
        finally:
            srv.shutdown()


def test_undo_by_filename():
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            _post(port, "/api/annotate", {"filename": "test_000.png", "cluster": "tips"})
            _post(port, "/api/annotate", {"filename": "test_001.png", "cluster": "pedestals"})
            _post(port, "/api/annotate", {"filename": "test_002.png", "cluster": "boundary"})

            undo = _post(port, "/api/undo", {"filename": "test_001.png"})
            assert undo["removed"][0]["filename"] == "test_001.png"

            stats = _get(port, "/api/stats")
            assert stats["total_annotated"] == 2
        finally:
            srv.shutdown()


def test_undo_empty_returns_empty():
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            undo = _post(port, "/api/undo", {})
            assert undo["removed"] == []
        finally:
            srv.shutdown()


def test_export_creates_timestamped_copy():
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            _post(port, "/api/annotate", {"filename": "test_000.png", "cluster": "tips"})
            _post(port, "/api/annotate", {"filename": "test_001.png", "cluster": "pedestals"})
            exp = _post(port, "/api/export", {})
            assert exp["path"], f"export path missing: {exp}"
            out = Path(exp["path"])
            assert out.exists()
            assert out.parent.name == "annotations_exports"
            assert "sft_annotations_" in out.name
            assert exp["rows"] == 2
            # Snapshot содержит те же данные
            rows = list(csv.DictReader(open(out)))
            assert len(rows) == 2
        finally:
            srv.shutdown()


def test_progress_by_class():
    catalog = _fake_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        ann_csv = Path(tmp) / "ann.csv"
        with open(ann_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "cluster"]).writeheader()

        port, srv = _start_server(catalog, ann_csv)
        try:
            # 3 tips, 1 pedestals, 0 boundary
            _post(port, "/api/annotate", {"filename": "test_000.png", "cluster": "tips"})
            _post(port, "/api/annotate", {"filename": "test_001.png", "cluster": "tips"})
            _post(port, "/api/annotate", {"filename": "test_002.png", "cluster": "tips"})
            _post(port, "/api/annotate", {"filename": "test_003.png", "cluster": "pedestals"})

            data = _get(port, "/api/progress_by_class?target=10")
            by_id = {r["cluster_id"]: r for r in data["per_class"]}
            assert by_id["tips"]["count"] == 3
            assert by_id["tips"]["pct"] == 0.3  # 3/10
            assert by_id["pedestals"]["count"] == 1
            assert by_id["boundary"]["count"] == 0
            assert by_id["boundary"]["pct"] == 0.0
        finally:
            srv.shutdown()


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name in sorted(globals()):
        if not name.startswith("test_"): continue
        fn = globals()[name]
        try:
            fn(); print(f"  PASS  {name}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}"); traceback.print_exc(); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
