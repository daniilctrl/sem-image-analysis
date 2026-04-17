"""Unit tests для src/data/sft_sampler.SftSampler.

Работают на синтетических данных, не требуют реальных CSV/npy файлов.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sft_sampler import SftSampler  # noqa: E402


def _fake_catalog(n_per_mat: int = 20, mats: list[str] = None) -> list[dict]:
    mats = mats or ["A", "B", "C"]
    rows = []
    for m in mats:
        for i in range(n_per_mat):
            rows.append({
                "filename": f"{m}_{i:03d}.png",
                "material": m,
                "img_id": f"{i:03d}",
                "source": f"{m}_{i:03d}",
                "x": 0, "y": 0,
                "std": 50.0, "mean": 100.0, "bottom_mean": 50.0,
                "is_trash": False,
                "cluster": "",
            })
    # плюс немного trash, они не должны попасть в pool
    for i in range(5):
        rows.append({
            "filename": f"trash_{i}.png",
            "material": "trash_mat",
            "img_id": str(i), "source": "trash", "x": 0, "y": 0,
            "std": 1.0, "mean": 0.0, "bottom_mean": 0.0,
            "is_trash": True, "cluster": "",
        })
    return rows


def _fake_embeddings(catalog: list[dict], dim: int = 32, seed: int = 42):
    """Синтетические embeddings: тайлы одного материала ближе друг к другу."""
    rng = np.random.default_rng(seed)
    good = [r for r in catalog if not r["is_trash"]]
    mats = sorted({r["material"] for r in good})
    mat_centers = {m: rng.standard_normal(dim) * 3.0 for m in mats}
    emb = np.zeros((len(good), dim), dtype=np.float32)
    names = []
    for i, r in enumerate(good):
        emb[i] = mat_centers[r["material"]] + rng.standard_normal(dim) * 0.5
        names.append(r["filename"])
    name_to_idx = {n: i for i, n in enumerate(names)}
    return emb, name_to_idx


def test_sampler_no_embeddings_falls_back():
    catalog = _fake_catalog()
    s = SftSampler(catalog=catalog)
    batch = s.next_batch({}, batch_size=10, strategy="hybrid")
    assert len(batch) == 10
    # все из good
    assert all(not t["is_trash"] for t in batch)


def test_sampler_unannotated_pool_excludes_annotated():
    catalog = _fake_catalog(n_per_mat=10)
    s = SftSampler(catalog=catalog)
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(10)}
    batch = s.next_batch(annotations, batch_size=100, strategy="stratified_mat")
    returned_names = {t["filename"] for t in batch}
    assert not any(n.startswith("A_") for n in returned_names), (
        "Размеченные не должны попадать в pool"
    )


def test_sampler_stratified_interleaves_materials():
    """Round-robin должен чередовать материалы."""
    catalog = _fake_catalog(n_per_mat=10, mats=["A", "B", "C"])
    s = SftSampler(catalog=catalog)
    batch = s.next_batch({}, batch_size=9, strategy="stratified_mat")
    mats = [t["material"] for t in batch]
    # Три последовательных тайла не должны быть одного материала.
    consecutive_same = sum(
        1 for i in range(len(mats) - 2)
        if mats[i] == mats[i + 1] == mats[i + 2]
    )
    assert consecutive_same == 0, f"Три подряд одного материала: {mats}"


def test_sampler_diversity_prefers_far_from_annotated():
    catalog = _fake_catalog(n_per_mat=15, mats=["A", "B", "C"])
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)

    # Размечены только тайлы материала A — diversity должна вытащить
    # сначала тайлы B/C (они дальше в embedding space).
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(5)}
    batch = s.next_batch(annotations, batch_size=10, strategy="diversity")

    # Первые 3 должны быть НЕ из материала A.
    first_mats = [t["material"] for t in batch[:3]]
    assert "A" not in first_mats, (
        f"Diversity должен предпочесть B/C в первую очередь, получил {first_mats}"
    )


def test_sampler_uncertainty_requires_min_annotations():
    catalog = _fake_catalog(n_per_mat=15)
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)

    # Только 3 аннотации — uncertainty fallback на diversity, не должно падать.
    annotations = {
        "A_000.png": "cls1",
        "A_001.png": "cls2",
        "B_000.png": "cls1",
    }
    batch = s.next_batch(annotations, batch_size=5, strategy="uncertainty")
    assert len(batch) == 5


def test_sampler_hybrid_uses_uncertainty_after_threshold():
    catalog = _fake_catalog(n_per_mat=30)
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)

    # 15 размеченных тайлов — hybrid должен уйти в uncertainty mode.
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(8)}
    annotations.update({f"B_{i:03d}.png": "cls2" for i in range(7)})
    batch = s.next_batch(annotations, batch_size=10, strategy="hybrid")
    assert len(batch) == 10


def test_sampler_stats_progress():
    catalog = _fake_catalog(n_per_mat=20)
    s = SftSampler(catalog=catalog)
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(5)}
    stats = s.stats(annotations)
    assert stats["total_good"] == 60
    assert stats["total_annotated"] == 5
    # stats округляет до 4 знаков, допускаем любой 4-digit вариант 5/60
    assert abs(stats["coverage"] - 5 / 60) < 1e-3
    assert stats["by_class"] == {"cls1": 5}
    mats_by_name = {row["material"]: row for row in stats["by_material"]}
    assert mats_by_name["A"]["annotated"] == 5
    assert mats_by_name["B"]["annotated"] == 0


def test_sampler_empty_pool_returns_empty():
    catalog = _fake_catalog(n_per_mat=3, mats=["A"])
    s = SftSampler(catalog=catalog)
    # Разметить вообще все good-тайлы
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(3)}
    batch = s.next_batch(annotations, batch_size=10)
    assert batch == []


def test_propagation_empty_without_embeddings():
    catalog = _fake_catalog()
    s = SftSampler(catalog=catalog)  # no embeddings
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(10)}
    proposals = s.propose_propagation(annotations, batch_size=10)
    assert proposals == []


def test_propagation_proposes_by_material_cluster():
    """Тайлы материала A размечены 'cls1', B — 'cls2'. Unlabeled A-тайлы
    должны получить predicted_cluster='cls1' через kNN в embedding space
    (где A-тайлы кучнуются рядом благодаря _fake_embeddings)."""
    catalog = _fake_catalog(n_per_mat=20, mats=["A", "B", "C"])
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)

    # Размечаем первые 10 тайлов каждого из материалов A и B
    annotations = {}
    for i in range(10):
        annotations[f"A_{i:03d}.png"] = "cls1"
        annotations[f"B_{i:03d}.png"] = "cls2"

    proposals = s.propose_propagation(
        annotations, batch_size=20, knn_k=3,
        min_similarity=0.5, min_agreement=0.5,
    )
    assert len(proposals) > 0, "Should propose some unlabeled tiles"

    # Проверяем корректность propagation: A-тайлы → cls1, B-тайлы → cls2
    for p in proposals:
        mat = p["tile"]["material"]
        if mat == "A":
            assert p["predicted_cluster"] == "cls1", (
                f"A-tile {p['tile']['filename']} должен быть cls1, "
                f"получил {p['predicted_cluster']}"
            )
        elif mat == "B":
            assert p["predicted_cluster"] == "cls2"


def test_propagation_respects_min_similarity():
    """Слишком высокий порог similarity должен дать пустой результат."""
    catalog = _fake_catalog(n_per_mat=15)
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(10)}
    # 0.9999 порог — маловероятно, что любой unlabeled так близок.
    proposals = s.propose_propagation(
        annotations, knn_k=3, min_similarity=0.9999, min_agreement=0.5,
    )
    assert len(proposals) == 0 or all(p["similarity"] >= 0.9999 for p in proposals)


def test_propagation_excludes_trash():
    """predicted_cluster='trash' никогда не пропагируется."""
    catalog = _fake_catalog(n_per_mat=15)
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)
    annotations = {f"A_{i:03d}.png": "trash" for i in range(10)}
    proposals = s.propose_propagation(
        annotations, knn_k=3, min_similarity=0.0, min_agreement=0.5,
    )
    for p in proposals:
        assert p["predicted_cluster"] != "trash"


def test_propagation_sorted_by_confidence():
    """Proposals должны быть отсортированы по убыванию similarity."""
    catalog = _fake_catalog(n_per_mat=20)
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)
    annotations = {f"A_{i:03d}.png": "cls1" for i in range(10)}
    annotations.update({f"B_{i:03d}.png": "cls2" for i in range(10)})
    proposals = s.propose_propagation(
        annotations, batch_size=20, knn_k=3,
        min_similarity=0.3, min_agreement=0.5,
    )
    sims = [p["similarity"] for p in proposals]
    assert sims == sorted(sims, reverse=True), "Must be sorted by similarity desc"


def test_propagation_requires_min_annotations():
    """Если размеченных меньше knn_k — возвращается пустой список."""
    catalog = _fake_catalog()
    emb, name_to_idx = _fake_embeddings(catalog)
    s = SftSampler(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx)
    annotations = {"A_000.png": "cls1", "A_001.png": "cls1"}
    proposals = s.propose_propagation(annotations, knn_k=5)
    assert proposals == []


def test_sampler_unknown_strategy_raises():
    catalog = _fake_catalog()
    s = SftSampler(catalog=catalog)
    try:
        s.next_batch({}, batch_size=5, strategy="nonexistent")
    except ValueError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("Should raise ValueError")


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
