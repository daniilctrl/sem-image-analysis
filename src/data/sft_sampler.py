"""Smart ordering strategies для SFT annotation.

Проблема: при разметке 23k тайлов случайный shuffle даёт плохую стратегию.
Хочется: следующий показанный тайл должен быть (а) максимально разнообразным
относительно уже размеченных, и (б) желательно пришёл из недопредставленного
материала.

Этот модуль реализует несколько стратегий выбора порядка показа:

  - `random`          — текущий shuffle (baseline).
  - `stratified_mat`  — по круговому round-robin между материалами, убирая
                         overrepresentation. Не требует эмбеддингов.
  - `diversity`       — max-min cosine distance к уже размеченному множеству
                         в baseline-эмбеддингах. Требует embeddings.
  - `uncertainty`     — выбирать в первую очередь тайлы, чьи ближайшие
                         размеченные соседи имеют разные метки (конфликт).
                         Требует embeddings и >= 2 размеченных тайлов.

Дефолтная стратегия — `hybrid`: diversity при наличии эмбеддингов,
stratified_mat в fallback.

Использование как библиотеки (вне annotate_tiles.py):

    from src.data.sft_sampler import SftSampler
    sampler = SftSampler.from_defaults()
    next_batch = sampler.next_batch(annotations_dict, batch_size=50)

Возвращает список tile dicts в порядке предлагаемой разметки.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG = _PROJECT_ROOT / "data" / "sft_catalog.csv"
_DEFAULT_EMB = _PROJECT_ROOT / "data" / "embeddings" / "resnet50_embeddings.npy"
_DEFAULT_EMB_NAMES = _PROJECT_ROOT / "data" / "embeddings" / "embedding_names.csv"


def _load_catalog(path: Path) -> list[dict]:
    """Читает sft_catalog.csv и приводит is_trash к bool."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["is_trash"] = row.get("is_trash", "False") == "True"
            rows.append(row)
    return rows


def _load_baseline_embeddings(
    emb_path: Path, names_path: Path,
) -> tuple[Optional[np.ndarray], Optional[dict[str, int]]]:
    """Возвращает (embeddings, name_to_idx) либо (None, None), если отсутствуют."""
    if not emb_path.exists() or not names_path.exists():
        return None, None
    emb = np.load(emb_path, mmap_mode="r")
    name_to_idx: dict[str, int] = {}
    with open(names_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            name_to_idx[row["tile_name"]] = i
    if len(name_to_idx) != emb.shape[0]:
        # Stale names file — отказываемся использовать, fallback на metadata.
        import warnings
        warnings.warn(
            f"embedding_names.csv ({len(name_to_idx)} rows) != embeddings "
            f"({emb.shape[0]} rows). Skipping embedding-based ordering; "
            f"falling back to stratified_mat. "
            f"Regenerate via scripts/regenerate_embedding_names.py.",
            stacklevel=2,
        )
        return None, None
    return emb, name_to_idx


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@dataclass
class SftSampler:
    """Сэмплер тайлов для разметки.

    Состояние загружается один раз (catalog + опциональные embeddings),
    затем каждый вызов `next_batch(annotations, batch_size)` возвращает
    упорядоченный список tile dict'ов.
    """

    catalog: list[dict]
    # good_idx: индексы в catalog, у которых is_trash=False
    good_idx: np.ndarray = field(init=False)
    # tile_name -> position in catalog
    name_to_catalog_idx: dict[str, int] = field(init=False)
    # Опциональные эмбеддинги (baseline ResNet50)
    embeddings: Optional[np.ndarray] = None
    emb_name_to_idx: Optional[dict[str, int]] = None
    seed: int = 42

    def __post_init__(self) -> None:
        self.good_idx = np.array(
            [i for i, r in enumerate(self.catalog) if not r["is_trash"]],
            dtype=np.int64,
        )
        self.name_to_catalog_idx = {
            r["filename"]: i for i, r in enumerate(self.catalog)
        }
        if self.embeddings is not None:
            # Кешируем L2-normalized копию для cosine-операций.
            self._normed = _l2_normalize(np.asarray(self.embeddings, dtype=np.float32))
        else:
            self._normed = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(
        cls,
        catalog_path: Path | None = None,
        emb_path: Path | None = None,
        emb_names_path: Path | None = None,
        seed: int = 42,
    ) -> "SftSampler":
        catalog_path = Path(catalog_path or _DEFAULT_CATALOG)
        emb_path = Path(emb_path or _DEFAULT_EMB)
        emb_names_path = Path(emb_names_path or _DEFAULT_EMB_NAMES)

        catalog = _load_catalog(catalog_path)
        emb, name_to_idx = _load_baseline_embeddings(emb_path, emb_names_path)
        return cls(catalog=catalog, embeddings=emb, emb_name_to_idx=name_to_idx,
                   seed=seed)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _annotated_indices_in_embeddings(
        self, annotations: dict[str, str],
    ) -> np.ndarray:
        """Индексы в self.embeddings для тайлов из annotations.keys().

        Только те, у кого есть baseline-эмбеддинг.
        """
        if self.emb_name_to_idx is None:
            return np.empty(0, dtype=np.int64)
        idx = [
            self.emb_name_to_idx[n]
            for n in annotations
            if n in self.emb_name_to_idx
        ]
        return np.asarray(idx, dtype=np.int64)

    def _unannotated_good(self, annotations: dict[str, str]) -> np.ndarray:
        """Индексы в catalog для good-тайлов, которых ещё нет в annotations."""
        annotated = set(annotations.keys())
        keep = [
            ci for ci in self.good_idx
            if self.catalog[ci]["filename"] not in annotated
        ]
        return np.asarray(keep, dtype=np.int64)

    def _rank_random(self, pool: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(pool))
        return pool[order]

    def _rank_stratified_mat(self, pool: np.ndarray) -> np.ndarray:
        """Round-robin по материалам — каждый материал вносит один тайл по очереди."""
        rng = np.random.default_rng(self.seed)
        buckets: dict[str, list[int]] = {}
        for ci in pool:
            m = self.catalog[int(ci)]["material"]
            buckets.setdefault(m, []).append(int(ci))
        # Перемешать внутри каждого bucket для разнообразия.
        for m in buckets:
            rng.shuffle(buckets[m])
        # Round-robin
        ordered: list[int] = []
        materials = list(buckets.keys())
        rng.shuffle(materials)
        while any(buckets.values()):
            for m in materials:
                if buckets[m]:
                    ordered.append(buckets[m].pop())
        return np.asarray(ordered, dtype=np.int64)

    def _rank_diversity(
        self, pool: np.ndarray, annotations: dict[str, str],
    ) -> np.ndarray:
        """Max-min cosine distance к уже размеченным.

        Если размеченных нет (cold start) — fallback на stratified_mat, потому
        что считать diversity «от 0 точек» бессмысленно.
        """
        if self._normed is None or self.emb_name_to_idx is None:
            return self._rank_stratified_mat(pool)

        annot_emb_idx = self._annotated_indices_in_embeddings(annotations)
        if len(annot_emb_idx) == 0:
            return self._rank_stratified_mat(pool)

        # Pool → массив emb-индексов (оставляем только тайлы, которые есть
        # в emb_name_to_idx; остальные дополним в конец через stratified_mat).
        pool_rows = []
        pool_catalog_idx = []
        missing_pool = []
        for ci in pool:
            name = self.catalog[int(ci)]["filename"]
            ei = self.emb_name_to_idx.get(name)
            if ei is None:
                missing_pool.append(int(ci))
            else:
                pool_rows.append(ei)
                pool_catalog_idx.append(int(ci))

        if not pool_rows:
            return self._rank_stratified_mat(pool)

        pool_normed = self._normed[pool_rows]            # (P, D)
        annot_normed = self._normed[annot_emb_idx]       # (A, D)

        # cosine sim матрица (P, A); distance = 1 - sim; хотим max min-distance
        # → min min-distance НЕ хотим, хотим max min-distance.
        # max_cos_sim = ближайший размеченный сосед (cos similarity).
        sim = pool_normed @ annot_normed.T               # (P, A)
        max_sim = sim.max(axis=1)                        # (P,)
        # Большая max_sim = тайл близок к уже размеченным (плохо для diversity).
        # Сортируем по возрастанию max_sim → первым идёт самый «далёкий».
        order = np.argsort(max_sim, kind="stable")
        ordered_with_emb = np.asarray(pool_catalog_idx, dtype=np.int64)[order]

        # Тайлы без эмбеддинга добавляем в конец через stratified_mat.
        tail = self._rank_stratified_mat(np.asarray(missing_pool, dtype=np.int64)) \
            if missing_pool else np.empty(0, dtype=np.int64)
        return np.concatenate([ordered_with_emb, tail])

    def _rank_uncertainty(
        self, pool: np.ndarray, annotations: dict[str, str],
    ) -> np.ndarray:
        """Uncertainty = вариативность меток среди k ближайших размеченных.

        Для каждого pool-тайла: находим K=10 ближайших уже размеченных по
        cosine, считаем entropy меток среди них. Высокая entropy = модель
        «не знает» → высокий приоритет для разметки.

        Fallback на diversity при недостаточном числе меток.
        """
        if self._normed is None or self.emb_name_to_idx is None:
            return self._rank_stratified_mat(pool)
        annot_emb_idx = self._annotated_indices_in_embeddings(annotations)
        if len(annot_emb_idx) < 10:
            return self._rank_diversity(pool, annotations)

        K_NN = 10
        pool_rows = []
        pool_catalog_idx = []
        missing_pool = []
        for ci in pool:
            name = self.catalog[int(ci)]["filename"]
            ei = self.emb_name_to_idx.get(name)
            if ei is None:
                missing_pool.append(int(ci))
            else:
                pool_rows.append(ei)
                pool_catalog_idx.append(int(ci))

        if not pool_rows:
            return self._rank_stratified_mat(pool)

        pool_normed = self._normed[pool_rows]
        annot_normed = self._normed[annot_emb_idx]
        sim = pool_normed @ annot_normed.T
        # top-K ближайших размеченных
        top_k_idx = np.argpartition(-sim, kth=K_NN - 1, axis=1)[:, :K_NN]

        # entropy меток в top-K
        annot_labels = [annotations[self._tile_name_of_emb_idx(ei)]
                        for ei in annot_emb_idx]
        annot_labels_arr = np.asarray(annot_labels)

        entropies = np.empty(len(pool_rows), dtype=np.float64)
        for i in range(len(pool_rows)):
            nn_labels = annot_labels_arr[top_k_idx[i]]
            _, counts = np.unique(nn_labels, return_counts=True)
            probs = counts / counts.sum()
            entropies[i] = -float(np.sum(probs * np.log(probs + 1e-12)))

        # Сортируем по убыванию entropy → первым идёт самый «неопределённый».
        order = np.argsort(-entropies, kind="stable")
        ordered_with_emb = np.asarray(pool_catalog_idx, dtype=np.int64)[order]

        tail = self._rank_stratified_mat(np.asarray(missing_pool, dtype=np.int64)) \
            if missing_pool else np.empty(0, dtype=np.int64)
        return np.concatenate([ordered_with_emb, tail])

    def _tile_name_of_emb_idx(self, emb_idx: int) -> str:
        # Обратный lookup: emb_idx -> tile_name. Кешируется при первом вызове.
        if not hasattr(self, "_emb_idx_to_name"):
            self._emb_idx_to_name = {
                v: k for k, v in (self.emb_name_to_idx or {}).items()
            }
        return self._emb_idx_to_name[int(emb_idx)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_batch(
        self,
        annotations: dict[str, str],
        batch_size: int = 50,
        strategy: str = "hybrid",
    ) -> list[dict]:
        """Возвращает упорядоченный список тайлов для разметки.

        Аргументы:
            annotations: {filename: cluster} уже размеченных тайлов.
            batch_size: сколько тайлов вернуть.
            strategy: 'random' | 'stratified_mat' | 'diversity' |
                      'uncertainty' | 'hybrid'.
                'hybrid' = uncertainty, если ≥ 10 размеченных;
                           diversity, если эмбеддинги есть;
                           иначе stratified_mat.
        """
        pool = self._unannotated_good(annotations)
        if len(pool) == 0:
            return []

        if strategy == "random":
            ranked = self._rank_random(pool)
        elif strategy == "stratified_mat":
            ranked = self._rank_stratified_mat(pool)
        elif strategy == "diversity":
            ranked = self._rank_diversity(pool, annotations)
        elif strategy == "uncertainty":
            ranked = self._rank_uncertainty(pool, annotations)
        elif strategy == "hybrid":
            if self._normed is None:
                ranked = self._rank_stratified_mat(pool)
            elif len(annotations) >= 10:
                ranked = self._rank_uncertainty(pool, annotations)
            else:
                ranked = self._rank_diversity(pool, annotations)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        result = [self.catalog[int(ci)] for ci in ranked[:batch_size]]
        return result

    def stats(self, annotations: dict[str, str]) -> dict:
        """Сводка: общий прогресс + покрытие по материалам и классам."""
        good = [self.catalog[int(i)] for i in self.good_idx]
        annotated_names = set(annotations.keys())
        annotated_good = [t for t in good if t["filename"] in annotated_names]

        from collections import Counter

        by_material = Counter(t["material"] for t in good)
        by_material_annot = Counter(
            t["material"] for t in annotated_good
        )
        by_class = Counter(annotations.values())

        per_material = [
            {
                "material": m,
                "total": by_material[m],
                "annotated": by_material_annot.get(m, 0),
                "coverage": round(by_material_annot.get(m, 0) / by_material[m], 4),
            }
            for m in sorted(by_material)
        ]

        return {
            "total_good": len(good),
            "total_annotated": len(annotated_good),
            "coverage": round(len(annotated_good) / max(len(good), 1), 4),
            "by_class": dict(by_class),
            "by_material": per_material,
            "has_embeddings": self._normed is not None,
        }
