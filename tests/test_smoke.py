"""
Smoke-check тесты для дипломного проекта.

Проверяют целостность данных, корректность размерностей эмбеддингов,
согласованность метаданных, и синтаксис ключевых модулей.

Запуск:
  python -m pytest tests/test_smoke.py -v
  python tests/test_smoke.py          # fallback без pytest
"""

import os
import sys
import ast
import csv
import unittest
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# Helpers
# ============================================================

def _count_csv_rows(path: Path) -> int:
    """Count data rows in CSV (excluding header)."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return sum(1 for _ in reader)


def _check_syntax(filepath: Path) -> bool:
    """Check Python file compiles without syntax errors."""
    source = filepath.read_text(encoding="utf-8")
    ast.parse(source, filename=str(filepath))
    return True


# ============================================================
# 1. Crystal Data Integrity
# ============================================================

class TestCrystalData:
    patches_path = ROOT / "data" / "crystal" / "patches" / "patches.npy"
    meta_path = ROOT / "data" / "crystal" / "patches" / "patches_metadata.csv"

    def test_patches_file_exists(self):
        assert self.patches_path.exists(), f"Missing: {self.patches_path}"

    def test_metadata_file_exists(self):
        assert self.meta_path.exists(), f"Missing: {self.meta_path}"

    def test_patches_shape(self):
        """patches.npy должен быть (N, 5, 32, 32)"""
        import numpy as np
        patches = np.load(self.patches_path, mmap_mode="r")
        assert patches.ndim == 4, f"Expected 4D, got {patches.ndim}D"
        N, C, H, W = patches.shape
        assert C == 5, f"Expected 5 channels, got {C}"
        assert H == 32, f"Expected height 32, got {H}"
        assert W == 32, f"Expected width 32, got {W}"

    def test_patches_metadata_count_match(self):
        """Number of patches must equal number of metadata rows."""
        import numpy as np
        n_patches = np.load(self.patches_path, mmap_mode="r").shape[0]
        n_meta = _count_csv_rows(self.meta_path)
        assert n_patches == n_meta, (
            f"patches ({n_patches}) != metadata ({n_meta}). "
            f"Run fix_embeddings_metadata.py or regenerate patches."
        )

    def test_metadata_has_required_columns(self):
        """patches_metadata.csv must have X, Y, Z columns."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        header_lower = [h.strip().lower() for h in header]
        for col in ["x", "y", "z"]:
            assert col in header_lower, f"Missing column '{col}' in metadata. Found: {header}"

    def test_patches_values_in_range(self):
        """Patch values should be in [0, 1] (normalized neighbor counts)."""
        import numpy as np
        patches = np.load(self.patches_path, mmap_mode="r")
        # Check a sample (first 100 patches) to avoid loading entire array
        sample = patches[:min(100, len(patches))]
        assert sample.min() >= 0.0, f"Negative values found: min={sample.min()}"
        assert sample.max() <= 1.0 + 1e-6, f"Values > 1 found: max={sample.max()}"


# ============================================================
# 2. Crystal Embeddings Integrity
# ============================================================

class TestCrystalEmbeddings:
    # extract_crystal_embeddings.py --output_dir defaults to data/crystal/embeddings/
    emb_path = ROOT / "data" / "crystal" / "embeddings" / "crystal_embeddings.npy"
    meta_path = ROOT / "data" / "crystal" / "embeddings" / "embeddings_metadata.csv"

    def test_embeddings_exist(self):
        if not self.emb_path.exists():
            raise unittest.SkipTest("Crystal embeddings not yet generated")

    def test_embeddings_dimension(self):
        """Embeddings should be (N, 512) — ResNet18 encoder dim."""
        if not self.emb_path.exists():
            raise unittest.SkipTest("Crystal embeddings not yet generated")
        import numpy as np
        emb = np.load(self.emb_path)
        assert emb.ndim == 2, f"Expected 2D, got {emb.ndim}D"
        assert emb.shape[1] == 512, f"Expected dim=512, got {emb.shape[1]}"

    def test_embeddings_metadata_sync(self):
        """Embedding count must match metadata rows."""
        if not self.emb_path.exists() or not self.meta_path.exists():
            raise unittest.SkipTest("Crystal embeddings/metadata not yet generated")
        import numpy as np
        n_emb = np.load(self.emb_path).shape[0]
        n_meta = _count_csv_rows(self.meta_path)
        assert n_emb == n_meta, f"embeddings ({n_emb}) != metadata ({n_meta})"

    def test_embeddings_l2_normalized(self):
        """Embeddings should be approximately L2-normalized."""
        if not self.emb_path.exists():
            raise unittest.SkipTest("Crystal embeddings not yet generated")
        import numpy as np
        emb = np.load(self.emb_path)
        norms = np.linalg.norm(emb[:100], axis=1)
        assert np.allclose(norms, 1.0, atol=0.02), (
            f"Embeddings not L2-normalized. Norms range: [{norms.min():.4f}, {norms.max():.4f}]"
        )


# ============================================================
# 3. SEM Data Integrity
# ============================================================

class TestSEMData:
    meta_path = ROOT / "data" / "processed" / "tiles_metadata.csv"
    baseline_emb = ROOT / "data" / "embeddings" / "resnet50_embeddings.npy"

    def test_metadata_exists(self):
        assert self.meta_path.exists(), f"Missing: {self.meta_path}"

    def test_baseline_embeddings_exist(self):
        assert self.baseline_emb.exists(), f"Missing: {self.baseline_emb}"

    def test_baseline_dimension(self):
        """Baseline embeddings should be (N, 2048) — ResNet50 encoder dim."""
        import numpy as np
        emb = np.load(self.baseline_emb)
        assert emb.ndim == 2, f"Expected 2D, got {emb.ndim}D"
        assert emb.shape[1] == 2048, f"Expected dim=2048, got {emb.shape[1]}"

    def test_baseline_metadata_count_match(self):
        """Baseline embedding count must match SEM metadata rows."""
        import numpy as np
        n_emb = np.load(self.baseline_emb).shape[0]
        n_meta = _count_csv_rows(self.meta_path)
        assert n_emb == n_meta, f"embeddings ({n_emb}) != metadata ({n_meta})"

    def test_sem_metadata_has_material_column(self):
        """tiles_metadata.csv should have a material/label column for evaluation."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        header_lower = [h.strip().lower() for h in header]
        # Check for material or filename (from which material is extracted)
        has_material = "material" in header_lower
        has_source = "source_image" in header_lower or "filename" in header_lower or "image_path" in header_lower
        assert has_material or has_source, (
            f"No material/source column in SEM metadata. Found: {header}"
        )

    def test_simclr_embeddings_if_present(self):
        """If SimCLR embeddings exist, verify shape matches baseline count."""
        simclr_path = ROOT / "data" / "embeddings" / "simclr" / "finetuned_embeddings.npy"
        if not simclr_path.exists():
            raise unittest.SkipTest("SimCLR embeddings not yet generated")
        import numpy as np
        emb = np.load(simclr_path)
        n_baseline = np.load(self.baseline_emb).shape[0]
        assert emb.shape[0] == n_baseline, (
            f"SimCLR count ({emb.shape[0]}) != baseline count ({n_baseline})"
        )

    def test_byol_embeddings_if_present(self):
        """If BYOL embeddings exist, verify shape matches baseline count."""
        byol_path = ROOT / "data" / "embeddings" / "byol" / "finetuned_embeddings.npy"
        if not byol_path.exists():
            raise unittest.SkipTest("BYOL embeddings not yet generated")
        import numpy as np
        emb = np.load(byol_path)
        n_baseline = np.load(self.baseline_emb).shape[0]
        assert emb.shape[0] == n_baseline, (
            f"BYOL count ({emb.shape[0]}) != baseline count ({n_baseline})"
        )

    def test_load_aligned_data_baseline(self):
        """Baseline alignment through embedding_names.csv should work."""
        from src.evaluation.eval_utils import load_aligned_data

        emb, df = load_aligned_data("Baseline (ImageNet)")
        assert len(emb) == len(df), f"aligned baseline mismatch: {len(emb)} != {len(df)}"
        assert len(df) > 0, "aligned baseline dataset is empty"
        for col in ["tile_name", "source_image", "mag"]:
            assert col in df.columns, f"Missing column '{col}' after baseline alignment"

    def test_simclr_names_file_if_present(self):
        """SimCLR names contract: exact match if file exists, fallback otherwise."""
        simclr_path = ROOT / "data" / "embeddings" / "simclr" / "finetuned_embeddings.npy"
        names_path = ROOT / "data" / "embeddings" / "simclr" / "finetuned_embedding_names.csv"
        if not simclr_path.exists():
            raise unittest.SkipTest("SimCLR embeddings not yet generated")
        import numpy as np
        if names_path.exists():
            n_emb = np.load(simclr_path).shape[0]
            n_names = _count_csv_rows(names_path)
            assert n_emb == n_names, f"SimCLR embeddings ({n_emb}) != names rows ({n_names})"
        else:
            import warnings
            from src.evaluation.eval_utils import load_aligned_data

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emb, df = load_aligned_data("SimCLR (30 ep)")
            assert len(emb) == len(df), f"aligned SimCLR mismatch: {len(emb)} != {len(df)}"
            assert any("Falling back" in str(w.message) for w in caught), (
                "Expected fallback warning when SimCLR names file is missing"
            )

    def test_byol_names_file_if_present(self):
        """BYOL names contract: exact match if file exists, fallback otherwise."""
        byol_path = ROOT / "data" / "embeddings" / "byol" / "finetuned_embeddings.npy"
        names_path = ROOT / "data" / "embeddings" / "byol" / "finetuned_embedding_names.csv"
        if not byol_path.exists():
            raise unittest.SkipTest("BYOL embeddings not yet generated")
        import numpy as np
        if names_path.exists():
            n_emb = np.load(byol_path).shape[0]
            n_names = _count_csv_rows(names_path)
            assert n_emb == n_names, f"BYOL embeddings ({n_emb}) != names rows ({n_names})"
        else:
            import warnings
            from src.evaluation.eval_utils import load_aligned_data

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emb, df = load_aligned_data("BYOL (30 ep)")
            assert len(emb) == len(df), f"aligned BYOL mismatch: {len(emb)} != {len(df)}"
            assert any("Falling back" in str(w.message) for w in caught), (
                "Expected fallback warning when BYOL names file is missing"
            )

    def test_load_aligned_data_simclr_if_present(self):
        """SimCLR aligned loader should return synchronized embeddings and metadata."""
        simclr_path = ROOT / "data" / "embeddings" / "simclr" / "finetuned_embeddings.npy"
        if not simclr_path.exists():
            raise unittest.SkipTest("SimCLR embeddings not yet generated")

        from src.evaluation.eval_utils import load_aligned_data

        emb, df = load_aligned_data("SimCLR (30 ep)")
        assert len(emb) == len(df), f"aligned SimCLR mismatch: {len(emb)} != {len(df)}"
        assert len(df) > 0, "aligned SimCLR dataset is empty"
        for col in ["tile_name", "source_image", "mag"]:
            assert col in df.columns, f"Missing column '{col}' after SimCLR alignment"

    def test_load_aligned_data_byol_if_present(self):
        """BYOL aligned loader should return synchronized embeddings and metadata."""
        byol_path = ROOT / "data" / "embeddings" / "byol" / "finetuned_embeddings.npy"
        if not byol_path.exists():
            raise unittest.SkipTest("BYOL embeddings not yet generated")

        from src.evaluation.eval_utils import load_aligned_data

        emb, df = load_aligned_data("BYOL (30 ep)")
        assert len(emb) == len(df), f"aligned BYOL mismatch: {len(emb)} != {len(df)}"
        assert len(df) > 0, "aligned BYOL dataset is empty"
        for col in ["tile_name", "source_image", "mag"]:
            assert col in df.columns, f"Missing column '{col}' after BYOL alignment"


# ============================================================
# 4. Module Syntax Checks
# ============================================================

class TestModuleSyntax:
    """Verify all key Python modules parse without syntax errors."""

    modules = [
        "src/crystal/train_crystal.py",
        "src/crystal/model_crystal.py",
        "src/crystal/patch_generator.py",
        "src/crystal/dataset_crystal.py",
        "src/crystal/augmentations_crystal.py",
        "src/crystal/extract_crystal_embeddings.py",
        "src/crystal/optimize_clusters.py",
        "src/crystal/analyze_miller_indices.py",
        "src/crystal/miller_utils.py",
        "src/crystal/retrieve_crystal.py",
        "src/models/deep_clustering/model.py",
        "src/models/deep_clustering/train.py",
        "src/models/deep_clustering/train_byol.py",
        "src/models/deep_clustering/loss.py",
        "src/models/deep_clustering/augmentations.py",
        "src/models/deep_clustering/extract_simclr_embeddings.py",
        "src/evaluation/eval_utils.py",
        "src/evaluation/cross_scale_comparison.py",
        "src/evaluation/scale_invariance_metrics.py",
        "src/evaluation/run_sem_evaluation.py",
        "src/evaluation/evaluate_sic_clustering.py",
    ]

    def test_all_modules_parse(self):
        errors = []
        for mod in self.modules:
            path = ROOT / mod
            if not path.exists():
                errors.append(f"MISSING: {mod}")
                continue
            try:
                _check_syntax(path)
            except SyntaxError as e:
                errors.append(f"SYNTAX ERROR in {mod}: {e}")
        assert not errors, "Module issues:\n" + "\n".join(errors)


# ============================================================
# 5. Miller Utils Unit Tests
# ============================================================

class TestMillerUtils:
    """Test miller_utils.py classification logic."""

    def test_import(self):
        from src.crystal.miller_utils import FAMILIES, FAMILY_NAMES
        # После расширения набора по рис. 3 [Никифоров 2009] семейств должно
        # быть значительно больше, чем в первой итерации (9). Ожидаем ≥ 20,
        # плюс Vicinal/Mixed в конце FAMILY_NAMES.
        assert len(FAMILIES) >= 20, f"Expected >=20 Miller families, got {len(FAMILIES)}"
        assert FAMILY_NAMES[-1] == "Vicinal/Mixed"
        assert len(FAMILY_NAMES) == len(FAMILIES) + 1

    def test_canonical_core_families(self):
        """Core Miller families (100, 110, 111) must classify correctly."""
        import numpy as np
        from src.crystal.miller_utils import assign_miller_labels

        vectors = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ], dtype=np.float64)

        labels, names = assign_miller_labels(vectors, tol_deg=6.0)
        assert names[labels[0]] == "{100}", f"[1,0,0] -> {names[labels[0]]}"
        assert names[labels[1]] == "{110}", f"[1,1,0] -> {names[labels[1]]}"
        assert names[labels[2]] == "{111}", f"[1,1,1] -> {names[labels[2]]}"
        assert names[labels[3]] == "{100}", f"[0,0,1] -> {names[labels[3]]}"

    def test_extended_families_present(self):
        """Ensure extended families beyond the initial 9 are registered."""
        from src.crystal.miller_utils import FAMILIES

        # Семейства, добавленные в расширенной версии по рис. 3 Никифорова 2009:
        expected_extra = {"{311}", "{331}", "{510}", "{531}", "{521}", "{711}"}
        missing = expected_extra - set(FAMILIES.keys())
        assert not missing, f"Missing extended families: {missing}"

    def test_vicinal_vector(self):
        """A vector far from all canonical families -> Vicinal/Mixed."""
        import numpy as np
        from src.crystal.miller_utils import assign_miller_labels

        vectors = np.array([[0.7, 0.5, 0.1]])
        labels, names = assign_miller_labels(vectors, tol_deg=1.0)
        assert names[labels[0]] == "Vicinal/Mixed"


# ============================================================
# Standalone runner (no pytest needed)
# ============================================================

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestCrystalData, TestCrystalEmbeddings, TestSEMData,
        TestModuleSyntax, TestMillerUtils,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for cls in test_classes:
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)
            full_name = f"{cls.__name__}.{name}"
            try:
                method()
                print(f"  PASS  {full_name}")
                passed += 1
            except Exception as e:
                if "skip" in str(type(e).__name__).lower() or "skip" in str(e).lower():
                    print(f"  SKIP  {full_name}: {e}")
                    skipped += 1
                else:
                    print(f"  FAIL  {full_name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)
