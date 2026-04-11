"""
Загрузка и предобработка данных кристаллической поверхности.

Входной файл: coordinates_and_neighbours_50.xlsx
  - 270510 строк-атомов
  - Столбцы: X Y Z n1 n2 n3 n4 n5
  - Радиус полусферической поверхности: 50 параметров решётки (BCC)

Выход: data/crystal/atoms.parquet (быстрая перезагрузка)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Максимальное число соседей каждого порядка (для BCC-решётки)
MAX_NEIGHBORS = {
    "n1": 8,   # 1-й порядок (ближайшие)
    "n2": 6,   # 2-й порядок
    "n3": 12,  # 3-й порядок
    "n4": 24,  # 4-й порядок
    "n5": 8,   # 5-й порядок
}
NORM_PRODUCT = 8 * 6 * 12 * 24 * 8  # = 110592


def load_excel(xlsx_path: str) -> pd.DataFrame:
    """Загрузка данных из Excel-файла."""
    print(f"Loading data from {xlsx_path} ...")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Стандартизуем имена столбцов
    expected_cols = ["X", "Y", "Z", "n1", "n2", "n3", "n4", "n5"]
    if len(df.columns) == len(expected_cols):
        df.columns = expected_cols
    else:
        raise ValueError(
            f"Expected {len(expected_cols)} columns, got {len(df.columns)}: {list(df.columns)}"
        )

    print(f"  Loaded {len(df):,} atoms")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет вычисляемые признаки:
      - rgb_r, rgb_g, rgb_b  — цветовое кодирование (слайд 9)
      - grayscale            — полутоновое кодирование (слайды 11-16)
      - radius               — расстояние от центра (для фильтрации)
      - n_norm_1..5           — нормированные числа соседей
    """
    # RGB — первые 3 порядка (как указывает руководитель)
    df["rgb_r"] = df["n1"] / MAX_NEIGHBORS["n1"]
    df["rgb_g"] = df["n2"] / MAX_NEIGHBORS["n2"]
    df["rgb_b"] = df["n3"] / MAX_NEIGHBORS["n3"]

    # Полутона серого (инвертированная формула руководителя)
    neighbor_product = df["n1"] * df["n2"] * df["n3"] * df["n4"] * df["n5"]
    df["grayscale"] = 1.0 - neighbor_product / NORM_PRODUCT

    # Расстояние от центра
    df["radius"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2)

    # Нормированные числа соседей (для 5-канальных патчей)
    for col, max_n in MAX_NEIGHBORS.items():
        df[f"{col}_norm"] = df[col] / max_n

    return df


def validate(df: pd.DataFrame) -> None:
    """Проверка корректности загруженных данных."""
    print("\n--- Data Validation ---")
    print(f"  Shape: {df.shape}")
    print(f"  Radius range: [{df['radius'].min():.2f}, {df['radius'].max():.2f}]")

    for col in ["n1", "n2", "n3", "n4", "n5"]:
        vmin, vmax = df[col].min(), df[col].max()
        expected_max = MAX_NEIGHBORS[col]
        status = "✓" if vmax <= expected_max else "✗ EXCEEDS MAX"
        print(f"  {col}: [{vmin}, {vmax}] (max expected: {expected_max}) {status}")

    print(f"  Grayscale range: [{df['grayscale'].min():.4f}, {df['grayscale'].max():.4f}]")
    print(f"  Any NaN: {df.isna().any().any()}")
    print()


def main(args):
    # Загрузка
    df = load_excel(args.input)

    # Производные признаки
    df = add_derived_features(df)

    # Валидация
    validate(df)

    # Сохранение в parquet для быстрой перезагрузки
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "atoms.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Статистика
    print("\n--- Sample (first 5 rows) ---")
    print(df.head().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load crystal surface data from Excel")
    _root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--input",
        type=str,
        default=str(_root / "data" / "crystal" / "coordinates_and_neighbours_50.xlsx"),
        help="Path to the .xlsx data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_root / "data" / "crystal"),
        help="Directory to save processed data",
    )
    args = parser.parse_args()
    main(args)
