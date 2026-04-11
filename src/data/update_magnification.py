"""
Скрипт для извлечения увеличения (magnification) из метаданных TIFF-файлов
микроскопа Zeiss Merlin SEM и обновления tiles_metadata.csv.

Zeiss Merlin хранит данные в проприетарном теге 34118, ключ 'ap_mag'.
Формат значения: ('Mag', '2.43 K X') или ('Mag', '20.00 K X')
"""
import re
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm


def parse_mag_string(mag_str):
    """
    Парсит строку увеличения Zeiss Merlin.
    '2.43 K X' -> 2430.0
    '20.00 K X' -> 20000.0
    '100 X' -> 100.0
    """
    if not isinstance(mag_str, str):
        return None
    
    mag_str = mag_str.strip()
    
    # Формат "XX.XX K X" (килократные увеличения)
    match = re.match(r'([\d.]+)\s*K\s*X', mag_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 1000
    
    # Формат "XX X" (просто кратность)
    match = re.match(r'([\d.]+)\s*X', mag_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def extract_magnification_from_tiff(tiff_path):
    """Извлекает увеличение из проприетарных тегов Zeiss Merlin SEM."""
    try:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            page = tif.pages[0]
            tag = page.tags.get(34118)
            if tag is None:
                return None
            
            val = tag.value
            if isinstance(val, dict):
                ap_mag = val.get('ap_mag')
                if ap_mag and isinstance(ap_mag, tuple) and len(ap_mag) >= 2:
                    return parse_mag_string(str(ap_mag[1]))
            
            return None
    except Exception as e:
        print(f"Error reading {tiff_path}: {e}")
        return None


def update_metadata():
    """Обновляет tiles_metadata.csv, добавляя реальные увеличения из TIFF."""
    _root = Path(__file__).resolve().parents[2]
    raw_dir = _root / "data" / "raw"
    csv_path = _root / "data" / "processed" / "tiles_metadata.csv"
    
    # 1. Считываем CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tiles from metadata.")
    
    # 2. Собираем увеличения из TIFF-файлов
    tiff_files = list(raw_dir.rglob("*.tif"))
    print(f"Found {len(tiff_files)} TIFF files.")
    
    source_to_mag = {}
    for tif_path in tqdm(tiff_files, desc="Reading TIFF metadata"):
        stem = tif_path.stem  # например "1-45degree__01"
        mag = extract_magnification_from_tiff(tif_path)
        source_to_mag[stem] = mag
    
    # 3. Статистика
    found = sum(1 for v in source_to_mag.values() if v is not None)
    print(f"\nExtracted magnification for {found}/{len(source_to_mag)} images.")
    
    # Уникальные увеличения
    mags = sorted(set(v for v in source_to_mag.values() if v is not None))
    print(f"Unique magnifications: {mags}")
    
    # 4. Обновляем DataFrame
    df['mag'] = df['source_image'].map(source_to_mag)
    
    # Добавляем удобную колонку material (часть до __)
    df['material'] = df['source_image'].str.split('__').str[0]
    
    # 5. Показываем cross-scale статистику
    print(f"\n--- Cross-Scale Analysis ---")
    for material in sorted(df['material'].unique()):
        sub = df[df['material'] == material]
        unique_mags = sorted(sub['mag'].dropna().unique())
        n_images = sub['source_image'].nunique()
        if len(unique_mags) > 1:
            print(f"  {material}: {n_images} images, magnifications = {unique_mags}")
    
    # 6. Сохраняем
    df.to_csv(csv_path, index=False)
    print(f"\nUpdated {csv_path}")
    print(f"Null mag count: {df['mag'].isna().sum()}/{len(df)}")


if __name__ == "__main__":
    update_metadata()
