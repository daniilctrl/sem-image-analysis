"""
Generate SFT labels CSV from tile filenames.

Scans data/processed/*.png, parses filenames, computes quality metrics,
and generates a catalog CSV for annotation and training.
"""

import os
import re
import csv
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "data" / "sft_catalog.csv"

# Quality thresholds
MIN_STD = 10.0  # Minimum pixel std to keep (filters blank tiles)
INFO_BAR_BRIGHTNESS_THRESHOLD = 200  # Bright bottom strip = info bar


def parse_tile_filename(filename: str) -> dict | None:
    """
    Parse tile filename into components.
    
    Patterns:
      {material}_{img_id}_x{X}_y{Y}.png       (e.g. 1-45degree_12_x0_y0.png)
      {material}__{img_id}_x{X}_y{Y}.png      (e.g. 1-45degree__01_x0_y0.png)
    """
    # Try pattern with double underscore first
    m = re.match(r'^(.+?)__(\d+)_x(\d+)_y(\d+)\.png$', filename)
    if m:
        return {
            'material': m.group(1),
            'img_id': m.group(2),
            'x': int(m.group(3)),
            'y': int(m.group(4)),
            'source': f"{m.group(1)}__{m.group(2)}"
        }
    
    # Try pattern with single underscore
    m = re.match(r'^(.+?)_(\d+)_x(\d+)_y(\d+)\.png$', filename)
    if m:
        return {
            'material': m.group(1),
            'img_id': m.group(2),
            'x': int(m.group(3)),
            'y': int(m.group(4)),
            'source': f"{m.group(1)}_{m.group(2)}"
        }
    
    return None


def compute_quality_metrics(img_path: Path) -> dict:
    """Compute quality metrics for a tile."""
    img = Image.open(img_path).convert('L')  # Grayscale
    arr = np.array(img, dtype=np.float32)
    
    h, w = arr.shape
    
    # Overall statistics
    std = float(np.std(arr))
    mean_val = float(np.mean(arr))
    
    # Check bottom 10% for info bar
    bottom_strip = arr[int(h * 0.9):, :]
    bottom_mean = float(np.mean(bottom_strip))
    
    # Check if tile is mostly uniform (low information)
    is_low_info = std < MIN_STD
    
    # Check for info bar (bright strip at bottom)
    has_info_bar = bottom_mean > INFO_BAR_BRIGHTNESS_THRESHOLD
    
    return {
        'std': round(std, 2),
        'mean': round(mean_val, 2),
        'bottom_mean': round(bottom_mean, 2),
        'is_low_info': is_low_info,
        'has_info_bar': has_info_bar,
    }


def generate_catalog():
    """Generate the full tile catalog CSV."""
    print(f"Scanning tiles in {PROCESSED_DIR}...")
    
    png_files = sorted(PROCESSED_DIR.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")
    
    rows = []
    skipped = 0
    
    for i, fpath in enumerate(png_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(png_files)}...")
        
        parsed = parse_tile_filename(fpath.name)
        if parsed is None:
            skipped += 1
            continue
        
        metrics = compute_quality_metrics(fpath)
        
        # Determine if tile should be filtered
        is_trash = metrics['is_low_info'] or metrics['has_info_bar']
        
        rows.append({
            'filename': fpath.name,
            'material': parsed['material'],
            'img_id': parsed['img_id'],
            'source': parsed['source'],
            'x': parsed['x'],
            'y': parsed['y'],
            'std': metrics['std'],
            'mean': metrics['mean'],
            'bottom_mean': metrics['bottom_mean'],
            'is_trash': is_trash,
            'cluster': '',  # To be filled by annotation tool
        })
    
    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['filename', 'material', 'img_id', 'source', 'x', 'y',
                  'std', 'mean', 'bottom_mean', 'is_trash', 'cluster']
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary
    materials = Counter(r['material'] for r in rows)
    trash_count = sum(1 for r in rows if r['is_trash'])
    
    print(f"\n{'='*50}")
    print(f"Catalog saved to: {OUTPUT_CSV}")
    print(f"Total tiles: {len(rows)}")
    print(f"Skipped (unparseable): {skipped}")
    print(f"Marked as trash: {trash_count}")
    print(f"Good tiles: {len(rows) - trash_count}")
    print(f"\nMaterials ({len(materials)}):")
    for mat, count in sorted(materials.items(), key=lambda x: -x[1]):
        print(f"  {mat}: {count} tiles")


if __name__ == '__main__':
    generate_catalog()
