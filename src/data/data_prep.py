import os
import glob
import pandas as pd
import numpy as np
import cv2
import tifffile
from pathlib import Path

def extract_metadata(tiff_path):
    """Извлекает масштаб и физические размеры из метаданных TIFF (SEM)."""
    # В сканирующей электронной микроскопии масштаб часто пишется в тегах image_description
    # или специальных полях. Попытаемся достать текст.
    metadata = {'scale': None, 'magnification': None, 'pixel_size': None}
    
    with tifffile.TiffFile(tiff_path) as tif:
        for page in tif.pages:
            try:
                # Читаем теги
                desc = page.tags.get('ImageDescription')
                if desc is not None:
                    text = desc.value
                    if isinstance(text, str):
                        # Ищем ключевые слова в описании (у разных приборов разные форматы)
                        lines = text.split('\r\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('Magnification='):
                                metadata['magnification'] = line.split('=')[1]
                            elif line.startswith('PixelSize='):
                                metadata['pixel_size'] = line.split('=')[1]
                            elif line.startswith('Scale='):
                                metadata['scale'] = line.split('=')[1]
            except Exception as e:
                pass
    
    return metadata

def process_images(input_dir, output_dir, tile_size=256, stride=256):
    """
    Нарезает все TIFF изображения в input_dir на тайлы размером tile_size x tile_size
    с заданным шагом stride и сохраняет их в output_dir.
    Возвращает DataFrame с метаданными всех тайлов.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_path.rglob("*.tif*"))
    records = []
    
    print(f"Found {len(tiff_files)} TIFF files in {input_dir}")
    
    for tiff_file in tiff_files:
        try:
            # Читаем метаданные
            metadata = extract_metadata(str(tiff_file))
            
            # Читаем изображение (в градациях серого)
            img = cv2.imread(str(tiff_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image {tiff_file}")
                continue
                
            h, w = img.shape
            filename_base = Path(tiff_file).stem
            
            # Нарезка
            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    tile = img[y:y+tile_size, x:x+tile_size]
                    
                    # Отбрасываем информационную панель РЭМ, которая часто бывает внизу
                    # или полностью черные/белые тайлы (пустые)
                    if np.std(tile) < 5.0:
                        continue
                        
                    tile_name = f"{filename_base}_x{x}_y{y}.png"
                    tile_path = output_path / tile_name
                    
                    cv2.imwrite(str(tile_path), tile)
                    
                    records.append({
                        'tile_name': tile_name,
                        'source_image': filename_base,
                        'x': x,
                        'y': y,
                        'mag': metadata.get('magnification'),
                        'pixel_size': metadata.get('pixel_size'),
                        'scale': metadata.get('scale')
                    })
        except Exception as e:
            print(f"Error processing {tiff_file}: {str(e)}")
            
    df = pd.DataFrame(records)
    csv_path = output_path / "tiles_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")
    print(f"Total tiles generated: {len(df)}")
    return df

if __name__ == "__main__":
    input_directory = r"c:\projects\diploma\data\raw"
    output_directory = r"c:\projects\diploma\data\processed"
    
    # Полный прогон датасета (нарезка без перекрытия)
    process_images(input_directory, output_directory, tile_size=256, stride=256)
