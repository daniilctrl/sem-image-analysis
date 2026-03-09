import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def zip_project_for_colab(output_filename="diploma_colab.zip"):
    """
    Упаковывает необходимые файлы (src/ и data/processed/) для загрузки в Google Colab.
    Исходники TIFF (data/raw) и тяжелые модели не включаются для экономии места.
    """
    root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    output_path = root_dir / output_filename
    
    # Что мы хотим положить в архив
    include_paths = [
        root_dir / "src",
        root_dir / "data" / "processed",
        root_dir / "requirements.txt"
    ]
    
    # Считаем количество файлов для прогресс-бара
    total_files = 0
    for path in include_paths:
        if path.is_file(): total_files += 1
        else:
            for _, _, files in os.walk(path): total_files += len(files)
            
    print(f"Archiving {total_files} files into {output_filename}...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Zipping") as pbar:
            for path in include_paths:
                if path.is_file():
                    arcname = path.relative_to(root_dir)
                    zipf.write(path, arcname)
                    pbar.update(1)
                else:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            # Пропускаем кэш питона
                            if "__pycache__" in root or file.endswith(".pyc"):
                                pbar.update(1)
                                continue
                            
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, root_dir)
                            zipf.write(file_path, arcname)
                            pbar.update(1)
                            
    print(f"\nSuccess! Please upload '{output_filename}' to your Google Drive.")

if __name__ == "__main__":
    zip_project_for_colab()
