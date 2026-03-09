$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "Starting Full Diploma Analysis Pipeline"
Write-Host "=========================================="

# 1. Очистка старых данных
Write-Host "[1/4] Cleaning previous processed data..."
Remove-Item -Path "c:\projects\diploma\data\processed\*" -Include *.png,*.csv -Force -ErrorAction SilentlyContinue
Remove-Item -Path "c:\projects\diploma\data\embeddings\*" -Include *.npy,*.csv,*.png -Force -ErrorAction SilentlyContinue

# 2. Нарезка (Full run, stride=256)
Write-Host "[2/4] Running data preparation (slicing TIFFs into tiles)..."
& c:\projects\diploma\.venv\Scripts\python.exe c:\projects\diploma\src\data\data_prep.py

# 3. Извлечение признаков
Write-Host "[3/4] Extracting ResNet50 features (Note: This might take 3-4 hours on CPU)..."
& c:\projects\diploma\.venv\Scripts\python.exe c:\projects\diploma\src\models\feature_extraction.py

# 4. Визуализация UMAP
Write-Host "[4/4] Running UMAP dimensionality reduction..."
& c:\projects\diploma\.venv\Scripts\python.exe c:\projects\diploma\src\visualization\visualize.py

Write-Host "=========================================="
Write-Host "Pipeline completed successfully!"
Write-Host "=========================================="

# 5. ������ �������������
Write-Host "[5/5] Evaluating clustering metrics..."
& c:\projects\diploma\.venv\Scripts\python.exe c:\projects\diploma\src\visualization\evaluate.py

