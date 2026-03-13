# SEM Image Analysis Pipeline

This repository contains an end-to-end pipeline for analyzing Scanning Electron Microscopy (SEM) images, specifically focused on achieving scale-invariant feature extraction using self-supervised learning methods (SimCLR, BYOL).

## Core Mechanisms
1. **Data Processing**: Specialized logic to break down massive TIFF images into smaller tiles and extract real-world magnification metadata from Zeiss Merlin tags (Tag 34118).
2. **Models**: ResNet-based SimCLR and BYOL models trained on aggressive augmentations to understand semantic features of materials, rather than just zooming details.
3. **Evaluation Framework**: A Cross-Scale Retrieval Test that calculates Precision@K to measure a model's ability to consistently identify similar material structures across varying magnifications.
4. **Tools**: FAISS integration for fast vector embeddings search and visualization scripts (UMAP/Silhouette Score).

## Clustering Targets (Silicon Carbide)
When applying models to SiC (e.g., *Serkov & Luchinin* arrays), the goal is to cluster images based on distinct micro-emitter geometries:
*   **Cluster A (Pedestals & Tips):** Images showing individual pedestals (base ~0.3 µm) topped with sharp nano-tips (radius ~30 nm, height ~2 µm).
*   **Cluster B (Defective Tips):** Smoothened or over-etched cones lacking a sharp emitter tip.
*   **Cluster C (High-Density Arrays):** Macro-views showing periodic arrays or rings (e.g., 200 µm diameter) of micro-emitters.
*   **Cluster D (Empty/Base Material):** Bare SiC substrate or aggressively etched regions.

The model must achieve *Scale Invariance*: recognizing that an array of dots (Cluster C) is physically composed of individual pedestal tips (Cluster A) when zoomed in.

## Project Structure
- `data/`: `raw`, `processed`, `embeddings`, `results`
- `src/`: 
  - `data_prep/`: Tiling and metadata extraction logic.
  - `models/`: Self-supervised model definitions and training scripts.
  - `visualization/`: UMAP and clustering metric evaluation.
- `notebooks/`: Optional Jupyter notebooks for interactive analysis.
- `models/` & `evaluate/`: Utilities for testing retrieval and scale consistency.

## Usage
Run the overall pipeline using the powershell script (if applicable) or by navigating into the `/src/` scripts respectively. Refer to `.cursorrules` for coding standards.
