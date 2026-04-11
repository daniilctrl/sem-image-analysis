import os
import gradio as gr
import pandas as pd
import numpy as np
import cv2
import faiss
from pathlib import Path

# ==================== НАСТРОЙКИ ====================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
META_FILE = PROCESSED_DATA_DIR / "tiles_metadata.csv"

# Доступные наборы эмбеддингов
EMBEDDING_CONFIGS = {
    "Baseline (ImageNet ResNet50)": {
        "emb_file": EMBEDDINGS_DIR / "resnet50_embeddings.npy",
        "names_file": EMBEDDINGS_DIR / "embedding_names.csv",
    },
    "SimCLR v1 (batch=64, temp=0.5)": {
        "emb_file": EMBEDDINGS_DIR / "finetuned_embeddings.npy",
        "names_file": EMBEDDINGS_DIR / "embedding_names.csv",
    },
    "SimCLR v2 (batch=80, temp=0.2)": {
        "emb_file": EMBEDDINGS_DIR / "finetuned_embeddings_v2.npy",
        "names_file": EMBEDDINGS_DIR / "embedding_names.csv",
    }
}

# ==================== ГЛОБАЛЬНОЕ СОСТОЯНИЕ ====================
state = {
    "embeddings": None,
    "names_df": None,
    "meta_df": None,
    "faiss_index": None,
    "current_model": None,
}

def build_faiss_index(embeddings):
    """Строит FAISS индекс для быстрого косинусного поиска."""
    # Нормализуем вектора, чтобы Inner Product == Cosine Similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Избегаем деления на 0
    normalized = embeddings / norms
    normalized = normalized.astype('float32')
    
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine после нормализации
    index.add(normalized)
    return index

def load_data(model_name="Baseline (ImageNet ResNet50)"):
    """Загружает эмбеддинги и строит FAISS индекс."""
    config = EMBEDDING_CONFIGS.get(model_name)
    if config is None:
        return False, f"Неизвестная модель: {model_name}"
    
    if not config["emb_file"].exists():
        return False, f"Файл {config['emb_file'].name} не найден. Сначала извлеките эмбеддинги."
    
    print(f"Loading embeddings for: {model_name}...")
    state["embeddings"] = np.load(config["emb_file"])
    state["names_df"] = pd.read_csv(config["names_file"])
    
    if state["meta_df"] is None:
        state["meta_df"] = pd.read_csv(META_FILE)
        state["meta_df"] = state["meta_df"].drop_duplicates(subset=['tile_name'])
    
    state["names_df"] = state["names_df"].merge(state["meta_df"], on='tile_name', how='inner')
    
    # Строим FAISS индекс
    print("Building FAISS index...")
    state["faiss_index"] = build_faiss_index(state["embeddings"])
    state["current_model"] = model_name
    
    n = len(state["embeddings"])
    return True, f"✅ Загружено **{n}** тайлов. Модель: **{model_name}**. FAISS индекс готов."

def find_similar_faiss(query_idx, top_k=9):
    """Мгновенный поиск через FAISS (миллисекунды вместо секунд)."""
    query_emb = state["embeddings"][query_idx].reshape(1, -1).astype('float32')
    
    # Нормализуем запрос
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm
    
    # FAISS поиск: top_k + 1 (включая саму картинку)
    similarities, indices = state["faiss_index"].search(query_emb, top_k + 1)
    
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == query_idx:
            continue
        if len(results) >= top_k:
            break
        
        tile_name = state["names_df"].iloc[idx]['tile_name']
        img_path = PROCESSED_DATA_DIR / tile_name
        source = state["names_df"].iloc[idx]['source_image']
        caption = f"{source} | Sim: {sim:.3f}"
        
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results.append((img, caption))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return results

def gui_search(selected_id_str):
    """Обработчик кнопки поиска."""
    try:
        query_idx = int(selected_id_str)
        if query_idx < 0 or query_idx >= len(state["names_df"]):
            return None, None, f"⚠️ Индекс должен быть от 0 до {len(state['names_df'])-1}"
        
        tile_name = state["names_df"].iloc[query_idx]['tile_name']
        source = state["names_df"].iloc[query_idx]['source_image']
        query_img_path = PROCESSED_DATA_DIR / tile_name
        query_img = cv2.cvtColor(cv2.imread(str(query_img_path)), cv2.COLOR_BGR2RGB)
        
        info = f"### 🔍 Результаты поиска\n**Запрос:** `{tile_name}` (ID: {query_idx}) | **Источник:** `{source}`\n\n**Модель:** {state['current_model']}"
        
        similar_images = find_similar_faiss(query_idx, top_k=9)
        
        return query_img, similar_images, info
    except ValueError:
        return None, None, "⚠️ Введите целое число."

def gui_switch_model(model_name):
    """Обработчик переключения модели."""
    success, msg = load_data(model_name)
    return msg

def create_ui():
    # Загружаем Baseline по умолчанию
    available_models = [m for m, c in EMBEDDING_CONFIGS.items() if c["emb_file"].exists()]
    
    if not available_models:
        available_models = list(EMBEDDING_CONFIGS.keys())
    
    default_model = available_models[0]
    success, init_msg = load_data(default_model)
    
    with gr.Blocks(title="SEM Image Clustering & Retrieval", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Диплом: Обратный поиск РЭМ-изображений")
        gr.Markdown("Демо-прототип для поиска участков с похожей текстурой на микрофотографиях наноматериалов.  \n"
                     "Используется **FAISS** для мгновенного поиска по 27 000+ эмбеддингам.")
        
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(EMBEDDING_CONFIGS.keys()),
                value=default_model,
                label="Выберите модель эмбеддингов",
                interactive=True
            )
        
        status_text = gr.Markdown(init_msg)
        model_selector.change(fn=gui_switch_model, inputs=[model_selector], outputs=[status_text])
        
        if success:
            with gr.Row():
                with gr.Column(scale=1):
                    query_id = gr.Textbox(
                        label=f"ID фрагмента для поиска (0 — {len(state['names_df'])-1})",
                        value="0"
                    )
                    search_btn = gr.Button("🔍 Найти похожие текстуры", variant="primary")
                    gr.Markdown("### Исходный фрагмент:")
                    query_gallery = gr.Image(label="Запрос")
                    
                with gr.Column(scale=3):
                    results_text = gr.Markdown("### Результаты поиска:")
                    results_gallery = gr.Gallery(label="Похожие тайлы", columns=3, rows=3, height="auto")
                    
            search_btn.click(
                fn=gui_search,
                inputs=[query_id],
                outputs=[query_gallery, results_gallery, results_text]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
