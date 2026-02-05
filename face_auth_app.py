"""
Система авторизации по лицу
Использует функции из face_det.py
"""

import gradio as gr
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Импортируем функции из face_det.py
from face_det import (
    get_face_embedding,
    load_existing_embeddings,
    UNIQUE_FACES_PATH,
    SIMILARITY_THRESHOLD,
    create_directories
)

# Инициализация
create_directories()


def recognize_face(image):
    """Распознавание лица из базы данных"""
    if image is None:
        return None, "⏳ Ожидание камеры..."
    
    # Сохраняем временное изображение
    temp_path = "temp_recognize.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Получаем эмбеддинг лица
        face_embeddings = get_face_embedding(temp_path)
        
        if not face_embeddings:
            return image, "❌ Лицо не обнаружено. Посмотрите в камеру."
        
        # Загружаем базу эмбеддингов
        embeddings_data = load_existing_embeddings()
        
        if not embeddings_data:
            return image, "⚠️ База данных пуста. Зарегистрируйтесь."
        
        # Ищем совпадение
        for face_embedding in face_embeddings:
            best_similarity = 0
            best_match = None
            
            for stored_embedding, user_folder in embeddings_data:
                similarity = cosine_similarity([face_embedding], [stored_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_folder
            
            if best_similarity >= SIMILARITY_THRESHOLD:
                similarity_pct = best_similarity * 100
                return image, f"✅ АВТОРИЗОВАН!\n\n👤 Пользователь: {best_match}\n📊 Совпадение: {similarity_pct:.1f}%"
            else:
                return image, f"❌ НЕ АВТОРИЗОВАН!\n\nПользователь не найден в базе.\nМаксимальное совпадение: {best_similarity*100:.1f}%\n\n➡️ Перейдите на вкладку 'Регистрация'"
    
    except Exception as e:
        return image, f"⚠️ Ошибка: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def register_user(image, full_name):
    """Регистрация нового пользователя"""
    if image is None:
        return "❌ Сделайте снимок!"
    
    if not full_name or not full_name.strip():
        return "❌ Введите ФИО!"
    
    # Очищаем имя для использования как папки
    safe_name = full_name.strip().replace(" ", "_")
    
    # Проверяем, существует ли уже такой пользователь
    user_folder = os.path.join(UNIQUE_FACES_PATH, safe_name)
    if os.path.exists(user_folder):
        return f"⚠️ Пользователь '{full_name}' уже существует!"
    
    # Сохраняем временное изображение
    temp_path = "temp_register.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Получаем эмбеддинг
        face_embeddings = get_face_embedding(temp_path)
        
        if not face_embeddings:
            return "❌ Лицо не обнаружено! Попробуйте ещё раз."
        
        # Создаем папку пользователя
        os.makedirs(user_folder, exist_ok=True)
        
        # Сохраняем изображение
        face_image_path = os.path.join(user_folder, f"{safe_name}.jpg")
        cv2.imwrite(face_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Сохраняем эмбеддинг
        embedding_path = os.path.join(user_folder, "embedding.npy")
        np.save(embedding_path, np.array(face_embeddings[0]))
        
        return f"✅ Пользователь '{full_name}' успешно зарегистрирован!"
    
    except Exception as e:
        return f"❌ Ошибка регистрации: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==================== ИНТЕРФЕЙС ====================
with gr.Blocks(title="🔐 Система авторизации по лицу") as demo:
    gr.Markdown("# 🔐 Система авторизации по лицу")
    
    with gr.Tabs():
        # Вкладка авторизации
        with gr.TabItem("🔍 Авторизация"):
            gr.Markdown("### Посмотрите в камеру для авторизации")
            with gr.Row():
                auth_image = gr.Image(sources=["webcam"], label="Камера")
                auth_result = gr.Textbox(label="Статус", lines=6)
            auth_btn = gr.Button("🔐 Авторизоваться", variant="primary", size="lg")
            auth_btn.click(recognize_face, inputs=auth_image, outputs=[auth_image, auth_result])
        
        # Вкладка регистрации
        with gr.TabItem("📝 Регистрация"):
            gr.Markdown("### Введите ФИО и сделайте снимок")
            reg_name = gr.Textbox(label="ФИО", placeholder="Иванов Иван Иванович")
            reg_image = gr.Image(sources=["webcam"], label="Камера")
            reg_result = gr.Textbox(label="Результат", lines=3)
            reg_btn = gr.Button("📸 Зарегистрироваться", variant="primary", size="lg")
            reg_btn.click(register_user, inputs=[reg_image, reg_name], outputs=reg_result)


if __name__ == "__main__":
    demo.launch(server_port=7860)
