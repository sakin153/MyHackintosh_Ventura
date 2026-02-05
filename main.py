import gradio as gr
from ultralytics import YOLO

# Загрузка модели
model = YOLO('yolov8n.pt')

# Классы модели
CLASS_CLEAR_FACE = 0
CLASS_MASKED_FACE = 1
CLASS_BADGE = 2

def detect_faces(image):
    if image is None:
        return None, "⚠️ Ожидание камеры..."
    
    # Детекция (320 для максимальной скорости)
    results = model(image, imgsz=320, conf=0.5, verbose=False)
    
    # Подсчет классов
    count_clear = 0
    count_masked = 0
    count_badge = 0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == CLASS_CLEAR_FACE:
                    count_clear += 1
                elif class_id == CLASS_MASKED_FACE:
                    count_masked += 1
                elif class_id == CLASS_BADGE:
                    count_badge += 1
    
    # Сообщение
    msg = []
    if count_masked > 0:
        msg.append("🚨 ЛИЦО ЗАКРЫТО! Снимите маску/очки!\n")
    
    msg.append(f"Чистое лицо: {count_clear}")
    msg.append(f"Закрытое лицо: {count_masked}")
    msg.append(f"Бейджик: {count_badge}")
    
    if count_masked > 0:
        msg.append("\n❌ СТАТУС: Детекция невозможна")
    elif count_clear > 0:
        msg.append("\n✅ СТАТУС: Всё хорошо")
    else:
        msg.append("\n⏳ Ищем лицо...")
    
    return results[0].plot(), "\n".join(msg)


# Простой интерфейс с реальным временем
demo = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Камера"),
    outputs=[
        gr.Image(label="Результат детекции"),
        gr.Textbox(label="Статус", lines=5)
    ],
    live=True,
    title="🎯 Детекция лиц в реальном времени"
)

if __name__ == "__main__":
    demo.launch(server_port=7860)