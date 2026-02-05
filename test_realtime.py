"""
Скрипт проверки YOLO детекции в реальном времени через OpenCV
Нажмите 'q' для выхода
"""
import cv2
from ultralytics import YOLO

# Загрузка модели
model = YOLO('yolov8n.pt')  # Замените на путь к вашей модели

# Классы (настройте под свою модель)
CLASS_NAMES = {
    0: "Чистое лицо",
    1: "Закрытое лицо", 
    2: "Бейджик"
}

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 50)
print("🎯 YOLO Детекция в реальном времени")
print("Нажмите 'q' для выхода")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр")
        break
    
    # Детекция
    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    
    # Подсчет объектов
    counts = {0: 0, 1: 0, 2: 0}
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in counts:
                    counts[class_id] += 1
    
    # Рисуем результаты
    annotated_frame = results[0].plot()
    
    # Добавляем текст статуса
    y_pos = 30
    cv2.putText(annotated_frame, f"Chistoe lico: {counts[0]}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_pos += 30
    cv2.putText(annotated_frame, f"Zakrytoe lico: {counts[1]}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_pos += 30
    cv2.putText(annotated_frame, f"Badge: {counts[2]}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Предупреждение если лицо закрыто
    if counts[1] > 0:
        cv2.putText(annotated_frame, "!!! SNIMITE MASKU/OCHKI !!!", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Показываем кадр
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Готово!")
