"""
Скрипт для обучения YOLO модели на детекцию лица и пропуска
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def train_face_badge_detector(
    data_yaml='dataset/data.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    img_size=640,
    batch_size=16,
    device='0',  # '0' для GPU, 'cpu' для CPU
    project='runs/train',
    name='face_badge_detector'
):
    """
    Обучает YOLO модель для детекции лица и пропуска
    
    Args:
        data_yaml: путь к конфигурации датасета
        model_size: размер модели (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: количество эпох обучения
        img_size: размер входного изображения
        batch_size: размер батча
        device: устройство для обучения
        project: папка для сохранения результатов
        name: название эксперимента
    """
    
    print("=" * 60)
    print("🚀 Начало обучения модели")
    print("=" * 60)
    
    # Проверяем наличие GPU
    if device == '0' and not torch.cuda.is_available():
        print("⚠️ GPU не доступен, используем CPU")
        device = 'cpu'
    else:
        print(f"✅ Используем устройство: {device}")
    
    # Проверяем датасет
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {data_yaml}")
    
    print(f"✅ Конфигурация датасета: {data_yaml}")
    
    # Загружаем базовую модель
    model_name = f'yolov8{model_size}.pt'
    print(f"📦 Загружаем базовую модель: {model_name}")
    model = YOLO(model_name)
    
    # Параметры обучения
    train_args = {
        'data': str(data_path),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'patience': 50,  # early stopping
        'save': True,
        'save_period': 10,  # сохранять каждые 10 эпох
        'cache': False,  # True для ускорения (если хватает RAM)
        'plots': True,  # сохранять графики
        'verbose': True,
        
        # Аугментации
        'hsv_h': 0.015,  # цветовой тон
        'hsv_s': 0.7,    # насыщенность
        'hsv_v': 0.4,    # яркость
        'degrees': 10,   # поворот
        'translate': 0.1,  # сдвиг
        'scale': 0.5,    # масштаб
        'shear': 0.0,    # сдвиг
        'perspective': 0.0,  # перспектива
        'flipud': 0.0,   # вертикальное отражение
        'fliplr': 0.5,   # горизонтальное отражение
        'mosaic': 1.0,   # мозаика
        'mixup': 0.0,    # смешивание
    }
    
    print(f"\n📋 Параметры обучения:")
    print(f"  Эпохи: {epochs}")
    print(f"  Размер изображения: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Модель: YOLOv8{model_size}")
    
    # Начинаем обучение
    print(f"\n{'=' * 60}")
    print("🎯 Запуск обучения...")
    print(f"{'=' * 60}\n")
    
    results = model.train(**train_args)
    
    print(f"\n{'=' * 60}")
    print("✅ Обучение завершено!")
    print(f"{'=' * 60}")
    
    # Валидация
    print("\n📊 Запуск валидации...")
    metrics = model.val()
    
    print(f"\n📈 Метрики:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # Путь к лучшей модели
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    print(f"\n💾 Лучшая модель сохранена: {best_model_path}")
    
    return model, results, metrics


def resume_training(checkpoint_path, epochs=50):
    """
    Продолжить обучение с checkpoint
    """
    model = YOLO(checkpoint_path)
    results = model.train(resume=True, epochs=epochs)
    return model, results


if __name__ == "__main__":
    # === БАЗОВОЕ ОБУЧЕНИЕ ===
    print("Начинаем обучение модели детекции лица и пропуска\n")
    
    # Для быстрого тестирования используйте меньше эпох
    # Для production обучения увеличьте до 100-300 эпох
    
    model, results, metrics = train_face_badge_detector(
        data_yaml=r'C:\Users\zoyaa\Desktop\REASKILLS\dataset\data.yaml',
        model_size='n',  # nano - самая легкая и быстрая модель
        epochs=100,      # уменьшите до 10-20 для быстрого теста
        img_size=640,
        batch_size=16,   # уменьшите если не хватает памяти
        device='0',      # '0' для GPU, 'cpu' для CPU
        project='runs/train',
        name='face_badge_v1'
    )
    
    # === ДОПОЛНИТЕЛЬНЫЕ ВАРИАНТЫ ===
    
    # Для более точной модели (но медленнее):
    # model, results, metrics = train_face_badge_detector(
    #     model_size='s',  # или 'm' для еще большей точности
    #     epochs=200,
    #     batch_size=8
    # )
    
    # Продолжить обучение с checkpoint:
    # model, results = resume_training(
    #     checkpoint_path='runs/train/face_badge_v1/weights/last.pt',
    #     epochs=50
    # )
