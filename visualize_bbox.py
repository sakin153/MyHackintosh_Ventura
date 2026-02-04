import cv2
import os
import numpy as np
from pathlib import Path


def draw_yolo_bbox(image_path, label_path, output_path=None, show=True):
    """
    Визуализирует bounding box из YOLO формата на изображении
    
    Args:
        image_path: путь к изображению
        label_path: путь к файлу с аннотациями
        output_path: путь для сохранения (опционально)
        show: показывать ли изображение
    """
    # Читаем изображение
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Читаем аннотации
    if not os.path.exists(label_path):
        print(f"Файл аннотации не найден: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Рисуем каждый bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_name = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])
        
        # Преобразуем нормализованные координаты в пиксели
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        bbox_width_px = int(bbox_width * width)
        bbox_height_px = int(bbox_height * height)
        
        # Вычисляем координаты углов
        x1 = int(x_center_px - bbox_width_px / 2)
        y1 = int(y_center_px - bbox_height_px / 2)
        x2 = int(x_center_px + bbox_width_px / 2)
        y2 = int(y_center_px + bbox_height_px / 2)
        
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Добавляем текст с классом
        label_text = f"{class_name}"
        cv2.putText(image, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Рисуем центральную точку
        cv2.circle(image, (x_center_px, y_center_px), 4, (255, 0, 0), -1)
    
    # Сохраняем результат
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Сохранено: {output_path}")
    
    # Показываем изображение
    if show:
        cv2.imshow('Bounding Box Visualization', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image


def visualize_dataset(dataset_path, output_dir=None, num_samples=10, show=True):
    """
    Визуализирует несколько примеров из датасета
    
    Args:
        dataset_path: путь к папке с датасетом
        output_dir: папка для сохранения результатов
        num_samples: количество примеров для визуализации
        show: показывать ли изображения
    """
    dataset_path = Path(dataset_path)
    
    # Создаем папку для выходных данных
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Получаем все изображения
    image_files = sorted(list(dataset_path.glob('*.jpg')))
    
    if not image_files:
        print(f"Изображения не найдены в {dataset_path}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Ограничиваем количество примеров
    if num_samples > 0:
        # Выбираем равномерно распределенные примеры
        step = max(1, len(image_files) // num_samples)
        image_files = image_files[::step][:num_samples]
    
    # Визуализируем каждое изображение
    for idx, img_path in enumerate(image_files):
        label_path = img_path.with_suffix('.txt')
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"visualized_{img_path.name}"
        
        print(f"\n[{idx + 1}/{len(image_files)}] Обработка: {img_path.name}")
        draw_yolo_bbox(img_path, label_path, output_path, show=show)


def batch_visualize(dataset_path, output_dir, show_progress=True):
    """
    Визуализирует весь датасет и сохраняет в папку
    
    Args:
        dataset_path: путь к папке с датасетом
        output_dir: папка для сохранения результатов
        show_progress: показывать ли прогресс
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_files = sorted(list(dataset_path.glob('*.jpg')))
    total = len(image_files)
    
    print(f"Обработка {total} изображений...")
    
    for idx, img_path in enumerate(image_files, 1):
        label_path = img_path.with_suffix('.txt')
        output_path = output_dir / f"vis_{img_path.name}"
        
        if show_progress and idx % 10 == 0:
            print(f"Обработано: {idx}/{total}")
        
        draw_yolo_bbox(img_path, label_path, output_path, show=False)
    
    print(f"\nГотово! Все изображения сохранены в: {output_dir}")


if __name__ == "__main__":
    # Путь к вашему датасету
    dataset_path = r"C:\Users\zoyaa\Desktop\REASKILLS\me"
    
    # Вариант 1: Визуализировать несколько примеров с отображением
    print("=== Визуализация примеров датасета ===")
    visualize_dataset(
        dataset_path=dataset_path,
        output_dir=r"C:\Users\zoyaa\Desktop\REASKILLS\visualized_samples",
        num_samples=10,  # Показать 10 примеров
        show=True  # Показывать изображения (нажмите любую клавишу для следующего)
    )
    
    # Вариант 2: Обработать весь датасет и сохранить (без показа)
    # Раскомментируйте для использования:
    # print("\n=== Пакетная визуализация всего датасета ===")
    # batch_visualize(
    #     dataset_path=dataset_path,
    #     output_dir=r"C:\Users\zoyaa\Desktop\REASKILLS\visualized_all"
    # )
    
    # Вариант 3: Визуализировать одно конкретное изображение
    # Раскомментируйте для использования:
    # draw_yolo_bbox(
    #     image_path=r"C:\Users\zoyaa\Desktop\REASKILLS\me\frame_000000.jpg",
    #     label_path=r"C:\Users\zoyaa\Desktop\REASKILLS\me\frame_000000.txt",
    #     output_path=r"C:\Users\zoyaa\Desktop\REASKILLS\test_visualization.jpg",
    #     show=True
    # )
