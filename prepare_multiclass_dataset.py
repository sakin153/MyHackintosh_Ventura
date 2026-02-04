"""
Простой скрипт для подготовки мультиклассового датасета YOLO
Принимает от 2 до 4 папок с разными классами и создает готовый датасет
"""

import shutil
import random
from pathlib import Path
import yaml


def merge_class_folders(class_folders, output_dir):
    """
    Объединяет несколько папок с разными классами в один датасет
    
    Args:
        class_folders: список кортежей [(class_id, class_name, folder_path), ...]
        output_dir: выходная папка для объединенного датасета
    
    Returns:
        Общее количество изображений
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Очищаем папку если она уже существует
    for file in output_path.glob('*'):
        file.unlink()
    
    all_items = []
    
    # Проходим по всем папкам классов
    for class_id, class_name, folder_path in class_folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"  Папка {folder} не найдена, пропускаем...")
            continue
        
        # Ищем все изображения
        image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        
        for img in image_files:
            txt = img.with_suffix('.txt')
            if txt.exists():
                all_items.append((class_id, class_name, img, txt))
    
    print(f"\nВсего найдено {len(all_items)} изображений")
    
    # Копируем все файлы с переименованием
    for idx, (class_id, class_name, img, txt) in enumerate(all_items):
        new_name = f"{class_name}_{idx:05d}"
        
        # Копируем изображение
        shutil.copy(img, output_path / f"{new_name}{img.suffix}")
        
        # Копируем и обновляем аннотацию (меняем ID класса)
        with open(txt, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Заменяем ID класса на нужный
                parts[0] = str(class_id)
                updated_lines.append(' '.join(parts))
        
        # Сохраняем обновленную аннотацию
        with open(output_path / f"{new_name}.txt", 'w') as f:
            f.write('\n'.join(updated_lines))
    
    print(f"Датасеты объединены в: {output_path}")
    return len(all_items)




def prepare_yolo_dataset(src_dir, dst_dir, split_ratio=(0.7, 0.2, 0.1), class_names=None):
    """
    Подготавливает датасет в формате YOLO с разделением на train/val/test
    
    Args:
        src_dir: папка с исходными данными
        dst_dir: выходная папка
        split_ratio: соотношение train/val/test
        class_names: список названий классов
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Создаем структуру папок
    for subset in ['train', 'val', 'test']:
        (dst_path / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (dst_path / 'labels' / subset).mkdir(parents=True, exist_ok=True)
    
    # Собираем все пары изображение + аннотация
    pairs = []
    for img in list(src_path.glob('*.jpg')) + list(src_path.glob('*.png')):
        txt = img.with_suffix('.txt')
        if txt.exists():
            pairs.append((img, txt))
    
    print(f"\nНайдено {len(pairs)} пар изображение+аннотация")
    
    if len(pairs) == 0:
        print("Не найдено ни одной пары! Проверьте путь к данным.")
        return
    
    # Перемешиваем и разделяем
    random.seed(42)
    random.shuffle(pairs)
    
    total = len(pairs)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    # Копируем файлы
    for subset, items in splits.items():
        for img, txt in items:
            shutil.copy(img, dst_path / 'images' / subset / img.name)
            shutil.copy(txt, dst_path / 'labels' / subset / txt.name)
    
    # Создаем data.yaml
    yaml_data = {
        'path': str(dst_path.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dst_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nДатасет подготовлен в: {dst_path}")
    print(f"Конфигурация: {yaml_path}")
    print(f"\nСтатистика:")
    print(f"  Train: {len(splits['train'])} изображений")
    print(f"  Val:   {len(splits['val'])} изображений")
    print(f"  Test:  {len(splits['test'])} изображений")
    print(f"  Классы ({len(class_names)}): {', '.join(class_names)}")


if __name__ == "__main__":
    print("=" * 70)
    print("ПОДГОТОВКА МУЛЬТИКЛАССОВОГО ДАТАСЕТА ДЛЯ YOLO")
    print("=" * 70)
    

    
    # Укажите папки с классами: (id, название, путь)
    # ID должны идти по порядку: 0, 1, 2, 3...
    CLASS_FOLDERS = [
        (0, 'face', r'C:\Users\zoyaa\Desktop\REASKILLS\me'),
        (1, 'badge', r'C:\Users\zoyaa\Desktop\REASKILLS\frames_pass'),
        # (2, 'helmet', r'C:\путь\к\папке\с\касками'),  # добавьте если нужен 3-й класс
        # (3, 'vest', r'C:\путь\к\папке\с\жилетами'),   # добавьте если нужен 4-й класс
    ]
    
    # Выходные папки
    COMBINED_DIR = r'C:\Users\zoyaa\Desktop\REASKILLS\combined_dataset'
    FINAL_DATASET_DIR = r'C:\Users\zoyaa\Desktop\REASKILLS\dataset'
    
    # Соотношение train/val/test
    SPLIT_RATIO = (0.7, 0.2, 0.1)


    # Собираем названия классов
    class_names = [name for _, name, _ in CLASS_FOLDERS]
    print(f"\nКлассы для обучения: {class_names}")
    print(f"Количество классов: {len(class_names)}")
    
    # ШАГ 1: Объединяем все папки
    print("\n" + "=" * 70)
    print("ШАГ 1/2: Объединение датасетов")
    print("=" * 70)
    
    total_images = merge_class_folders(CLASS_FOLDERS, COMBINED_DIR)
    
    if total_images == 0:
        print("\nОшибка: не найдено изображений для обработки!")
        print("Проверьте пути к папкам в настройках.")
        exit(1)
    
    # ШАГ 2: Создаем финальный датасет YOLO
    print("\n" + "=" * 70)
    print("ШАГ 2/2: Создание YOLO датасета (train/val/test)")
    print("=" * 70)
    
    prepare_yolo_dataset(
        src_dir=COMBINED_DIR,
        dst_dir=FINAL_DATASET_DIR,
        split_ratio=SPLIT_RATIO,
        class_names=class_names
    )
    
    # Финальное сообщение
    print("\n" + "=" * 70)
    print("ДАТАСЕТ ПОЛНОСТЬЮ ГОТОВ К ОБУЧЕНИЮ!")
    print("=" * 70)
    print(f"\nПуть к датасету: {FINAL_DATASET_DIR}")
    print(f"Конфигурация: {FINAL_DATASET_DIR}\\data.yaml")
    print(f"\nСледующий шаг: python train_model.py")
    print("=" * 70)