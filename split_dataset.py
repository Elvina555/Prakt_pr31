# Импорт необходимых модулей
import os  # Для работы с файловой системой
import random  # Для случайного перемешивания данных
import shutil  # Для операций копирования файлов

def split_dataset(images_dir, labels_dir, output_dir, val_split=0.2, seed=42):
    """
    Разбивает датасет на тренировочную и валидационную выборки.

    Параметры:
    images_dir: путь к директории с исходными изображениями
    labels_dir: путь к директории с исходными аннотациями
    output_dir: корневая директория для выходных данных
    val_split: доля данных для валидации (по умолчанию 20%)
    seed: значение для инициализации генератора случайных чисел (для воспроизводимости)
    """
    
    # Инициализация генератора случайных чисел
    random.seed(seed)

    # Проверка существования исходных директорий
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Директория изображений не найдена: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Директория аннотаций не найдена: {labels_dir}")

    # Создание структуры выходных директорий
    # Пути для тренировочных данных
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    # Пути для валидационных данных
    valid_img_dir = os.path.join(output_dir, 'valid', 'images')
    valid_lbl_dir = os.path.join(output_dir, 'valid', 'labels')
    
    # Создание директорий (если не существуют)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)

    # Сбор базовых имен файлов изображений (без расширения)
    image_bases = set()  # Используем множество для избежания дубликатов
    # Перебор всех файлов в директории с изображениями
    for file in os.listdir(images_dir):
        # Проверка поддерживаемых расширений изображений
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Извлечение базового имени без расширения
            base_name = os.path.splitext(file)[0]
            # Добавление имени в множество
            image_bases.add(base_name)

    # Вывод общего количества найденных изображений
    print(f"Найдено {len(image_bases)} изображений.")

    # Фильтрация файлов: оставляем только те, у которых есть аннотации
    valid_files = [] # Создание пустого массива
    # Итерируемся по именам
    for base in image_bases:
        # Формирование пути к файлу аннотации
        label_path = os.path.join(labels_dir, f"{base}.txt")
        # Проверка существования аннотации
        if os.path.exists(label_path):
            valid_files.append(base) # Добавление имени в массив
        else:
            # Предупреждение об отсутствующей аннотации
            print(f"Предупреждение: Аннотация для {base}.txt не найдена")

    # Вывод количества валидных пар изображение-аннотация
    print(f"Действительных пар image-label: {len(valid_files)}")
    
    # Перемешивание списка файлов для случайного распределения
    random.shuffle(valid_files)

    # Разделение на тренировочную и валидационную выборки
    # Вычисление индекса разделения
    split_idx = int(len(valid_files) * (1 - val_split))
    # Разделение файлов на две группы
    train_files = valid_files[:split_idx]    # Тренировочная выборка
    valid_files = valid_files[split_idx:]    # Валидационная выборка

    # Вывод размеров выборок
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}")

    # Функция для копирования файлов
    def copy_files(bases, src_img, src_lbl, dst_img, dst_lbl):
        """
        Копирует изображения и соответствующие аннотации в целевые директории
        
        Параметры:
        bases: список базовых имен файлов (без расширений)
        src_img: исходная директория с изображениями
        src_lbl: исходная директория с аннотациями
        dst_img: целевая директория для изображений
        dst_lbl: целевая директория для аннотаций
        """
        # Перебор всех базовых имен
        for base in bases:
            # Поиск и копирование изображения (с учетом разных расширений)
            img_copied = False
            # Проверка всех возможных расширений изображений
            for ext in ['.jpg', '.jpeg', '.png']:
                src_path = os.path.join(src_img, base + ext) # Полный путь к изображению
                # Если файл существует - копируем
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_img)
                    img_copied = True
                    break # Прерываем цикл
            
            # Копирование аннотации
            lbl_src = os.path.join(src_lbl, base + '.txt') # Полный путь к файлу с аннотациями
            # Если файл существует - копируем
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, dst_lbl)
            else:
                # Сообщение об ошибке, если аннотация не найдена
                print(f"Ошибка: Аннотация {base}.txt отсутствует")

    # Копирование тренировочных данных
    print("Копирую train данные...")
    copy_files(
        train_files, 
        images_dir, 
        labels_dir,
        train_img_dir, 
        train_lbl_dir
    )
    
    # Копирование валидационных данных
    print("Копирую valid данные...")
    copy_files(
        valid_files, 
        images_dir, 
        labels_dir,
        valid_img_dir, 
        valid_lbl_dir
    )

    # Финальное сообщение об успешном завершении
    print("Разделение датасета завершено успешно!")

# Точка входа при запуске скрипта напрямую
if __name__ == '__main__':
    # Пример использования функции
    split_dataset(
        images_dir='data/images',      # Исходные изображения
        labels_dir='data/labels',      # Исходные аннотации
        output_dir='data',             # Целевая директория
        val_split=0.2,                 # 20% данных для валидации
        seed=42                        # Seed для воспроизводимости
    )