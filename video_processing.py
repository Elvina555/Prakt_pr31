# Импорт необходимых библиотек
import cv2  # OpenCV для работы с видео и изображениями
from ultralytics import YOLO  # YOLO модель для детекции объектов
import numpy as np  # NumPy для работы с массивами
from sort import *  # SORT (Simple Online and Realtime Tracking) для отслеживания объектов

# Пути к входному и выходному видеофайлам
input_video_path = 'video/cut_video_2.mp4'  # Путь к исходному видео
output_video_path = 'video/output_2.mp4'  # Путь для сохранения обработанного видео

# Загрузка модели YOLO и перемещение ее на GPU (CUDA)
model = YOLO('model/mel_model.pt').to('cuda')

# Инициализация трекера SORT с параметрами:
# max_age=5 - максимальное количество кадров без обнаружения перед удалением трека
# min_hits=2 - минимальное количество обнаружений для инициализации трека
# iou_threshold=0.1 - порог IoU для ассоциации обнаружений
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.1)

def process_images(img, model):
    """
    Обрабатывает один кадр: детекция объектов, трекинг и визуализация результатов.
    
    Параметры:
    img: входное изображение (кадр видео)
    model: модель YOLO для детекции объектов
    
    Возвращает:
    img: обработанное изображение с bounding boxes и метками
    """
    # Выполняем предсказание с помощью модели YOLO
    # conf=0.8 - минимальная уверенность для детекции
    # iou=0.5 - порог IoU для NMS (Non-Maximum Suppression)
    # agnostic_nms=True - использовать класс-независимый NMS
    results = model(img, conf=0.8, iou=0.5, agnostic_nms=True)
    
    # Обрабатываем результаты для каждого обнаруженного объекта
    for result in results:
        # Извлекаем координаты bounding boxes в формате [x1, y1, x2, y2]
        boxes = result.boxes.xyxy.cpu().numpy()
        # Извлекаем уверенность (confidence) для каждого обнаружения
        confidences = result.boxes.conf.cpu().numpy()
        # Извлекаем идентификаторы классов
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Формируем список обнаружений для трекера
        detections = []
        # Объединяем информацию по каждому обнаружению и итерируемся по ней
        for detection in zip(boxes, confidences, class_ids):
            # Создаем массив [x1, y1, x2, y2, conf, class_id] для каждого объекта
            detections.append(np.hstack(detection))
        
        # Сортируем обнаружения по высоте объектов (от низа к верху)
        detections = sorted(detections, key=lambda x: x[1], reverse=False)
        # Преобразуем в numpy массив
        detections = np.array(detections)
        
        # Подготавливаем данные для трекера (формат [x1, y1, x2, y2, conf])
        if len(detections) == 0:
            # Если обнаружений нет - пустой массив
            detections_ = np.empty((0, 5))
        else:
            # Иначе берем первые 5 столбцов (x1, y1, x2, y2, conf)
            detections_ = detections[:, :5]
        
        # Обновляем трекер с новыми обнаружениями
        # Возвращаются обновленные треки в формате [x1, y1, x2, y2, track_id]
        track_ids = tracker.update(detections_)
        # Реверсируем порядок треков для визуализации
        track_ids = track_ids[::-1]
        
        # Отрисовываем результаты для каждого трека
        # Цикл по количеству объектов
        for i in range(len(track_ids)):
            # Преобразуем координаты bounding box в целые числа
            x1, y1, x2, y2 = map(int, detections[i][:4])
            # Извлекаем уверенность и класс
            conf, class_id = detections[i][4:]
            
            # Выбираем цвет bounding box в зависимости от класса
            if class_id == 1:
                color = (0, 0, 255)  # Красный для класса 1 (BGR)
            else:
                color = (0, 255, 0)  # Зеленый для других классов
            
            # Рисуем прямоугольник вокруг объекта
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Добавляем текст с информацией об объекте
            cv2.putText(
                img, 
                f'conf: {conf:.2f}, class: {int(class_id)}, id: {track_ids[i][4]}', 
                (x1, y1 - 10),  # Позиция текста (над bounding box)
                cv2.FONT_HERSHEY_SIMPLEX,  # Шрифт
                1,  # Масштаб шрифта
                color,  # Цвет текста
                2  # Толщина текста
            )
    
    return img  # Возвращаем обработанное изображение

def video_processing(input_video_path, output_video_path, model):
    """
    Обрабатывает видео: чтение кадров, обработка каждого кадра, сохранение результата.
    
    Параметры:
    input_video_path: путь к исходному видео
    output_video_path: путь для сохранения обработанного видео
    model: модель YOLO для детекции объектов
    """
    # Открываем видеофайл для чтения
    cap = cv2.VideoCapture(input_video_path)

    # Проверяем успешность открытия видеофайла
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        exit()  # Выход из программы при ошибке

    # Получаем параметры видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
    fps = cap.get(cv2.CAP_PROP_FPS)  # Кадров в секунду (FPS)

    # Определяем кодек для выходного видео (MP4V)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Создаем объект для записи видео
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Основной цикл обработки видео
    while True:
        # Читаем очередной кадр
        ret, frame = cap.read()
        # Если кадр прочитать не удалось (конец видео)
        if not ret:
            break  # Выходим из цикла
        
        # Обрабатываем текущий кадр (применяем функцию process_images)
        processed_frame = process_images(frame, model)

        # Записываем обработанный кадр в выходной файл
        out.write(processed_frame)

    # Освобождаем ресурсы
    cap.release()  # Закрываем входное видео
    out.release()  # Закрываем выходное видео

    print("Видео успешно обработано и сохранено.")

# Запуск обработки видео
video_processing(input_video_path, output_video_path, model)