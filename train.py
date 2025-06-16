# Импорт библиотек
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics import YOLO
import cv2

# Функция инициализации класса Albumentations
def __init__(self, p=1.0):
    # Инициализация переменных
    self.p = p # вероятность применения аугментаций
    self.transform = None # Объект класса transform
    prefix = colorstr("albumentations: ") # Для логов

    try: # Если не возникнет исключений
        import albumentations as A # Импорт библиотеки

        # Список возможных пространственных преобразований (аугментаций)
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

        # Список из применяемых трансформаций (p - вероятность применения)
        T = [
            A.Blur(p=0.17), # Равномерное размытие на изображении
            A.MedianBlur(p=0.17), # Медианное размытие
            A.CLAHE(p=0.17), # Увеличение контраста изображения
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.2, 0.5), p=0.35), # Изменение яркости и контраста на изображении
        ]

        # Проверяем, содержит ли список преобразований T хотя бы одно пространственное преобразование
        # Пространственные преобразования (например, поворот, масштабирование, кадрирование) изменяют геометрию изображения
        # и требуют специальной обработки bounding boxes
        self.contains_spatial = any(
            # Проверяем для каждого преобразования в списке T:
            # Итерируемся по всем преобразованиям в списке T
            # - Получаем имя класса преобразования через transform.__class__.__name__
            # - Сравниваем с заранее определенным списком spatial_transforms (должен быть объявлен ранее)
            transform.__class__.__name__ in spatial_transforms for transform in T)
        # Создаем композицию преобразований с помощью библиотеки Albumentations
        self.transform = (
            # Если обнаружены пространственные преобразования:
            A.Compose(
                T, # Список преобразований
                # Специальные параметры для обработки bounding boxes:
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])) # Указываем формат bounding boxes (YOLO: center_x, center_y, width, height)
            if self.contains_spatial # Условие применения параметров для bbox
            # Если пространственных преобразований нет:
            else A.Compose(T) # Создаем простую композицию без обработки bounding boxes
        )
        # Отображаем в логах применение преобразований
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # Ошибка импорта (пакет не установлен)
        pass # Пропускаем
    # Пишем ошибку в логи
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

if __name__ == '__main__':
    Albumentations.__init__ = __init__ # Инициализируем класс инициализации для Albumentations
    model = YOLO("yolov8s.pt") # Загружаем предварительно обученную модель YOLO
    model.to('cuda') # Переводим на GPU (видекарту)

    # Запуск тренировки модели
    model.train(
                data="data.yaml", # конфигурация с путями файлов (изображения и аннотации)
                epochs=100, # количество эпох обучения (циклов)
                batch=8, # размер батча
                imgsz=640, # Размер изображений (YOLO обучается на квадратных)
                patience=10, # количество попыток для early stopping (через сколько эпох останавливать обучение, если лоссы не уменьшаются)
                plots=True, # Сохранение графиков обучения
                amp=False, # автоматическое обучение смешанной точности, что обеспечивает более высокую производительность и экономию памяти до 50% на графических процессорах Tensor Core.
                fliplr=0.0,       # горизонтальные отражения (трубы симметричны по длине)
                flipud=0.0,      # вертикальные отражения (не имеют смысла)
                degrees=0.0,      # повороты (трубы строго горизонтальны)
                shear=0.0,        # сдвиги
                translate=0.0,  # <-- отключить смещение
                scale=0.0      # <-- отключить масштабирование
                )
    
    model.val(conf=0.5, iou=0.5) # Проверка модели на валидации (conf - порог уверенности, iou - коэффициент перекрытия между объектами)