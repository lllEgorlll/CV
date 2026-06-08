import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from typing import Optional, Tuple


def preprocess_frame(frame, enhance_contrast: bool = False, denoise: bool = False) -> np.ndarray:
    """
    Предобработка кадра для улучшения детекции

    Args:
        frame: изображение в формате numpy array (RGB или BGR)
        enhance_contrast: применять CLAHE для улучшения контраста
        denoise: применять шумоподавление

    Returns:
        обработанное изображение
    """
    # Конвертируем в RGB если нужно
    if len(frame.shape) == 2:  # grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    result = frame.copy()

    # Шумоподавление
    if denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)

    # Улучшение контраста (CLAHE)
    if enhance_contrast:
        # Конвертируем в LAB для CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Применяем CLAHE к L-каналу
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Объединяем обратно
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


def extract_roi(frame: np.ndarray, roi_coords: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Вырезание области интереса (ROI) для ускорения обработки

    Args:
        frame: исходное изображение
        roi_coords: (x, y, w, h) - координаты области. Если None, возвращает весь кадр

    Returns:
        вырезанная область
    """
    if roi_coords is None:
        return frame

    x, y, w, h = roi_coords
    h_frame, w_frame = frame.shape[:2]

    # Проверка границ
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    return frame[y:y + h, x:x + w]


def preprocess_image_file(image_path: str, enhance_contrast: bool = False) -> str:
    """
    Предобработка изображения из файла

    Returns:
        путь к обработанному временному файлу
    """
    frame = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    processed = preprocess_frame(frame_rgb, enhance_contrast=enhance_contrast)

    # Сохраняем во временный файл
    temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_fd)

    Image.fromarray(processed).save(temp_path)

    return temp_path