from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

import os
from typing import List, Dict, Any, Optional

# Цвета для визуализации по классам
CLASS_COLORS = {
    "skimmer_overlay": (255, 0, 0),  # Красный
    "hidden_camera": (255, 165, 0),  # Оранжевый
    "keypad_overlay": (255, 255, 0),  # Жёлтый
    "cash_trapping": (128, 0, 128),  # Фиолетовый
    "card_trapping": (255, 51, 153),  # Розовый
    "suspicious_hand": (0, 102, 255),  # Синий
    "card_reader": (0, 255, 0),  # Зелёный (штатный, не тревога)
}

# Сопоставление ID классов с названиями
CLASS_NAMES = {
    0: "card_reader",
    1: "skimmer_overlay",
    2: "hidden_camera",
    3: "keypad_overlay",
    4: "cash_trapping",
    5: "card_trapping",
    6: "suspicious_hand",
}

# Классы, которые считаются угрозами
THREAT_CLASSES = {
    "skimmer_overlay": "high",
    "hidden_camera": "high",
    "keypad_overlay": "high",
    "cash_trapping": "medium",
    "card_trapping": "medium",
    "suspicious_hand": "low",
}


class ATMFraudDetector:
    """Детектор подделок на банкоматах на базе YOLOv8"""

    def __init__(self, model_path: str = "models/yolov8m_atm.pt", device: str = "cpu"):
        """
        Инициализация детектора

        Args:
            model_path: путь к весам модели
            device: "cpu", "cuda", "mps"
        """
        self.device = device
        self.model = YOLO(model_path)

        # Перемещаем модель на нужное устройство
        if device == "cuda":
            self.model.to("cuda")
        elif device == "mps":
            self.model.to("mps")
        else:
            self.model.to("cpu")

        print(f"[Detector] Загружена модель {model_path} на устройстве {device}")

    def detect(self, image, confidence_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Обнаружение объектов на изображении

        Args:
            image: PIL Image, путь к файлу или numpy array
            confidence_threshold: порог уверенности (0-1)

        Returns:
            список обнаруженных объектов
        """
        # Выполняем инференс
        results = self.model(image, conf=confidence_threshold, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Получаем координаты
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Получаем класс и уверенность
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = CLASS_NAMES.get(class_id, "unknown")
                threat_level = THREAT_CLASSES.get(class_name, "none")

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "threat_level": threat_level
                })

        return detections

    def detect_with_visualization(self, image, confidence_threshold: float = 0.35) -> tuple:
        """
        Обнаружение объектов с визуализацией

        Returns:
            (detections, image_with_boxes, annotated_image_path)
        """
        # Загружаем изображение
        if isinstance(image, str):
            img = cv2.imread(image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_rgb = np.array(image)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img_rgb = image
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Неверный тип image")

        # Детекция
        detections = self.detect(img_rgb, confidence_threshold)

        # Рисуем bounding boxes
        img_with_boxes = img_rgb.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            # Рисуем прямоугольник
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)

            # Рисуем подпись
            label = f"{class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            cv2.rectangle(
                img_with_boxes,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                img_with_boxes,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Сохраняем во временный файл
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)

        # Конвертируем обратно в PIL для сохранения
        Image.fromarray(img_with_boxes).save(temp_path)

        return detections, img_with_boxes, temp_path

    def process_video_frame(self, frame, confidence_threshold: float = 0.35) -> tuple:
        """
        Обработка одного кадра видео (с трекингом будет в отдельном модуле)
        """
        return self.detect_with_visualization(frame, confidence_threshold)


# Для тестирования
if __name__ == "__main__":
    detector = ATMFraudDetector(device="cpu")
    print("Детектор готов к работе")