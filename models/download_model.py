import os
from ultralytics import YOLO


def download_model():
    """Скачивание базовой модели YOLOv8m"""
    model_path = "models/yolov8m_atm.pt"

    if os.path.exists(model_path):
        print(f"Модель уже существует: {model_path}")
        return

    os.makedirs("models", exist_ok=True)

    print("Скачивание базовой модели YOLOv8m...")
    model = YOLO("yolov8m.pt")
    model.save(model_path)

    print(f"Модель сохранена: {model_path}")
    print("\n⚠️ ВНИМАНИЕ: Это базовая модель YOLO, не обученная на ATM-данных.")
    print("Для реального использования требуется дообучение на датасете скиммеров.")


if __name__ == "__main__":
    download_model()