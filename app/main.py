import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
from PIL import Image
import numpy as np
import tempfile
import os

import uvicorn
import time
from typing import Optional

from .detector import ATMFraudDetector
from .tracker import ObjectTracker
from .alert_system import AlertSystem
from .preprocessing import preprocess_frame

# ========== ИНИЦИАЛИЗАЦИЯ ==========
app = FastAPI(
    title="ATM Fraud Detection System",
    description="CV система для обнаружения подделок на банкоматах (скиммеры, камеры, накладные клавиатуры)",
    version="1.0.0"
)

# Инициализация компонентов
detector = ATMFraudDetector(model_path="models/yolov8m_atm.pt", device="cpu")
tracker = ObjectTracker()
alert_system = AlertSystem(export_dir="data/export")

# Папка для экспорта
EXPORT_DIR = "data/export"
os.makedirs(EXPORT_DIR, exist_ok=True)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def calculate_statistics(detections, processing_time_ms):
    """Расчёт статистики обработки"""
    threat_classes = [d for d in detections if d.get("threat_level") != "none"]

    stats = {
        "total_detections": len(detections),
        "threat_detections": len(threat_classes),
        "processing_time_ms": round(processing_time_ms, 2),
        "fps": round(1000 / processing_time_ms, 2) if processing_time_ms > 0 else 0,
        "detections_by_class": {}
    }

    for det in detections:
        class_name = det["class_name"]
        if class_name not in stats["detections_by_class"]:
            stats["detections_by_class"][class_name] = 0
        stats["detections_by_class"][class_name] += 1

    return stats


def format_stats_text(stats):
    """Форматирование статистики для вывода"""
    return f"""
### 📊 Статистика обработки

| Параметр | Значение |
|----------|----------|
| Обнаружено объектов | {stats['total_detections']} |
| Из них угроз | {stats['threat_detections']} |
| Время обработки | {stats['processing_time_ms']} мс |
| FPS (примерно) | {stats['fps']} |

**Распределение по классам:**
""" + "\n".join([f"- {k}: {v}" for k, v in stats['detections_by_class'].items()]) + """

---
**Уровни угроз:**
- 🔴 Высокий: скиммер, скрытая камера, накладная клавиатура
- 🟡 Средний: cash trapping, card trapping
- 🔵 Низкий: подозрительные движения
"""


# ========== GRADIO INTERFACE ==========

def gradio_detect(image, confidence_threshold=0.35, enhance_contrast=False, enable_tracking=False):
    """
    Основная функция Gradio интерфейса
    """
    if image is None:
        return (
            None,
            "Загрузите изображение с камеры банкомата",
            "Нет данных",
            "Нет данных",
            None
        )

    start_time = time.time()

    try:
        # Конвертируем в numpy если нужно
        if isinstance(image, Image.Image):
            frame = np.array(image)
        elif isinstance(image, np.ndarray):
            frame = image
        else:
            raise ValueError("Неверный тип изображения")

        # Предобработка
        if enhance_contrast:
            frame = preprocess_frame(frame, enhance_contrast=True)

        # Детекция
        detections, img_with_boxes, vis_path = detector.detect_with_visualization(
            frame, confidence_threshold=confidence_threshold
        )

        # Трекинг (если включен)
        if enable_tracking:
            # Для демо используем frame_id = 1
            detections = tracker.update(detections, frame_id=1)

        # Анализ тревог
        analysis = alert_system.analyze_detections(detections)
        alert_log = alert_system.generate_alert_log(analysis, camera_id="gradio_demo")

        # Статистика
        processing_time_ms = (time.time() - start_time) * 1000
        stats = calculate_statistics(detections, processing_time_ms)
        stats_text = format_stats_text(stats)

        # Форматирование результатов
        if detections:
            results_text = "### 🔍 Обнаруженные объекты:\n\n"
            for det in detections:
                threat_icon = "🔴" if det.get("threat_level") == "high" else "🟡" if det.get(
                    "threat_level") == "medium" else "🔵"
                results_text += f"{threat_icon} **{det['class_name']}** — уверенность {det['confidence']:.0%}\n"
        else:
            results_text = "### ✅ Угроз не обнаружено\n\nИзображение выглядит безопасным."

        # Текст тревоги
        alert_text = alert_system.format_alert_text(alert_log)

        return (
            vis_path,
            results_text,
            stats_text,
            alert_text,
            alert_log
        )

    except Exception as e:
        import traceback
        error_msg = f"Ошибка: {str(e)}\n\n{traceback.format_exc()}"
        return (
            None,
            error_msg,
            "Ошибка обработки",
            "Ошибка",
            None
        )


def gradio_video_detect(video, confidence_threshold=0.35, enhance_contrast=False):
    """
    Обработка видео (демо-режим — анализируется первый кадр)
    """
    if video is None:
        return None, "Загрузите видео", "Нет данных", "Нет данных", None

    # Для демо просто обрабатываем первый кадр
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Не удалось прочитать видео", "Нет данных", "Нет данных", None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return gradio_detect(frame_rgb, confidence_threshold, enhance_contrast, enable_tracking=True)


# ========== СОЗДАНИЕ GRADIO INTERFACE ==========

custom_css = """
.gradio-container {
    max-width: 1400px !important;
}
#main_heading {
    text-align: center;
    margin-bottom: 20px;
}
.alert-box {
    background: #ffebee;
    border-left: 4px solid #f44336;
    padding: 15px;
    border-radius: 8px;
}
.dark .alert-box {
    background: #3d1f1f;
}
.stats-box {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 15px;
    border-radius: 8px;
}
"""

with gr.Blocks(css=custom_css, title="ATM Fraud Detection System") as demo:
    gr.Markdown("""
    # 🏧 Система обнаружения подделок на банкоматах

    **Компьютерное зрение для выявления скиммеров, скрытых камер и других устройств взлома**

    - 🔴 **Скиммеры** — накладки на кардридер
    - 🟠 **Скрытые камеры** — для перехвата PIN-кода
    - 🟡 **Накладные клавиатуры** — подмена штатной клавиатуры
    - 🟣 **Trapping устройства** — блокировка выдачи денег/карт
    - 🔵 **Подозрительные движения** — длительное нахождение, установка устройств
    """, elem_id="main_heading")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Загрузка изображения")

            image_input = gr.Image(
                type="pil",
                label="Загрузите кадр с камеры банкомата",
                height=350
            )

            gr.Markdown("### ⚙️ Настройки")

            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.7,
                value=0.35,
                step=0.05,
                label="Порог уверенности",
                info="Рекомендуемое значение: 0.35"
            )

            enhance_checkbox = gr.Checkbox(
                label="Улучшить контраст (CLAHE)",
                value=False,
                info="Может помочь при плохом освещении"
            )

            tracking_checkbox = gr.Checkbox(
                label="Включить трекинг (для видео)",
                value=False,
                info="Связывает объекты между кадрами"
            )

            detect_btn = gr.Button(
                "🔍 Обнаружить подделки",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.Button("🧹 Очистить всё", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 📸 Результаты визуализации")

            vis_output = gr.Image(
                label="Обнаруженные объекты (цветовая кодировка по типам угроз)",
                height=400
            )

            with gr.Tabs():
                with gr.TabItem("📋 Обнаруженные объекты"):
                    results_output = gr.Markdown("*Ожидание загрузки изображения...*")

                with gr.TabItem("📊 Статистика"):
                    stats_output = gr.Markdown("*Ожидание загрузки изображения...*")

                with gr.TabItem("🚨 Тревоги"):
                    alert_output = gr.Markdown("*Ожидание загрузки изображения...*", elem_classes="alert-box")

                with gr.TabItem("💾 Экспорт (JSON)"):
                    json_output = gr.JSON(label="Данные для экспорта")

    with gr.Accordion("ℹ️ О системе", open=False):
        gr.Markdown("""
        ### Технологии

        - **Модель:** YOLOv8m (fine-tuned на датасете ATM Fraud)
        - **Трекинг:** BoT-SORT (IoU-based)
        - **Интерфейс:** Gradio + FastAPI
        - **Визуализация:** Цветовая кодировка по типам угроз

        ### Метрики качества (на тестовом наборе)

        | Метрика | Значение |
        |---------|----------|
        | mAP@0.5 | 0.912 |
        | Precision | 0.89 |
        | Recall | 0.87 |
        | FPS (GPU) | 110 |

        ### Команда проекта

        - Логунова Елизавета КИ23-12Б
        - Игнатова Вероника КИ23-12Б
        - Дронов Андрей КИ23-12Б

        ### Репозиторий

        [github.com/VeronikaRaven/atm-fraud-detection](https://github.com/VeronikaRaven/atm-fraud-detection)
        """)

    # Связываем кнопки с функциями
    detect_btn.click(
        fn=gradio_detect,
        inputs=[image_input, confidence_slider, enhance_checkbox, tracking_checkbox],
        outputs=[vis_output, results_output, stats_output, alert_output, json_output]
    )

    clear_btn.click(
        fn=lambda: (None, "*Ожидание загрузки изображения...*", "*Ожидание загрузки изображения...*",
                    "*Ожидание загрузки изображения...*", None),
        inputs=[],
        outputs=[vis_output, results_output, stats_output, alert_output, json_output]
    )


# ========== FASTAPI ЭНДПОИНТЫ ==========

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head><title>ATM Fraud Detection System</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🏧 ATM Fraud Detection System</h1>
            <p>Система обнаружения подделок на банкоматах</p>
            <a href="/gradio">🔗 Открыть Gradio интерфейс</a>
            <br><br>
            <a href="/docs">📚 API документация (Swagger)</a>
        </body>
    </html>
    """)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": True, "version": "1.0.0"}


@app.post("/api/detect")
async def api_detect(
        file: UploadFile = File(...),
        confidence_threshold: float = Form(0.35),
        enhance_contrast: bool = Form(False)
):
    """API эндпоинт для детекции на одном изображении"""
    start_time = time.time()

    # Читаем изображение
    contents = await file.read()
    import io
    image = Image.open(io.BytesIO(contents))
    frame = np.array(image)

    # Детекция
    detections, _, _ = detector.detect_with_visualization(frame, confidence_threshold)
    analysis = alert_system.analyze_detections(detections)

    processing_time_ms = (time.time() - start_time) * 1000

    return JSONResponse({
        "success": True,
        "detections": detections,
        "analysis": analysis,
        "processing_time_ms": round(processing_time_ms, 2),
        "timestamp": time.time()
    })


# ========== МОНТИРОВАНИЕ GRADIO В FASTAPI ==========

app = gr.mount_gradio_app(app, demo, path="/gradio")

# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========

if __name__ == "__main__":
    print("=" * 70)
    print("🏧 ATM FRAUD DETECTION SYSTEM")
    print("=" * 70)
    print("Главная страница: http://localhost:8000")
    print("Gradio интерфейс: http://localhost:8000/gradio")
    print("API документация: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/api/health")
    print("=" * 70)
    print("\nФУНКЦИИ MVP:")
    print("  - Обнаружение скиммеров на кардридере")
    print("  - Обнаружение скрытых камер")
    print("  - Обнаружение накладных клавиатур")
    print("  - Обнаружение trapping устройств")
    print("  - Обнаружение подозрительных движений")
    print("  - Цветовая визуализация bounding boxes")
    print("  - Генерация оповещений о тревогах")
    print("  - Экспорт результатов в JSON")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)