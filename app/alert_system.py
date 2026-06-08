import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional



class AlertSystem:
    """Система оповещений для службы безопасности"""

    def __init__(self, export_dir: str = "data/export"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

        # История тревог
        self.alert_history = []

        # Пороги для генерации тревог
        self.THREAT_THRESHOLDS = {
            "high": 0.65,  # высокая угроза
            "medium": 0.50,  # средняя
            "low": 0.35  # низкая
        }

    def analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ детекций и генерация тревог

        Returns:
            словарь с результатами анализа
        """
        threats = []
        alert_level = "none"
        alert_reason = None

        for det in detections:
            threat_level = det.get("threat_level", "none")
            confidence = det["confidence"]

            if threat_level != "none":
                threshold = self.THREAT_THRESHOLDS.get(threat_level, 0.5)

                if confidence >= threshold:
                    threats.append(det)

                    # Определяем максимальный уровень тревоги
                    if threat_level == "high":
                        alert_level = "high"
                        alert_reason = f"Обнаружен {det['class_name']} (уверенность {confidence:.0%})"
                    elif threat_level == "medium" and alert_level != "high":
                        alert_level = "medium"
                        alert_reason = f"Обнаружен {det['class_name']} (уверенность {confidence:.0%})"
                    elif threat_level == "low" and alert_level not in ["high", "medium"]:
                        alert_level = "low"
                        alert_reason = f"Обнаружена подозрительная активность: {det['class_name']}"

        return {
            "alert": alert_level != "none",
            "alert_level": alert_level,
            "alert_reason": alert_reason,
            "threats": threats,
            "total_detections": len(detections)
        }

    def generate_alert_log(self, analysis: Dict[str, Any], camera_id: str = "unknown") -> Dict[str, Any]:
        """Генерация структурированного лога тревоги"""
        alert_log = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "alert": analysis["alert"],
            "alert_level": analysis["alert_level"],
            "alert_reason": analysis["alert_reason"],
            "threats": analysis["threats"],
            "total_detections": analysis["total_detections"]
        }

        if analysis["alert"]:
            self.alert_history.append(alert_log)

            # Сохраняем в JSON-файл
            self._save_alert_to_file(alert_log)

        return alert_log

    def _save_alert_to_file(self, alert_log: Dict[str, Any]):
        """Сохранение тревоги в файл"""
        timestamp = alert_log["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"alert_{timestamp}.json"
        filepath = os.path.join(self.export_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(alert_log, f, ensure_ascii=False, indent=2)

        # Также сохраняем в текстовый лог
        txt_filename = f"alert_{timestamp}.txt"
        txt_filepath = os.path.join(self.export_dir, txt_filename)

        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ОБНАРУЖЕНА УГРОЗА НА БАНКОМАТЕ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Время: {alert_log['timestamp']}\n")
            f.write(f"Камера: {alert_log['camera_id']}\n")
            f.write(f"Уровень угрозы: {alert_log['alert_level'].upper()}\n")
            f.write(f"Причина: {alert_log['alert_reason']}\n\n")
            f.write("Обнаруженные объекты:\n")
            for threat in alert_log.get("threats", []):
                f.write(f"  - {threat['class_name']} (уверенность: {threat['confidence']:.0%})\n")
            f.write("\n" + "=" * 70 + "\n")

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получить историю тревог"""
        return self.alert_history[-limit:]

    def export_to_json(self, detections: List[Dict[str, Any]], analysis: Dict[str, Any],
                       camera_id: str = "unknown") -> str:
        """Экспорт результатов в JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_result_{timestamp}.json"
        filepath = os.path.join(self.export_dir, filename)

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "detections": detections,
            "analysis": analysis
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return filepath

    def format_alert_text(self, alert_log: Dict[str, Any]) -> str:
        """Форматирование тревоги для текстового вывода"""
        if not alert_log["alert"]:
            return "Угроз не обнаружено"

        lines = [
            "=" * 60,
            f"🚨 ТРЕВОГА! 🚨",
            "=" * 60,
            f"Уровень: {alert_log['alert_level'].upper()}",
            f"Причина: {alert_log['alert_reason']}",
            f"Время: {alert_log['timestamp']}",
            "-" * 60
        ]

        for threat in alert_log.get("threats", []):
            lines.append(f"• {threat['class_name']}: {threat['confidence']:.0%} уверенности")

        lines.append("=" * 60)

        return "\n".join(lines)