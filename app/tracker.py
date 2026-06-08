from typing import List, Dict, Any, Optional
import numpy as np



class ObjectTracker:
    """Простой трекер объектов на основе IoU (Intersection over Union)"""

    def __init__(self, iou_threshold: float = 0.5, max_lost_frames: int = 10):
        """
        Args:
            iou_threshold: порог IoU для связывания объектов
            max_lost_frames: количество кадров, после которых объект считается потерянным
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks = []  # список активных треков
        self.next_track_id = 1

    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
        """
        Обновление треков на основе новых детекций

        Args:
            detections: список детекций с полями bbox, class_name, confidence
            frame_id: номер кадра

        Returns:
            detections с добавленным track_id
        """
        if not detections:
            # Нет детекций, увеличиваем счетчик потерянных кадров
            for track in self.tracks:
                track["lost_frames"] += 1
            # Удаляем потерянные треки
            self.tracks = [t for t in self.tracks if t["lost_frames"] <= self.max_lost_frames]
            return []

        # Вычисляем IoU между существующими треками и новыми детекциями
        matches = []

        for track in self.tracks:
            track_bbox = track["bbox"]
            track_class = track["class_name"]

            for i, det in enumerate(detections):
                det_bbox = det["bbox"]
                det_class = det["class_name"]

                # Сравниваем только объекты одного класса
                if track_class != det_class:
                    continue

                iou = self._calculate_iou(track_bbox, det_bbox)

                if iou >= self.iou_threshold:
                    matches.append({
                        "track_id": track["track_id"],
                        "detection_idx": i,
                        "iou": iou
                    })

        # Сортируем по убыванию IoU и назначаем соответствия
        matches.sort(key=lambda x: x["iou"], reverse=True)

        used_tracks = set()
        used_detections = set()

        for match in matches:
            if match["track_id"] in used_tracks or match["detection_idx"] in used_detections:
                continue

            # Обновляем существующий трек
            track = next(t for t in self.tracks if t["track_id"] == match["track_id"])
            det = detections[match["detection_idx"]]

            track["bbox"] = det["bbox"]
            track["confidence"] = det["confidence"]
            track["last_seen_frame"] = frame_id
            track["lost_frames"] = 0
            track["detection_history"].append(det)

            det["track_id"] = track["track_id"]

            used_tracks.add(match["track_id"])
            used_detections.add(match["detection_idx"])

        # Создаём новые треки для несоответствующих детекций
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_track = {
                    "track_id": self.next_track_id,
                    "class_name": det["class_name"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "first_seen_frame": frame_id,
                    "last_seen_frame": frame_id,
                    "lost_frames": 0,
                    "detection_history": [det]
                }
                self.tracks.append(new_track)
                det["track_id"] = self.next_track_id
                self.next_track_id += 1

        # Увеличиваем счетчик потерянных кадров для необновлённых треков
        for track in self.tracks:
            if track["track_id"] not in used_tracks:
                track["lost_frames"] += 1

        # Удаляем потерянные треки
        self.tracks = [t for t in self.tracks if t["lost_frames"] <= self.max_lost_frames]

        return detections

    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Вычисление IoU между двумя bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Координаты пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Получить активные треки"""
        return [t for t in self.tracks if t["lost_frames"] == 0]