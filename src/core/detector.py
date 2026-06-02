"""核心检测器 — 封装模型推理逻辑."""

from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from src.utils.box_utils import filter_by_class, compute_iou


class Detector:
    """YOLO-based reflective vest detector."""

    def __init__(self, weights: str, device: str = "0", half: bool = False):
        self.model = YOLO(weights)
        self.device = device
        self.half = half and device != "cpu"

    def predict(self, source, conf: float = 0.5, iou: float = 0.45, classes: Optional[List[int]] = None, **kwargs):
        """Run inference on source (image path / numpy array / video stream)."""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            half=self.half,
            classes=classes,
            verbose=False,
            **kwargs,
        )
        return results

    def get_detections(self, result, target_classes: Optional[List[int]] = None):
        """Extract detections from a single YOLO result."""
        boxes = result.boxes
        if boxes is None:
            return []

        detections = []
        for i, cls_id in enumerate(boxes.cls):
            if target_classes is not None and int(cls_id) not in target_classes:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu())
            class_id = int(cls_id)
            class_name = self.model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return detections
