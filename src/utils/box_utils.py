"""Bounding box 工具函数."""

from typing import List


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def person_vest_overlap(person_bbox: List[float], vest_bbox: List[float]) -> float:
    """
    Compute the overlap ratio of vest box within person box.
    Returns fraction of vest area that falls inside person area.
    """
    x1 = max(person_bbox[0], vest_bbox[0])
    y1 = max(person_bbox[1], vest_bbox[1])
    x2 = min(person_bbox[2], vest_bbox[2])
    y2 = min(person_bbox[3], vest_bbox[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    vest_area = (vest_bbox[2] - vest_bbox[0]) * (vest_bbox[3] - vest_bbox[1])

    return inter_area / vest_area if vest_area > 0 else 0.0


def filter_by_class(detections: list, class_name: str) -> list:
    """Filter detections by class name."""
    return [d for d in detections if d.get("class_name") == class_name]


def align_to_stride(h: int, w: int, stride: int = 32) -> tuple:
    """
    Round h/w UP to the nearest multiple of stride for YOLO input.

    YOLO's backbone FPN has stride=32, so both dimensions must be
    divisible by 32 — otherwise YOLO silently resizes and warns.
    """
    new_h = ((h + stride - 1) // stride) * stride
    new_w = ((w + stride - 1) // stride) * stride
    return (new_h, new_w)


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
