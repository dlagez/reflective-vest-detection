"""可视化绘制工具函数."""

import cv2
import numpy as np

# Color palette
COLORS = {
    "person": (0, 255, 0),       # Green
    "vest": (0, 165, 255),       # Orange
    "helmet": (255, 0, 0),       # Blue
    "violation": (0, 0, 255),    # Red
}


def draw_bbox(image, bbox, label, color, thickness=2):
    """Draw a bounding box with label on image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def draw_detections(image, detections, show_conf=True):
    """Draw all detections on an image."""
    for det in detections:
        color = COLORS.get(det["class_name"], (128, 128, 128))
        label = det["class_name"]
        if show_conf:
            label += f" {det['confidence']:.2f}"
        draw_bbox(image, det["bbox"], label, color)
    return image


def draw_compliance(image, analysis_results, overlap_threshold=0.5):
    """
    Draw compliance results — green box for compliant, red for violation.
    """
    for person in analysis_results:
        color = (0, 255, 0) if person["wearing_vest"] else (0, 0, 255)
        label = "VEST OK" if person["wearing_vest"] else "NO VEST!"
        draw_bbox(image, person["person_bbox"], label, color, thickness=3)
    return image
