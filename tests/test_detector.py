"""检测器单元测试."""

import pytest
from pathlib import Path
import numpy as np

from src.core.detector import Detector
from src.utils.box_utils import compute_iou, person_vest_overlap, xyxy_to_xywh


class TestBoxUtils:
    def test_iou_same_box(self):
        box = [0, 0, 100, 100]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        assert compute_iou(box1, box2) == 0.0

    def test_iou_partial_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1

    def test_xyxy_to_xywh(self):
        assert xyxy_to_xywh([10, 20, 110, 220]) == [10, 20, 100, 200]

    def test_person_vest_overlap_fully_inside(self):
        person = [0, 0, 200, 300]
        vest = [50, 50, 150, 200]
        assert person_vest_overlap(person, vest) == pytest.approx(1.0)

    def test_person_vest_overlap_no_overlap(self):
        person = [0, 0, 100, 100]
        vest = [200, 200, 300, 300]
        assert person_vest_overlap(person, vest) == 0.0


class TestDetector:
    def test_get_detections_empty_result(self):
        """Test detection extraction with mock data."""
        # Placeholder: requires actual model weights for full integration test
        pass
