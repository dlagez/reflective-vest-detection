"""JSON 输出单元测试."""

import json
import pytest
from pathlib import Path
import tempfile

from src.services.result_formatter import format_detection_result, format_compliance_report, save_json
from src.schemas.response_schema import DetectionResponse, ComplianceStats, Violation


class TestResultFormatter:
    def test_format_detection_result(self):
        detections = [
            {"class_id": 0, "class_name": "person", "confidence": 0.95, "bbox": [10, 20, 100, 200]},
        ]
        result = format_detection_result(detections, "test.jpg")

        assert result["image"] == "test.jpg"
        assert result["detection_count"] == 1
        assert "timestamp" in result
        assert len(result["detections"]) == 1

    def test_format_compliance_report(self):
        stats = {"total_persons": 3, "wearing_vest": 2, "not_wearing_vest": 1, "compliance_rate": 0.6667}
        violations = [{"person_bbox": [0, 0, 100, 100], "person_confidence": 0.9, "violation": "no_reflective_vest"}]

        report = format_compliance_report(stats, violations, "test.jpg")
        assert report["source"] == "test.jpg"
        assert "summary" in report
        assert "violations" in report
        assert len(report["violations"]) == 1

    def test_save_json(self):
        data = {"key": "value", "nested": {"a": 1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json(data, f"{tmpdir}/output/test.json")
            assert Path(path).exists()

            with open(path, "r") as f:
                loaded = json.load(f)
            assert loaded == data


class TestResponseSchema:
    def test_detection_response(self):
        stats = ComplianceStats(total_persons=5, wearing_vest=4, not_wearing_vest=1, compliance_rate=0.8)
        violations = [Violation(person_bbox=[0, 0, 100, 100], person_confidence=0.9)]

        response = DetectionResponse(
            success=True,
            message="OK",
            source="test.jpg",
            stats=stats,
            violations=violations,
        )

        d = response.to_dict()
        assert d["success"] is True
        assert d["stats"]["compliance_rate"] == 0.8
        assert len(d["violations"]) == 1
