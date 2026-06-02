"""结果格式化工具 — 标准化输出结构."""

import json
from pathlib import Path
from datetime import datetime


def format_detection_result(detections: list, image_path: str) -> dict:
    """Format detections into a standardized dict."""
    return {
        "image": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "detection_count": len(detections),
        "detections": detections,
    }


def format_compliance_report(stats: dict, violations: list, source: str) -> dict:
    """Format compliance analysis into a report."""
    return {
        "source": str(source),
        "timestamp": datetime.now().isoformat(),
        "summary": stats,
        "violations": violations,
    }


def save_json(data: dict, output_path: str) -> str:
    """Save result dict to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path
