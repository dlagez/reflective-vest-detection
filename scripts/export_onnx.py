"""导出 ONNX 模型脚本."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.logger import logger

load_dotenv()


def run(weights: str = None, output: str = None):
    weights = weights or os.getenv("MODEL_WEIGHTS", "weights/yolo11m_safety.pt")
    output = output or "weights/yolo11m_safety.onnx"

    if not Path(weights).exists():
        logger.error(f"Weights not found: {weights}")
        return

    logger.info(f"Loading model: {weights}")
    model = YOLO(weights)

    logger.info(f"Exporting to ONNX: {output}")
    model.export(format="onnx", dynamic=True, simplify=True)

    logger.info(f"Export complete: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="PT weights path")
    parser.add_argument("--output", type=str, default=None, help="ONNX output path")
    args = parser.parse_args()
    run(weights=args.weights, output=args.output)
