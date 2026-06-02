"""图像检测脚本."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.model_loader import load_config, resolve_device, validate_weights
from src.core.detector import Detector
from src.core.image_processor import ImageProcessor
from src.services.vest_detection_service import VestDetectionService
from src.services.compliance_service import ComplianceService
from src.services.result_formatter import format_compliance_report, save_json
from src.utils.draw_utils import draw_detections, draw_compliance
from src.utils.file_utils import ensure_dir
from src.utils.logger import logger

load_dotenv()


def run(source: str = None, config: str = "configs/model.yaml"):
    cfg = load_config(config)
    weights = os.getenv("MODEL_WEIGHTS", cfg["model"]["weights"])

    if not validate_weights(weights):
        logger.error(f"Weights not found: {weights}")
        return

    device = resolve_device(cfg["model"].get("device", "0"))
    half = cfg["model"].get("half", False)
    conf = float(os.getenv("CONF_THRESHOLD", cfg["inference"]["conf"]))
    iou = float(os.getenv("IOU_THRESHOLD", cfg["inference"]["iou"]))
    output_dir = os.getenv("OUTPUT_DIR", cfg["output"]["output_dir"])

    detector = Detector(weights=weights, device=device, half=half)
    processor = ImageProcessor(detector)
    service = VestDetectionService(detector)

    if source is None:
        source = "data/images/"

    logger.info(f"Processing images from: {source}")
    results = processor.process(source, conf=conf, iou=iou)

    all_analysis = []
    for result in results:
        detections = detector.get_detections(result)
        analysis = service.analyze(result.orig_img if hasattr(result, "orig_img") else result, conf=conf, iou=iou)
        all_analysis.extend(analysis)

    stats = ComplianceService.compute_stats(all_analysis)
    violations = ComplianceService.get_violations(all_analysis)
    report = format_compliance_report(stats, violations, source)

    # Save JSON
    json_path = save_json(report, f"{output_dir}/json/image_result.json")
    logger.info(f"Results saved to: {json_path}")

    # Save visualization
    if cfg["output"].get("save_viz"):
        viz_dir = ensure_dir(f"{output_dir}/images")
        logger.info(f"Visualization saved to: {viz_dir}")

    logger.info(f"Compliance rate: {stats['compliance_rate']:.2%}")
