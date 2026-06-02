"""视频检测脚本."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.model_loader import load_config, resolve_device, validate_weights
from src.core.detector import Detector
from src.core.video_processor import VideoProcessor
from src.services.vest_detection_service import VestDetectionService
from src.services.compliance_service import ComplianceService
from src.services.result_formatter import format_compliance_report, save_json
from src.utils.draw_utils import draw_detections
from src.utils.file_utils import ensure_dir, get_output_path
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

    detector = Detector(weights=weights, device=device,half=half)
    processor = VideoProcessor(detector)

    if source is None:
        video_files = list(Path("data/videos/").glob("*.mp4"))
        if not video_files:
            logger.error("No videos found in data/videos/")
            return
        source = str(video_files[0])

    logger.info(f"Processing video: {source}")

    output_path = get_output_path(f"{output_dir}/videos", source, suffix="_result")

    def on_frame(frame, results):
        for result in results:
            detections = detector.get_detections(result)
            draw_detections(frame, detections)
        return frame

    all_results = processor.process(
        source=source,
        output_path=output_path,
        conf=conf,
        iou=iou,
        frame_callback=on_frame,
        show_preview=False,
    )

    logger.info(f"Video output saved to: {output_path}")
