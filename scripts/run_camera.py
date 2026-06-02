"""摄像头实时检测脚本."""

import os
import cv2

from dotenv import load_dotenv

from src.core.model_loader import load_config, resolve_device, validate_weights
from src.core.detector import Detector
from src.services.vest_detection_service import VestDetectionService
from src.utils.draw_utils import draw_detections, draw_compliance
from src.utils.logger import logger

load_dotenv()


def run(config: str = "configs/model.yaml", camera_id: int = 0):
    cfg = load_config(config)
    weights = os.getenv("MODEL_WEIGHTS", cfg["model"]["weights"])

    if not validate_weights(weights):
        logger.error(f"Weights not found: {weights}")
        return

    device = resolve_device(cfg["model"].get("device", "0"))
    half = cfg["model"].get("half", False)
    conf = float(os.getenv("CONF_THRESHOLD", cfg["inference"]["conf"]))
    iou = float(os.getenv("IOU_THRESHOLD", cfg["inference"]["iou"]))

    detector = Detector(weights=weights, device=device, half=half)
    service = VestDetectionService(detector)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return

    logger.info(f"Camera started (device={camera_id}). Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.predict(frame, conf=conf, iou=iou)
            analysis = service.analyze(frame, conf=conf, iou=iou)

            draw_detections(frame, detector.get_detections(results[0]) if results else [])
            draw_compliance(frame, analysis)

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera stopped.")
