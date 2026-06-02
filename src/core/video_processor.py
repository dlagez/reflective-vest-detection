"""视频处理器 — 逐帧推理与结果写入."""

import cv2
from pathlib import Path
from typing import Callable, Optional

from src.core.detector import Detector


class VideoProcessor:
    """Process video frame-by-frame with a detector."""

    def __init__(self, detector: Detector):
        self.detector = detector

    def process(
        self,
        source: str,
        output_path: str,
        conf: float = 0.5,
        iou: float = 0.45,
        frame_callback: Optional[Callable] = None,
        show_preview: bool = False,
    ) -> list:
        """
        Process a video file frame-by-frame.

        Returns list of per-frame detection results.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.predict(frame, conf=conf, iou=iou)
            all_results.append(results)

            viz_frame = frame.copy()
            if frame_callback:
                viz_frame = frame_callback(viz_frame, results)

            out.write(viz_frame)

            if show_preview:
                cv2.imshow("Video", viz_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return all_results
