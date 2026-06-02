"""图像处理器 — 单张/批量图像推理."""

from pathlib import Path
from typing import Optional

from src.core.detector import Detector


class ImageProcessor:
    """Process images with a detector."""

    def __init__(self, detector: Detector):
        self.detector = detector

    def process(
        self,
        source: str,
        conf: float = 0.5,
        iou: float = 0.45,
        save_dir: Optional[str] = None,
    ) -> list:
        """
        Process a single image or directory of images.

        Returns list of YOLO results.
        """
        source_path = Path(source)

        if source_path.is_file():
            image_files = [source_path]
        elif source_path.is_dir():
            image_files = sorted(source_path.glob("*.jpg"))
            image_files += sorted(source_path.glob("*.png"))
            image_files += sorted(source_path.glob("*.jpeg"))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        all_results = []
        for img_path in image_files:
            results = self.detector.predict(img_path, conf=conf, iou=iou)
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                for r in results:
                    out_name = Path(save_dir) / img_path.name
                    r.save(filename=str(out_name))
            all_results.extend(results)

        return all_results
