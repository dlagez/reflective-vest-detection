"""反光衣检测业务服务."""

from src.core.detector import Detector
from src.utils.box_utils import person_vest_overlap


class VestDetectionService:
    """Detect people and determine vest compliance."""

    def __init__(self, detector: Detector, overlap_threshold: float = 0.5):
        self.detector = detector
        self.overlap_threshold = overlap_threshold

    def analyze(self, source, conf: float = 0.5, iou: float = 0.45):
        """
        Run detection and classify each person as wearing vest or not.

        Returns list of person detections with vest compliance status.
        """
        results = self.detector.predict(source, conf=conf, iou=iou)
        analysis = []

        for result in results:
            detections = self.detector.get_detections(result)
            persons = [d for d in detections if d["class_name"] == "person"]
            vests = [d for d in detections if d["class_name"] == "vest"]

            for person in persons:
                has_vest = False
                for vest in vests:
                    iou = person_vest_overlap(person["bbox"], vest["bbox"])
                    if iou >= self.overlap_threshold:
                        has_vest = True
                        break

                analysis.append({
                    "person_bbox": person["bbox"],
                    "person_confidence": person["confidence"],
                    "wearing_vest": has_vest,
                    "vest_confidence": max(
                        (v["confidence"] for v in vests
                         if person_vest_overlap(person["bbox"], v["bbox"]) >= self.overlap_threshold),
                        default=0.0,
                    ),
                })

        return analysis
