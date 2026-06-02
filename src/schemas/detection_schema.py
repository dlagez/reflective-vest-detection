"""检测数据结构定义."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> List[float]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_list(),
        }


@dataclass
class PersonAnalysis:
    person_bbox: BoundingBox
    person_confidence: float
    wearing_vest: bool
    vest_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "person_bbox": self.person_bbox.to_list(),
            "person_confidence": self.person_confidence,
            "wearing_vest": self.wearing_vest,
            "vest_confidence": self.vest_confidence,
        }


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    person_analyses: List[PersonAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
            "person_analyses": [p.to_dict() for p in self.person_analyses],
        }
