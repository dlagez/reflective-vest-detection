"""API 响应数据结构."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ComplianceStats:
    total_persons: int
    wearing_vest: int
    not_wearing_vest: int
    compliance_rate: float

    def to_dict(self) -> dict:
        return {
            "total_persons": self.total_persons,
            "wearing_vest": self.wearing_vest,
            "not_wearing_vest": self.not_wearing_vest,
            "compliance_rate": round(self.compliance_rate, 4),
        }


@dataclass
class Violation:
    person_bbox: List[float]
    person_confidence: float
    violation: str = "no_reflective_vest"

    def to_dict(self) -> dict:
        return {
            "person_bbox": self.person_bbox,
            "person_confidence": round(self.person_confidence, 4),
            "violation": self.violation,
        }


@dataclass
class DetectionResponse:
    success: bool
    message: str
    source: str
    stats: Optional[ComplianceStats] = None
    violations: List[Violation] = field(default_factory=list)
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "message": self.message,
            "source": self.source,
        }
        if self.stats:
            result["stats"] = self.stats.to_dict()
        if self.violations:
            result["violations"] = [v.to_dict() for v in self.violations]
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ErrorResponse:
    success: bool = False
    message: str = ""
    error_code: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "error_code": self.error_code,
        }
