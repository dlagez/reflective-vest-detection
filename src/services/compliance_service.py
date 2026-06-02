"""合规性服务 — 统计与违规报告."""

from typing import List


class ComplianceService:
    """Analyze detection results for safety compliance."""

    @staticmethod
    def compute_stats(analysis_results: List[dict]) -> dict:
        """
        Compute compliance statistics from analysis results.

        Args:
            analysis_results: List of per-person analysis dicts from VestDetectionService.

        Returns:
            Stats dict with counts and compliance rate.
        """
        total = len(analysis_results)
        compliant = sum(1 for r in analysis_results if r.get("wearing_vest"))
        non_compliant = total - compliant

        return {
            "total_persons": total,
            "wearing_vest": compliant,
            "not_wearing_vest": non_compliant,
            "compliance_rate": compliant / total if total > 0 else 1.0,
        }

    @staticmethod
    def get_violations(analysis_results: List[dict]) -> List[dict]:
        """Return list of persons not wearing vests."""
        return [
            {
                "person_bbox": r["person_bbox"],
                "person_confidence": r["person_confidence"],
                "violation": "no_reflective_vest",
            }
            for r in analysis_results
            if not r.get("wearing_vest")
        ]
