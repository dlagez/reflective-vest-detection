"""API 路由定义."""

import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.core.model_loader import load_config, resolve_device, validate_weights
from src.core.detector import Detector
from src.services.vest_detection_service import VestDetectionService
from src.services.compliance_service import ComplianceService
from src.services.result_formatter import format_compliance_report
from src.schemas.response_schema import DetectionResponse, ErrorResponse

router = APIRouter()


def get_detector() -> Detector:
    """Lazy-load detector singleton."""
    config = load_config()
    weights = os.getenv("MODEL_WEIGHTS", config["model"]["weights"])

    if not validate_weights(weights):
        raise RuntimeError(f"Weights not found: {weights}")

    device = resolve_device(config["model"].get("device", "0"))
    half = config["model"].get("half", False)

    return Detector(weights=weights, device=device, half=half)


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image for vest detection."""
    try:
        detector = get_detector()
        service = VestDetectionService(detector)

        # Save uploaded file temporarily
        upload_path = Path(f"outputs/uploads/{file.filename}")
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        upload_path.write_bytes(content)

        analysis = service.analyze(str(upload_path))
        stats = ComplianceService.compute_stats(analysis)
        violations = ComplianceService.get_violations(analysis)

        report = format_compliance_report(stats, violations, str(upload_path))

        return JSONResponse(content=report)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/url")
async def detect_from_url(source_url: str):
    """Detect from image URL."""
    try:
        detector = get_detector()
        service = VestDetectionService(detector)

        analysis = service.analyze(source_url)
        stats = ComplianceService.compute_stats(analysis)
        violations = ComplianceService.get_violations(analysis)

        report = format_compliance_report(stats, violations, source_url)
        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def system_status():
    """System health and model status."""
    config = load_config()
    weights = os.getenv("MODEL_WEIGHTS", config["model"]["weights"])
    weights_ok = validate_weights(weights)

    return {
        "model_loaded": weights_ok,
        "weights_path": weights,
        "device": config["model"].get("device", "0"),
    }
