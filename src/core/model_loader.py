"""模型加载器 — 统一模型加载与验证."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/model.yaml") -> dict:
    """Load model configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config_device: str) -> str:
    """Resolve device string from config."""
    if config_device == "auto":
        return "0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    return str(config_device)


def validate_weights(weights_path: str) -> bool:
    """Check if weights file exists."""
    path = Path(weights_path)
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    return True
