"""文件操作工具函数."""

import os
from pathlib import Path
from typing import List


def find_images(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all images in a directory."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    images = []
    for ext in extensions:
        images.extend(dir_path.glob(f"*{ext}"))
        images.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted([str(p) for p in images])


def find_videos(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all videos in a directory."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    videos = []
    for ext in extensions:
        videos.extend(dir_path.glob(f"*{ext}"))
        videos.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted([str(p) for p in videos])


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if needed, return path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(output_dir: str, filename: str, suffix: str = "") -> str:
    """Generate an output path in the output directory."""
    p = Path(filename)
    out_name = f"{p.stem}{suffix}{p.suffix}"
    return str(Path(output_dir) / out_name)
