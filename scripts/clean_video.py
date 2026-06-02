"""清理视频中的损坏帧/空帧.

读取原始视频，丢弃无法解码的帧和全黑/纯色帧，输出干净视频。
不依赖 YOLO，纯 OpenCV 操作。

用法:
    # 单个文件
    python scripts/clean_video.py --source data/videos/broken.mp4

    # 整个文件夹
    python scripts/clean_video.py --source data/videos/

    # 调整亮度阈值
    python scripts/clean_video.py --source data/videos/ --brightness 5.0

    # 保留中间文件（默认会保存在原文件同级 _cleaned 后缀）
    python scripts/clean_video.py --source data/videos/broken.mp4
"""

import cv2
import argparse
from pathlib import Path

from src.utils.logger import logger


def clean_video(input_path: str, output_path: str = None,
                brightness_threshold: float = 1.0) -> dict:
    """
    Read a video, drop broken/blank frames, write a clean copy.

    A frame is dropped if:
      - OpenCV cannot decode it
      - Mean brightness < threshold (all black)
      - Pixel std < 1.0 (pure solid color, no detail)

    Args:
        input_path: Path to the broken video.
        output_path: Path for the cleaned video.
                     Defaults to <original>_cleaned.mp4 next to the original.
        brightness_threshold: Minimum mean brightness (0-255) to keep a frame.

    Returns:
        Dict with keys: total, kept, dropped.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_cleaned{p.suffix}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total = 0
    kept = 0
    dropped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Retry once — sometimes decode glitch recovers
            ret, frame = cap.read()
            if not ret:
                break
            total += 1
            dropped += 1
            continue

        total += 1

        # Skip blank / solid-color frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        std = gray.std()

        if mean_brightness < brightness_threshold or std < 1.0:
            dropped += 1
            logger.debug(
                f"Frame {total} dropped "
                f"(brightness={mean_brightness:.1f}, std={std:.1f})"
            )
            continue

        out.write(frame)
        kept += 1

    cap.release()
    out.release()

    logger.info(
        f"[{Path(input_path).name}] "
        f"{kept}/{total} frames kept, {dropped} dropped "
        f"({dropped / max(total, 1) * 100:.1f}%)"
    )
    logger.info(f"Cleaned video saved to: {output_path}")

    return {"total": total, "kept": kept, "dropped": dropped}


def main():
    parser = argparse.ArgumentParser(description="Clean broken frames from video(s)")
    parser.add_argument("--source", required=True, help="Video file or directory")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Min mean brightness (0-255) to keep frame")
    args = parser.parse_args()

    p = Path(args.source)
    files = []

    if p.is_file():
        files = [p]
    elif p.is_dir():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            files.extend(p.glob(ext))
        files.sort()
    else:
        logger.error(f"Path not found: {args.source}")
        return

    logger.info(f"Found {len(files)} video(s) to clean")

    total_dropped = 0
    for f in files:
        result = clean_video(str(f), brightness_threshold=args.brightness)
        total_dropped += result["dropped"]

    logger.info(f"Done. Total {total_dropped} frames dropped across {len(files)} video(s)")


if __name__ == "__main__":
    main()
