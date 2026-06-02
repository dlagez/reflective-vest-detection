"""Simple YOLO video detection script.

This script intentionally delegates preprocessing, box scaling, drawing, and
video writing to Ultralytics YOLO. Avoid passing a 2D 1080p imgsz such as
[1080, 1920]; if a larger inference size is needed, pass a single int like
--imgsz 1920 so YOLO can handle stride alignment itself.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def parse_classes(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def default_run_name(video: str) -> str:
    stem = Path(video).stem
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]
    return f"{stem}_detect"


class FFmpegNVENCWriter:
    def __init__(self, output_path: Path, fps: float, width: int, height: int, cq: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            str(cq),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")


def read_video_info(video: str, vid_stride: int) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if vid_stride > 1:
        fps = fps / vid_stride
        frame_count = frame_count // vid_stride if frame_count > 0 else 0
    return fps, width, height, frame_count


def run(
    video: str,
    weights: str,
    output: str,
    name: str | None,
    conf: float,
    iou: float,
    device: str | None,
    imgsz: int | None,
    half: bool,
    classes: list[int] | None,
    vid_stride: int,
    verbose: bool,
    encoder: str,
    cq: int,
    progress: bool,
) -> None:
    model = YOLO(weights)
    project = str(Path(output).resolve())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = name or default_run_name(video)
    run_name = f"{base_run_name}_{timestamp}"
    run_dir = Path(project) / run_name
    output_video = run_dir / f"{Path(video).stem}.mp4"
    fps, width, height, total_frames = read_video_info(video, vid_stride)

    predict_kwargs = {
        "source": video,
        "save": encoder == "yolo",
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "conf": conf,
        "iou": iou,
        "stream": True,
        "verbose": verbose,
        "vid_stride": vid_stride,
    }

    if device:
        predict_kwargs["device"] = device
    if imgsz:
        predict_kwargs["imgsz"] = imgsz
    if half:
        predict_kwargs["half"] = True
    if classes is not None:
        predict_kwargs["classes"] = classes

    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / "detect_params.json"
    output_path = run_dir if encoder == "yolo" else output_video
    params = {
        "created_at": timestamp,
        "video": video,
        "weights": weights,
        "output": str(output_path),
        "run_dir": str(run_dir),
        "yolo": {
            "conf": conf,
            "iou": iou,
            "device": device,
            "imgsz": imgsz,
            "half": half,
            "classes": classes,
            "vid_stride": vid_stride,
            "verbose": verbose,
        },
        "encoder": {
            "type": encoder,
            "cq": cq if encoder == "nvenc" else None,
        },
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "frames": total_frames,
        },
        "progress": progress,
    }
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("YOLO video detection")
    print(f"Video:      {video}")
    print(f"Weights:    {weights}")
    print(f"Output:     {output_path}")
    print(f"Params:     {params_path}")
    print(f"Device:     {device or 'YOLO default'}")
    print(f"imgsz:      {imgsz if imgsz else 'YOLO default'}")
    print(f"conf/iou:   {conf}/{iou}")
    print(f"half:       {half}")
    print(f"classes:    {classes if classes is not None else 'all'}")
    print(f"vid_stride: {vid_stride}")
    print(f"verbose:    {verbose}")
    print(f"encoder:    {encoder}")
    print(f"progress:   {progress}")
    print("=" * 80, flush=True)

    started_at = time.perf_counter()
    frame_count = 0
    last_result = None
    writer = None
    pbar = None
    try:
        if encoder == "nvenc":
            writer = FFmpegNVENCWriter(output_video, fps, width, height, cq)

        if progress:
            pbar = tqdm(total=total_frames or None, desc="Processing", unit="frame", ncols=100)

        for last_result in model.predict(**predict_kwargs):
            frame_count += 1
            if writer is not None:
                writer.write(last_result.plot())
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started_at
    if encoder == "nvenc":
        print(f"Output saved to: {output_video}")
    elif last_result is not None:
        print(f"Output saved to: {last_result.save_dir}")
    if elapsed > 0:
        print(f"Processed frames: {frame_count}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Speed: {frame_count / elapsed:.2f} FPS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on a video.")
    parser.add_argument("--video", default="data/videos/cv-xiaomi-1080.mp4", help="input video path")
    parser.add_argument("--weights", default="weights/yolo11m_safety.pt", help="YOLO weights path")
    parser.add_argument("--output", default="outputs/videos", help="output directory")
    parser.add_argument("--name", default=None, help="output run name; default: <video>_detect")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="device passed to YOLO, for example 0 or cpu")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO image size as one int",
    )
    parser.add_argument("--no-half", action="store_true", help="disable FP16")
    parser.add_argument("--classes", default=None, help="optional class ids, comma-separated, for example 0,1")
    parser.add_argument("--vid-stride", type=int, default=1, help="YOLO video frame stride")
    parser.add_argument("--verbose", action="store_true", help="enable YOLO per-frame logs")
    parser.add_argument("--encoder", choices=["nvenc", "yolo"], default="nvenc", help="video writer")
    parser.add_argument("--cq", type=int, default=28, help="NVENC quality, lower is better")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bar")
    args = parser.parse_args()

    run(
        video=args.video,
        weights=args.weights,
        output=args.output,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        half=not args.no_half,
        classes=parse_classes(args.classes),
        vid_stride=args.vid_stride,
        verbose=args.verbose,
        encoder=args.encoder,
        cq=args.cq,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
