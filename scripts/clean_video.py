"""Clean broken video streams before YOLO detection.

This script uses FFmpeg to decode the source video, drop corrupt packets/frames,
rebuild timestamps, remove audio, and write a normal H.264 MP4. It is intended
for camera files that print errors such as:

    non-existing PPS 0 referenced
    non monotonically increasing dts

Examples:
    .venv/bin/python scripts/clean_video.py --source data/videos/cv-hk-camera-拼色.mp4
    .venv/bin/python scripts/clean_video.py --source data/videos
"""

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".ts"}


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = run_cmd(cmd)
    return json.loads(result.stdout)


def parse_fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def video_info(path: Path) -> dict:
    data = ffprobe(path)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found: {path}")

    fps = parse_fps(video_stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = parse_fps(video_stream.get("r_frame_rate"))
    if fps <= 0:
        fps = 25.0

    return {
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration") or 0),
        "format": data.get("format", {}).get("format_name"),
        "video_codec": video_stream.get("codec_name"),
    }


def output_path_for(source: Path, output: Path | None, source_is_dir: bool) -> Path:
    if output is None:
        return source.parent / f"{source.stem}_cleaned.mp4"
    if source_is_dir:
        return output / f"{source.stem}_cleaned.mp4"
    if output.suffix:
        return output
    return output / f"{source.stem}_cleaned.mp4"


def build_ffmpeg_cmd(source: Path, target: Path, fps: float, encoder: str, cq: int, crf: int) -> list[str]:
    if encoder == "nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(cq)]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]

    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-fflags",
        "+genpts+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={fps:.6f},setpts=N/FRAME_RATE/TB",
        "-vsync",
        "cfr",
        *codec_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]


def clean_one(source: Path, output: Path | None, source_is_dir: bool, encoder: str, cq: int, crf: int) -> dict:
    info = video_info(source)
    target = output_path_for(source, output, source_is_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Source:  {source}")
    print(f"Output:  {target}")
    print(f"Input:   {info['width']}x{info['height']} {info['fps']:.3f} FPS, codec={info['video_codec']}, format={info['format']}")
    print(f"Encoder: {encoder}")
    print("=" * 80, flush=True)

    cmd = build_ffmpeg_cmd(source, target, info["fps"], encoder, cq, crf)
    result = run_cmd(cmd, check=False)

    if result.returncode != 0 and encoder == "nvenc":
        print("NVENC failed, retrying with libx264...")
        encoder = "libx264"
        cmd = build_ffmpeg_cmd(source, target, info["fps"], encoder, cq, crf)
        result = run_cmd(cmd, check=False)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

    cleaned_info = video_info(target)
    report = {
        "source": str(source),
        "output": str(target),
        "encoder": encoder,
        "input": info,
        "cleaned": cleaned_info,
        "ffmpeg_command": cmd,
    }
    report_path = target.with_suffix(".clean_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cleaned video saved to: {target}")
    print(f"Report saved to:        {report_path}")
    print(f"Cleaned: {cleaned_info['width']}x{cleaned_info['height']} {cleaned_info['fps']:.3f} FPS")
    return report


def find_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    raise FileNotFoundError(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean corrupt video frames/packets before YOLO detection.")
    parser.add_argument("--source", required=True, help="video file or directory")
    parser.add_argument(
        "--output",
        default=None,
        help="output file or directory; default: each video is saved as <filename>_cleaned.mp4",
    )
    parser.add_argument("--encoder", choices=["nvenc", "libx264"], default="nvenc", help="video encoder")
    parser.add_argument("--cq", type=int, default=28, help="NVENC quality, lower is better")
    parser.add_argument("--crf", type=int, default=23, help="libx264 quality, lower is better")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output) if args.output else None
    source_is_dir = source.is_dir()
    files = find_videos(source)
    if not files:
        raise RuntimeError(f"No videos found: {source}")

    for file in files:
        clean_one(file, output, source_is_dir, args.encoder, args.cq, args.crf)


if __name__ == "__main__":
    main()
