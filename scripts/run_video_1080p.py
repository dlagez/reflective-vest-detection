"""1080p 视频检测脚本 — 原始分辨率输入，带进度条."""

import argparse
import json
import sys
import time
from pathlib import Path

# 把项目根目录加入 sys.path，使 `from src...` 可导入
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.utils.box_utils import align_to_stride


# ● 脚本已创建。下面是运行命令和参数说明：

#   ---
#   默认运行（直接跑 1080p 视频）：

#   .venv/bin/python scripts/run_video_1080p.py --video data/videos/cv-hk-camera-拼色.mp4

#   完整参数说明：

#   ┌───────────┬────────────────────────────────┬────────────────┐
#   │   参数    │             默认值             │      说明      │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --video   │ data/videos/cv-xiaomi-1080.mp4 │ 视频路径       │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --weights │ weights/yolo11m_safety.pt      │ YOLO 权重      │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --output  │ outputs/videos                 │ 输出目录       │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --conf    │ 0.5                            │ 检测置信度阈值 │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --iou     │ 0.45                           │ NMS IoU 阈值   │
#   ├───────────┼────────────────────────────────┼────────────────┤
#   │ --device  │ 0                              │ 0=GPU, cpu=CPU │
#   └───────────┴────────────────────────────────┴────────────────┘

#   关键行为：

#   - 模型输入尺寸自动对齐 stride=32（1080→1088），bbox 映射回原始帧
#   - 带 tqdm 进度条，显示帧进度
#   - 输出带标注的视频：outputs/videos/cv-xiaomi-1080_detect.mp4
#   - 输出 JSON 结果：outputs/videos/cv-xiaomi-1080_result.json（逐帧检测详情 + 统计）
#   - 绿色框 = 穿了反光衣，红色框 = 未穿反光衣，橙色框 = 反光衣本身

#   运行前确保 weights/yolo11m_safety.pt 已经放入对应目录。

def run_detection(video_path: str, weights: str, output_dir: str, conf: float, iou: float, device):
    """
    Run vest detection on video at original resolution.

    Key: imgsz is aligned to stride=32 so YOLO does not need to
    resize the frames. Output bbox coords are mapped back to
    the original frame resolution.
    """
    # ── Resolve device label for logging ────────────────────────────
    if isinstance(device, int) and device >= 0:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        device_label = f"GPU — {gpu_name} ({gpu_mem:.1f} GB)"
    else:
        device_label = "CPU"

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"[1/5] Loading model: {weights}  [{device_label}]")
    model = YOLO(weights)
    model.fuse()  # fuse Conv + BN for faster inference

    # ── 2. Open video, get native resolution ───────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[2/5] Video info:")
    print(f"       Resolution: {vid_width}x{vid_height}")
    print(f"       FPS: {fps:.2f}")
    print(f"       Total frames: {total_frames}")
    print(f"       Duration: {total_frames / fps:.1f}s")

    # Compute YOLO input size aligned to stride=32.
    # This only affects the inference tensor — output bbox coordinates are
    # automatically mapped back to the original frame resolution by Ultralytics.
    imgsz = align_to_stride(vid_height, vid_width)
    print(f"[3/5] Model input size: {imgsz[0]}x{imgsz[1]} "
          f"(stride-aligned; video stays {vid_width}x{vid_height})")

    # ── 3. Setup output ────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    viz_path = str(out_dir / f"{video_name}_detect.mp4")
    json_path = str(out_dir / f"{video_name}_result.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(viz_path, fourcc, fps, (vid_width, vid_height))

    # ── 4. Frame-by-frame inference ────────────────────────────────
    print(f"[4/5] Running inference...")
    print()

    # Classes: person=0, vest=1
    target_classes = [0, 1]

    all_frame_results = []
    total_persons = 0
    total_vests = 0
    violations_count = 0

    class_names = {0: "person", 1: "vest", 2: "helmet"}

    with tqdm(total=total_frames, desc="Processing", unit="frame", ncols=100) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference at native resolution
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=target_classes,
                verbose=False,
                device=device,
            )

            frame_detections = []
            persons_in_frame = []
            vests_in_frame = []

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf_val = float(result.boxes.conf[i].cpu())

                    det = {
                        "class_id": cls_id,
                        "class_name": class_names.get(cls_id, "unknown"),
                        "confidence": round(conf_val, 4),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    frame_detections.append(det)

                    if cls_id == 0:
                        persons_in_frame.append(det)
                    elif cls_id == 1:
                        vests_in_frame.append(det)

            # Simple vest-person association via IoU
            for person in persons_in_frame:
                px1, py1, px2, py2 = person["bbox"]
                p_area = (px2 - px1) * (py2 - py1)
                wearing = False
                for vest in vests_in_frame:
                    vx1, vy1, vx2, vy2 = vest["bbox"]
                    inter_x1 = max(px1, vx1)
                    inter_y1 = max(py1, vy1)
                    inter_x2 = min(px2, vx2)
                    inter_y2 = min(py2, vy2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    vest_area = (vx2 - vx1) * (vy2 - vy1)
                    overlap = inter_area / vest_area if vest_area > 0 else 0
                    if overlap >= 0.5:
                        wearing = True
                        break

                person["wearing_vest"] = wearing
                if not wearing:
                    violations_count += 1

            # ── Draw results on frame ──────────────────────────────
            for det in frame_detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class_name']} {det['confidence']:.2f}"

                if det["class_name"] == "person":
                    wearing = det.get("wearing_vest", False)
                    color = (0, 255, 0) if wearing else (0, 0, 255)
                    label = f"{'VEST OK' if wearing else 'NO VEST!'} {det['confidence']:.2f}"
                    thickness = 3
                elif det["class_name"] == "vest":
                    color = (0, 165, 255)
                    thickness = 2
                else:
                    color = (128, 128, 128)
                    thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # FPS counter on frame
            cv2.putText(frame, f"Frame {frame_idx + 1}/{total_frames}",
                        (10, vid_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out_writer.write(frame)

            # Stats
            total_persons += len(persons_in_frame)
            total_vests += len(vests_in_frame)

            all_frame_results.append({
                "frame": frame_idx,
                "detections": frame_detections,
                "person_count": len(persons_in_frame),
                "vest_count": len(vests_in_frame),
            })

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out_writer.release()

    # ── 5. Save results ────────────────────────────────────────────
    print()
    print(f"[5/5] Saving results...")

    stats = {
        "video": video_path,
        "resolution": f"{vid_width}x{vid_height}",
        "model_input": f"{imgsz[1]}x{imgsz[0]}",
        "device": device_label,
        "total_frames_processed": frame_idx,
        "total_persons_detected": total_persons,
        "total_vests_detected": total_vests,
        "total_violations": violations_count,
        "compliance_rate": round(
            (total_persons - violations_count) / total_persons if total_persons > 0 else 1.0, 4
        ),
    }

    report = {
        "stats": stats,
        "frame_results": all_frame_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Video:           {video_path}")
    print(f"  Device:          {device_label}")
    print(f"  Resolution:      {vid_width}x{vid_height}")
    print(f"  Model input:     {imgsz[1]}x{imgsz[0]} (stride-aligned)")
    print(f"  Frames:          {frame_idx}")
    print(f"  Persons detected: {total_persons}")
    print(f"  Vests detected:   {total_vests}")
    print(f"  Violations:       {violations_count}")
    print(f"  Compliance rate:  {stats['compliance_rate']:.2%}")
    print(f"  Output video:     {viz_path}")
    print(f"  Output JSON:      {json_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="1080p 反光衣视频检测 — 原始分辨率输入")
    parser.add_argument("--video", type=str, default="data/videos/cv-xiaomi-1080.mp4",
                        help="视频路径 (default: data/videos/cv-xiaomi-1080.mp4)")
    parser.add_argument("--weights", type=str, default="weights/yolo11m_safety.pt",
                        help="模型权重路径")
    parser.add_argument("--output", type=str, default="outputs/videos",
                        help="输出目录")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值 (default: 0.45)")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备: auto(默认优先GPU), 0=cuda:0, cpu (default: auto)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  反光衣检测 — 1080p 原始分辨率推理")
    print("=" * 60)
    print()
    print("  运行参数:")
    print(f"    --video    {args.video}")
    print(f"    --weights  {args.weights}")
    print(f"    --output   {args.output}")
    print(f"    --conf     {args.conf}")
    print(f"    --iou      {args.iou}")
    print(f"    --device   {args.device}")
    print()

    # ── Resolve device: GPU first (auto), or explicit ──────────────
    device_requested = args.device.lower()
    if device_requested == "cpu":
        device = "cpu"
        device_label = "CPU"
    elif device_requested == "auto":
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_label = f"GPU — {gpu_name} ({gpu_mem:.1f} GB)"
        else:
            device = "cpu"
            device_label = "CPU (GPU not available, fallback)"
    else:
        # Explicit device index like "0", "1"
        device = int(device_requested) if device_requested.isdigit() else device_requested
        if isinstance(device, int) and device >= 0:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(device)
                gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
                device_label = f"GPU — {gpu_name} ({gpu_mem:.1f} GB)"
            else:
                device = "cpu"
                device_label = "CPU (GPU not available, fallback)"
        else:
            device_label = "CPU"

    print(f"  推理设备: {device_label}")
    print()

    run_detection(
        video_path=args.video,
        weights=args.weights,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )


if __name__ == "__main__":
    main()
