"""反光衣检测主入口."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Reflective Vest Detection")
    parser.add_argument(
        "--task",
        type=str,
        choices=["image", "video", "camera", "api"],
        default="image",
        help="Detection task to run",
    )
    parser.add_argument("--source", type=str, default=None, help="Input source path")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "image":
        from scripts.run_image import run as run_image
        run_image(source=args.source, config=args.config)
    elif args.task == "video":
        from scripts.run_video import run as run_video
        run_video(source=args.source, config=args.config)
    elif args.task == "camera":
        from scripts.run_camera import run as run_camera
        run_camera(config=args.config)
    elif args.task == "api":
        import uvicorn
        uvicorn.run("src.api.app:create_app()", factory=True, reload=True)
    else:
        print(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
