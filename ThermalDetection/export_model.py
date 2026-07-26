"""Export the trained model with a fixed 1x3x512x640 input contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermal_detection.model_utils import require_three_class_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("onnx", "engine", "openvino", "torchscript"),
        default="onnx",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Allow dynamic shapes; omit for the requested fixed input.",
    )
    parser.add_argument(
        "--traditional-nms",
        action="store_true",
        help="Export one-to-many output with NMS-compatible behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    require_three_class_model(model)
    exported = model.export(
        format=args.format,
        imgsz=(512, 640),
        batch=args.batch,
        dynamic=args.dynamic,
        half=args.half,
        simplify=True,
        opset=args.opset,
        device=args.device,
        end2end=not args.traditional_nms,
    )
    print(f"Export complete: {exported}")
    if not args.dynamic:
        print(f"Fixed input contract: [{args.batch}, 3, 512, 640]")


if __name__ == "__main__":
    main()

