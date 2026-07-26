"""Run fixed-shape 640x512 inference with the trained thermal model."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermal_detection.data_utils import TARGET_NAMES
from thermal_detection.model_utils import require_three_class_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source",
        required=True,
        help="Image, directory, video, stream URL or camera index.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "predict",
    )
    parser.add_argument("--name", default="thermal_640x512")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--traditional-nms",
        action="store_true",
        help="Use YOLO26 one-to-many predictions with NMS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    require_three_class_model(model)
    source: str | int = int(args.source) if args.source.isdecimal() else args.source
    results = model.predict(
        source=source,
        imgsz=(512, 640),
        rect=False,
        classes=[0, 1, 2],
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        end2end=not args.traditional_nms,
        stream=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        verbose=True,
    )
    image_count = 0
    detection_count = 0
    class_counts = {name: 0 for name in TARGET_NAMES.values()}
    save_dir: Path | None = None
    for result in results:
        image_count += 1
        detection_count += len(result.boxes)
        for class_id in result.boxes.cls.detach().cpu().tolist():
            class_counts[TARGET_NAMES[int(class_id)]] += 1
        save_dir = Path(result.save_dir)
    print(f"Processed inputs: {image_count}")
    print(f"Total detections: {detection_count}")
    if save_dir is not None:
        print(f"Saved predictions: {save_dir}")
        from thermal_detection.reporting import generate_prediction_report

        generated = generate_prediction_report(
            class_counts,
            image_count,
            save_dir / "graph-report",
        )
        print(f"Prediction graph report: {generated['html']}")


if __name__ == "__main__":
    main()
