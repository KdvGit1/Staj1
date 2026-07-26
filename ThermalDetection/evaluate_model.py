"""Evaluate a trained three-class YOLO26n model on val or test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thermal_detection.data_utils import TARGET_NAMES
from thermal_detection.model_utils import (
    load_dataset_config,
    require_three_class_model,
    to_jsonable,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "yolo_thermal" / "dataset.yaml",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low confidence is appropriate for AP computation.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "evaluation",
    )
    parser.add_argument("--name", default="yolo26n_eval")
    parser.add_argument(
        "--traditional-nms",
        action="store_true",
        help="Use the one-to-many head with NMS instead of default end-to-end.",
    )
    parser.add_argument(
        "--compare-heads",
        action="store_true",
        help="Evaluate both YOLO26 detection heads in separate run directories.",
    )
    return parser.parse_args()


def per_class_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    payload: dict[str, Any] = {}
    for class_id, class_name in TARGET_NAMES.items():
        row: dict[str, Any] = {}
        for attribute in ("p", "r", "f1", "ap50", "ap"):
            values = getattr(box, attribute, None)
            if values is None:
                continue
            try:
                row[attribute] = float(values[class_id])
            except (IndexError, TypeError, ValueError):
                row[attribute] = to_jsonable(values)
        maps = getattr(box, "maps", None)
        if maps is not None:
            try:
                row["map50_95"] = float(maps[class_id])
            except (IndexError, TypeError, ValueError):
                pass
        payload[class_name] = row
    return payload


def run_evaluation(model: Any, args: argparse.Namespace, end2end: bool) -> Path:
    mode_name = "end2end" if end2end else "traditional_nms"
    run_name = f"{args.name}_{args.split}_{mode_name}"
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=640,
        rect=True,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        end2end=end2end,
        plots=True,
        save_json=True,
        project=str(args.project.resolve()),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    save_dir = Path(metrics.save_dir)
    summary = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "input_tensor": ["batch", 3, 512, 640],
        "head": mode_name,
        "conf": args.conf,
        "iou": args.iou,
        "overall": getattr(metrics, "results_dict", {}),
        "per_class": per_class_metrics(metrics),
        "speed_ms_per_image": getattr(metrics, "speed", {}),
        "save_dir": str(save_dir),
    }
    summary_path = save_dir / "thermal_metrics_summary.json"
    write_json(summary_path, summary)
    print(f"Evaluation summary: {summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    load_dataset_config(args.data)
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    from ultralytics import YOLO

    def load_model() -> Any:
        """Load a fresh model because validation may fuse away the other head."""
        model = YOLO(str(args.model.resolve()))
        require_three_class_model(model)
        return model

    if args.compare_heads:
        summary_paths = [
            run_evaluation(load_model(), args, end2end=True),
            run_evaluation(load_model(), args, end2end=False),
        ]
    else:
        summary_paths = [
            run_evaluation(
                load_model(),
                args,
                end2end=not args.traditional_nms,
            )
        ]
    from thermal_detection.reporting import generate_evaluation_report

    generated = generate_evaluation_report(
        summary_paths,
        args.project.resolve() / f"{args.name}_{args.split}_graph-report",
    )
    print(f"Evaluation graph report: {generated['html']}")


if __name__ == "__main__":
    main()
