"""Verify a converted thermal YOLO dataset and render sample labels."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from thermal_detection.array_backend import select_array_backend
from thermal_detection.data_utils import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    TARGET_NAMES,
    atomic_write_text,
    load_dataset_yaml,
    load_json,
    validate_normalized_boxes,
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLORS = {
    0: (255, 80, 80),
    1: (80, 210, 255),
    2: (120, 255, 100),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "yolo_thermal" / "dataset.yaml",
        help="Generated dataset.yaml file.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "numpy"),
        default="auto",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=12,
        help="Number of random labeled images to render across all splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-image-header-check",
        action="store_true",
        help="Skip repeated image size/mode checks; labels are still fully verified.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_verification",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Do not generate the SVG/HTML dataset graph report.",
    )
    return parser.parse_args()


def resolve_dataset_root(data_path: Path, config: dict[str, Any]) -> Path:
    raw_path = Path(str(config.get("path", data_path.parent)))
    if not raw_path.is_absolute():
        raw_path = data_path.parent / raw_path
    return raw_path.resolve()


def read_label_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}:{line_number} must contain 5 fields, got {len(fields)}"
            )
        try:
            class_value = float(fields[0])
            class_id = int(class_value)
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric label at {path}:{line_number}") from exc
        if class_value != class_id:
            raise ValueError(f"Non-integer class ID at {path}:{line_number}")
        classes.append(class_id)
        boxes.append(values)
    return (
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    )


def render_sample(
    image_path: Path,
    classes: np.ndarray,
    boxes: np.ndarray,
    destination: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for class_id, (cx, cy, box_width, box_height) in zip(
        classes.tolist(),
        boxes.tolist(),
        strict=True,
    ):
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        color = COLORS[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((max(0, x1 + 2), max(0, y1 + 2)), TARGET_NAMES[class_id], fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def verify_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    backend: Any,
    check_image_headers: bool,
) -> tuple[dict[str, Any], list[tuple[Path, np.ndarray, np.ndarray]]]:
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted(labels_dir.glob("*.txt"))
    label_by_stem = {path.stem.casefold(): path for path in labels}
    image_stems = {path.stem.casefold() for path in images}
    orphan_labels = [
        str(path) for path in labels if path.stem.casefold() not in image_stems
    ]
    missing_labels: list[str] = []
    class_counts: Counter[str] = Counter()
    empty_images = 0
    visual_candidates: list[tuple[Path, np.ndarray, np.ndarray]] = []
    box_chunks: list[np.ndarray] = []

    for index, image_path in enumerate(images, start=1):
        label_path = label_by_stem.get(image_path.stem.casefold())
        if label_path is None:
            missing_labels.append(str(image_path))
            continue
        if check_image_headers:
            with Image.open(image_path) as image:
                if image.size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
                    raise ValueError(f"Unexpected size {image.size}: {image_path}")
                if image.mode != "L":
                    raise ValueError(f"Expected L mode, got {image.mode}: {image_path}")
        classes, boxes = read_label_file(label_path)
        if np.any((classes < 0) | (classes >= len(TARGET_NAMES))):
            raise ValueError(f"Class outside 0..2 in {label_path}")
        if len(boxes):
            box_chunks.append(boxes)
        if len(classes) == 0:
            empty_images += 1
        else:
            visual_candidates.append((image_path, classes, boxes))
        for class_id in classes.tolist():
            class_counts[TARGET_NAMES[class_id]] += 1
        if index % 1000 == 0 or index == len(images):
            print(f"[{split}] verified {index}/{len(images)} images")

    if missing_labels or orphan_labels:
        raise ValueError(
            f"{split} pairing failure: {len(missing_labels)} missing labels, "
            f"{len(orphan_labels)} orphan labels"
        )
    all_boxes = (
        np.concatenate(box_chunks, axis=0)
        if box_chunks
        else np.empty((0, 4), dtype=np.float32)
    )
    errors = validate_normalized_boxes(all_boxes, backend)
    if errors:
        raise ValueError(f"{split} contains invalid boxes: {', '.join(errors)}")
    return (
        {
            "images": len(images),
            "labels": len(labels),
            "empty_images": empty_images,
            "class_counts": {
                name: int(class_counts.get(name, 0)) for name in TARGET_NAMES.values()
            },
        },
        visual_candidates,
    )


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    config = load_dataset_yaml(data_path)
    dataset_root = resolve_dataset_root(data_path, config)
    backend = select_array_backend(args.backend)
    print(f"Array backend: {backend.name} ({backend.detail})")

    configured_names = {
        int(key): str(value) for key, value in dict(config.get("names", {})).items()
    }
    if configured_names != TARGET_NAMES:
        raise ValueError(
            f"dataset.yaml names must be exactly {TARGET_NAMES}, got {configured_names}"
        )

    report: dict[str, Any] = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "array_backend": {"name": backend.name, "detail": backend.detail},
        "expected_tensor": ["batch", 3, EXPECTED_HEIGHT, EXPECTED_WIDTH],
        "splits": {},
    }
    all_candidates: list[tuple[str, Path, np.ndarray, np.ndarray]] = []
    for split in ("train", "val", "test"):
        relative_images = Path(str(config[split]))
        images_dir = dataset_root / relative_images
        labels_dir = dataset_root / "labels" / split
        split_report, candidates = verify_split(
            split,
            images_dir,
            labels_dir,
            backend,
            check_image_headers=not args.skip_image_header_check,
        )
        report["splits"][split] = split_report
        all_candidates.extend(
            (split, image_path, classes, boxes)
            for image_path, classes, boxes in candidates
        )

    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        for split in ("train", "val", "test"):
            expected = manifest["splits"][split]
            actual = report["splits"][split]
            comparisons = {
                "images": (expected["images"], actual["images"]),
                "empty_images": (
                    expected["empty_target_images"],
                    actual["empty_images"],
                ),
                "class_counts": (
                    expected["class_counts"],
                    actual["class_counts"],
                ),
            }
            mismatches = {
                key: {"manifest": left, "verified": right}
                for key, (left, right) in comparisons.items()
                if left != right
            }
            if mismatches:
                raise ValueError(f"{split} differs from manifest: {mismatches}")
        report["manifest_match"] = True
    else:
        report["manifest_match"] = None

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sample_count = min(max(args.visualize, 0), len(all_candidates))
    for sample_index, (split, image_path, classes, boxes) in enumerate(
        rng.sample(all_candidates, sample_count),
        start=1,
    ):
        destination = (
            report_dir
            / "samples"
            / f"{sample_index:03d}_{split}_{image_path.stem}.jpg"
        )
        render_sample(image_path, classes, boxes, destination)

    report["visualization_samples"] = sample_count
    report_path = report_dir / "verification_report.json"
    atomic_write_text(
        report_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Dataset verification passed. Report: {report_path}")
    if not args.no_graphs and manifest_path.is_file():
        from thermal_detection.reporting import generate_dataset_report

        generated = generate_dataset_report(
            manifest_path,
            report_path,
            PROJECT_ROOT / "reports" / "graphs" / "dataset",
        )
        print(f"Dataset graph report: {generated['html']}")


if __name__ == "__main__":
    main()
