"""Convert the supplied thermal COCO dataset to a three-class YOLO dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from PIL import Image

from thermal_detection.array_backend import select_array_backend
from thermal_detection.data_utils import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    SOURCE_TO_TARGET,
    TARGET_NAMES,
    SplitSummary,
    atomic_write_text,
    dataset_yaml_text,
    load_json,
    locate_source_image,
    materialize_image,
    normalize_coco_xywh,
    prepare_clean_output,
    validate_normalized_boxes,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = {
    "train": PROJECT_ROOT / "images_thermal_train",
    "val": PROJECT_ROOT / "images_thermal_val",
    "test": PROJECT_ROOT / "video_thermal_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert thermal COCO annotations into YOLO labels with exactly "
            "person, bike_motorcycle and car classes."
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_SPLITS["train"],
        help="COCO training split directory.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=DEFAULT_SPLITS["val"],
        help="COCO validation split directory.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_SPLITS["test"],
        help="COCO test split directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "yolo_thermal",
        help="Destination for the converted YOLO dataset.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="Use hardlinks to avoid duplication, with copy fallback in auto mode.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "numpy"),
        default="auto",
        help="Array backend for vectorized bbox conversion.",
    )
    parser.add_argument(
        "--skip-image-header-check",
        action="store_true",
        help="Skip opening every image to verify 640x512 grayscale headers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and calculate labels without writing output files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing output only when it contains this tool's "
            "safety marker."
        ),
    )
    return parser.parse_args()


def validate_coco_categories(coco: dict[str, Any], source: Path) -> dict[int, int]:
    category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    available = set(category_names.values())
    missing = set(SOURCE_TO_TARGET) - available
    if missing:
        raise ValueError(f"{source} is missing required categories: {sorted(missing)}")
    return {
        source_id: SOURCE_TO_TARGET[source_name]
        for source_id, source_name in category_names.items()
        if source_name in SOURCE_TO_TARGET
    }


def audit_image(
    image_path: Path,
    expected_width: int,
    expected_height: int,
) -> None:
    with Image.open(image_path) as image:
        if image.size != (expected_width, expected_height):
            raise ValueError(
                f"Unexpected image size for {image_path}: {image.size}; "
                f"expected {(expected_width, expected_height)}"
            )
        if image.mode != "L":
            raise ValueError(
                f"Expected a true grayscale L image, got mode {image.mode!r}: "
                f"{image_path}"
            )


def convert_split(
    split: str,
    split_root: Path,
    output_root: Path,
    image_mode: str,
    backend: Any,
    check_headers: bool,
    dry_run: bool,
) -> SplitSummary:
    coco_path = split_root / "coco.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing COCO file: {coco_path}")
    coco = load_json(coco_path)
    category_mapping = validate_coco_categories(coco, coco_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_by_id = {int(item["id"]): item for item in images}
    if len(image_by_id) != len(images):
        raise ValueError(f"Duplicate image IDs found in {coco_path}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotations_total = 0
    for annotation in annotations:
        annotations_total += 1
        image_id = int(annotation["image_id"])
        if image_id not in image_by_id:
            raise ValueError(
                f"Annotation {annotation.get('id')} references unknown image {image_id}"
            )
        source_category = int(annotation["category_id"])
        if source_category in category_mapping:
            annotations_by_image[image_id].append(annotation)

    # Convert each split in one vectorized operation. This avoids thousands of
    # tiny host/device transfers when the selected backend is CuPy.
    flat_annotations = [
        annotation
        for image_info in images
        for annotation in annotations_by_image.get(int(image_info["id"]), [])
    ]
    flat_boxes = normalize_coco_xywh(
        (annotation["bbox"] for annotation in flat_annotations),
        EXPECTED_WIDTH,
        EXPECTED_HEIGHT,
        backend,
    )
    errors = validate_normalized_boxes(flat_boxes, backend)
    if errors:
        raise ValueError(f"Invalid target boxes in {split}: {', '.join(errors)}")
    converted_by_image: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for annotation, box in zip(flat_annotations, flat_boxes, strict=True):
        converted_by_image[int(annotation["image_id"])].append(
            (category_mapping[int(annotation["category_id"])], box)
        )

    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    seen_stems: set[str] = set()
    class_counts: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()
    target_box_count = 0
    empty_images = 0

    for index, image_info in enumerate(images, start=1):
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        source_image = locate_source_image(split_root, file_name)
        destination_name = source_image.name
        key = source_image.stem.casefold()
        if key in seen_stems:
            raise ValueError(
                f"Duplicate destination image stem in {split}: {source_image.stem}"
            )
        seen_stems.add(key)

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError(
                f"COCO dimensions for {source_image} are {(width, height)}, "
                f"expected {(EXPECTED_WIDTH, EXPECTED_HEIGHT)}"
            )
        if check_headers:
            audit_image(source_image, width, height)

        converted = converted_by_image.get(image_id, [])
        classes = [class_id for class_id, _ in converted]
        boxes = [box for _, box in converted]

        if not converted:
            empty_images += 1
        for class_id in classes:
            class_counts[TARGET_NAMES[class_id]] += 1
        target_box_count += len(converted)

        if not dry_run:
            destination_image = output_images / destination_name
            mechanism = materialize_image(source_image, destination_image, image_mode)
            link_counts[mechanism] += 1

            label_lines = [
                f"{class_id} "
                + " ".join(f"{float(value):.8f}" for value in box)
                for class_id, box in zip(classes, boxes, strict=True)
            ]
            label_text = "\n".join(label_lines)
            if label_text:
                label_text += "\n"
            atomic_write_text(output_labels / f"{source_image.stem}.txt", label_text)

        if index % 1000 == 0 or index == len(images):
            print(f"[{split}] audited {index}/{len(images)} images")

    return SplitSummary(
        split=split,
        images=len(images),
        annotations_total=annotations_total,
        target_boxes=target_box_count,
        empty_target_images=empty_images,
        class_counts={
            name: int(class_counts.get(name, 0)) for name in TARGET_NAMES.values()
        },
        link_counts=dict(link_counts),
    )


def main() -> None:
    args = parse_args()
    backend = select_array_backend(args.backend)
    print(f"Array backend: {backend.name} ({backend.detail})")

    split_roots = {
        "train": args.train_root.resolve(),
        "val": args.val_root.resolve(),
        "test": args.test_root.resolve(),
    }
    output = args.output.resolve()
    if not args.dry_run:
        prepare_clean_output(output, args.force)

    summaries: list[SplitSummary] = []
    for split, split_root in split_roots.items():
        summaries.append(
            convert_split(
                split=split,
                split_root=split_root,
                output_root=output,
                image_mode=args.image_mode,
                backend=backend,
                check_headers=not args.skip_image_header_check,
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "array_backend": {"name": backend.name, "detail": backend.detail},
        "image_size": {"width": EXPECTED_WIDTH, "height": EXPECTED_HEIGHT},
        "grayscale_policy": (
            "Source JPEG files remain mode L. Ultralytics/PyTorch loads them as "
            "three equal channels for the pretrained YOLO26n backbone."
        ),
        "source_to_target": SOURCE_TO_TARGET,
        "target_names": TARGET_NAMES,
        "ignored_source_categories_are_background": True,
        "splits": {item.split: item.as_dict() for item in summaries},
    }

    if not args.dry_run:
        atomic_write_text(output / "dataset.yaml", dataset_yaml_text())
        atomic_write_text(
            output / "manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry run complete; no dataset files were written.")
    else:
        print(f"YOLO dataset created at: {output}")
        print(f"Dataset config: {output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
