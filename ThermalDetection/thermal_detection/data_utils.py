"""Shared dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from .array_backend import ArrayBackend

TARGET_NAMES = {0: "person", 1: "bike_motorcycle", 2: "car"}
SOURCE_TO_TARGET = {
    "person": 0,
    "bike": 1,
    "motor": 1,
    "car": 2,
}
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 512
DATASET_MARKER = ".thermal_yolo_dataset"


@dataclass
class SplitSummary:
    split: str
    images: int
    annotations_total: int
    target_boxes: int
    empty_target_images: int
    class_counts: dict[str, int]
    link_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "images": self.images,
            "annotations_total": self.annotations_total,
            "target_boxes": self.target_boxes,
            "empty_target_images": self.empty_target_images,
            "class_counts": self.class_counts,
            "image_materialization": self.link_counts,
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    """Load dataset YAML, with a dependency-free fallback for our own schema."""

    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a mapping in {path}")
        return loaded
    except ImportError:
        config: dict[str, Any] = {"names": {}}
        in_names = False
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "names:":
                in_names = True
                continue
            if ":" not in stripped:
                raise ValueError(f"Unsupported fallback YAML line in {path}: {raw_line}")
            key, value = (part.strip() for part in stripped.split(":", 1))
            if in_names and key.isdigit():
                config["names"][int(key)] = value
            else:
                in_names = False
                config[key] = value
        required = {"train", "val", "test", "names"}
        missing = required - set(config)
        if missing:
            raise ValueError(f"Missing dataset YAML keys in {path}: {sorted(missing)}")
        return config


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def locate_source_image(split_root: Path, file_name: str) -> Path:
    """Resolve an image only from the raw ``data`` directory.

    ``analyticsData`` is intentionally never searched.
    """

    relative = Path(file_name.replace("\\", "/"))
    candidates = [
        split_root / relative,
        split_root / "data" / relative.name,
    ]
    data_root = (split_root / "data").resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved.parent == data_root or data_root in resolved.parents:
            return resolved
    raise FileNotFoundError(
        f"COCO image {file_name!r} was not found under {split_root / 'data'}"
    )


def normalize_coco_xywh(
    boxes: Iterable[Iterable[float]],
    width: int,
    height: int,
    backend: ArrayBackend,
) -> np.ndarray:
    """Convert COCO ``x,y,w,h`` boxes to normalized YOLO ``cx,cy,w,h``."""

    box_list = list(boxes)
    if not box_list:
        return np.empty((0, 4), dtype=np.float32)

    xp = backend.xp
    values = backend.asarray(box_list, dtype=xp.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected an Nx4 bbox array, got shape {values.shape}")

    result = xp.empty_like(values)
    result[:, 0] = (values[:, 0] + values[:, 2] / 2.0) / float(width)
    result[:, 1] = (values[:, 1] + values[:, 3] / 2.0) / float(height)
    result[:, 2] = values[:, 2] / float(width)
    result[:, 3] = values[:, 3] / float(height)
    backend.synchronize()
    return backend.to_numpy(result)


def validate_normalized_boxes(
    boxes: np.ndarray,
    backend: ArrayBackend,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return validation errors for normalized YOLO boxes."""

    if boxes.size == 0:
        return []
    xp = backend.xp
    values = backend.asarray(boxes, dtype=xp.float32)
    errors: list[str] = []
    if not bool(backend.to_numpy(xp.all(xp.isfinite(values)))):
        errors.append("non-finite coordinate")
    if bool(backend.to_numpy(xp.any(values[:, 2:] <= 0.0))):
        errors.append("non-positive width or height")
    if bool(backend.to_numpy(xp.any(values < -tolerance))):
        errors.append("coordinate below zero")
    if bool(backend.to_numpy(xp.any(values > 1.0 + tolerance))):
        errors.append("coordinate above one")
    x1 = values[:, 0] - values[:, 2] / 2.0
    y1 = values[:, 1] - values[:, 3] / 2.0
    x2 = values[:, 0] + values[:, 2] / 2.0
    y2 = values[:, 1] + values[:, 3] / 2.0
    if bool(
        backend.to_numpy(
            xp.any(
                (x1 < -tolerance)
                | (y1 < -tolerance)
                | (x2 > 1.0 + tolerance)
                | (y2 > 1.0 + tolerance)
            )
        )
    ):
        errors.append("box extends beyond image bounds")
    return errors


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    """Create a hardlink/copy and return the mechanism that was used."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise

    shutil.copy2(source, destination)
    return "copy"


def prepare_clean_output(output: Path, force: bool) -> None:
    """Create an output directory without risking deletion of unrelated data."""

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        marker = output / DATASET_MARKER
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output}. "
                "Use --force only for a dataset previously created by this tool."
            )
        if not marker.is_file():
            raise RuntimeError(
                f"Refusing to remove {output}: marker {DATASET_MARKER} is missing."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / DATASET_MARKER, "thermal-yolo-dataset\n")


def dataset_yaml_text() -> str:
    lines = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for class_id, name in TARGET_NAMES.items())
    return "\n".join(lines) + "\n"
