"""Shared checks for YOLO26n training, evaluation and deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data_utils import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    TARGET_NAMES,
    load_dataset_yaml,
)


def parse_batch_size(value: str) -> int | float:
    """Parse an integer batch or Ultralytics automatic-memory fraction."""

    text = value.strip()
    if "." in text:
        result = float(text)
        if not 0.0 < result < 1.0:
            raise ValueError("A float batch value must be between 0 and 1.")
        return result
    result = int(text)
    if result == -1 or result > 0:
        return result
    raise ValueError("Batch must be -1, a positive integer, or a fraction such as 0.70.")


def validate_training_device(device: str) -> None:
    """Prevent an accidental multi-day CPU training run."""

    normalized = device.strip().lower()
    if normalized in {"cpu", "mps"}:
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build first."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Training device {device!r} requires CUDA, but "
            "torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch "
            "build or explicitly pass --device cpu for a short diagnostic only."
        )
    print(
        f"PyTorch CUDA: {torch.version.cuda}; "
        f"GPU 0: {torch.cuda.get_device_name(0)}"
    )


def load_dataset_config(data_path: Path) -> tuple[dict[str, Any], Path]:
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}. Run prepare_dataset.py first."
        )
    config = load_dataset_yaml(data_path)
    names = {int(key): str(value) for key, value in dict(config["names"]).items()}
    if names != TARGET_NAMES:
        raise ValueError(f"Expected exactly {TARGET_NAMES}, got {names}")
    root = Path(str(config.get("path", data_path.parent)))
    if not root.is_absolute():
        root = data_path.parent / root
    return config, root.resolve()


def verify_first_image_tensor_contract(data_path: Path) -> Path:
    """Check that OpenCV expands L-mode JPEG to three equal channels."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is missing. Install requirements.txt before training."
        ) from exc

    config, dataset_root = load_dataset_config(data_path)
    train_dir = dataset_root / Path(str(config["train"]))
    image_path = next(
        (
            path
            for path in sorted(train_dir.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ),
        None,
    )
    if image_path is None:
        raise FileNotFoundError(f"No training images found in {train_dir}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {image_path}")
    expected_shape = (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
    if image.shape != expected_shape:
        raise ValueError(f"Expected OpenCV shape {expected_shape}, got {image.shape}")
    if not (
        np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 1], image[:, :, 2])
    ):
        raise ValueError(
            f"Expected three identical grayscale channels after loading: {image_path}"
        )
    return image_path


def model_names(model: Any) -> dict[int, str]:
    raw_names = model.names
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    return {int(key): str(value) for key, value in dict(raw_names).items()}


def require_three_class_model(model: Any) -> None:
    names = model_names(model)
    if names != TARGET_NAMES:
        raise ValueError(
            "The checkpoint is not this project's three-class model. "
            f"Expected {TARGET_NAMES}, got {names}"
        )


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
