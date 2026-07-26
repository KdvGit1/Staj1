from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from thermal_detection.array_backend import select_array_backend
from thermal_detection.data_utils import (
    load_dataset_yaml,
    normalize_coco_xywh,
    validate_normalized_boxes,
)


class DataUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = select_array_backend("numpy")

    def test_full_image_box(self) -> None:
        result = normalize_coco_xywh(
            [[0.0, 0.0, 640.0, 512.0]],
            640,
            512,
            self.backend,
        )
        np.testing.assert_allclose(result, [[0.5, 0.5, 1.0, 1.0]])
        self.assertEqual(validate_normalized_boxes(result, self.backend), [])

    def test_known_box(self) -> None:
        result = normalize_coco_xywh(
            [[64.0, 51.2, 128.0, 102.4]],
            640,
            512,
            self.backend,
        )
        np.testing.assert_allclose(result, [[0.2, 0.2, 0.2, 0.2]], atol=1e-6)

    def test_out_of_bounds_box_is_rejected(self) -> None:
        boxes = np.asarray([[0.95, 0.5, 0.2, 0.2]], dtype=np.float32)
        errors = validate_normalized_boxes(boxes, self.backend)
        self.assertIn("box extends beyond image bounds", errors)

    def test_auto_backend_always_returns_a_backend(self) -> None:
        backend = select_array_backend("auto")
        self.assertIn(backend.name, {"numpy", "cupy"})

    def test_project_dataset_yaml_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.yaml"
            path.write_text(
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n"
                "  0: person\n"
                "  1: bike_motorcycle\n"
                "  2: car\n",
                encoding="utf-8",
            )
            config = load_dataset_yaml(path)
            self.assertEqual(config["names"][1], "bike_motorcycle")
            self.assertEqual(config["train"], "images/train")


if __name__ == "__main__":
    unittest.main()
