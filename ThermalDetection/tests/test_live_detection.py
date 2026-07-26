from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from thermal_live_detection.app import _letterbox_for_detector
from thermal_live_detection.butiv_demo import _fit_native_preserve_aspect
from thermal_live_detection.butiv_random_report import (
    _iou,
    _select_frames,
)
from thermal_live_detection.butiv import (
    load_butiv_annotations,
    transform_to_detector_input,
)
from thermal_live_detection.collector import CaptureSession
from thermal_live_detection.stream import build_hikvision_rtsp_url


class LiveDetectionTests(unittest.TestCase):
    def test_rtsp_credentials_are_url_encoded(self) -> None:
        url = build_hikvision_rtsp_url(
            ip="192.0.2.10",
            username="thermal user",
            password="p@ss:/?#",
            channel="202",
        )
        self.assertEqual(
            url,
            (
                "rtsp://thermal%20user:p%40ss%3A%2F%3F%23@"
                "192.0.2.10:554/Streaming/Channels/202"
            ),
        )

    def test_pseudo_label_is_standard_five_field_yolo(self) -> None:
        detections = [
            {
                "class_id": 2,
                "confidence": 0.876,
                "xyxy": [64.0, 48.0, 192.0, 144.0],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "frame.txt"
            CaptureSession._write_yolo_labels(
                label_path,
                detections=detections,
                width=640,
                height=480,
            )
            fields = label_path.read_text(encoding="utf-8").split()

        self.assertEqual(len(fields), 5)
        self.assertEqual(fields[0], "2")
        self.assertEqual(
            [float(value) for value in fields[1:]],
            [0.2, 0.2, 0.2, 0.2],
        )

    def test_edsr_shape_is_letterboxed_without_stretching(self) -> None:
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        letterboxed = _letterbox_for_detector(image)

        self.assertEqual(letterboxed.shape, (512, 640, 3))
        self.assertTrue(np.all(letterboxed[:16] == 114))
        self.assertTrue(np.all(letterboxed[16:496] == 200))
        self.assertTrue(np.all(letterboxed[496:] == 114))

    def test_butiv_labels_are_mapped_and_transformed(self) -> None:
        xml = """\
<dataset name="test">
  <frame number="1">
    <objectlist>
      <object category="people" id="7"
              x1="0" y1="0" x2="1024" y2="512"/>
      <object category="motorcyclist" id="8"
              x1="128" y1="64" x2="256" y2="128"/>
    </objectlist>
  </frame>
</dataset>
"""
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "labels.xml"
            xml_path.write_text(xml, encoding="utf-8")
            labels = load_butiv_annotations(xml_path)[1]

        self.assertEqual(
            [(item.class_id, item.class_name) for item in labels],
            [(0, "person"), (1, "bike_motorcycle")],
        )
        transformed = transform_to_detector_input(labels)
        self.assertEqual(transformed[0].xyxy, (0.0, 16.0, 640.0, 496.0))
        self.assertEqual(
            transformed[1].xyxy,
            (80.0, 76.0, 160.0, 136.0),
        )
        preserved = transform_to_detector_input(
            labels,
            content_width=640,
            content_height=320,
        )
        self.assertEqual(
            preserved[0].xyxy,
            (0.0, 96.0, 640.0, 416.0),
        )

    def test_butiv_native_resize_preserves_two_to_one_aspect(self) -> None:
        image = np.full((256, 512, 3), 200, dtype=np.uint8)
        native = _fit_native_preserve_aspect(image)

        self.assertEqual(native.shape, (120, 160))
        self.assertTrue(np.all(native[:20] == 114))
        self.assertTrue(np.all(native[20:100] == 200))
        self.assertTrue(np.all(native[100:] == 114))

    def test_random_butiv_selection_is_reproducible(self) -> None:
        annotations = {
            "3": {number: [] for number in range(1, 6)},
            "4": {number: [] for number in range(1, 6)},
        }
        first = _select_frames(annotations, samples=4, seed=42)
        second = _select_frames(annotations, samples=4, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len({(x.sequence, x.frame_number) for x in first}), 4)

    def test_iou_for_identical_and_disjoint_boxes(self) -> None:
        self.assertEqual(_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


if __name__ == "__main__":
    unittest.main()
