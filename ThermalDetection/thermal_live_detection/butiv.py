"""BU-TIV marathon annotation parsing and detector-coordinate transforms."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANNOTATION_WIDTH = 1024
ANNOTATION_HEIGHT = 512

CATEGORY_MAP = {
    "people": (0, "person"),
    "cyclist": (1, "bike_motorcycle"),
    "motorcyclist": (1, "bike_motorcycle"),
    "car": (2, "car"),
}


@dataclass(frozen=True)
class BUAnnotation:
    class_id: int
    class_name: str
    track_id: int
    xyxy: tuple[float, float, float, float]
    source_category: str


def load_butiv_annotations(
    xml_path: Path,
) -> dict[int, list[BUAnnotation]]:
    """Read BU-TIV XML with bounded memory and return 1-based frame labels."""
    xml_path = xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"BU-TIV annotation file not found: {xml_path}")

    frames: dict[int, list[BUAnnotation]] = {}
    for _event, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag != "frame":
            continue
        frame_number = int(element.attrib["number"])
        labels: list[BUAnnotation] = []
        object_list = element.find("objectlist")
        if object_list is not None:
            for item in object_list.findall("object"):
                category = item.attrib["category"].strip().lower()
                mapped = CATEGORY_MAP.get(category)
                if mapped is None:
                    continue
                class_id, class_name = mapped
                labels.append(
                    BUAnnotation(
                        class_id=class_id,
                        class_name=class_name,
                        track_id=int(item.attrib["id"]),
                        xyxy=(
                            float(item.attrib["x1"]),
                            float(item.attrib["y1"]),
                            float(item.attrib["x2"]),
                            float(item.attrib["y2"]),
                        ),
                        source_category=category,
                    )
                )
        frames[frame_number] = labels
        element.clear()
    return frames


def transform_to_detector_input(
    labels: list[BUAnnotation],
    detector_width: int = 640,
    detector_height: int = 512,
    content_width: int = 640,
    content_height: int = 480,
    annotation_width: int = ANNOTATION_WIDTH,
    annotation_height: int = ANNOTATION_HEIGHT,
) -> list[BUAnnotation]:
    """Map XML boxes through resize and centered letterbox into 640x512."""
    left = (detector_width - content_width) / 2.0
    top = (detector_height - content_height) / 2.0
    x_scale = content_width / annotation_width
    y_scale = content_height / annotation_height
    transformed: list[BUAnnotation] = []
    for label in labels:
        x1, y1, x2, y2 = label.xyxy
        x1 = float(np.clip(x1 * x_scale + left, 0, detector_width))
        x2 = float(np.clip(x2 * x_scale + left, 0, detector_width))
        y1 = float(np.clip(y1 * y_scale + top, 0, detector_height))
        y2 = float(np.clip(y2 * y_scale + top, 0, detector_height))
        if x2 <= x1 or y2 <= y1:
            continue
        transformed.append(
            BUAnnotation(
                class_id=label.class_id,
                class_name=label.class_name,
                track_id=label.track_id,
                xyxy=(x1, y1, x2, y2),
                source_category=label.source_category,
            )
        )
    return transformed


def draw_ground_truth(
    image: np.ndarray,
    labels: list[BUAnnotation],
) -> np.ndarray:
    """Draw thin class-colored ground-truth boxes on an image copy."""
    import cv2

    output = image.copy()
    colors = {
        0: (80, 220, 80),
        1: (220, 80, 220),
        2: (30, 170, 255),
    }
    for label in labels:
        x1, y1, x2, y2 = (round(value) for value in label.xyxy)
        color = colors[label.class_id]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        text = f"GT {label.class_name} #{label.track_id}"
        cv2.putText(
            output,
            text,
            (x1, max(11, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )
    return output
