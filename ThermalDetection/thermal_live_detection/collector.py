"""Capture review candidates and unreviewed YOLO pseudo-labels."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CollectionPolicy:
    mode: str = "manual"
    interval_seconds: float = 10.0
    event_interval_seconds: float = 2.0
    event_confidence: float = 0.25
    uncertainty_low: float = 0.20
    uncertainty_high: float = 0.45
    novelty_threshold: float = 3.0
    max_items: int = 0


class CaptureSession:
    """Write aligned camera/native/SR/detector frames and review metadata."""

    SUBDIRECTORIES = (
        "source_frames",
        "native_frames",
        "sr_frames",
        "detector_inputs",
        "previews",
        "pseudo_labels",
    )

    def __init__(
        self,
        root: Path,
        policy: CollectionPolicy,
        class_names: dict[int, str],
        detector_input_name: str,
        camera_id: str,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.policy = policy
        self.class_names = class_names
        self.detector_input_name = detector_input_name
        self.camera_id = camera_id
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = root.resolve() / f"session_{timestamp}"
        for directory in self.SUBDIRECTORIES:
            (self.session_dir / directory).mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.session_dir / "metadata.jsonl"
        self._last_capture = 0.0
        self._last_event = 0.0
        self._last_thumbnail: np.ndarray | None = None
        self.count = 0
        (self.session_dir / "session.json").write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "camera_id": camera_id,
                    "detector_input": detector_input_name,
                    "class_names": class_names,
                    "policy": asdict(policy),
                    "label_warning": (
                        "pseudo_labels are model suggestions, not ground truth; "
                        "every file must be human-reviewed before training."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _novel_enough(self, gray: np.ndarray) -> bool:
        thumbnail = self._cv2.resize(gray, (64, 48))
        if self._last_thumbnail is None:
            return True
        difference = np.mean(
            np.abs(
                thumbnail.astype(np.float32)
                - self._last_thumbnail.astype(np.float32)
            )
        )
        return bool(difference >= self.policy.novelty_threshold)

    def should_capture(
        self,
        gray: np.ndarray,
        detections: list[dict[str, Any]],
        manual: bool = False,
    ) -> str | None:
        now = time.monotonic()
        if manual:
            return "manual"
        if self.policy.max_items and self.count >= self.policy.max_items:
            return None
        mode = self.policy.mode
        if mode in {"off", "manual"}:
            return None

        if mode in {"detections", "hybrid"}:
            uncertain = any(
                self.policy.uncertainty_low
                <= float(item["confidence"])
                <= self.policy.uncertainty_high
                for item in detections
            )
            event = any(
                float(item["confidence"]) >= self.policy.event_confidence
                for item in detections
            )
            if (
                (uncertain or event)
                and now - self._last_event
                >= self.policy.event_interval_seconds
            ):
                return "uncertain_detection" if uncertain else "detection_event"

        if mode in {"interval", "hybrid"}:
            if (
                now - self._last_capture >= self.policy.interval_seconds
                and self._novel_enough(gray)
            ):
                return "novel_interval"
        return None

    @staticmethod
    def _write_yolo_labels(
        path: Path,
        detections: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> None:
        lines = []
        for item in detections:
            x1, y1, x2, y2 = (float(value) for value in item["xyxy"])
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            lines.append(
                f"{int(item['class_id'])} {center_x:.8f} {center_y:.8f} "
                f"{box_w:.8f} {box_h:.8f}"
            )
        path.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )

    def save(
        self,
        reason: str,
        source_frame: np.ndarray,
        native_frame: np.ndarray,
        sr_frame: np.ndarray,
        detector_input: np.ndarray,
        preview: np.ndarray,
        detections: list[dict[str, Any]],
        frame_sequence: int,
        timings_ms: dict[str, float],
    ) -> Path:
        now = datetime.now(timezone.utc)
        stem = (
            f"{now.strftime('%Y%m%dT%H%M%S_%fZ')}"
            f"_frame{frame_sequence:09d}"
        )
        images = {
            "source_frames": source_frame,
            "native_frames": native_frame,
            "sr_frames": sr_frame,
            "detector_inputs": detector_input,
            "previews": preview,
        }
        for directory, image in images.items():
            path = self.session_dir / directory / f"{stem}.png"
            if not self._cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not save capture image: {path}")

        height, width = detector_input.shape[:2]
        label_path = self.session_dir / "pseudo_labels" / f"{stem}.txt"
        self._write_yolo_labels(
            label_path,
            detections=detections,
            width=width,
            height=height,
        )
        record = {
            "id": stem,
            "captured_utc": now.isoformat(),
            "frame_sequence": frame_sequence,
            "camera_id": self.camera_id,
            "reason": reason,
            "detector_input": self.detector_input_name,
            "source_shape": list(source_frame.shape),
            "native_shape": list(native_frame.shape),
            "sr_shape": list(sr_frame.shape),
            "detector_shape": list(detector_input.shape),
            "detections": detections,
            "timings_ms": timings_ms,
            "label_status": "pseudo_unreviewed",
        }
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.count += 1
        self._last_capture = time.monotonic()
        if reason in {"uncertain_detection", "detection_event"}:
            self._last_event = self._last_capture
        gray = (
            detector_input
            if detector_input.ndim == 2
            else self._cv2.cvtColor(
                detector_input, self._cv2.COLOR_BGR2GRAY
            )
        )
        self._last_thumbnail = self._cv2.resize(gray, (64, 48))
        return self.session_dir / "detector_inputs" / f"{stem}.png"
