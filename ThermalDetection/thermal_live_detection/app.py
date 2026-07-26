"""Run camera -> EDSR x4 -> YOLO26 detection and capture review data."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from thermal_detection.model_utils import model_names, require_three_class_model
from thermal_live_detection.collector import CaptureSession, CollectionPolicy
from thermal_live_detection.edsr import EDSRUpscaler
from thermal_live_detection.stream import (
    LatestFrameStream,
    build_hikvision_rtsp_url,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_EDSR = PACKAGE_ROOT / "weights" / "edsr_x4_best.pth"
DEFAULT_YOLO = PACKAGE_ROOT / "weights" / "yolo26n_thermal_best.pt"
DETECTOR_WIDTH = 640
DETECTOR_HEIGHT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help=(
            "Optional RTSP URL, video path or camera index. If omitted, the "
            "Hikvision URL is built from camera arguments and environment."
        ),
    )
    parser.add_argument(
        "--camera-ip",
        default=os.getenv("THERMAL_CAMERA_IP", ""),
    )
    parser.add_argument(
        "--camera-user",
        default=os.getenv("THERMAL_CAMERA_USER", "admin"),
    )
    parser.add_argument(
        "--password-env",
        default="THERMAL_CAMERA_PASSWORD",
        help="Environment variable containing the camera password.",
    )
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--channel", default="202", choices=("201", "202"))
    parser.add_argument("--edsr-weights", type=Path, default=DEFAULT_EDSR)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--native-width", type=int, default=160)
    parser.add_argument("--native-height", type=int, default=120)
    parser.add_argument(
        "--detector-input",
        choices=("edsr", "bicubic", "source"),
        default="edsr",
        help="Image variant passed to YOLO; use this for controlled A/B runs.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Use the NMS-free one-to-one head instead of traditional NMS.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=("off", "manual", "interval", "detections", "hybrid"),
        default="manual",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=PACKAGE_ROOT / "captures",
    )
    parser.add_argument("--capture-interval", type=float, default=10.0)
    parser.add_argument("--event-interval", type=float, default=2.0)
    parser.add_argument("--event-confidence", type=float, default=0.25)
    parser.add_argument("--uncertainty-low", type=float, default=0.20)
    parser.add_argument("--uncertainty-high", type=float, default=0.45)
    parser.add_argument("--novelty-threshold", type=float, default=3.0)
    parser.add_argument("--max-captures", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N processed frames; zero means unlimited.",
    )
    return parser.parse_args()


def _resolve_source(args: argparse.Namespace) -> tuple[str | int, str]:
    if args.source:
        source: str | int = (
            int(args.source) if args.source.isdecimal() else args.source
        )
        return source, "explicit_source"
    password = os.getenv(args.password_env)
    if not password:
        raise RuntimeError(
            f"Camera password is missing. Set {args.password_env} without "
            "placing the secret in source code or command history."
        )
    url = build_hikvision_rtsp_url(
        ip=args.camera_ip,
        username=args.camera_user,
        password=password,
        channel=args.channel,
        rtsp_port=args.rtsp_port,
    )
    return url, f"{args.camera_ip}_ch{args.channel}"


def _resolve_devices(requested: str) -> tuple[str, str]:
    import torch

    if requested != "auto":
        yolo_device = requested
        edsr_device = (
            f"cuda:{requested}"
            if requested.isdecimal()
            else requested
        )
        return edsr_device, yolo_device
    if torch.cuda.is_available():
        return "cuda:0", "0"
    return "cpu", "cpu"


def _detections_from_result(
    result: Any,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    return [
        {
            "class_id": int(class_id),
            "class_name": class_names[int(class_id)],
            "confidence": round(float(confidence), 6),
            "xyxy": [round(float(value), 3) for value in box],
        }
        for box, confidence, class_id in zip(boxes, confidences, classes)
    ]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _letterbox_for_detector(image: np.ndarray) -> np.ndarray:
    """Fit an image into the fixed 640x512 contract without distortion."""
    import cv2

    height, width = image.shape[:2]
    scale = min(DETECTOR_WIDTH / width, DETECTOR_HEIGHT / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (DETECTOR_WIDTH - resized_width) // 2
    top = (DETECTOR_HEIGHT - resized_height) // 2
    right = DETECTOR_WIDTH - resized_width - left
    bottom = DETECTOR_HEIGHT - resized_height - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def main() -> None:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    source, camera_id = _resolve_source(args)
    edsr_device, yolo_device = _resolve_devices(args.device)
    upscaler = EDSRUpscaler(
        checkpoint_path=args.edsr_weights,
        device=edsr_device,
        fp16=args.fp16,
        native_size=(args.native_width, args.native_height),
    )
    detector = YOLO(str(args.detector_weights.resolve()))
    require_three_class_model(detector)
    class_names = model_names(detector)

    policy = CollectionPolicy(
        mode=args.capture_mode,
        interval_seconds=args.capture_interval,
        event_interval_seconds=args.event_interval,
        event_confidence=args.event_confidence,
        uncertainty_low=args.uncertainty_low,
        uncertainty_high=args.uncertainty_high,
        novelty_threshold=args.novelty_threshold,
        max_items=args.max_captures,
    )
    collector = (
        None
        if args.capture_mode == "off"
        else CaptureSession(
            root=args.capture_root,
            policy=policy,
            class_names=class_names,
            detector_input_name=args.detector_input,
            camera_id=camera_id,
        )
    )

    print(
        "Pipeline ready: camera -> 160x120 -> EDSR 640x480 -> "
        f"YOLO ({'end-to-end' if args.end2end else 'traditional NMS'})"
    )
    print(
        f"Devices: EDSR={edsr_device}, YOLO={yolo_device}; "
        f"capture={args.capture_mode}"
    )
    if collector is not None:
        print(f"Capture session: {collector.session_dir}")
    if not args.headless:
        print("Keys: [s] save review candidate, [q] quit")

    sequence = 0
    processed = 0
    smoothed_fps = 0.0
    with LatestFrameStream(source) as stream:
        try:
            while True:
                packet = stream.read(after_sequence=sequence, timeout=3.0)
                if packet is None:
                    print("Waiting for a new camera frame...")
                    continue
                sequence = packet.sequence
                loop_start = time.perf_counter()
                source_frame = packet.image

                sr_start = time.perf_counter()
                native, sr_gray = upscaler.upscale(source_frame)
                sr_ms = (time.perf_counter() - sr_start) * 1000.0
                bicubic = cv2.resize(
                    native,
                    (args.native_width * 4, args.native_height * 4),
                    interpolation=cv2.INTER_CUBIC,
                )
                variants = {
                    "edsr": _as_bgr(sr_gray),
                    "bicubic": _as_bgr(bicubic),
                    "source": _as_bgr(source_frame),
                }
                detector_input = _letterbox_for_detector(
                    variants[args.detector_input]
                )

                detect_start = time.perf_counter()
                result = detector.predict(
                    source=detector_input,
                    imgsz=(DETECTOR_HEIGHT, DETECTOR_WIDTH),
                    rect=False,
                    classes=[0, 1, 2],
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=yolo_device,
                    end2end=args.end2end,
                    verbose=False,
                )[0]
                detect_ms = (time.perf_counter() - detect_start) * 1000.0
                detections = _detections_from_result(result, class_names)
                preview = result.plot()

                elapsed = time.perf_counter() - loop_start
                instantaneous_fps = 1.0 / max(elapsed, 1e-6)
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps == 0.0
                    else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
                )
                cv2.putText(
                    preview,
                    (
                        f"EDSR {sr_ms:.1f} ms | YOLO {detect_ms:.1f} ms | "
                        f"{smoothed_fps:.1f} FPS | {args.detector_input}"
                    ),
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

                manual_capture = False
                quit_requested = False
                if not args.headless:
                    cv2.imshow("Thermal EDSR + YOLO", preview)
                    key = cv2.waitKey(1) & 0xFF
                    manual_capture = key == ord("s")
                    quit_requested = key == ord("q")

                if collector is not None:
                    reason = collector.should_capture(
                        gray=sr_gray,
                        detections=detections,
                        manual=manual_capture,
                    )
                    if reason:
                        saved = collector.save(
                            reason=reason,
                            source_frame=source_frame,
                            native_frame=native,
                            sr_frame=sr_gray,
                            detector_input=detector_input,
                            preview=preview,
                            detections=detections,
                            frame_sequence=sequence,
                            timings_ms={
                                "edsr": round(sr_ms, 3),
                                "yolo": round(detect_ms, 3),
                                "loop": round(elapsed * 1000.0, 3),
                            },
                        )
                        print(
                            f"Captured {reason}: {saved.name} "
                            f"({len(detections)} pseudo-labels)"
                        )

                processed += 1
                if processed % 100 == 0:
                    print(
                        f"Processed={processed}, FPS={smoothed_fps:.2f}, "
                        f"detections={len(detections)}"
                    )
                if quit_requested:
                    break
                if args.max_frames and processed >= args.max_frames:
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cv2.destroyAllWindows()

    print(
        f"Stopped after {processed} frames"
        + (
            f"; captured {collector.count} candidates in {collector.session_dir}"
            if collector is not None
            else ""
        )
    )


if __name__ == "__main__":
    main()
