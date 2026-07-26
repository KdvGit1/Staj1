"""Play BU-TIV through blur -> 160x120 -> EDSR x4 -> YOLO."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from thermal_detection.model_utils import (
    model_names,
    require_three_class_model,
)
from thermal_live_detection.app import (
    DEFAULT_EDSR,
    DEFAULT_YOLO,
    _letterbox_for_detector,
    _resolve_devices,
)
from thermal_live_detection.butiv import (
    draw_ground_truth,
    load_butiv_annotations,
    transform_to_detector_input,
)
from thermal_live_detection.edsr import EDSRUpscaler

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data" / "bu_tiv"

SEQUENCES = {
    "3": (
        DATA_ROOT / "test_seq3.mp4",
        DATA_ROOT / "marathon_3_2d.xml",
    ),
    "4": (
        DATA_ROOT / "test_seq4.mp4",
        DATA_ROOT / "marathon_4_2d.xml",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        choices=("3", "4", "all"),
        default="all",
    )
    parser.add_argument("--edsr-weights", type=Path, default=DEFAULT_EDSR)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--lens-blur-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian optical blur applied before 160x120 sampling; "
            "zero disables it (default)."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--end2end", action="store_true")
    parser.add_argument("--hide-ground-truth", action="store_true")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum total processed frames; zero plays all frames.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.75,
        help="Scale the 1280x1024 comparison window for the desktop.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional MP4 path for saving the side-by-side visualization.",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        help="Optional PNG path; saves the first processed comparison frame.",
    )
    return parser.parse_args()


def _apply_lens_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    if sigma < 0:
        raise ValueError("--lens-blur-sigma cannot be negative.")
    if sigma == 0:
        return frame
    return cv2.GaussianBlur(
        frame,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )


def _fit_native_preserve_aspect(
    frame: np.ndarray,
    width: int = 160,
    height: int = 120,
) -> np.ndarray:
    """Create a 160x120 grayscale sensor frame without geometry distortion."""
    import cv2

    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3
        else frame
    )
    source_height, source_width = gray.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    resized = cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    right = width - resized_width - left
    bottom = height - resized_height - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=114,
    )


def build_comparison_inputs(
    source_frame: np.ndarray,
    upscaler: EDSRUpscaler,
    lens_blur_sigma: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Build aligned source, bicubic and EDSR detector inputs."""
    import cv2

    source_input = _letterbox_for_detector(source_frame)
    blurred = _apply_lens_blur(source_frame, lens_blur_sigma)
    native = _fit_native_preserve_aspect(blurred)
    sr_start = time.perf_counter()
    native, sr_gray = upscaler.upscale(native)
    sr_ms = (time.perf_counter() - sr_start) * 1000.0
    bicubic_gray = cv2.resize(
        native,
        (640, 480),
        interpolation=cv2.INTER_CUBIC,
    )
    bicubic_input = _letterbox_for_detector(
        cv2.cvtColor(bicubic_gray, cv2.COLOR_GRAY2BGR)
    )
    edsr_input = _letterbox_for_detector(
        cv2.cvtColor(sr_gray, cv2.COLOR_GRAY2BGR)
    )
    return (
        {
            "source": source_input,
            "bicubic": bicubic_input,
            "edsr": edsr_input,
        },
        native,
        sr_ms,
    )


def _put_banner(
    image: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (15, 15, 15), -1)
    cv2.putText(
        image,
        title,
        (10, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        subtitle,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )


def _native_panel(native: np.ndarray, blur_sigma: float) -> np.ndarray:
    """Magnify pixels for viewing only; no interpolation or SR is applied."""
    import cv2

    magnified = cv2.resize(
        native,
        (640, 480),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = cv2.copyMakeBorder(
        cv2.cvtColor(magnified, cv2.COLOR_GRAY2BGR),
        16,
        16,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    _put_banner(
        panel,
        "NATIVE INPUT: 160x120 (content 160x80)",
        (
            "Aspect preserved, nearest display only | "
            f"lens sigma={blur_sigma:.2f}"
        ),
    )
    return panel


def _prediction_count(result: Any) -> int:
    return 0 if result.boxes is None else len(result.boxes)


def _sequence_ids(requested: str) -> list[str]:
    return ["3", "4"] if requested == "all" else [requested]


def main() -> None:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    if args.display_scale <= 0:
        raise ValueError("--display-scale must be positive.")

    edsr_device, yolo_device = _resolve_devices(args.device)
    upscaler = EDSRUpscaler(
        checkpoint_path=args.edsr_weights,
        device=edsr_device,
        fp16=args.fp16,
        native_size=(160, 120),
    )
    detector = YOLO(str(args.detector_weights.resolve()))
    require_three_class_model(detector)
    names = model_names(detector)

    annotations = {}
    for sequence_id in _sequence_ids(args.sequence):
        video_path, xml_path = SEQUENCES[sequence_id]
        if not video_path.is_file():
            raise FileNotFoundError(f"BU-TIV video not found: {video_path}")
        annotations[sequence_id] = load_butiv_annotations(xml_path)
        print(
            f"Seq{sequence_id}: {len(annotations[sequence_id])} "
            f"annotated frames loaded"
        )

    writer = None
    preview_saved = False
    total_processed = 0
    smoothed_fps = 0.0
    stop_requested = False

    if args.output_video:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            for sequence_id in _sequence_ids(args.sequence):
                video_path, _xml_path = SEQUENCES[sequence_id]
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open video: {video_path}")
                source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    max(0, args.start_frame - 1),
                )
                print(
                    f"Playing Seq{sequence_id}: {video_path.name}, "
                    f"{source_fps:.2f} FPS"
                )

                try:
                    while True:
                        frame_start = time.perf_counter()
                        ok, source_frame = capture.read()
                        if not ok or source_frame is None:
                            break
                        frame_number = int(
                            capture.get(cv2.CAP_PROP_POS_FRAMES)
                        )

                        inputs, native, sr_ms = build_comparison_inputs(
                            source_frame,
                            upscaler,
                            args.lens_blur_sigma,
                        )

                        detect_start = time.perf_counter()
                        results = detector.predict(
                            source=[
                                inputs["source"],
                                inputs["bicubic"],
                                inputs["edsr"],
                            ],
                            imgsz=(512, 640),
                            rect=False,
                            classes=[0, 1, 2],
                            conf=args.conf,
                            iou=args.iou,
                            max_det=args.max_det,
                            device=yolo_device,
                            end2end=args.end2end,
                            verbose=False,
                        )
                        yolo_ms = (
                            time.perf_counter() - detect_start
                        ) * 1000.0

                        ground_truth = transform_to_detector_input(
                            annotations[sequence_id].get(
                                frame_number,
                                [],
                            ),
                            content_width=640,
                            content_height=320,
                        )

                        elapsed = time.perf_counter() - frame_start
                        fps = 1.0 / max(elapsed, 1e-6)
                        smoothed_fps = (
                            fps
                            if smoothed_fps == 0.0
                            else 0.9 * smoothed_fps + 0.1 * fps
                        )
                        panel_titles = (
                            "SOURCE + YOLO26n",
                            "160x120 BICUBIC + YOLO26n",
                            "160x120 EDSR x4 + YOLO26n",
                        )
                        prediction_panels = []
                        for title, result in zip(panel_titles, results):
                            panel = result.plot(labels=True, conf=True)
                            if not args.hide_ground_truth:
                                panel = draw_ground_truth(
                                    panel,
                                    ground_truth,
                                )
                            _put_banner(
                                panel,
                                title,
                                (
                                    f"Seq{sequence_id} "
                                    f"frame={frame_number} | "
                                    f"GT={len(ground_truth)} "
                                    f"PRED={_prediction_count(result)} | "
                                    f"batch YOLO={yolo_ms:.1f}ms"
                                ),
                            )
                            prediction_panels.append(panel)
                        native_panel = _native_panel(
                            native,
                            args.lens_blur_sigma,
                        )
                        cv2.putText(
                            native_panel,
                            (
                                f"EDSR={sr_ms:.1f}ms | "
                                f"pipeline FPS={smoothed_fps:.1f}"
                            ),
                            (10, 504),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 230, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        comparison = np.vstack(
                            (
                                np.hstack(
                                    (
                                        prediction_panels[0],
                                        prediction_panels[1],
                                    )
                                ),
                                np.hstack(
                                    (
                                        prediction_panels[2],
                                        native_panel,
                                    )
                                ),
                            )
                        )

                        if args.save_preview and not preview_saved:
                            if not cv2.imwrite(
                                str(args.save_preview.resolve()),
                                comparison,
                            ):
                                raise RuntimeError(
                                    "Could not save preview image: "
                                    f"{args.save_preview}"
                                )
                            preview_saved = True

                        if args.output_video:
                            if writer is None:
                                writer = cv2.VideoWriter(
                                    str(args.output_video.resolve()),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    source_fps,
                                    (comparison.shape[1], comparison.shape[0]),
                                )
                                if not writer.isOpened():
                                    raise RuntimeError(
                                        "Could not create output video: "
                                        f"{args.output_video}"
                                    )
                            writer.write(comparison)

                        if not args.headless:
                            displayed = comparison
                            if args.display_scale != 1.0:
                                displayed = cv2.resize(
                                    comparison,
                                    None,
                                    fx=args.display_scale,
                                    fy=args.display_scale,
                                    interpolation=cv2.INTER_AREA,
                                )
                            cv2.imshow(
                                "BU-TIV | EDSR+YOLO vs 160x120",
                                displayed,
                            )
                            remaining_ms = (
                                1000.0 / source_fps
                                - (
                                    time.perf_counter() - frame_start
                                )
                                * 1000.0
                            )
                            key = cv2.waitKey(
                                max(1, round(remaining_ms))
                            ) & 0xFF
                            if key in (ord("q"), 27):
                                stop_requested = True
                            elif key in (ord("p"), ord(" ")):
                                while True:
                                    pause_key = cv2.waitKey(0) & 0xFF
                                    if pause_key in (
                                        ord("p"),
                                        ord(" "),
                                    ):
                                        break
                                    if pause_key in (ord("q"), 27):
                                        stop_requested = True
                                        break

                        total_processed += 1
                        if (
                            args.max_frames
                            and total_processed >= args.max_frames
                        ):
                            stop_requested = True
                        if stop_requested:
                            break
                finally:
                    capture.release()
                if stop_requested:
                    break
            if stop_requested or not args.loop:
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(
        f"Finished: {total_processed} frames; classes={names}; "
        f"lens_sigma={args.lens_blur_sigma}"
    )


if __name__ == "__main__":
    main()
