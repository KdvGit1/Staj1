"""Evaluate random BU-TIV frames and save four-panel images and reports."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    _resolve_devices,
)
from thermal_live_detection.butiv import (
    BUAnnotation,
    draw_ground_truth,
    load_butiv_annotations,
    transform_to_detector_input,
)
from thermal_live_detection.butiv_demo import (
    DATA_ROOT,
    SEQUENCES,
    _native_panel,
    _put_banner,
    build_comparison_inputs,
)
from thermal_live_detection.edsr import EDSRUpscaler

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "results"
PIPELINES = ("source", "bicubic", "edsr")
PIPELINE_TITLES = {
    "source": "SOURCE + YOLO26n",
    "bicubic": "160x120 BICUBIC + YOLO26n",
    "edsr": "160x120 EDSR x4 + YOLO26n",
}
CLASS_NAMES = {0: "person", 1: "bike_motorcycle", 2: "car"}


@dataclass(frozen=True)
class SelectedFrame:
    sample_index: int
    sequence: str
    frame_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=200,
        help="Number of random annotated frames to save (default: 200).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence",
        choices=("3", "4", "all"),
        default="all",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--run-name")
    parser.add_argument("--edsr-weights", type=Path, default=DEFAULT_EDSR)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lens-blur-sigma", type=float, default=0.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--end2end", action="store_true")
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=4,
        help="Frames prepared before one three-pipeline YOLO batch.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
    )
    return parser.parse_args()


def _sequence_ids(requested: str) -> list[str]:
    return ["3", "4"] if requested == "all" else [requested]


def _iou(
    first: tuple[float, float, float, float] | list[float],
    second: tuple[float, float, float, float] | list[float],
) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0,
        float(first[3]) - float(first[1]),
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0,
        float(second[3]) - float(second[1]),
    )
    union = first_area + second_area - intersection
    return intersection / max(union, 1e-9)


def _empty_counts() -> dict[str, float | int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "predictions": 0,
        "ground_truth": 0,
        "confidence_sum": 0.0,
        "matched_iou_sum": 0.0,
    }


def _finalize_counts(
    counts: dict[str, float | int],
) -> dict[str, float | int]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    predictions = int(counts["predictions"])
    ground_truth = int(counts["ground_truth"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "mean_confidence": (
            float(counts["confidence_sum"]) / max(predictions, 1)
        ),
        "mean_matched_iou": (
            float(counts["matched_iou_sum"]) / max(tp, 1)
        ),
        "ground_truth": ground_truth,
    }


def _match_result(
    result: Any,
    ground_truth: list[BUAnnotation],
    match_iou: float,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box, confidence, class_id in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().tolist(),
        ):
            predictions.append(
                {
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in box],
                }
            )
    predictions.sort(key=lambda item: -item["confidence"])

    overall = _empty_counts()
    by_class = {class_id: _empty_counts() for class_id in CLASS_NAMES}
    for label in ground_truth:
        overall["ground_truth"] += 1
        by_class[label.class_id]["ground_truth"] += 1

    matched_ground_truth: set[int] = set()
    for prediction in predictions:
        class_id = prediction["class_id"]
        overall["predictions"] += 1
        overall["confidence_sum"] += prediction["confidence"]
        by_class[class_id]["predictions"] += 1
        by_class[class_id]["confidence_sum"] += prediction["confidence"]

        candidates = [
            (_iou(prediction["xyxy"], label.xyxy), index)
            for index, label in enumerate(ground_truth)
            if label.class_id == class_id
            and index not in matched_ground_truth
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= match_iou:
            matched_ground_truth.add(best_index)
            overall["tp"] += 1
            overall["matched_iou_sum"] += best_iou
            by_class[class_id]["tp"] += 1
            by_class[class_id]["matched_iou_sum"] += best_iou
        else:
            overall["fp"] += 1
            by_class[class_id]["fp"] += 1

    for class_id in CLASS_NAMES:
        class_counts = by_class[class_id]
        class_counts["fn"] = (
            int(class_counts["ground_truth"]) - int(class_counts["tp"])
        )
    overall["fn"] = int(overall["ground_truth"]) - int(overall["tp"])
    return {
        "overall": _finalize_counts(overall),
        "by_class": {
            class_id: _finalize_counts(counts)
            for class_id, counts in by_class.items()
        },
    }


def _render_comparison(
    sequence: str,
    frame_number: int,
    results: dict[str, Any],
    ground_truth: list[BUAnnotation],
    native: np.ndarray,
    lens_blur_sigma: float,
    sr_ms: float,
    yolo_batch_ms: float,
) -> np.ndarray:
    import cv2

    panels = {}
    for pipeline in PIPELINES:
        panel = results[pipeline].plot(labels=True, conf=True)
        panel = draw_ground_truth(panel, ground_truth)
        _put_banner(
            panel,
            PIPELINE_TITLES[pipeline],
            (
                f"Seq{sequence} frame={frame_number} | "
                f"GT={len(ground_truth)} "
                f"PRED={0 if results[pipeline].boxes is None else len(results[pipeline].boxes)} "
                f"| batch YOLO={yolo_batch_ms:.1f}ms"
            ),
        )
        panels[pipeline] = panel

    native_panel = _native_panel(native, lens_blur_sigma)
    cv2.putText(
        native_panel,
        f"EDSR={sr_ms:.1f}ms | random report sample",
        (10, 504),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack(
        (
            np.hstack((panels["source"], panels["bicubic"])),
            np.hstack((panels["edsr"], native_panel)),
        )
    )


def _create_run_directory(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or (
        f"butiv_random_n{args.samples}_seed{args.seed}_{timestamp}"
    )
    run_dir = args.results_root.resolve() / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Result directory already exists; choose another --run-name: {run_dir}"
        )
    (run_dir / "frames").mkdir(parents=True)
    (run_dir / "graphs").mkdir()
    return run_dir


def _select_frames(
    annotations: dict[str, dict[int, list[BUAnnotation]]],
    samples: int,
    seed: int,
) -> list[SelectedFrame]:
    if samples <= 0:
        raise ValueError("--samples must be positive.")
    candidates = [
        (sequence, frame_number)
        for sequence, frames in annotations.items()
        for frame_number in frames
    ]
    if samples > len(candidates):
        raise ValueError(
            f"Requested {samples} frames, but only {len(candidates)} "
            "annotated frames are available."
        )
    chosen = random.Random(seed).sample(candidates, samples)
    chosen.sort(key=lambda item: (int(item[0]), item[1]))
    return [
        SelectedFrame(index, sequence, frame_number)
        for index, (sequence, frame_number) in enumerate(chosen, start=1)
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for pipeline in PIPELINES:
        pipeline_rows = [
            row for row in rows if row["pipeline"] == pipeline
        ]
        for class_id, class_name in [
            ("all", "all"),
            *[(str(key), value) for key, value in CLASS_NAMES.items()],
        ]:
            suffix = "" if class_id == "all" else f"_class_{class_id}"
            tp = sum(int(row[f"tp{suffix}"]) for row in pipeline_rows)
            fp = sum(int(row[f"fp{suffix}"]) for row in pipeline_rows)
            fn = sum(int(row[f"fn{suffix}"]) for row in pipeline_rows)
            predictions = sum(
                int(row[f"predictions{suffix}"])
                for row in pipeline_rows
            )
            ground_truth = sum(
                int(row[f"ground_truth{suffix}"])
                for row in pipeline_rows
            )
            confidence_sum = sum(
                float(row[f"mean_confidence{suffix}"])
                * int(row[f"predictions{suffix}"])
                for row in pipeline_rows
            )
            matched_iou_sum = sum(
                float(row[f"mean_matched_iou{suffix}"])
                * int(row[f"tp{suffix}"])
                for row in pipeline_rows
            )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            summary.append(
                {
                    "pipeline": pipeline,
                    "class_id": class_id,
                    "class_name": class_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "precision": precision,
                    "recall": recall,
                    "f1": (
                        2.0
                        * precision
                        * recall
                        / max(precision + recall, 1e-12)
                    ),
                    "mean_confidence": confidence_sum
                    / max(predictions, 1),
                    "mean_matched_iou": matched_iou_sum / max(tp, 1),
                }
            )
    return summary


def _save_graphs(
    run_dir: Path,
    summary: list[dict[str, Any]],
    timing_rows: list[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "source": "#4C78A8",
        "bicubic": "#F58518",
        "edsr": "#54A24B",
    }
    graph_dir = run_dir / "graphs"
    overall = {
        row["pipeline"]: row
        for row in summary
        if row["class_id"] == "all"
    }

    metrics = ("precision", "recall", "f1")
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for index, pipeline in enumerate(PIPELINES):
        values = [overall[pipeline][metric] for metric in metrics]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=pipeline,
            color=colors[pipeline],
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    axis.set_xticks(x, ["Precision", "Recall", "F1"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Skor")
    axis.set_title("BU-TIV rastgele örneklem: genel tespit metrikleri")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        graph_dir / "pipeline_overall_metrics.png",
        dpi=160,
    )
    plt.close(figure)

    class_rows = {
        (row["pipeline"], row["class_name"]): row
        for row in summary
        if row["class_id"] != "all"
    }
    classes = list(CLASS_NAMES.values())
    x = np.arange(len(classes))
    figure, axis = plt.subplots(figsize=(10, 5.2))
    for index, pipeline in enumerate(PIPELINES):
        values = [
            class_rows[(pipeline, class_name)]["recall"]
            for class_name in classes
        ]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=pipeline,
            color=colors[pipeline],
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    axis.set_xticks(x, classes)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_title("Sınıf bazlı recall")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_dir / "class_recall.png", dpi=160)
    plt.close(figure)

    counts = ("tp", "fp", "fn")
    x = np.arange(len(counts))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for index, pipeline in enumerate(PIPELINES):
        values = [overall[pipeline][metric] for metric in counts]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=pipeline,
            color=colors[pipeline],
        )
        axis.bar_label(bars, fmt="%d", fontsize=8)
    axis.set_xticks(x, ["TP", "FP", "FN"])
    axis.set_ylabel("Kutu sayısı")
    axis.set_title("Doğru, yanlış ve kaçırılan kutular")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_dir / "tp_fp_fn.png", dpi=160)
    plt.close(figure)

    mean_edsr = float(np.mean([row["edsr_ms"] for row in timing_rows]))
    mean_yolo = float(
        np.mean([row["yolo_batch_ms_per_frame"] for row in timing_rows])
    )
    mean_total = float(
        np.mean([row["total_ms_per_frame"] for row in timing_rows])
    )
    values = [mean_edsr, mean_yolo, mean_total]
    figure, axis = plt.subplots(figsize=(8, 5.2))
    bars = axis.bar(
        ["EDSR", "YOLO batch/frame", "Toplam/frame"],
        values,
        color=["#54A24B", "#4C78A8", "#B279A2"],
    )
    axis.bar_label(bars, fmt="%.1f ms", fontsize=9)
    axis.set_ylabel("Milisaniye")
    axis.set_title("Ortalama CPU/GPU işlem süreleri")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(graph_dir / "runtime_ms.png", dpi=160)
    plt.close(figure)


def _write_text_report(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    summary: list[dict[str, Any]],
    timing_rows: list[dict[str, float]],
) -> None:
    overall = [
        row for row in summary if row["class_id"] == "all"
    ]
    lines = [
        "BU-TIV RASTGELE DÖRT-PANEL TEST RAPORU",
        "=" * 44,
        "",
        f"Rastgele kare sayısı : {args.samples}",
        f"Seed                : {args.seed}",
        f"Sekans              : {args.sequence}",
        f"Confidence          : {args.conf}",
        f"NMS IoU             : {args.iou}",
        f"Eşleştirme IoU      : {args.match_iou}",
        f"Lens blur sigma     : {args.lens_blur_sigma}",
        f"Sonuç klasörü       : {run_dir}",
        "",
        "GENEL SONUÇLAR",
        "-" * 44,
        (
            f"{'pipeline':<10} {'TP':>7} {'FP':>7} {'FN':>7} "
            f"{'precision':>11} {'recall':>9} {'F1':>9}"
        ),
    ]
    for row in overall:
        lines.append(
            f"{row['pipeline']:<10} {row['tp']:>7} {row['fp']:>7} "
            f"{row['fn']:>7} {row['precision']:>11.4f} "
            f"{row['recall']:>9.4f} {row['f1']:>9.4f}"
        )
    lines.extend(
        [
            "",
            "SINIF BAZLI SONUÇLAR",
            "-" * 44,
        ]
    )
    for row in summary:
        if row["class_id"] == "all":
            continue
        lines.append(
            f"{row['pipeline']:<10} {row['class_name']:<18} "
            f"TP={row['tp']:<6} FP={row['fp']:<6} FN={row['fn']:<6} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "ORTALAMA SÜRELER",
            "-" * 44,
            (
                "EDSR: "
                f"{np.mean([row['edsr_ms'] for row in timing_rows]):.2f} ms"
            ),
            (
                "YOLO üç-pipeline batch / kare: "
                f"{np.mean([row['yolo_batch_ms_per_frame'] for row in timing_rows]):.2f} ms"
            ),
            (
                "Toplam / kare: "
                f"{np.mean([row['total_ms_per_frame'] for row in timing_rows]):.2f} ms"
            ),
            "",
            "DOSYALAR",
            "-" * 44,
            "frames/                   Dört panelli rastgele kareler",
            "per_frame_metrics.csv     Kare ve pipeline bazlı metrikler",
            "summary.csv               Toplu ve sınıf bazlı sonuçlar",
            "summary.json              Makine tarafından okunabilir özet",
            "selected_frames.csv       Tekrar üretilebilir örnek listesi",
            "graphs/*.png              Sonuç grafikleri",
            "",
            "NOT",
            "-" * 44,
            (
                "Bu rapor seçilen rastgele karelerin tanısal karşılaştırmasıdır. "
                "Tam BU-TIV benchmark sonucu değildir."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    if not 0 <= args.lens_blur_sigma:
        raise ValueError("--lens-blur-sigma cannot be negative.")
    if not 0 < args.match_iou <= 1:
        raise ValueError("--match-iou must be in (0, 1].")
    if args.batch_frames <= 0:
        raise ValueError("--batch-frames must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    sequence_ids = _sequence_ids(args.sequence)
    annotations = {
        sequence: load_butiv_annotations(SEQUENCES[sequence][1])
        for sequence in sequence_ids
    }
    selected = _select_frames(
        annotations,
        samples=args.samples,
        seed=args.seed,
    )
    run_dir = _create_run_directory(args)

    edsr_device, yolo_device = _resolve_devices(args.device)
    upscaler = EDSRUpscaler(
        checkpoint_path=args.edsr_weights,
        device=edsr_device,
        fp16=args.fp16,
        native_size=(160, 120),
    )
    detector = YOLO(str(args.detector_weights.resolve()))
    require_three_class_model(detector)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "classes": model_names(detector),
        "edsr_metadata": upscaler.checkpoint_metadata,
        "data_root": str(DATA_ROOT.resolve()),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "selected_frames.csv",
        [asdict(frame) for frame in selected],
    )

    captures = {
        sequence: cv2.VideoCapture(str(SEQUENCES[sequence][0]))
        for sequence in sequence_ids
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("At least one BU-TIV video could not be opened.")

    metric_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float]] = []
    processed = 0
    started = time.perf_counter()
    try:
        for chunk_start in range(0, len(selected), args.batch_frames):
            chunk = selected[
                chunk_start : chunk_start + args.batch_frames
            ]
            prepared = []
            flat_inputs = []
            for selected_frame in chunk:
                capture = captures[selected_frame.sequence]
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    selected_frame.frame_number - 1,
                )
                ok, source_frame = capture.read()
                if not ok or source_frame is None:
                    raise RuntimeError(
                        "Could not read "
                        f"Seq{selected_frame.sequence} "
                        f"frame {selected_frame.frame_number}"
                    )
                inputs, native, sr_ms = build_comparison_inputs(
                    source_frame,
                    upscaler,
                    args.lens_blur_sigma,
                )
                ground_truth = transform_to_detector_input(
                    annotations[selected_frame.sequence][
                        selected_frame.frame_number
                    ],
                    content_width=640,
                    content_height=320,
                )
                prepared.append(
                    (
                        selected_frame,
                        inputs,
                        native,
                        sr_ms,
                        ground_truth,
                    )
                )
                flat_inputs.extend(inputs[pipeline] for pipeline in PIPELINES)

            yolo_start = time.perf_counter()
            flat_results = detector.predict(
                source=flat_inputs,
                imgsz=(512, 640),
                rect=False,
                classes=[0, 1, 2],
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=yolo_device,
                end2end=args.end2end,
                batch=len(flat_inputs),
                verbose=False,
            )
            yolo_batch_ms = (time.perf_counter() - yolo_start) * 1000.0
            yolo_per_frame_ms = yolo_batch_ms / len(chunk)

            for chunk_index, (
                selected_frame,
                _inputs,
                native,
                sr_ms,
                ground_truth,
            ) in enumerate(prepared):
                result_slice = flat_results[
                    chunk_index * len(PIPELINES) :
                    (chunk_index + 1) * len(PIPELINES)
                ]
                results = dict(zip(PIPELINES, result_slice))
                frame_file = (
                    f"sample{selected_frame.sample_index:04d}_"
                    f"seq{selected_frame.sequence}_"
                    f"frame{selected_frame.frame_number:06d}.jpg"
                )
                comparison = _render_comparison(
                    sequence=selected_frame.sequence,
                    frame_number=selected_frame.frame_number,
                    results=results,
                    ground_truth=ground_truth,
                    native=native,
                    lens_blur_sigma=args.lens_blur_sigma,
                    sr_ms=sr_ms,
                    yolo_batch_ms=yolo_batch_ms,
                )
                frame_path = run_dir / "frames" / frame_file
                if not cv2.imwrite(
                    str(frame_path),
                    comparison,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                ):
                    raise RuntimeError(f"Could not save {frame_path}")

                for pipeline in PIPELINES:
                    metrics = _match_result(
                        results[pipeline],
                        ground_truth,
                        args.match_iou,
                    )
                    overall = metrics["overall"]
                    row: dict[str, Any] = {
                        "sample_index": selected_frame.sample_index,
                        "sequence": selected_frame.sequence,
                        "frame_number": selected_frame.frame_number,
                        "pipeline": pipeline,
                        "frame_file": f"frames/{frame_file}",
                        "sr_ms": round(sr_ms, 3),
                        "yolo_batch_ms_per_frame": round(
                            yolo_per_frame_ms,
                            3,
                        ),
                    }
                    for key in (
                        "tp",
                        "fp",
                        "fn",
                        "predictions",
                        "ground_truth",
                        "precision",
                        "recall",
                        "f1",
                        "mean_confidence",
                        "mean_matched_iou",
                    ):
                        row[key] = overall[key]
                    for class_id, class_metrics in metrics[
                        "by_class"
                    ].items():
                        for key in (
                            "tp",
                            "fp",
                            "fn",
                            "predictions",
                            "ground_truth",
                            "precision",
                            "recall",
                            "f1",
                            "mean_confidence",
                            "mean_matched_iou",
                        ):
                            row[f"{key}_class_{class_id}"] = (
                                class_metrics[key]
                            )
                    metric_rows.append(row)

                processed += 1
                timing_rows.append(
                    {
                        "edsr_ms": sr_ms,
                        "yolo_batch_ms_per_frame": yolo_per_frame_ms,
                        "total_ms_per_frame": sr_ms + yolo_per_frame_ms,
                    }
                )
                if processed % 10 == 0 or processed == len(selected):
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {processed}/{len(selected)} frames "
                        f"({elapsed:.1f}s elapsed)",
                        flush=True,
                    )
    finally:
        for capture in captures.values():
            capture.release()

    summary = _aggregate(metric_rows)
    _write_csv(run_dir / "per_frame_metrics.csv", metric_rows)
    _write_csv(run_dir / "summary.csv", summary)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "summary": summary,
                "timing": {
                    "mean_edsr_ms": float(
                        np.mean([row["edsr_ms"] for row in timing_rows])
                    ),
                    "mean_yolo_batch_ms_per_frame": float(
                        np.mean(
                            [
                                row["yolo_batch_ms_per_frame"]
                                for row in timing_rows
                            ]
                        )
                    ),
                    "mean_total_ms_per_frame": float(
                        np.mean(
                            [
                                row["total_ms_per_frame"]
                                for row in timing_rows
                            ]
                        )
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_graphs(run_dir, summary, timing_rows)
    _write_text_report(
        run_dir / "REPORT.txt",
        args,
        run_dir,
        summary,
        timing_rows,
    )
    print(f"Report complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
