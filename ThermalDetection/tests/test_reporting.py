from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from thermal_detection.reporting import (
    generate_dataset_report,
    generate_evaluation_report,
    generate_prediction_report,
    generate_training_report,
)


class ReportingTests(unittest.TestCase):
    def test_dataset_report_contains_all_splits_and_classes(self) -> None:
        manifest = {
            "splits": {
                "train": {
                    "images": 100,
                    "target_boxes": 1000,
                    "empty_target_images": 5,
                    "class_counts": {
                        "person": 400,
                        "bike_motorcycle": 100,
                        "car": 500,
                    },
                },
                "val": {
                    "images": 20,
                    "target_boxes": 150,
                    "empty_target_images": 2,
                    "class_counts": {
                        "person": 60,
                        "bike_motorcycle": 10,
                        "car": 80,
                    },
                },
                "test": {
                    "images": 30,
                    "target_boxes": 300,
                    "empty_target_images": 3,
                    "class_counts": {
                        "person": 100,
                        "bike_motorcycle": 50,
                        "car": 150,
                    },
                },
            }
        }
        verification = {"manifest_match": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            verification_path = root / "verification.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            outputs = generate_dataset_report(
                manifest_path,
                verification_path,
                root / "report",
            )
            self.assertTrue(outputs["html"].is_file())
            self.assertTrue(outputs["distribution"].is_file())
            ET.parse(outputs["distribution"])
            html = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("bike_motorcycle", html)
            self.assertIn("Manifest", html)

    def test_training_report_from_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            csv_path = run_dir / "results.csv"
            fieldnames = [
                "epoch",
                "train/box_loss",
                "train/cls_loss",
                "val/box_loss",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "lr/pg0",
                "time",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "train/box_loss": 2.0,
                        "train/cls_loss": 1.5,
                        "val/box_loss": 1.9,
                        "metrics/precision(B)": 0.4,
                        "metrics/recall(B)": 0.3,
                        "metrics/mAP50(B)": 0.35,
                        "metrics/mAP50-95(B)": 0.2,
                        "lr/pg0": 0.01,
                        "time": 10.0,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 2,
                        "train/box_loss": 1.5,
                        "train/cls_loss": 1.1,
                        "val/box_loss": 1.4,
                        "metrics/precision(B)": 0.6,
                        "metrics/recall(B)": 0.5,
                        "metrics/mAP50(B)": 0.55,
                        "metrics/mAP50-95(B)": 0.4,
                        "lr/pg0": 0.005,
                        "time": 22.0,
                    }
                )
            outputs = generate_training_report(run_dir)
            self.assertTrue(outputs["html"].is_file())
            self.assertTrue(outputs["metrics"].is_file())
            self.assertTrue(outputs["time"].is_file())
            ET.parse(outputs["metrics"])
            self.assertIn(
                "En iyi epoch",
                outputs["summary"].read_text(encoding="utf-8"),
            )

    def test_evaluation_and_prediction_reports(self) -> None:
        summary = {
            "head": "end2end",
            "overall": {
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.5,
                "fitness": 0.52,
            },
            "per_class": {
                name: {
                    "p": 0.8,
                    "r": 0.7,
                    "ap50": 0.75,
                    "map50_95": 0.5,
                }
                for name in ("person", "bike_motorcycle", "car")
            },
            "speed_ms_per_image": {
                "preprocess": 1.0,
                "inference": 4.0,
                "postprocess": 0.5,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "summary.json"
            comparison = json.loads(json.dumps(summary))
            comparison["head"] = "traditional_nms"
            comparison["overall"]["metrics/mAP50-95(B)"] = 0.55
            comparison_path = root / "comparison.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            outputs = generate_evaluation_report(
                [summary_path, comparison_path],
                root / "evaluation",
            )
            self.assertTrue(outputs["html"].is_file())
            self.assertTrue(outputs["metrics_end2end"].is_file())
            self.assertTrue(outputs["head_comparison"].is_file())
            ET.parse(outputs["head_comparison"])
            prediction = generate_prediction_report(
                {"person": 10, "bike_motorcycle": 2, "car": 15},
                3,
                root / "prediction",
            )
            self.assertTrue(prediction["chart"].is_file())
            self.assertIn(
                "bike_motorcycle",
                prediction["html"].read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
