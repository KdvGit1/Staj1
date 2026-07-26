"""Generate understandable SVG/HTML reports from existing project outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermal_detection.reporting import (
    generate_dataset_report,
    generate_evaluation_report,
    generate_training_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "yolo_thermal" / "manifest.json",
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "dataset_verification"
        / "verification_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "graphs",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional Ultralytics run directory containing results.csv.",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        action="append",
        default=[],
        help="Evaluation summary JSON; repeat for end2end/NMS comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.manifest.is_file():
        generated = generate_dataset_report(
            args.manifest.resolve(),
            args.verification.resolve() if args.verification.is_file() else None,
            output / "dataset",
        )
        print(f"Dataset graph report: {generated['html']}")
    if args.run_dir:
        generated = generate_training_report(args.run_dir.resolve())
        print(f"Training graph report: {generated['html']}")
    if args.evaluation_summary:
        generated = generate_evaluation_report(
            [path.resolve() for path in args.evaluation_summary],
            output / "evaluation",
        )
        print(f"Evaluation graph report: {generated['html']}")


if __name__ == "__main__":
    main()

