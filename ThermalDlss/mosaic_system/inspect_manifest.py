"""Dataset'i kopyalamadan manifest boyut ve benzersizlik raporu üret."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import PROJECT_ROOT
from .manifest import MosaicManifestPlanner, discover_images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mozaik manifestini denetle")
    parser.add_argument(
        "--hr-base",
        type=Path,
        default=PROJECT_ROOT / "thermal database" / "thermal_dataset_split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        planner = MosaicManifestPlanner(
            discover_images(args.hr_base / split),
            seed=args.seed,
        )
        epoch_signatures: list[set[str]] = []
        for epoch in range(args.epochs):
            groups = planner.build_epoch(epoch)
            signatures = {group.signature for group in groups}
            if len(signatures) != len(groups):
                raise AssertionError(f"{split} epoch {epoch}: tekrar eden grup")
            if any(not signatures.isdisjoint(previous) for previous in epoch_signatures):
                raise AssertionError(f"{split} epoch {epoch}: önceki epoch ile tekrar")
            epoch_signatures.append(signatures)
        report[split] = {
            **planner.summary(0),
            "checked_epochs": args.epochs,
            "cross_epoch_duplicate_groups": 0,
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

