"""Train YOLO26n on the 640x512 three-class thermal dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermal_detection.array_backend import select_array_backend
from thermal_detection.model_utils import (
    load_dataset_config,
    parse_batch_size,
    validate_training_device,
    verify_first_image_tensor_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def generate_run_report(save_dir: Path) -> None:
    try:
        from thermal_detection.reporting import generate_training_report

        generated = generate_training_report(save_dir)
        print(f"Training graph report: {generated['html']}")
    except Exception as exc:
        print(f"WARNING: Training completed but graph report failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "yolo_thermal" / "dataset.yaml",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=parse_batch_size, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "thermal_detection",
    )
    parser.add_argument("--name", default="yolo26n_640x512")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "numpy"),
        default="auto",
        help="Backend for auxiliary preflight operations; training uses PyTorch.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 epochs on 5%% of training data before a full run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run from an Ultralytics last.pt checkpoint.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse the requested run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    load_dataset_config(data_path)
    validate_training_device(args.device)
    first_image = verify_first_image_tensor_contract(data_path)
    backend = select_array_backend(args.backend)
    print(f"Auxiliary array backend: {backend.name} ({backend.detail})")
    print(
        "Thermal channel contract passed: "
        f"{first_image} -> [3, 512, 640] with equal channels"
    )

    from ultralytics import YOLO, __version__ as ultralytics_version

    print(f"Ultralytics version: {ultralytics_version}")
    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(resume=True, device=args.device)
        print(f"Resumed training output: {results.save_dir}")
        generate_run_report(Path(results.save_dir))
        return

    epochs = 2 if args.smoke_test else args.epochs
    fraction = min(args.fraction, 0.05) if args.smoke_test else args.fraction
    name = f"{args.name}_smoke" if args.smoke_test else args.name
    cache: bool | str = False if args.cache == "false" else args.cache

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        imgsz=640,
        rect=True,
        epochs=epochs,
        patience=2 if args.smoke_test else args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=True,
        fraction=fraction,
        cache=cache,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=min(10, max(1, epochs // 10)),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=0.0,
        translate=0.08,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.4,
        mixup=0.0,
        cutmix=0.0,
        cls_pw=0.25,
        max_det=300,
        val=True,
        plots=True,
        save=True,
        save_period=args.save_period,
        project=str(args.project.resolve()),
        name=name,
        exist_ok=args.exist_ok,
        verbose=True,
    )
    print(f"Training complete: {results.save_dir}")
    print(f"Best checkpoint: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    generate_run_report(Path(results.save_dir))


if __name__ == "__main__":
    main()
