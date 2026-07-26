"""Report PyTorch CUDA, Ultralytics and CuPy/NumPy readiness."""

from __future__ import annotations

import argparse
import platform
import sys

from thermal_detection.array_backend import select_array_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "numpy"),
        default="auto",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with an error if PyTorch CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import ultralytics

        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics: NOT INSTALLED")

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for device_id in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(device_id)
                memory_gib = properties.total_memory / (1024**3)
                print(
                    f"  [{device_id}] {properties.name}, "
                    f"{memory_gib:.1f} GiB VRAM"
                )
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    backend = select_array_backend(args.backend)
    print(f"Auxiliary array backend: {backend.name}")
    print(f"Backend detail: {backend.detail}")
    if args.require_cuda and not cuda_available:
        raise SystemExit(
            "CUDA is required for full training, but the installed PyTorch "
            "cannot access an NVIDIA GPU."
        )


if __name__ == "__main__":
    main()

