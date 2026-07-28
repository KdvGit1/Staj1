"""testFoto klasorundeki *_T termal goruntuleri EDSR ile dogrudan 4x buyut.

Girdiler kucultulmez. Bellek kullanimini sinirlamak icin goruntu, genis bir
halo ile ortusen karolarda islenir ve yalnizca karolarin guvenli merkezleri
nihai ciktiya yazilir.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import EDSR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_thermal_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.lower().endswith("_t")
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EDSR, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})

    scale_factor = int(saved_args.get("scale_factor", 4))
    model = EDSR(
        scale_factor=scale_factor,
        num_channels=1,
        num_features=int(saved_args.get("num_features", 64)),
        num_residual_blocks=int(saved_args.get("num_residual_blocks", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scale_factor


@torch.inference_mode()
def upscale_tiled(
    model: EDSR,
    image: torch.Tensor,
    scale_factor: int,
    tile_size: int,
    halo: int,
) -> torch.Tensor:
    """[1, 1, H, W] tensorunu cozunurlugu dusurmeden karolarla buyut."""
    _, _, height, width = image.shape
    output = torch.empty(
        (1, 1, height * scale_factor, width * scale_factor),
        dtype=image.dtype,
        device="cpu",
    )

    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        extended_y0 = max(0, y0 - halo)
        extended_y1 = min(height, y1 + halo)

        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            extended_x0 = max(0, x0 - halo)
            extended_x1 = min(width, x1 + halo)

            tile = image[
                ...,
                extended_y0:extended_y1,
                extended_x0:extended_x1,
            ]
            prediction = model(tile).clamp_(0.0, 1.0).cpu()

            crop_y0 = (y0 - extended_y0) * scale_factor
            crop_y1 = crop_y0 + (y1 - y0) * scale_factor
            crop_x0 = (x0 - extended_x0) * scale_factor
            crop_x1 = crop_x0 + (x1 - x0) * scale_factor

            output[
                ...,
                y0 * scale_factor:y1 * scale_factor,
                x0 * scale_factor:x1 * scale_factor,
            ] = prediction[..., crop_y0:crop_y1, crop_x0:crop_x1]

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="testFoto icindeki *_T termal goruntuleri EDSR ile direkt 4x buyut."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("thermal database") / "testFoto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thermal database") / "testFoto_upscale_x4",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "best_model.pth",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=160,
        help="LR cekirdek karo boyutu; girdi cozunurlugunu degistirmez.",
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=40,
        help="Karo dikislerini engelleyen LR baglam genisligi.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU PyTorch thread sayisi.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Girdi klasoru bulunamadi: {args.input_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {args.checkpoint}")
    if args.tile_size <= 0:
        raise ValueError("--tile-size pozitif olmali")
    if args.halo < 36:
        raise ValueError("--halo, EDSR alici alani icin en az 36 olmali")

    images = find_thermal_images(args.input_dir)
    if not images:
        raise FileNotFoundError(f"*_T termal goruntu bulunamadi: {args.input_dir}")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale_factor = load_model(args.checkpoint, device)
    if scale_factor != 4:
        raise ValueError(f"Checkpoint x4 degil: scale_factor={scale_factor}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cihaz: {device}")
    print(f"CPU thread: {torch.get_num_threads()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Girdi: {args.input_dir}")
    print(f"Cikti: {args.output_dir}")
    print(f"Termal dosya: {len(images)}")
    print(f"Tile/halo: {args.tile_size}/{args.halo}")

    total_start = time.perf_counter()
    for index, image_path in enumerate(images, start=1):
        started = time.perf_counter()
        with Image.open(image_path) as source:
            source_gray = source.convert("L")
            source_array = np.asarray(source_gray, dtype=np.float32) / 255.0

        input_tensor = (
            torch.from_numpy(source_array)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        prediction = upscale_tiled(
            model=model,
            image=input_tensor,
            scale_factor=scale_factor,
            tile_size=args.tile_size,
            halo=args.halo,
        )

        output_array = (
            (prediction.squeeze(0).squeeze(0) * 255.0).round().byte().numpy()
        )
        output_path = args.output_dir / f"{image_path.stem}_EDSR_X4.png"
        Image.fromarray(output_array, mode="L").save(output_path)

        elapsed = time.perf_counter() - started
        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"{source_array.shape[1]}x{source_array.shape[0]} -> "
            f"{output_array.shape[1]}x{output_array.shape[0]} "
            f"({elapsed:.2f} sn)"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"Tamamlandi: {len(images)} dosya, toplam {total_elapsed:.2f} sn")


if __name__ == "__main__":
    main()
