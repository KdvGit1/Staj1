"""Sabit test mozaiklerinde tiled EDSR, bicubic ve artefakt değerlendirmesi."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .backend import resolve_backends
from .bootstrap import PROJECT_ROOT, ensure_project_root
from .data import MosaicSRDataset
from .metrics import artifact_metrics, masked_psnr_numpy
from .modeling import build_model
from .mosaic_io import BICUBIC

ensure_project_root()

import torch
from train import calculate_psnr, calculate_ssim
from upscale_testfoto_x4 import upscale_tiled


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sabit 4×4 pseudo-HR test mozaiklerini değerlendir"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "mosaic_system"
        / "runs"
        / "native_x4"
        / "best_model.pth",
    )
    parser.add_argument(
        "--hr-base",
        type=Path,
        default=PROJECT_ROOT
        / "thermal database"
        / "thermal_dataset_split",
        help="train/val/test HR split'lerinin ana dizini.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Bilimsel pseudo-HR evaluation split'i (varsayılan: val).",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        default=None,
        help="İleri kullanım: --hr-base/--split yerine doğrudan HR dizini.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "mosaic_system"
        / "runs"
        / "native_x4"
        / "evaluation",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0=tümü")
    parser.add_argument("--save-previews", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--halo", type=int, default=40)
    parser.add_argument("--seam-margin-lr", type=int, default=4)
    parser.add_argument("--skip-ssim", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--preprocess-backend",
        choices=["auto", "cpu", "cupy"],
        default="cpu",
    )
    return parser.parse_args(argv)


def _tensor_image(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).view(
        1, 1, *array.shape
    )


def _bicubic_from_lr(lr: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((lr * 255.0).round().clip(0, 255).astype(np.uint8), "L")
    return np.asarray(image.resize(output_size, BICUBIC), dtype=np.float32) / 255.0


def _save_preview(
    path: Path,
    hr: np.ndarray,
    bicubic: np.ndarray,
    pred: np.ndarray,
    *,
    group_id: str,
    psnr_bicubic: float,
    psnr_model: float,
) -> None:
    target_size = (640, 512)
    panels = []
    for array in (hr, bicubic, pred):
        panel = Image.fromarray(
            (array * 255.0).round().clip(0, 255).astype(np.uint8), "L"
        ).resize(target_size, BICUBIC)
        panels.append(panel.convert("RGB"))
    header = 48
    canvas = Image.new("RGB", (target_size[0] * 3, target_size[1] + header), "#151821")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * target_size[0], header))
    draw = ImageDraw.Draw(canvas)
    labels = (
        f"Pseudo-HR hedef | {group_id}",
        f"Bicubic | {psnr_bicubic:.3f} dB",
        f"EDSR | {psnr_model:.3f} dB",
    )
    for index, label in enumerate(labels):
        draw.text((index * target_size[0] + 10, 15), label, fill="white")
    canvas.save(path)


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
    return float(sum(values) / len(values)) if values else float("nan")


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {args.checkpoint}")
    if args.halo < 36:
        raise ValueError("--halo en az 36 olmalı")

    backend = resolve_backends(
        device_request=args.device,
        preprocess_request=args.preprocess_backend,
    )
    model, checkpoint = build_model(
        device=backend.device,
        checkpoint_path=args.checkpoint,
        post_shuffle_relu=None,
    )
    model.eval()
    scale = int(model.mosaic_config["scale_factor"])
    if scale != 4:
        raise ValueError(f"Checkpoint ×4 değil: {scale}")

    hr_dir = args.hr_dir or args.hr_base / args.split
    dataset = MosaicSRDataset(
        hr_dir,
        patch_size=None,
        seed=args.seed,
        augment=False,
        seam_mode="mask",
        seam_margin_lr=args.seam_margin_lr,
        preprocess_backend=backend.preprocess_backend,
        cache_mode="memory",
    )
    total = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "bicubic_x4_pseudo_hr_mosaic",
        "split": args.split if args.hr_dir is None else "custom",
        "source_hr_dir": str(hr_dir.resolve()),
        "tiles": 16,
        "grid": "4x4",
        "native_tile_size": [640, 512],
        "target_pseudo_hr_size": [2560, 2048],
        "input_generation": "PIL bicubic downsample x4 from the same pseudo-HR target",
        "input_lr_size": [640, 512],
        "model_mapping": "640x512 -> 2560x2048",
        "baseline": "bicubic upsample x4 from the same 640x512 LR input",
        "reference_metrics": ["PSNR", "SSIM"],
        "native_testFoto_used": False,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preview_dir = args.output_dir / "previews"
    if args.save_previews > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    manifest_handle = (args.output_dir / "evaluated_manifest.jsonl").open(
        "w", encoding="utf-8"
    )
    started = time.perf_counter()
    try:
        for index in range(total):
            lr_tensor, hr_tensor, mask_tensor, _ = dataset[index]
            group = dataset.groups[index]
            lr_batch = lr_tensor.unsqueeze(0).to(backend.device)
            hr_batch = hr_tensor.unsqueeze(0).cpu()
            mask = mask_tensor.squeeze().cpu().numpy()

            pred_batch = upscale_tiled(
                model=model,
                image=lr_batch,
                scale_factor=scale,
                tile_size=args.tile_size,
                halo=args.halo,
            ).clamp(0.0, 1.0)

            lr = lr_tensor.squeeze().detach().cpu().numpy()
            hr = hr_tensor.squeeze().detach().cpu().numpy()
            if lr.shape != (512, 640) or hr.shape != (2048, 2560):
                raise AssertionError(
                    "Evaluation protokol boyutu bozuldu: "
                    f"LR={lr.shape}, HR={hr.shape}; beklenen "
                    "LR=(512, 640), HR=(2048, 2560)"
                )
            pred = pred_batch.squeeze().numpy()
            bicubic = _bicubic_from_lr(lr, (hr.shape[1], hr.shape[0]))
            bicubic_batch = _tensor_image(bicubic)

            psnr_model = calculate_psnr(pred_batch, hr_batch)
            psnr_bicubic = calculate_psnr(bicubic_batch, hr_batch)
            ssim_model = (
                float("nan")
                if args.skip_ssim
                else calculate_ssim(pred_batch, hr_batch)
            )
            ssim_bicubic = (
                float("nan")
                if args.skip_ssim
                else calculate_ssim(bicubic_batch, hr_batch)
            )
            model_art = artifact_metrics(pred)
            bicubic_art = artifact_metrics(bicubic)
            row: dict[str, float | str | int] = {
                "index": index,
                "group_id": group.group_id,
                "split": protocol["split"],
                "distinct_videos": group.distinct_video_count,
                "psnr_model": psnr_model,
                "psnr_bicubic": psnr_bicubic,
                "psnr_gain": psnr_model - psnr_bicubic,
                "ssim_model": ssim_model,
                "ssim_bicubic": ssim_bicubic,
                "ssim_gain": ssim_model - ssim_bicubic,
                "psnr_model_interior": masked_psnr_numpy(pred, hr, mask),
                "psnr_model_seam": masked_psnr_numpy(pred, hr, 1.0 - mask),
            }
            for key, value in model_art.items():
                row[f"model_{key}"] = value
            for key, value in bicubic_art.items():
                row[f"bicubic_{key}"] = value
            rows.append(row)  # type: ignore[arg-type]

            manifest_handle.write(
                json.dumps(
                    {
                        "group_id": group.group_id,
                        "signature": group.signature,
                        "distinct_videos": group.distinct_video_count,
                        "sources": [path.name for path in group.image_paths],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index < args.save_previews:
                _save_preview(
                    preview_dir / f"{index:04d}_{group.group_id}.png",
                    hr,
                    bicubic,
                    pred,
                    group_id=group.group_id,
                    psnr_bicubic=psnr_bicubic,
                    psnr_model=psnr_model,
                )
            print(
                f"[{index + 1}/{total}] {group.group_id}: "
                f"EDSR={psnr_model:.3f} bicubic={psnr_bicubic:.3f} "
                f"gain={psnr_model - psnr_bicubic:+.3f} dB"
            )
    finally:
        manifest_handle.close()
        dataset.close()

    fieldnames = list(rows[0].keys()) if rows else []
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": protocol,
        "backend": backend.description,
        "samples": len(rows),
        "seconds": time.perf_counter() - started,
        "psnr_model": _json_number(_mean(rows, "psnr_model")),
        "psnr_bicubic": _json_number(_mean(rows, "psnr_bicubic")),
        "psnr_gain": _json_number(_mean(rows, "psnr_gain")),
        "ssim_model": _json_number(_mean(rows, "ssim_model")),
        "ssim_bicubic": _json_number(_mean(rows, "ssim_bicubic")),
        "ssim_gain": _json_number(_mean(rows, "ssim_gain")),
        "model_clip_ratio": _json_number(_mean(rows, "model_clip_ratio")),
        "bicubic_clip_ratio": _json_number(_mean(rows, "bicubic_clip_ratio")),
        "model_phase_mean_std": _json_number(
            _mean(rows, "model_phase_mean_std")
        ),
        "bicubic_phase_mean_std": _json_number(
            _mean(rows, "bicubic_phase_mean_std")
        ),
        "model_gradient_x": _json_number(_mean(rows, "model_gradient_x")),
        "bicubic_gradient_x": _json_number(_mean(rows, "bicubic_gradient_x")),
        "checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
