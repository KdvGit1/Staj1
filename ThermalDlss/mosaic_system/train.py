"""Mevcut EDSR ağırlıklarından 16'lı pseudo-HR mixed-replay fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np

from .backend import resolve_backends
from .bootstrap import PROJECT_ROOT, ensure_project_root
from .data import (
    DeterministicMixedDataset,
    MosaicSRDataset,
    PairedReplayDataset,
    create_dataloader,
)
from .losses_ext import MaskedThermalSRLoss
from .modeling import build_model, load_checkpoint_payload, save_checkpoint

ensure_project_root()

import torch
import torch.optim as optim
from train import calculate_psnr, calculate_ssim


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ThermalDlss 4×4 pseudo-HR EDSR fine-tuning"
    )
    parser.add_argument(
        "--hr-base",
        type=Path,
        default=PROJECT_ROOT / "thermal database" / "thermal_dataset_split",
    )
    parser.add_argument(
        "--lr-base",
        type=Path,
        default=PROJECT_ROOT
        / "thermal database"
        / "thermal_dataset_degraded"
        / "x4",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pth",
        help="Yalnız model ağırlıkları yüklenir; optimizer yeniden kurulur.",
    )
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Mosaic-system checkpoint'ından optimizer dahil devam et.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "mosaic_system" / "runs" / "native_x4",
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=0, help="0=otomatik")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--edge-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Tekrarlanabilir deney için deterministik PyTorch/CUDA davranışını "
            "etkinleştirir (bir miktar yavaşlatabilir)."
        ),
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-max-samples", type=int, default=97)
    parser.add_argument("--samples-per-epoch", type=int, default=0)

    parser.add_argument("--paired-ratio", type=float, default=0.70)
    parser.add_argument("--mosaic-ratio", type=float, default=0.30)
    parser.add_argument(
        "--seam-mode",
        choices=["avoid", "mask", "include"],
        default="avoid",
    )
    parser.add_argument("--seam-margin-lr", type=int, default=4)
    parser.add_argument(
        "--cache-mode",
        choices=["memory", "rolling_disk"],
        default="memory",
    )
    parser.add_argument("--cache-size", type=int, default=8)

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--preprocess-backend",
        choices=["auto", "cpu", "cupy"],
        default="auto",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--no-post-shuffle-relu",
        action="store_true",
        help="Mimari ablation: PixelShuffle sonrası ReLU'ları Identity yap.",
    )
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-residual-blocks", type=int, default=16)
    return parser.parse_args(argv)


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _auto_batch_size(device: torch.device, patch_size: int) -> int:
    if device.type == "cpu":
        return 1 if patch_size >= 96 else 2
    vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
    base = 16 if vram >= 10 else 8 if vram >= 6 else 4
    scaled = int(base * (48.0 / patch_size) ** 2)
    return max(1, scaled)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def _move_batch(batch, device: torch.device):
    lr, hr, mask, source = batch
    non_blocking = device.type == "cuda"
    return (
        lr.to(device, non_blocking=non_blocking),
        hr.to(device, non_blocking=non_blocking),
        mask.to(device, non_blocking=non_blocking),
        source,
    )


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    *,
    amp_enabled: bool,
    accumulation: int,
    gradient_clip: float,
    interrupted,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "pixel": 0.0, "edge": 0.0, "batches": 0.0}

    for batch_index, batch in enumerate(loader):
        if interrupted():
            break
        lr, hr, mask, _ = _move_batch(batch, device)
        with _autocast(amp_enabled):
            pred = model(lr)
            loss, details = criterion(pred, hr, mask)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()

        should_step = (
            (batch_index + 1) % accumulation == 0
            or (batch_index + 1) == len(loader)
        )
        if should_step:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += details["total"]
        totals["pixel"] += details["pixel"]
        totals["edge"] += details["edge"]
        totals["batches"] += 1
        if (batch_index + 1) % 50 == 0:
            print(
                f"    train {batch_index + 1}/{len(loader)} "
                f"loss={details['total']:.5f}"
            )

    count = max(totals.pop("batches"), 1.0)
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def validate(
    model,
    loader,
    criterion,
    device,
    *,
    amp_enabled: bool,
    max_samples: int,
    interrupted,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "samples": 0.0}
    for batch in loader:
        if interrupted():
            break
        lr, hr, mask, _ = _move_batch(batch, device)
        with _autocast(amp_enabled):
            pred = model(lr).clamp(0.0, 1.0)
            loss, _ = criterion(pred, hr, mask)
        batch_size = lr.shape[0]
        totals["loss"] += float(loss.item()) * batch_size
        totals["psnr"] += calculate_psnr(pred.float(), hr.float()) * batch_size
        totals["ssim"] += calculate_ssim(pred.float(), hr.float()) * batch_size
        totals["samples"] += batch_size
        if max_samples > 0 and totals["samples"] >= max_samples:
            break
    count = max(totals.pop("samples"), 1.0)
    return {key: value / count for key, value in totals.items()}


def _jsonable_config(args: argparse.Namespace, backend_description: str) -> dict:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["backend_description"] = backend_description
    config["scale_factor"] = 4
    return config


EpochCallback = Callable[[int, dict[str, float], dict[str, float]], None]


def run(
    args: argparse.Namespace,
    *,
    epoch_callback: EpochCallback | None = None,
) -> int:
    if args.from_scratch and args.resume:
        raise ValueError("--from-scratch ile --resume birlikte kullanılamaz")
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation pozitif olmalı")
    if args.paired_ratio <= 0 or args.mosaic_ratio <= 0:
        raise ValueError("paired/mosaic oranları pozitif olmalı")

    backend = resolve_backends(
        device_request=args.device,
        preprocess_request=args.preprocess_backend,
    )
    _seed_everything(args.seed, deterministic=args.deterministic)
    if args.batch_size == 0:
        args.batch_size = _auto_batch_size(backend.device, args.patch_size)
    amp_enabled = bool(backend.device.type == "cuda" and not args.no_amp)

    weight_path = None
    if args.resume:
        weight_path = args.resume
    elif not args.from_scratch:
        weight_path = args.pretrained
    if weight_path and not Path(weight_path).is_file():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {weight_path}")

    model, initial_checkpoint = build_model(
        device=backend.device,
        checkpoint_path=weight_path,
        scale_factor=4,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
        post_shuffle_relu=not args.no_post_shuffle_relu,
    )
    criterion = MaskedThermalSRLoss(args.edge_weight).to(backend.device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = _make_scaler(amp_enabled)
    start_epoch, best_psnr = 0, -math.inf

    if args.resume:
        resume_payload = initial_checkpoint or load_checkpoint_payload(
            args.resume, backend.device
        )
        if not resume_payload.get("mosaic_system"):
            raise ValueError(
                "--resume yalnız mosaic_system checkpoint'ı kabul eder; "
                "eski checkpoint için --pretrained kullanın"
            )
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict"):
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_psnr = float(resume_payload.get("best_psnr", -math.inf))

    cupy_active = backend.preprocess_backend == "cupy"
    train_mosaic = MosaicSRDataset(
        args.hr_base / "train",
        patch_size=args.patch_size,
        seed=args.seed,
        augment=True,
        seam_mode=args.seam_mode,
        seam_margin_lr=args.seam_margin_lr,
        preprocess_backend=backend.preprocess_backend,
        cache_mode=args.cache_mode,
        cache_size=args.cache_size,
        cache_dir=args.output_dir.parent / ".cache" / "train",
    )
    train_paired = PairedReplayDataset(
        args.hr_base / "train",
        args.lr_base / "train",
        patch_size=args.patch_size,
        scale_factor=4,
        seed=args.seed,
        augment=True,
        use_cupy=cupy_active,
    )
    mixed = DeterministicMixedDataset(
        [train_paired, train_mosaic],
        [args.paired_ratio, args.mosaic_ratio],
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
        sequential_source_ids=(1,) if args.cache_mode == "rolling_disk" else (),
    )

    val_patch = min(args.patch_size, 96)
    val_mosaic = MosaicSRDataset(
        args.hr_base / "val",
        patch_size=val_patch,
        seed=args.seed + 10_000,
        augment=False,
        seam_mode="mask",
        seam_margin_lr=args.seam_margin_lr,
        preprocess_backend=backend.preprocess_backend,
        cache_mode="memory",
    )
    train_loader = create_dataloader(
        mixed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        cuda_active=backend.cuda_available,
        cupy_active=cupy_active,
        rolling_cache_active=args.cache_mode == "rolling_disk",
    )
    val_loader = create_dataloader(
        val_mosaic,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        cuda_active=backend.cuda_available,
        cupy_active=cupy_active,
        rolling_cache_active=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _jsonable_config(args, backend.description)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_path = args.output_dir / "training_log.csv"
    append_log = args.resume is not None and log_path.exists()
    log_handle = log_path.open("a" if append_log else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_handle)
    if not append_log:
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_pixel",
                "train_edge",
                "val_loss",
                "val_psnr",
                "val_ssim",
                "learning_rate",
                "seconds",
            ]
        )

    stop_requested = False

    def signal_handler(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nDurdurma istendi; güvenli checkpoint epoch sonunda yazılacak.")

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 72)
    print("ThermalDlss Mosaic Fine-tuning")
    print(f"Backend: {backend.description}; AMP={amp_enabled}")
    print(f"Model: {model.get_param_count():,} parametre")
    print(f"Patch/batch/accum: {args.patch_size}/{args.batch_size}/"
          f"{args.gradient_accumulation}")
    print(f"Epoch örnekleri: {len(mixed)} {mixed.source_counts()}")
    print(f"Train mozaikleri: {train_mosaic.planner.summary(0)}")
    print(f"Çıktı: {args.output_dir}")
    print("=" * 72)

    patience_counter = 0
    last_epoch = max(start_epoch - 1, 0)
    try:
        for epoch in range(start_epoch, args.epochs):
            if stop_requested:
                break
            mixed.set_epoch(epoch)
            val_mosaic.set_epoch(0)  # Validation manifesti sabit tutulur.
            started = time.perf_counter()
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                backend.device,
                amp_enabled=amp_enabled,
                accumulation=args.gradient_accumulation,
                gradient_clip=args.gradient_clip,
                interrupted=lambda: stop_requested,
            )
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                backend.device,
                amp_enabled=amp_enabled,
                max_samples=args.val_max_samples,
                interrupted=lambda: stop_requested,
            )
            scheduler.step()
            elapsed = time.perf_counter() - started
            last_epoch = epoch
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{train_metrics['loss']:.7f}",
                    f"{train_metrics['pixel']:.7f}",
                    f"{train_metrics['edge']:.7f}",
                    f"{val_metrics['loss']:.7f}",
                    f"{val_metrics['psnr']:.5f}",
                    f"{val_metrics['ssim']:.7f}",
                    f"{current_lr:.3e}",
                    f"{elapsed:.2f}",
                ]
            )
            log_handle.flush()
            save_checkpoint(
                args.output_dir / "last_checkpoint.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_psnr=best_psnr,
                config=config,
            )
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                patience_counter = 0
                save_checkpoint(
                    args.output_dir / "best_model.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    config=config,
                )
            else:
                patience_counter += 1
            print(
                f"Epoch {epoch:03d}: train={train_metrics['loss']:.5f} "
                f"val={val_metrics['psnr']:.3f}dB/"
                f"{val_metrics['ssim']:.5f} {elapsed:.1f}s"
            )
            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, val_metrics)
            if patience_counter >= args.patience:
                print(f"Early stopping: {args.patience} epoch iyileşme yok.")
                break
    finally:
        if stop_requested:
            save_checkpoint(
                args.output_dir / f"interrupted_epoch_{last_epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=last_epoch,
                best_psnr=best_psnr,
                config=config,
            )
        log_handle.close()
        mixed.close()
        val_mosaic.close()

    print(f"Tamamlandı. En iyi validation PSNR: {best_psnr:.4f} dB")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
