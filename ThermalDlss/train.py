"""
Thermal Süper Çözünürlük — Eğitim Scripti
============================================
EDSR modelini thermal LR-HR çiftleriyle eğitir.

Özellikler:
  - Otomatik CUDA/CPU algılama
  - VRAM'e göre otomatik batch size belirleme
  - Ctrl+C ile durdurulduğunda otomatik checkpoint kaydetme
  - Checkpoint'tan eğitime devam etme (--resume)
  - PSNR/SSIM metrik takibi
  - Early stopping
  - CSV loglama
"""

import argparse
import csv
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from losses import ThermalSRLoss
from model import EDSR


# ──────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ──────────────────────────────────────────────


def get_device() -> torch.device:
    """CUDA varsa kullan, yoksa CPU'ya düş."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Cihaz:  {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("  Cihaz:  CPU (CUDA bulunamadı)")
    return device


def auto_batch_size(device: torch.device, patch_size: int, scale_factor: int) -> int:
    """VRAM'e göre otomatik batch size belirler.

    RTX 3060 12GB → batch 16
    8GB RAM CPU   → batch 4
    """
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 10:
            batch = 16
        elif vram_gb >= 6:
            batch = 8
        elif vram_gb >= 4:
            batch = 4
        else:
            batch = 2
    else:
        # CPU: Düşük RAM için güvenli değer
        batch = 4

    print(f"  Batch:  {batch} (otomatik belirlenmiş)")
    return batch


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR hesaplar (dB). Girdi: [0, 1] aralığında tensorlar."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> float:
    """Basit SSIM hesaplaması. Girdi: [B, 1, H, W], [0, 1] aralığında."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian benzeri ortalama — basit uniform pencere
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device)
    kernel = kernel / (window_size ** 2)

    mu_pred = nn.functional.conv2d(pred, kernel, padding=window_size // 2)
    mu_target = nn.functional.conv2d(target, kernel, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = nn.functional.conv2d(pred ** 2, kernel, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = nn.functional.conv2d(target ** 2, kernel, padding=window_size // 2) - mu_target_sq
    sigma_cross = nn.functional.conv2d(pred * target, kernel, padding=window_size // 2) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean().item()


# ──────────────────────────────────────────────
# Checkpoint Yönetimi
# ──────────────────────────────────────────────


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
):
    """Eğitim durumunu kaydeder."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_psnr": best_psnr,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None,
) -> dict:
    """Checkpoint'tan yükler."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ──────────────────────────────────────────────
# Eğitim ve Doğrulama
# ──────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: ThermalSRLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    check_interrupted=None,
) -> dict:
    """Bir epoch eğitim."""
    model.train()
    total_loss = 0.0
    total_pixel = 0.0
    total_edge = 0.0
    num_batches = 0

    for batch_idx, (lr, hr) in enumerate(dataloader):
        if check_interrupted and check_interrupted():
            print("    [!] Eğitim sırasında durdurma isteği alındı.")
            break
        lr = lr.to(device)
        hr = hr.to(device)

        # İleri geçiş
        pred = model(lr)
        loss, loss_dict = criterion(pred, hr)

        # Geri yayılım
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total"]
        total_pixel += loss_dict["pixel"]
        total_edge += loss_dict["edge"]
        num_batches += 1

        # Her 50 batch'te durum yazdır
        if (batch_idx + 1) % 50 == 0:
            print(
                f"    Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss_dict['total']:.4f} "
                f"(pixel={loss_dict['pixel']:.4f}, edge={loss_dict['edge']:.4f})"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "pixel": total_pixel / max(num_batches, 1),
        "edge": total_edge / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: ThermalSRLoss,
    device: torch.device,
    max_samples: int = 0,
    check_interrupted=None,
) -> dict:
    """Doğrulama seti üzerinde değerlendirme."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    total_batches = len(dataloader)

    print("  Doğrulama (Validation) hesaplanıyor...")

    for batch_idx, (lr, hr) in enumerate(dataloader):
        if check_interrupted and check_interrupted():
            print("    [!] Doğrulama sırasında durdurma isteği alındı.")
            break

        lr = lr.to(device)
        hr = hr.to(device)

        pred = model(lr)
        pred = torch.clamp(pred, 0.0, 1.0)

        loss, _ = criterion(pred, hr)
        batch_size_curr = lr.size(0)

        total_loss += loss.item() * batch_size_curr
        total_psnr += calculate_psnr(pred, hr) * batch_size_curr
        total_ssim += calculate_ssim(pred, hr) * batch_size_curr
        num_samples += batch_size_curr

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"    [Val {batch_idx + 1}/{total_batches}] PSNR={total_psnr / max(num_samples, 1):.2f}dB")

        if max_samples > 0 and num_samples >= max_samples:
            break

    n = max(num_samples, 1)
    return {
        "loss": total_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }


# ──────────────────────────────────────────────
# Ana Eğitim Döngüsü
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Thermal SR EDSR Eğitim")

    # Veri yolları
    parser.add_argument("--hr_base", type=str,
                        default=os.path.join("thermal database", "thermal_dataset_split"))
    parser.add_argument("--lr_base", type=str,
                        default=os.path.join("thermal database", "thermal_dataset_degraded", "x4"))

    # Model
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_residual_blocks", type=int, default=16)

    # Eğitim
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = otomatik belirle (VRAM'e göre)")
    parser.add_argument("--patch_size", type=int, default=48,
                        help="LR patch boyutu (HR patch = patch_size × scale)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--edge_weight", type=float, default=0.1,
                        help="Edge loss ağırlığı")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cupy", action="store_true",
                        help="CuPy GPU ivmelendirmesini kapatıp NumPy CPU moduna geç")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Devam edilecek checkpoint dosyası")

    # Early stopping & Val
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epoch)")
    parser.add_argument("--val_max_samples", type=int, default=300,
                        help="Doğrulamada kullanılacak maksimum görüntü sayısı (0 = tümü, varsayılan: 300)")

    args = parser.parse_args()

    # ── Cihaz ve batch size ──
    print("=" * 60)
    print("Thermal SR — EDSR Eğitim")
    print("=" * 60)

    device = get_device()

    if args.batch_size == 0:
        args.batch_size = auto_batch_size(device, args.patch_size, args.scale_factor)
    else:
        print(f"  Batch:  {args.batch_size} (kullanıcı belirlemiş)")

    # ── Model ──
    model = EDSR(
        scale_factor=args.scale_factor,
        num_channels=1,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
    ).to(device)
    print(f"  Model:  EDSR (parametre: {model.get_param_count():,})")

    # ── Loss, Optimizer, Scheduler ──
    criterion = ThermalSRLoss(edge_weight=args.edge_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Checkpoint'tan devam et ──
    start_epoch = 1
    best_psnr = 0.0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n  Checkpoint yükleniyor: {args.resume}")
            ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
            start_epoch = ckpt["epoch"] + 1
            best_psnr = ckpt.get("best_psnr", 0.0)
            print(f"  Epoch {ckpt['epoch']}'den devam ediliyor (best PSNR: {best_psnr:.2f} dB)")
        else:
            print(f"  [UYARI] Checkpoint bulunamadı: {args.resume}, sıfırdan başlanıyor.")

    # ── DataLoader ──
    print(f"\n  Veri yükleniyor...")
    train_loader, val_loader, _ = create_dataloaders(
        hr_base=args.hr_base,
        lr_base=args.lr_base,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cupy=not args.no_cupy,
    )

    if train_loader is None:
        print("[HATA] Train DataLoader oluşturulamadı!")
        sys.exit(1)

    # ── Checkpoint dizini ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── CSV log dosyası ──
    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    log_exists = os.path.exists(log_path) and args.resume
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch", "train_loss", "train_pixel", "train_edge",
            "val_loss", "val_psnr", "val_ssim", "lr",
        ])

    # ── Ctrl+C handler: Otomatik checkpoint kaydetme ──
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n  [!] Durdurma sinyali alındı, checkpoint kaydediliyor...")

    signal.signal(signal.SIGINT, signal_handler)

    def is_interrupted():
        return interrupted

    # ── Eğitim döngüsü ──
    patience_counter = 0
    epoch = start_epoch

    print(f"\n{'=' * 60}")
    print(f"  Eğitim başlıyor: epoch {start_epoch} → {args.epochs}")
    print(f"{'=' * 60}\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if interrupted:
                break

            epoch_start = time.time()

            # Eğitim
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, is_interrupted
            )

            # Eğitilen model durumunu anında kaydet
            last_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pth")
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_psnr, args)

            if interrupted:
                break

            # Doğrulama
            val_metrics = {"loss": 0, "psnr": 0, "ssim": 0}
            if val_loader:
                val_metrics = validate(
                    model, val_loader, criterion, device,
                    max_samples=args.val_max_samples,
                    check_interrupted=is_interrupted,
                )

            if interrupted:
                break

            # Scheduler adımı
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            elapsed = time.time() - epoch_start

            # Log yazdır
            print(
                f"  Epoch {epoch:03d}/{args.epochs} ({elapsed:.0f}s) │ "
                f"Train loss={train_metrics['loss']:.4f} │ "
                f"Val PSNR={val_metrics['psnr']:.2f}dB SSIM={val_metrics['ssim']:.4f} │ "
                f"LR={current_lr:.2e}"
            )

            # CSV log
            log_writer.writerow([
                epoch,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['pixel']:.6f}",
                f"{train_metrics['edge']:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['psnr']:.4f}",
                f"{val_metrics['ssim']:.6f}",
                f"{current_lr:.2e}",
            ])
            log_file.flush()

            # En iyi model kaydet
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                patience_counter = 0
                best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_psnr, args)
                print(f"    ★ Yeni en iyi model! PSNR={best_psnr:.2f}dB → {best_path}")
            else:
                patience_counter += 1

            # Her 10 epoch'ta checkpoint kaydet
            if epoch % 10 == 0:
                periodic_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
                save_checkpoint(periodic_path, model, optimizer, scheduler, epoch, best_psnr, args)

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n  [Early Stopping] {args.patience} epoch boyunca iyileşme yok, durduruluyor.")
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  [!] KeyboardInterrupt algılandı. Durduruluyor...")

    # ── Durdurma halinde otomatik checkpoint ──
    if interrupted:
        interrupt_path = os.path.join(args.checkpoint_dir, f"interrupted_epoch_{epoch:03d}.pth")
        save_checkpoint(interrupt_path, model, optimizer, scheduler, epoch, best_psnr, args)
        print(f"  Checkpoint kaydedildi: {interrupt_path}")
        print(f"  Devam etmek için: python train.py --resume {interrupt_path}")

    log_file.close()

    print(f"\n{'=' * 60}")
    print(f"  Eğitim tamamlandı!")
    print(f"  En iyi PSNR: {best_psnr:.2f} dB")
    print(f"  Checkpoint dizini: {args.checkpoint_dir}")
    print(f"  Log dosyası: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
