"""
Thermal Süper Çözünürlük — Değerlendirme Scripti
===================================================
Eğitilmiş modeli test seti üzerinde değerlendirir.
PSNR/SSIM metrikleri ve görsel karşılaştırmalar üretir.
"""

import argparse
import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from model import EDSR
from train import calculate_psnr, calculate_ssim


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def _pil_to_tensor(img: Image.Image, device: torch.device = None) -> torch.Tensor:
    """PIL grayscale Image → [1, H, W] float tensor [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    if HAS_CUPY and device is not None and device.type == "cuda":
        gpu_arr = cp.ascontiguousarray(cp.asarray(arr)[cp.newaxis, ...])
        return torch.as_tensor(gpu_arr, device=device)
    else:
        return torch.from_numpy(arr).unsqueeze(0)


def _get_font(size: int = 14):
    """Sistemde mevcut fontu yükler veya varsayılan fontu döndürür."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                return ImageFont.load_default()


def evaluate_model(

    model: nn.Module,
    hr_dir: str,
    lr_dir: str,
    device: torch.device,
    output_dir: str | None = None,
    max_samples: int = 0,
    num_save_images: int = 50,
    seed: int = 42,
    checkpoint_path: str = "",
    scale_factor: int = 4,
    enable_16x: bool = True,
):
    """Test seti üzerinde modeli değerlendirir (4x ve isteğe bağlı 16x Cascade).

    Args:
        model: Eğitilmiş EDSR modeli.
        hr_dir: HR test görüntü dizini.
        lr_dir: LR test görüntü dizini.
        device: Hesaplama cihazı.
        output_dir: Karşılaştırma görüntülerinin kaydedileceği dizin.
        max_samples: Değerlendirilecek maksimum görüntü sayısı (0 = tümü).
        num_save_images: Kaydedilecek rastgele karşılaştırma görüntüsü sayısı (varsayılan: 50).
        seed: Rastgele görüntü seçimi için seed.
        checkpoint_path: Yüklenen checkpoint dosya yolu.
        scale_factor: Büyütme katsayısı (örn: 4).
        enable_16x: 16x Cascade (EDSR(EDSR(LR))) değerlendirmesini çalıştırır.
    """
    from pathlib import Path

    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)

    # Eşleşen dosyaları bul
    lr_files = {f.name for f in lr_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    hr_files = {f.name for f in hr_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    common = sorted(lr_files & hr_files)

    if max_samples > 0:
        common = common[:max_samples]

    if not common:
        print("[HATA] Eşleşen dosya bulunamadı!")
        return

    standalone_16x_dir = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if enable_16x:
            standalone_16x_dir = os.path.join(output_dir, "16x_standalone")
            os.makedirs(standalone_16x_dir, exist_ok=True)

    # Rastgele kaydedilecek görüntü indekslerini seç (Dataset genelinden rastgele)
    if output_dir and num_save_images > 0:
        num_to_save = min(num_save_images, len(common))
        rng = random.Random(seed)
        save_indices = set(rng.sample(range(len(common)), num_to_save))
        print(f"  [BİLGİ] Toplam {len(common)} test görüntüsünden rastgele seçilen {num_to_save} tanesi '{output_dir}' dizinine kaydedilecek.")
        if enable_16x:
            print(f"  [BİLGİ] Tekil 16x büyütülmüş görseller '{standalone_16x_dir}' klasörüne kaydedilecek.")
    else:
        save_indices = set()

    model.eval()
    total_psnr_4x = 0.0
    total_ssim_4x = 0.0
    bicubic_psnr_4x = 0.0
    bicubic_ssim_4x = 0.0

    total_psnr_16x_rel = 0.0
    total_ssim_16x_rel = 0.0

    results_list = []

    print(f"\n  {len(common)} görüntü değerlendiriliyor...")
    if enable_16x:
        print("  [16x CASCADE] EDSR(EDSR(LR)) 16x büyütme aktif (Bicubic 16x ile kıyaslamalı).")

    for i, fname in enumerate(common):
        # Yükle
        lr_img = Image.open(lr_path / fname).convert("L")
        hr_img = Image.open(hr_path / fname).convert("L")

        lr_tensor = _pil_to_tensor(lr_img).unsqueeze(0).to(device)  # [1, 1, H, W]
        hr_tensor = _pil_to_tensor(hr_img).unsqueeze(0).to(device)

        # Bicubic 4x baseline
        bicubic_img_4x = lr_img.resize(hr_img.size, Image.BICUBIC)
        bicubic_tensor_4x = _pil_to_tensor(bicubic_img_4x).unsqueeze(0).to(device)

        # Model tahmini (4x Pas 1)
        with torch.no_grad():
            pred_4x = model(lr_tensor)
            pred_4x = torch.clamp(pred_4x, 0.0, 1.0)

            # 16x Cascade tahmini (Pas 2: EDSR(EDSR(LR)))
            pred_16x = None
            psnr_16x_rel = 0.0
            ssim_16x_rel = 0.0
            bic_16x_psnr_rel = 0.0
            bic_16x_ssim_rel = 0.0

            if enable_16x:
                pred_16x = model(pred_4x)
                pred_16x = torch.clamp(pred_16x, 0.0, 1.0)

                # 16x Bicubic baseline (LR -> 16x)
                bic_16x_size = (lr_img.width * 16, lr_img.height * 16)
                bicubic_img_16x = lr_img.resize(bic_16x_size, Image.BICUBIC)
                bicubic_tensor_16x = _pil_to_tensor(bicubic_img_16x).unsqueeze(0).to(device)

                # 16x Relatif Metrikler (EDSR 16x vs Bicubic 16x)
                psnr_16x_rel = calculate_psnr(pred_16x, bicubic_tensor_16x)
                ssim_16x_rel = calculate_ssim(pred_16x, bicubic_tensor_16x)

                total_psnr_16x_rel += psnr_16x_rel
                total_ssim_16x_rel += ssim_16x_rel

        # 4x Metrikler
        psnr_val_4x = calculate_psnr(pred_4x, hr_tensor)
        ssim_val_4x = calculate_ssim(pred_4x, hr_tensor)
        bic_psnr_4x = calculate_psnr(bicubic_tensor_4x, hr_tensor)
        bic_ssim_4x = calculate_ssim(bicubic_tensor_4x, hr_tensor)

        total_psnr_4x += psnr_val_4x
        total_ssim_4x += ssim_val_4x
        bicubic_psnr_4x += bic_psnr_4x
        bicubic_ssim_4x += bic_ssim_4x

        sample_res = {
            "index": i,
            "filename": fname,
            "bic_psnr_4x": bic_psnr_4x,
            "bic_ssim_4x": bic_ssim_4x,
            "psnr_val_4x": psnr_val_4x,
            "ssim_val_4x": ssim_val_4x,
        }
        if enable_16x:
            sample_res["psnr_16x_rel"] = psnr_16x_rel
            sample_res["ssim_16x_rel"] = ssim_16x_rel

        results_list.append(sample_res)

        # Karşılaştırma görüntüsü & Tekil 16x Görsel Kaydı
        if output_dir and i in save_indices:
            # 1. 5-Kolonlu Karşılaştırma Görseli Kaydet
            _save_comparison(
                lr_img=lr_img,
                bicubic_img=bicubic_img_4x,
                pred_tensor_4x=pred_4x,
                hr_tensor=hr_tensor,
                save_path=os.path.join(output_dir, f"compare_{i:04d}_{fname}"),
                psnr_model=psnr_val_4x,
                ssim_model=ssim_val_4x,
                psnr_bicubic=bic_psnr_4x,
                ssim_bicubic=bic_ssim_4x,
                pred_tensor_16x=pred_16x if enable_16x else None,
                psnr_16x_rel=psnr_16x_rel,
                ssim_16x_rel=ssim_16x_rel,
                fname=fname,
                sample_idx=i,
            )

            # 2. Tekil 16x Görsel Kaydet (Ham 2560x1920 boyutta)
            if enable_16x and pred_16x is not None:
                pred_16x_np = pred_16x.squeeze().cpu().numpy()
                pred_16x_img = Image.fromarray((pred_16x_np * 255).clip(0, 255).astype("uint8"), mode="L")
                standalone_path = os.path.join(standalone_16x_dir, f"16x_{i:04d}_{fname}")
                pred_16x_img.save(standalone_path)

        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(common)}] 4x PSNR={psnr_val_4x:.2f}dB" + (f" | 16x Rel={psnr_16x_rel:.2f}dB" if enable_16x else ""))

    n = len(common)
    print(f"\n{'=' * 60}")
    print(f"  Sonuçlar ({n} görüntü)")
    print(f"{'=' * 60}")
    print(f"  {'':25s} {'PSNR (dB)':>12s} {'SSIM':>10s}")
    print(f"  {'---------------------------------------------------'}")

    print(f"  {'Bicubic 4x (baseline)':25s} {bicubic_psnr_4x / n:12.2f} {bicubic_ssim_4x / n:10.4f}")
    print(f"  {'EDSR 4x (model)':25s} {total_psnr_4x / n:12.2f} {total_ssim_4x / n:10.4f}")
    print(f"  {'4x Fark':25s} {(total_psnr_4x - bicubic_psnr_4x) / n:+12.2f} {(total_ssim_4x - bicubic_ssim_4x) / n:+10.4f}")

    if enable_16x:
        print(f"  {'---------------------------------------------------'}")
        print(f"  {'EDSR 16x vs Bic16x (Rel)':25s} {total_psnr_16x_rel / n:12.2f} {total_ssim_16x_rel / n:10.4f}")
    print(f"{'=' * 60}")

    if output_dir:
        print(f"  Karşılaştırma görüntüleri kaydedildi ({len(save_indices)} adet): {output_dir}")
        if enable_16x:
            print(f"  Tekil 16x büyütülmüş ham görseller kaydedildi: {standalone_16x_dir}")
        _save_evaluation_logs_and_plots(
            results_list=results_list,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            scale_factor=scale_factor,
            enable_16x=enable_16x,
        )


def _save_evaluation_logs_and_plots(
    results_list: list[dict],
    output_dir: str,
    checkpoint_path: str,
    scale_factor: int,
    enable_16x: bool = True,
):
    """Değerlendirme metriklerini CSV, JSON, TXT dosyalarına kaydeder ve açıklayıcı grafikler çizer."""
    import csv
    import json

    if not output_dir or not results_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Detaylı CSV Logu
    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
    fieldnames = [
        "index", "filename",
        "bicubic_4x_psnr", "bicubic_4x_ssim",
        "model_4x_psnr", "model_4x_ssim",
        "psnr_gain_4x", "ssim_gain_4x"
    ]
    if enable_16x:
        fieldnames.extend(["psnr_16x_rel", "ssim_16x_rel"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_list:
            row = {
                "index": r["index"],
                "filename": r["filename"],
                "bicubic_4x_psnr": round(r["bic_psnr_4x"], 4),
                "bicubic_4x_ssim": round(r["bic_ssim_4x"], 6),
                "model_4x_psnr": round(r["psnr_val_4x"], 4),
                "model_4x_ssim": round(r["ssim_val_4x"], 6),
                "psnr_gain_4x": round(r["psnr_val_4x"] - r["bic_psnr_4x"], 4),
                "ssim_gain_4x": round(r["ssim_val_4x"] - r["bic_ssim_4x"], 6),
            }
            if enable_16x:
                row["psnr_16x_rel"] = round(r.get("psnr_16x_rel", 0.0), 4)
                row["ssim_16x_rel"] = round(r.get("ssim_16x_rel", 0.0), 6)
            writer.writerow(row)
    print(f"  [LOG] Detaylı metrik tablosu kaydedildi: {csv_path}")

    # 2. Özet JSON & TXT Raporu
    n = len(results_list)
    avg_bic_psnr_4x = sum(r["bic_psnr_4x"] for r in results_list) / n
    avg_bic_ssim_4x = sum(r["bic_ssim_4x"] for r in results_list) / n
    avg_mod_psnr_4x = sum(r["psnr_val_4x"] for r in results_list) / n
    avg_mod_ssim_4x = sum(r["ssim_val_4x"] for r in results_list) / n
    avg_gain_psnr_4x = avg_mod_psnr_4x - avg_bic_psnr_4x
    avg_gain_ssim_4x = avg_mod_ssim_4x - avg_bic_ssim_4x

    best_sample_4x = max(results_list, key=lambda x: x["psnr_val_4x"] - x["bic_psnr_4x"])

    summary_data = {
        "total_test_samples": n,
        "checkpoint": os.path.basename(checkpoint_path) if checkpoint_path else "Unknown",
        "scale_factor_model": scale_factor,
        "enable_16x_cascade": enable_16x,
        "metrics_4x": {
            "bicubic": {"mean_psnr_db": round(avg_bic_psnr_4x, 4), "mean_ssim": round(avg_bic_ssim_4x, 6)},
            "model_edsr": {"mean_psnr_db": round(avg_mod_psnr_4x, 4), "mean_ssim": round(avg_mod_ssim_4x, 6)},
            "gain": {"psnr_gain_db": round(avg_gain_psnr_4x, 4), "ssim_gain": round(avg_gain_ssim_4x, 6)}
        },
        "best_improvement_sample_4x": {
            "filename": best_sample_4x["filename"],
            "psnr_gain": round(best_sample_4x["psnr_val_4x"] - best_sample_4x["bic_psnr_4x"], 4)
        }
    }

    if enable_16x:
        avg_16x_psnr_rel = sum(r.get("psnr_16x_rel", 0.0) for r in results_list) / n
        avg_16x_ssim_rel = sum(r.get("ssim_16x_rel", 0.0) for r in results_list) / n
        summary_data["metrics_16x_relative_to_bicubic16x"] = {
            "mean_psnr_db": round(avg_16x_psnr_rel, 4),
            "mean_ssim": round(avg_16x_ssim_rel, 6)
        }

    json_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(" THERMAL SUPER-RESOLUTION DEĞERLENDİRME ÖZET RAPORU\n")
        f.write("=" * 65 + "\n")
        f.write(f" Değerlendirilen Görsel Sayısı: {n}\n")
        f.write(f" Checkpoint: {checkpoint_path}\n")
        f.write(f" Model Büyütme Faktörü: x{scale_factor}\n")
        f.write(f" 16x Cascade Büyütme Modu: {'Aktif' if enable_16x else 'Pasif'}\n")
        f.write("-" * 65 + "\n")
        f.write(" 4x DEĞERLENDİRME (Native HR ile):\n")
        f.write(f"   Bicubic 4x Baseline PSNR: {avg_bic_psnr_4x:.2f} dB  | SSIM: {avg_bic_ssim_4x:.4f}\n")
        f.write(f"   EDSR 4x Model PSNR      : {avg_mod_psnr_4x:.2f} dB  | SSIM: {avg_mod_ssim_4x:.4f}\n")
        f.write(f"   4x Ortalama Kazanım     : {avg_gain_psnr_4x:+.2f} dB | SSIM: {avg_gain_ssim_4x:+.4f}\n")
        f.write(f"   En Çok İyileşen Örnek: {best_sample_4x['filename']} (+{best_sample_4x['psnr_val_4x'] - best_sample_4x['bic_psnr_4x']:.2f} dB)\n")

        if enable_16x:
            f.write("-" * 65 + "\n")
            f.write(" 16x CASCADE DEĞERLENDİRME (16x Bicubic Baseline ile Relatif):\n")
            f.write(f"   EDSR 16x vs Bicubic 16x PSNR: {avg_16x_psnr_rel:.2f} dB  | SSIM: {avg_16x_ssim_rel:.4f}\n")
        f.write("=" * 65 + "\n")
    print(f"  [LOG] Özet raporlar kaydedildi: {txt_path}, {json_path}")

    # 3. Grafik Çizimleri (Matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Grafik 1: Bicubic vs EDSR Metrik Karşılaştırma Bar Grafiği (4x)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        methods = ["Bicubic 4x\n(Baseline)", "EDSR 4x\n(Model)"]
        psnr_vals = [avg_bic_psnr_4x, avg_mod_psnr_4x]
        ssim_vals = [avg_bic_ssim_4x, avg_mod_ssim_4x]
        colors = ["#7f8c8d", "#2ecc71"]

        bars1 = ax1.bar(methods, psnr_vals, color=colors, width=0.45)
        ax1.set_title("4x Ortalama PSNR Karşılaştırması (dB)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_ylim(bottom=max(0, min(psnr_vals) - 2), top=max(psnr_vals) + 2.5)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f} dB", ha="center", va="bottom", fontweight="bold")
        ax1.axhline(avg_bic_psnr_4x, color="gray", linestyle="--", alpha=0.7)

        bars2 = ax2.bar(methods, ssim_vals, color=["#95a5a6", "#3498db"], width=0.45)
        ax2.set_title("4x Ortalama SSIM Karşılaştırması", fontsize=12, fontweight="bold")
        ax2.set_ylabel("SSIM")
        ax2.set_ylim(bottom=max(0, min(ssim_vals) - 0.05), top=min(1.0, max(ssim_vals) + 0.06))
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
        ax2.axhline(avg_bic_ssim_4x, color="gray", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "metrics_comparison_bar.png")
        plt.savefig(plot1_path, dpi=200)
        plt.close()

        # Grafik 2: Örnek Bazlı PSNR Dağılım Çizgi Grafiği (4x)
        fig, ax = plt.subplots(figsize=(12, 5))
        indices = [r["index"] for r in results_list]
        mod_psnrs = [r["psnr_val_4x"] for r in results_list]
        bic_psnrs = [r["bic_psnr_4x"] for r in results_list]

        ax.plot(indices, mod_psnrs, label="EDSR 4x Model", color="#2ecc71", linewidth=1.5)
        ax.plot(indices, bic_psnrs, label="Bicubic 4x Baseline", color="#e74c3c", linestyle="--", linewidth=1.2)
        ax.fill_between(indices, bic_psnrs, mod_psnrs, where=[m >= b for m, b in zip(mod_psnrs, bic_psnrs)],
                        color="#2ecc71", alpha=0.25, label="4x Model PSNR Kazancı (+dB)")

        ax.set_title("Test Örnekleri Bazında 4x PSNR Karşılaştırması (dB)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Örnek İndeksi")
        ax.set_ylabel("PSNR (dB)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "psnr_distribution_line.png")
        plt.savefig(plot2_path, dpi=200)
        plt.close()

        # Grafik 3: PSNR İyileşme (Gain) Dağılım Histogramı
        fig, ax = plt.subplots(figsize=(10, 5))
        gains = [r["psnr_val_4x"] - r["bic_psnr_4x"] for r in results_list]
        ax.hist(gains, bins=min(30, max(5, n // 2)), color="#3498db", edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="0 dB (Fark Yok)")
        ax.axvline(avg_gain_psnr_4x, color="green", linestyle="-", linewidth=2.0, label=f"Ortalama Kazanç (+{avg_gain_psnr_4x:.2f} dB)")

        ax.set_title("4x Model PSNR Kazanım Dağılımı (Histogram)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PSNR Kazancı (dB)")
        ax.set_ylabel("Örnek Sayısı")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, "psnr_gain_histogram.png")
        plt.savefig(plot3_path, dpi=200)
        plt.close()

        print(f"  [GRAFİK] Açıklayıcı değerlendirme grafikleri kaydedildi:")
        print(f"    - {plot1_path}")
        print(f"    - {plot2_path}")
        print(f"    - {plot3_path}")

    except Exception as e:
        print(f"  [UYARI] Grafik oluşturulurken bir hata oluştu: {e}")


def _save_comparison(
    lr_img: Image.Image,
    bicubic_img: Image.Image,
    pred_tensor_4x: torch.Tensor,
    hr_tensor: torch.Tensor,
    save_path: str,
    psnr_model: float,
    ssim_model: float,
    psnr_bicubic: float,
    ssim_bicubic: float,
    pred_tensor_16x: torch.Tensor | None = None,
    psnr_16x_rel: float = 0.0,
    ssim_16x_rel: float = 0.0,
    fname: str = "",
    sample_idx: int = 0,
):
    """5'li (veya 4'lü) karşılaştırma görüntüsü kaydeder:
    LR | Bicubic 4x | EDSR 4x Model | HR Orijinal | EDSR 16x Cascade
    Görüntülerin üzerine açıklayıcı başlıklar ve metrikler ekler.
    """
    # Model çıktısını PIL Image'a dönüştür (4x)
    pred_np_4x = pred_tensor_4x.squeeze().cpu().numpy()
    pred_img_4x = Image.fromarray((pred_np_4x * 255).clip(0, 255).astype("uint8"), mode="L")

    hr_np = hr_tensor.squeeze().cpu().numpy()
    hr_img = Image.fromarray((hr_np * 255).clip(0, 255).astype("uint8"), mode="L")

    # LR'yi HR boyutuna büyüt (görsel karşılaştırma için, nearest)
    lr_display = lr_img.resize(hr_img.size, Image.NEAREST)

    w, h = hr_img.size
    top_bar_h = 24
    col_bar_h = 42
    header_h = top_bar_h + col_bar_h

    has_16x = pred_tensor_16x is not None
    num_panels = 5 if has_16x else 4

    # Tuval oluştur (RGB formatında renkli yazılar için)
    canvas = Image.new("RGB", (w * num_panels, h + header_h), color=(20, 20, 28))

    # Görüntüleri yapıştır (header altında kalacak şekilde)
    canvas.paste(lr_display.convert("RGB"), (0, header_h))
    canvas.paste(bicubic_img.convert("RGB"), (w, header_h))
    canvas.paste(pred_img_4x.convert("RGB"), (w * 2, header_h))
    canvas.paste(hr_img.convert("RGB"), (w * 3, header_h))

    if has_16x:
        pred_np_16x = pred_tensor_16x.squeeze().cpu().numpy()
        pred_img_16x = Image.fromarray((pred_np_16x * 255).clip(0, 255).astype("uint8"), mode="L")
        # 16x çıktıyı tuval paneli boyutuna (w, h) sığacak şekilde yeniden boyutlandırıp yapıştır
        pred_img_16x_panel = pred_img_16x.resize((w, h), Image.BICUBIC)
        canvas.paste(pred_img_16x_panel.convert("RGB"), (w * 4, header_h))

    draw = ImageDraw.Draw(canvas)
    title_font = _get_font(15)
    sub_font = _get_font(12)

    # 1. Üst Bilgi Çubuğu (Dosya Adı & İndeks)
    draw.rectangle([0, 0, w * num_panels, top_bar_h], fill=(15, 15, 22))
    info_text = f"  GÖRÜNTÜ KARŞILAŞTIRMASI ({num_panels} KOLON)  |  Dosya: {fname}  |  Test Örnek Index: {sample_idx}"
    draw.text((10, 4), info_text, fill=(220, 220, 240), font=sub_font)

    # 2. Kolon Başlıkları & Metrikler
    panels = [
        {
            "title": "LR (Düşük Çözünürlük)",
            "subtitle": f"Giriş ({lr_img.width}x{lr_img.height})",
            "color": (255, 190, 40),  # Sarı/Turuncu
        },
        {
            "title": "Bicubic 4x (Baseline)",
            "subtitle": f"PSNR: {psnr_bicubic:.2f} dB  |  SSIM: {ssim_bicubic:.4f}",
            "color": (210, 210, 210),  # Açık Gri
        },
        {
            "title": "EDSR 4x (Model Çıktısı)",
            "subtitle": f"PSNR: {psnr_model:.2f} dB  |  SSIM: {ssim_model:.4f}",
            "color": (50, 230, 120),  # Canlı Yeşil
        },
        {
            "title": "HR (Gerçek / Hedef)",
            "subtitle": f"Orijinal ({w}x{h})",
            "color": (90, 180, 255),  # Açık Mavi
        },
    ]

    if has_16x:
        panels.append({
            "title": "EDSR 16x (Cascade Çıktı)",
            "subtitle": f"Çıktı ({lr_img.width*16}x{lr_img.height*16}) | Rel PSNR: {psnr_16x_rel:.2f}dB",
            "color": (255, 110, 210),  # Pembe/Magenta
        })

    for idx, panel in enumerate(panels):
        x_start = idx * w
        # Kolon başlık arka planı
        draw.rectangle([x_start, top_bar_h, x_start + w, header_h], fill=(30, 32, 44))

        # Paneller arası dikey ayraç çizgisi
        draw.line([(x_start, 0), (x_start, h + header_h)], fill=(60, 65, 80), width=2)

        # Metinleri çiz
        draw.text((x_start + 10, top_bar_h + 3), panel["title"], fill=panel["color"], font=title_font)
        draw.text((x_start + 10, top_bar_h + 22), panel["subtitle"], fill=(180, 185, 200), font=sub_font)

    canvas.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Thermal SR Model Değerlendirme")

    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "best_model.pth"),
                        help="Model checkpoint dosyası")
    parser.add_argument("--hr_dir", type=str,
                        default=os.path.join("thermal database", "thermal_dataset_split", "test"))
    parser.add_argument("--lr_dir", type=str,
                        default=os.path.join("thermal database", "thermal_dataset_degraded", "x4", "test"))
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Karşılaştırma görüntülerinin kaydedileceği dizin")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Değerlendirilecek maksimum görüntü sayısı (0 = tümü)")
    parser.add_argument("--num_save_images", type=int, default=50,
                        help="Rastgele kaydedilecek karşılaştırma görüntüsü sayısı (varsayılan: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgele fotoğraf seçimi için seed (varsayılan: 42)")
    parser.add_argument("--enable_16x", type=lambda x: str(x).lower() not in ("false", "0", "no"),
                        default=True, help="16x Cascade (EDSR(EDSR(LR))) değerlendirmesini çalıştırır (varsayılan: True)")

    # Model parametreleri (checkpoint'taki argümanlardan otomatik alınır)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_residual_blocks", type=int, default=16)

    args = parser.parse_args()

    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Cihaz: {device}")

    # Model oluştur
    model = EDSR(
        scale_factor=args.scale_factor,
        num_channels=1,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
    ).to(device)

    # Checkpoint yükle
    if not os.path.exists(args.checkpoint):
        print(f"[HATA] Checkpoint bulunamadı: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Checkpoint'taki args'ı kullan (varsa)
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        epoch = checkpoint.get("epoch", "?")
        best_psnr = checkpoint.get("best_psnr", 0)
        print(f"  Checkpoint: Epoch {epoch}, Best PSNR={best_psnr:.2f}dB")

    print(f"  Model: EDSR ({model.get_param_count():,} parametre)")

    # Değerlendir
    evaluate_model(
        model=model,
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        device=device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_save_images=args.num_save_images,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        scale_factor=args.scale_factor,
        enable_16x=args.enable_16x,
    )


if __name__ == "__main__":
    main()


