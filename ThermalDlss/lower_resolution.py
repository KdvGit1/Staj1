"""
Thermal Görüntü Kalite Düşürme (Lower Resolution) Scripti
==========================================================
Yüksek çözünürlüklü (640×512) thermal görüntülerden düşük çözünürlüklü
versiyonlar oluşturur. Super-resolution model eğitimi için paired dataset
hazırlar.

Kullanım:
    python lower_resolution.py
    python lower_resolution.py --scales 2 4 --quality 95 --workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def degrade_image(
    image_path: str,
    output_path: str,
    scale_factor: int,
    jpeg_quality: int,
) -> bool:
    """Bir thermal görüntüyü bicubic interpolation ile küçültür.

    Args:
        image_path: Kaynak HR görüntü yolu.
        output_path: Hedef LR görüntü yolu.
        scale_factor: Küçültme faktörü (2 veya 4).
        jpeg_quality: JPEG kayıt kalitesi (1-100).

    Returns:
        True ise başarılı, False ise hatalı.
    """
    try:
        img = Image.open(image_path)

        # Grayscale olduğundan emin ol
        if img.mode != "L":
            imgmodeold = img.mode
            img = img.convert("L")
            print(f"Image converted to grayscale: {image_path},original mode was {imgmodeold}")

        w, h = img.size
        new_w = w // scale_factor
        new_h = h // scale_factor

        # Bicubic downscale — küçük boyutta bırak
        img_lr = img.resize((new_w, new_h), Image.BICUBIC)

        # Kaydet
        img_lr.save(output_path, "JPEG", quality=jpeg_quality)
        return True

    except Exception as e:
        print(f"[HATA] {image_path}: {e}", file=sys.stderr)
        return False


def _worker(args: tuple) -> bool:
    """Multiprocessing için wrapper fonksiyon."""
    return degrade_image(*args)


def process_split(
    input_dir: Path,
    output_dir: Path,
    scale_factor: int,
    jpeg_quality: int,
    num_workers: int,
) -> tuple[int, int]:
    """Bir split klasörünü (train/val/test) toplu olarak işler.

    Args:
        input_dir: Kaynak split klasörü.
        output_dir: Hedef split klasörü.
        scale_factor: Küçültme faktörü.
        jpeg_quality: JPEG kayıt kalitesi.
        num_workers: Paralel worker sayısı.

    Returns:
        (başarılı_sayısı, hatalı_sayısı) tuple'ı.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tüm jpg dosyalarını topla
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not image_files:
        print(f"  [UYARI] {input_dir} klasöründe görüntü bulunamadı.")
        return 0, 0

    # Worker argümanlarını hazırla
    tasks = [
        (str(img_path), str(output_dir / img_path.name), scale_factor, jpeg_quality)
        for img_path in image_files
    ]

    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}

        with tqdm(total=len(futures), desc=f"  {input_dir.name}", unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Thermal görüntüleri bicubic downscale ile küçültür (SR dataset hazırlığı)."
    )
    parser.add_argument(
        "--input_base",
        type=str,
        default=os.path.join("thermal database", "thermal_dataset_split"),
        help="Kaynak HR dataset dizini (default: thermal database/thermal_dataset_split)",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=os.path.join("thermal database", "thermal_dataset_degraded"),
        help="Çıktı LR dataset dizini (default: thermal database/thermal_dataset_degraded)",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Küçültme faktörleri (default: 2 4)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG kayıt kalitesi (default: 95)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Paralel worker sayısı (default: CPU sayısı)",
    )

    args = parser.parse_args()

    input_base = Path(args.input_base)
    output_base = Path(args.output_base)
    num_workers = args.workers or os.cpu_count()
    splits = ["train", "val", "test"]

    # Giriş dizini kontrolü
    if not input_base.exists():
        print(f"[HATA] Giriş dizini bulunamadı: {input_base}")
        sys.exit(1)

    print("=" * 60)
    print("Thermal Görüntü Kalite Düşürme (Lower Resolution)")
    print("=" * 60)
    print(f"  Kaynak:    {input_base.resolve()}")
    print(f"  Çıktı:     {output_base.resolve()}")
    print(f"  Ölçekler:  {args.scales}")
    print(f"  JPEG kal:  {args.quality}")
    print(f"  Workers:   {num_workers}")
    print("=" * 60)

    total_start = time.time()
    grand_success = 0
    grand_error = 0

    for scale in args.scales:
        scale_dir = output_base / f"x{scale}"
        target_w = 640 // scale
        target_h = 512 // scale

        print(f"\n{'─' * 60}")
        print(f"  x{scale} küçültme → {target_w}×{target_h}")
        print(f"{'─' * 60}")

        for split in splits:
            input_dir = input_base / split
            output_dir = scale_dir / split

            if not input_dir.exists():
                print(f"  [UYARI] {input_dir} bulunamadı, atlanıyor.")
                continue

            success, errors = process_split(
                input_dir, output_dir, scale, args.quality, num_workers
            )
            grand_success += success
            grand_error += errors

    elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("ÖZET")
    print(f"{'=' * 60}")
    print(f"  Toplam başarılı: {grand_success}")
    print(f"  Toplam hatalı:   {grand_error}")
    print(f"  Süre:            {elapsed:.1f} saniye")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
