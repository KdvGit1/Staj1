"""
Thermal Süper Çözünürlük — Dataset (CuPy GPU Destekli)
======================================================
LR-HR eşleştirilmiş termal görüntü veri seti.
Patch-based eğitim, CuPy GPU tabanlı data augmentation (flip + rotate) destekler.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class ThermalSRDataset(Dataset):
    """LR-HR eşleştirilmiş thermal görüntü veri seti.

    Dosya adlarına göre eşleştirme yapar:
        HR: thermal_dataset_split/{split}/abc.jpg
        LR: thermal_dataset_degraded/x{scale}/{split}/abc.jpg

    Args:
        hr_dir: Yüksek çözünürlüklü görüntü dizini.
        lr_dir: Düşük çözünürlüklü görüntü dizini.
        patch_size: Eğitimde kullanılacak LR patch boyutu (None = tam görüntü).
        scale_factor: Büyütme faktörü (HR patch = LR patch × scale).
        augment: Data augmentation uygulansın mı.
        use_cupy: GPU ivmelendirmesi için CuPy kullanılsın mı.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        patch_size: int | None = 48,
        scale_factor: int = 4,
        augment: bool = True,
        use_cupy: bool = True,
    ):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.use_cupy = use_cupy and HAS_CUPY and torch.cuda.is_available()

        # Dosya listesini oluştur ve eşleştir
        lr_files = {f.name for f in self.lr_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
        hr_files = {f.name for f in self.hr_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}

        # Kesişim — her iki dizinde de bulunan dosyalar
        common_files = sorted(lr_files & hr_files)

        if not common_files:
            raise FileNotFoundError(
                f"LR ({self.lr_dir}) ve HR ({self.hr_dir}) dizinlerinde "
                f"eşleşen dosya bulunamadı."
            )

        self.filenames = common_files

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL grayscale Image → [1, H, W] float tensor [0, 1]."""
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _pil_to_cupy(img: Image.Image):
        """PIL grayscale Image → [1, H, W] CuPy float32 GPU array [0, 1]."""
        arr = np.array(img, dtype=np.float32) / 255.0
        gpu_arr = cp.asarray(arr)
        return gpu_arr[cp.newaxis, ...]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]

        # Görüntüleri yükle (grayscale)
        lr_img = Image.open(self.lr_dir / filename).convert("L")
        hr_img = Image.open(self.hr_dir / filename).convert("L")

        if self.use_cupy:
            # CuPy ile GPU VRAM üzerinde ön işleme ve augmentation
            lr_arr = self._pil_to_cupy(lr_img)
            hr_arr = self._pil_to_cupy(hr_img)

            if self.patch_size is not None:
                lr_arr, hr_arr = self._random_crop_cupy(lr_arr, hr_arr)

            if self.augment:
                lr_arr, hr_arr = self._augment_cupy(lr_arr, hr_arr)

            # CuPy GPU array -> PyTorch CUDA Tensor (Bitişik bellek düzeni sağlama)
            lr_arr = cp.ascontiguousarray(lr_arr)
            hr_arr = cp.ascontiguousarray(hr_arr)

            lr_tensor = torch.as_tensor(lr_arr, device="cuda")
            hr_tensor = torch.as_tensor(hr_arr, device="cuda")
            return lr_tensor, hr_tensor
        else:
            # Orijinal CPU NumPy / PyTorch işleme
            lr_tensor = self._pil_to_tensor(lr_img)
            hr_tensor = self._pil_to_tensor(hr_img)

            if self.patch_size is not None:
                lr_tensor, hr_tensor = self._random_crop(lr_tensor, hr_tensor)

            if self.augment:
                lr_tensor, hr_tensor = self._augment(lr_tensor, hr_tensor)

            return lr_tensor, hr_tensor

    def _random_crop_cupy(self, lr, hr):
        """CuPy GPU matrisleri üzerinde patch kırpar."""
        _, lr_h, lr_w = lr.shape
        ps = self.patch_size

        top = random.randint(0, lr_h - ps)
        left = random.randint(0, lr_w - ps)

        lr_patch = lr[:, top : top + ps, left : left + ps]

        hr_top = top * self.scale_factor
        hr_left = left * self.scale_factor
        hr_ps = ps * self.scale_factor
        hr_patch = hr[:, hr_top : hr_top + hr_ps, hr_left : hr_left + hr_ps]

        return lr_patch, hr_patch

    def _augment_cupy(self, lr, hr):
        """CuPy GPU matrisleri üzerinde flip ve rotasyon uygular."""
        if random.random() > 0.5:
            lr = cp.flip(lr, axis=2)
            hr = cp.flip(hr, axis=2)

        if random.random() > 0.5:
            lr = cp.flip(lr, axis=1)
            hr = cp.flip(hr, axis=1)

        if random.random() > 0.5:
            lr = cp.rot90(lr, 1, (1, 2))
            hr = cp.rot90(hr, 1, (1, 2))

        return lr, hr

    def _random_crop(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LR'den rastgele patch kırpar, HR'den karşılık gelen büyük patch'i alır."""
        _, lr_h, lr_w = lr.shape
        ps = self.patch_size

        top = random.randint(0, lr_h - ps)
        left = random.randint(0, lr_w - ps)

        lr_patch = lr[:, top : top + ps, left : left + ps]

        hr_top = top * self.scale_factor
        hr_left = left * self.scale_factor
        hr_ps = ps * self.scale_factor
        hr_patch = hr[:, hr_top : hr_top + hr_ps, hr_left : hr_left + hr_ps]

        return lr_patch, hr_patch

    def _augment(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rastgele flip ve 90° dönüşüm uygular."""
        if random.random() > 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])

        if random.random() > 0.5:
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])

        if random.random() > 0.5:
            lr = torch.rot90(lr, 1, [1, 2])
            hr = torch.rot90(hr, 1, [1, 2])

        return lr, hr


def create_dataloaders(
    hr_base: str,
    lr_base: str,
    scale_factor: int = 4,
    patch_size: int = 48,
    batch_size: int = 16,
    num_workers: int = 4,
    use_cupy: bool = True,
) -> tuple:
    """Train, val, test DataLoader'larını oluşturur."""
    from torch.utils.data import DataLoader

    loaders = {}
    use_cupy_active = use_cupy and HAS_CUPY and torch.cuda.is_available()

    for split in ["train", "val", "test"]:
        hr_dir = os.path.join(hr_base, split)
        lr_dir = os.path.join(lr_base, split)

        if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
            print(f"[UYARI] {split} dizini bulunamadı, atlanıyor.")
            loaders[split] = None
            continue

        is_train = split == "train"
        dataset = ThermalSRDataset(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            patch_size=patch_size if is_train else None,
            scale_factor=scale_factor,
            augment=is_train,
            use_cupy=use_cupy_active,
        )

        split_workers = 0 if os.name == "nt" else num_workers
        split_batch = batch_size if is_train else max(1, batch_size // 4)

        # CuPy aktifken tensörler zaten CUDA'da üretildiği için pin_memory False kalabilir
        loaders[split] = DataLoader(
            dataset,
            batch_size=split_batch,
            shuffle=is_train,
            num_workers=split_workers,
            pin_memory=not use_cupy_active if torch.cuda.is_available() else False,
            drop_last=is_train,
        )

        mode_str = "CuPy (GPU)" if dataset.use_cupy else "NumPy (CPU)"
        print(f"  {split}: {len(dataset)} görüntü, batch_size={split_batch}, mod={mode_str}")

    return loaders.get("train"), loaders.get("val"), loaders.get("test")

