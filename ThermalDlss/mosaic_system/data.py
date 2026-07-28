"""Mozaik, mevcut paired replay ve deterministik karma eğitim datasetleri."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from .backend import cupy_module
from .bootstrap import ensure_project_root
from .manifest import MosaicGroup, MosaicManifestPlanner, discover_images
from .mosaic_io import RollingMosaicCache, build_mosaic_pair

ensure_project_root()

import torch
from torch.utils.data import DataLoader, Dataset
from dataset import ThermalSRDataset


SOURCE_PAIRED = 0
SOURCE_MOSAIC = 1


def _stable_seed(base_seed: int, epoch: int, index: int, namespace: str) -> int:
    payload = f"{base_seed}:{epoch}:{index}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _uint8_to_tensor(array: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array, dtype=np.float32) / 255.0
    return torch.from_numpy(contiguous).unsqueeze(0)


def _seam_mask(
    *,
    crop_top_lr: int,
    crop_left_lr: int,
    patch_h_lr: int,
    patch_w_lr: int,
    scale_factor: int,
    seam_margin_lr: int,
    full_lr_shape: tuple[int, int],
) -> np.ndarray:
    """HR kayıp maskesinde mozaik dikişlerinin çevresini sıfırla."""
    hr_h, hr_w = patch_h_lr * scale_factor, patch_w_lr * scale_factor
    mask = np.ones((hr_h, hr_w), dtype=np.float32)
    full_lr_h, full_lr_w = full_lr_shape
    vertical = [full_lr_w * i // 4 for i in range(1, 4)]
    horizontal = [full_lr_h * i // 4 for i in range(1, 4)]
    margin_hr = seam_margin_lr * scale_factor

    for seam_lr in vertical:
        local_hr = (seam_lr - crop_left_lr) * scale_factor
        start = max(0, local_hr - margin_hr)
        end = min(hr_w, local_hr + margin_hr)
        if start < end:
            mask[:, start:end] = 0.0
    for seam_lr in horizontal:
        local_hr = (seam_lr - crop_top_lr) * scale_factor
        start = max(0, local_hr - margin_hr)
        end = min(hr_h, local_hr + margin_hr)
        if start < end:
            mask[start:end, :] = 0.0
    return mask


class MosaicSRDataset(Dataset):
    """16 native HR görüntüden geçici 4×4 pseudo-HR örnekleri üretir."""

    def __init__(
        self,
        hr_dir: Path | str,
        *,
        patch_size: int | None = 96,
        scale_factor: int = 4,
        seed: int = 42,
        augment: bool = True,
        seam_mode: str = "avoid",
        seam_margin_lr: int = 4,
        preprocess_backend: str = "cpu",
        cache_mode: str = "memory",
        cache_size: int = 8,
        cache_dir: Path | str | None = None,
    ):
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = int(scale_factor)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.seam_mode = seam_mode.lower()
        self.seam_margin_lr = int(seam_margin_lr)
        self.preprocess_backend = preprocess_backend.lower()
        self.cache_mode = cache_mode.lower()
        self.cache_size = int(cache_size)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.scale_factor != 4:
            raise ValueError("Bu sistem 4×4 mozaik için scale_factor=4 bekler")
        if self.seam_mode not in {"avoid", "mask", "include"}:
            raise ValueError("seam_mode: avoid, mask veya include olmalı")
        if self.preprocess_backend not in {"cpu", "cupy"}:
            raise ValueError("preprocess_backend: cpu veya cupy olmalı")
        if self.cache_mode not in {"memory", "rolling_disk"}:
            raise ValueError("cache_mode: memory veya rolling_disk olmalı")

        self.image_paths = discover_images(self.hr_dir)
        self.planner = MosaicManifestPlanner(self.image_paths, seed=self.seed)
        self.epoch = -1
        self.groups: tuple[MosaicGroup, ...] = ()
        self._cache: RollingMosaicCache | None = None
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.groups)

    def _create_cache(self) -> None:
        if self.cache_mode != "rolling_disk":
            return
        if self.cache_dir is None:
            raise ValueError("rolling_disk için cache_dir verilmelidir")
        self._cache = RollingMosaicCache(
            self.cache_dir,
            capacity=self.cache_size,
            total_items=len(self.groups),
        )

    def set_epoch(self, epoch: int) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        self.epoch = int(epoch)
        self.groups = self.planner.build_epoch(self.epoch)
        self._create_cache()

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def _build_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return build_mosaic_pair(
            self.groups[index].image_paths,
            grid_size=4,
            tile_size=(640, 512),
            scale_factor=self.scale_factor,
        )

    def _get_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is None:
            return self._build_pair(index)
        return self._cache.get(index, self._build_pair)

    def _crop(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lr_h, lr_w = lr.shape
        if self.patch_size is None:
            return lr, hr, _seam_mask(
                crop_top_lr=0,
                crop_left_lr=0,
                patch_h_lr=lr_h,
                patch_w_lr=lr_w,
                scale_factor=self.scale_factor,
                seam_margin_lr=self.seam_margin_lr
                if self.seam_mode == "mask"
                else 0,
                full_lr_shape=(lr_h, lr_w),
            )

        ps = int(self.patch_size)
        if ps <= 0 or ps > lr_h or ps > lr_w:
            raise ValueError(f"Geçersiz patch_size={ps}; LR boyutu={lr_w}×{lr_h}")
        rng = random.Random(_stable_seed(self.seed, self.epoch, index, "crop"))

        if self.seam_mode == "avoid":
            tile_w_lr, tile_h_lr = lr_w // 4, lr_h // 4
            margin = self.seam_margin_lr
            min_x, min_y = margin, margin
            max_x = tile_w_lr - margin - ps
            max_y = tile_h_lr - margin - ps
            if max_x < min_x or max_y < min_y:
                raise ValueError(
                    f"patch_size={ps}, seam_margin_lr={margin} ile "
                    f"{tile_w_lr}×{tile_h_lr} LR karo içine sığmıyor. "
                    "Patch'i/marjı küçültin veya seam_mode=mask kullanın."
                )
            tile_row = rng.randrange(4)
            tile_col = rng.randrange(4)
            left = tile_col * tile_w_lr + rng.randint(min_x, max_x)
            top = tile_row * tile_h_lr + rng.randint(min_y, max_y)
        else:
            left = rng.randint(0, lr_w - ps)
            top = rng.randint(0, lr_h - ps)

        hr_top, hr_left = top * self.scale_factor, left * self.scale_factor
        hr_ps = ps * self.scale_factor
        lr_patch = lr[top : top + ps, left : left + ps]
        hr_patch = hr[hr_top : hr_top + hr_ps, hr_left : hr_left + hr_ps]
        mask = (
            _seam_mask(
                crop_top_lr=top,
                crop_left_lr=left,
                patch_h_lr=ps,
                patch_w_lr=ps,
                scale_factor=self.scale_factor,
                seam_margin_lr=self.seam_margin_lr,
                full_lr_shape=(lr_h, lr_w),
            )
            if self.seam_mode == "mask"
            else np.ones_like(hr_patch, dtype=np.float32)
        )
        return lr_patch, hr_patch, mask

    def _augment_numpy(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        mask: np.ndarray,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return lr, hr, mask
        rng = random.Random(_stable_seed(self.seed, self.epoch, index, "augment"))
        if rng.random() < 0.5:
            lr, hr, mask = np.flip(lr, 1), np.flip(hr, 1), np.flip(mask, 1)
        if rng.random() < 0.5:
            lr, hr, mask = np.flip(lr, 0), np.flip(hr, 0), np.flip(mask, 0)
        if lr.shape[0] == lr.shape[1] and rng.random() < 0.5:
            lr, hr, mask = (
                np.rot90(lr, 1),
                np.rot90(hr, 1),
                np.rot90(mask, 1),
            )
        return lr, hr, mask

    def _to_cupy_tensors(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        mask: np.ndarray,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cp = cupy_module()
        lr_cp = cp.asarray(lr, dtype=cp.float32)[cp.newaxis, ...] / 255.0
        hr_cp = cp.asarray(hr, dtype=cp.float32)[cp.newaxis, ...] / 255.0
        mask_cp = cp.asarray(mask, dtype=cp.float32)[cp.newaxis, ...]
        if self.augment:
            rng = random.Random(
                _stable_seed(self.seed, self.epoch, index, "augment")
            )
            if rng.random() < 0.5:
                lr_cp, hr_cp, mask_cp = (
                    cp.flip(lr_cp, 2),
                    cp.flip(hr_cp, 2),
                    cp.flip(mask_cp, 2),
                )
            if rng.random() < 0.5:
                lr_cp, hr_cp, mask_cp = (
                    cp.flip(lr_cp, 1),
                    cp.flip(hr_cp, 1),
                    cp.flip(mask_cp, 1),
                )
            if lr_cp.shape[1] == lr_cp.shape[2] and rng.random() < 0.5:
                lr_cp, hr_cp, mask_cp = (
                    cp.rot90(lr_cp, 1, (1, 2)),
                    cp.rot90(hr_cp, 1, (1, 2)),
                    cp.rot90(mask_cp, 1, (1, 2)),
                )
        return (
            torch.as_tensor(cp.ascontiguousarray(lr_cp), device="cuda"),
            torch.as_tensor(cp.ascontiguousarray(hr_cp), device="cuda"),
            torch.as_tensor(cp.ascontiguousarray(mask_cp), device="cuda"),
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        lr, hr = self._get_pair(index)
        lr, hr, mask = self._crop(lr, hr, index)
        if self.preprocess_backend == "cupy":
            lr_tensor, hr_tensor, mask_tensor = self._to_cupy_tensors(
                lr, hr, mask, index
            )
        else:
            lr, hr, mask = self._augment_numpy(lr, hr, mask, index)
            lr_tensor = _uint8_to_tensor(lr)
            hr_tensor = _uint8_to_tensor(hr)
            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(mask, dtype=np.float32)
            ).unsqueeze(0)
        return lr_tensor, hr_tensor, mask_tensor, SOURCE_MOSAIC


class PairedReplayDataset(Dataset):
    """Mevcut `ThermalSRDataset` sınıfını değiştirmeden karma eğitime uyarlar."""

    def __init__(
        self,
        hr_dir: Path | str,
        lr_dir: Path | str,
        *,
        patch_size: int,
        scale_factor: int,
        seed: int,
        augment: bool,
        use_cupy: bool,
    ):
        self.seed = int(seed)
        self.epoch = 0
        self.dataset = ThermalSRDataset(
            hr_dir=str(hr_dir),
            lr_dir=str(lr_dir),
            patch_size=patch_size,
            scale_factor=scale_factor,
            augment=augment,
            use_cupy=use_cupy,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Orijinal dataset Python `random` kullanıyor. Durumu geçici değiştirerek
        # aynı seed/epoch/index için aynı crop ve augmentasyonu elde et.
        state = random.getstate()
        random.seed(_stable_seed(self.seed, self.epoch, index, "paired"))
        try:
            lr, hr = self.dataset[index]
        finally:
            random.setstate(state)
        return lr, hr, torch.ones_like(hr), SOURCE_PAIRED


class DeterministicMixedDataset(Dataset):
    """Kaynak oranlarını tekrar kullanmadan karşılayan epoch planı."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        ratios: Sequence[float],
        *,
        seed: int,
        samples_per_epoch: int = 0,
        sequential_source_ids: Sequence[int] = (),
    ):
        if len(datasets) != len(ratios) or not datasets:
            raise ValueError("datasets ve ratios aynı, sıfırdan büyük uzunlukta olmalı")
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Tüm dataset oranları pozitif olmalı")
        self.datasets = tuple(datasets)
        ratio_sum = float(sum(ratios))
        self.ratios = tuple(float(ratio) / ratio_sum for ratio in ratios)
        self.seed = int(seed)
        self.sequential_source_ids = frozenset(int(i) for i in sequential_source_ids)

        maximum_unique = min(
            int(len(dataset) / ratio)
            for dataset, ratio in zip(self.datasets, self.ratios)
        )
        self.samples_per_epoch = int(samples_per_epoch or maximum_unique)
        if self.samples_per_epoch > maximum_unique:
            raise ValueError(
                f"samples_per_epoch={self.samples_per_epoch}, benzersiz örnek "
                f"sınırı {maximum_unique}. Tekrarı önlemek için değeri düşürün."
            )
        self.epoch = -1
        self.schedule: tuple[tuple[int, int], ...] = ()
        self.set_epoch(0)

    def _source_counts(self) -> list[int]:
        raw = [self.samples_per_epoch * ratio for ratio in self.ratios]
        counts = [int(value) for value in raw]
        missing = self.samples_per_epoch - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda i: raw[i] - counts[i],
            reverse=True,
        )
        for source_id in order[:missing]:
            counts[source_id] += 1
        for count, dataset in zip(counts, self.datasets):
            if count > len(dataset):
                raise ValueError("Kaynak dataset aynı epoch içinde tekrar gerektiriyor")
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(_stable_seed(self.seed, self.epoch, 0, "mixed"))
        source_tokens: list[int] = []
        source_indices: list[list[int]] = []
        for source_id, (dataset, count) in enumerate(
            zip(self.datasets, self._source_counts())
        ):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.epoch)
            indices = list(range(len(dataset)))
            if source_id not in self.sequential_source_ids:
                rng.shuffle(indices)
            source_indices.append(indices[:count])
            source_tokens.extend([source_id] * count)
        rng.shuffle(source_tokens)

        cursors = [0] * len(self.datasets)
        schedule: list[tuple[int, int]] = []
        for source_id in source_tokens:
            source_index = source_indices[source_id][cursors[source_id]]
            cursors[source_id] += 1
            schedule.append((source_id, source_index))
        self.schedule = tuple(schedule)

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int):
        source_id, source_index = self.schedule[index]
        return self.datasets[source_id][source_index]

    def source_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for source_id, _ in self.schedule:
            counts[source_id] = counts.get(source_id, 0) + 1
        return counts

    def close(self) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, "close"):
                dataset.close()


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    cuda_active: bool,
    cupy_active: bool,
    rolling_cache_active: bool,
) -> DataLoader:
    """Platform ve backend kısıtlarını koruyan DataLoader oluştur."""
    if os.name == "nt" or cupy_active or rolling_cache_active:
        num_workers = 0
    if rolling_cache_active and shuffle:
        # Manifest sırası zaten epoch bazında karıştırılmıştır. Disk cache için
        # indislerin ardışık okunması gerekir.
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(cuda_active and not cupy_active),
        drop_last=False,
    )
