"""Pseudo-HR tuvali üretme ve isteğe bağlı sınırlı rolling disk cache."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
OWNER_MARKER = ".thermal_dlss_mosaic_cache"


def load_grayscale_uint8(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    """Kaynak görüntüyü kopyalamadan tek kanallı uint8 olarak oku."""
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != expected_size:
            raise ValueError(
                f"Beklenmeyen görüntü boyutu {gray.size}; beklenen "
                f"{expected_size}: {path}"
            )
        return np.asarray(gray, dtype=np.uint8)


def build_mosaic_pair(
    image_paths: tuple[Path, ...] | list[Path],
    *,
    grid_size: int = 4,
    tile_size: tuple[int, int] = (640, 512),
    scale_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """16 kaynaktan RAM'de HR tuval ve bicubic LR üret.

    Dönüş sırası `(lr_uint8, hr_uint8)` biçimindedir. Hiçbir görüntü kalıcı
    olarak kaydedilmez.
    """
    expected_count = grid_size * grid_size
    if len(image_paths) != expected_count:
        raise ValueError(
            f"{expected_count} görüntü gerekli; verilen: {len(image_paths)}"
        )
    tile_w, tile_h = tile_size
    hr_w, hr_h = tile_w * grid_size, tile_h * grid_size
    if hr_w % scale_factor or hr_h % scale_factor:
        raise ValueError("Mozaik boyutu scale_factor ile tam bölünmeli")

    hr = np.empty((hr_h, hr_w), dtype=np.uint8)
    for tile_index, path in enumerate(image_paths):
        row, col = divmod(tile_index, grid_size)
        y0, x0 = row * tile_h, col * tile_w
        hr[y0 : y0 + tile_h, x0 : x0 + tile_w] = load_grayscale_uint8(
            Path(path), tile_size
        )

    lr_image = Image.fromarray(hr, mode="L").resize(
        (hr_w // scale_factor, hr_h // scale_factor),
        BICUBIC,
    )
    lr = np.asarray(lr_image, dtype=np.uint8)
    return lr, hr


class RollingMosaicCache:
    """Yalnız geçerli `N` örneği diskte tutan ve pencere sonunda silen cache.

    Cache yalnız bu sınıfın marker bıraktığı dizinde temizlik yapar. Başlangıçta
    önceki yarım kalmış cache temizlenir; normal kapanışta tüm dosyalar silinir.
    """

    def __init__(self, cache_dir: Path, capacity: int, total_items: int):
        self.cache_dir = Path(cache_dir).resolve()
        self.capacity = int(capacity)
        self.total_items = int(total_items)
        if self.capacity <= 0:
            raise ValueError("Rolling cache kapasitesi pozitif olmalı")
        if self.total_items <= 0:
            raise ValueError("Rolling cache total_items pozitif olmalı")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.marker = self.cache_dir / OWNER_MARKER
        self.marker.write_text(
            "Bu dizin yalnız ThermalDlss mosaic_system geçici cache verisidir.\n",
            encoding="utf-8",
        )
        self._current_start: int | None = None
        self._cleanup_payloads()
        atexit.register(self.close)

    def _assert_owned(self) -> None:
        if not self.marker.is_file():
            raise RuntimeError(
                f"Cache güvenlik marker'ı yok; temizlik reddedildi: {self.cache_dir}"
            )

    def _cleanup_payloads(self) -> None:
        self._assert_owned()
        for path in self.cache_dir.iterdir():
            if path == self.marker:
                continue
            if (
                path.is_file()
                and path.name.startswith("mosaic_")
                and path.suffix == ".npz"
            ):
                path.unlink()

    def _path_for(self, index: int) -> Path:
        return self.cache_dir / f"mosaic_{index:06d}.npz"

    def get(
        self,
        index: int,
        builder: Callable[[int], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if index < 0 or index >= self.total_items:
            raise IndexError(index)
        chunk_start = (index // self.capacity) * self.capacity
        if chunk_start != self._current_start:
            self._cleanup_payloads()
            chunk_end = min(chunk_start + self.capacity, self.total_items)
            for item_index in range(chunk_start, chunk_end):
                lr, hr = builder(item_index)
                target = self._path_for(item_index)
                temporary = target.with_suffix(".tmp.npz")
                np.savez(temporary, lr=lr, hr=hr)
                os.replace(temporary, target)
            self._current_start = chunk_start

        payload = np.load(self._path_for(index), allow_pickle=False)
        try:
            return payload["lr"], payload["hr"]
        finally:
            payload.close()

    def close(self) -> None:
        if not self.cache_dir.exists() or not self.marker.exists():
            return
        self._cleanup_payloads()
        self.marker.unlink(missing_ok=True)
        try:
            self.cache_dir.rmdir()
        except OSError:
            # Kullanıcı dosyası veya eşzamanlı süreç varsa dizine dokunma.
            pass
