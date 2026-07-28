"""Deterministik, tekrarsız ve mümkün olduğunca video-çeşitli mozaik planları."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_PATTERN = re.compile(r"^(?P<video>.+?)-frame-", re.IGNORECASE)


def discover_images(image_dir: Path) -> list[Path]:
    """Dizindeki desteklenen görüntüleri sıralı olarak bul; dosya kopyalamaz."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Görüntü dizini bulunamadı: {image_dir}")
    images = sorted(
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_dir}")
    return images


def extract_video_id(path: Path) -> str:
    """Projede kullanılan `...-frame-...` adından video kimliğini çıkar."""
    match = VIDEO_PATTERN.match(path.stem)
    return match.group("video") if match else path.stem


def group_signature(paths: Sequence[Path]) -> str:
    """Karo sırasından bağımsız mozaik birleşim imzası."""
    payload = "\n".join(sorted(str(path) for path in paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_for(base_seed: int, epoch: int, nonce: int) -> int:
    payload = f"{base_seed}:{epoch}:{nonce}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class MosaicGroup:
    """Bir 4×4 pseudo-HR tuvalin kaynak dosyaları ve izlenebilir kimliği."""

    group_id: str
    epoch: int
    index: int
    image_paths: tuple[Path, ...]
    video_ids: tuple[str, ...]
    signature: str

    @property
    def distinct_video_count(self) -> int:
        return len(set(self.video_ids))


class MosaicManifestPlanner:
    """Her epoch için tekrarsız 16'lı gruplar üretir.

    Bir epoch içinde her kaynak dosya en fazla bir kez kullanılır. Önceki
    epoch'lardaki tam 16'lı birleşim imzaları da bellekte tutulur; aynı
    kombinasyon tekrar oluşursa epoch farklı bir deterministik nonce ile
    yeniden planlanır. Resume sırasında önceki epoch planları aynı seed ile
    yeniden hesaplanır, bu yüzden büyük bir manifest dosyasına gerek kalmaz.
    """

    def __init__(
        self,
        image_paths: Iterable[Path],
        *,
        tiles_per_mosaic: int = 16,
        seed: int = 42,
        max_nonce_attempts: int = 128,
    ):
        self.image_paths = tuple(sorted(Path(path).resolve() for path in image_paths))
        self.tiles_per_mosaic = int(tiles_per_mosaic)
        self.seed = int(seed)
        self.max_nonce_attempts = int(max_nonce_attempts)

        if self.tiles_per_mosaic <= 1:
            raise ValueError("tiles_per_mosaic en az 2 olmalı")
        if len(self.image_paths) < self.tiles_per_mosaic:
            raise ValueError(
                f"En az {self.tiles_per_mosaic} görüntü gerekli; bulunan: "
                f"{len(self.image_paths)}"
            )
        if len(set(self.image_paths)) != len(self.image_paths):
            raise ValueError("Kaynak görüntü listesinde yinelenen dosya var")

        self._seen_signatures: set[str] = set()
        self._highest_built_epoch = -1
        self._current_groups: tuple[MosaicGroup, ...] = ()

    @property
    def groups_per_epoch(self) -> int:
        return len(self.image_paths) // self.tiles_per_mosaic

    @property
    def leftovers_per_epoch(self) -> int:
        return len(self.image_paths) % self.tiles_per_mosaic

    def build_epoch(self, epoch: int) -> tuple[MosaicGroup, ...]:
        """İstenen epoch planını, önceki epoch birleşimleriyle çakışmadan kur."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch negatif olamaz")
        if epoch == self._highest_built_epoch:
            return self._current_groups
        if epoch < self._highest_built_epoch:
            # Eski bir epoch yeniden istenirse deterministik diziyi baştan kur.
            self._seen_signatures.clear()
            self._highest_built_epoch = -1
            self._current_groups = ()

        # Resume için eksik önceki epoch'ları deterministik olarak yeniden kur.
        for current_epoch in range(self._highest_built_epoch + 1, epoch + 1):
            groups = self._build_unique_epoch(current_epoch)
            self._seen_signatures.update(group.signature for group in groups)
            self._highest_built_epoch = current_epoch
            self._current_groups = groups
        return self._current_groups

    def _build_unique_epoch(self, epoch: int) -> tuple[MosaicGroup, ...]:
        for nonce in range(self.max_nonce_attempts):
            groups = self._build_candidate(epoch, nonce)
            signatures = [group.signature for group in groups]
            if len(signatures) != len(set(signatures)):
                continue
            if self._seen_signatures.isdisjoint(signatures):
                return groups
        raise RuntimeError(
            f"Epoch {epoch} için {self.max_nonce_attempts} denemede benzersiz "
            "mozaik planı üretilemedi"
        )

    def _build_candidate(self, epoch: int, nonce: int) -> tuple[MosaicGroup, ...]:
        rng = random.Random(_seed_for(self.seed, epoch, nonce))
        buckets: dict[str, list[Path]] = defaultdict(list)
        for path in self.image_paths:
            buckets[extract_video_id(path)].append(path)
        for paths in buckets.values():
            rng.shuffle(paths)

        target_count = self.groups_per_epoch
        group_paths: list[list[Path]] = [[] for _ in range(target_count)]
        group_videos: list[list[str]] = [[] for _ in range(target_count)]

        # Büyük video kovalarını bütün mozaiklere yay. Böylece bir büyük video
        # sona yığılıp son mozaiklerin tek sahneden oluşmasına neden olmaz.
        ranked_videos = sorted(
            buckets,
            key=lambda video_id: (len(buckets[video_id]), rng.random()),
            reverse=True,
        )
        for video_id in ranked_videos:
            for path in buckets[video_id]:
                available = [
                    index
                    for index, paths in enumerate(group_paths)
                    if len(paths) < self.tiles_per_mosaic
                ]
                if not available:
                    break  # Yalnız `leftovers_per_epoch` kadar dosya dışarıda kalır.
                without_same_video = [
                    index
                    for index in available
                    if video_id not in group_videos[index]
                ]
                candidates = without_same_video or available
                minimum_fill = min(len(group_paths[index]) for index in candidates)
                least_filled = [
                    index
                    for index in candidates
                    if len(group_paths[index]) == minimum_fill
                ]
                target = rng.choice(least_filled)
                group_paths[target].append(path)
                group_videos[target].append(video_id)

        if any(len(paths) != self.tiles_per_mosaic for paths in group_paths):
            sizes = sorted({len(paths) for paths in group_paths})
            raise RuntimeError(f"Eksik mozaik grubu oluştu; grup boyutları: {sizes}")

        groups: list[MosaicGroup] = []
        for group_index, (selected, selected_videos) in enumerate(
            zip(group_paths, group_videos)
        ):

            tile_order = list(range(self.tiles_per_mosaic))
            rng.shuffle(tile_order)
            selected = [selected[i] for i in tile_order]
            selected_videos = [selected_videos[i] for i in tile_order]
            signature = group_signature(selected)
            groups.append(
                MosaicGroup(
                    group_id=f"e{epoch:04d}-m{group_index:05d}-{signature[:12]}",
                    epoch=epoch,
                    index=group_index,
                    image_paths=tuple(selected),
                    video_ids=tuple(selected_videos),
                    signature=signature,
                )
            )
        return tuple(groups)

    def summary(self, epoch: int = 0) -> dict[str, int | float]:
        groups = self.build_epoch(epoch)
        distinct_counts = [group.distinct_video_count for group in groups]
        return {
            "source_images": len(self.image_paths),
            "groups": len(groups),
            "leftovers": self.leftovers_per_epoch,
            "tiles_per_group": self.tiles_per_mosaic,
            "min_distinct_videos": min(distinct_counts),
            "mean_distinct_videos": sum(distinct_counts) / len(distinct_counts),
            "max_distinct_videos": max(distinct_counts),
        }
