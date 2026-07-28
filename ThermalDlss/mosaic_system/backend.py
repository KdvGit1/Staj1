"""CPU, CUDA ve mevcut projedeki CuPy desteğini tek yerde çözümle."""

from __future__ import annotations

from dataclasses import dataclass

from .bootstrap import ensure_project_root

ensure_project_root()

import torch
from dataset import HAS_CUPY, cp


@dataclass(frozen=True)
class BackendInfo:
    device: torch.device
    preprocess_backend: str
    cupy_available: bool
    cuda_available: bool
    description: str


def resolve_backends(
    *,
    device_request: str = "auto",
    preprocess_request: str = "auto",
) -> BackendInfo:
    """Model cihazını ve dataset ön-işleme yolunu doğrula."""
    device_request = device_request.lower()
    preprocess_request = preprocess_request.lower()

    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError("device: auto, cpu veya cuda olmalı")
    if preprocess_request not in {"auto", "cpu", "cupy"}:
        raise ValueError("preprocess_backend: auto, cpu veya cupy olmalı")

    cuda_available = torch.cuda.is_available()
    if device_request == "cuda" and not cuda_available:
        raise RuntimeError("CUDA istendi fakat torch.cuda.is_available() False")
    device = torch.device(
        "cuda"
        if device_request == "cuda"
        or (device_request == "auto" and cuda_available)
        else "cpu"
    )

    cupy_available = bool(HAS_CUPY and cuda_available)
    if preprocess_request == "cupy" and not cupy_available:
        raise RuntimeError("CuPy istendi fakat CuPy + CUDA birlikte kullanılamıyor")
    preprocess_backend = (
        "cupy"
        if preprocess_request == "cupy"
        or (preprocess_request == "auto" and cupy_available)
        else "cpu"
    )
    if preprocess_backend == "cupy" and device.type != "cuda":
        raise ValueError("CuPy ön-işleme yalnız CUDA model cihazıyla kullanılabilir")

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        device_text = f"CUDA: {gpu}, {vram:.1f} GB"
    else:
        device_text = "CPU"
    description = f"{device_text}; ön-işleme={preprocess_backend}"
    return BackendInfo(
        device=device,
        preprocess_backend=preprocess_backend,
        cupy_available=cupy_available,
        cuda_available=cuda_available,
        description=description,
    )


def cupy_module():
    """Aktif CuPy modülünü döndür; yoksa açıklayıcı hata üret."""
    if not HAS_CUPY or cp is None:
        raise RuntimeError("CuPy modülü mevcut değil")
    return cp

