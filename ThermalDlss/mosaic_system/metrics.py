"""Referanslı ve native görsel artefakt metrikleri."""

from __future__ import annotations

import math

import numpy as np


def masked_psnr_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = 1.0,
) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    squared = (pred - target) ** 2
    if mask is None:
        mse = float(squared.mean())
    else:
        valid = np.asarray(mask, dtype=np.float64)
        denominator = float(valid.sum())
        mse = float((squared * valid).sum() / max(denominator, 1.0))
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10((data_range**2) / mse)


def artifact_metrics(image: np.ndarray, phase: int = 4) -> dict[str, float]:
    """[0,1] tek kanallı görüntüde clipping, keskinlik ve faz farkı ölç."""
    array = np.asarray(image, dtype=np.float32).squeeze()
    grad_x = np.abs(np.diff(array, axis=1)).mean()
    grad_y = np.abs(np.diff(array, axis=0)).mean()
    center = array[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )
    phase_means = [
        float(array[row::phase, col::phase].mean())
        for row in range(phase)
        for col in range(phase)
    ]
    return {
        "clip_ratio": float(((array <= 0.0) | (array >= 1.0)).mean()),
        "gradient_x": float(grad_x),
        "gradient_y": float(grad_y),
        "laplacian_abs": float(np.abs(laplacian).mean()),
        "phase_mean_std": float(np.std(phase_means)),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }

