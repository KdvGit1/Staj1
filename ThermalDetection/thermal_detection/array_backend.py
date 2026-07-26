"""Automatic CuPy/NumPy selection for auxiliary numerical operations.

Ultralytics performs model training with PyTorch, so CUDA training does not
depend on this module. This backend is used by dataset conversion and label
validation, where NumPy-compatible array operations are useful.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

import numpy as np

BackendPreference = Literal["auto", "cupy", "numpy"]


@dataclass(frozen=True)
class ArrayBackend:
    """A small compatibility wrapper around NumPy or CuPy."""

    name: str
    xp: Any
    detail: str

    @property
    def is_cuda(self) -> bool:
        return self.name == "cupy"

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.is_cuda:
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.is_cuda:
            self.xp.cuda.Stream.null.synchronize()


def select_array_backend(preference: BackendPreference | None = None) -> ArrayBackend:
    """Select CuPy when it is installed and has a working CUDA device.

    Selection order:
      1. Explicit ``preference`` argument.
      2. ``THERMAL_ARRAY_BACKEND`` environment variable.
      3. ``auto``.

    ``auto`` safely falls back to NumPy on missing CuPy, a missing NVIDIA
    device, or a CuPy/CUDA runtime mismatch. Explicit ``cupy`` raises a clear
    error instead of silently changing the requested backend.
    """

    requested = (
        preference
        or os.environ.get("THERMAL_ARRAY_BACKEND", "auto")
    ).strip().lower()
    if requested not in {"auto", "cupy", "numpy"}:
        raise ValueError(
            "Array backend must be one of: auto, cupy, numpy. "
            f"Received: {requested!r}"
        )

    if requested == "numpy":
        return ArrayBackend("numpy", np, f"NumPy {np.__version__} (forced)")

    try:
        import cupy as cp  # type: ignore

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("CuPy found no CUDA-capable device")

        # A tiny real operation catches common driver/runtime incompatibilities.
        probe = cp.asarray([1.0], dtype=cp.float32)
        _ = probe + 1.0
        cp.cuda.Stream.null.synchronize()
        device_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", "unknown CUDA device")
        device_name = (
            raw_name.decode("utf-8", errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        return ArrayBackend(
            "cupy",
            cp,
            f"CuPy {cp.__version__}, CUDA device {device_id}: {device_name}",
        )
    except Exception as exc:
        if requested == "cupy":
            raise RuntimeError(
                "CuPy was explicitly requested but a working CUDA backend "
                f"could not be initialized: {exc}"
            ) from exc
        return ArrayBackend(
            "numpy",
            np,
            f"NumPy {np.__version__}; CuPy auto-detection failed: {exc}",
        )

