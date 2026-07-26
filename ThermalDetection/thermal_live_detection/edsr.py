"""Single-channel EDSR x4 model and inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for EDSR inference. Install the project's "
            "CUDA-enabled PyTorch build before running the live pipeline."
        ) from exc
    return torch


class ResidualBlock:
    """Factory namespace kept private; the concrete module is built lazily."""

    @staticmethod
    def build(num_features: int, res_scale: float) -> Any:
        torch = _torch()

        class _ResidualBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.res_scale = res_scale
                self.conv1 = torch.nn.Conv2d(
                    num_features, num_features, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv2d(
                    num_features, num_features, kernel_size=3, padding=1
                )

            def forward(self, x: Any) -> Any:
                residual = self.conv2(self.relu(self.conv1(x)))
                return x + residual * self.res_scale

        return _ResidualBlock()


class Upscaler:
    """Factory for one PixelShuffle x2 stage."""

    @staticmethod
    def build(num_features: int) -> Any:
        torch = _torch()

        class _Upscaler(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    num_features,
                    num_features * 4,
                    kernel_size=3,
                    padding=1,
                )
                self.shuffle = torch.nn.PixelShuffle(2)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                return self.relu(self.shuffle(self.conv(x)))

        return _Upscaler()


def _build_edsr_class() -> type:
    torch = _torch()

    class _EDSR(torch.nn.Module):
        def __init__(
            self,
            scale_factor: int = 4,
            num_channels: int = 1,
            num_features: int = 64,
            num_residual_blocks: int = 16,
            res_scale: float = 1.0,
        ) -> None:
            super().__init__()
            if scale_factor not in (2, 4):
                raise ValueError("EDSR scale_factor must be 2 or 4.")
            self.scale_factor = scale_factor
            self.head = torch.nn.Conv2d(
                num_channels, num_features, kernel_size=3, padding=1
            )
            body = [
                ResidualBlock.build(num_features, res_scale)
                for _ in range(num_residual_blocks)
            ]
            body.append(
                torch.nn.Conv2d(
                    num_features, num_features, kernel_size=3, padding=1
                )
            )
            self.body = torch.nn.Sequential(*body)
            self.upscale = torch.nn.Sequential(
                *[
                    Upscaler.build(num_features)
                    for _ in range(scale_factor // 2)
                ]
            )
            self.tail = torch.nn.Conv2d(
                num_features, num_channels, kernel_size=3, padding=1
            )

        def forward(self, x: Any) -> Any:
            head_out = self.head(x)
            body_out = self.body(head_out) + head_out
            return self.tail(self.upscale(body_out))

        def get_param_count(self) -> int:
            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

    return _EDSR


EDSR = _build_edsr_class()


class EDSRUpscaler:
    """Load the ThermalDlss checkpoint and upscale 8-bit thermal frames."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        fp16: bool = False,
        native_size: tuple[int, int] = (160, 120),
    ) -> None:
        torch = _torch()
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EDSR checkpoint not found: {checkpoint_path}")

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.native_size = native_size

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        saved_args = checkpoint.get("args", {}) or {}
        scale_factor = int(saved_args.get("scale_factor", 4))
        num_features = int(saved_args.get("num_features", 64))
        num_blocks = int(saved_args.get("num_residual_blocks", 16))
        if scale_factor != 4:
            raise ValueError(
                f"Integrated pipeline requires an x4 EDSR checkpoint, got x{scale_factor}."
            )

        self.model = EDSR(
            scale_factor=scale_factor,
            num_channels=1,
            num_features=num_features,
            num_residual_blocks=num_blocks,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "best_psnr": checkpoint.get("best_psnr"),
            "scale_factor": scale_factor,
            "num_features": num_features,
            "num_residual_blocks": num_blocks,
            "parameters": self.model.get_param_count(),
        }

    def upscale(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the 160x120 model input and the 640x480 EDSR output."""
        import cv2

        torch = _torch()
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError(f"Unsupported camera frame shape: {frame.shape}")

        native = cv2.resize(
            gray,
            self.native_size,
            interpolation=cv2.INTER_AREA,
        )
        tensor = (
            torch.from_numpy(native.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        if self.fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            prediction = self.model(tensor).clamp_(0.0, 1.0)
        sr = (
            prediction.squeeze(0)
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )
        return native, np.rint(sr * 255.0).astype(np.uint8)
