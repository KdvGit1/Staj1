"""Kök EDSR modelini import ederek checkpoint/fine-tuning yönetimi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bootstrap import ensure_project_root

ensure_project_root()

import torch
import torch.nn as nn
from model import EDSR


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args
    return vars(args) if hasattr(args, "__dict__") else {}


def load_checkpoint_payload(path: Path | str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint sözlük değil: {path}")
    return payload


def build_model(
    *,
    device: torch.device,
    checkpoint_path: Path | str | None = None,
    scale_factor: int = 4,
    num_features: int = 64,
    num_residual_blocks: int = 16,
    post_shuffle_relu: bool | None = True,
) -> tuple[EDSR, dict[str, Any] | None]:
    """Modeli kur; checkpoint verilirse önce mimari argümanlarını ondan al."""
    checkpoint = None
    if checkpoint_path:
        checkpoint = load_checkpoint_payload(checkpoint_path, device)
        saved = _checkpoint_args(checkpoint)
        scale_factor = int(saved.get("scale_factor", scale_factor))
        num_features = int(saved.get("num_features", num_features))
        num_residual_blocks = int(
            saved.get("num_residual_blocks", num_residual_blocks)
        )
        if post_shuffle_relu is None:
            post_shuffle_relu = bool(
                saved.get(
                    "post_shuffle_relu",
                    not saved.get("no_post_shuffle_relu", False),
                )
            )
    if post_shuffle_relu is None:
        post_shuffle_relu = True

    model = EDSR(
        scale_factor=scale_factor,
        num_channels=1,
        num_features=num_features,
        num_residual_blocks=num_residual_blocks,
    ).to(device)

    if checkpoint is not None:
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)

    if not post_shuffle_relu:
        for block in model.upscale:
            if hasattr(block, "relu"):
                block.relu = nn.Identity()

    model.mosaic_config = {
        "scale_factor": scale_factor,
        "num_features": num_features,
        "num_residual_blocks": num_residual_blocks,
        "post_shuffle_relu": bool(post_shuffle_relu),
    }
    return model, checkpoint


def save_checkpoint(
    path: Path | str,
    *,
    model: EDSR,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_psnr: float,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_psnr": float(best_psnr),
            "args": dict(config),
            "mosaic_system": True,
        },
        Path(path),
    )
