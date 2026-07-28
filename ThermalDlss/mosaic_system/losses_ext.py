"""Mevcut Sobel uygulamasını kullanan dikiş-maskeli termal SR kaybı."""

from __future__ import annotations

from .bootstrap import ensure_project_root

ensure_project_root()

import torch
import torch.nn as nn
from losses import SobelEdgeLoss


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask
    denominator = mask.sum().clamp_min(1.0)
    return weighted.sum() / denominator


class MaskedThermalSRLoss(nn.Module):
    """`L1 + edge_weight × Sobel` kaybını geçerli HR piksellerinde hesapla."""

    def __init__(self, edge_weight: float = 0.01):
        super().__init__()
        self.edge_weight = float(edge_weight)
        self.edge_loss = SobelEdgeLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mask is None:
            mask = torch.ones_like(target)
        if mask.shape != target.shape:
            mask = mask.expand_as(target)
        mask = mask.to(dtype=pred.dtype)

        pixel = _masked_mean(torch.abs(pred - target), mask)
        pred_edges = self.edge_loss._sobel_edges(pred)
        target_edges = self.edge_loss._sobel_edges(target)
        edge = _masked_mean(torch.abs(pred_edges - target_edges), mask)
        total = pixel + self.edge_weight * edge
        return total, {
            "pixel": float(pixel.detach().item()),
            "edge": float(edge.detach().item()),
            "total": float(total.detach().item()),
        }

