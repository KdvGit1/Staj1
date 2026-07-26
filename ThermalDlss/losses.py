"""
Thermal Süper Çözünürlük — Kayıp Fonksiyonları
================================================
L1 piksel kaybı + Sobel kenar kaybı kombinasyonu.
Perceptual loss ileride eklenecek.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Sobel filtresi ile kenar (gradyan) farkı kaybı.

    Hem yatay hem dikey Sobel gradyanlarını hesaplar, tahmin ile gerçek
    görüntünün kenar haritaları arasındaki L1 farkını döndürür.

    Thermal görüntülerde kenarlar = sıcaklık geçişleri. Bu kayıp,
    modelin kenar netliğini korumasını teşvik eder.
    """

    def __init__(self):
        super().__init__()
        # Sobel filtreleri (sabit, öğrenilmez)
        # Yatay kenar filtresi
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        # Dikey kenar filtresi
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel_edges(self, img: torch.Tensor) -> torch.Tensor:
        """Sobel gradyan büyüklüğünü hesaplar.

        Args:
            img: [B, 1, H, W] grayscale görüntü

        Returns:
            [B, 1, H, W] kenar haritası (gradyan büyüklüğü)
        """
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model tahmini [B, 1, H, W]
            target: Gerçek HR görüntü [B, 1, H, W]

        Returns:
            Kenar kaybı (scalar)
        """
        edges_pred = self._sobel_edges(pred)
        edges_target = self._sobel_edges(target)
        return F.l1_loss(edges_pred, edges_target)


class ThermalSRLoss(nn.Module):
    """Thermal süper çözünürlük için kombine kayıp fonksiyonu.

    L_total = L_pixel + λ_edge * L_edge

    Args:
        edge_weight: Kenar kaybı ağırlığı (λ_edge).
    """

    def __init__(self, edge_weight: float = 0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = SobelEdgeLoss()
        self.edge_weight = edge_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred: Model tahmini [B, 1, H, W]
            target: Gerçek HR görüntü [B, 1, H, W]

        Returns:
            (total_loss, loss_dict) — loss_dict loglama için ayrıntıları içerir
        """
        loss_pixel = self.l1_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)
        loss_total = loss_pixel + self.edge_weight * loss_edge

        loss_dict = {
            "pixel": loss_pixel.item(),
            "edge": loss_edge.item(),
            "total": loss_total.item(),
        }

        return loss_total, loss_dict
