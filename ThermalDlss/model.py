"""
EDSR (Enhanced Deep Residual Super-Resolution) — Thermal Görüntü için
======================================================================
Tek kanallı (grayscale) thermal görüntüler için uyarlanmış EDSR mimarisi.
Batch Normalization KULLANILMAZ (EDSR makalesinin temel bulgularından biri).

Referans: Lim ve ark., "Enhanced Deep Residual Networks for Single Image
          Super-Resolution", CVPR 2017 Workshop.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Temel EDSR residual bloğu: Conv → ReLU → Conv + skip connection.

    Batch Normalization yok — EDSR'nin performans artışının temel kaynağı.
    """

    def __init__(self, num_features: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.res_scale


class Upscaler(nn.Module):
    """Sub-pixel convolution (PixelShuffle) ile 2x büyütme bloğu.

    Conv2d ile kanal sayısını 4 katına çıkarır, PixelShuffle ile
    uzaysal boyutu 2x büyütür.
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.conv(x)))


class EDSR(nn.Module):
    """Thermal görüntüler için EDSR modeli.

    Args:
        scale_factor: Büyütme faktörü (2 veya 4).
        num_channels: Girdi/çıktı kanal sayısı (thermal = 1).
        num_features: Ara katman kanal sayısı.
        num_residual_blocks: Residual blok sayısı.
        res_scale: Residual ölçekleme faktörü (stabilite için).
    """

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 1,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Baş: Girdi konvolüsyonu
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        # Gövde: Residual bloklar
        body = [
            ResidualBlock(num_features, res_scale)
            for _ in range(num_residual_blocks)
        ]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Upscale: PixelShuffle ile büyütme
        # 4x = 2 aşamalı PixelShuffle(2), 2x = 1 aşamalı
        num_upscale_blocks = scale_factor // 2  # 4→2, 2→1
        upscale = [Upscaler(num_features) for _ in range(num_upscale_blocks)]
        self.upscale = nn.Sequential(*upscale)

        # Kuyruk: Çıktı konvolüsyonu
        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Düşük çözünürlüklü girdi [B, 1, H, W]

        Returns:
            Yüksek çözünürlüklü tahmin [B, 1, H*scale, W*scale]
        """
        head_out = self.head(x)             # Öznitelik çıkarma
        body_out = self.body(head_out)       # Derin residual öğrenme
        body_out = body_out + head_out       # Global skip connection
        upscaled = self.upscale(body_out)    # PixelShuffle büyütme
        output = self.tail(upscaled)         # 1 kanala projeksiyon
        return output

    def get_param_count(self) -> int:
        """Toplam eğitilebilir parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Hızlı mimari doğrulama
    model = EDSR(scale_factor=4, num_channels=1, num_features=64, num_residual_blocks=16)
    print(f"Parametre sayısı: {model.get_param_count():,}")

    dummy_lr = torch.randn(1, 1, 128, 160)  # 160×128 LR girdi (4x küçültülmüş)
    dummy_hr = model(dummy_lr)
    print(f"Girdi:  {dummy_lr.shape}")       # [1, 1, 128, 160]
    print(f"Çıktı:  {dummy_hr.shape}")       # [1, 1, 512, 640]
