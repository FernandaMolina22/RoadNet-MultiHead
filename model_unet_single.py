# model_unet_single.py
"""
Baseline single-head U-Net for road segmentation on AWR patches.

- Input:  (B, 3, 128, 128) RGB image
- Output: (B, 1, 128, 128) logit map (no sigmoid applied here)

Use BCEWithLogitsLoss or a combo BCE+Dice on the raw logits.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.

    If bilinear=True, uses nn.Upsample + Conv2d to reduce channels.
    Otherwise, uses ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # in_channels = channels_from_decoder + channels_from_skip
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.bilinear = bilinear

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: feature map from decoder (coarse)
        x2: feature map from encoder (skip connection)
        """
        x1 = self.up(x1)

        # Input is CHW
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        # Pad x1 to match x2 spatially (important if sizes are odd)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 conv to map to n_classes channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------

class UNetSingle(nn.Module):
    """
    Baseline U-Net for binary road segmentation.

    n_channels: number of input channels (3 for RGB)
    n_classes:  number of output channels (1 for road mask)
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # factor is used to reduce the number of channels in the bottleneck
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)      # 64, 128,128
        x2 = self.down1(x1)   # 128,64,64
        x3 = self.down2(x2)   # 256,32,32
        x4 = self.down3(x3)   # 512,16,16
        x5 = self.down4(x4)   # 1024 or 512,8,8

        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512,16,16
        x = self.up2(x,  x3)  # 256,32,32
        x = self.up3(x,  x2)  # 128,64,64
        x = self.up4(x,  x1)  # 64,128,128

        logits = self.outc(x) # (B, 1, 128,128)
        return logits


def build_unet_single(device: Optional[torch.device] = None) -> UNetSingle:
    """
    Convenience function to build the baseline U-Net and (optionally) move it to a device.
    """
    model = UNetSingle(n_channels=3, n_classes=1, bilinear=True)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------
# Quick self-test (including GPU)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_unet_single(device)
    x = torch.randn(2, 3, 128, 128, device=device)  # batch of 2 patches
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)  # expect (2, 1, 128, 128)
