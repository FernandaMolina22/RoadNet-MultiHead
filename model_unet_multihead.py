#model_unet_multihead.py
"""
Multi-head U-Net for topology-aware road extraction on AWR patches.

- Input:  (B, 3, 128, 128) RGB image
- Output: dict with three logit maps (no sigmoid):

    {
        "mask":       (B, 1, 128, 128),  # road mask
        "centerline": (B, 1, 128, 128),  # road centerlines
        "junctions":  (B, 1, 128, 128),  # junction heatmap
    }

Use BCEWithLogitsLoss (and optionally Dice) on each head.
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

# Reuse the same U-Net building blocks from the single-head model
from model_unet_single import DoubleConv, Down, Up, OutConv


class UNetMultiHead(nn.Module):
    """
    Multi-head U-Net with a shared encoder-decoder backbone and
    three separate 1x1 conv heads:

    - mask_head:       road mask
    - centerline_head: road centerlines
    - junction_head:   junction heatmap
    """

    def __init__(
        self,
        n_channels: int = 3,
        bilinear: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes_mask = 1
        self.n_classes_center = 1
        self.n_classes_junc = 1
        self.bilinear = bilinear

        # Same backbone as UNetSingle
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)

        # --- Multi-head outputs ---
        # All heads operate on the final 64-channel feature map.
        self.out_mask       = OutConv(64, self.n_classes_mask)
        self.out_centerline = OutConv(64, self.n_classes_center)
        self.out_junction   = OutConv(64, self.n_classes_junc)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns a dict:
            {
                "mask":       logits_mask,
                "centerline": logits_center,
                "junctions":  logits_junc,
            }
        """
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

        # Multi-head outputs (logits, no sigmoid)
        logits_mask       = self.out_mask(x)
        logits_centerline = self.out_centerline(x)
        logits_junction   = self.out_junction(x)

        return {
            "mask":       logits_mask,
            "centerline": logits_centerline,
            "junctions":  logits_junction,
        }


def build_unet_multihead(device: Optional[torch.device] = None) -> UNetMultiHead:
    """
    Convenience builder to instantiate the multi-head U-Net and optionally
    move it to a device.
    """
    model = UNetMultiHead(n_channels=3, bilinear=True)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------
# Quick self-test (including GPU)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_unet_multihead(device)
    x = torch.randn(2, 3, 128, 128, device=device)  # batch of 2 patches
    outputs = model(x)

    print("Input shape:", x.shape)
    for k, v in outputs.items():
        print(f"{k} shape:", v.shape)
