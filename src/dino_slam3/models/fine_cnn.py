from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class FineCNN(nn.Module):
    """A lightweight CNN producing fine features at stride 4 by default.

    Purpose: re-inject local texture/geometry lost by DINO semantic invariances.
    """
    def __init__(self, in_ch: int = 3, channels: int = 64, num_blocks: int = 6, out_stride: int = 4):
        super().__init__()
        assert out_stride in (2, 4, 8)
        self.out_stride = out_stride

        # Simple stem to reach stride 4
        layers = [
            nn.Conv2d(in_ch, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        c = channels
        for _ in range(num_blocks):
            layers.append(ConvBlock(c, c))
        if out_stride == 8:
            layers += [
                nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
        elif out_stride == 2:
            # upsample to stride 2 (rarely needed)
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
