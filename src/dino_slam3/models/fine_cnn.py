from __future__ import annotations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FineCNN(nn.Module):
    """
    Stride-4 local geometric features (localizable).
    Keeps resolution relatively high as recommended by lightweight local-feature designs.
    """
    def __init__(self, in_ch: int = 3, channels: int = 96, num_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ConvBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))
