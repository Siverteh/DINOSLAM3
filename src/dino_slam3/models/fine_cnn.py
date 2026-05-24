from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Multi-scale local geometric features.

    Output pyramid:
      - stage2: stride-2
      - stage4: stride-4

    `forward(x)` preserves the legacy contract and returns stage4 only.
    """
    def __init__(
        self,
        in_ch: int = 3,
        channels: int = 96,
        num_blocks: int = 8,
        enable_depth_aux: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.channels = int(channels)
        self.enable_depth_aux = bool(enable_depth_aux)
        blocks = max(1, int(num_blocks))
        n_stage2 = max(1, blocks // 3)
        n_stage4 = max(1, blocks - n_stage2)

        self.conv_s2 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.stage2_blocks = nn.Sequential(*[ConvBlock(self.channels) for _ in range(n_stage2)])

        self.conv_s4 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.stage4_blocks = nn.Sequential(*[ConvBlock(self.channels) for _ in range(n_stage4)])

        if self.enable_depth_aux:
            self.depth_s2 = nn.Sequential(
                nn.Conv2d(1, self.channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                ConvBlock(self.channels),
            )
            self.depth_s4 = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                ConvBlock(self.channels),
            )
        else:
            self.depth_s2 = None
            self.depth_s4 = None

    def forward_pyramid(
        self,
        x: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        s2 = self.conv_s2(x)
        s2 = self.stage2_blocks(s2)
        s4 = self.conv_s4(s2)
        s4 = self.stage4_blocks(s4)
        out = {"stage2": s2, "stage4": s4}

        if self.enable_depth_aux and self.depth_s2 is not None and self.depth_s4 is not None:
            if depth is None:
                # Inference path has no depth; use a lightweight intensity proxy.
                d = x.mean(dim=1, keepdim=True)
            else:
                d = depth
                if d.dim() != 4:
                    d = x.mean(dim=1, keepdim=True)
                if d.shape[1] != 1:
                    d = d[:, :1]
                if d.shape[-2:] != x.shape[-2:]:
                    d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
                d = d.clamp(min=0.0)
                # Compress dynamic range to stabilize depth auxiliary branch.
                d = torch.log1p(d)
            ds2 = self.depth_s2(d)
            ds4 = self.depth_s4(ds2)
            out["depth_s4"] = ds4
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyr = self.forward_pyramid(x)
        return pyr["stage4"]
