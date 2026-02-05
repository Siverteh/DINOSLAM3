from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.dinov3 import DinoV3Backbone
from .fine_cnn import FineCNN
from .heads import HybridHeads, FeatureOutputs

class LocalFeatureNet(nn.Module):
    """DINOv3 tokens + fine CNN -> detector+descriptor(+offset,+reliability) at fine stride."""

    def __init__(
        self,
        patch_size: int = 16,
        descriptor_dim: int = 128,
        fine_channels: int = 64,
        fine_blocks: int = 6,
        fine_stride: int = 4,
        use_offset: bool = True,
        use_reliability: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = DinoV3Backbone(patch_size=patch_size, freeze=freeze_backbone)
        self.fine = FineCNN(in_ch=3, channels=fine_channels, num_blocks=fine_blocks, out_stride=fine_stride)

        # DINO tokens will be upsampled to match fine feature resolution
        # Concatenate fine features (C_f) + dino (C_d)
        # We don't know C_d until DINO is loaded, so we build heads lazily.
        self.descriptor_dim = descriptor_dim
        self.use_offset = use_offset
        self.use_reliability = use_reliability
        self.fine_stride = fine_stride

        self._heads: Optional[HybridHeads] = None

    def _build_heads_if_needed(self, dino_c: int, fine_c: int, device: torch.device) -> None:
        if self._heads is not None:
            return
        self._heads = HybridHeads(
            in_ch=dino_c + fine_c,
            descriptor_dim=self.descriptor_dim,
            use_offset=self.use_offset,
            use_reliability=self.use_reliability,
        ).to(device)

    def forward(self, x: torch.Tensor) -> FeatureOutputs:
        # x: B,3,H,W
        fine = self.fine(x)  # B,Cf,Hf,Wf (Hf=H/stride)
        dino = self.backbone(x).tokens  # B,Cd,H/patch,W/patch

        # upsample dino to fine resolution
        Hf, Wf = fine.shape[-2:]
        dino_up = F.interpolate(dino, size=(Hf, Wf), mode="bilinear", align_corners=False)

        self._build_heads_if_needed(dino_c=dino_up.shape[1], fine_c=fine.shape[1], device=x.device)
        feat = torch.cat([fine, dino_up], dim=1)
        return self._heads(feat)  # type: ignore[arg-type]
