from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.dinov3 import DinoV3Backbone
from .fine_cnn import FineCNN
from .heads import Heads, FeatureOutputs

class LocalFeatureNet(nn.Module):
    """
    DINOv3 tokens (stride=16) + FineCNN (stride=4) -> detector/descriptor/offset/reliability at stride=4.
    """
    def __init__(
        self,
        dinov3_name: str,
        patch_size: int = 16,
        descriptor_dim: int = 256,
        fine_channels: int = 96,
        fine_blocks: int = 8,
        freeze_backbone: bool = True,
        use_offset: bool = True,
        use_reliability: bool = True,
        dinov3_dtype: str = "bf16",
    ):
        super().__init__()
        self.patch_size = int(patch_size)

        self.backbone = DinoV3Backbone(
            name_or_path=dinov3_name,
            patch_size=self.patch_size,
            freeze=freeze_backbone,
            dtype=dinov3_dtype,
        )
        self.backbone.load()
        assert self.backbone.embed_dim is not None and self.backbone.embed_dim > 0

        self.fine = FineCNN(in_ch=3, channels=int(fine_channels), num_blocks=int(fine_blocks))
        in_ch = int(fine_channels) + int(self.backbone.embed_dim)

        self.heads = Heads(
            in_ch=in_ch,
            descriptor_dim=int(descriptor_dim),
            use_offset=bool(use_offset),
            use_reliability=bool(use_reliability),
            max_offset=0.5,
        )

    def forward(self, x: torch.Tensor) -> FeatureOutputs:
        fine = self.fine(x)  # (B,Cf,H/4,W/4)

        # DINO tokens at stride 16, then upsample to stride 4
        dino = self.backbone(x).tokens
        dino_up = F.interpolate(dino, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        dino_up = dino_up.to(dtype=fine.dtype)

        feat = torch.cat([fine, dino_up], dim=1)
        return self.heads(feat)
