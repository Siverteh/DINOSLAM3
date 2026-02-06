from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.dinov3 import DinoV3Backbone
from .fine_cnn import FineCNN
from .heads import HybridHeads, FeatureOutputs


class LocalFeatureNet(nn.Module):
    """
    DINOv3 tokens (stride=16) + FineCNN (stride=4) -> detector/descriptor/offset/reliability at stride=4.

    Heads are built in __init__ so the optimizer includes them.
    """

    def __init__(
        self,
        patch_size: int = 16,
        descriptor_dim: int = 128,
        fine_channels: int = 64,
        fine_blocks: int = 6,
        use_offset: bool = True,
        use_reliability: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = DinoV3Backbone(patch_size=patch_size, freeze=freeze_backbone)
        self.backbone.load()
        assert self.backbone.embed_dim is not None, "DINOv3 embed_dim not set after load()"

        self.fine = FineCNN(in_ch=3, channels=fine_channels, num_blocks=fine_blocks, out_stride=4)

        in_ch = int(fine_channels) + int(self.backbone.embed_dim)

        self._heads = HybridHeads(
            in_ch=in_ch,
            descriptor_dim=int(descriptor_dim),
            use_offset=bool(use_offset),
            use_reliability=bool(use_reliability),
            max_offset=0.5,
        )

    def forward(self, x: torch.Tensor) -> FeatureOutputs:
        fine = self.fine(x)  # B,Cf,H/4,W/4

        # EVA backbone in fp32 for stability
        with torch.autocast("cuda", enabled=False):
            dino = self.backbone(x.float()).tokens  # B,Cd,H/16,W/16 (fp32)

        Hf, Wf = fine.shape[-2:]
        dino_up = F.interpolate(dino, size=(Hf, Wf), mode="bilinear", align_corners=False)

        if dino_up.dtype != fine.dtype:
            dino_up = dino_up.to(dtype=fine.dtype)

        feat = torch.cat([fine, dino_up], dim=1)
        return self._heads(feat)
