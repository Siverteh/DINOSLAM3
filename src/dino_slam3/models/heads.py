from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FeatureOutputs:
    heatmap: torch.Tensor          # B,1,Hf,Wf  (detector logits)
    desc: torch.Tensor             # B,D,Hf,Wf
    offset: torch.Tensor | None    # B,2,Hf,Wf  (dx,dy in pixels at full-res)
    reliability: torch.Tensor | None  # B,1,Hf,Wf (logits)

class HybridHeads(nn.Module):
    """Heads operating on concatenated [fine_cnn_features, dino_tokens_upsampled]."""

    def __init__(self, in_ch: int, descriptor_dim: int = 128, use_offset: bool = True, use_reliability: bool = True):
        super().__init__()
        hid = max(64, in_ch)
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.detector = nn.Conv2d(hid, 1, 1)
        self.descriptor = nn.Sequential(
            nn.Conv2d(hid, descriptor_dim, 1),
        )

        self.use_offset = use_offset
        self.use_reliability = use_reliability
        self.offset = nn.Conv2d(hid, 2, 1) if use_offset else None
        self.reliability = nn.Conv2d(hid, 1, 1) if use_reliability else None

    def forward(self, feat: torch.Tensor) -> FeatureOutputs:
        h = self.shared(feat)
        heat = self.detector(h)
        desc = self.descriptor(h)
        desc = F.normalize(desc, dim=1, eps=1e-6)

        off = self.offset(h) if self.use_offset else None
        rel = self.reliability(h) if self.use_reliability else None
        return FeatureOutputs(heatmap=heat, desc=desc, offset=off, reliability=rel)
