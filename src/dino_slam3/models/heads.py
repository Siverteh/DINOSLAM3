from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FeatureOutputs:
    heatmap: torch.Tensor                 # (B,1,Hf,Wf) logits
    desc: torch.Tensor                    # (B,D,Hf,Wf) L2-normalized
    offset: Optional[torch.Tensor]        # (B,2,Hf,Wf) feature offset
    reliability: Optional[torch.Tensor]   # (B,1,Hf,Wf) logits

class Heads(nn.Module):
    def __init__(
        self,
        in_ch: int,
        descriptor_dim: int = 256,
        use_offset: bool = True,
        use_reliability: bool = True,
        max_offset: float = 0.5,
    ):
        super().__init__()
        hid = max(192, in_ch)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.det = nn.Conv2d(hid, 1, 1)
        self.desc = nn.Conv2d(hid, descriptor_dim, 1)

        self.use_offset = bool(use_offset)
        self.use_reliability = bool(use_reliability)
        self.max_offset = float(max_offset)

        self.off = nn.Conv2d(hid, 2, 1) if self.use_offset else None
        self.rel = nn.Conv2d(hid, 1, 1) if self.use_reliability else None

    def forward(self, feat: torch.Tensor) -> FeatureOutputs:
        h = self.shared(feat)
        heat = self.det(h)
        desc = F.normalize(self.desc(h), dim=1, eps=1e-6)

        off = None
        if self.off is not None:
            off = torch.tanh(self.off(h)) * self.max_offset

        rel = self.rel(h) if self.rel is not None else None
        return FeatureOutputs(heatmap=heat, desc=desc, offset=off, reliability=rel)
