from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureOutputs:
    heatmap: torch.Tensor                 # B,1,Hf,Wf (logits)
    desc: torch.Tensor                    # B,D,Hf,Wf (L2-normalized)
    offset: Optional[torch.Tensor]        # B,2,Hf,Wf (feature-coord delta; sub-cell)
    reliability: Optional[torch.Tensor]   # B,1,Hf,Wf (logits)


class HybridHeads(nn.Module):
    def __init__(
        self,
        in_ch: int,
        descriptor_dim: int = 128,
        use_offset: bool = True,
        use_reliability: bool = True,
        max_offset: float = 0.5,  # in feature coords, i.e. +-0.5 cell
    ):
        super().__init__()
        hid = max(128, in_ch)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.detector = nn.Conv2d(hid, 1, 1)
        self.descriptor = nn.Conv2d(hid, descriptor_dim, 1)

        self.use_offset = bool(use_offset)
        self.use_reliability = bool(use_reliability)
        self.max_offset = float(max_offset)

        self.offset_head = nn.Conv2d(hid, 2, 1) if self.use_offset else None
        self.rel_head = nn.Conv2d(hid, 1, 1) if self.use_reliability else None

    def forward(self, feat: torch.Tensor) -> FeatureOutputs:
        h = self.shared(feat)

        heat = self.detector(h)
        desc = F.normalize(self.descriptor(h), dim=1, eps=1e-6)

        off = None
        if self.offset_head is not None:
            off = torch.tanh(self.offset_head(h)) * self.max_offset

        rel = self.rel_head(h) if self.rel_head is not None else None

        return FeatureOutputs(heatmap=heat, desc=desc, offset=off, reliability=rel)
