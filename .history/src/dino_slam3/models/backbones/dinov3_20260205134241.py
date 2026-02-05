from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class DinoV3Output:
    tokens: torch.Tensor  # B,C,Ht,Wt (Ht=H/patch, Wt=W/patch)

class DinoV3Backbone(nn.Module):
    """Thin wrapper around a DINOv3 model.

    This repository intentionally does NOT ship DINOv3 weights.
    You must implement `load()` to return a nn.Module that, when called on an
    ImageNet-normalized tensor (B,3,H,W), returns patch tokens.

    Required output format from `forward()`:
      tokens: (B, C, H/patch, W/patch)
    """

    def __init__(self, patch_size: int = 16, freeze: bool = True):
        super().__init__()
        self.patch_size = int(patch_size)
        self.freeze = bool(freeze)
        self.model: Optional[nn.Module] = None
        self.embed_dim: Optional[int] = None

    def load(self) -> None:
        """USER: implement this to load your DINOv3.

        Examples (choose one):
        - torch.hub load (if you have the repo)
        - local checkpoint + custom DINOv3 module
        - HF transformers (if you wrap outputs)

        After loading, set:
          self.model
          self.embed_dim
        """
        raise NotImplementedError(
            "Implement DinoV3Backbone.load() with your DINOv3 model/weights."
        )

    def forward(self, x: torch.Tensor) -> DinoV3Output:
        if self.model is None or self.embed_dim is None:
            self.load()

        with torch.set_grad_enabled(not self.freeze):
            out = self.model(x)

        # Accept a few common shapes and normalize to B,C,Ht,Wt
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.dim() == 3:
            # B, N, C -> reshape tokens
            B, N, C = out.shape
            # assume includes cls token?
            # if N == (Ht*Wt)+1: drop first token
            H = x.shape[-2] // self.patch_size
            W = x.shape[-1] // self.patch_size
            if N == H * W + 1:
                out = out[:, 1:, :]
                N = H * W
            out = out.transpose(1, 2).contiguous().view(B, C, H, W)
        elif out.dim() == 4:
            # could be B,C,Ht,Wt already
            pass
        else:
            raise ValueError(f"Unexpected DINO output shape: {tuple(out.shape)}")

        return DinoV3Output(tokens=out)
