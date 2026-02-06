from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn


@dataclass
class DinoV3Output:
    tokens: torch.Tensor  # [B, C, H/patch, W/patch]


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze: bool = True,
        remove_register_tokens: bool = True,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.model_name = model_name
        self.freeze = bool(freeze)
        self.remove_register_tokens = bool(remove_register_tokens)

        self.model: Optional[nn.Module] = None
        self.embed_dim: Optional[int] = None

    def load(self) -> None:
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            dynamic_img_size=True,
        )
        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Could not infer embed_dim from timm model.")

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> DinoV3Output:
        if self.model is None:
            self.load()

        # Ensure device placement
        if next(self.model.parameters()).device != x.device:
            self.model.to(x.device)

        B, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input H,W must be divisible by {self.patch_size}, got {H}x{W}")

        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("timm model has no forward_features(); cannot extract tokens reliably.")

        tokens = self.model.forward_features(x)
        if isinstance(tokens, dict):
            tokens = tokens.get("x", tokens.get("last_hidden_state", None))
        if tokens is None or tokens.dim() != 3:
            raise RuntimeError("Unexpected token output shape/type from timm forward_features().")

        _, N, C = tokens.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        n_patches = gh * gw

        # Slice off cls/register tokens robustly
        if self.remove_register_tokens:
            if N < (1 + n_patches):
                raise RuntimeError(f"Token count {N} < 1+patches {1+n_patches}. Cannot slice.")
            patch_tokens = tokens[:, -n_patches:, :]
        else:
            patch_tokens = tokens[:, 1 : 1 + n_patches, :]

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)
        return DinoV3Output(tokens=feat)
