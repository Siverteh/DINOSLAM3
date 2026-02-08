from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class DinoTokens:
    tokens: torch.Tensor  # (B, C, H/ps, W/ps)


class DinoV3Backbone(nn.Module):
    """
    DINOv3 backbone via HuggingFace Transformers.
    Returns patch tokens reshaped to (B, C, H/ps, W/ps).

    H100 FIX (bulletproof):
      - Run backbone in pure FP32 with autocast disabled.
      - This avoids cublasGemmEx fp16/bf16 INVALID_VALUE issues seen on some H100 stacks.
      - Keep AMP for the rest of your network (heads) outside this backbone.
    """

    def __init__(
        self,
        name_or_path: str,
        patch_size: int = 16,
        freeze: bool = True,
        dtype: str = "fp32",  # ignored on CUDA: we force fp32
    ):
        super().__init__()
        self.name_or_path = str(name_or_path)
        self.patch_size = int(patch_size)
        self.freeze = bool(freeze)
        self.dtype_name = str(dtype).lower()

        self.model: Optional[nn.Module] = None
        self.embed_dim: Optional[int] = None
        self.num_register_tokens: int = 0
        self._loaded: bool = False

    def load(self) -> None:
        # Load on CPU first; we'll move to GPU on first forward() based on x.device.
        self.model = AutoModel.from_pretrained(self.name_or_path, trust_remote_code=True)
        self.model.eval()

        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            self.embed_dim = int(getattr(cfg, "hidden_size", getattr(cfg, "embed_dim", 0)) or 0)
            self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0) or 0)
        else:
            self.embed_dim = None
            self.num_register_tokens = 0

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self._loaded = True

    def _ensure_on_device_fp32(self, device: torch.device) -> None:
        assert self.model is not None
        cur_dev = next(self.model.parameters()).device
        cur_dtype = next(self.model.parameters()).dtype

        if cur_dev != device or cur_dtype != torch.float32:
            self.model.to(device=device, dtype=torch.float32)
            self.model.eval()
            if self.freeze:
                for p in self.model.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> DinoTokens:
        assert self.model is not None and self._loaded, "Call load() before forward()"
        assert x.dim() == 4 and x.size(1) == 3, "Expected input (B,3,H,W)"

        # Avoid weird strides/layouts that can trigger GEMM edge-cases
        x = x.contiguous()

        B, _, H, W = x.shape
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise ValueError(f"Input must be padded to patch size {ps}. Got H={H}, W={W}")

        # Force backbone to FP32 on whatever device x is on
        self._ensure_on_device_fp32(x.device)

        # Run backbone with autocast OFF (pure fp32)
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.model(pixel_values=x.float(), return_dict=True)

        tokens = out.last_hidden_state  # (B, seq, C)

        # Layout: [CLS] + [REG]*n + [PATCHES]
        start = 1 + int(self.num_register_tokens)
        patch_tokens = tokens[:, start:, :]  # (B, N, C)

        Ht = H // ps
        Wt = W // ps

        # (B, N, C) -> (B, C, Ht, Wt)
        patch_tokens = patch_tokens.transpose(1, 2).contiguous().reshape(B, -1, Ht, Wt)
        return DinoTokens(tokens=patch_tokens)
