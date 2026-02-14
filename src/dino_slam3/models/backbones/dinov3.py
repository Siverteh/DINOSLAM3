from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

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
        requested_name = os.environ.get("DINOSLAM3_DINOV3_NAME_OR_PATH", self.name_or_path)
        allow_fallback = os.environ.get("DINOSLAM3_ALLOW_BACKBONE_FALLBACK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        fallback_map = {
            # Ungated equivalent used when the facebook repo is inaccessible.
            "facebook/dinov3-vits16-pretrain-lvd1689m": "timm/vit_small_patch16_dinov3.lvd1689m"
        }
        # Load on CPU first; we'll move to GPU on first forward() based on x.device.
        try:
            self.model = AutoModel.from_pretrained(requested_name, trust_remote_code=True)
            self.name_or_path = requested_name
        except OSError as e:
            fallback_name = fallback_map.get(requested_name)
            if fallback_name is not None and allow_fallback:
                print(
                    f"[DinoV3Backbone] WARNING: failed to load '{requested_name}' ({e}). "
                    f"Falling back to '{fallback_name}' because DINOSLAM3_ALLOW_BACKBONE_FALLBACK is enabled."
                )
                self.model = AutoModel.from_pretrained(fallback_name, trust_remote_code=True)
                self.name_or_path = fallback_name
            else:
                hint = (
                    "Run `hf auth login` with a valid token that has access to the requested model. "
                    f"Requested model: '{requested_name}'."
                )
                if fallback_name is not None:
                    hint += (
                        f" If you intentionally want the ungated fallback '{fallback_name}', "
                        "set DINOSLAM3_ALLOW_BACKBONE_FALLBACK=1 (this may be incompatible with checkpoints trained on the gated backbone)."
                    )
                raise RuntimeError(
                    f"[DinoV3Backbone] Cannot load backbone '{requested_name}'. {hint}"
                ) from e
        self.model.eval()

        cfg = getattr(self.model, "config", None)
        self.embed_dim = 0
        self.num_register_tokens = 0
        if cfg is not None:
            self.embed_dim = int(
                getattr(cfg, "hidden_size", 0)
                or getattr(cfg, "embed_dim", 0)
                or getattr(cfg, "num_features", 0)
                or 0
            )
            self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0) or 0)

        if self.embed_dim <= 0:
            tm = getattr(self.model, "timm_model", None)
            self.embed_dim = int(
                getattr(self.model, "embed_dim", 0)
                or getattr(self.model, "num_features", 0)
                or getattr(tm, "embed_dim", 0)
                or getattr(tm, "num_features", 0)
                or 0
            )

        if self.num_register_tokens <= 0:
            reg = getattr(self.model, "reg_token", None)
            if reg is None and hasattr(self.model, "timm_model"):
                reg = getattr(self.model.timm_model, "reg_token", None)
            if reg is not None and hasattr(reg, "shape") and len(reg.shape) >= 2:
                self.num_register_tokens = int(reg.shape[1])

        if self.embed_dim <= 0:
            # Final fallback from known weight tensor shape.
            for k, v in self.model.state_dict().items():
                if k.endswith("patch_embed.proj.weight") and hasattr(v, "shape") and len(v.shape) >= 1:
                    self.embed_dim = int(v.shape[0])
                    break

        if self.embed_dim <= 0:
            raise RuntimeError(
                f"Could not infer DINO backbone embed_dim for '{self.name_or_path}'."
            )

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
