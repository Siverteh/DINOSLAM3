from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import os
from contextlib import nullcontext

import torch
import torch.nn as nn


@dataclass
class DinoTokens:
    tokens: torch.Tensor
    hidden_states: Dict[int, torch.Tensor] | None = None
    register_tokens: torch.Tensor | None = None
    pooled_register_tokens: torch.Tensor | None = None


class DinoV3Backbone(nn.Module):
    """
    DINOv3 backbone via HuggingFace Transformers.
    Returns patch tokens reshaped to (B, C, H/ps, W/ps).
    """

    def __init__(
        self,
        name_or_path: str,
        patch_size: int = 16,
        freeze: bool = True,
        dtype: str = "bf16",
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
        self._runtime_force_fp32: bool = False

    def _compute_dtype(self) -> torch.dtype:
        if self.dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.float32

    def load(self) -> None:
        from transformers import AutoModel

        requested_name = os.environ.get("DINOSLAM3_DINOV3_NAME_OR_PATH", self.name_or_path)
        allow_fallback = os.environ.get("DINOSLAM3_ALLOW_BACKBONE_FALLBACK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        fallback_map = {
            "facebook/dinov3-vits16-pretrain-lvd1689m": "timm/vit_small_patch16_dinov3.lvd1689m"
        }
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
                        "set DINOSLAM3_ALLOW_BACKBONE_FALLBACK=1."
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

    def _ensure_on_device(self, device: torch.device) -> None:
        assert self.model is not None
        cur_dev = next(self.model.parameters()).device
        if cur_dev != device:
            self.model.to(device=device)
            self.model.eval()
            if self.freeze:
                for p in self.model.parameters():
                    p.requires_grad_(False)

    def _reshape_patch_tokens(self, tokens: torch.Tensor, h_tokens: int, w_tokens: int) -> torch.Tensor:
        return tokens.transpose(1, 2).contiguous().reshape(tokens.shape[0], -1, h_tokens, w_tokens)

    def _extract_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        start = 1 + int(self.num_register_tokens)
        return tokens[:, start:, :]

    def _extract_register_tokens(self, tokens: torch.Tensor) -> torch.Tensor | None:
        if self.num_register_tokens <= 0:
            return None
        start = 1
        end = start + int(self.num_register_tokens)
        return tokens[:, start:end, :]

    def has_trainable_params(self) -> bool:
        assert self.model is not None and self._loaded, "Call load() before querying trainable params"
        return any(bool(p.requires_grad) for p in self.model.parameters())

    def set_trainable_top_blocks(self, num_blocks: int, *, train_norm: bool = True) -> None:
        assert self.model is not None and self._loaded, "Call load() before changing trainable blocks"

        for p in self.model.parameters():
            p.requires_grad_(False)

        num_blocks = int(max(0, num_blocks))
        layers = getattr(self.model, "layer", None)
        if layers is None and hasattr(self.model, "encoder"):
            layers = getattr(self.model.encoder, "layer", None)
        if layers is None and hasattr(self.model, "timm_model"):
            layers = getattr(self.model.timm_model, "blocks", None)

        if num_blocks > 0:
            if layers is None:
                raise RuntimeError(
                    f"Backbone '{self.name_or_path}' does not expose transformer blocks through a supported attribute."
                )
            for block in list(layers)[-num_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)

        if bool(train_norm):
            norm = getattr(self.model, "norm", None)
            if norm is None and hasattr(self.model, "timm_model"):
                norm = getattr(self.model.timm_model, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad_(True)

        self.freeze = not self.has_trainable_params()

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_hidden_states: bool = False,
        hidden_state_indices: Sequence[int] | None = None,
    ) -> DinoTokens:
        assert self.model is not None and self._loaded, "Call load() before forward()"
        assert x.dim() == 4 and x.size(1) == 3, "Expected input (B,3,H,W)"

        x = x.contiguous()

        _, _, H, W = x.shape
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise ValueError(f"Input must be padded to patch size {ps}. Got H={H}, W={W}")

        self._ensure_on_device(x.device)

        compute_dtype = torch.float32 if self._runtime_force_fp32 else self._compute_dtype()
        use_cuda_amp = x.is_cuda and compute_dtype != torch.float32
        amp_ctx = (
            torch.autocast(device_type="cuda", enabled=True, dtype=compute_dtype)
            if use_cuda_amp
            else (torch.autocast(device_type="cuda", enabled=False) if x.is_cuda else nullcontext())
        )
        try:
            with amp_ctx:
                model_kwargs = {
                    "pixel_values": x if use_cuda_amp else x.float(),
                    "return_dict": True,
                }
                if return_hidden_states:
                    model_kwargs["output_hidden_states"] = True
                out = self.model(**model_kwargs)
        except RuntimeError as exc:
            if use_cuda_amp and "CUBLAS_STATUS_INVALID_VALUE" in str(exc):
                self._runtime_force_fp32 = True
                print("[DinoV3Backbone] WARNING: bf16/fp16 GEMM failed on this stack; falling back to fp32 backbone.")
                with (torch.autocast(device_type="cuda", enabled=False) if x.is_cuda else nullcontext()):
                    model_kwargs = {
                        "pixel_values": x.float(),
                        "return_dict": True,
                    }
                    if return_hidden_states:
                        model_kwargs["output_hidden_states"] = True
                    out = self.model(**model_kwargs)
            else:
                raise

        tokens = out.last_hidden_state
        patch_tokens = self._extract_patch_tokens(tokens)
        register_tokens = self._extract_register_tokens(tokens)
        pooled_register_tokens = None if register_tokens is None else register_tokens.mean(dim=1)

        Ht = H // ps
        Wt = W // ps
        patch_tokens = self._reshape_patch_tokens(patch_tokens, Ht, Wt)

        hidden_states_out: Dict[int, torch.Tensor] | None = None
        if return_hidden_states:
            raw_hidden_states = getattr(out, "hidden_states", None)
            if raw_hidden_states is None:
                raise RuntimeError(
                    f"Backbone '{self.name_or_path}' did not return hidden states."
                )
            requested = tuple(int(i) for i in (hidden_state_indices or ()))
            hidden_states_out = {}
            for block_idx in requested:
                tuple_idx = int(block_idx) + 1
                if tuple_idx < 0 or tuple_idx >= len(raw_hidden_states):
                    raise IndexError(
                        f"Requested encoder block {block_idx}, but backbone only exposed "
                        f"{max(0, len(raw_hidden_states) - 1)} encoder outputs."
                    )
                hs_patch = self._extract_patch_tokens(raw_hidden_states[tuple_idx])
                hidden_states_out[block_idx] = self._reshape_patch_tokens(hs_patch, Ht, Wt)

        return DinoTokens(
            tokens=patch_tokens,
            hidden_states=hidden_states_out,
            register_tokens=register_tokens,
            pooled_register_tokens=pooled_register_tokens,
        )
