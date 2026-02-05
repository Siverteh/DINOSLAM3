import timm
import torch
from torch import amp
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class DinoV3Output:
    tokens: torch.Tensor  # [B, C, H/patch, W/patch]


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze: bool = True,
        use_feature_bn: bool = False,
        remove_register_tokens: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_feature_bn = use_feature_bn
        self.remove_register_tokens = remove_register_tokens

        # IMPORTANT: must match what network.py passes in
        self.patch_size = int(patch_size)

        self.model: Optional[nn.Module] = None
        self.embed_dim: Optional[int] = None

        self.feature_norm: Optional[nn.BatchNorm1d] = None

        # If you want eager loading, uncomment next line:
        # self.load()

    def load(self) -> None:
        # Create model
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            dynamic_img_size=True,
        )

        # Infer embed dim
        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Could not infer embed_dim from timm model.")

        # Optional BN over feature dimension
        if self.use_feature_bn:
            self.feature_norm = nn.BatchNorm1d(self.embed_dim, affine=True)

        # Freeze backbone
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DinoV3Backbone.model is None. Did you call load()?")

        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("timm model has no forward_features(); cannot extract tokens reliably.")

        # Make sure patch_embed conv params match the dtype that autocast uses for x
        # (this prevents Half-vs-float bias mismatches)
        pe = getattr(self.model, "patch_embed", None)
        if pe is not None and hasattr(pe, "proj"):
            proj = pe.proj
            # match input dtype (x is often fp16 under autocast)
            target_dtype = x.dtype
            if proj.weight.dtype != target_dtype:
                proj.weight.data = proj.weight.data.to(target_dtype)
            if proj.bias is not None and proj.bias.dtype != target_dtype:
                proj.bias.data = proj.bias.data.to(target_dtype)

        print("x dtype:", x.dtype, "proj.weight:", self.model.patch_embed.proj.weight.dtype, "proj.bias:", self.model.patch_embed.proj.bias.dtype)
        tokens = self.model.forward_features(x)

        if isinstance(tokens, dict):
            if "x" in tokens:
                tokens = tokens["x"]
            elif "last_hidden_state" in tokens:
                tokens = tokens["last_hidden_state"]
            else:
                raise RuntimeError(f"Unknown forward_features dict keys: {list(tokens.keys())}")

        if tokens.dim() != 3:
            raise RuntimeError(f"Expected tokens [B,N,C], got shape {tuple(tokens.shape)}")

        return tokens



    def forward(self, x: torch.Tensor) -> DinoV3Output:
        """
        Returns DinoV3Output(tokens=[B,C,H/patch,W/patch])

        Fixes:
        - Lazy-load timm model on CPU then input on CUDA -> moves model to x.device
        - dtype/device mismatch inside timm patch_embed.proj -> aligns proj params to x
        """
        # --- lazy load (model created on CPU by default) ---
        if self.model is None:
            self.load()

        # --- ensure the timm model is on the same device as the input ---
        # (important because load() happens after parent .to(device) in your net)
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = x.device  # edge case: no parameters

        if model_device != x.device:
            self.model.to(device=x.device)

        B, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input H,W must be divisible by {self.patch_size}, got {H}x{W}")

        # --- HARD GUARANTEE: patch_embed.proj weights/bias match x device + dtype ---
        # This prevents: "Input type (...) and weight/bias type (...) should be the same"
        pe = getattr(self.model, "patch_embed", None)
        if pe is not None and hasattr(pe, "proj"):
            proj = pe.proj
            # move module first (device), then fix dtype
            if proj.weight.device != x.device:
                proj.to(device=x.device)
            if proj.weight.dtype != x.dtype:
                proj.weight.data = proj.weight.data.to(dtype=x.dtype)
            if proj.bias is not None and proj.bias.dtype != x.dtype:
                proj.bias.data = proj.bias.data.to(dtype=x.dtype)

        # --- forward tokens ---
        tokens = self._forward_tokens(x)  # [B, N, C]
        B2, N, C = tokens.shape

        if self.embed_dim is None:
            raise RuntimeError("embed_dim is None; load() did not set it.")
        if B2 != B:
            raise RuntimeError(f"Token batch {B2} != input batch {B}")
        if C != self.embed_dim:
            raise RuntimeError(f"Token dim C={C} does not match embed_dim={self.embed_dim}")

        gh, gw = H // self.patch_size, W // self.patch_size
        n_patches = gh * gw

        # Robust slicing: last n_patches tokens are the spatial grid (drops CLS + register tokens)
        if self.remove_register_tokens:
            if N < (1 + n_patches):
                raise RuntimeError(f"Token count {N} < 1+patches {1+n_patches}. Cannot slice.")
            patch_tokens = tokens[:, -n_patches:, :]  # [B, n_patches, C]
        else:
            patch_tokens = tokens[:, 1:1 + n_patches, :]

        # Optional BN over feature dimension
        if self.feature_norm is not None:
            flat = patch_tokens.reshape(-1, C)  # [B*n_patches, C]
            flat = self.feature_norm(flat)
            patch_tokens = flat.view(B, n_patches, C)

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)  # [B,C,gh,gw]
        return DinoV3Output(tokens=feat)
