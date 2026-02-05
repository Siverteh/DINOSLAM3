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

        # IMPORTANT: force DINO forward in FP32, outside autocast
        device_type = "cuda" if x.is_cuda else "cpu"
        with amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            tokens = self.model.forward_features(x_fp32)

        # timm sometimes returns a dict
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
        """
        # Lazy load to avoid init-time downloads / failures
        if self.model is None:
            self.load()

        B, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input H,W must be divisible by {self.patch_size}, got {H}x{W}")

        tokens = self._forward_tokens(x)  # [B, N, C]
        B2, N, C = tokens.shape

        if self.embed_dim is None:
            raise RuntimeError("embed_dim is None; load() did not set it.")
        if C != self.embed_dim:
            raise RuntimeError(f"Token dim C={C} does not match embed_dim={self.embed_dim}")

        gh, gw = H // self.patch_size, W // self.patch_size
        n_patches = gh * gw

        # Robust: assume patch tokens are last n_patches tokens
        # (drops CLS + any register tokens automatically)
        if self.remove_register_tokens:
            if N < (1 + n_patches):
                raise RuntimeError(f"Token count {N} < 1+patches {1+n_patches}. Cannot slice.")
            patch_tokens = tokens[:, -n_patches:, :]  # [B, n_patches, C]
        else:
            # Keep tokens right after CLS (less robust if register tokens exist)
            patch_tokens = tokens[:, 1:1 + n_patches, :]

        # Optional BN over feature dimension
        if self.feature_norm is not None:
            flat = patch_tokens.reshape(-1, C)  # [B*n_patches, C]
            flat = self.feature_norm(flat)
            patch_tokens = flat.view(B, n_patches, C)

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)  # [B,C,gh,gw]
        return DinoV3Output(tokens=feat)
