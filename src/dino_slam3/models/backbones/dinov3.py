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
        """
        Return token embeddings [B, N, C] from timm ViT/EVA.

        HARD FIX:
          - Always run backbone in FP32
          - Always disable autocast inside backbone to avoid BF16 GEMMEx issues on some stacks
        """
        if self.model is None:
            raise RuntimeError("DinoV3Backbone.model is None. Did you call load()?")

        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("timm model has no forward_features(); cannot extract tokens reliably.")

        # Ensure model is on correct device and in fp32
        if next(self.model.parameters()).device != x.device:
            self.model.to(device=x.device)
        if next(self.model.parameters()).dtype != torch.float32:
            self.model.to(dtype=torch.float32)

        # Disable autocast inside backbone and cast input to fp32
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            tokens = self.model.forward_features(x_fp32)

        # timm may return dict
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

    def forward(self, x: torch.Tensor):
        """
        Returns DinoV3Output(tokens=[B,C,H/16,W/16])

        Backbone always runs FP32 (see _forward_tokens).
        Output tokens are FP32; the rest of the network can still run under AMP.
        """
        if self.model is None:
            self.load()

        B, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input H,W must be divisible by {self.patch_size}, got {H}x{W}")

        tokens = self._forward_tokens(x)  # [B, N, C] FP32
        B2, N, C = tokens.shape

        if self.embed_dim is None:
            raise RuntimeError("embed_dim is None; load() did not set it.")
        if B2 != B:
            raise RuntimeError(f"Token batch {B2} != input batch {B}")
        if C != self.embed_dim:
            raise RuntimeError(f"Token dim C={C} does not match embed_dim={self.embed_dim}")

        gh, gw = H // self.patch_size, W // self.patch_size
        n_patches = gh * gw

        # robust slicing: patch tokens are last n_patches tokens
        if self.remove_register_tokens:
            if N < (1 + n_patches):
                raise RuntimeError(f"Token count {N} < 1+patches {1+n_patches}. Cannot slice.")
            patch_tokens = tokens[:, -n_patches:, :]  # [B, n_patches, C]
        else:
            patch_tokens = tokens[:, 1:1 + n_patches, :]

        # optional BN (fp32)
        if self.feature_norm is not None:
            flat = patch_tokens.reshape(-1, C)
            flat = self.feature_norm(flat)
            patch_tokens = flat.view(B, n_patches, C)

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)  # [B,C,gh,gw] fp32
        return DinoV3Output(tokens=feat)
