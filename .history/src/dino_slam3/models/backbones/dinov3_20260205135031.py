import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
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

        self.model = None
        self.embed_dim = None
        self.patch_size = 16

        self.feature_norm = None
        self.n_register_tokens = 0  # will infer if possible

    def load(self) -> None:
        # 1) create DINOv3 model
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            dynamic_img_size=True,   # allows variable H,W (still must be divisible by patch size)
        )

        # 2) infer embed dim
        # timm ViT models commonly expose embed_dim; fallback to num_features
        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Could not infer embed_dim from timm model.")

        # 3) optional: feature BN (not strictly required; see notes below)
        if self.use_feature_bn:
            self.feature_norm = nn.BatchNorm1d(self.embed_dim, affine=True)

        # 4) try to infer register/storage tokens count (optional)
        # Some DINOv3 variants use "reg" or "register" tokens in addition to CLS.
        # timm may expose them in different ways; safest is to remove tokens by *count inference at runtime*.
        # We'll keep n_register_tokens as 0 here and infer in forward by token count vs grid.
        self.n_register_tokens = 0

        # 5) freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns token embeddings [B, N, C] from timm ViT.
        Uses forward_features if available.
        """
        if hasattr(self.model, "forward_features"):
            tokens = self.model.forward_features(x)
            # timm ViT forward_features sometimes returns:
            # - [B, N, C]
            # - dict with 'x' or similar
            if isinstance(tokens, dict):
                # common keys: 'x', 'last_hidden_state'
                if "x" in tokens:
                    tokens = tokens["x"]
                elif "last_hidden_state" in tokens:
                    tokens = tokens["last_hidden_state"]
                else:
                    raise RuntimeError(f"Unknown forward_features dict keys: {list(tokens.keys())}")
        else:
            raise RuntimeError("timm model has no forward_features(); cannot extract tokens reliably.")
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected tokens [B,N,C], got shape {tuple(tokens.shape)}")
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns dense map: [B, C, H/16, W/16]
        """
        B, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input H,W must be divisible by {self.patch_size}, got {H}x{W}")

        tokens = self._forward_tokens(x)  # [B, N, C]
        B2, N, C = tokens.shape
        assert B2 == B and C == self.embed_dim

        gh, gw = H // self.patch_size, W // self.patch_size
        n_patches = gh * gw

        # Typical ViT token layout:
        # [CLS] + (optional register tokens) + patch tokens
        # So patch tokens are the last n_patches tokens in many implementations.
        if self.remove_register_tokens:
            if N < (1 + n_patches):
                raise RuntimeError(f"Token count {N} < 1+patches {1+n_patches}. Cannot slice.")
            patch_tokens = tokens[:, -n_patches:, :]  # take last patches
        else:
            # if you want ALL tokens (rarely useful for matching)
            patch_tokens = tokens[:, 1:1+n_patches, :]

        # Optional BN over feature dimension (flatten tokens first)
        if self.feature_norm is not None:
            flat = patch_tokens.reshape(-1, C)  # [B*n_patches, C]
            flat = self.feature_norm(flat)
            patch_tokens = flat.view(B, n_patches, C)

        # reshape to feature map
        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)  # [B,C,gh,gw]
        return feat
