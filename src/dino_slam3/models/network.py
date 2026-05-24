from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.dinov3 import DinoV3Backbone
from .fine_cnn import FineCNN
from .heads import Heads, FeatureOutputs


class _FusionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        return self.act(self.refine(y) + y)


class _FusionBlockV3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        local = self.local(y)
        context = self.context(y)
        fused = self.mix(torch.cat([local, context], dim=1))
        return self.act(fused + y)


class _DepthwiseRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FusionBlockV4FPN(nn.Module):
    """
    Multi-scale fusion:
      - fine stage2 (stride=2) -> down to stride=4
      - fine stage4 (stride=4)
      - upsampled dino tokens (stride=4)
    """
    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.refine = _DepthwiseRefine(c)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)
        fused = self.mix(torch.cat([p2, p4, pd], dim=1))
        return self.refine(fused + p4)


class _LocalGatedAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)
        self.local_ctx = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        gate = torch.sigmoid((q * k).mean(dim=1, keepdim=True))
        gated = v * gate
        ctx = self.local_ctx(x)
        y = self.out(torch.cat([gated, ctx], dim=1))
        return self.norm(y + x)


class _FusionBlockV5FPNAttn(nn.Module):
    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.base = _FusionBlockV4FPN(
            fine_channels=int(fine_channels),
            dino_channels=int(dino_channels),
            out_ch=c,
        )
        self.attn = _LocalGatedAttention(c)
        self.refine = _DepthwiseRefine(c)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        y = self.base(fine_s2=fine_s2, fine_s4=fine_s4, dino_s4=dino_s4)
        y = self.attn(y)
        return self.refine(y)


class _FusionBlockV6FPNXGate(nn.Module):
    """
    Cross-gated multi-scale fusion:
      - fine stage2 (stride=2) -> down to stride=4
      - fine stage4 (stride=4), gated by dino projection
      - dino stage4, gated by fine projections
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.gate_dino = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )
        self.gate_fine = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.refine = _DepthwiseRefine(c)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)

        g_d = self.gate_dino(torch.cat([p2, p4], dim=1))
        g_f = self.gate_fine(pd)

        fused = self.mix(torch.cat([p2, p4 * g_f, pd * g_d], dim=1))
        return self.refine(fused + p4)


class _FusionBlockV7FPNASPP(nn.Module):
    """
    FPN fusion with ASPP-style context refinement.
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.base = _FusionBlockV4FPN(
            fine_channels=int(fine_channels),
            dino_channels=int(dino_channels),
            out_ch=c,
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=4, dilation=4, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.b4_pool = nn.AdaptiveAvgPool2d(1)
        self.b4_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(4 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.refine = _DepthwiseRefine(c)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        y = self.base(fine_s2=fine_s2, fine_s4=fine_s4, dino_s4=dino_s4)
        b1 = self.b1(y)
        b2 = self.b2(y)
        b3 = self.b3(y)
        b4 = self.b4_proj(self.b4_pool(y))
        b4 = F.interpolate(b4, size=y.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.mix(torch.cat([b1, b2, b3, b4], dim=1))
        return self.refine(fused + y)


class _FusionBlockV8BiFPNLite(nn.Module):
    """
    Lightweight BiFPN-style weighted fusion at stride 4.
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        # Positive fusion weights (ReLU + normalize).
        self.w_top = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.w_out = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.top_refine = _DepthwiseRefine(c)
        self.out_refine = _DepthwiseRefine(c)
        self.mix = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _norm_w(w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + 1e-6)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)

        wt = self._norm_w(self.w_top)
        top = wt[0] * p4 + wt[1] * pd
        top = self.top_refine(top)

        wo = self._norm_w(self.w_out)
        out = wo[0] * p2 + wo[1] * top
        out = self.out_refine(out)
        return self.mix(out + p4)


class _FusionBlockV9FPNDeformLite(nn.Module):
    """
    FPN fusion with lightweight offset-conditioned local resampling.
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2, 1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.refine = _DepthwiseRefine(c)

    @staticmethod
    def _base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).unsqueeze(0)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)
        bsz, _, h, w = p4.shape
        off = torch.tanh(self.offset(torch.cat([p4, pd], dim=1))) * 2.0
        base_grid = self._base_grid(h=h, w=w, device=p4.device, dtype=p4.dtype)
        dx = off[:, 0] / float(max(w - 1, 1))
        dy = off[:, 1] / float(max(h - 1, 1))
        flow = torch.stack([dx, dy], dim=-1)
        grid = base_grid.expand(bsz, -1, -1, -1) + flow
        pd_shift = F.grid_sample(
            pd,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        fused = self.mix(torch.cat([p2, p4, pd_shift], dim=1))
        return self.refine(fused + p4)


class _FusionBlockV10TokenCrossAttn(nn.Module):
    """
    Lightweight local cross-attention from fine features to DINO token map.
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)
        self.token_ctx = nn.Sequential(
            nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.refine = _DepthwiseRefine(c)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)
        q = self.q(p4)
        k = self.k(pd)
        v = self.v(pd)
        gate = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / (q.shape[1] ** 0.5))
        token_fused = self.token_ctx(v * gate + pd)
        fused = self.mix(torch.cat([p2, p4, token_fused], dim=1))
        return self.refine(fused + p4)


class _FusionBlockV11BiFPNDepthAux(nn.Module):
    """
    BiFPN-lite with train-time depth auxiliary context branch.
    """

    def __init__(self, fine_channels: int, dino_channels: int, out_ch: int):
        super().__init__()
        c = max(64, int(out_ch))
        self.s2_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s2_down = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.s4_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(int(dino_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv2d(int(fine_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.w_top = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.w_out = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.top_refine = _DepthwiseRefine(c)
        self.out_refine = _DepthwiseRefine(c)
        self.mix = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _norm_w(w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + 1e-6)

    def forward(
        self,
        fine_s2: torch.Tensor,
        fine_s4: torch.Tensor,
        dino_s4: torch.Tensor,
        depth_aux_s4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p2 = self.s2_down(self.s2_proj(fine_s2))
        p4 = self.s4_proj(fine_s4)
        pd = self.dino_proj(dino_s4)
        if depth_aux_s4 is None:
            pz = torch.zeros_like(p4)
        else:
            pz = self.depth_proj(depth_aux_s4)

        wt = self._norm_w(self.w_top)
        top = wt[0] * p4 + wt[1] * pd
        top = self.top_refine(top)

        wo = self._norm_w(self.w_out)
        out = wo[0] * p2 + wo[1] * top + wo[2] * pz
        out = self.out_refine(out)
        return self.mix(out + p4)


class _PseudoObjectAffinity(nn.Module):
    """
    Lightweight pseudo-object embedding from local token affinity.
    """

    def __init__(self, channels: int, k: int = 5, temperature: float = 10.0):
        super().__init__()
        c = int(channels)
        self.k = max(1, min(9, int(k)))
        self.temperature = float(max(1.0e-3, temperature))
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = x.shape
        x32 = x.float()
        xn = F.normalize(x32, dim=1, eps=1e-6)
        patches = F.unfold(xn, kernel_size=3, padding=1)  # (B, C*9, H*W)
        patches = patches.view(bsz, ch, 9, h * w)
        center = xn.view(bsz, ch, 1, h * w)
        sim = (patches * center).sum(dim=1)  # (B,9,HW)
        k = min(int(self.k), int(sim.shape[1]))
        vals, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        weights = F.softmax(vals * float(self.temperature), dim=1)  # (B,k,HW)
        idx_exp = idx.unsqueeze(1).expand(-1, ch, -1, -1)
        top_feat = torch.gather(patches, 2, idx_exp)  # (B,C,k,HW)
        obj = (top_feat * weights.unsqueeze(1)).sum(dim=2).view(bsz, ch, h, w)
        obj = self.refine(obj)
        g = self.gate(torch.cat([x32, obj], dim=1))
        out = g * obj + (1.0 - g) * x32
        return out.to(dtype=x.dtype)


class LocalFeatureNet(nn.Module):
    """
    DINOv3 tokens (stride=16) + FineCNN (stride=4) -> detector/descriptor/offset/reliability at stride=4.
    """
    def __init__(
        self,
        dinov3_name: str,
        patch_size: int = 16,
        descriptor_dim: int = 256,
        fine_channels: int = 96,
        fine_blocks: int = 8,
        freeze_backbone: bool = True,
        use_offset: bool = True,
        use_reliability: bool = True,
        dinov3_dtype: str = "bf16",
        head_variant: str = "v1",
        head_channels: int | None = None,
        head_tower_depth: int = 2,
        head_norm: str = "group",
        head_act: str = "silu",
        detector_logit_scale: float = 1.0,
        reliability_logit_scale: float = 1.0,
        fusion_variant: str = "v1",
        fusion_channels: int = 256,
        desc_relgate_detach: bool = False,
        pseudo_object_enabled: bool = False,
        pseudo_object_k: int = 5,
        pseudo_object_temperature: float = 10.0,
    ):
        super().__init__()
        self.patch_size = int(patch_size)

        self.backbone = DinoV3Backbone(
            name_or_path=dinov3_name,
            patch_size=self.patch_size,
            freeze=freeze_backbone,
            dtype=dinov3_dtype,
        )
        self.backbone.load()
        assert self.backbone.embed_dim is not None and self.backbone.embed_dim > 0

        self.fusion_variant = str(fusion_variant).lower()
        self.use_depth_aux = self.fusion_variant == "v11_bifpn_depthaux"
        self.fine = FineCNN(
            in_ch=3,
            channels=int(fine_channels),
            num_blocks=int(fine_blocks),
            enable_depth_aux=self.use_depth_aux,
        )
        in_ch = int(fine_channels) + int(self.backbone.embed_dim)
        if self.fusion_variant == "v2":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlock(in_ch=in_ch, out_ch=fch)
            head_in_ch = fch
        elif self.fusion_variant == "v3":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV3(in_ch=in_ch, out_ch=fch)
            head_in_ch = fch
        elif self.fusion_variant == "v4_fpn":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV4FPN(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v5_fpn_attn":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV5FPNAttn(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v6_fpn_xgate":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV6FPNXGate(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v7_fpn_aspp":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV7FPNASPP(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v8_bifpn_lite":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV8BiFPNLite(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v9_fpn_deformlite":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV9FPNDeformLite(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v10_token_crossattn":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV10TokenCrossAttn(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        elif self.fusion_variant == "v11_bifpn_depthaux":
            fch = max(64, int(fusion_channels))
            self.fusion = _FusionBlockV11BiFPNDepthAux(
                fine_channels=int(fine_channels),
                dino_channels=int(self.backbone.embed_dim),
                out_ch=fch,
            )
            head_in_ch = fch
        else:
            self.fusion = None
            head_in_ch = in_ch

        self.pseudo_object_enabled = bool(pseudo_object_enabled)
        if self.pseudo_object_enabled:
            self.pseudo_object = _PseudoObjectAffinity(
                channels=int(self.backbone.embed_dim),
                k=int(pseudo_object_k),
                temperature=float(pseudo_object_temperature),
            )
            self.pseudo_to_head = nn.Sequential(
                nn.Conv2d(int(self.backbone.embed_dim), int(head_in_ch), 1, bias=False),
                nn.BatchNorm2d(int(head_in_ch)),
                nn.SiLU(inplace=True),
            )
        else:
            self.pseudo_object = None
            self.pseudo_to_head = None

        self.heads = Heads(
            in_ch=head_in_ch,
            descriptor_dim=int(descriptor_dim),
            use_offset=bool(use_offset),
            use_reliability=bool(use_reliability),
            max_offset=0.5,
            variant=str(head_variant),
            head_channels=head_channels,
            tower_depth=int(head_tower_depth),
            norm=str(head_norm),
            act=str(head_act),
            detector_logit_scale=float(detector_logit_scale),
            reliability_logit_scale=float(reliability_logit_scale),
            desc_relgate_detach=bool(desc_relgate_detach),
        )

    def forward(self, x: torch.Tensor, depth: torch.Tensor | None = None) -> FeatureOutputs:
        fine_pyr = self.fine.forward_pyramid(x, depth=depth if self.use_depth_aux else None)
        fine_s2 = fine_pyr["stage2"]  # (B,Cf,H/2,W/2)
        fine_s4 = fine_pyr["stage4"]  # (B,Cf,H/4,W/4)
        depth_aux_s4 = fine_pyr.get("depth_s4")

        # DINO tokens at stride 16, then upsample to stride 4
        dino = self.backbone(x).tokens
        dino_up = F.interpolate(dino, size=fine_s4.shape[-2:], mode="bilinear", align_corners=False)
        dino_up = dino_up.to(dtype=fine_s4.dtype)
        pseudo_ctx = None
        if self.pseudo_object is not None:
            pseudo_ctx = self.pseudo_object(dino_up)
        if self.fusion_variant in {
            "v4_fpn",
            "v5_fpn_attn",
            "v6_fpn_xgate",
            "v7_fpn_aspp",
            "v8_bifpn_lite",
            "v9_fpn_deformlite",
            "v10_token_crossattn",
            "v11_bifpn_depthaux",
        } and self.fusion is not None:
            if self.fusion_variant == "v11_bifpn_depthaux":
                feat = self.fusion(
                    fine_s2=fine_s2,
                    fine_s4=fine_s4,
                    dino_s4=dino_up,
                    depth_aux_s4=depth_aux_s4,
                )
            else:
                feat = self.fusion(fine_s2=fine_s2, fine_s4=fine_s4, dino_s4=dino_up)
        else:
            feat = torch.cat([fine_s4, dino_up], dim=1)
            if self.fusion is not None:
                feat = self.fusion(feat)
        if pseudo_ctx is not None and self.pseudo_to_head is not None:
            feat = feat + self.pseudo_to_head(pseudo_ctx.to(dtype=feat.dtype))
        return self.heads(feat)
