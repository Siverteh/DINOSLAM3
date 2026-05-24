from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureOutputs:
    heatmap: torch.Tensor                 # (B,1,Hf,Wf) logits
    desc: torch.Tensor                    # (B,D,Hf,Wf) L2-normalized
    offset: Optional[torch.Tensor]        # (B,2,Hf,Wf) feature offset
    reliability: Optional[torch.Tensor]   # (B,1,Hf,Wf) logits


def _make_norm(norm: str, channels: int) -> nn.Module:
    name = str(norm).lower()
    if name in {"batch", "bn", "batchnorm"}:
        return nn.BatchNorm2d(channels)
    groups = min(16, int(channels))
    while groups > 1 and (int(channels) % groups) != 0:
        groups -= 1
    return nn.GroupNorm(max(1, groups), int(channels))


def _make_act(act: str) -> nn.Module:
    name = str(act).lower()
    if name in {"relu"}:
        return nn.ReLU(inplace=True)
    if name in {"gelu"}:
        return nn.GELU()
    return nn.SiLU(inplace=True)


class _ResBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "group", act: str = "silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _make_norm(norm, channels)
        self.act1 = _make_act(act)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(norm, channels)
        self.act2 = _make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y + x
        return self.act2(y)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, int(channels) // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.pool(x)
        gate = self.fc1(gate)
        gate = self.act(gate)
        gate = torch.sigmoid(self.fc2(gate))
        return x * gate


class _LayerNorm2d(nn.Module):
    """
    Channel-wise layer norm on BCHW tensors.
    """

    def __init__(self, channels: int, eps: float = 1.0e-6):
        super().__init__()
        self.ln = nn.LayerNorm(int(channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.ln(y)
        return y.permute(0, 3, 1, 2).contiguous()


class _DescriptorContextRefine(nn.Module):
    def __init__(self, channels: int, norm: str = "group", act: str = "silu"):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            _make_norm(norm, channels),
            _make_act(act),
            nn.Conv2d(channels, channels, 1, bias=False),
            _make_norm(norm, channels),
            _make_act(act),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            _make_norm(norm, channels),
            _make_act(act),
            nn.Conv2d(channels, channels, 1, bias=False),
            _make_norm(norm, channels),
            _make_act(act),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            _make_norm(norm, channels),
            _make_act(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        context = self.context(x)
        fused = self.mix(torch.cat([local, context], dim=1))
        return fused + x


class _DescriptorDualBranch(nn.Module):
    """
    Descriptor refinement that explicitly separates local and context branches.
    """

    def __init__(self, channels: int, norm: str = "group", act: str = "silu"):
        super().__init__()
        c = int(channels)
        self.local = nn.Sequential(
            _ResBlock(c, norm=norm, act=act),
            _ResBlock(c, norm=norm, act=act),
        )
        self.context = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False),
            _make_norm(norm, c),
            _make_act(act),
            nn.Conv2d(c, c, 1, bias=False),
            _make_norm(norm, c),
            _make_act(act),
            _ResBlock(c, norm=norm, act=act),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            _make_norm(norm, c),
            _make_act(act),
            _ResBlock(c, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.local(x)
        c = self.context(x)
        return self.fuse(torch.cat([l, c], dim=1))


def _make_tower(channels: int, depth: int, norm: str, act: str) -> nn.Sequential:
    d = max(1, int(depth))
    blocks = [_ResBlock(channels=channels, norm=norm, act=act) for _ in range(d)]
    return nn.Sequential(*blocks)


class Heads(nn.Module):
    def __init__(
        self,
        in_ch: int,
        descriptor_dim: int = 256,
        use_offset: bool = True,
        use_reliability: bool = True,
        max_offset: float = 0.5,
        variant: str = "v1",
        head_channels: Optional[int] = None,
        tower_depth: int = 2,
        norm: str = "group",
        act: str = "silu",
        detector_logit_scale: float = 1.0,
        reliability_logit_scale: float = 1.0,
        desc_relgate_detach: bool = False,
    ):
        super().__init__()
        self.variant = str(variant).lower()
        hid = int(head_channels) if head_channels is not None else max(192, int(in_ch))
        self.offset_gated = False
        self.dual_desc = False
        self.desc_relgated = False
        self.desc_heatmap_mod = False
        self.desc_saliency_gate = False
        self.desc_layernorm = False
        self.offset_residual = False
        self.offset_confidence = False
        self.desc_moe = False
        self.desc_scale_pyramid = False
        self.det_rel_crossgate = False
        self.desc_relgate_detach = bool(desc_relgate_detach)

        if self.variant in {
            "v2",
            "v3",
            "v4_dual_desc",
            "v5_offset_gated",
            "v6_dual_desc_relgate",
            "v7_heatmap_mod_desc",
            "v8_offset_residual",
            "v9_dual_desc_saliencygate",
            "v10_dual_desc_layernorm",
            "v11_offset_confidence",
            "v12_moe_desc",
            "v13_scale_pyramid_desc",
            "v14_det_rel_crossgate",
        }:
            self.shared = nn.Sequential(
                nn.Conv2d(in_ch, hid, 3, padding=1, bias=False),
                _make_norm(norm, hid),
                _make_act(act),
                _ResBlock(hid, norm=norm, act=act),
            )
            self.det_tower = _make_tower(hid, depth=tower_depth, norm=norm, act=act)
            self.desc_tower = _make_tower(hid, depth=tower_depth, norm=norm, act=act)
            tower_aux_depth = max(1, int(tower_depth) - 1)
            self.off_tower = _make_tower(hid, depth=tower_aux_depth, norm=norm, act=act)
            self.rel_tower = _make_tower(hid, depth=tower_aux_depth, norm=norm, act=act)
            if self.variant == "v3":
                self.desc_context = _DescriptorContextRefine(hid, norm=norm, act=act)
                self.desc_channel_attn = _ChannelAttention(hid, reduction=8)
                self.desc_dual = None
            elif self.variant in {
                "v4_dual_desc",
                "v5_offset_gated",
                "v6_dual_desc_relgate",
                "v7_heatmap_mod_desc",
                "v8_offset_residual",
                "v9_dual_desc_saliencygate",
                "v10_dual_desc_layernorm",
                "v11_offset_confidence",
                "v12_moe_desc",
                "v13_scale_pyramid_desc",
                "v14_det_rel_crossgate",
            }:
                self.desc_context = None
                self.desc_channel_attn = None
                self.desc_dual = _DescriptorDualBranch(hid, norm=norm, act=act)
                self.dual_desc = True
            else:
                self.desc_context = None
                self.desc_channel_attn = None
                self.desc_dual = None
            if self.variant == "v5_offset_gated":
                self.offset_gated = True
            if self.variant == "v6_dual_desc_relgate":
                self.desc_relgated = True
            if self.variant == "v7_heatmap_mod_desc":
                self.desc_heatmap_mod = True
            if self.variant == "v8_offset_residual":
                self.offset_residual = True
            if self.variant == "v9_dual_desc_saliencygate":
                self.desc_saliency_gate = True
            if self.variant == "v10_dual_desc_layernorm":
                self.desc_layernorm = True
            if self.variant == "v11_offset_confidence":
                self.offset_confidence = True
            if self.variant == "v12_moe_desc":
                self.desc_moe = True
            if self.variant == "v13_scale_pyramid_desc":
                self.desc_scale_pyramid = True
            if self.variant == "v14_det_rel_crossgate":
                self.det_rel_crossgate = True
        else:
            self.shared = nn.Sequential(
                nn.Conv2d(in_ch, hid, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, hid, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.det_tower = None
            self.desc_tower = None
            self.off_tower = None
            self.rel_tower = None
            self.desc_context = None
            self.desc_channel_attn = None
            self.desc_dual = None

        self.saliency_head = nn.Conv2d(hid, 1, 1) if self.desc_saliency_gate else None
        self.desc_post_norm = _LayerNorm2d(hid) if self.desc_layernorm else None
        self.desc_pyr_fuse = nn.Sequential(
            nn.Conv2d(2 * hid, hid, 1, bias=False),
            _make_norm(norm, hid),
            _make_act(act),
            _ResBlock(hid, norm=norm, act=act),
        ) if self.desc_scale_pyramid else None
        self.desc_gate = nn.Conv2d(hid, 1, 1) if self.desc_moe else None
        self.desc_expert_a = nn.Conv2d(hid, descriptor_dim, 1) if self.desc_moe else None
        self.desc_expert_b = nn.Conv2d(hid, descriptor_dim, 1) if self.desc_moe else None
        self.det = nn.Conv2d(hid, 1, 1)
        self.desc = nn.Conv2d(hid, descriptor_dim, 1)

        self.use_offset = bool(use_offset)
        self.use_reliability = bool(use_reliability)
        self.max_offset = float(max_offset)

        self.off = nn.Conv2d(hid, 2, 1) if self.use_offset else None
        self.off_res = nn.Conv2d(hid, 2, 1) if (self.use_offset and self.offset_residual) else None
        self.off_conf = nn.Conv2d(hid, 1, 1) if (self.use_offset and self.offset_confidence) else None
        self.rel = nn.Conv2d(hid, 1, 1) if self.use_reliability else None
        det_scale_init = max(1e-3, float(detector_logit_scale))
        rel_scale_init = max(1e-3, float(reliability_logit_scale))
        self.det_scale_log = nn.Parameter(torch.tensor(math.log(det_scale_init), dtype=torch.float32))
        self.rel_scale_log = nn.Parameter(torch.tensor(math.log(rel_scale_init), dtype=torch.float32))

    def forward(self, feat: torch.Tensor) -> FeatureOutputs:
        h = self.shared(feat)
        if self.det_tower is not None and self.desc_tower is not None:
            h_det = self.det_tower(h)
            h_desc = self.desc_tower(h)
            if self.desc_context is not None and self.desc_channel_attn is not None:
                h_desc = self.desc_context(h_desc)
                h_desc = self.desc_channel_attn(h_desc)
            if self.desc_dual is not None:
                h_desc = self.desc_dual(h_desc)
            h_off = self.off_tower(h) if self.off_tower is not None else h
            h_rel = self.rel_tower(h) if self.rel_tower is not None else h
        else:
            h_det = h
            h_desc = h
            h_off = h
            h_rel = h

        det_logits = self.det(h_det)
        rel_logits = None
        if self.rel is not None:
            rel_logits = self.rel(h_rel)

        if self.desc_relgated and rel_logits is not None:
            rel_gate = torch.sigmoid(rel_logits.detach() if self.desc_relgate_detach else rel_logits)
            h_desc = h_desc * rel_gate
        if self.desc_heatmap_mod:
            h_desc = h_desc * torch.sigmoid(det_logits)
        if self.saliency_head is not None:
            h_desc = h_desc * torch.sigmoid(self.saliency_head(h_det))
        if self.det_rel_crossgate:
            g_det = torch.sigmoid(det_logits)
            if rel_logits is not None:
                g_rel = torch.sigmoid(rel_logits.detach() if self.desc_relgate_detach else rel_logits)
            else:
                g_rel = torch.ones_like(g_det)
            h_desc = h_desc * (0.25 + 0.75 * g_det) * (0.25 + 0.75 * g_rel)
        if self.desc_pyr_fuse is not None:
            pooled = F.avg_pool2d(h_desc, kernel_size=2, stride=2)
            pooled = F.interpolate(pooled, size=h_desc.shape[-2:], mode="bilinear", align_corners=False)
            h_desc = self.desc_pyr_fuse(torch.cat([h_desc, pooled], dim=1))
        if self.desc_post_norm is not None:
            h_desc = self.desc_post_norm(h_desc)

        det_scale = self.det_scale_log.exp().clamp(min=0.25, max=4.0)
        heat = det_logits * det_scale
        if self.desc_moe and self.desc_gate is not None and self.desc_expert_a is not None and self.desc_expert_b is not None:
            da = self.desc_expert_a(h_desc)
            db = self.desc_expert_b(h_desc)
            if rel_logits is not None:
                rel_src = rel_logits.detach() if self.desc_relgate_detach else rel_logits
                gate_src = rel_src.expand(-1, h_desc.shape[1], -1, -1)
            else:
                gate_src = h_rel
            gate = torch.sigmoid(self.desc_gate(gate_src))
            desc_raw = gate * da + (1.0 - gate) * db
        else:
            desc_raw = self.desc(h_desc)
        desc = F.normalize(desc_raw, dim=1, eps=1e-6)

        off = None
        if self.off is not None:
            raw = self.off(h_off)
            if self.off_res is not None:
                raw = raw + self.off_res(h_desc)
            raw_off = torch.tanh(raw) * self.max_offset
            if self.off_conf is not None:
                conf = torch.sigmoid(self.off_conf(h_off))
                if rel_logits is not None:
                    conf = conf * (0.1 + 0.9 * torch.sigmoid(rel_logits))
                conf = conf.clamp(min=0.05, max=1.0)
                raw_off = raw_off * conf
            elif self.offset_gated and rel_logits is not None:
                raw_off = raw_off * torch.sigmoid(rel_logits)
            off = raw_off

        if rel_logits is not None:
            rel_scale = self.rel_scale_log.exp().clamp(min=0.25, max=4.0)
            rel = rel_logits * rel_scale
        else:
            rel = None
        return FeatureOutputs(heatmap=heat, desc=desc, offset=off, reliability=rel)
