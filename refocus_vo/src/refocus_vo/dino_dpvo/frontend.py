from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from refocus_vo.backbones.dinov3 import DinoTokens, DinoV3Backbone
from refocus_vo.patchgraph.teacher import DinoPatchTeacher, PseudoObjectPatchProposal

from .config import DinoDPVOConfig, load_dino_dpvo_config


@dataclass
class DinoDPVOFrameOutput:
    proposal: PseudoObjectPatchProposal
    selector_logits: torch.Tensor
    staticness_logits: torch.Tensor
    gradient_score: torch.Tensor
    qualities: torch.Tensor
    descriptor_bias: torch.Tensor | None = None
    gmap_descriptor_bias: torch.Tensor | None = None
    register_context: torch.Tensor | None = None


@dataclass
class DinoDPVOBatchOutput:
    fused: torch.Tensor
    selector_logits: torch.Tensor
    staticness_logits: torch.Tensor
    gradient_score: torch.Tensor
    observations: list[list[DinoDPVOFrameOutput]]
    register_context: torch.Tensor | None = None


class SmallLocalEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientPyramidEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        hidden_dim = max(32, int(out_dim))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, int(out_dim), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        h4 = self.conv2(x)
        h8 = self.conv3(h4)
        return h4, h8


class ConvGRUCell2d(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        gate_dim = int(input_dim) + int(hidden_dim)
        self.gates = nn.Conv2d(gate_dim, int(hidden_dim) * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(gate_dim, int(hidden_dim), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                x.shape[0],
                self.hidden_dim,
                x.shape[-2],
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        stacked = torch.cat([x, h], dim=1)
        update_gate, reset_gate = self.gates(stacked).chunk(2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset_gate * h], dim=1)))
        return ((1.0 - update_gate) * h) + (update_gate * candidate)


class DPTFusionDecoder(nn.Module):
    def __init__(self, layer_indices: Sequence[int], in_dim: int, out_dim: int):
        super().__init__()
        self.layer_indices = tuple(int(v) for v in layer_indices)
        self.proj = nn.ModuleDict(
            {
                str(idx): nn.Conv2d(int(in_dim), int(out_dim), kernel_size=1)
                for idx in self.layer_indices
            }
        )
        self.smooth = nn.ModuleDict(
            {
                str(idx): nn.Sequential(
                    nn.Conv2d(int(out_dim), int(out_dim), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for idx in self.layer_indices
            }
        )

    def forward(self, hidden_states: dict[int, torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        if not hidden_states:
            return fallback
        decoded = None
        for idx in reversed(self.layer_indices):
            feat = hidden_states.get(int(idx))
            if feat is None:
                continue
            projected = self.proj[str(idx)](feat)
            if decoded is None:
                decoded = projected
            else:
                decoded = F.interpolate(decoded, size=projected.shape[-2:], mode="bilinear", align_corners=False)
                decoded = decoded + projected
            decoded = self.smooth[str(idx)](decoded)
        return fallback if decoded is None else decoded


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dense_gradient_offset_targets(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.dim() != 5:
        raise ValueError(f"Expected images with shape (B,T,3,H,W), got {tuple(images.shape)}")
    b, t, _, h, w = images.shape
    gray = (0.2989 * images[:, :, 0]) + (0.5870 * images[:, :, 1]) + (0.1140 * images[:, :, 2])
    flat = gray.reshape(b * t, 1, h, w)
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=flat.device)
    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=flat.device)
    gx = F.conv2d(flat, sobel_x, padding=1)
    gy = F.conv2d(flat, sobel_y, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
    unfold = F.unfold(mag, kernel_size=patch_size, stride=patch_size)
    patch_area = patch_size * patch_size
    ht = h // patch_size
    wt = w // patch_size
    patches = unfold.transpose(1, 2).reshape(b * t, ht * wt, patch_area)
    yy, xx = torch.meshgrid(
        torch.arange(patch_size, device=flat.device),
        torch.arange(patch_size, device=flat.device),
        indexing="ij",
    )
    xx = xx.reshape(1, 1, patch_area).float()
    yy = yy.reshape(1, 1, patch_area).float()
    weight = patches / patches.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    cx = (weight * xx).sum(dim=-1)
    cy = (weight * yy).sum(dim=-1)
    center = (float(patch_size) - 1.0) * 0.5
    offsets = torch.stack([cx - center, cy - center], dim=-1)
    return offsets.reshape(b, t, ht, wt, 2)


class DinoProposalFrontend(nn.Module):
    def __init__(
        self,
        *,
        dino_name_or_path: str,
        dino_layers: Sequence[int] = (6, 11),
        dino_dtype: str = "bf16",
        image_size: Sequence[int] = (240, 320),
        patch_size: int = 16,
        candidate_pool: int = 48,
        patch_budget: int = 24,
        max_nodes_per_object_ratio: float = 0.20,
        k_mutual_neighbors: int = 4,
        local_patch_dim: int = 64,
        dpvo_dim: int = 384,
        use_offset_refinement: bool = True,
        use_descriptor_bias: bool = False,
        descriptor_bias_mode: str = "none",
        descriptor_bias_scale: float = 1.0,
        gmap_bias_scale: float = 1.0,
        static_score_weight: float = 0.35,
        quality_floor: float = 0.05,
        use_register_context: bool = False,
        register_context_scale: float = 0.0,
        register_context_target: str = "fused",
        enable_gradient_branch: bool = False,
        gradient_branch_dim: int = 32,
        proposal_decoder_mode: str = "base",
        temporal_memory_mode: str = "none",
        temporal_memory_dim: int = 64,
        geometry_score_mode: str = "gradient",
        semantic_grid_rows: int = 6,
        semantic_grid_cols: int = 8,
        dino_unfreeze_blocks: int = 0,
        device: str | None = None,
    ):
        super().__init__()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.patch_size = int(patch_size)
        self.layer_indices = tuple(int(v) for v in dino_layers)
        self.candidate_pool = int(candidate_pool)
        self.patch_budget = int(patch_budget)
        self.use_offset_refinement = bool(use_offset_refinement)
        legacy_bias = bool(use_descriptor_bias)
        mode = str(descriptor_bias_mode or ("imap" if legacy_bias else "none")).lower()
        if mode not in {"none", "imap", "imap_gmap"}:
            raise ValueError(f"Unsupported descriptor_bias_mode: {descriptor_bias_mode}")
        self.descriptor_bias_mode = mode
        self.use_descriptor_bias = self.descriptor_bias_mode != "none"
        self.descriptor_bias_scale = float(descriptor_bias_scale)
        self.gmap_bias_scale = float(gmap_bias_scale)
        self.static_score_weight = float(static_score_weight)
        self.quality_floor = float(quality_floor)
        self.use_register_context = bool(use_register_context)
        self.register_context_scale = float(register_context_scale)
        target = str(register_context_target or "fused").lower()
        if target not in {"fused", "anchor_refresh", "both"}:
            raise ValueError(f"Unsupported register_context_target: {register_context_target}")
        self.register_context_target = target
        self.enable_gradient_branch = bool(enable_gradient_branch)
        self.gradient_branch_dim = int(gradient_branch_dim)
        decoder_mode = str(proposal_decoder_mode or "base").lower()
        if decoder_mode not in {"base", "dpt_fpn", "dual_stream"}:
            raise ValueError(f"Unsupported proposal_decoder_mode: {proposal_decoder_mode}")
        self.proposal_decoder_mode = decoder_mode
        memory_mode = str(temporal_memory_mode or "none").lower()
        if memory_mode not in {"none", "convgru", "token_gru"}:
            raise ValueError(f"Unsupported temporal_memory_mode: {temporal_memory_mode}")
        self.temporal_memory_mode = memory_mode
        geometry_mode = str(geometry_score_mode or "gradient").lower()
        if geometry_mode not in {"none", "gradient", "corner", "gradient_corner"}:
            raise ValueError(f"Unsupported geometry_score_mode: {geometry_score_mode}")
        self.geometry_score_mode = geometry_mode
        self.temporal_memory_dim = int(temporal_memory_dim)
        self.semantic_grid_rows = max(1, int(semantic_grid_rows))
        self.semantic_grid_cols = max(1, int(semantic_grid_cols))
        self.dino_unfreeze_blocks = int(dino_unfreeze_blocks)
        self.device_name = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.backbone = DinoV3Backbone(
            name_or_path=str(dino_name_or_path),
            patch_size=self.patch_size,
            freeze=True,
            dtype=str(dino_dtype),
        )
        self.backbone.load()
        if self.dino_unfreeze_blocks > 0:
            self.backbone.set_trainable_top_blocks(self.dino_unfreeze_blocks, train_norm=True)

        self.teacher = DinoPatchTeacher(
            patch_size=self.patch_size,
            num_patches=self.candidate_pool,
            max_nodes_per_object_ratio=float(max_nodes_per_object_ratio),
            k_mutual_neighbors=int(k_mutual_neighbors),
        )

        embed_dim = int(self.backbone.embed_dim or dpvo_dim)
        self.register_context_proj = (
            nn.Linear(embed_dim, embed_dim)
            if self.use_register_context
            else None
        )
        self.local_encoder = SmallLocalEncoder(int(local_patch_dim))
        if self.enable_gradient_branch:
            self.gradient_encoder = GradientPyramidEncoder(int(self.gradient_branch_dim))
            self.local_fuse = nn.Sequential(
                nn.Conv2d(int(local_patch_dim) + int(self.gradient_branch_dim), int(local_patch_dim), kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.mid_fuse = nn.Sequential(
                nn.Conv2d(int(local_patch_dim) + int(self.gradient_branch_dim), int(local_patch_dim), kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.gradient_encoder = None
            self.local_fuse = None
            self.mid_fuse = None
        self.dpt_decoder = (
            DPTFusionDecoder(self.layer_indices, embed_dim, embed_dim)
            if self.proposal_decoder_mode in {"dpt_fpn", "dual_stream"}
            else None
        )
        self.local_to_embed = (
            nn.Sequential(
                nn.Conv2d(int(local_patch_dim), embed_dim, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            if self.proposal_decoder_mode == "dual_stream"
            else None
        )
        self.dual_stream_gate = (
            nn.Sequential(
                nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            if self.proposal_decoder_mode == "dual_stream"
            else None
        )
        self.temporal_convgru = (
            ConvGRUCell2d(embed_dim, self.temporal_memory_dim)
            if self.temporal_memory_mode == "convgru"
            else None
        )
        self.temporal_convgru_out = (
            nn.Conv2d(self.temporal_memory_dim, embed_dim, kernel_size=1)
            if self.temporal_memory_mode == "convgru"
            else None
        )
        self.temporal_token_gru = (
            nn.GRU(embed_dim, self.temporal_memory_dim, batch_first=True)
            if self.temporal_memory_mode == "token_gru"
            else None
        )
        self.temporal_token_out = (
            nn.Linear(self.temporal_memory_dim, embed_dim)
            if self.temporal_memory_mode == "token_gru"
            else None
        )
        self.selector_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )
        self.staticness_head = nn.Sequential(
            nn.Conv2d(embed_dim + int(local_patch_dim), embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )
        hidden_dim = embed_dim + int(local_patch_dim)
        offset_hidden_dim = hidden_dim + (int(self.gradient_branch_dim) if self.enable_gradient_branch else 0)
        self.offset_head = MLP(offset_hidden_dim, max(128, offset_hidden_dim // 2), 2)
        self.descriptor_head = MLP(hidden_dim, max(128, hidden_dim // 2), int(dpvo_dim))
        self.gmap_descriptor_head = (
            MLP(hidden_dim, max(128, hidden_dim // 2), 128)
            if self.descriptor_bias_mode == "imap_gmap"
            else None
        )
        self.corner_head = nn.Sequential(
            nn.Conv2d(int(local_patch_dim), max(16, int(local_patch_dim) // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, int(local_patch_dim) // 2), 1, kernel_size=1),
        )

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def _extract_backbone_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[int, torch.Tensor] | None]:
        b, t, _, h, w = images.shape
        flat = images.reshape(b * t, 3, h, w)
        backbone_ctx = torch.inference_mode() if not self.backbone.has_trainable_params() else nullcontext()
        with backbone_ctx:
            out: DinoTokens = self.backbone(
                flat,
                return_hidden_states=True,
                hidden_state_indices=self.layer_indices,
            )
        hidden_states = {
            int(idx): tensor.clone()
            for idx, tensor in dict(out.hidden_states or {}).items()
        }
        fused = self.teacher.fuse_layers(hidden_states, self.layer_indices)
        fused = fused.reshape(b, t, fused.shape[1], fused.shape[2], fused.shape[3])
        pooled_register = out.pooled_register_tokens
        register_context = None
        if pooled_register is not None:
            # Frozen-backbone outputs may be inference tensors, which cannot be
            # consumed directly by trainable heads that need autograd state.
            register_context = pooled_register.reshape(b, t, pooled_register.shape[-1]).clone()
        projected_context = None
        if self.register_context_proj is not None and register_context is not None:
            projected_context = self.register_context_proj(
                register_context.reshape(b * t, register_context.shape[-1])
            ).reshape(b, t, -1)
        if (
            self.use_register_context
            and self.register_context_target in {"fused", "both"}
            and projected_context is not None
        ):
            fused = fused + float(self.register_context_scale) * projected_context.unsqueeze(-1).unsqueeze(-1)
        return {
            "fused": fused,
            "hidden_states": {
                int(idx): tensor.reshape(b, t, tensor.shape[1], tensor.shape[2], tensor.shape[3])
                for idx, tensor in hidden_states.items()
            },
            "register_context": (projected_context if projected_context is not None else register_context),
        }

    def _encode_backbone(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        payload = self._extract_backbone_features(images)
        return payload["fused"], payload["register_context"]  # type: ignore[return-value]

    def _decode_semantic_map(
        self,
        *,
        fused: torch.Tensor,
        hidden_states: dict[int, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.dpt_decoder is None or self.proposal_decoder_mode == "base":
            return fused
        b, t = fused.shape[:2]
        fused_flat = fused.reshape(b * t, fused.shape[2], fused.shape[3], fused.shape[4])
        hidden_flat = {
            int(idx): tensor.reshape(b * t, tensor.shape[2], tensor.shape[3], tensor.shape[4])
            for idx, tensor in (hidden_states or {}).items()
        }
        decoded = self.dpt_decoder(hidden_flat, fused_flat)
        return decoded.reshape(b, t, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    def _apply_temporal_memory(self, dense_map: torch.Tensor) -> torch.Tensor:
        if self.temporal_memory_mode == "none":
            return dense_map
        b, t, c, h, w = dense_map.shape
        if self.temporal_memory_mode == "convgru":
            assert self.temporal_convgru is not None and self.temporal_convgru_out is not None
            hidden = None
            outputs: list[torch.Tensor] = []
            for ti in range(t):
                hidden = self.temporal_convgru(dense_map[:, ti], hidden)
                outputs.append(self.temporal_convgru_out(hidden))
            return torch.stack(outputs, dim=1)

        assert self.temporal_token_gru is not None and self.temporal_token_out is not None
        pooled = F.adaptive_avg_pool2d(
            dense_map.reshape(b * t, c, h, w),
            (self.semantic_grid_rows, self.semantic_grid_cols),
        ).reshape(b, t, c, self.semantic_grid_rows, self.semantic_grid_cols)
        cell_count = self.semantic_grid_rows * self.semantic_grid_cols
        tokens = pooled.permute(0, 3, 4, 1, 2).reshape(b * cell_count, t, c)
        encoded, _ = self.temporal_token_gru(tokens)
        encoded = self.temporal_token_out(encoded.reshape(-1, encoded.shape[-1])).reshape(
            b,
            self.semantic_grid_rows,
            self.semantic_grid_cols,
            t,
            c,
        )
        encoded = encoded.permute(0, 3, 4, 1, 2).reshape(b * t, c, self.semantic_grid_rows, self.semantic_grid_cols)
        encoded = F.interpolate(encoded, size=(h, w), mode="bilinear", align_corners=False)
        encoded = encoded.reshape(b, t, c, h, w)
        return dense_map + encoded

    def _proposal_feature_map(
        self,
        *,
        fused: torch.Tensor,
        hidden_states: dict[int, torch.Tensor] | None,
        local_map: torch.Tensor,
    ) -> torch.Tensor:
        semantic_map = self._decode_semantic_map(fused=fused, hidden_states=hidden_states)
        if self.proposal_decoder_mode != "dual_stream":
            return self._apply_temporal_memory(semantic_map)

        assert self.local_to_embed is not None and self.dual_stream_gate is not None
        b, t = semantic_map.shape[:2]
        local_flat = local_map.reshape(b * t, local_map.shape[2], local_map.shape[3], local_map.shape[4])
        semantic_flat = semantic_map.reshape(b * t, semantic_map.shape[2], semantic_map.shape[3], semantic_map.shape[4])
        local_semantic = self.local_to_embed(local_flat)
        gate = self.dual_stream_gate(torch.cat([semantic_flat, local_semantic], dim=1))
        mixed = (gate * semantic_flat) + ((1.0 - gate) * local_semantic)
        mixed = mixed.reshape(b, t, mixed.shape[1], mixed.shape[2], mixed.shape[3])
        return self._apply_temporal_memory(mixed)

    def _gradient_score(self, images: torch.Tensor) -> torch.Tensor:
        b, t, _, _, _ = images.shape
        gray = (0.2989 * images[:, :, 0]) + (0.5870 * images[:, :, 1]) + (0.1140 * images[:, :, 2])
        flat = gray.reshape(b * t, 1, gray.shape[-2], gray.shape[-1])
        sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=flat.device)
        sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=flat.device)
        gx = F.conv2d(flat, sobel_x, padding=1)
        gy = F.conv2d(flat, sobel_y, padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        pooled = F.avg_pool2d(mag, kernel_size=self.patch_size, stride=self.patch_size)
        pooled = pooled.reshape(b, t, pooled.shape[-2], pooled.shape[-1])
        flat_scores = pooled.reshape(b, t, -1)
        min_v = flat_scores.min(dim=-1, keepdim=True).values
        max_v = flat_scores.max(dim=-1, keepdim=True).values
        norm = (flat_scores - min_v) / (max_v - min_v + 1e-6)
        return norm.reshape(b, t, pooled.shape[-2], pooled.shape[-1])

    def _corner_score(self, local_map: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = local_map.shape
        score = self.corner_head(local_map.reshape(b * t, local_map.shape[2], h, w))
        flat = score.reshape(b * t, -1)
        min_v = flat.min(dim=-1, keepdim=True).values
        max_v = flat.max(dim=-1, keepdim=True).values
        norm = (flat - min_v) / (max_v - min_v + 1e-6)
        return norm.reshape(b, t, h, w)

    def _geometry_score(self, images: torch.Tensor, local_map: torch.Tensor) -> torch.Tensor:
        if self.geometry_score_mode == "none":
            b, t = images.shape[:2]
            return torch.zeros(
                b,
                t,
                local_map.shape[-2],
                local_map.shape[-1],
                dtype=local_map.dtype,
                device=local_map.device,
            )
        gradient = self._gradient_score(images)
        if self.geometry_score_mode == "gradient":
            return gradient
        corner = self._corner_score(local_map)
        if self.geometry_score_mode == "corner":
            return corner
        return 0.5 * (gradient + corner)

    def _gradient_channels_flat(self, flat_images: torch.Tensor) -> torch.Tensor:
        gray = (0.2989 * flat_images[:, 0:1]) + (0.5870 * flat_images[:, 1:2]) + (0.1140 * flat_images[:, 2:3])
        sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=flat_images.device)
        sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=flat_images.device)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        denom = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.cat([gx / denom, gy / denom, mag / denom], dim=1)

    def _encode_local_features(
        self,
        images: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        b, t, _, image_h, image_w = images.shape
        flat_images = images.reshape(b * t, 3, image_h, image_w)

        local_rgb_h8 = self.local_encoder(flat_images)
        local_mid = F.interpolate(local_rgb_h8, size=(image_h // 4, image_w // 4), mode="bilinear", align_corners=False)
        offset_fine = local_mid

        if self.enable_gradient_branch:
            assert self.gradient_encoder is not None
            assert self.local_fuse is not None
            assert self.mid_fuse is not None
            grad_inputs = self._gradient_channels_flat(flat_images)
            grad_h4, grad_h8 = self.gradient_encoder(grad_inputs)
            local_fine = self.local_fuse(torch.cat([local_rgb_h8, grad_h8], dim=1))
            local_mid = self.mid_fuse(
                torch.cat(
                    [
                        F.interpolate(local_fine, size=grad_h4.shape[-2:], mode="bilinear", align_corners=False),
                        grad_h4,
                    ],
                    dim=1,
                )
            )
            offset_fine = grad_h4
        else:
            local_fine = local_rgb_h8

        local_map = F.interpolate(local_fine, size=target_hw, mode="bilinear", align_corners=False)
        return {
            "local_map": local_map.reshape(b, t, local_map.shape[1], local_map.shape[2], local_map.shape[3]),
            "local_fine": local_fine.reshape(b, t, local_fine.shape[1], local_fine.shape[2], local_fine.shape[3]),
            "local_mid": local_mid.reshape(b, t, local_mid.shape[1], local_mid.shape[2], local_mid.shape[3]),
            "offset_fine": offset_fine.reshape(b, t, offset_fine.shape[1], offset_fine.shape[2], offset_fine.shape[3]),
        }

    def _sample_feature_vectors(
        self,
        feature_map: torch.Tensor,
        pixel_xy: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        if pixel_xy.numel() == 0:
            return torch.zeros((0, feature_map.shape[1]), device=feature_map.device, dtype=feature_map.dtype)
        grid_x = (pixel_xy[:, 0] / max(image_width - 1, 1)) * 2.0 - 1.0
        grid_y = (pixel_xy[:, 1] / max(image_height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
        return sampled[:, :, :, 0].permute(0, 2, 1).reshape(-1, feature_map.shape[1])

    def _normalize_patch_score(self, score: torch.Tensor) -> torch.Tensor:
        flat = score.reshape(-1)
        return (score - flat.min()) / (flat.max() - flat.min() + 1e-6)

    def _prepare_frame_output(
        self,
        *,
        proposal_bt: torch.Tensor,
        local_bt: torch.Tensor,
        local_fine_bt: torch.Tensor,
        offset_bt: torch.Tensor,
        selector_bt: torch.Tensor,
        staticness_bt: torch.Tensor,
        geometry_bt: torch.Tensor,
        register_context_bt: torch.Tensor | None,
        image_height: int,
        image_width: int,
    ) -> DinoDPVOFrameOutput:
        object_ids = self.teacher.build_pseudo_objects(proposal_bt)[0]
        combined_score = self._normalize_patch_score(
            torch.sigmoid(selector_bt[0, 0]) +
            (self.static_score_weight * torch.sigmoid(staticness_bt[0, 0])) +
            geometry_bt
        )
        proposal = self.teacher.select_patches(
            fused=proposal_bt,
            local_features=local_bt,
            patch_score=combined_score.unsqueeze(0),
            object_ids=object_ids.unsqueeze(0),
            selector_logits=None,
            num_patches=self.candidate_pool,
        )[0]

        sampled_local = self._sample_feature_vectors(
            local_fine_bt,
            proposal.coarse_pixel_xy,
            image_height=image_height,
            image_width=image_width,
        )
        proposal.local_features = sampled_local
        descriptor_in = torch.cat([proposal.descriptors, sampled_local], dim=1)
        if self.use_offset_refinement and descriptor_in.numel() > 0:
            offset_in = descriptor_in
            if self.enable_gradient_branch:
                sampled_offset = self._sample_feature_vectors(
                    offset_bt,
                    proposal.coarse_pixel_xy,
                    image_height=image_height,
                    image_width=image_width,
                )
                offset_in = torch.cat([offset_in, sampled_offset], dim=1)
            offset = torch.tanh(self.offset_head(offset_in)) * (float(self.patch_size) * 0.5)
            refined_xy = proposal.coarse_pixel_xy + offset
            refined_xy = torch.stack(
                [
                    refined_xy[:, 0].clamp(0.0, float(image_width - 1)),
                    refined_xy[:, 1].clamp(0.0, float(image_height - 1)),
                ],
                dim=1,
            )
            proposal.pixel_xy = refined_xy
            proposal.offset_xy = refined_xy - proposal.coarse_pixel_xy
        else:
            proposal.pixel_xy = proposal.coarse_pixel_xy
            proposal.offset_xy = torch.zeros_like(proposal.coarse_pixel_xy)

        qualities = proposal.scores.clamp(self.quality_floor, 1.0)
        descriptor_bias = None
        if self.descriptor_bias_mode in {"imap", "imap_gmap"}:
            descriptor_bias = self.descriptor_head(descriptor_in) * float(self.descriptor_bias_scale)
        gmap_descriptor_bias = None
        if self.gmap_descriptor_head is not None:
            gmap_descriptor_bias = self.gmap_descriptor_head(descriptor_in) * float(self.gmap_bias_scale)
        if proposal.patch_indices.numel() > self.patch_budget:
            ranking = torch.argsort(qualities, descending=True)[: self.patch_budget]
            proposal = PseudoObjectPatchProposal(
                patch_indices=proposal.patch_indices[ranking],
                patch_xy=proposal.patch_xy[ranking],
                coarse_pixel_xy=proposal.coarse_pixel_xy[ranking],
                pixel_xy=proposal.pixel_xy[ranking],
                offset_xy=proposal.offset_xy[ranking],
                scores=proposal.scores[ranking],
                object_ids=proposal.object_ids[ranking],
                descriptors=proposal.descriptors[ranking],
                local_features=proposal.local_features[ranking],
            )
            qualities = qualities[ranking]
            if descriptor_bias is not None:
                descriptor_bias = descriptor_bias[ranking]
            if gmap_descriptor_bias is not None:
                gmap_descriptor_bias = gmap_descriptor_bias[ranking]

        return DinoDPVOFrameOutput(
            proposal=proposal,
            selector_logits=selector_bt[0, 0],
            staticness_logits=staticness_bt[0, 0],
            gradient_score=geometry_bt,
            qualities=qualities,
            descriptor_bias=descriptor_bias,
            gmap_descriptor_bias=gmap_descriptor_bias,
            register_context=None if register_context_bt is None else register_context_bt[0],
        )

    def forward(self, images: torch.Tensor) -> DinoDPVOBatchOutput:
        if images.dim() != 5:
            raise ValueError(f"Expected images with shape (B,T,3,H,W), got {tuple(images.shape)}")
        b, t, _, image_h, image_w = images.shape
        backbone = self._extract_backbone_features(images)
        fused = backbone["fused"]  # type: ignore[assignment]
        hidden_states = backbone["hidden_states"]  # type: ignore[assignment]
        register_context = backbone["register_context"]  # type: ignore[assignment]
        local_feats = self._encode_local_features(images, target_hw=(fused.shape[-2], fused.shape[-1]))
        local_map = local_feats["local_map"]
        local_fine = local_feats["local_fine"]
        offset_fine = local_feats["offset_fine"]
        proposal_map = self._proposal_feature_map(
            fused=fused,
            hidden_states=hidden_states if isinstance(hidden_states, dict) else None,
            local_map=local_map,
        )

        proposal_flat = proposal_map.reshape(b * t, proposal_map.shape[2], proposal_map.shape[3], proposal_map.shape[4])
        selector_logits = self.selector_head(proposal_flat).reshape(
            b,
            t,
            1,
            proposal_map.shape[-2],
            proposal_map.shape[-1],
        )
        static_in = torch.cat(
            [
                proposal_flat,
                local_map.reshape(b * t, local_map.shape[2], local_map.shape[3], local_map.shape[4]),
            ],
            dim=1,
        )
        staticness_logits = self.staticness_head(static_in).reshape(
            b,
            t,
            1,
            proposal_map.shape[-2],
            proposal_map.shape[-1],
        )
        geometry_score = self._geometry_score(images, local_map)

        observations: list[list[DinoDPVOFrameOutput]] = []
        for bi in range(b):
            frame_outputs: list[DinoDPVOFrameOutput] = []
            for ti in range(t):
                frame_outputs.append(
                    self._prepare_frame_output(
                        proposal_bt=proposal_map[bi, ti].unsqueeze(0),
                        local_bt=local_map[bi, ti].unsqueeze(0),
                        local_fine_bt=local_fine[bi, ti].unsqueeze(0),
                        offset_bt=offset_fine[bi, ti].unsqueeze(0),
                        selector_bt=selector_logits[bi, ti].unsqueeze(0),
                        staticness_bt=staticness_logits[bi, ti].unsqueeze(0),
                        geometry_bt=geometry_score[bi, ti],
                        register_context_bt=(
                            None if register_context is None else register_context[bi, ti].unsqueeze(0)
                        ),
                        image_height=image_h,
                        image_width=image_w,
                    )
                )
            observations.append(frame_outputs)

        return DinoDPVOBatchOutput(
            fused=proposal_map,
            selector_logits=selector_logits[:, :, 0],
            staticness_logits=staticness_logits[:, :, 0],
            gradient_score=geometry_score,
            observations=observations,
            register_context=register_context,
        )

    @torch.no_grad()
    def infer_single_frame(self, image: torch.Tensor) -> DinoDPVOFrameOutput:
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:
            image = image.unsqueeze(1)
        output = self.forward(image.to(self.device))
        return output.observations[0][0]


def build_dino_dpvo_frontend(cfg: DinoDPVOConfig) -> DinoProposalFrontend:
    model_cfg = cfg.model
    return DinoProposalFrontend(
        dino_name_or_path=str(model_cfg.get("dino_name_or_path")),
        dino_layers=tuple(int(v) for v in model_cfg.get("dino_layers", [6, 11])),
        dino_dtype=str(model_cfg.get("dino_dtype", "bf16")),
        image_size=tuple(int(v) for v in model_cfg.get("image_size", [240, 320])),
        patch_size=int(model_cfg.get("patch_size", 16)),
        candidate_pool=int(model_cfg.get("candidate_pool", 48)),
        patch_budget=int(model_cfg.get("patch_budget", 24)),
        max_nodes_per_object_ratio=float(model_cfg.get("max_nodes_per_object_ratio", 0.20)),
        k_mutual_neighbors=int(model_cfg.get("k_mutual_neighbors", 4)),
        local_patch_dim=int(model_cfg.get("local_patch_dim", 64)),
        dpvo_dim=int(model_cfg.get("dpvo_dim", 384)),
        use_offset_refinement=bool(model_cfg.get("use_offset_refinement", True)),
        use_descriptor_bias=bool(model_cfg.get("use_descriptor_bias", False)),
        descriptor_bias_mode=str(
            model_cfg.get(
                "descriptor_bias_mode",
                "imap" if bool(model_cfg.get("use_descriptor_bias", False)) else "none",
            )
        ),
        descriptor_bias_scale=float(model_cfg.get("descriptor_bias_scale", 1.0)),
        gmap_bias_scale=float(model_cfg.get("gmap_bias_scale", 1.0)),
        static_score_weight=float(model_cfg.get("static_score_weight", 0.35)),
        quality_floor=float(model_cfg.get("quality_floor", 0.05)),
        use_register_context=bool(model_cfg.get("use_register_context", False)),
        register_context_scale=float(model_cfg.get("register_context_scale", 0.0)),
        register_context_target=str(model_cfg.get("register_context_target", "fused")),
        enable_gradient_branch=bool(model_cfg.get("enable_gradient_branch", False)),
        gradient_branch_dim=int(model_cfg.get("gradient_branch_dim", 32)),
        proposal_decoder_mode=str(model_cfg.get("proposal_decoder_mode", "base")),
        temporal_memory_mode=str(model_cfg.get("temporal_memory_mode", "none")),
        temporal_memory_dim=int(model_cfg.get("temporal_memory_dim", 64)),
        geometry_score_mode=str(model_cfg.get("geometry_score_mode", "gradient")),
        semantic_grid_rows=int(model_cfg.get("semantic_grid_rows", 6)),
        semantic_grid_cols=int(model_cfg.get("semantic_grid_cols", 8)),
        dino_unfreeze_blocks=int(model_cfg.get("dino_unfreeze_blocks", 0)),
        device=str(cfg.training.get("device", "cuda")),
    ).to(torch.device(str(cfg.training.get("device", "cuda"))))


def load_matching_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str | None = None,
) -> dict[str, int]:
    module_state = module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state_dict.items():
        target_key = key
        if prefix is not None:
            if not key.startswith(prefix):
                continue
            target_key = key[len(prefix) :]
        if target_key in module_state and module_state[target_key].shape == value.shape:
            filtered[target_key] = value
        else:
            skipped += 1

    load_info = module.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing": len(load_info.missing_keys),
        "unexpected": len(load_info.unexpected_keys),
        "skipped": skipped,
    }


def load_dino_dpvo_frontend_checkpoint(
    checkpoint: str | Path,
    *,
    config: str | Path | None = None,
    device: str = "cuda",
) -> tuple[DinoDPVOConfig, DinoProposalFrontend]:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = load_dino_dpvo_config(config or payload["config_path"])
    cfg.raw.setdefault("training", {})["device"] = str(device)
    model = build_dino_dpvo_frontend(cfg)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        info = load_matching_state_dict(model, state_dict)
        if int(info["loaded"]) == 0:
            info = load_matching_state_dict(model, state_dict, prefix="patchify.semantic.")
        if int(info["loaded"]) == 0:
            raise
    model.eval()
    return cfg, model
