from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PseudoObjectPatchProposal:
    patch_indices: torch.Tensor
    patch_xy: torch.Tensor
    coarse_pixel_xy: torch.Tensor
    pixel_xy: torch.Tensor
    offset_xy: torch.Tensor
    scores: torch.Tensor
    object_ids: torch.Tensor
    descriptors: torch.Tensor
    local_features: torch.Tensor


class DinoPatchTeacher:
    def __init__(
        self,
        *,
        patch_size: int = 16,
        num_patches: int = 64,
        max_nodes_per_object_ratio: float = 0.20,
        k_mutual_neighbors: int = 4,
        teacher_weights: dict[str, float] | None = None,
    ):
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.max_nodes_per_object_ratio = float(max_nodes_per_object_ratio)
        self.k_mutual_neighbors = int(k_mutual_neighbors)
        weights = {
            "boundary": 0.45,
            "consistency": 0.30,
            "depth_safe": 0.10,
            "reprojection": 0.15,
        }
        if teacher_weights:
            for key, value in teacher_weights.items():
                if key in weights:
                    weights[key] = float(value)
        total = sum(max(0.0, v) for v in weights.values()) or 1.0
        self.weights = {k: max(0.0, v) / total for k, v in weights.items()}

    def fuse_layers(self, layers: dict[int, torch.Tensor], layer_indices: Sequence[int]) -> torch.Tensor:
        feats = []
        for idx in layer_indices:
            feat = layers[int(idx)]
            feats.append(F.normalize(feat.float(), dim=1, eps=1e-6))
        fused = torch.stack(feats, dim=0).mean(dim=0)
        return F.normalize(fused, dim=1, eps=1e-6)

    def pool_depth(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (depth > 1e-6).float()
        depth_sum = F.avg_pool2d(
            (depth * valid).unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        valid_sum = F.avg_pool2d(valid.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        patch_depth = depth_sum[:, 0] / valid_sum[:, 0].clamp_min(1e-6)
        patch_valid = valid_sum[:, 0] > 0.25
        return patch_depth, patch_valid

    def build_pseudo_objects(self, fused: torch.Tensor) -> torch.Tensor:
        b, c, ht, wt = fused.shape
        out = torch.zeros((b, ht, wt), dtype=torch.long, device=fused.device)
        feats = fused.permute(0, 2, 3, 1).reshape(b, ht * wt, c)
        feats = F.normalize(feats, dim=-1, eps=1e-6)
        for bi in range(b):
            sim = feats[bi] @ feats[bi].t()
            k = min(max(1, self.k_mutual_neighbors), sim.shape[0] - 1)
            topk_idx = torch.topk(sim, k=k + 1, dim=1, largest=True, sorted=False).indices[:, 1:]
            adj = torch.zeros((sim.shape[0], sim.shape[0]), dtype=torch.bool, device=sim.device)
            rows = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1).expand(-1, k)
            adj[rows, topk_idx] = True
            mutual = adj & adj.t()
            strong = mutual & (sim > 0.70)
            labels = self._connected_components(strong)
            out[bi] = labels.reshape(ht, wt)
        return out

    def score_adjacent_pair(
        self,
        fused_src: torch.Tensor,
        fused_tgt: torch.Tensor,
        rgb_src: torch.Tensor,
        rgb_tgt: torch.Tensor,
        depth_src: torch.Tensor,
        depth_tgt: torch.Tensor,
        rel_pose_tgt_src: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        b, c, ht, wt = fused_src.shape
        patch_depth, patch_valid = self.pool_depth(depth_src)
        centers_u, centers_v = self.patch_centers(ht, wt, fused_src.device)
        centers_u = centers_u.unsqueeze(0).expand(b, -1, -1).reshape(b, -1)
        centers_v = centers_v.unsqueeze(0).expand(b, -1, -1).reshape(b, -1)
        z = patch_depth.reshape(b, -1)
        valid = patch_valid.reshape(b, -1) & (z > 1e-6)

        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]
        x = ((centers_u - cx[:, None]) / fx[:, None]) * z
        y = ((centers_v - cy[:, None]) / fy[:, None]) * z
        xyz = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
        xyz_t = torch.einsum("bij,bnj->bni", rel_pose_tgt_src, xyz)[..., :3]
        z_t = xyz_t[..., 2]
        valid = valid & (z_t > 1e-6)
        u_t = (fx[:, None] * xyz_t[..., 0] / z_t.clamp_min(1e-6)) + cx[:, None]
        v_t = (fy[:, None] * xyz_t[..., 1] / z_t.clamp_min(1e-6)) + cy[:, None]
        valid = valid & (u_t >= 0.0) & (u_t <= float(rgb_src.shape[-1] - 1)) & (v_t >= 0.0) & (
            v_t <= float(rgb_src.shape[-2] - 1)
        )

        grid_x = ((u_t + 0.5) / float(self.patch_size)) - 0.5
        grid_y = ((v_t + 0.5) / float(self.patch_size)) - 0.5
        tgt_sample = self.sample_feature_map(fused_tgt, grid_x, grid_y)
        src_flat = fused_src.permute(0, 2, 3, 1).reshape(b, ht * wt, c)
        src_flat = F.normalize(src_flat, dim=-1, eps=1e-6)
        tgt_sample = F.normalize(tgt_sample, dim=-1, eps=1e-6)
        consistency = 0.5 * ((src_flat * tgt_sample).sum(dim=-1) + 1.0)
        consistency = torch.where(valid, consistency, torch.zeros_like(consistency)).reshape(b, ht, wt)

        boundary = self.normalize_map(self.feature_boundary_strength(fused_src))
        patch_depth_src, _ = self.pool_depth(depth_src)
        depth_safe = 1.0 - self.normalize_map(self.scalar_gradient(patch_depth_src))

        rgb_src_patch = F.avg_pool2d(rgb_src, kernel_size=self.patch_size, stride=self.patch_size)
        tgt_rgb = self.sample_rgb(rgb_tgt, u_t, v_t)
        src_rgb = rgb_src_patch.permute(0, 2, 3, 1).reshape(b, ht * wt, 3)
        rgb_err = (src_rgb - tgt_rgb).abs().mean(dim=-1).reshape(b, ht, wt)

        tgt_depth = self.sample_scalar(depth_tgt, u_t, v_t).reshape(b, ht, wt)
        depth_err = (tgt_depth - z_t.reshape(b, ht, wt)).abs()
        reprojection = 1.0 - torch.clamp(0.5 * (rgb_err / 0.20 + depth_err / 0.20), 0.0, 1.0)

        score = (
            self.weights["boundary"] * boundary
            + self.weights["consistency"] * consistency
            + self.weights["depth_safe"] * depth_safe
            + self.weights["reprojection"] * reprojection
        )
        return torch.where(patch_valid, score, torch.zeros_like(score))

    def select_patches(
        self,
        *,
        fused: torch.Tensor,
        local_features: torch.Tensor,
        patch_score: torch.Tensor,
        object_ids: torch.Tensor,
        selector_logits: torch.Tensor | None = None,
        num_patches: int | None = None,
    ) -> list[PseudoObjectPatchProposal]:
        num_patches = int(self.num_patches if num_patches is None else num_patches)
        b, c, ht, wt = fused.shape
        descriptors = fused.permute(0, 2, 3, 1).reshape(b, ht * wt, c)
        locals_flat = local_features.permute(0, 2, 3, 1).reshape(b, ht * wt, local_features.shape[1])
        score_flat = patch_score.reshape(b, ht * wt)
        if selector_logits is not None:
            pred = torch.sigmoid(selector_logits.reshape(b, ht * wt))
            score_flat = 0.75 * score_flat + 0.25 * pred
        object_flat = object_ids.reshape(b, ht * wt)

        centers_u, centers_v = self.patch_centers(ht, wt, fused.device)
        centers_u = centers_u.reshape(-1)
        centers_v = centers_v.reshape(-1)
        patch_xy_all = torch.stack(
            [
                torch.arange(wt, device=fused.device).repeat(ht),
                torch.arange(ht, device=fused.device).repeat_interleave(wt),
            ],
            dim=1,
        ).float()
        pixel_xy_all = torch.stack([centers_u, centers_v], dim=1).float()

        proposals: list[PseudoObjectPatchProposal] = []
        max_per_object = max(1, int(round(num_patches * self.max_nodes_per_object_ratio)))
        for bi in range(b):
            ranking = torch.argsort(score_flat[bi], descending=True)
            chosen = []
            per_object: dict[int, int] = {}
            for idx in ranking.tolist():
                object_id = int(object_flat[bi, idx].item())
                if per_object.get(object_id, 0) >= max_per_object:
                    continue
                chosen.append(idx)
                per_object[object_id] = per_object.get(object_id, 0) + 1
                if len(chosen) >= num_patches:
                    break
            if len(chosen) < num_patches:
                for idx in ranking.tolist():
                    if idx in chosen:
                        continue
                    chosen.append(idx)
                    if len(chosen) >= num_patches:
                        break

            chosen_t = torch.as_tensor(chosen, device=fused.device, dtype=torch.long)
            proposals.append(
                PseudoObjectPatchProposal(
                    patch_indices=chosen_t,
                    patch_xy=patch_xy_all[chosen_t],
                    coarse_pixel_xy=pixel_xy_all[chosen_t],
                    pixel_xy=pixel_xy_all[chosen_t],
                    offset_xy=torch.zeros_like(pixel_xy_all[chosen_t]),
                    scores=score_flat[bi, chosen_t],
                    object_ids=object_flat[bi, chosen_t],
                    descriptors=descriptors[bi, chosen_t],
                    local_features=locals_flat[bi, chosen_t],
                )
            )
        return proposals

    def patch_centers(self, ht: int, wt: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        step = float(self.patch_size)
        u = (torch.arange(wt, device=device).float() + 0.5) * step
        v = (torch.arange(ht, device=device).float() + 0.5) * step
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return uu, vv

    @staticmethod
    def sample_feature_map(feature_map: torch.Tensor, x_patch: torch.Tensor, y_patch: torch.Tensor) -> torch.Tensor:
        b, c, ht, wt = feature_map.shape
        grid_x = (x_patch / max(wt - 1, 1)) * 2.0 - 1.0
        grid_y = (y_patch / max(ht - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b, ht, wt, 2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
        return sampled.permute(0, 2, 3, 1).reshape(b, ht * wt, c)

    @staticmethod
    def sample_rgb(rgb: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return DinoPatchTeacher.sample_scalar_map(rgb, u, v).reshape(rgb.shape[0], -1, 3)

    @staticmethod
    def sample_scalar(depth: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        depth = depth.unsqueeze(1)
        return DinoPatchTeacher.sample_scalar_map(depth, u, v).reshape(depth.shape[0], -1)

    @staticmethod
    def sample_scalar_map(value: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, c, h, w = value.shape
        grid_x = (u / max(w - 1, 1)) * 2.0 - 1.0
        grid_y = (v / max(h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b, -1, 1, 2)
        sampled = F.grid_sample(value, grid, mode="bilinear", align_corners=True)
        return sampled[:, :, :, 0].permute(0, 2, 1)

    @staticmethod
    def normalize_map(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(value.shape[0], -1)
        min_v = flat.min(dim=1).values[:, None, None]
        max_v = flat.max(dim=1).values[:, None, None]
        return (value - min_v) / (max_v - min_v + 1e-6)

    @staticmethod
    def feature_boundary_strength(fused: torch.Tensor) -> torch.Tensor:
        dx = fused[:, :, :, 1:] - fused[:, :, :, :-1]
        dy = fused[:, :, 1:, :] - fused[:, :, :-1, :]
        dx = F.pad(dx.abs().mean(dim=1), (0, 1, 0, 0))
        dy = F.pad(dy.abs().mean(dim=1), (0, 0, 0, 1))
        return torch.sqrt(dx * dx + dy * dy + 1e-9)

    @staticmethod
    def scalar_gradient(value: torch.Tensor) -> torch.Tensor:
        dx = value[:, :, 1:] - value[:, :, :-1]
        dy = value[:, 1:, :] - value[:, :-1, :]
        dx = F.pad(dx.abs(), (0, 1, 0, 0))
        dy = F.pad(dy.abs(), (0, 0, 0, 1))
        return torch.sqrt(dx * dx + dy * dy + 1e-9)

    @staticmethod
    def _connected_components(adjacency: torch.Tensor) -> torch.Tensor:
        n = int(adjacency.shape[0])
        labels = torch.full((n,), -1, dtype=torch.long, device=adjacency.device)
        next_label = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            stack = [i]
            labels[i] = next_label
            while stack:
                cur = stack.pop()
                neighbors = torch.where(adjacency[cur])[0].tolist()
                for nb in neighbors:
                    if labels[nb] >= 0:
                        continue
                    labels[nb] = next_label
                    stack.append(nb)
            next_label += 1
        return labels
