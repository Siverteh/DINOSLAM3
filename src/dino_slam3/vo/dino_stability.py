from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dino_slam3.models.backbones.dinov3 import DinoV3Backbone


@dataclass
class DinoObservation:
    rgb: torch.Tensor  # (3, H, W) in [0, 1]
    depth: torch.Tensor  # (H, W) in meters
    layers: Dict[int, torch.Tensor]  # block idx -> (C, Ht, Wt)
    fused: torch.Tensor  # (C, Ht, Wt)
    patch_rgb: torch.Tensor  # (3, Ht, Wt)
    patch_depth: torch.Tensor  # (Ht, Wt)
    patch_valid: torch.Tensor  # (Ht, Wt)


@dataclass
class DinoStabilityMap:
    patch_score: torch.Tensor  # (Ht, Wt)
    pixel_score: torch.Tensor  # (H, W)
    pixel_mask: torch.Tensor  # (H, W) bool
    consistency: torch.Tensor  # (Ht, Wt)
    boundary: torch.Tensor  # (Ht, Wt)
    depth_edge_risk: torch.Tensor  # (Ht, Wt)
    reprojection_risk: torch.Tensor  # (Ht, Wt)


class DinoStabilityScorer:
    """
    Training-free DINOv3 stability scorer for dense RGB-D odometry.

    The score combines:
      - cross-view DINO consistency after reprojection
      - local DINO boundary contrast
      - depth-edge risk
      - RGB-D reprojection disagreement
    """

    def __init__(
        self,
        *,
        backbone_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        layer_indices: Sequence[int] = (6, 11),
        patch_size: int = 16,
        dtype: str = "bf16",
        device: str | None = None,
        keep_ratio: float = 0.35,
        min_keep_pixels: int = 2048,
        photometric_scale: float = 0.15,
        depth_reproj_scale_m: float = 0.08,
        weights: Dict[str, float] | None = None,
    ):
        self.patch_size = int(patch_size)
        self.layer_indices = tuple(int(v) for v in layer_indices)
        if len(self.layer_indices) == 0:
            raise ValueError("layer_indices must not be empty")

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.keep_ratio = float(keep_ratio)
        self.min_keep_pixels = max(1, int(min_keep_pixels))
        self.photometric_scale = max(1e-4, float(photometric_scale))
        self.depth_reproj_scale_m = max(1e-4, float(depth_reproj_scale_m))

        score_weights = {
            "consistency": 0.45,
            "boundary": 0.20,
            "depth_safe": 0.15,
            "reprojection": 0.20,
        }
        if isinstance(weights, dict):
            for key, value in weights.items():
                if key in score_weights:
                    score_weights[key] = float(value)
        weight_sum = sum(max(0.0, float(v)) for v in score_weights.values()) or 1.0
        self.weights = {k: max(0.0, float(v)) / weight_sum for k, v in score_weights.items()}

        self.backbone = DinoV3Backbone(
            name_or_path=str(backbone_name_or_path),
            patch_size=self.patch_size,
            freeze=True,
            dtype=str(dtype),
        )
        self.backbone.load()
        self.backbone.to(self.device)
        self.backbone.eval()

    def observe(self, rgb: np.ndarray, depth_m: np.ndarray) -> DinoObservation:
        rgb_t = self._to_rgb_tensor(rgb)
        depth_t = self._to_depth_tensor(depth_m)
        with torch.inference_mode():
            out = self.backbone(
                rgb_t.unsqueeze(0),
                return_hidden_states=True,
                hidden_state_indices=self.layer_indices,
            )

        layers = dict(out.hidden_states or {})
        missing = [idx for idx in self.layer_indices if idx not in layers]
        if missing:
            raise RuntimeError(f"Missing DINO hidden states for encoder blocks: {missing}")

        layer_feats = {
            idx: F.normalize(layers[idx][0].float(), dim=0, eps=1e-6).contiguous()
            for idx in self.layer_indices
        }
        fused = self._fuse_layers(layer_feats)
        patch_rgb = F.avg_pool2d(
            rgb_t.unsqueeze(0), kernel_size=self.patch_size, stride=self.patch_size
        )[0].float()
        patch_depth, patch_valid = self._pool_depth(depth_t)

        return DinoObservation(
            rgb=rgb_t,
            depth=depth_t,
            layers=layer_feats,
            fused=fused,
            patch_rgb=patch_rgb,
            patch_depth=patch_depth,
            patch_valid=patch_valid,
        )

    def score_pair(
        self,
        source: DinoObservation,
        target: DinoObservation,
        transform_target_source: np.ndarray,
        intrinsics: np.ndarray,
        *,
        keep_ratio: float | None = None,
    ) -> DinoStabilityMap:
        transform = torch.as_tensor(
            np.asarray(transform_target_source, dtype=np.float32),
            device=self.device,
            dtype=torch.float32,
        )
        K = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32), device=self.device)

        H, W = int(source.depth.shape[0]), int(source.depth.shape[1])
        Ht, Wt = int(source.fused.shape[-2]), int(source.fused.shape[-1])

        centers_u, centers_v = self._patch_centers(Ht, Wt, device=self.device)
        z = source.patch_depth.reshape(-1)
        valid = source.patch_valid.reshape(-1) & (z > 1e-6)

        x = ((centers_u.reshape(-1) - K[0, 2]) / K[0, 0]) * z
        y = ((centers_v.reshape(-1) - K[1, 2]) / K[1, 1]) * z
        xyz = torch.stack([x, y, z], dim=1)

        xyz_h = torch.cat([xyz, torch.ones_like(z).unsqueeze(1)], dim=1)
        xyz_target = (transform @ xyz_h.t()).t()[:, :3]
        z_target = xyz_target[:, 2]
        valid = valid & (z_target > 1e-6)

        z_target_safe = z_target.clamp_min(1e-6)
        u_target = (K[0, 0] * xyz_target[:, 0] / z_target_safe) + K[0, 2]
        v_target = (K[1, 1] * xyz_target[:, 1] / z_target_safe) + K[1, 2]
        valid = valid & (u_target >= 0.0) & (u_target <= (W - 1)) & (v_target >= 0.0) & (v_target <= (H - 1))

        safe_u_target = torch.where(valid, u_target, torch.zeros_like(u_target))
        safe_v_target = torch.where(valid, v_target, torch.zeros_like(v_target))

        feat_x = ((safe_u_target + 0.5) / float(self.patch_size)) - 0.5
        feat_y = ((safe_v_target + 0.5) / float(self.patch_size)) - 0.5

        sampled_target_feat = self._sample_feature_map(target.fused, feat_x, feat_y)
        source_feat = source.fused.permute(1, 2, 0).reshape(-1, source.fused.shape[0]).float()
        source_feat = F.normalize(source_feat, dim=1, eps=1e-6)
        sampled_target_feat = F.normalize(sampled_target_feat, dim=1, eps=1e-6)
        consistency = ((source_feat * sampled_target_feat).sum(dim=1) + 1.0) * 0.5
        consistency = torch.where(valid, consistency, torch.zeros_like(consistency))
        consistency = consistency.reshape(Ht, Wt)

        boundary = self._normalize_map(self._feature_boundary_strength(source.fused))
        depth_edge_risk = self._normalize_map(self._scalar_gradient(source.patch_depth))

        sampled_target_rgb = self._sample_image(target.rgb, safe_u_target, safe_v_target)
        source_patch_rgb = source.patch_rgb.permute(1, 2, 0).reshape(-1, 3)
        rgb_err = (sampled_target_rgb - source_patch_rgb).abs().mean(dim=1)
        rgb_risk = torch.clamp(rgb_err / self.photometric_scale, 0.0, 1.0)

        sampled_target_depth = self._sample_scalar_image(target.depth, safe_u_target, safe_v_target)
        valid = valid & (sampled_target_depth > 1e-6)
        depth_rel_err = (sampled_target_depth - z_target).abs()
        depth_risk = torch.clamp(depth_rel_err / self.depth_reproj_scale_m, 0.0, 1.0)
        reprojection_risk = 0.5 * (rgb_risk + depth_risk)
        reprojection_risk = torch.where(valid, reprojection_risk, torch.ones_like(reprojection_risk))
        reprojection_risk = reprojection_risk.reshape(Ht, Wt)

        patch_score = (
            self.weights["consistency"] * consistency
            + self.weights["boundary"] * boundary
            + self.weights["depth_safe"] * (1.0 - depth_edge_risk)
            + self.weights["reprojection"] * (1.0 - reprojection_risk)
        )
        patch_score = torch.where(source.patch_valid, patch_score, torch.zeros_like(patch_score))
        patch_score = torch.clamp(patch_score, 0.0, 1.0)

        pixel_score = F.interpolate(
            patch_score.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        pixel_mask = self.build_depth_mask(
            pixel_score,
            source.depth,
            keep_ratio=self.keep_ratio if keep_ratio is None else float(keep_ratio),
        )

        return DinoStabilityMap(
            patch_score=patch_score,
            pixel_score=pixel_score,
            pixel_mask=pixel_mask,
            consistency=consistency,
            boundary=boundary,
            depth_edge_risk=depth_edge_risk,
            reprojection_risk=reprojection_risk,
        )

    def build_depth_mask(
        self,
        pixel_score: torch.Tensor,
        depth_m: torch.Tensor,
        *,
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        valid = depth_m > 1e-6
        valid_scores = pixel_score[valid]
        if valid_scores.numel() == 0:
            return valid

        ratio = float(self.keep_ratio if keep_ratio is None else keep_ratio)
        ratio = min(max(ratio, 1e-3), 1.0)
        if ratio >= 0.999:
            return valid

        keep_pixels = max(self.min_keep_pixels, int(round(ratio * float(valid_scores.numel()))))
        keep_pixels = min(keep_pixels, int(valid_scores.numel()))
        if keep_pixels <= 0:
            return valid

        topk = torch.topk(valid_scores, k=keep_pixels, largest=True, sorted=False).values
        threshold = topk.min()
        return valid & (pixel_score >= threshold)

    @staticmethod
    def apply_mask_to_depth(depth_m: np.ndarray, mask: torch.Tensor) -> np.ndarray:
        out = np.asarray(depth_m, dtype=np.float32).copy()
        mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
        out[~mask_np] = 0.0
        return out

    def _to_rgb_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape={arr.shape}")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if float(arr.max()) > 1.5:
            arr = arr / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor.to(self.device, dtype=torch.float32)

    def _to_depth_tensor(self, depth_m: np.ndarray) -> torch.Tensor:
        arr = np.asarray(depth_m, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected depth image (H, W), got shape={arr.shape}")
        return torch.from_numpy(arr).to(self.device, dtype=torch.float32)

    def _fuse_layers(self, layers: Dict[int, torch.Tensor]) -> torch.Tensor:
        fused = None
        for idx in self.layer_indices:
            feat = layers[idx]
            fused = feat if fused is None else (fused + feat)
        assert fused is not None
        return F.normalize(fused / float(len(self.layer_indices)), dim=0, eps=1e-6)

    def _pool_depth(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth_4d = depth.unsqueeze(0).unsqueeze(0)
        valid = (depth_4d > 1e-6).float()
        pooled_sum = F.avg_pool2d(
            depth_4d * valid,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ) * float(self.patch_size * self.patch_size)
        pooled_valid = F.avg_pool2d(
            valid,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ) * float(self.patch_size * self.patch_size)
        pooled_depth = pooled_sum / pooled_valid.clamp_min(1.0)
        patch_valid = pooled_valid[0, 0] >= max(1.0, 0.10 * float(self.patch_size * self.patch_size))
        return pooled_depth[0, 0], patch_valid

    def _feature_boundary_strength(self, feat: torch.Tensor) -> torch.Tensor:
        dx = (feat[:, :, 1:] - feat[:, :, :-1]).abs().mean(dim=0)
        dy = (feat[:, 1:, :] - feat[:, :-1, :]).abs().mean(dim=0)

        dx = F.pad(dx.unsqueeze(0).unsqueeze(0), (0, 1, 0, 0), mode="replicate")[0, 0]
        dy = F.pad(dy.unsqueeze(0).unsqueeze(0), (0, 0, 0, 1), mode="replicate")[0, 0]
        return dx + dy

    def _scalar_gradient(self, scalar_map: torch.Tensor) -> torch.Tensor:
        dx = (scalar_map[:, 1:] - scalar_map[:, :-1]).abs()
        dy = (scalar_map[1:, :] - scalar_map[:-1, :]).abs()
        dx = F.pad(dx.unsqueeze(0).unsqueeze(0), (0, 1, 0, 0), mode="replicate")[0, 0]
        dy = F.pad(dy.unsqueeze(0).unsqueeze(0), (0, 0, 0, 1), mode="replicate")[0, 0]
        return dx + dy

    def _sample_feature_map(self, feat: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Ht, Wt = int(feat.shape[-2]), int(feat.shape[-1])
        grid = torch.stack(
            [
                ((x + 0.5) / float(Wt)) * 2.0 - 1.0,
                ((y + 0.5) / float(Ht)) * 2.0 - 1.0,
            ],
            dim=-1,
        )
        sampled = F.grid_sample(
            feat.unsqueeze(0),
            grid.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled[0, :, :, 0].transpose(0, 1).contiguous()

    def _sample_image(self, img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        H, W = int(img.shape[-2]), int(img.shape[-1])
        grid = torch.stack(
            [
                ((u + 0.5) / float(W)) * 2.0 - 1.0,
                ((v + 0.5) / float(H)) * 2.0 - 1.0,
            ],
            dim=-1,
        )
        sampled = F.grid_sample(
            img.unsqueeze(0),
            grid.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled[0, :, :, 0].transpose(0, 1).contiguous()

    def _sample_scalar_image(self, img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sampled = self._sample_image(img.unsqueeze(0), u, v)
        return sampled[:, 0]

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros_like(x)
        scale = torch.quantile(flat, 0.95).clamp_min(1e-6)
        return torch.clamp(x / scale, 0.0, 1.0)

    def _patch_centers(self, ht: int, wt: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(ht, device=device, dtype=torch.float32)
        xs = torch.arange(wt, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        offset = 0.5 * float(self.patch_size) - 0.5
        u = xx * float(self.patch_size) + offset
        v = yy * float(self.patch_size) + offset
        return u, v
