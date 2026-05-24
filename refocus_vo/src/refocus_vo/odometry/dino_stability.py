from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from refocus_vo.backbones.dinov3 import DinoV3Backbone


@dataclass
class DinoObservation:
    rgb: torch.Tensor
    depth: torch.Tensor
    layers: Dict[int, torch.Tensor]
    fused: torch.Tensor
    patch_rgb: torch.Tensor
    patch_depth: torch.Tensor
    patch_valid: torch.Tensor


@dataclass
class DinoStabilityMap:
    patch_score: torch.Tensor
    pixel_score: torch.Tensor
    pixel_mask: torch.Tensor
    consistency: torch.Tensor
    boundary: torch.Tensor
    depth_edge_risk: torch.Tensor
    reprojection_risk: torch.Tensor


class DinoStabilityScorer:
    """
    Training-free DINOv3 stability scorer for dense RGB-D odometry.
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
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
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
        min_keep_pixels: int | None = None,
    ) -> torch.Tensor:
        valid = depth_m > 1e-6
        valid_scores = pixel_score[valid]
        if valid_scores.numel() == 0:
            return valid

        ratio = float(self.keep_ratio if keep_ratio is None else keep_ratio)
        ratio = min(max(ratio, 1e-3), 1.0)
        if ratio >= 0.999:
            return valid

        keep_floor = int(self.min_keep_pixels if min_keep_pixels is None else min_keep_pixels)
        keep_pixels = max(keep_floor, int(round(ratio * float(valid_scores.numel()))))
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

    def normalize_score_map(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize_map(value)

    def build_pixel_mask_from_patch_score(
        self,
        patch_score: torch.Tensor,
        depth_m: torch.Tensor,
        *,
        keep_ratio: float | None = None,
        min_keep_pixels: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = int(depth_m.shape[0]), int(depth_m.shape[1])
        pixel_score = F.interpolate(
            patch_score.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        pixel_mask = self.build_depth_mask(
            pixel_score,
            depth_m,
            keep_ratio=keep_ratio,
            min_keep_pixels=min_keep_pixels,
        )
        return pixel_score, pixel_mask

    def warp_patch_score_to_source(
        self,
        source: DinoObservation,
        target_patch_score: torch.Tensor,
        transform_target_source: np.ndarray | torch.Tensor,
        intrinsics: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        transform = torch.as_tensor(
            transform_target_source,
            device=self.device,
            dtype=torch.float32,
        )
        K = torch.as_tensor(intrinsics, device=self.device, dtype=torch.float32)
        projected_x, projected_y, valid = self._project_source_patch_coordinates(
            source,
            transform,
            K,
        )
        sampled = self._sample_feature_map(
            target_patch_score.unsqueeze(0),
            projected_x,
            projected_y,
        )[:, 0]
        sampled = torch.where(valid, sampled, torch.zeros_like(sampled))
        return sampled.reshape_as(source.patch_depth)

    def build_consensus_mask(
        self,
        source: DinoObservation,
        source_map: DinoStabilityMap,
        target_map: DinoStabilityMap,
        transform_target_source: np.ndarray | torch.Tensor,
        intrinsics: np.ndarray | torch.Tensor,
        *,
        keep_ratio: float | None = None,
        min_keep_pixels: int | None = None,
        fallback_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_norm = self.normalize_score_map(source_map.patch_score)
        target_warped = self.warp_patch_score_to_source(
            source,
            target_map.patch_score,
            transform_target_source,
            intrinsics,
        )
        target_norm = self.normalize_score_map(target_warped)
        consensus_patch = 0.5 * (source_norm + target_norm)
        pixel_score, pixel_mask = self.build_pixel_mask_from_patch_score(
            consensus_patch,
            source.depth,
            keep_ratio=keep_ratio,
            min_keep_pixels=min_keep_pixels,
        )
        if fallback_mask is not None and int(pixel_mask.sum().item()) < int(
            self.min_keep_pixels if min_keep_pixels is None else min_keep_pixels
        ):
            return consensus_patch, pixel_score, fallback_mask
        return consensus_patch, pixel_score, pixel_mask

    def _to_rgb_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float().div(255.0).to(self.device)

    def _to_depth_tensor(self, depth_m: np.ndarray) -> torch.Tensor:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Expected depth image of shape (H, W), got {depth.shape}")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 5000.0
        elif depth.max(initial=0.0) > 100.0:
            depth = depth.astype(np.float32) / 5000.0
        return torch.from_numpy(depth).float().to(self.device)

    def _pool_depth(self, depth_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (depth_t > 1e-6).float()
        depth_sum = F.avg_pool2d(
            (depth_t * valid).unsqueeze(0).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )[0, 0]
        valid_ratio = F.avg_pool2d(
            valid.unsqueeze(0).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )[0, 0]
        patch_valid = valid_ratio > 0.25
        patch_depth = torch.where(
            patch_valid,
            depth_sum / valid_ratio.clamp_min(1e-6),
            torch.zeros_like(depth_sum),
        )
        return patch_depth, patch_valid

    def _fuse_layers(self, layer_feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        ordered = [layer_feats[idx] for idx in self.layer_indices]
        fused = torch.stack(ordered, dim=0).mean(dim=0)
        return F.normalize(fused, dim=0, eps=1e-6)

    def _project_source_patch_coordinates(
        self,
        source: DinoObservation,
        transform: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        patch_x = ((u_target + 0.5) / float(self.patch_size)) - 0.5
        patch_y = ((v_target + 0.5) / float(self.patch_size)) - 0.5
        patch_x = torch.where(valid, patch_x, torch.zeros_like(patch_x))
        patch_y = torch.where(valid, patch_y, torch.zeros_like(patch_y))
        return patch_x, patch_y, valid

    def _patch_centers(self, Ht: int, Wt: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(Ht, device=device, dtype=torch.float32) * float(self.patch_size)
        xs = torch.arange(Wt, device=device, dtype=torch.float32) * float(self.patch_size)
        centers_v, centers_u = torch.meshgrid(
            ys + (float(self.patch_size) * 0.5 - 0.5),
            xs + (float(self.patch_size) * 0.5 - 0.5),
            indexing="ij",
        )
        return centers_u, centers_v

    def _sample_feature_map(self, feat_map: torch.Tensor, x_patch: torch.Tensor, y_patch: torch.Tensor) -> torch.Tensor:
        Ht, Wt = feat_map.shape[-2:]
        grid = torch.stack(
            [
                (x_patch / max(Wt - 1, 1)) * 2.0 - 1.0,
                (y_patch / max(Ht - 1, 1)) * 2.0 - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            feat_map.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, :, 0].t()
        return sampled

    def _sample_image(self, image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[-2:]
        grid = torch.stack(
            [
                (u / max(W - 1, 1)) * 2.0 - 1.0,
                (v / max(H - 1, 1)) * 2.0 - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, :, 0].t()
        return sampled

    def _sample_scalar_image(self, image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[-2:]
        grid = torch.stack(
            [
                (u / max(W - 1, 1)) * 2.0 - 1.0,
                (v / max(H - 1, 1)) * 2.0 - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0, :, 0]
        return sampled

    def _feature_boundary_strength(self, feat_map: torch.Tensor) -> torch.Tensor:
        dx = feat_map[:, :, 1:] - feat_map[:, :, :-1]
        dy = feat_map[:, 1:, :] - feat_map[:, :-1, :]
        dx_mag = torch.zeros(feat_map.shape[-2:], device=feat_map.device, dtype=torch.float32)
        dy_mag = torch.zeros(feat_map.shape[-2:], device=feat_map.device, dtype=torch.float32)
        dx_mag[:, 1:] = dx.norm(dim=0)
        dy_mag[1:, :] = dy.norm(dim=0)
        return dx_mag + dy_mag

    def _scalar_gradient(self, scalar_map: torch.Tensor) -> torch.Tensor:
        dx = torch.zeros_like(scalar_map)
        dy = torch.zeros_like(scalar_map)
        dx[:, 1:] = (scalar_map[:, 1:] - scalar_map[:, :-1]).abs()
        dy[1:, :] = (scalar_map[1:, :] - scalar_map[:-1, :]).abs()
        return dx + dy

    def _normalize_map(self, value: torch.Tensor) -> torch.Tensor:
        v = value.float()
        finite = torch.isfinite(v)
        if not finite.any():
            return torch.zeros_like(v)
        vals = v[finite]
        vmin = vals.min()
        vmax = vals.max()
        if float(vmax - vmin) < 1e-6:
            return torch.zeros_like(v)
        out = (v - vmin) / (vmax - vmin)
        return torch.where(finite, out, torch.zeros_like(out))
