from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import yaml

from .dino_stability import DinoObservation, DinoStabilityScorer
from .rgbd_odometry import CameraShim, VisualOdometryRgbdTensor, invert_transform


def _tum_settings_name(sequence: str) -> str:
    seq = str(sequence).replace("rgbd_dataset_", "", 1)
    if seq.startswith("freiburg1_"):
        return "TUM1.yaml"
    if seq.startswith("freiburg2_"):
        return "TUM2.yaml"
    if seq.startswith("freiburg3_"):
        return "TUM3.yaml"
    raise ValueError(f"Unsupported TUM sequence name: {sequence}")


def load_tum_camera(sequence: str, settings_dir: str | Path | None = None) -> CameraShim:
    settings_root = (
        Path(settings_dir)
        if settings_dir is not None
        else Path(__file__).resolve().parents[3] / "pyslam" / "settings"
    )
    settings_path = settings_root / _tum_settings_name(sequence)
    if not settings_path.exists():
        raise FileNotFoundError(f"TUM settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    depth_map_factor = float(cfg.get("DepthMapFactor", 5000.0))
    if depth_map_factor <= 0.0:
        raise ValueError(f"Invalid DepthMapFactor in {settings_path}: {depth_map_factor}")

    D = np.array(
        [
            float(cfg.get("Camera.k1", 0.0)),
            float(cfg.get("Camera.k2", 0.0)),
            float(cfg.get("Camera.p1", 0.0)),
            float(cfg.get("Camera.p2", 0.0)),
            float(cfg.get("Camera.k3", 0.0)),
        ],
        dtype=np.float64,
    )

    return CameraShim(
        width=int(cfg["Camera.width"]),
        height=int(cfg["Camera.height"]),
        fx=float(cfg["Camera.fx"]),
        fy=float(cfg["Camera.fy"]),
        cx=float(cfg["Camera.cx"]),
        cy=float(cfg["Camera.cy"]),
        D=D,
        depth_factor=1.0 / depth_map_factor,
    )


class DinoGuidedVisualOdometryRgbdTensor(VisualOdometryRgbdTensor):
    """
    Two-pass dense RGB-D odometry:
      1. coarse hybrid odometry on the full RGB-D pair
      2. DINOv3 stability masking
      3. point-to-plane refinement on the masked pair
    """

    def __init__(
        self,
        cam,
        scorer: DinoStabilityScorer,
        *,
        keep_ratio: float = 0.35,
        coarse_method: str = "hybrid",
        refine_method: str = "point_to_plane",
        min_refine_pixels: int = 4096,
        device: str = "cuda",
    ):
        super().__init__(cam, groundtruth=None, method_name=refine_method, device=device)
        self.scorer = scorer
        self.keep_ratio = float(keep_ratio)
        self.min_refine_pixels = max(1, int(min_refine_pixels))
        self.coarse_method = self._resolve_method(str(coarse_method))
        self.refine_method = self._resolve_method(str(refine_method))

        self.prev_color_undistorted: np.ndarray | None = None
        self.prev_depth_undistorted: np.ndarray | None = None
        self.prev_observation: DinoObservation | None = None
        self.last_stability_stats: dict[str, float] = {}

        self._intrinsics_np = self.intrinsics.cpu().numpy().astype(np.float32)

    def process_first_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()

        if self.cur_depth is None:
            raise ValueError("Depth image is None, are you using a dataset with depth images?")

        color_undistorted, depth_undistorted = self.rectify_in_needed(self.cur_image, self.cur_depth)
        depth_undistorted = np.asarray(depth_undistorted, dtype=np.float32)

        self.prev_rgbd = self._make_rgbd(color_undistorted, depth_undistorted)
        self.prev_color_undistorted = color_undistorted
        self.prev_depth_undistorted = depth_undistorted
        self.prev_observation = self.scorer.observe(color_undistorted, depth_undistorted)

    def process_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()
        self.update_gt_data(frame_id)

        assert self.prev_rgbd is not None
        assert self.prev_color_undistorted is not None
        assert self.prev_depth_undistorted is not None
        assert self.prev_observation is not None

        color_undistorted, depth_undistorted = self.rectify_in_needed(self.cur_image, self.cur_depth)
        depth_undistorted = np.asarray(depth_undistorted, dtype=np.float32)

        current_full_rgbd = self._make_rgbd(color_undistorted, depth_undistorted)
        current_observation = self.scorer.observe(color_undistorted, depth_undistorted)

        self.timer_pose_est.start()
        coarse_transform = self._run_odometry(
            self.prev_rgbd,
            current_full_rgbd,
            method=self.coarse_method,
            init_transform=np.eye(4, dtype=np.float64),
        )

        prev_map = self.scorer.score_pair(
            self.prev_observation,
            current_observation,
            coarse_transform,
            self._intrinsics_np,
            keep_ratio=self.keep_ratio,
        )
        current_map = self.scorer.score_pair(
            current_observation,
            self.prev_observation,
            np.linalg.inv(coarse_transform),
            self._intrinsics_np,
            keep_ratio=self.keep_ratio,
        )

        prev_kept = int(prev_map.pixel_mask.sum().item())
        current_kept = int(current_map.pixel_mask.sum().item())
        refine_ok = min(prev_kept, current_kept) >= self.min_refine_pixels

        if refine_ok:
            masked_prev_depth = self.scorer.apply_mask_to_depth(
                self.prev_depth_undistorted,
                prev_map.pixel_mask,
            )
            masked_current_depth = self.scorer.apply_mask_to_depth(
                depth_undistorted,
                current_map.pixel_mask,
            )
            masked_prev_rgbd = self._make_rgbd(self.prev_color_undistorted, masked_prev_depth)
            masked_current_rgbd = self._make_rgbd(color_undistorted, masked_current_depth)
            rel_transform = self._run_odometry(
                masked_prev_rgbd,
                masked_current_rgbd,
                method=self.refine_method,
                init_transform=coarse_transform,
            )
        else:
            rel_transform = coarse_transform

        self.timer_pose_est.refresh()

        self.cur_rgbd = current_full_rgbd
        self.prev_rgbd = current_full_rgbd.clone()
        self.prev_color_undistorted = color_undistorted
        self.prev_depth_undistorted = depth_undistorted
        self.prev_observation = current_observation

        self.last_stability_stats = {
            "prev_kept_pixels": float(prev_kept),
            "current_kept_pixels": float(current_kept),
            "prev_keep_ratio": float(prev_kept) / float(prev_map.pixel_mask.numel()),
            "current_keep_ratio": float(current_kept) / float(current_map.pixel_mask.numel()),
            "refine_ok": 1.0 if refine_ok else 0.0,
        }
        self.num_matched_kps = prev_kept
        self.num_inliers = current_kept

        inv_pose = invert_transform(rel_transform)
        R, t = inv_pose[:3, :3], inv_pose[:3, 3]
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)

        self.cur_R = self.cur_R @ R
        self.cur_t = self.cur_t + self.cur_R @ t

    def _run_odometry(
        self,
        prev_rgbd: o3d.t.geometry.RGBDImage,
        cur_rgbd: o3d.t.geometry.RGBDImage,
        *,
        method,
        init_transform: np.ndarray,
    ) -> np.ndarray:
        result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            prev_rgbd,
            cur_rgbd,
            self.intrinsics,
            o3c.Tensor(np.asarray(init_transform, dtype=np.float64)),
            self.depth_scale,
            self.max_depth,
            self.criteria_list,
            method,
        )
        transform = result.transformation.cpu().numpy()
        if not np.all(np.isfinite(transform)):
            raise RuntimeError("Open3D returned a non-finite RGB-D transform")
        return transform

    def _make_rgbd(self, color_rgb: np.ndarray, depth_m: np.ndarray) -> o3d.t.geometry.RGBDImage:
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.asarray(color_rgb)).to(self.device),
            o3d.t.geometry.Image(np.asarray(depth_m, dtype=np.float32)).to(self.device),
        )

    @staticmethod
    def _resolve_method(name: str):
        if name == "hybrid":
            return o3d.t.pipelines.odometry.Method.Hybrid
        if name == "point_to_plane":
            return o3d.t.pipelines.odometry.Method.PointToPlane
        raise ValueError(f"Unsupported RGB-D odometry method: {name}")
