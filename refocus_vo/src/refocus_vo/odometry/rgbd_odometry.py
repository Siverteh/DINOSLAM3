from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c


@dataclass
class CameraShim:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    D: np.ndarray
    depth_factor: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class VoState(Enum):
    NO_IMAGES_YET = 0
    GOT_FIRST_IMAGE = 1


class _NullTimer:
    def start(self) -> None:
        pass

    def refresh(self) -> None:
        pass


def invert_transform(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    trans = np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
    out[:3, :3] = rot.T
    out[:3, 3] = -(rot.T @ trans)
    return out


class VisualOdometryRgbdTensor:
    """
    Lightweight dense RGB-D odometry used by the refocus path.
    """

    def __init__(self, cam: CameraShim, groundtruth=None, method_name: str = "hybrid", device: str = "cuda"):
        del groundtruth
        self.cam = cam
        self.state = VoState.NO_IMAGES_YET

        self.cur_image = None
        self.cur_depth = None
        self.cur_timestamp = None
        self.prev_image = None
        self.prev_depth = None
        self.prev_timestamp = None

        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        self.num_matched_kps = 0
        self.num_inliers = 0
        self.draw_img = None
        self.timer_pose_est = _NullTimer()

        self.depth_factor = float(cam.depth_factor)
        self.prev_rgbd = None
        self.cur_rgbd = None

        h, w = int(cam.height), int(cam.width)
        D = np.asarray(cam.D if cam.D is not None else [0, 0, 0, 0, 0], dtype=np.float64).flatten()
        K = np.asarray(cam.K, dtype=np.float64)
        if np.linalg.norm(D) <= 1e-10:
            self.new_K = K
            self.calib_map1 = None
            self.calib_map2 = None
        else:
            self.new_K = K
            self.calib_map1, self.calib_map2 = cv2.initUndistortRectifyMap(
                K, D, None, self.new_K, (w, h), cv2.CV_32FC1
            )

        self.rectified_fx = float(self.new_K[0, 0])
        self.rectified_fy = float(self.new_K[1, 1])
        self.rectified_cx = float(self.new_K[0, 2])
        self.rectified_cy = float(self.new_K[1, 2])

        device_name = "CUDA:0" if device == "cuda" and o3d.core.cuda.is_available() else "CPU:0"
        self.device = o3c.Device(device_name)
        self.intrinsics = o3d.core.Tensor(
            np.array(
                [
                    [self.rectified_fx, 0.0, self.rectified_cx],
                    [0.0, self.rectified_fy, self.rectified_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            o3d.core.Dtype.Float64,
        )
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
        ]
        self.method = self._resolve_method(str(method_name))
        self.max_depth = 10.0
        self.depth_scale = 1.0

    def rectify_in_needed(self, color: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth_out = np.asarray(depth, dtype=np.float32)
        if self.depth_factor != 1.0:
            depth_out = depth_out * self.depth_factor

        color_out = np.asarray(color)
        if color_out.ndim == 2:
            color_out = cv2.cvtColor(color_out, cv2.COLOR_GRAY2BGR)

        if self.calib_map1 is not None and self.calib_map2 is not None:
            color_out = cv2.remap(color_out, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_LINEAR)
            depth_out = cv2.remap(depth_out, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_NEAREST)

        return cv2.cvtColor(color_out, cv2.COLOR_BGR2RGB), depth_out

    def track(self, img, img_right, depth, frame_id, timestamp) -> None:
        del img_right
        assert img.shape[0] == self.cam.height and img.shape[1] == self.cam.width
        self.cur_image = img
        self.cur_depth = depth
        self.cur_timestamp = timestamp
        if self.state == VoState.GOT_FIRST_IMAGE:
            self.process_frame(frame_id)
        else:
            self.process_first_frame(frame_id)
            self.state = VoState.GOT_FIRST_IMAGE
        self.prev_image = self.cur_image
        self.prev_depth = self.cur_depth
        self.prev_timestamp = self.cur_timestamp

    def update_gt_data(self, frame_id) -> None:
        del frame_id

    def process_first_frame(self, frame_id) -> None:
        del frame_id
        self.draw_img = self.cur_image.copy()
        if self.cur_depth is None:
            raise ValueError("Depth image is None, are you using a dataset with depth images?")
        color_undistorted, depth_undistorted = self.rectify_in_needed(self.cur_image, self.cur_depth)
        self.prev_rgbd = self._make_rgbd(color_undistorted, depth_undistorted)

    def process_frame(self, frame_id) -> None:
        del frame_id
        self.draw_img = self.cur_image.copy()
        color_undistorted, depth_undistorted = self.rectify_in_needed(self.cur_image, self.cur_depth)
        self.cur_rgbd = self._make_rgbd(color_undistorted, depth_undistorted)

        rel_transform = self._run_odometry(
            self.prev_rgbd,
            self.cur_rgbd,
            method=self.method,
            init_transform=np.eye(4, dtype=np.float64),
        )
        self.prev_rgbd = self.cur_rgbd.clone()

        inv_pose = invert_transform(rel_transform)
        rot = inv_pose[:3, :3]
        trans = inv_pose[:3, 3].reshape(3, 1)
        self.cur_R = self.cur_R @ rot
        self.cur_t = self.cur_t + self.cur_R @ trans

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
