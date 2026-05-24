from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .model import DinoPatchGraphVO, FrameObservation, PairPrediction


@dataclass
class PatchGraphTrackState:
    observation: FrameObservation
    pose_w_c: np.ndarray
    timestamp: float


def pose_vec_to_matrix(pose_vec: torch.Tensor) -> np.ndarray:
    pose_vec = pose_vec.detach().cpu().float().reshape(-1)
    rot = R.from_rotvec(pose_vec[:3].numpy()).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pose_vec[3:6].numpy()
    return T


class DinoPatchGraphTracker:
    def __init__(
        self,
        model: DinoPatchGraphVO,
        *,
        intrinsics: np.ndarray,
        track_confidence_threshold: float = 0.35,
        max_history: int = 3,
        use_geometric_pose: bool = True,
    ):
        self.model = model
        self.device = model.device
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.track_confidence_threshold = float(track_confidence_threshold)
        self.max_history = int(max_history)
        self.use_geometric_pose = bool(use_geometric_pose)
        self.history: list[PatchGraphTrackState] = []
        self.current_pose = np.eye(4, dtype=np.float64)
        self.num_matched_patches = 0
        self.graph_hidden: torch.Tensor | None = None

    def reset(self) -> None:
        self.history.clear()
        self.current_pose = np.eye(4, dtype=np.float64)
        self.num_matched_patches = 0
        self.graph_hidden = None

    def _image_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float().div(255.0)
        return image.to(self.device)

    def _estimate_pose_from_geometry(
        self,
        prev_obs: FrameObservation,
        cur_obs: FrameObservation,
        pair: PairPrediction,
    ) -> np.ndarray | None:
        if pair.src_indices.numel() < 8:
            return None
        conf = torch.sigmoid(pair.confidence_logits)
        keep = conf >= self.track_confidence_threshold
        if int(keep.sum().item()) < 8:
            return None

        src_xy = prev_obs.proposal.pixel_xy[pair.src_indices[keep]].detach().cpu().numpy().astype(np.float64)
        tgt_xy = cur_obs.proposal.pixel_xy[pair.tgt_indices[keep]].detach().cpu().numpy().astype(np.float64)
        fx, fy, cx, cy = self.intrinsics
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        E, mask = cv2.findEssentialMat(
            src_xy,
            tgt_xy,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0,
        )
        if E is None:
            return None
        _, R_est, t_est, _ = cv2.recoverPose(E, src_xy, tgt_xy, K, mask=mask)
        scale = float(np.linalg.norm(pose_vec_to_matrix(pair.pose_vec)[:3, 3]))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_est
        T[:3, 3] = t_est.reshape(3) * max(scale, 1e-3)
        return T

    def step(self, rgb: np.ndarray, timestamp: float) -> np.ndarray:
        image_t = self._image_to_tensor(rgb)
        cur_obs = self.model.infer_single_frame(image_t)

        if not self.history:
            self.history.append(
                PatchGraphTrackState(
                    observation=cur_obs,
                    pose_w_c=self.current_pose.copy(),
                    timestamp=float(timestamp),
                )
            )
            self.num_matched_patches = 0
            self.graph_hidden = None
            return self.current_pose.copy()

        history_obs = [state.observation for state in self.history[-self.max_history :]]
        incoming_pairs, pose_vec, self.graph_hidden = self.model.predict_target_from_history(
            history_obs,
            cur_obs,
            prev_hidden=self.graph_hidden,
        )
        self.num_matched_patches = int(sum(pair.src_indices.numel() for pair in incoming_pairs))

        rel_pose = None
        if self.use_geometric_pose and not self.model.use_multiframe_graph and incoming_pairs:
            prev_state = self.history[-1]
            rel_pose = self._estimate_pose_from_geometry(prev_state.observation, cur_obs, incoming_pairs[-1])
        if rel_pose is None:
            rel_pose = pose_vec_to_matrix(pose_vec)

        prev_pose = self.history[-1].pose_w_c
        self.current_pose = prev_pose @ rel_pose
        self.history.append(
            PatchGraphTrackState(
                observation=cur_obs,
                pose_w_c=self.current_pose.copy(),
                timestamp=float(timestamp),
            )
        )
        self.history = self.history[-self.max_history :]
        return self.current_pose.copy()
