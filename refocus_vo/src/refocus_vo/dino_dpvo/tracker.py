from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .adapter import build_dpvo_patch_input
from .config import DinoDPVOConfig
from .frontend import DinoProposalFrontend


class DinoDPVOTracker:
    def __init__(
        self,
        slam: Any,
        *,
        frontend: DinoProposalFrontend | None = None,
        frontend_cfg: DinoDPVOConfig | None = None,
        frontend_mode: str = "dpvo_native",
        patch_budget: int = 24,
        dpvo_res: int = 4,
        collect_diagnostics: bool = False,
        hybrid_grid_rows: int = 6,
        hybrid_grid_cols: int = 8,
    ):
        self.slam = slam
        self.frontend = frontend
        self.frontend_cfg = frontend_cfg
        self.frontend_mode = str(frontend_mode)
        self.patch_budget = int(patch_budget)
        self.dpvo_res = int(dpvo_res)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.hybrid_grid_rows = int(hybrid_grid_rows)
        self.hybrid_grid_cols = int(hybrid_grid_cols)
        self.patch_runtime_state: dict[str, Any] = {}
        self.slam.collect_patch_diagnostics = bool(collect_diagnostics)
        self.slam.patch_diag_grid_rows = int(hybrid_grid_rows)
        self.slam.patch_diag_grid_cols = int(hybrid_grid_cols)

    def _to_frontend_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach()
            if tensor.dim() != 3:
                raise ValueError(f"Expected CHW image tensor, got {tuple(tensor.shape)}")
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div(255.0)
            return tensor

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
            if tensor.max() > 1.0:
                tensor = tensor.div(255.0)
            return tensor
        if array.ndim == 3 and array.shape[2] in {1, 3}:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
            if tensor.max() > 1.0:
                tensor = tensor.div(255.0)
            return tensor
        raise ValueError(f"Unsupported image shape for DINO frontend: {array.shape}")

    def _to_slam_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach()
            if tensor.dtype != torch.uint8:
                tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return tensor.to("cuda")
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.uint8, device="cuda")
        if array.ndim == 3 and array.shape[2] in {1, 3}:
            return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).to(torch.uint8).to("cuda")
        raise ValueError(f"Unsupported image shape for DPVO runtime: {array.shape}")

    def step(self, tstamp: float, image: np.ndarray | torch.Tensor, intrinsics: np.ndarray | torch.Tensor) -> None:
        if self.frontend is not None and self.frontend_mode != "dpvo_native":
            frontend_image = self._to_frontend_tensor(image).to(self.frontend.device)
            frame_output = self.frontend.infer_single_frame(frontend_image)
            patch_input = build_dpvo_patch_input(
                frame_output,
                patch_budget=self.patch_budget,
                frontend_mode=self.frontend_mode,
                dpvo_res=self.dpvo_res,
                image_height=int(frontend_image.shape[-2]),
                image_width=int(frontend_image.shape[-1]),
                config=self.frontend_cfg.model if self.frontend_cfg is not None else None,
                runtime_state=self.patch_runtime_state,
            )
            self.slam.pending_patch_metadata = patch_input.pop("patch_metadata", None)
            self.slam.pending_patch_input = patch_input

        slam_image = self._to_slam_tensor(image)
        slam_intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device="cuda")
        self.slam(float(tstamp), slam_image, slam_intrinsics)

    def terminate(self):
        return self.slam.terminate()
