from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace

import torch

from dino_slam3.losses.two_view_loss import compute_losses


class GeometryDepthMaskTests(unittest.TestCase):
    def test_requires_valid_depth2(self) -> None:
        torch.manual_seed(0)
        B, H, W = 1, 32, 32
        stride = 4
        Hf, Wf = H // stride, W // stride
        D = 16

        rgb1 = torch.rand(B, 3, H, W)
        rgb2 = torch.rand(B, 3, H, W)
        depth1 = torch.ones(B, 1, H, W)
        depth2 = torch.zeros(B, 1, H, W)  # all invalid
        valid1 = torch.ones(B, 1, H, W)
        valid2 = torch.zeros(B, 1, H, W)

        K = torch.tensor(
            [[[120.0, 0.0, (W - 1) / 2.0], [0.0, 120.0, (H - 1) / 2.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        T21 = torch.eye(4, dtype=torch.float32).unsqueeze(0)

        batch = {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "depth1": depth1,
            "depth2": depth2,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": K,
            "relative_pose": T21,
        }

        out1 = SimpleNamespace(
            heatmap=torch.randn(B, 1, Hf, Wf),
            desc=torch.randn(B, D, Hf, Wf),
            offset=torch.zeros(B, 2, Hf, Wf),
            reliability=None,
        )
        out2 = SimpleNamespace(
            heatmap=torch.randn(B, 1, Hf, Wf),
            desc=torch.randn(B, D, Hf, Wf),
            offset=torch.zeros(B, 2, Hf, Wf),
            reliability=None,
        )

        cfg = {
            "geom": {
                "sample_points": 128,
                "border": 2,
                "depth_consistency_m": 0.05,
                "depth_consistency_rel": 0.03,
                "require_valid_depth2": True,
                "fb_consistency_px": 2.0,
                "pose_weight": 0.0,
                "epipolar_weight": 0.0,
                "pose_det_weight": 0.0,
            },
            "contrastive": {"weight": 0.0},
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0, "mode": "none"},
            "z_min_m": 0.1,
        }

        _, stats = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertEqual(stats.num_valid, 0)
        self.assertEqual(stats.valid_ratio, 0.0)
        self.assertGreaterEqual(stats.occlusion_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
