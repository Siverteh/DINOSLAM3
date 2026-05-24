from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.losses.two_view_loss import compute_losses
from dino_slam3.models.heads import FeatureOutputs


class HardSamplingLossTests(unittest.TestCase):
    def _build_minibatch(self):
        torch.manual_seed(0)
        B, H, W = 1, 32, 32
        stride = 4
        Hf, Wf = H // stride, W // stride
        D = 16

        rgb1 = torch.rand(B, 3, H, W)
        rgb2 = torch.rand(B, 3, H, W)
        depth = torch.ones(B, 1, H, W)
        valid = torch.ones(B, 1, H, W)

        K = torch.tensor(
            [[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
        )
        T = torch.eye(4, dtype=torch.float32).unsqueeze(0)

        desc1 = torch.nn.functional.normalize(torch.rand(B, D, Hf, Wf), dim=1)
        desc2 = torch.nn.functional.normalize(torch.rand(B, D, Hf, Wf), dim=1)

        out1 = FeatureOutputs(
            heatmap=torch.zeros(B, 1, Hf, Wf),
            desc=desc1,
            offset=torch.zeros(B, 2, Hf, Wf),
            reliability=None,
        )
        out2 = FeatureOutputs(
            heatmap=torch.zeros(B, 1, Hf, Wf),
            desc=desc2,
            offset=torch.zeros(B, 2, Hf, Wf),
            reliability=None,
        )

        batch = {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "depth1": depth,
            "depth2": depth,
            "valid_depth1": valid,
            "valid_depth2": valid,
            "K": K,
            "relative_pose": T,
            "frame_delta": torch.tensor([100], dtype=torch.int64),
        }
        return batch, out1, out2, stride

    def test_hard_sampling_handles_empty_hard_pool(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()

        cfg = {
            "geom": {
                "sample_points": 128,
                "depth_consistency_m": 0.05,
                "depth_consistency_rel": 0.03,
                "require_valid_depth2": True,
                "fb_consistency_px": 2.0,
            },
            "contrastive": {
                "temperature": 0.07,
                "max_positives": 64,
                "hard_sample_fraction": 0.9,
                "hard_min_displacement_px": 999.0,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }

        losses, _stats = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_teacher_loss_zero_when_disabled(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        B, _, Hf, Wf = out1.heatmap.shape
        batch_with_teacher = dict(batch)
        batch_with_teacher["teacher_heatmap1"] = torch.rand(B, 1, Hf, Wf)
        batch_with_teacher["teacher_heatmap2"] = torch.rand(B, 1, Hf, Wf)

        cfg = {
            "geom": {"sample_points": 64},
            "contrastive": {"max_positives": 64, "weight": 0.0},
            "detector": {"weight": 1.0, "sparsity_weight": 0.0, "teacher_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }

        torch.manual_seed(7)
        losses_a, _ = compute_losses(batch_with_teacher, out1, out2, cfg=cfg, epoch=1, stride=stride)
        torch.manual_seed(7)
        losses_b, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertAlmostEqual(float(losses_a["loss_repeat"]), float(losses_b["loss_repeat"]), places=6)

    def test_long_range_weighting_stable_edge_cases(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        base = {
            "geom": {"sample_points": 128},
            "contrastive": {
                "temperature": 0.07,
                "max_positives": 64,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }

        cfg_no_long = dict(base)
        cfg_no_long["contrastive"] = dict(base["contrastive"], long_disp_px=9999.0, long_weight=2.5)
        cfg_all_long = dict(base)
        cfg_all_long["contrastive"] = dict(base["contrastive"], long_disp_px=0.0, long_weight=2.5)

        losses_no_long, _ = compute_losses(batch, out1, out2, cfg=cfg_no_long, epoch=1, stride=stride)
        losses_all_long, _ = compute_losses(batch, out1, out2, cfg=cfg_all_long, epoch=1, stride=stride)
        self.assertTrue(torch.isfinite(losses_no_long["loss_total"]).item())
        self.assertTrue(torch.isfinite(losses_all_long["loss_total"]).item())

    def test_circle_mode_finite_for_degenerate_batches(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {"sample_points": 32},
            "contrastive": {
                "mode": "circle",
                "max_positives": 16,
                "min_pairs": 2,
                "circle_margin": 0.25,
                "circle_gamma": 32.0,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_hard_mining_warmup_disables_hard_fraction_before_start(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg_warm = {
            "geom": {"sample_points": 128},
            "contrastive": {
                "temperature": 0.07,
                "max_positives": 64,
                "hard_sample_fraction": 0.9,
                "hard_mining_start_epoch": 5,
                "hard_min_displacement_px": 0.0,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        cfg_off = {
            **cfg_warm,
            "contrastive": {
                **cfg_warm["contrastive"],
                "hard_sample_fraction": 0.0,
            },
        }
        torch.manual_seed(11)
        losses_warm, _ = compute_losses(batch, out1, out2, cfg=cfg_warm, epoch=1, stride=stride)
        torch.manual_seed(11)
        losses_off, _ = compute_losses(batch, out1, out2, cfg=cfg_off, epoch=1, stride=stride)
        self.assertAlmostEqual(float(losses_warm["loss_desc"]), float(losses_off["loss_desc"]), places=6)

    def test_offset_enable_false_zeroes_refine_loss(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {"sample_points": 64},
            "contrastive": {"max_positives": 64, "weight": 0.0},
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {
                "enable": False,
                "weight": 1.0,
                "smoothness_weight": 1.0,
                "bias_weight": 1.0,
            },
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertAlmostEqual(float(losses["loss_refine"]), 0.0, places=7)

    def test_reliability_geom_correctness_mode_is_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        out1_rel = FeatureOutputs(
            heatmap=out1.heatmap,
            desc=out1.desc,
            offset=out1.offset,
            reliability=torch.zeros_like(out1.heatmap),
        )
        out2_rel = FeatureOutputs(
            heatmap=out2.heatmap,
            desc=out2.desc,
            offset=out2.offset,
            reliability=torch.zeros_like(out2.heatmap),
        )
        cfg = {
            "geom": {"sample_points": 128},
            "contrastive": {"max_positives": 64, "weight": 0.0},
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {
                "weight": 1.0,
                "mode": "geom_correctness",
                "target_mean": 0.1,
                "mean_reg_weight": 0.1,
            },
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1_rel, out2_rel, cfg=cfg, epoch=1, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_rel"]).item())

    def test_temperature_schedule_and_uniformity_are_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {"sample_points": 128},
            "contrastive": {
                "mode": "infonce",
                "temperature": 0.07,
                "temperature_schedule": {"start": 0.06, "end": 0.03},
                "uniformity_weight": 0.03,
                "max_positives": 64,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
            "__total_epochs": 10,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=5, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_desc"]).item())
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_adaptive_temperature_and_saliency_terms_are_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        B, _, Hf, Wf = out1.heatmap.shape
        batch["teacher_heatmap1"] = torch.rand(B, 1, Hf, Wf)
        batch["teacher_heatmap2"] = torch.rand(B, 1, Hf, Wf)
        cfg = {
            "geom": {"sample_points": 128},
            "contrastive": {
                "mode": "infonce",
                "temperature": 0.07,
                "adaptive_temperature_by_gap": True,
                "temperature_gap_low": 0.025,
                "temperature_gap_high": 0.07,
                "long_disp_px": 20.0,
                "max_positives": 64,
                "weight": 1.0,
            },
            "detector": {
                "weight": 1.0,
                "sparsity_weight": 0.0,
                "saliency_consistency_weight": 0.2,
                "saliency_teacher_mix": 0.5,
                "saliency_entropy_weight": 0.01,
            },
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=2, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_repeat"]).item())
        self.assertTrue(torch.isfinite(losses["loss_desc"]).item())
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_pose_charbonnier_mode_is_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {
                "sample_points": 128,
                "pose_weight": 1.0,
                "pose_error_mode": "charbonnier",
                "pose_charb_eps": 1.0,
            },
            "contrastive": {"max_positives": 64, "weight": 0.0},
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_pose"]).item())

    def test_multisim_and_memory_bank_are_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {"sample_points": 128},
            "contrastive": {
                "mode": "multisim",
                "max_positives": 64,
                "memory_bank_size": 256,
                "memory_momentum": 0.5,
                "weight": 1.0,
            },
            "detector": {"weight": 0.0, "sparsity_weight": 0.0},
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses1, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=1, stride=stride)
        losses2, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=2, stride=stride)
        self.assertTrue(torch.isfinite(losses1["loss_desc"]).item())
        self.assertTrue(torch.isfinite(losses2["loss_desc"]).item())

    def test_loop_and_depth_edge_objectives_are_finite(self) -> None:
        batch, out1, out2, stride = self._build_minibatch()
        cfg = {
            "geom": {
                "sample_points": 128,
                "loop_consistency_weight": 0.3,
                "loop_min_gap": 80,
                "loop_pose_dist_m": 0.4,
                "loop_yaw_deg": 20.0,
                "pose_weight": 0.0,
                "cycle_weight": 0.0,
            },
            "contrastive": {
                "max_positives": 64,
                "depth_edge_separation_weight": 0.05,
                "weight": 1.0,
            },
            "detector": {
                "weight": 1.0,
                "sparsity_weight": 0.0,
                "depth_edge_consistency_weight": 0.05,
            },
            "offset": {"weight": 0.0},
            "reliability": {"weight": 0.0},
            "z_min_m": 0.1,
        }
        losses, _ = compute_losses(batch, out1, out2, cfg=cfg, epoch=6, stride=stride)
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())


if __name__ == "__main__":
    unittest.main()
