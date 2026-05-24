from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.dino_dpvo import (
    DinoDPVOFrameOutput,
    DinoDPVOTracker,
    build_dpvo_patch_input,
    dense_gradient_offset_targets,
    load_dino_dpvo_config,
    pixel_to_dpvo_coords,
)
from refocus_vo.dino_dpvo.frontend import load_matching_state_dict
from refocus_vo.eval.aggregate_trial_medians import main as aggregate_trial_medians_main
from refocus_vo.eval.dpvo_style_metrics import write_dpvo_style_csv
from refocus_vo.patchgraph.teacher import PseudoObjectPatchProposal
from refocus_vo.train_dino_dpvo_frontend import (
    _checkpoint_candidate_key,
    _compute_coverage_regularizer,
    _read_diagnostics_summary_metrics,
    _selection_score,
)
from refocus_vo.train_dino_dpvo_semantic_full import _load_dpvo_init_weights, _semantic_fraction_target


def _make_frame_output(*, with_descriptor_bias: bool) -> DinoDPVOFrameOutput:
    proposal = PseudoObjectPatchProposal(
        patch_indices=torch.tensor([0, 1], dtype=torch.long),
        patch_xy=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        coarse_pixel_xy=torch.tensor([[16.0, 24.0], [48.0, 40.0]], dtype=torch.float32),
        pixel_xy=torch.tensor([[18.0, 26.0], [52.0, 44.0]], dtype=torch.float32),
        offset_xy=torch.tensor([[2.0, 2.0], [4.0, 4.0]], dtype=torch.float32),
        scores=torch.tensor([0.8, 0.4], dtype=torch.float32),
        object_ids=torch.tensor([0, 1], dtype=torch.long),
        descriptors=torch.zeros((2, 4), dtype=torch.float32),
        local_features=torch.zeros((2, 4), dtype=torch.float32),
    )
    return DinoDPVOFrameOutput(
        proposal=proposal,
        selector_logits=torch.zeros((2, 2), dtype=torch.float32),
        staticness_logits=torch.zeros((2, 2), dtype=torch.float32),
        gradient_score=torch.zeros((2, 2), dtype=torch.float32),
        qualities=torch.tensor([0.8, 0.4], dtype=torch.float32),
        descriptor_bias=torch.arange(8, dtype=torch.float32).reshape(2, 4) if with_descriptor_bias else None,
    )


class _FrontendStub:
    def __init__(self, frame_output: DinoDPVOFrameOutput):
        self.device = torch.device("cpu")
        self.frame_output = frame_output

    def infer_single_frame(self, image: torch.Tensor) -> DinoDPVOFrameOutput:
        return self.frame_output


class _SlamStub:
    def __init__(self):
        self.pending_patch_input = None
        self.calls: list[tuple[float, tuple[int, ...], tuple[float, ...]]] = []

    def __call__(self, tstamp: float, image: torch.Tensor, intrinsics: torch.Tensor) -> None:
        self.calls.append((float(tstamp), tuple(image.shape), tuple(float(v) for v in intrinsics.tolist())))


class DinoDPVOPipelineTests(unittest.TestCase):
    def test_pixel_to_dpvo_coords_maps_pixel_centers(self) -> None:
        pixel_xy = torch.tensor([[18.0, 26.0]], dtype=torch.float32)
        coords = pixel_to_dpvo_coords(pixel_xy, dpvo_res=4)
        self.assertTrue(torch.allclose(coords, torch.tensor([[4.0, 6.0]], dtype=torch.float32)))

    def test_build_dpvo_patch_input_repeats_to_budget_and_adds_descriptor_bias_only_for_full(self) -> None:
        frame_output = _make_frame_output(with_descriptor_bias=True)
        proposals = build_dpvo_patch_input(
            frame_output,
            patch_budget=4,
            frontend_mode="dino_proposals",
            dpvo_res=4,
        )
        self.assertEqual(tuple(proposals["external_coords"].shape), (1, 4, 2))
        self.assertEqual(tuple(proposals["external_quality"].shape), (1, 4, 1))
        self.assertNotIn("external_descriptor_bias", proposals)

        full = build_dpvo_patch_input(
            frame_output,
            patch_budget=4,
            frontend_mode="dino_full",
            dpvo_res=4,
        )
        self.assertEqual(tuple(full["external_descriptor_bias"].shape), (1, 4, 4))
        self.assertIn("patch_metadata", full)

    def test_build_dpvo_patch_input_hybrid_mixes_native_and_dino(self) -> None:
        torch.manual_seed(7)
        frame_output = _make_frame_output(with_descriptor_bias=False)
        hybrid = build_dpvo_patch_input(
            frame_output,
            patch_budget=8,
            frontend_mode="dino_hybrid",
            dpvo_res=4,
            image_height=64,
            image_width=64,
            config={
                "native_fraction": 0.75,
                "dino_fraction": 0.25,
                "static_score_weight": 0.35,
                "hybrid_grid_rows": 6,
                "hybrid_grid_cols": 8,
                "max_dino_per_cell": 1,
                "dedupe_radius_px": 0.0,
            },
        )
        self.assertEqual(tuple(hybrid["external_coords"].shape), (1, 8, 2))
        self.assertEqual(tuple(hybrid["external_quality"].shape), (1, 8, 1))
        self.assertTrue(torch.allclose(hybrid["external_quality"], torch.ones_like(hybrid["external_quality"])))
        sources = hybrid["patch_metadata"]["source_labels"]
        self.assertEqual(int((sources == 0).sum().item()), 6)
        self.assertEqual(int((sources == 1).sum().item()), 2)

    def test_build_dpvo_patch_input_semantic_fullcover_stays_semantic_only(self) -> None:
        torch.manual_seed(7)
        frame_output = _make_frame_output(with_descriptor_bias=False)
        fullcover = build_dpvo_patch_input(
            frame_output,
            patch_budget=4,
            frontend_mode="dino_proposals",
            dpvo_res=4,
            image_height=64,
            image_width=64,
            config={
                "enforce_unique_semantic": True,
                "semantic_backfill_source": "dino",
                "semantic_grid_rows": 6,
                "semantic_grid_cols": 8,
                "max_semantic_per_cell": 2,
                "static_score_weight": 0.35,
                "dedupe_radius_px": 0.0,
            },
        )
        self.assertEqual(tuple(fullcover["external_coords"].shape), (1, 4, 2))
        self.assertEqual(tuple(fullcover["external_quality"].shape), (1, 4, 1))
        sources = fullcover["patch_metadata"]["source_labels"]
        self.assertEqual(int((sources == 1).sum().item()), 4)
        self.assertEqual(int((sources == 0).sum().item()), 0)

    def test_build_dpvo_patch_input_semantic_fullcover_records_repeat_metadata(self) -> None:
        torch.manual_seed(7)
        frame_output = _make_frame_output(with_descriptor_bias=False)
        fullcover = build_dpvo_patch_input(
            frame_output,
            patch_budget=4,
            frontend_mode="dino_proposals",
            dpvo_res=4,
            image_height=64,
            image_width=64,
            config={
                "enforce_unique_semantic": True,
                "semantic_backfill_source": "dino",
                "semantic_grid_rows": 6,
                "semantic_grid_cols": 8,
                "max_semantic_per_cell": 2,
                "static_score_weight": 0.35,
                "dedupe_radius_px": 8.0,
                "semantic_dedupe_schedule_px": [8.0, 6.0, 4.0],
            },
        )
        metadata = fullcover["patch_metadata"]
        self.assertIn("repeated_patch_flags", metadata)
        self.assertIn("unique_semantic_count_before_repeat", metadata)
        self.assertIn("dedupe_radius_used", metadata)
        self.assertEqual(tuple(metadata["repeated_patch_flags"].shape), (4,))
        self.assertGreater(float(metadata["repeated_patch_flags"].sum().item()), 0.0)

    def test_dense_gradient_offset_targets_returns_bounded_offsets(self) -> None:
        x = torch.linspace(0.0, 1.0, 32)
        y = torch.linspace(0.0, 1.0, 32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        image = torch.stack([xx, yy, xx], dim=0).unsqueeze(0).unsqueeze(0)
        offsets = dense_gradient_offset_targets(image, patch_size=16)
        self.assertEqual(tuple(offsets.shape), (1, 1, 2, 2, 2))
        self.assertLessEqual(float(offsets.abs().max().item()), 7.5 + 1e-5)

    def test_tracker_only_sets_pending_patch_input_for_non_native_mode(self) -> None:
        frame_output = _make_frame_output(with_descriptor_bias=True)
        image = torch.zeros((3, 32, 32), dtype=torch.uint8)
        intrinsics = torch.tensor([100.0, 100.0, 16.0, 16.0], dtype=torch.float32)

        native_slam = _SlamStub()
        native_tracker = DinoDPVOTracker(native_slam, frontend=None, frontend_mode="dpvo_native", patch_budget=4)
        original_to_slam_tensor = DinoDPVOTracker._to_slam_tensor
        DinoDPVOTracker._to_slam_tensor = lambda self, img: torch.zeros((3, 32, 32), dtype=torch.uint8)
        try:
            native_tracker.step(0.0, image, intrinsics)
        finally:
            DinoDPVOTracker._to_slam_tensor = original_to_slam_tensor
        self.assertIsNone(native_slam.pending_patch_input)

        slam = _SlamStub()
        tracker = DinoDPVOTracker(
            slam,
            frontend=_FrontendStub(frame_output),
            frontend_mode="dino_full",
            patch_budget=4,
            frontend_cfg=None,
        )
        DinoDPVOTracker._to_slam_tensor = lambda self, img: torch.zeros((3, 32, 32), dtype=torch.uint8)
        try:
            tracker.step(1.0, image, intrinsics)
        finally:
            DinoDPVOTracker._to_slam_tensor = original_to_slam_tensor
        self.assertIsNotNone(slam.pending_patch_input)
        self.assertIsNotNone(slam.pending_patch_metadata)
        self.assertIn("external_descriptor_bias", slam.pending_patch_input)
        self.assertEqual(len(slam.calls), 1)

    def test_config_loader_reads_loss_block(self) -> None:
        payload = {
            "method_id": "demo",
            "feature_type": "DEMO",
            "model": {"patch_budget": 24},
            "losses": {"selector_bce": 1.0, "offset_l1": 0.25, "coverage_kl": 0.05},
            "eval": {"frontend_mode": "dino_proposals"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            cfg = load_dino_dpvo_config(path)
        self.assertEqual(cfg.method_id, "demo")
        self.assertEqual(cfg.feature_type, "DEMO")
        self.assertEqual(cfg.losses["offset_l1"], 0.25)
        self.assertEqual(cfg.losses["coverage_kl"], 0.05)

    def test_coverage_regularizer_prefers_spread_selector_mass(self) -> None:
        teacher = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        concentrated = torch.full((1, 1, 4, 4), -4.0, dtype=torch.float32)
        concentrated[0, 0, 0, 0] = 4.0
        spread = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

        concentrated_loss = _compute_coverage_regularizer(
            concentrated,
            teacher,
            grid_rows=2,
            grid_cols=2,
            uniform_mix=0.15,
        )
        spread_loss = _compute_coverage_regularizer(
            spread,
            teacher,
            grid_rows=2,
            grid_cols=2,
            uniform_mix=0.15,
        )

        self.assertGreater(float(concentrated_loss.item()), float(spread_loss.item()))

    def test_load_matching_state_dict_skips_shape_mismatches(self) -> None:
        module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
        state_dict = {
            "0.weight": torch.randn(3, 4),
            "0.bias": torch.randn(3),
            "2.weight": torch.randn(5, 3),
            "2.bias": torch.randn(5),
        }
        info = load_matching_state_dict(module, state_dict)
        self.assertEqual(info["loaded"], 2)
        self.assertGreaterEqual(info["skipped"], 2)

    def test_semantic_fraction_schedule_ramps_and_holds(self) -> None:
        payload = {
            "method_id": "demo",
            "feature_type": "DEMO",
            "training": {
                "semantic_fraction_start": 0.25,
                "semantic_fraction_mid": 0.40,
                "semantic_fraction_end": 0.50,
                "mix_ramp_1_end_step": 5000,
                "mix_ramp_2_end_step": 15000,
                "mix_hold_step": 22000,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            cfg = load_dino_dpvo_config(path)

        self.assertAlmostEqual(_semantic_fraction_target(cfg, 1), 0.25, places=6)
        self.assertAlmostEqual(_semantic_fraction_target(cfg, 5000), 0.25, places=6)
        self.assertGreater(_semantic_fraction_target(cfg, 12000), 0.25)
        self.assertLess(_semantic_fraction_target(cfg, 12000), 0.50)
        self.assertAlmostEqual(_semantic_fraction_target(cfg, 30000), 0.50, places=6)
        self.assertAlmostEqual(_semantic_fraction_target(cfg, 16000, hold_fraction=0.37), 0.37, places=6)

    def test_update_only_dpvo_init_accepts_module_prefix(self) -> None:
        class _FakeModel:
            def __init__(self) -> None:
                self.update = torch.nn.Sequential(
                    torch.nn.Linear(4, 3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(3, 2),
                )

        model = _FakeModel()
        payload = {
            "module.update.0.weight": torch.randn(3, 4),
            "module.update.0.bias": torch.randn(3),
            "module.update.2.weight": torch.randn(2, 3),
            "module.update.2.bias": torch.randn(2),
        }
        info = _load_dpvo_init_weights(model, payload, mode="update_only")
        self.assertEqual(info["loaded"], 4)
        self.assertGreaterEqual(info["missing"], 0)

    def test_selection_score_prefers_associated_or_coverage_aware_metric(self) -> None:
        metrics = {
            "external_mean_ate": 0.25,
            "external_mean_ate_associated": 0.05,
        }
        self.assertAlmostEqual(_selection_score(metrics, "associated_ate"), 0.05, places=6)
        self.assertAlmostEqual(_selection_score(metrics, "coverage_aware_ate"), 0.25, places=6)

    def test_checkpoint_candidate_key_uses_repetition_then_coverage_aware_tiebreakers(self) -> None:
        better_repeat = {
            "external_mean_ate": 0.20,
            "external_mean_ate_associated": 0.05,
            "repeated_patch_fraction": 0.01,
        }
        worse_repeat = {
            "external_mean_ate": 0.19,
            "external_mean_ate_associated": 0.05,
            "repeated_patch_fraction": 0.03,
        }
        key_better = _checkpoint_candidate_key(
            better_repeat,
            "associated_ate",
            tie_breakers=("repeated_patch_fraction", "external_mean_ate"),
        )
        key_worse = _checkpoint_candidate_key(
            worse_repeat,
            "associated_ate",
            tie_breakers=("repeated_patch_fraction", "external_mean_ate"),
        )
        self.assertIsNotNone(key_better)
        self.assertIsNotNone(key_worse)
        self.assertLess(key_better, key_worse)

    def test_read_diagnostics_summary_metrics_averages_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "diagnostics_summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "sequence,feature_type,status,coverage,ate_rmse,ate_rmse_associated,native_patch_fraction,dino_patch_fraction,mean_grid_occupancy,grid_occupancy_std,mean_unique_semantic_count_before_repeat,repeated_patch_fraction,mean_dedupe_radius_used,per_cell_semantic_histogram,mean_track_age,survival_rate_1,survival_rate_3,survival_rate_5",
                        "a,FEATURE,ok,1.0,0.2,0.1,0.0,1.0,1.0,0.0,94.0,0.02,6.0,[],10.0,1.0,0.5,0.2",
                        "b,FEATURE,ok,1.0,0.3,0.2,0.0,1.0,1.0,0.0,96.0,0.04,8.0,[],10.0,1.0,0.5,0.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = _read_diagnostics_summary_metrics(summary_path)

        self.assertAlmostEqual(metrics["mean_unique_semantic_count_before_repeat"], 95.0, places=6)
        self.assertAlmostEqual(metrics["repeated_patch_fraction"], 0.03, places=6)
        self.assertAlmostEqual(metrics["mean_dedupe_radius_used"], 7.0, places=6)

    def test_write_dpvo_style_csv_promotes_associated_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "metrics_summary.csv"
            output_path = Path(tmpdir) / "dpvo_style_metrics_summary.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "sequence,feature_type,status,ate_rmse,ate_mean,ate_median,ate_rmse_associated,ate_mean_associated,ate_median_associated,rpe_trans_rmse,rpe_rot_rmse,coverage",
                        "demo,FEATURE,ok,0.5,0.4,0.3,0.05,0.04,0.03,0.01,1.0,0.99",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_dpvo_style_csv(input_path, output_path)
            with output_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ate_rmse"], "0.05")
        self.assertEqual(rows[0]["ate_mean"], "0.04")
        self.assertEqual(rows[0]["ate_median"], "0.03")

    def test_aggregate_trial_medians_builds_avg_row_and_paper_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            trial_paths = []
            trial_values = [0.040, 0.030, 0.050]
            for idx, value in enumerate(trial_values, start=1):
                csv_path = tmpdir_path / f"trial_{idx:02d}.csv"
                csv_path.write_text(
                    "\n".join(
                        [
                            "sequence,feature_type,status,ate_rmse,ate_mean,ate_median,ate_rmse_associated,ate_mean_associated,ate_median_associated,rpe_trans_rmse,rpe_rot_rmse,coverage",
                            f"freiburg1_desk,FEATURE,ok,{value},{value},{value},{value},{value},{value},0.01,1.0,0.99",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                trial_paths.append(csv_path)

            output_path = tmpdir_path / "tum_freiburg1_paper_table.csv"
            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    "aggregate_trial_medians.py",
                    "--output-csv",
                    str(output_path),
                    "--benchmark",
                    "tum_freiburg1_paper",
                ]
                for trial_path in trial_paths:
                    sys.argv.extend(["--trial-csv", str(trial_path)])
                aggregate_trial_medians_main()
            finally:
                sys.argv = argv_backup

            with output_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["sequence"], "freiburg1_desk")
        self.assertEqual(rows[0]["ate_rmse"], "0.04")
        self.assertEqual(rows[0]["paper_default_ate"], "0.038")
        self.assertEqual(rows[1]["sequence"], "AVG")


if __name__ == "__main__":
    unittest.main()
