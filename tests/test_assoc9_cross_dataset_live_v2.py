from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.sweeps.run_assoc9_cross_dataset_sweep import (  # noqa: E402
    DatasetEvalSpec,
    DatasetSummary,
    EvalMethodSpec,
    _build_live_proxy_eval_cfg,
    _load_live_candidate_proxy_summaries,
)
from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest, _row_key  # noqa: E402


def _summary(dataset_id: str, ate: float, *, pressure: float | None = None) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=dataset_id,
        row_count=2,
        finite_count=2,
        ok_count=2,
        non_ok_count=0,
        failed_count=0,
        mean_ate_rmse_associated=ate,
        mean_rpe_trans_rmse=0.10,
        mean_rpe_rot_rmse=0.20,
        mean_scale_correction=1.0,
        mean_scale_error_abs=0.0,
        mean_scale_error_abs_log=0.05,
        mean_coverage=1.0,
        pressure_mean_ate_rmse_associated=ate if pressure is None else pressure,
        mean_kitti_trans_percent=1.5,
        mean_kitti_rot_deg_per_m=0.01,
    )


class CrossDatasetLiveTriProxyV2Tests(unittest.TestCase):
    def test_manifest_live_v2_enables_live_proxy_training(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_cross_dataset_tri_proxy_live_v2.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_cross_dataset_tri_proxy_live_v2")
        self.assertEqual(len(runs), 10)
        self.assertTrue(bool(payload["proxy_validation"]["use_live_training_eval"]))
        self.assertEqual(payload["sweep"]["config_overrides"]["eval"]["selection_metric"], "tri_proxy_score")
        self.assertEqual(int(payload["sweep"]["worse_on_both_patience"]), 0)

    def test_row_key_prefers_live_tri_proxy_score_when_gate_passes(self) -> None:
        row = {
            "selection_metric": "tri_proxy_score",
            "selection_passed_gate": "1",
            "live_tri_proxy_score": 0.95,
            "live_weighted_rpe_trans_score": 1.10,
            "live_weighted_rpe_rot_score": 1.20,
            "live_weighted_scale_error_abs_log_score": 1.30,
            "tum_pressure_mean_ate_associated": 0.42,
        }
        blocked = dict(row)
        blocked["selection_passed_gate"] = "0"

        self.assertEqual(
            _row_key(row, 0.95),
            (0.95, 1.10, 1.20, 1.30, 0.42),
        )
        self.assertIsNone(_row_key(blocked, 0.95))

    def test_build_live_proxy_cfg_uses_baseline_references(self) -> None:
        datasets = {
            "tum": DatasetEvalSpec("tum", "m", Path("/tmp/tum"), ("a",), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "A", ("a",)),
            "euroc": DatasetEvalSpec("euroc", "m", Path("/tmp/euroc"), ("b",), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "B"),
            "kitti": DatasetEvalSpec("kitti", "m", Path("/tmp/kitti"), ("c",), 0.01, 3.0, 0.95, 240, 320, 4, 32.0, "C"),
        }
        baselines = {
            "current_multiscale": {"tum": _summary("tum", 0.20, pressure=0.25), "euroc": _summary("euroc", 0.30), "kitti": _summary("kitti", 0.40)},
            "dpvo_native": {"tum": _summary("tum", 0.50), "euroc": _summary("euroc", 0.35), "kitti": _summary("kitti", 0.45)},
        }

        cfg = _build_live_proxy_eval_cfg(
            proxy_datasets=datasets,
            baseline_summaries=baselines,
            reference_ids={"tum": "current_multiscale", "euroc": "dpvo_native", "kitti": "dpvo_native"},
            dataset_weights={"tum": 0.5, "euroc": 0.3, "kitti": 0.2},
            tum_no_regression_multiplier=1.03,
        )

        self.assertTrue(bool(cfg["enabled"]))
        self.assertEqual(cfg["datasets"]["kitti"]["dpvo_opts"], "C")
        self.assertAlmostEqual(cfg["references"]["tum_mean_ate_rmse_associated"], 0.20)
        self.assertAlmostEqual(cfg["references"]["tum_pressure_mean_ate_rmse_associated"], 0.25)
        self.assertAlmostEqual(cfg["references"]["euroc_mean_ate_rmse_associated"], 0.35)

    def test_load_live_candidate_proxy_summaries_reads_live_checkpoint_csvs(self) -> None:
        candidate = EvalMethodSpec(
            method_id="candidate_x",
            frontend_mode="dino_hybrid",
            frontend_config=None,
            frontend_checkpoint=None,
            kind="candidate",
            source_run_id="candidate_x",
        )
        datasets = {
            "tum": DatasetEvalSpec("tum", "m", Path("/tmp/tum"), ("seq_a", "seq_b"), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "A", ("seq_b",)),
            "euroc": DatasetEvalSpec("euroc", "m", Path("/tmp/euroc"), ("MH02", "MH04"), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "B"),
            "kitti": DatasetEvalSpec("kitti", "m", Path("/tmp/kitti"), ("02", "04"), 0.01, 3.0, 0.95, 240, 320, 4, 32.0, "C"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dataset_id, spec in datasets.items():
                csv_path = root / "train" / "candidate_x" / "live_proxy_eval" / "step_000500" / dataset_id / "metrics_summary.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "sequence",
                            "status",
                            "ate_rmse",
                            "ate_rmse_associated",
                            "rpe_trans_rmse",
                            "rpe_rot_rmse",
                            "scale_correction",
                            "scale_error_abs",
                            "scale_error_abs_log",
                            "coverage",
                            "kitti_trans_percent",
                            "kitti_rot_deg_per_m",
                        ]
                    )
                    for idx, seq in enumerate(spec.sequences):
                        writer.writerow(
                            [
                                seq,
                                "ok",
                                f"{0.1 + idx:.6f}",
                                f"{0.2 + idx:.6f}",
                                "0.01",
                                "0.02",
                                "1.0",
                                "0.0",
                                "0.0",
                                "1.0",
                                "1.5",
                                "0.01",
                            ]
                        )

            summaries = _load_live_candidate_proxy_summaries(
                candidate_methods=[candidate],
                leaderboard_rows=[{"run_id": "candidate_x", "best_step": "500"}],
                datasets=datasets,
                base_output_dir=root,
            )

        self.assertIn("candidate_x", summaries)
        self.assertAlmostEqual(summaries["candidate_x"]["tum"].mean_ate_rmse_associated, 0.7)
        self.assertAlmostEqual(summaries["candidate_x"]["tum"].pressure_mean_ate_rmse_associated, 1.2)
        self.assertAlmostEqual(summaries["candidate_x"]["kitti"].mean_kitti_trans_percent, 1.5)


if __name__ == "__main__":
    unittest.main()
