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
    _rank_proxy_candidates,
    _select_full_winner,
    _summarize_dataset,
)
from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest  # noqa: E402


def _summary(
    dataset_id: str,
    *,
    ate: float,
    rpe_t: float = 0.1,
    rpe_r: float = 0.2,
    scale_log: float = 0.05,
    pressure: float | None = None,
    failed_count: int = 0,
    finite_count: int = 1,
    row_count: int = 1,
) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=dataset_id,
        row_count=row_count,
        finite_count=finite_count,
        ok_count=row_count - failed_count,
        non_ok_count=failed_count,
        failed_count=failed_count,
        mean_ate_rmse_associated=ate,
        mean_rpe_trans_rmse=rpe_t,
        mean_rpe_rot_rmse=rpe_r,
        mean_scale_correction=1.0,
        mean_scale_error_abs=0.0,
        mean_scale_error_abs_log=scale_log,
        mean_coverage=1.0,
        pressure_mean_ate_rmse_associated=ate if pressure is None else pressure,
        mean_kitti_trans_percent=1.0,
        mean_kitti_rot_deg_per_m=0.01,
    )


class CrossDatasetTriProxySweepTests(unittest.TestCase):
    def test_manifest_has_10_runs_and_expected_proxy_packs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_cross_dataset_tri_proxy_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_cross_dataset_tri_proxy_sweep_v1")
        self.assertEqual(len(runs), 10)
        self.assertEqual(payload["sweep"]["config_overrides"]["eval"]["selection_mode"], "hybrid_only")
        self.assertFalse(bool(payload["sweep"]["config_overrides"]["eval"]["save_best_pure100"]))
        self.assertTrue(bool(payload["sweep"]["config_overrides"]["eval"]["save_best_hybrid"]))

        proxy = payload["proxy_validation"]["datasets"]
        full = payload["full_benchmark"]["datasets"]
        self.assertEqual(len(proxy["tum"]["sequences"]), 10)
        self.assertEqual(len(proxy["tum"]["pressure_sequences"]), 4)
        self.assertEqual(len(proxy["euroc"]["sequences"]), 6)
        self.assertEqual(len(proxy["kitti"]["sequences"]), 5)
        self.assertEqual(len(full["tum"]["sequences"]), 38)
        self.assertEqual(len(full["euroc"]["sequences"]), 11)
        self.assertEqual(len(full["kitti"]["sequences"]), 11)
        self.assertEqual(int(payload["proxy_validation"]["top_k"]), 3)

    def test_summarize_dataset_reads_scale_and_kitti_fields(self) -> None:
        sequences = ("seq_a", "seq_b", "seq_c")
        spec = DatasetEvalSpec(
            dataset_id="kitti",
            module="demo.module",
            dataset_root=Path("/tmp/demo"),
            sequences=sequences,
            max_dt=0.01,
            missing_penalty_m=3.0,
            min_coverage_ok=0.95,
            image_height=240,
            image_width=320,
            stride=4,
            backend_thresh=32.0,
            dpvo_opts="BUFFER_SIZE=2048",
            pressure_sequences=("seq_b",),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics_summary.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sequence",
                        "status",
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
                writer.writerow(["seq_a", "ok", "0.10", "0.01", "0.02", "1.10", "0.10", "0.095310", "1.0", "2.0", "0.003"])
                writer.writerow(["seq_b", "partial_low_coverage", "0.20", "0.03", "0.04", "0.95", "0.05", "0.051293", "0.5", "3.0", "0.004"])
                writer.writerow(["seq_c", "ok", "0.30", "0.05", "0.06", "1.00", "0.00", "0.000000", "1.0", "4.0", "0.005"])

            summary = _summarize_dataset(csv_path, spec)

        self.assertEqual(summary.row_count, 3)
        self.assertEqual(summary.finite_count, 3)
        self.assertEqual(summary.failed_count, 0)
        self.assertAlmostEqual(summary.mean_ate_rmse_associated, 0.20)
        self.assertAlmostEqual(summary.pressure_mean_ate_rmse_associated, 0.20)
        self.assertAlmostEqual(summary.mean_rpe_trans_rmse, 0.03)
        self.assertAlmostEqual(summary.mean_rpe_rot_rmse, 0.04)
        self.assertAlmostEqual(summary.mean_scale_correction, 1.0166666666666666)
        self.assertAlmostEqual(summary.mean_scale_error_abs, 0.05)
        self.assertAlmostEqual(summary.mean_scale_error_abs_log, (0.095310 + 0.051293 + 0.0) / 3.0, places=6)
        self.assertAlmostEqual(summary.mean_coverage, (1.0 + 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(summary.mean_kitti_trans_percent, 3.0)
        self.assertAlmostEqual(summary.mean_kitti_rot_deg_per_m, 0.004)

    def test_rank_proxy_candidates_applies_gates_and_sorting(self) -> None:
        candidate_a = EvalMethodSpec(
            method_id="candidate_a",
            frontend_mode="dino_hybrid",
            frontend_config=None,
            frontend_checkpoint=None,
            kind="candidate",
        )
        candidate_b = EvalMethodSpec(
            method_id="candidate_b",
            frontend_mode="dino_hybrid",
            frontend_config=None,
            frontend_checkpoint=None,
            kind="candidate",
        )
        candidate_c = EvalMethodSpec(
            method_id="candidate_c",
            frontend_mode="dino_hybrid",
            frontend_config=None,
            frontend_checkpoint=None,
            kind="candidate",
        )

        summaries = {
            "current_multiscale_32x4_v1_hybrid75_25": {
                "tum": _summary("tum", ate=1.0, pressure=1.0),
                "euroc": _summary("euroc", ate=1.2),
                "kitti": _summary("kitti", ate=1.1),
            },
            "dpvo_native_matched": {
                "tum": _summary("tum", ate=1.4),
                "euroc": _summary("euroc", ate=1.0),
                "kitti": _summary("kitti", ate=1.0),
            },
            "candidate_a": {
                "tum": _summary("tum", ate=1.01, pressure=1.00, scale_log=0.04),
                "euroc": _summary("euroc", ate=0.80, scale_log=0.03),
                "kitti": _summary("kitti", ate=0.90, scale_log=0.02),
            },
            "candidate_b": {
                "tum": _summary("tum", ate=1.04, pressure=1.02),
                "euroc": _summary("euroc", ate=0.70),
                "kitti": _summary("kitti", ate=0.70),
            },
            "candidate_c": {
                "tum": _summary("tum", ate=1.00, pressure=1.01, scale_log=0.06),
                "euroc": _summary("euroc", ate=1.00, scale_log=0.05),
                "kitti": _summary("kitti", ate=1.00, scale_log=0.05),
            },
        }

        rows = _rank_proxy_candidates(
            candidate_methods=[candidate_a, candidate_b, candidate_c],
            summaries_by_method=summaries,
            reference_ids={
                "tum": "current_multiscale_32x4_v1_hybrid75_25",
                "euroc": "dpvo_native_matched",
                "kitti": "dpvo_native_matched",
            },
            dataset_weights={"tum": 0.50, "euroc": 0.30, "kitti": 0.20},
            tum_no_regression_multiplier=1.03,
        )

        self.assertEqual([row["method_id"] for row in rows], ["candidate_a", "candidate_c", "candidate_b"])
        self.assertEqual(rows[0]["proxy_rank"], 1)
        self.assertEqual(rows[1]["proxy_rank"], 2)
        self.assertEqual(rows[2]["proxy_rank"], "")
        self.assertEqual(int(rows[0]["passes_proxy_gate"]), 1)
        self.assertEqual(int(rows[1]["passes_proxy_gate"]), 1)
        self.assertEqual(int(rows[2]["passes_tum_gate"]), 0)
        self.assertAlmostEqual(float(rows[0]["weighted_ate_score"]), 0.925)
        self.assertAlmostEqual(float(rows[1]["weighted_ate_score"]), 1.0)

    def test_select_full_winner_uses_transfer_score_for_tum_ties(self) -> None:
        rows = [
            {
                "method_id": "baseline",
                "tum_mean_ate_rmse_associated": "0.120000",
                "transfer_ate_score": "1.000000",
            },
            {
                "method_id": "candidate_a",
                "tum_mean_ate_rmse_associated": "0.100000",
                "transfer_ate_score": "0.900000",
            },
            {
                "method_id": "candidate_b",
                "tum_mean_ate_rmse_associated": "0.103000",
                "transfer_ate_score": "0.700000",
            },
        ]

        winner = _select_full_winner(rows, {"candidate_a", "candidate_b"}, tum_tie_abs_threshold=0.005)
        self.assertEqual(winner["method_id"], "candidate_b")


if __name__ == "__main__":
    unittest.main()
