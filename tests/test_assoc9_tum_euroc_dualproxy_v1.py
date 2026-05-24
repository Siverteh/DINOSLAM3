from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.sweeps.run_assoc9_cross_dataset_sweep import (  # noqa: E402
    DatasetEvalSpec,
    DatasetSummary,
    EvalMethodSpec,
    _build_live_proxy_eval_cfg,
)
from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest, _row_key  # noqa: E402
from refocus_vo.sweeps.run_assoc9_tum_euroc_dualproxy_sweep import (  # noqa: E402
    _frozen_sequence_baseline_rows,
    _rank_dual_proxy_candidates,
)


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


class TumEurocDualProxyV1Tests(unittest.TestCase):
    def test_manifest_uses_dual_proxy_without_kitti(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_tum_euroc_live_dualproxy_v1.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_tum_euroc_live_dualproxy_v1")
        self.assertEqual(len(runs), 10)
        self.assertNotIn("kitti", payload["proxy_validation"]["datasets"])
        self.assertEqual(payload["sweep"]["config_overrides"]["eval"]["selection_metric"], "dual_proxy_score")
        self.assertEqual(int(payload["sweep"]["min_failure_step"]), 1500)

    def test_build_live_proxy_cfg_omits_kitti_references(self) -> None:
        datasets = {
            "tum": DatasetEvalSpec("tum", "m", Path("/tmp/tum"), ("a",), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "A", ("a",)),
            "euroc": DatasetEvalSpec("euroc", "m", Path("/tmp/euroc"), ("MH01",), 0.02, 3.0, 0.95, 240, 320, 4, 32.0, "B"),
        }
        baselines = {
            "current_multiscale": {
                "tum": _summary("tum", 0.20, pressure=0.25),
                "euroc": _summary("euroc", 0.30),
            },
            "dpvo_native_matched": {
                "tum": _summary("tum", 0.40, pressure=0.45),
                "euroc": _summary("euroc", 0.35),
            },
        }

        cfg = _build_live_proxy_eval_cfg(
            proxy_datasets=datasets,
            baseline_summaries=baselines,
            reference_ids={"tum": "current_multiscale", "euroc": "dpvo_native_matched"},
            dataset_weights={"tum": 0.65, "euroc": 0.35},
            win_weights={"tum": 0.15, "euroc": 0.05},
            tum_no_regression_multiplier=1.03,
            sequence_assoc_baselines={"tum": {"a": 0.3}, "euroc": {"MH01": 0.4}},
            gate_overrides={"tum_proxy_min_wins_vs_dpvo": 8, "required_valid_datasets": ["euroc"]},
        )

        self.assertTrue(bool(cfg["enabled"]))
        self.assertNotIn("kitti_mean_ate_rmse_associated", cfg["references"])
        self.assertEqual(cfg["references"]["sequence_assoc_baselines"]["tum"]["a"], 0.3)
        self.assertEqual(cfg["gate"]["tum_proxy_min_wins_vs_dpvo"], 8)

    def test_row_key_prefers_dual_proxy_score(self) -> None:
        row = {
            "selection_metric": "dual_proxy_score",
            "selection_passed_gate": "1",
            "live_dual_proxy_score": 0.91,
            "live_weighted_rpe_trans_score": 1.05,
            "live_weighted_rpe_rot_score": 1.10,
            "live_weighted_scale_error_abs_log_score": 1.15,
            "tum_pressure_mean_ate_associated": 0.42,
            "euroc_proxy_wins_vs_dpvo": 9,
        }
        blocked = dict(row)
        blocked["selection_passed_gate"] = "0"

        self.assertEqual(
            _row_key(row, 0.95),
            (0.91, 0.42, -9.0, 1.05, 1.10, 1.15),
        )
        self.assertIsNone(_row_key(blocked, 0.95))

    def test_rank_candidates_applies_tum_and_euroc_gates(self) -> None:
        candidate_methods = [
            EvalMethodSpec("candidate_good", "dino_hybrid", None, None, "candidate"),
            EvalMethodSpec("candidate_bad", "dino_hybrid", None, None, "candidate"),
        ]
        summaries = {
            "current_multiscale": {
                "tum": _summary("tum", 0.35, pressure=0.70),
                "euroc": _summary("euroc", 0.36),
            },
            "dpvo_native_matched": {
                "tum": _summary("tum", 0.31, pressure=0.60),
                "euroc": _summary("euroc", 0.39),
            },
            "candidate_good": {
                "tum": _summary("tum", 0.33, pressure=0.69),
                "euroc": _summary("euroc", 0.30),
            },
            "candidate_bad": {
                "tum": _summary("tum", 0.34, pressure=0.73),
                "euroc": _summary("euroc", 0.42),
            },
        }
        leaderboard_rows = [
            {"run_id": "candidate_good", "best_tum_proxy_wins_vs_dpvo": "9", "best_euroc_proxy_wins_vs_dpvo": "7"},
            {"run_id": "candidate_bad", "best_tum_proxy_wins_vs_dpvo": "7", "best_euroc_proxy_wins_vs_dpvo": "4"},
        ]

        ranked = _rank_dual_proxy_candidates(
            candidate_methods=candidate_methods,
            summaries_by_method=summaries,
            leaderboard_rows=leaderboard_rows,
            reference_ids={"tum": "current_multiscale", "euroc": "dpvo_native_matched"},
            dataset_weights={"tum": 0.65, "euroc": 0.35},
            win_weights={"tum": 0.15, "euroc": 0.05},
            tum_no_regression_multiplier=1.03,
            tum_pressure_multiplier=1.05,
            tum_min_wins_vs_dpvo=8,
            euroc_min_wins_vs_dpvo=6,
        )

        self.assertEqual(ranked[0]["method_id"], "candidate_good")
        self.assertEqual(int(ranked[0]["passes_advancement_gate"]), 1)
        self.assertEqual(int(ranked[1]["passes_advancement_gate"]), 0)

    def test_rank_candidates_requires_euroc_wins_for_advancement(self) -> None:
        candidate_methods = [EvalMethodSpec("candidate_almost", "dino_hybrid", None, None, "candidate")]
        summaries = {
            "current_multiscale": {
                "tum": _summary("tum", 0.35, pressure=0.70),
                "euroc": _summary("euroc", 0.36),
            },
            "dpvo_native_matched": {
                "tum": _summary("tum", 0.31, pressure=0.60),
                "euroc": _summary("euroc", 0.39),
            },
            "candidate_almost": {
                "tum": _summary("tum", 0.32, pressure=0.68),
                "euroc": _summary("euroc", 0.30),
            },
        }
        leaderboard_rows = [
            {"run_id": "candidate_almost", "best_tum_proxy_wins_vs_dpvo": "9", "best_euroc_proxy_wins_vs_dpvo": "5"}
        ]

        ranked = _rank_dual_proxy_candidates(
            candidate_methods=candidate_methods,
            summaries_by_method=summaries,
            leaderboard_rows=leaderboard_rows,
            reference_ids={"tum": "current_multiscale", "euroc": "dpvo_native_matched"},
            dataset_weights={"tum": 0.55, "euroc": 0.25},
            win_weights={"tum": 0.15, "euroc": 0.05},
            tum_no_regression_multiplier=1.03,
            tum_pressure_multiplier=1.05,
            tum_min_wins_vs_dpvo=8,
            euroc_min_wins_vs_dpvo=6,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(int(ranked[0]["passes_tum_gate"]), 1)
        self.assertEqual(int(ranked[0]["beats_dpvo_on_euroc"]), 1)
        self.assertEqual(int(ranked[0]["passes_advancement_gate"]), 0)

    def test_frozen_sequence_baseline_rows_are_flattened(self) -> None:
        rows = _frozen_sequence_baseline_rows(
            {
                "tum": {"seq_b": 0.2, "seq_a": 0.1},
                "euroc": {"MH01": 0.3},
            }
        )
        self.assertEqual(
            rows,
            [
                {"dataset_id": "euroc", "sequence": "MH01", "baseline_ate_rmse_associated": "0.300000"},
                {"dataset_id": "tum", "sequence": "seq_a", "baseline_ate_rmse_associated": "0.100000"},
                {"dataset_id": "tum", "sequence": "seq_b", "baseline_ate_rmse_associated": "0.200000"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
