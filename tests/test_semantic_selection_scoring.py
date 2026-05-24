from __future__ import annotations

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.training.trainer import _score_semantic_eval, _selection_candidate_from_semantic_result


class SemanticSelectionScoringTests(unittest.TestCase):
    def test_weighted_penalized_score(self) -> None:
        rows = [
            {"sequence": "freiburg1_plant", "status": "ok", "ate_rmse": 0.8, "coverage": 0.97},
            {"sequence": "freiburg1_desk", "status": "partial_failed", "ate_rmse": math.nan, "coverage": math.nan},
            {"sequence": "freiburg1_room", "status": "ok", "ate_rmse": 1.2, "coverage": 0.96},
            # long_office intentionally missing
        ]
        sequences = [
            "freiburg1_plant",
            "freiburg1_desk",
            "freiburg1_room",
            "freiburg3_long_office_household",
        ]
        weights = {
            "freiburg1_plant": 0.40,
            "freiburg1_desk": 0.30,
            "freiburg1_room": 0.20,
            "freiburg3_long_office_household": 0.10,
        }
        status_pen = {
            "partial_failed": 1.0,
            "tracking_failed": 1.5,
        }
        scored = _score_semantic_eval(
            rows=rows,
            sequences=sequences,
            sequence_weights_cfg=weights,
            missing_penalty=3.0,
            status_penalties_cfg=status_pen,
            min_coverage_ok=0.95,
            primary_ate_field="ate_rmse",
            robust_ate_field="ate_rmse",
        )

        # weighted_mean_ok = (0.4*0.8 + 0.2*1.2)/(0.6) = 0.933333...
        self.assertAlmostEqual(float(scored["weighted_mean_ok"]), 0.9333333333, places=6)
        # missing weight = desk(0.3) + loh(0.1) => 0.4
        self.assertAlmostEqual(float(scored["weighted_missing_ratio"]), 0.4, places=6)
        # status penalty only desk partial_failed (1.0) weighted by 0.3 / 1.0 sum = 0.3
        self.assertAlmostEqual(float(scored["weighted_status_penalty"]), 0.3, places=6)
        # total = 0.933333 + 3*0.4 + 0.3 = 2.433333...
        self.assertAlmostEqual(float(scored["weighted_penalized_score"]), 2.4333333333, places=6)
        self.assertTrue(bool(scored["coverage_ok"]))
        self.assertEqual(int(scored["statuses"].get("missing", 0)), 1)

    def test_primary_field_prefers_associated_when_configured(self) -> None:
        rows = [
            {
                "sequence": "freiburg1_desk",
                "status": "ok",
                "ate_rmse": 0.90,
                "ate_rmse_associated": 0.04,
                "coverage": 0.99,
            }
        ]
        scored = _score_semantic_eval(
            rows=rows,
            sequences=["freiburg1_desk"],
            sequence_weights_cfg={"freiburg1_desk": 1.0},
            missing_penalty=3.0,
            status_penalties_cfg={},
            min_coverage_ok=0.95,
            primary_ate_field="ate_rmse_associated",
            robust_ate_field="ate_rmse",
        )
        self.assertAlmostEqual(float(scored["weighted_penalized_score"]), 0.04, places=6)
        self.assertAlmostEqual(float(scored["weighted_penalized_score_robust"]), 0.90, places=6)

    def test_selection_candidate_prefers_holdout_when_requested(self) -> None:
        ate_result = {
            "weighted_penalized_score": 0.50,
            "selected_profile": "holdout",
            "profiles": {
                "overfit": {
                    "weighted_penalized_score": 0.03,
                    "weighted_mean_ok": 0.03,
                    "penalized_mean": 0.03,
                    "mean_ok": 0.03,
                },
                "holdout": {
                    "weighted_penalized_score": 0.08,
                    "weighted_mean_ok": 0.08,
                    "penalized_mean": 0.08,
                    "mean_ok": 0.08,
                },
            },
        }
        cand, used = _selection_candidate_from_semantic_result("holdout_weighted_penalized_ate", ate_result)
        self.assertEqual(used, "holdout_weighted_penalized_ate")
        self.assertAlmostEqual(float(cand), 0.08, places=7)

        cand_sel, _used_sel = _selection_candidate_from_semantic_result("weighted_penalized_ate", ate_result)
        self.assertAlmostEqual(float(cand_sel), 0.08, places=7)

    def test_selection_candidate_supports_overfit_weighted_mean_ate(self) -> None:
        ate_result = {
            "weighted_penalized_score": 0.50,
            "selected_profile": "overfit",
            "profiles": {
                "overfit": {
                    "weighted_penalized_score": 0.06,
                    "weighted_mean_ok": 0.05,
                },
                "holdout": {
                    "weighted_penalized_score": 0.09,
                    "weighted_mean_ok": 0.08,
                },
            },
        }
        cand, used = _selection_candidate_from_semantic_result("overfit_weighted_mean_ate", ate_result)
        self.assertEqual(used, "overfit_weighted_mean_ate")
        self.assertAlmostEqual(float(cand), 0.05, places=7)


if __name__ == "__main__":
    unittest.main()
