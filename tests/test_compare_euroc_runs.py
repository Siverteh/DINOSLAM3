from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.compare_euroc_runs import compare_euroc_runs  # noqa: E402


CSV_HEADER = [
    "sequence",
    "feature_type",
    "status",
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "ate_rmse_associated",
    "ate_mean_associated",
    "ate_median_associated",
    "rpe_trans_rmse",
    "rpe_rot_rmse",
    "coverage",
]


def _write_eval_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for sequence, status, value in rows:
            metric = f"{value:.6f}" if math.isfinite(value) else "NaN"
            coverage = "1.000000" if status == "ok" else "0.000000"
            writer.writerow(
                [
                    sequence,
                    "TEST",
                    status,
                    metric,
                    metric,
                    metric,
                    metric,
                    metric,
                    metric,
                    "0.010000",
                    "0.020000",
                    coverage,
                ]
            )


class CompareEuRoCRunsTests(unittest.TestCase):
    def test_compare_euroc_runs_writes_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            winner_csv = root / "winner.csv"
            baseline_csv = root / "baseline.csv"
            output_dir = root / "summary"

            _write_eval_csv(
                winner_csv,
                [
                    ("MH01", "ok", 0.10),
                    ("MH02", "ok", 0.12),
                    ("V101", "ok", 0.08),
                ],
            )
            _write_eval_csv(
                baseline_csv,
                [
                    ("MH01", "ok", 0.20),
                    ("MH02", "invalid_trajectory", math.nan),
                    ("V101", "ok", 0.07),
                ],
            )

            outputs = compare_euroc_runs(
                winner_csv=winner_csv,
                baseline_csv=baseline_csv,
                output_dir=output_dir,
                winner_id="winner_model",
                baseline_id="baseline_model",
            )

            self.assertTrue(outputs["per_sequence"].exists())
            self.assertTrue(outputs["headline_csv"].exists())
            self.assertTrue(outputs["headline_md"].exists())

            with outputs["headline_csv"].open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["method_id"], "winner_model")
            self.assertEqual(rows[1]["method_id"], "baseline_model")
            self.assertAlmostEqual(float(rows[0]["mean_ate_rmse_associated"]), 0.10)
            self.assertEqual(rows[0]["wins_vs_other"], "2")
            self.assertEqual(rows[0]["losses_vs_other"], "1")
            self.assertEqual(rows[0]["non_ok_sequence_count"], "0")
            self.assertEqual(rows[1]["non_ok_sequence_count"], "1")

            with outputs["per_sequence"].open("r", encoding="utf-8", newline="") as f:
                per_sequence = list(csv.DictReader(f))
            mh02 = [row for row in per_sequence if row["sequence"] == "MH02"][0]
            self.assertEqual(mh02["better_method"], "winner")


if __name__ == "__main__":
    unittest.main()
