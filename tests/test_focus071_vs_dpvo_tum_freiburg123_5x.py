from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.aggregate_tum_freiburg123_repeats import aggregate_benchmark  # noqa: E402
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import (  # noqa: E402
    _enumerate_freiburg_sequences,
    _split_dpvo_opts,
)


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


class Focus071VsDpvoTumFreiburg123Tests(unittest.TestCase):
    def test_enumerate_freiburg_sequences_filters_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in [
                "rgbd_dataset_freiburg2_desk",
                "rgbd_dataset_freiburg1_room",
                "rgbd_dataset_freiburg3_teddy",
                "rgbd_dataset_not_used",
                "other_dir",
            ]:
                (root / name).mkdir(parents=True, exist_ok=True)
            sequences = _enumerate_freiburg_sequences(root)
            self.assertEqual(
                sequences,
                ["freiburg1_room", "freiburg2_desk", "freiburg3_teddy"],
            )

    def test_split_dpvo_opts_converts_key_value_pairs(self) -> None:
        self.assertEqual(
            _split_dpvo_opts("BUFFER_SIZE=384 PATCHES_PER_FRAME=24 REMOVAL_WINDOW=12"),
            ["BUFFER_SIZE", "384", "PATCHES_PER_FRAME", "24", "REMOVAL_WINDOW", "12"],
        )

    def test_aggregate_benchmark_writes_expected_summaries(self) -> None:
        sequences = ["freiburg1_desk", "freiburg2_desk", "freiburg3_teddy"]
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = Path(tmpdir)
            for method, base_values in {
                "dpvo_native": [0.10, 0.20, 0.30],
                "focus071_best": [0.09, 0.25, 0.28],
            }.items():
                for repeat_idx in range(1, 6):
                    repeat_dir = benchmark_root / method / f"repeat_{repeat_idx:02d}"
                    repeat_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(CSV_HEADER)
                        for seq, base in zip(sequences, base_values):
                            value = base + (0.01 * (repeat_idx - 3))
                            writer.writerow(
                                [
                                    seq,
                                    method.upper(),
                                    "ok",
                                    f"{value:.6f}",
                                    f"{value:.6f}",
                                    f"{value:.6f}",
                                    f"{value:.6f}",
                                    f"{value:.6f}",
                                    f"{value:.6f}",
                                    "0.010000",
                                    "0.020000",
                                    "1.000000",
                                ]
                            )

            aggregate_benchmark(benchmark_root=benchmark_root, expected_sequences=sequences)

            summary_dir = benchmark_root / "summary"
            self.assertTrue((summary_dir / "repeat_summary.csv").exists())
            self.assertTrue((summary_dir / "per_sequence_median.csv").exists())
            self.assertTrue((summary_dir / "method_comparison.csv").exists())
            self.assertTrue((summary_dir / "method_comparison.md").exists())

            with (summary_dir / "per_sequence_median.csv").open("r", encoding="utf-8", newline="") as f:
                per_sequence_rows = list(csv.DictReader(f))
            row_map = {
                (row["method"], row["sequence"]): row for row in per_sequence_rows
            }
            self.assertAlmostEqual(
                float(row_map[("dpvo_native", "freiburg1_desk")]["median_ate_rmse_associated"]),
                0.10,
            )
            self.assertAlmostEqual(
                float(row_map[("focus071_best", "freiburg3_teddy")]["median_ate_rmse_associated"]),
                0.28,
            )

            with (summary_dir / "method_comparison.csv").open("r", encoding="utf-8", newline="") as f:
                comparison_rows = list(csv.DictReader(f))
            comp_map = {row["method"]: row for row in comparison_rows}
            self.assertAlmostEqual(
                float(comp_map["dpvo_native"]["full_mean_of_sequence_medians_ate_rmse_associated"]),
                (0.10 + 0.20 + 0.30) / 3.0,
            )
            self.assertAlmostEqual(
                float(comp_map["focus071_best"]["full_mean_of_sequence_medians_ate_rmse_associated"]),
                (0.09 + 0.25 + 0.28) / 3.0,
            )

    def test_aggregate_benchmark_allows_partial_low_coverage_rows(self) -> None:
        sequences = ["freiburg1_desk", "freiburg2_desk", "freiburg3_teddy"]
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = Path(tmpdir)
            for method in ("dpvo_native", "focus071_best"):
                for repeat_idx in range(1, 6):
                    repeat_dir = benchmark_root / method / f"repeat_{repeat_idx:02d}"
                    repeat_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(CSV_HEADER)
                        writer.writerow(
                            [
                                "freiburg1_desk",
                                method.upper(),
                                "ok",
                                "0.100000",
                                "0.100000",
                                "0.100000",
                                "0.100000",
                                "0.100000",
                                "0.100000",
                                "0.010000",
                                "0.020000",
                                "1.000000",
                            ]
                        )
                        writer.writerow(
                            [
                                "freiburg2_desk",
                                method.upper(),
                                "partial_low_coverage",
                                "0.200000",
                                "0.200000",
                                "0.200000",
                                "0.200000",
                                "0.200000",
                                "0.200000",
                                "0.010000",
                                "0.020000",
                                "0.750000",
                            ]
                        )
                        writer.writerow(
                            [
                                "freiburg3_teddy",
                                method.upper(),
                                "ok",
                                "0.300000",
                                "0.300000",
                                "0.300000",
                                "0.300000",
                                "0.300000",
                                "0.300000",
                                "0.010000",
                                "0.020000",
                                "1.000000",
                            ]
                        )

            aggregate_benchmark(benchmark_root=benchmark_root, expected_sequences=sequences)
            self.assertTrue((benchmark_root / "summary" / "method_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
