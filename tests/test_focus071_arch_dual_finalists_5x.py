from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.aggregate_arch_dual_finalists_5x import aggregate_dual_finalists_benchmark  # noqa: E402
from refocus_vo.eval.focus071_arch_dual_finalists_5x import (  # noqa: E402
    _ensure_reused_repeat01,
    _load_locked_finalists,
    _validate_repeat_dir,
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


def _write_eval_csv(path: Path, *, sequences: list[str], values: list[float], feature_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for sequence, value in zip(sequences, values):
            writer.writerow(
                [
                    sequence,
                    feature_type,
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


class Focus071ArchDualFinalists5xTests(unittest.TestCase):
    def test_load_locked_finalists_reads_expected_rows(self) -> None:
        sequences = ["freiburg1_room", "freiburg2_coke", "freiburg3_teddy"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ablation_root = root / "ablation"
            screening_summary = ablation_root / "screening_summary.csv"
            screening_dir_a = ablation_root / "screening" / "multiscale_32x4_v1" / "hybrid75_25"
            screening_dir_b = ablation_root / "screening" / "micro4_grid_v1" / "hybrid90_10"
            _write_eval_csv(
                screening_dir_a / "dpvo_style_metrics_summary.csv",
                sequences=sequences,
                values=[0.10, 0.20, 0.30],
                feature_type="A",
            )
            _write_eval_csv(
                screening_dir_b / "dpvo_style_metrics_summary.csv",
                sequences=sequences,
                values=[0.11, 0.21, 0.31],
                feature_type="B",
            )
            ckpt_a = root / "multiscale.pt"
            ckpt_b = root / "micro4.pt"
            cfg_a = root / "multiscale.yaml"
            cfg_b = root / "micro4.yaml"
            for path in (ckpt_a, ckpt_b, cfg_a, cfg_b):
                path.write_text("x", encoding="utf-8")
            with screening_summary.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "mode",
                        "ratio_id",
                        "frontend_mode",
                        "checkpoint_path",
                        "config_path",
                    ]
                )
                writer.writerow(["multiscale_32x4_v1", "hybrid75_25", "hybrid75_25", "dino_hybrid", str(ckpt_a), str(cfg_a)])
                writer.writerow(["micro4_grid_v1", "hybrid90_10", "hybrid90_10", "dino_hybrid", str(ckpt_b), str(cfg_b)])

            finalists = _load_locked_finalists(
                screening_summary_path=screening_summary,
                ablation_root=ablation_root,
            )

        self.assertEqual([item.finalist_id for item in finalists], ["multiscale_32x4_v1_hybrid75_25", "micro4_grid_v1_hybrid90_10"])
        self.assertEqual(finalists[0].run_id, "multiscale_32x4_v1")
        self.assertEqual(finalists[1].ratio_id, "hybrid90_10")

    def test_ensure_reused_repeat01_copies_screening_dir(self) -> None:
        sequences = ["freiburg1_room", "freiburg2_coke", "freiburg3_teddy"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "screening"
            dest_dir = root / "benchmark" / "repeat_01"
            _write_eval_csv(
                source_dir / "dpvo_style_metrics_summary.csv",
                sequences=sequences,
                values=[0.10, 0.20, 0.30],
                feature_type="SRC",
            )
            (source_dir / "command.txt").write_text("demo command\n", encoding="utf-8")

            _ensure_reused_repeat01(
                source_dir=source_dir,
                dest_dir=dest_dir,
                expected_sequences=sequences,
            )

            self.assertTrue((dest_dir / "dpvo_style_metrics_summary.csv").exists())
            self.assertEqual((dest_dir / "command.txt").read_text(encoding="utf-8"), "demo command\n")

    def test_aggregate_dual_finalists_benchmark_writes_expected_outputs(self) -> None:
        sequences = ["freiburg1_room", "freiburg2_coke", "freiburg3_teddy"]
        finalist_ids = ["multiscale_32x4_v1_hybrid75_25", "micro4_grid_v1_hybrid90_10"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_root = root / "benchmark"
            for finalist_id, base_values in {
                finalist_ids[0]: [0.10, 0.12, 0.08],
                finalist_ids[1]: [0.14, 0.18, 0.11],
            }.items():
                for repeat_idx in range(1, 6):
                    repeat_dir = benchmark_root / finalist_id / f"repeat_{repeat_idx:02d}"
                    values = [value + 0.01 * (repeat_idx - 3) for value in base_values]
                    _write_eval_csv(
                        repeat_dir / "dpvo_style_metrics_summary.csv",
                        sequences=sequences,
                        values=values,
                        feature_type=finalist_id.upper(),
                    )

            baseline_per_sequence = root / "per_sequence_median.csv"
            with baseline_per_sequence.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "method",
                        "sequence",
                        "family",
                        "median_ate_rmse",
                        "median_ate_rmse_associated",
                        "median_coverage",
                    ]
                )
                for sequence, dpvo_value, old_value in zip(sequences, [0.20, 0.22, 0.18], [0.16, 0.19, 0.14]):
                    family = sequence.split("_", 1)[0]
                    writer.writerow(["dpvo_native", sequence, family, f"{dpvo_value:.6f}", f"{dpvo_value:.6f}", "1.000000"])
                    writer.writerow(["focus071_best", sequence, family, f"{old_value:.6f}", f"{old_value:.6f}", "1.000000"])

            historical_method_comparison = root / "method_comparison.csv"
            with historical_method_comparison.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "method",
                        "full_mean_of_sequence_medians_ate_rmse_associated",
                    ]
                )
                writer.writerow(["dpvo_native", "0.200000"])
                writer.writerow(["focus071_best", "0.170000"])

            historical_repeat_summary = root / "repeat_summary.csv"
            with historical_repeat_summary.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["method", "repeat_id", "full_mean_ate_rmse_associated"])
                for repeat_idx, value in enumerate([0.20, 0.21, 0.19, 0.20, 0.22], start=1):
                    writer.writerow(["dpvo_native", f"repeat_{repeat_idx:02d}", f"{value:.6f}"])
                for repeat_idx, value in enumerate([0.17, 0.18, 0.16, 0.17, 0.19], start=1):
                    writer.writerow(["focus071_best", f"repeat_{repeat_idx:02d}", f"{value:.6f}"])

            outputs = aggregate_dual_finalists_benchmark(
                benchmark_root=benchmark_root,
                finalist_ids=finalist_ids,
                expected_sequences=sequences,
                baseline_per_sequence_path=baseline_per_sequence,
                historical_method_comparison_path=historical_method_comparison,
                historical_repeat_summary_path=historical_repeat_summary,
                repeats=5,
            )

            self.assertTrue(outputs["repeat_summary"].exists())
            self.assertTrue(outputs["per_sequence_median"].exists())
            self.assertTrue(outputs["method_comparison"].exists())
            self.assertTrue(outputs["method_comparison_md"].exists())

            with outputs["method_comparison"].open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["finalist_id"], finalist_ids[0])
            self.assertAlmostEqual(float(rows[0]["full_mean_of_repeat_means_ate_rmse_associated"]), 0.10)
            self.assertEqual(rows[0]["per_sequence_median_wins_vs_dpvo"], "3")

    def test_validate_repeat_dir_allows_invalid_trajectory_rows(self) -> None:
        sequences = ["freiburg1_room", "freiburg2_coke", "freiburg3_teddy"]
        with tempfile.TemporaryDirectory() as tmpdir:
            repeat_dir = Path(tmpdir) / "repeat_05"
            csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerow(
                    [
                        "freiburg1_room",
                        "TEST",
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
                        "freiburg2_coke",
                        "TEST",
                        "invalid_trajectory",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "0.000000",
                    ]
                )
                writer.writerow(
                    [
                        "freiburg3_teddy",
                        "TEST",
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

            _validate_repeat_dir(repeat_dir, expected_sequences=sequences)

    def test_aggregate_counts_nonfinite_rows_as_losses_vs_baseline(self) -> None:
        sequences = ["freiburg1_room", "freiburg2_coke", "freiburg3_teddy"]
        finalist_ids = ["multiscale_32x4_v1_hybrid75_25", "micro4_grid_v1_hybrid90_10"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_root = root / "benchmark"
            for finalist_id in finalist_ids:
                for repeat_idx in range(1, 6):
                    repeat_dir = benchmark_root / finalist_id / f"repeat_{repeat_idx:02d}"
                    repeat_dir.mkdir(parents=True, exist_ok=True)
                    with (repeat_dir / "dpvo_style_metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(CSV_HEADER)
                        writer.writerow(
                            [
                                "freiburg1_room",
                                finalist_id,
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
                                "freiburg2_coke",
                                finalist_id,
                                "invalid_trajectory",
                                "NaN",
                                "NaN",
                                "NaN",
                                "NaN",
                                "NaN",
                                "NaN",
                                "NaN",
                                "NaN",
                                "0.000000",
                            ]
                        )
                        writer.writerow(
                            [
                                "freiburg3_teddy",
                                finalist_id,
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

            baseline_per_sequence = root / "per_sequence_median.csv"
            with baseline_per_sequence.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "method",
                        "sequence",
                        "family",
                        "median_ate_rmse",
                        "median_ate_rmse_associated",
                        "median_coverage",
                    ]
                )
                for sequence, dpvo_value, old_value in zip(sequences, [0.20, 0.20, 0.20], [0.15, 0.15, 0.15]):
                    family = sequence.split("_", 1)[0]
                    writer.writerow(["dpvo_native", sequence, family, f"{dpvo_value:.6f}", f"{dpvo_value:.6f}", "1.000000"])
                    writer.writerow(["focus071_best", sequence, family, f"{old_value:.6f}", f"{old_value:.6f}", "1.000000"])

            historical_method_comparison = root / "method_comparison.csv"
            with historical_method_comparison.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["method", "full_mean_of_sequence_medians_ate_rmse_associated"])
                writer.writerow(["dpvo_native", "0.200000"])
                writer.writerow(["focus071_best", "0.150000"])

            historical_repeat_summary = root / "repeat_summary.csv"
            with historical_repeat_summary.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["method", "repeat_id", "full_mean_ate_rmse_associated"])
                for repeat_idx in range(1, 6):
                    writer.writerow(["dpvo_native", f"repeat_{repeat_idx:02d}", "0.200000"])
                    writer.writerow(["focus071_best", f"repeat_{repeat_idx:02d}", "0.150000"])

            outputs = aggregate_dual_finalists_benchmark(
                benchmark_root=benchmark_root,
                finalist_ids=finalist_ids,
                expected_sequences=sequences,
                baseline_per_sequence_path=baseline_per_sequence,
                historical_method_comparison_path=historical_method_comparison,
                historical_repeat_summary_path=historical_repeat_summary,
                repeats=5,
            )

            with outputs["repeat_summary"].open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["wins_vs_frozen_dpvo_median"], "2")
            self.assertEqual(rows[0]["losses_vs_frozen_dpvo_median"], "1")
            self.assertEqual(rows[0]["ties_vs_frozen_dpvo_median"], "0")


if __name__ == "__main__":
    unittest.main()
