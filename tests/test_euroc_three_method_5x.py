from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.aggregate_euroc_three_method_5x import aggregate_euroc_three_method_benchmark  # noqa: E402
from refocus_vo.eval.euroc_three_method_5x import (  # noqa: E402
    _ensure_reused_repeat01,
    _parse_method_ids,
    _selected_methods,
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
SEQUENCES = ["MH01", "MH02", "MH03", "MH04", "MH05", "V101", "V102", "V103", "V201", "V202", "V203"]


def _write_eval_csv(
    path: Path,
    *,
    sequences: list[str],
    values: list[float],
    feature_type: str,
    non_ok_sequences: set[str] | None = None,
) -> None:
    non_ok_sequences = non_ok_sequences or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for sequence, value in zip(sequences, values):
            status = "partial_low_coverage" if sequence in non_ok_sequences else "ok"
            writer.writerow(
                [
                    sequence,
                    feature_type,
                    status,
                    f"{value:.6f}",
                    f"{value:.6f}",
                    f"{value:.6f}",
                    f"{value:.6f}",
                    f"{value:.6f}",
                    f"{value:.6f}",
                    "0.010000",
                    "0.020000",
                    "1.000000" if status == "ok" else "0.500000",
                ]
            )


class EurocThreeMethod5xTests(unittest.TestCase):
    def test_parse_method_ids_and_selected_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = {
                "dpvo_repeat01": root / "dpvo_repeat01",
                "multiscale_repeat01": root / "multiscale_repeat01",
                "multiscale_checkpoint": root / "multiscale.pt",
                "multiscale_config": root / "multiscale.yaml",
                "micro_repeat01": root / "micro_repeat01",
                "micro_checkpoint": root / "micro.pt",
                "micro_config": root / "micro.yaml",
            }
            for path in defaults.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix:
                    path.write_text("x", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)

            method_ids = _parse_method_ids("dpvo_native_matched,multiscale_32x4_v1_hybrid75_25")
            methods = _selected_methods(defaults=defaults, method_ids=method_ids)

        self.assertEqual([item.method_id for item in methods], method_ids)
        self.assertEqual(methods[0].frontend_mode, "dpvo_native")
        self.assertEqual(methods[1].frontend_mode, "dino_hybrid")

    def test_ensure_reused_repeat01_copies_source_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "benchmark" / "repeat_01"
            _write_eval_csv(
                source_dir / "dpvo_style_metrics_summary.csv",
                sequences=SEQUENCES,
                values=[0.10 + i * 0.01 for i in range(len(SEQUENCES))],
                feature_type="SRC",
            )
            (source_dir / "command.txt").write_text("demo\n", encoding="utf-8")

            _ensure_reused_repeat01(source_dir=source_dir, dest_dir=dest_dir, expected_sequences=SEQUENCES)

            self.assertTrue((dest_dir / "dpvo_style_metrics_summary.csv").exists())
            self.assertEqual((dest_dir / "command.txt").read_text(encoding="utf-8"), "demo\n")

    def test_validate_repeat_dir_allows_partial_low_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repeat_dir = Path(tmpdir) / "repeat_01"
            _write_eval_csv(
                repeat_dir / "dpvo_style_metrics_summary.csv",
                sequences=SEQUENCES,
                values=[0.10 + i * 0.01 for i in range(len(SEQUENCES))],
                feature_type="TEST",
                non_ok_sequences={"V203"},
            )
            _validate_repeat_dir(repeat_dir, expected_sequences=SEQUENCES)

    def test_aggregate_benchmark_writes_expected_outputs(self) -> None:
        method_ids = ["dpvo_native_matched", "multiscale_32x4_v1_hybrid75_25", "micro4_grid_v1_hybrid90_10"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_root = root / "benchmark"
            base_values = {
                "dpvo_native_matched": [0.30] * len(SEQUENCES),
                "multiscale_32x4_v1_hybrid75_25": [0.20] * len(SEQUENCES),
                "micro4_grid_v1_hybrid90_10": [0.24] * len(SEQUENCES),
            }
            for method_id in method_ids:
                for repeat_idx in range(1, 6):
                    repeat_dir = benchmark_root / method_id / f"repeat_{repeat_idx:02d}"
                    values = [value + 0.01 * (repeat_idx - 3) for value in base_values[method_id]]
                    _write_eval_csv(
                        repeat_dir / "dpvo_style_metrics_summary.csv",
                        sequences=SEQUENCES,
                        values=values,
                        feature_type=method_id,
                    )

            outputs = aggregate_euroc_three_method_benchmark(
                benchmark_root=benchmark_root,
                method_ids=method_ids,
                expected_sequences=SEQUENCES,
                repeats=5,
            )

            self.assertTrue(outputs["repeat_summary"].exists())
            self.assertTrue(outputs["per_sequence_median"].exists())
            self.assertTrue(outputs["method_comparison"].exists())
            self.assertTrue(outputs["method_comparison_md"].exists())

            with outputs["method_comparison"].open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["method_id"], "multiscale_32x4_v1_hybrid75_25")
            self.assertAlmostEqual(float(rows[0]["full_mean_of_repeat_means_ate_rmse_associated"]), 0.20)
            self.assertEqual(rows[0]["per_sequence_median_wins_vs_dpvo_native_matched"], "11")
            self.assertEqual(rows[1]["method_id"], "micro4_grid_v1_hybrid90_10")


if __name__ == "__main__":
    unittest.main()
