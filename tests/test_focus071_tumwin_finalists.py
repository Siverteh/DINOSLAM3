from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.focus071_tumwin_finalists import (  # noqa: E402
    _read_frozen_dpvo_baseline,
    _read_leaderboard_candidates,
    _screening_summary_row,
    _winner_per_sequence_medians,
)


class Focus071TumwinFinalistsTests(unittest.TestCase):
    def test_read_frozen_dpvo_baseline_filters_dpvo_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "per_sequence_median.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
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
                writer.writerow(["dpvo_native", "freiburg1_room", "freiburg1", "0.30", "0.30", "1.0"])
                writer.writerow(["focus071_best", "freiburg1_room", "freiburg1", "0.20", "0.20", "1.0"])
                writer.writerow(["dpvo_native", "freiburg2_coke", "freiburg2", "0.10", "0.10", "1.0"])
            baseline = _read_frozen_dpvo_baseline(
                csv_path,
                expected_sequences=["freiburg1_room", "freiburg2_coke"],
            )
        self.assertEqual(baseline, {"freiburg1_room": 0.30, "freiburg2_coke": 0.10})

    def test_read_leaderboard_candidates_uses_secondary_assoc_as_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            leaderboard_path = root / "leaderboard_dev.csv"
            candidates_dir = root / "candidates"
            candidates_dir.mkdir(parents=True, exist_ok=True)

            def _touch(name: str) -> tuple[Path, Path]:
                cfg = candidates_dir / f"{name}.yaml"
                ckpt = candidates_dir / f"{name}.pt"
                cfg.write_text("method_id: demo\nfeature_type: DEMO\n", encoding="utf-8")
                ckpt.write_text("x", encoding="utf-8")
                return cfg, ckpt

            cfg_a, ckpt_a = _touch("run_a")
            cfg_b, ckpt_b = _touch("run_b")
            cfg_c, ckpt_c = _touch("run_c")

            with leaderboard_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "status",
                        "expected_eval_mode",
                        "config_path",
                        "subset_config",
                        "init_checkpoint",
                        "best_step",
                        "best_assoc",
                        "best_ate",
                        "best_coverage",
                        "best_lowtex_assoc",
                        "best_lowtex_coverage",
                        "last_step",
                        "last_assoc",
                        "last_ate",
                        "last_coverage",
                        "last_lowtex_assoc",
                        "last_lowtex_coverage",
                        "checkpoint_path",
                    ]
                )
                writer.writerow(
                    ["run_a", "completed", "pure100", str(cfg_a), "-", "-", "500", "0.080000", "0.20", "0.99", "0.110000", "0.99", "", "", "", "", "", "", str(ckpt_a)]
                )
                writer.writerow(
                    ["run_b", "completed", "pure100", str(cfg_b), "-", "-", "500", "0.080000", "0.20", "0.99", "0.090000", "0.99", "", "", "", "", "", "", str(ckpt_b)]
                )
                writer.writerow(
                    ["run_c", "completed", "pure100", str(cfg_c), "-", "-", "500", "0.081000", "0.20", "0.99", "0.050000", "0.99", "", "", "", "", "", "", str(ckpt_c)]
                )

            candidates = _read_leaderboard_candidates(
                leaderboard_path=leaderboard_path,
                top_k=2,
                coverage_gate=0.95,
                secondary_coverage_gate=0.95,
            )
        self.assertEqual([candidate.run_id for candidate in candidates], ["run_b", "run_a"])

    def test_screening_summary_row_counts_wins_and_losses(self) -> None:
        rows = [
            {"sequence": "freiburg1_room", "ate_rmse": "0.20", "ate_rmse_associated": "0.20", "coverage": "1.0"},
            {"sequence": "freiburg2_coke", "ate_rmse": "0.15", "ate_rmse_associated": "0.15", "coverage": "1.0"},
            {"sequence": "freiburg3_teddy", "ate_rmse": "0.05", "ate_rmse_associated": "0.05", "coverage": "1.0"},
        ]
        baseline = {
            "freiburg1_room": 0.30,
            "freiburg2_coke": 0.10,
            "freiburg3_teddy": 0.05,
        }
        row = _screening_summary_row(
            run_id="demo",
            rows=rows,
            baseline_assoc=baseline,
            checkpoint_path=Path("/tmp/demo.pt"),
            config_path=Path("/tmp/demo.yaml"),
            best_assoc=0.08,
            best_secondary_assoc=0.11,
        )
        self.assertEqual(row["wins_vs_dpvo_median"], 1)
        self.assertEqual(row["losses_vs_dpvo_median"], 1)
        self.assertEqual(row["ties_vs_dpvo_median"], 1)
        self.assertEqual(row["freiburg1_wins"], 1)
        self.assertEqual(row["freiburg2_losses"], 1)

    def test_winner_per_sequence_medians_compares_against_baseline(self) -> None:
        repeat_rows = [
            [
                {"sequence": "freiburg1_room", "ate_rmse": "0.20", "ate_rmse_associated": "0.20", "coverage": "1.0"},
                {"sequence": "freiburg2_coke", "ate_rmse": "0.12", "ate_rmse_associated": "0.12", "coverage": "1.0"},
            ],
            [
                {"sequence": "freiburg1_room", "ate_rmse": "0.22", "ate_rmse_associated": "0.22", "coverage": "1.0"},
                {"sequence": "freiburg2_coke", "ate_rmse": "0.08", "ate_rmse_associated": "0.08", "coverage": "1.0"},
            ],
            [
                {"sequence": "freiburg1_room", "ate_rmse": "0.21", "ate_rmse_associated": "0.21", "coverage": "1.0"},
                {"sequence": "freiburg2_coke", "ate_rmse": "0.09", "ate_rmse_associated": "0.09", "coverage": "1.0"},
            ],
        ]
        per_sequence = _winner_per_sequence_medians(
            repeat_rows=repeat_rows,
            expected_sequences=["freiburg1_room", "freiburg2_coke"],
            baseline_assoc={"freiburg1_room": 0.30, "freiburg2_coke": 0.10},
        )
        row_map = {row["sequence"]: row for row in per_sequence}
        self.assertEqual(row_map["freiburg1_room"]["winner_vs_baseline"], "focus071_best")
        self.assertEqual(row_map["freiburg2_coke"]["winner_vs_baseline"], "focus071_best")
        self.assertEqual(row_map["freiburg1_room"]["winner_assoc_median"], "0.210000")


if __name__ == "__main__":
    unittest.main()
