from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.focus071_arch_ratio_ablation import (  # noqa: E402
    _best_ratio_rows,
    _load_ranked_candidates,
    _materialize_ratio_config,
    _parse_ratio_specs,
)


class Focus071ArchRatioAblationTests(unittest.TestCase):
    def test_parse_ratio_specs_supports_hybrid_and_pure(self) -> None:
        specs = _parse_ratio_specs("90/10,75/25,50/50,pure100")
        self.assertEqual([item.ratio_id for item in specs], ["hybrid90_10", "hybrid75_25", "hybrid50_50", "pure100"])
        self.assertEqual(specs[0].frontend_mode, "dino_hybrid")
        self.assertEqual(specs[-1].frontend_mode, "dino_proposals")

    def test_load_ranked_candidates_adds_lowtex_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leaderboard = root / "leaderboard_dev.csv"

            def _touch(run_id: str) -> tuple[Path, Path]:
                cfg = root / f"{run_id}.yaml"
                ckpt = root / f"{run_id}.pt"
                cfg.write_text("method_id: demo\nfeature_type: DEMO\n", encoding="utf-8")
                ckpt.write_text("x", encoding="utf-8")
                return cfg, ckpt

            cfg_a, ckpt_a = _touch("overall_a")
            cfg_b, ckpt_b = _touch("overall_b")
            cfg_c, ckpt_c = _touch("lowtex_c")

            with leaderboard.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "status",
                        "config_path",
                        "best_assoc",
                        "best_ate",
                        "best_coverage",
                        "best_lowtex_assoc",
                        "best_lowtex_coverage",
                        "best_mode",
                        "checkpoint_path",
                    ]
                )
                writer.writerow(["overall_a", "early_stopped", str(cfg_a), "0.20", "0.30", "0.99", "0.70", "0.99", "hybrid", str(ckpt_a)])
                writer.writerow(["overall_b", "early_stopped", str(cfg_b), "0.21", "0.31", "0.99", "0.60", "0.99", "hybrid", str(ckpt_b)])
                writer.writerow(["lowtex_c", "early_stopped", str(cfg_c), "0.30", "0.40", "0.99", "0.10", "0.99", "hybrid", str(ckpt_c)])

            candidates = _load_ranked_candidates(
                leaderboard_path=leaderboard,
                coverage_gate=0.95,
                secondary_coverage_gate=0.95,
                overall_top_k=2,
                include_lowtex_specialist=True,
            )
        self.assertEqual([item.run_id for item in candidates], ["overall_a", "overall_b", "lowtex_c"])

    def test_materialize_ratio_config_overrides_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.yaml"
            output = root / "ratio.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "method_id": "demo",
                        "feature_type": "DEMO",
                        "model": {"native_fraction": 0.9, "dino_fraction": 0.1},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            path = _materialize_ratio_config(
                source_config=source,
                output_config=output,
                run_label="ratio_test",
                native_fraction=0.75,
                dino_fraction=0.25,
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"]["native_fraction"], 0.75)
        self.assertEqual(payload["model"]["dino_fraction"], 0.25)
        self.assertIn("ratio_test", payload["method_id"])

    def test_best_ratio_rows_prefers_more_wins_then_lower_assoc(self) -> None:
        rows = [
            {
                "run_id": "demo",
                "ratio_id": "hybrid90_10",
                "wins_vs_dpvo_median": 24,
                "full_mean_ate_rmse_associated": "0.20",
                "freiburg2_wins": 8,
                "freiburg3_wins": 7,
                "dino_fraction": "0.10",
            },
            {
                "run_id": "demo",
                "ratio_id": "hybrid50_50",
                "wins_vs_dpvo_median": 26,
                "full_mean_ate_rmse_associated": "0.22",
                "freiburg2_wins": 9,
                "freiburg3_wins": 7,
                "dino_fraction": "0.50",
            },
        ]
        best = _best_ratio_rows(rows)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["ratio_id"], "hybrid50_50")


if __name__ == "__main__":
    unittest.main()
