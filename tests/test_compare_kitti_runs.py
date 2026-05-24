from __future__ import annotations

import csv
from pathlib import Path

from refocus_vo.eval.compare_kitti_runs import compare_kitti_runs


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "feature_type",
                "status",
                "kitti_trans_percent",
                "kitti_rot_deg_per_m",
                "ate_rmse_associated",
                "coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_compare_kitti_runs_writes_headline_and_per_sequence_outputs(tmp_path: Path) -> None:
    winner_csv = tmp_path / "winner.csv"
    baseline_csv = tmp_path / "baseline.csv"
    output_dir = tmp_path / "summary"

    _write_csv(
        winner_csv,
        [
            {
                "sequence": "00",
                "feature_type": "DINO",
                "status": "ok",
                "kitti_trans_percent": "1.0",
                "kitti_rot_deg_per_m": "0.010",
                "ate_rmse_associated": "2.0",
                "coverage": "1.0",
            },
            {
                "sequence": "01",
                "feature_type": "DINO",
                "status": "ok",
                "kitti_trans_percent": "2.0",
                "kitti_rot_deg_per_m": "0.020",
                "ate_rmse_associated": "3.0",
                "coverage": "1.0",
            },
        ],
    )
    _write_csv(
        baseline_csv,
        [
            {
                "sequence": "00",
                "feature_type": "DPVO",
                "status": "ok",
                "kitti_trans_percent": "1.5",
                "kitti_rot_deg_per_m": "0.030",
                "ate_rmse_associated": "4.0",
                "coverage": "1.0",
            },
            {
                "sequence": "01",
                "feature_type": "DPVO",
                "status": "ok",
                "kitti_trans_percent": "1.0",
                "kitti_rot_deg_per_m": "0.010",
                "ate_rmse_associated": "1.0",
                "coverage": "1.0",
            },
        ],
    )

    outputs = compare_kitti_runs(
        winner_csv=winner_csv,
        baseline_csv=baseline_csv,
        output_dir=output_dir,
        winner_id="winner",
        baseline_id="baseline",
    )

    assert outputs["headline_csv"].exists()
    assert outputs["headline_md"].exists()
    assert outputs["per_sequence"].exists()
    headline_text = outputs["headline_md"].read_text(encoding="utf-8")
    assert "Translation wins" in headline_text
    assert "ATE wins" in headline_text

