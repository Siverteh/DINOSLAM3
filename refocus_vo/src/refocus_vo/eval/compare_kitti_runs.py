from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.kitti_odometry_metrics import KITTI_SEQUENCE_ORDER


@dataclass(frozen=True)
class SequenceRow:
    sequence: str
    status: str
    kitti_trans_percent: float
    kitti_rot_deg_per_m: float
    ate_rmse_associated: float
    coverage: float


def _parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _load_rows(csv_path: Path) -> dict[str, SequenceRow]:
    rows: dict[str, SequenceRow] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            sequence = str(raw.get("sequence", "")).strip()
            if not sequence:
                continue
            rows[sequence] = SequenceRow(
                sequence=sequence,
                status=str(raw.get("status", "")).strip(),
                kitti_trans_percent=_parse_float(raw.get("kitti_trans_percent")),
                kitti_rot_deg_per_m=_parse_float(raw.get("kitti_rot_deg_per_m")),
                ate_rmse_associated=_parse_float(raw.get("ate_rmse_associated")),
                coverage=_parse_float(raw.get("coverage")),
            )
    return rows


def _ordered_sequences(left: dict[str, SequenceRow], right: dict[str, SequenceRow]) -> list[str]:
    seen = set(left.keys()) | set(right.keys())
    ordered = [seq for seq in KITTI_SEQUENCE_ORDER if seq in seen]
    extras = sorted(seq for seq in seen if seq not in KITTI_SEQUENCE_ORDER)
    return ordered + extras


def _winner_label(left_value: float, right_value: float) -> str:
    left_ok = math.isfinite(left_value)
    right_ok = math.isfinite(right_value)
    if left_ok and right_ok:
        if left_value < right_value:
            return "winner"
        if right_value < left_value:
            return "baseline"
        return "tie"
    if left_ok and not right_ok:
        return "winner"
    if right_ok and not left_ok:
        return "baseline"
    return "tie"


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def compare_kitti_runs(
    *,
    winner_csv: Path,
    baseline_csv: Path,
    output_dir: Path,
    winner_id: str,
    baseline_id: str,
) -> dict[str, Path]:
    winner_rows = _load_rows(winner_csv)
    baseline_rows = _load_rows(baseline_csv)
    sequences = _ordered_sequences(winner_rows, baseline_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_path = output_dir / "per_sequence_comparison.csv"
    headline_csv_path = output_dir / "headline_comparison.csv"
    headline_md_path = output_dir / "headline_comparison.md"

    winner_trans_values: list[float] = []
    winner_rot_values: list[float] = []
    winner_ate_values: list[float] = []
    baseline_trans_values: list[float] = []
    baseline_rot_values: list[float] = []
    baseline_ate_values: list[float] = []

    winner_ok = 0
    baseline_ok = 0
    winner_non_ok = 0
    baseline_non_ok = 0

    trans_counts = {"winner": 0, "baseline": 0, "tie": 0}
    rot_counts = {"winner": 0, "baseline": 0, "tie": 0}
    ate_counts = {"winner": 0, "baseline": 0, "tie": 0}

    with per_sequence_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "winner_status",
                "winner_kitti_trans_percent",
                "winner_kitti_rot_deg_per_m",
                "winner_ate_rmse_associated",
                "winner_coverage",
                "baseline_status",
                "baseline_kitti_trans_percent",
                "baseline_kitti_rot_deg_per_m",
                "baseline_ate_rmse_associated",
                "baseline_coverage",
                "better_translation",
                "better_rotation",
                "better_ate",
            ]
        )
        for sequence in sequences:
            winner_row = winner_rows.get(sequence, SequenceRow(sequence, "missing", math.nan, math.nan, math.nan, math.nan))
            baseline_row = baseline_rows.get(sequence, SequenceRow(sequence, "missing", math.nan, math.nan, math.nan, math.nan))

            winner_trans_values.append(winner_row.kitti_trans_percent)
            winner_rot_values.append(winner_row.kitti_rot_deg_per_m)
            winner_ate_values.append(winner_row.ate_rmse_associated)
            baseline_trans_values.append(baseline_row.kitti_trans_percent)
            baseline_rot_values.append(baseline_row.kitti_rot_deg_per_m)
            baseline_ate_values.append(baseline_row.ate_rmse_associated)

            if winner_row.status == "ok":
                winner_ok += 1
            else:
                winner_non_ok += 1
            if baseline_row.status == "ok":
                baseline_ok += 1
            else:
                baseline_non_ok += 1

            trans_better = _winner_label(winner_row.kitti_trans_percent, baseline_row.kitti_trans_percent)
            rot_better = _winner_label(winner_row.kitti_rot_deg_per_m, baseline_row.kitti_rot_deg_per_m)
            ate_better = _winner_label(winner_row.ate_rmse_associated, baseline_row.ate_rmse_associated)
            trans_counts[trans_better] += 1
            rot_counts[rot_better] += 1
            ate_counts[ate_better] += 1

            writer.writerow(
                [
                    sequence,
                    winner_row.status,
                    f"{winner_row.kitti_trans_percent:.6f}" if math.isfinite(winner_row.kitti_trans_percent) else "NaN",
                    f"{winner_row.kitti_rot_deg_per_m:.6f}" if math.isfinite(winner_row.kitti_rot_deg_per_m) else "NaN",
                    f"{winner_row.ate_rmse_associated:.6f}" if math.isfinite(winner_row.ate_rmse_associated) else "NaN",
                    f"{winner_row.coverage:.6f}" if math.isfinite(winner_row.coverage) else "NaN",
                    baseline_row.status,
                    f"{baseline_row.kitti_trans_percent:.6f}" if math.isfinite(baseline_row.kitti_trans_percent) else "NaN",
                    f"{baseline_row.kitti_rot_deg_per_m:.6f}" if math.isfinite(baseline_row.kitti_rot_deg_per_m) else "NaN",
                    f"{baseline_row.ate_rmse_associated:.6f}" if math.isfinite(baseline_row.ate_rmse_associated) else "NaN",
                    f"{baseline_row.coverage:.6f}" if math.isfinite(baseline_row.coverage) else "NaN",
                    trans_better,
                    rot_better,
                    ate_better,
                ]
            )

    winner_trans_mean = _mean_finite(winner_trans_values)
    winner_rot_mean = _mean_finite(winner_rot_values)
    winner_ate_mean = _mean_finite(winner_ate_values)
    baseline_trans_mean = _mean_finite(baseline_trans_values)
    baseline_rot_mean = _mean_finite(baseline_rot_values)
    baseline_ate_mean = _mean_finite(baseline_ate_values)

    with headline_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method_id",
                "mean_kitti_trans_percent",
                "mean_kitti_rot_deg_per_m",
                "mean_ate_rmse_associated",
                "finite_sequence_count",
                "ok_sequence_count",
                "non_ok_sequence_count",
                "translation_wins_vs_other",
                "translation_losses_vs_other",
                "translation_ties_vs_other",
                "rotation_wins_vs_other",
                "rotation_losses_vs_other",
                "rotation_ties_vs_other",
                "ate_wins_vs_other",
                "ate_losses_vs_other",
                "ate_ties_vs_other",
            ]
        )
        writer.writerow(
            [
                winner_id,
                f"{winner_trans_mean:.6f}" if math.isfinite(winner_trans_mean) else "NaN",
                f"{winner_rot_mean:.6f}" if math.isfinite(winner_rot_mean) else "NaN",
                f"{winner_ate_mean:.6f}" if math.isfinite(winner_ate_mean) else "NaN",
                sum(1 for value in winner_ate_values if math.isfinite(value)),
                winner_ok,
                winner_non_ok,
                trans_counts["winner"],
                trans_counts["baseline"],
                trans_counts["tie"],
                rot_counts["winner"],
                rot_counts["baseline"],
                rot_counts["tie"],
                ate_counts["winner"],
                ate_counts["baseline"],
                ate_counts["tie"],
            ]
        )
        writer.writerow(
            [
                baseline_id,
                f"{baseline_trans_mean:.6f}" if math.isfinite(baseline_trans_mean) else "NaN",
                f"{baseline_rot_mean:.6f}" if math.isfinite(baseline_rot_mean) else "NaN",
                f"{baseline_ate_mean:.6f}" if math.isfinite(baseline_ate_mean) else "NaN",
                sum(1 for value in baseline_ate_values if math.isfinite(value)),
                baseline_ok,
                baseline_non_ok,
                trans_counts["baseline"],
                trans_counts["winner"],
                trans_counts["tie"],
                rot_counts["baseline"],
                rot_counts["winner"],
                rot_counts["tie"],
                ate_counts["baseline"],
                ate_counts["winner"],
                ate_counts["tie"],
            ]
        )

    headline_md_path.write_text(
        "\n".join(
            [
                "# KITTI Headline Comparison",
                "",
                f"- Winner method: `{winner_id}`",
                f"- Baseline method: `{baseline_id}`",
                f"- Mean KITTI translation drift: `{winner_id}` `{winner_trans_mean:.6f}`, `{baseline_id}` `{baseline_trans_mean:.6f}`",
                f"- Mean KITTI rotation drift: `{winner_id}` `{winner_rot_mean:.6f}`, `{baseline_id}` `{baseline_rot_mean:.6f}`",
                f"- Mean ATE-style sidecar: `{winner_id}` `{winner_ate_mean:.6f}`, `{baseline_id}` `{baseline_ate_mean:.6f}`",
                f"- Translation wins: `{winner_id}` `{trans_counts['winner']}`, `{baseline_id}` `{trans_counts['baseline']}`, ties `{trans_counts['tie']}`",
                f"- Rotation wins: `{winner_id}` `{rot_counts['winner']}`, `{baseline_id}` `{rot_counts['baseline']}`, ties `{rot_counts['tie']}`",
                f"- ATE wins: `{winner_id}` `{ate_counts['winner']}`, `{baseline_id}` `{ate_counts['baseline']}`, ties `{ate_counts['tie']}`",
                f"- Non-ok sequences: `{winner_id}` `{winner_non_ok}`, `{baseline_id}` `{baseline_non_ok}`",
                "",
                "Per-sequence details are in `per_sequence_comparison.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "per_sequence": per_sequence_path,
        "headline_csv": headline_csv_path,
        "headline_md": headline_md_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two KITTI metrics_summary.csv files.")
    ap.add_argument("--winner-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--winner-id", default="winner")
    ap.add_argument("--baseline-id", default="baseline")
    args = ap.parse_args()

    compare_kitti_runs(
        winner_csv=Path(args.winner_csv).expanduser().resolve(),
        baseline_csv=Path(args.baseline_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        winner_id=str(args.winner_id),
        baseline_id=str(args.baseline_id),
    )


if __name__ == "__main__":
    main()

