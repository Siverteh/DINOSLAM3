from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


EUROC_SEQUENCE_ORDER = [
    "MH01",
    "MH02",
    "MH03",
    "MH04",
    "MH05",
    "V101",
    "V102",
    "V103",
    "V201",
    "V202",
    "V203",
]


@dataclass(frozen=True)
class SequenceRow:
    sequence: str
    status: str
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
                ate_rmse_associated=_parse_float(raw.get("ate_rmse_associated")),
                coverage=_parse_float(raw.get("coverage")),
            )
    return rows


def _ordered_sequences(left: dict[str, SequenceRow], right: dict[str, SequenceRow]) -> list[str]:
    seen = set(left.keys()) | set(right.keys())
    ordered = [seq for seq in EUROC_SEQUENCE_ORDER if seq in seen]
    extras = sorted(seq for seq in seen if seq not in EUROC_SEQUENCE_ORDER)
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


def compare_euroc_runs(
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

    winner_values: list[float] = []
    baseline_values: list[float] = []
    winner_wins = 0
    baseline_wins = 0
    ties = 0
    winner_ok = 0
    baseline_ok = 0
    winner_non_ok = 0
    baseline_non_ok = 0

    with per_sequence_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "winner_status",
                "winner_ate_rmse_associated",
                "winner_coverage",
                "baseline_status",
                "baseline_ate_rmse_associated",
                "baseline_coverage",
                "better_method",
            ]
        )

        for sequence in sequences:
            winner_row = winner_rows.get(sequence, SequenceRow(sequence=sequence, status="missing", ate_rmse_associated=math.nan, coverage=math.nan))
            baseline_row = baseline_rows.get(sequence, SequenceRow(sequence=sequence, status="missing", ate_rmse_associated=math.nan, coverage=math.nan))

            winner_values.append(winner_row.ate_rmse_associated)
            baseline_values.append(baseline_row.ate_rmse_associated)

            if winner_row.status == "ok":
                winner_ok += 1
            else:
                winner_non_ok += 1
            if baseline_row.status == "ok":
                baseline_ok += 1
            else:
                baseline_non_ok += 1

            better_method = _winner_label(winner_row.ate_rmse_associated, baseline_row.ate_rmse_associated)
            if better_method == "winner":
                winner_wins += 1
            elif better_method == "baseline":
                baseline_wins += 1
            else:
                ties += 1

            writer.writerow(
                [
                    sequence,
                    winner_row.status,
                    f"{winner_row.ate_rmse_associated:.6f}" if math.isfinite(winner_row.ate_rmse_associated) else "NaN",
                    f"{winner_row.coverage:.6f}" if math.isfinite(winner_row.coverage) else "NaN",
                    baseline_row.status,
                    f"{baseline_row.ate_rmse_associated:.6f}" if math.isfinite(baseline_row.ate_rmse_associated) else "NaN",
                    f"{baseline_row.coverage:.6f}" if math.isfinite(baseline_row.coverage) else "NaN",
                    better_method,
                ]
            )

    winner_mean = _mean_finite(winner_values)
    baseline_mean = _mean_finite(baseline_values)
    delta = winner_mean - baseline_mean if math.isfinite(winner_mean) and math.isfinite(baseline_mean) else math.nan

    with headline_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method_id",
                "mean_ate_rmse_associated",
                "finite_sequence_count",
                "ok_sequence_count",
                "non_ok_sequence_count",
                "wins_vs_other",
                "losses_vs_other",
                "ties_vs_other",
            ]
        )
        writer.writerow(
            [
                winner_id,
                f"{winner_mean:.6f}" if math.isfinite(winner_mean) else "NaN",
                sum(1 for value in winner_values if math.isfinite(value)),
                winner_ok,
                winner_non_ok,
                winner_wins,
                baseline_wins,
                ties,
            ]
        )
        writer.writerow(
            [
                baseline_id,
                f"{baseline_mean:.6f}" if math.isfinite(baseline_mean) else "NaN",
                sum(1 for value in baseline_values if math.isfinite(value)),
                baseline_ok,
                baseline_non_ok,
                baseline_wins,
                winner_wins,
                ties,
            ]
        )

    better_overall = "winner" if math.isfinite(delta) and delta < 0 else "baseline" if math.isfinite(delta) and delta > 0 else "tie"
    headline_md_path.write_text(
        "\n".join(
            [
                "# EuRoC Headline Comparison",
                "",
                f"- Winner method: `{winner_id}`",
                f"- Baseline method: `{baseline_id}`",
                f"- Winner mean associated ATE: `{winner_mean:.6f}`" if math.isfinite(winner_mean) else f"- Winner mean associated ATE: `NaN`",
                f"- Baseline mean associated ATE: `{baseline_mean:.6f}`" if math.isfinite(baseline_mean) else f"- Baseline mean associated ATE: `NaN`",
                f"- Delta (winner - baseline): `{delta:.6f}`" if math.isfinite(delta) else "- Delta (winner - baseline): `NaN`",
                f"- Better overall by mean: `{better_overall}`",
                f"- Sequence wins: `{winner_id}` `{winner_wins}`, `{baseline_id}` `{baseline_wins}`, ties `{ties}`",
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
    ap = argparse.ArgumentParser(description="Compare two EuRoC dpvo_style_metrics_summary.csv files.")
    ap.add_argument("--winner-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--winner-id", default="winner")
    ap.add_argument("--baseline-id", default="baseline")
    args = ap.parse_args()

    compare_euroc_runs(
        winner_csv=Path(args.winner_csv).expanduser().resolve(),
        baseline_csv=Path(args.baseline_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        winner_id=str(args.winner_id),
        baseline_id=str(args.baseline_id),
    )


if __name__ == "__main__":
    main()
