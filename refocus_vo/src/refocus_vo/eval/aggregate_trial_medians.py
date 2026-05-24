from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TUM_FREIBURG1_PAPER_DEFAULT = {
    "freiburg1_360": 0.135,
    "freiburg1_desk": 0.038,
    "freiburg1_desk2": 0.048,
    "freiburg1_floor": 0.040,
    "freiburg1_plant": 0.036,
    "freiburg1_room": 0.394,
    "freiburg1_rpy": 0.034,
    "freiburg1_teddy": 0.064,
    "freiburg1_xyz": 0.012,
}

TUM_FREIBURG1_PAPER_FAST = {
    "freiburg1_360": 0.169,
    "freiburg1_desk": 0.029,
    "freiburg1_desk2": 0.064,
    "freiburg1_floor": 0.047,
    "freiburg1_plant": 0.047,
    "freiburg1_room": 0.396,
    "freiburg1_rpy": 0.034,
    "freiburg1_teddy": 0.074,
    "freiburg1_xyz": 0.012,
}

NUMERIC_FIELDS = [
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


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _paper_value(benchmark: str, sequence: str, mode: str) -> float:
    if benchmark != "tum_freiburg1_paper":
        return math.nan
    if mode == "default":
        return float(TUM_FREIBURG1_PAPER_DEFAULT.get(sequence, math.nan))
    if mode == "fast":
        return float(TUM_FREIBURG1_PAPER_FAST.get(sequence, math.nan))
    return math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate repeated benchmark CSVs into median tables.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--trial-csv", action="append", default=[])
    ap.add_argument("--benchmark", default="")
    args = ap.parse_args()

    grouped: dict[str, list[dict[str, str]]] = {}
    for csv_path in args.trial_csv:
        path = Path(csv_path).expanduser().resolve()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                grouped.setdefault(str(row.get("sequence", "")), []).append(row)

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "sequence",
        "trial_count",
        "status",
        *NUMERIC_FIELDS,
        "paper_default_ate",
        "paper_fast_ate",
        "delta_vs_paper_default",
        "delta_vs_paper_fast",
    ]

    rows_out: list[dict[str, str | float | int]] = []
    for sequence in sorted(k for k in grouped.keys() if k):
        trial_rows = grouped[sequence]
        numeric_values = {
            field: [
                _parse_float(row.get(field))
                for row in trial_rows
                if math.isfinite(_parse_float(row.get(field)))
            ]
            for field in NUMERIC_FIELDS
        }
        status = "ok"
        if any(str(row.get("status", "")) != "ok" for row in trial_rows):
            status = str(trial_rows[0].get("status", ""))

        ate_median = _median(numeric_values["ate_rmse"])
        paper_default = _paper_value(str(args.benchmark), sequence, "default")
        paper_fast = _paper_value(str(args.benchmark), sequence, "fast")
        rows_out.append(
            {
                "sequence": sequence,
                "trial_count": len(trial_rows),
                "status": status,
                **{field: _median(values) for field, values in numeric_values.items()},
                "paper_default_ate": paper_default,
                "paper_fast_ate": paper_fast,
                "delta_vs_paper_default": ate_median - paper_default if math.isfinite(ate_median) and math.isfinite(paper_default) else math.nan,
                "delta_vs_paper_fast": ate_median - paper_fast if math.isfinite(ate_median) and math.isfinite(paper_fast) else math.nan,
            }
        )

    if rows_out:
        avg_row = {
            "sequence": "AVG",
            "trial_count": int(_median([float(row["trial_count"]) for row in rows_out])),
            "status": "aggregate",
        }
        for field in NUMERIC_FIELDS:
            avg_row[field] = _mean([float(row[field]) for row in rows_out if math.isfinite(float(row[field]))])
        avg_row["paper_default_ate"] = _mean(
            [float(row["paper_default_ate"]) for row in rows_out if math.isfinite(float(row["paper_default_ate"]))]
        )
        avg_row["paper_fast_ate"] = _mean(
            [float(row["paper_fast_ate"]) for row in rows_out if math.isfinite(float(row["paper_fast_ate"]))]
        )
        avg_row["delta_vs_paper_default"] = (
            float(avg_row["ate_rmse"]) - float(avg_row["paper_default_ate"])
            if math.isfinite(float(avg_row["ate_rmse"])) and math.isfinite(float(avg_row["paper_default_ate"]))
            else math.nan
        )
        avg_row["delta_vs_paper_fast"] = (
            float(avg_row["ate_rmse"]) - float(avg_row["paper_fast_ate"])
            if math.isfinite(float(avg_row["ate_rmse"])) and math.isfinite(float(avg_row["paper_fast_ate"]))
            else math.nan
        )
        rows_out.append(avg_row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)


if __name__ == "__main__":
    main()
