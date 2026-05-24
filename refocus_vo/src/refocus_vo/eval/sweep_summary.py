from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _method_rows(sweep_root: Path) -> dict[str, list[dict]]:
    out = {}
    for method_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        csv_path = method_dir / "metrics_summary.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            out[method_dir.name] = list(csv.DictReader(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a DINO-guided VO sweep.")
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method_id, method_rows in _method_rows(sweep_root).items():
        ate_assoc = [_safe_float(row.get("ate_rmse_associated", "NaN")) for row in method_rows]
        coverage = [_safe_float(row.get("coverage", "NaN")) for row in method_rows]
        rpe_t = [_safe_float(row.get("rpe_trans_rmse", "NaN")) for row in method_rows]
        rpe_r = [_safe_float(row.get("rpe_rot_rmse", "NaN")) for row in method_rows]
        partial_count = sum(1 for row in method_rows if row.get("status") == "partial_low_coverage")
        eligible = 1 if partial_count <= 1 else 0
        rows.append(
            {
                "method_id": method_id,
                "eligible": str(eligible),
                "num_sequences": str(len(method_rows)),
                "partial_low_coverage_count": str(partial_count),
                "mean_ate_rmse_associated": f"{_mean(ate_assoc):.6f}",
                "mean_coverage": f"{_mean(coverage):.6f}",
                "mean_rpe_trans_rmse": f"{_mean(rpe_t):.6f}",
                "mean_rpe_rot_rmse": f"{_mean(rpe_r):.6f}",
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item["eligible"]),
            _safe_float(item["mean_ate_rmse_associated"]),
            int(item["partial_low_coverage_count"]),
            -_safe_float(item["mean_coverage"]),
            _safe_float(item["mean_rpe_trans_rmse"]),
            _safe_float(item["mean_rpe_rot_rmse"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)

    header = [
        "rank",
        "method_id",
        "eligible",
        "num_sequences",
        "partial_low_coverage_count",
        "mean_ate_rmse_associated",
        "mean_coverage",
        "mean_rpe_trans_rmse",
        "mean_rpe_rot_rmse",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
