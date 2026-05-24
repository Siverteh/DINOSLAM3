from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_spec(spec: str) -> tuple[Path, str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError(
            f"Invalid input spec '{spec}'. Expected 'csv_path|method_id|family|modality|feature_type'."
        )
    csv_path, method_id, family, modality, feature_type = parts
    return Path(csv_path).expanduser().resolve(), method_id, family, modality, feature_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-method metrics CSVs into one comparison CSV.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--input-spec", action="append", default=[])
    args = ap.parse_args()

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "sequence",
        "method_id",
        "family",
        "modality",
        "feature_type",
        "status",
        "ate_rmse",
        "ate_rmse_associated",
        "rpe_trans_rmse",
        "rpe_rot_rmse",
        "coverage",
    ]

    merged_rows = []
    for spec in args.input_spec:
        csv_path, method_id, family, modality, feature_type = _parse_spec(spec)
        if not csv_path.exists():
            continue
        for row in _read_rows(csv_path):
            merged_rows.append(
                {
                    "sequence": row.get("sequence", ""),
                    "method_id": method_id,
                    "family": family,
                    "modality": modality,
                    "feature_type": feature_type or row.get("feature_type", ""),
                    "status": row.get("status", ""),
                    "ate_rmse": row.get("ate_rmse", "NaN"),
                    "ate_rmse_associated": row.get("ate_rmse_associated", "NaN"),
                    "rpe_trans_rmse": row.get("rpe_trans_rmse", "NaN"),
                    "rpe_rot_rmse": row.get("rpe_rot_rmse", "NaN"),
                    "coverage": row.get("coverage", "NaN"),
                }
            )

    merged_rows.sort(key=lambda item: (item["sequence"], item["method_id"]))
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
