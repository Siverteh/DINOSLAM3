from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


FAMILY_ORDER = ("freiburg1", "freiburg2", "freiburg3")
METHOD_ORDER = ("dpvo_native", "focus071_best")
ALLOWED_STATUSES = {"ok", "partial_low_coverage"}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _family_for_sequence(sequence: str) -> str:
    seq = str(sequence).strip()
    for family in FAMILY_ORDER:
        if seq.startswith(family + "_"):
            return family
    raise ValueError(f"Unsupported Freiburg sequence: {sequence}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(sum(usable) / len(usable))


def _median(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(statistics.median(usable))


def _validate_repeat_rows(
    rows: list[dict[str, str]],
    *,
    expected_sequences: list[str],
    csv_path: Path,
) -> None:
    expected = list(expected_sequences)
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if len(rows) != len(expected):
        raise ValueError(
            f"{csv_path} has {len(rows)} rows; expected {len(expected)} Freiburg rows"
        )
    if seen != expected:
        raise ValueError(
            f"{csv_path} sequence order/content mismatch; expected {expected}, got {seen}"
        )
    bad = [
        row
        for row in rows
        if str(row.get("status", "")).strip() not in ALLOWED_STATUSES
    ]
    if bad:
        raise ValueError(
            f"{csv_path} contains unsupported status rows: "
            + ", ".join(
                f"{row.get('sequence')}:{row.get('status')}" for row in bad[:5]
            )
        )


def _repeat_summary_row(
    *,
    method: str,
    repeat_id: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    by_family = {
        family: [row for row in rows if _family_for_sequence(str(row.get("sequence", ""))) == family]
        for family in FAMILY_ORDER
    }
    out: dict[str, object] = {
        "method": method,
        "repeat_id": repeat_id,
        "full_mean_ate_rmse": _mean([_safe_float(row.get("ate_rmse")) for row in rows]),
        "full_mean_ate_rmse_associated": _mean(
            [_safe_float(row.get("ate_rmse_associated")) for row in rows]
        ),
        "full_mean_coverage": _mean([_safe_float(row.get("coverage")) for row in rows]),
    }
    for family, family_rows in by_family.items():
        out[f"{family}_mean_ate_rmse"] = _mean(
            [_safe_float(row.get("ate_rmse")) for row in family_rows]
        )
        out[f"{family}_mean_ate_rmse_associated"] = _mean(
            [_safe_float(row.get("ate_rmse_associated")) for row in family_rows]
        )
        out[f"{family}_mean_coverage"] = _mean(
            [_safe_float(row.get("coverage")) for row in family_rows]
        )
    return out


def _per_sequence_medians(
    *,
    expected_sequences: list[str],
    repeat_rows: dict[str, list[list[dict[str, str]]]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        rows_by_repeat = repeat_rows.get(method, [])
        for sequence in expected_sequences:
            seq_rows = [
                next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
                for rows in rows_by_repeat
            ]
            output.append(
                {
                    "method": method,
                    "sequence": sequence,
                    "family": _family_for_sequence(sequence),
                    "median_ate_rmse": _median(
                        [_safe_float(row.get("ate_rmse")) for row in seq_rows]
                    ),
                    "median_ate_rmse_associated": _median(
                        [_safe_float(row.get("ate_rmse_associated")) for row in seq_rows]
                    ),
                    "median_coverage": _median(
                        [_safe_float(row.get("coverage")) for row in seq_rows]
                    ),
                }
            )
    return output


def _method_comparison_rows(
    *,
    repeat_summary_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_repeat = [row for row in repeat_summary_rows if str(row.get("method")) == method]
        method_seq = [row for row in per_sequence_rows if str(row.get("method")) == method]
        row: dict[str, object] = {
            "method": method,
            "sequence_count": len(method_seq),
            "full_mean_of_sequence_medians_ate_rmse": _mean(
                [_safe_float(item.get("median_ate_rmse")) for item in method_seq]
            ),
            "full_mean_of_sequence_medians_ate_rmse_associated": _mean(
                [_safe_float(item.get("median_ate_rmse_associated")) for item in method_seq]
            ),
            "full_mean_of_sequence_medians_coverage": _mean(
                [_safe_float(item.get("median_coverage")) for item in method_seq]
            ),
            "full_median_of_repeat_means_ate_rmse": _median(
                [_safe_float(item.get("full_mean_ate_rmse")) for item in method_repeat]
            ),
            "full_median_of_repeat_means_ate_rmse_associated": _median(
                [_safe_float(item.get("full_mean_ate_rmse_associated")) for item in method_repeat]
            ),
            "full_median_of_repeat_means_coverage": _median(
                [_safe_float(item.get("full_mean_coverage")) for item in method_repeat]
            ),
        }
        for family in FAMILY_ORDER:
            family_seq = [item for item in method_seq if str(item.get("family")) == family]
            family_repeat = [item for item in method_repeat]
            row[f"{family}_mean_of_sequence_medians_ate_rmse"] = _mean(
                [_safe_float(item.get("median_ate_rmse")) for item in family_seq]
            )
            row[f"{family}_mean_of_sequence_medians_ate_rmse_associated"] = _mean(
                [_safe_float(item.get("median_ate_rmse_associated")) for item in family_seq]
            )
            row[f"{family}_mean_of_sequence_medians_coverage"] = _mean(
                [_safe_float(item.get("median_coverage")) for item in family_seq]
            )
            row[f"{family}_median_of_repeat_means_ate_rmse"] = _median(
                [_safe_float(item.get(f"{family}_mean_ate_rmse")) for item in family_repeat]
            )
            row[f"{family}_median_of_repeat_means_ate_rmse_associated"] = _median(
                [
                    _safe_float(item.get(f"{family}_mean_ate_rmse_associated"))
                    for item in family_repeat
                ]
            )
            row[f"{family}_median_of_repeat_means_coverage"] = _median(
                [_safe_float(item.get(f"{family}_mean_coverage")) for item in family_repeat]
            )
        output.append(row)
    return output


def _wins_and_losses_md(per_sequence_rows: list[dict[str, object]]) -> list[str]:
    by_method_and_sequence = {
        (str(row.get("method")), str(row.get("sequence"))): row for row in per_sequence_rows
    }
    deltas: list[tuple[str, float]] = []
    for sequence in sorted(
        {str(row.get("sequence")) for row in per_sequence_rows if str(row.get("sequence"))}
    ):
        dpvo = _safe_float(
            by_method_and_sequence[("dpvo_native", sequence)].get("median_ate_rmse_associated")
        )
        dino = _safe_float(
            by_method_and_sequence[("focus071_best", sequence)].get("median_ate_rmse_associated")
        )
        if math.isfinite(dpvo) and math.isfinite(dino):
            deltas.append((sequence, dino - dpvo))
    dino_best = sorted([item for item in deltas if item[1] < 0.0], key=lambda item: item[1])[:5]
    dpvo_best = sorted([item for item in deltas if item[1] > 0.0], key=lambda item: item[1], reverse=True)[:5]
    lines = [
        "## Largest Per-Sequence Margins",
        "",
        "### Focus071 better than DPVO",
        "",
    ]
    if dino_best:
        for sequence, delta in dino_best:
            lines.append(f"- `{sequence}`: `{abs(delta):.6f}` lower assoc ATE")
    else:
        lines.append("- none")
    lines.extend(["", "### DPVO better than Focus071", ""])
    if dpvo_best:
        for sequence, delta in dpvo_best:
            lines.append(f"- `{sequence}`: `{abs(delta):.6f}` lower assoc ATE")
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _write_method_comparison_md(
    *,
    output_path: Path,
    method_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
) -> None:
    by_method = {str(row.get("method")): row for row in method_rows}
    dino = by_method.get("focus071_best", {})
    dpvo = by_method.get("dpvo_native", {})
    lines = [
        "# TUM RGB-D Freiburg1/2/3: DPVO vs Focus071",
        "",
        "## Method Summary",
        "",
        "| Method | Full Mean of Sequence Medians (Assoc ATE) | Full Median of Repeat Means (Assoc ATE) | Freiburg1 | Freiburg2 | Freiburg3 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `dpvo_native` | "
            f"{_safe_float(dpvo.get('full_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dpvo.get('full_median_of_repeat_means_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dpvo.get('freiburg1_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dpvo.get('freiburg2_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dpvo.get('freiburg3_mean_of_sequence_medians_ate_rmse_associated')):.6f} |"
        ),
        (
            f"| `focus071_best` | "
            f"{_safe_float(dino.get('full_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dino.get('full_median_of_repeat_means_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dino.get('freiburg1_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dino.get('freiburg2_mean_of_sequence_medians_ate_rmse_associated')):.6f} | "
            f"{_safe_float(dino.get('freiburg3_mean_of_sequence_medians_ate_rmse_associated')):.6f} |"
        ),
        "",
        "## Per-Sequence Median Assoc ATE",
        "",
        "| Sequence | DPVO | Focus071 | Winner |",
        "|---|---:|---:|---|",
    ]
    by_method_and_sequence = {
        (str(row.get("method")), str(row.get("sequence"))): row for row in per_sequence_rows
    }
    sequences = sorted(
        {str(row.get("sequence")) for row in per_sequence_rows if str(row.get("sequence"))},
        key=lambda sequence: (_family_for_sequence(sequence), sequence),
    )
    for sequence in sequences:
        dpvo_val = _safe_float(
            by_method_and_sequence[("dpvo_native", sequence)].get("median_ate_rmse_associated")
        )
        dino_val = _safe_float(
            by_method_and_sequence[("focus071_best", sequence)].get("median_ate_rmse_associated")
        )
        winner = "tie"
        if math.isfinite(dpvo_val) and math.isfinite(dino_val):
            if dino_val < dpvo_val:
                winner = "focus071_best"
            elif dpvo_val < dino_val:
                winner = "dpvo_native"
        lines.append(f"| `{sequence}` | {dpvo_val:.6f} | {dino_val:.6f} | `{winner}` |")
    lines.extend(_wins_and_losses_md(per_sequence_rows))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_benchmark(
    *,
    benchmark_root: Path,
    expected_sequences: list[str],
    repeats: int = 5,
) -> None:
    repeat_summary_rows: list[dict[str, object]] = []
    repeat_rows_by_method: dict[str, list[list[dict[str, str]]]] = {method: [] for method in METHOD_ORDER}
    for method in METHOD_ORDER:
        for repeat_idx in range(1, int(repeats) + 1):
            repeat_id = f"repeat_{repeat_idx:02d}"
            csv_path = benchmark_root / method / repeat_id / "dpvo_style_metrics_summary.csv"
            rows = _read_csv_rows(csv_path)
            _validate_repeat_rows(rows, expected_sequences=expected_sequences, csv_path=csv_path)
            repeat_rows_by_method[method].append(rows)
            repeat_summary_rows.append(
                _repeat_summary_row(method=method, repeat_id=repeat_id, rows=rows)
            )

    summary_dir = benchmark_root / "summary"
    repeat_fieldnames = [
        "method",
        "repeat_id",
        "full_mean_ate_rmse",
        "full_mean_ate_rmse_associated",
        "full_mean_coverage",
    ]
    for family in FAMILY_ORDER:
        repeat_fieldnames.extend(
            [
                f"{family}_mean_ate_rmse",
                f"{family}_mean_ate_rmse_associated",
                f"{family}_mean_coverage",
            ]
        )
    _write_csv(summary_dir / "repeat_summary.csv", repeat_summary_rows, repeat_fieldnames)

    per_sequence_rows = _per_sequence_medians(
        expected_sequences=expected_sequences,
        repeat_rows=repeat_rows_by_method,
    )
    _write_csv(
        summary_dir / "per_sequence_median.csv",
        per_sequence_rows,
        [
            "method",
            "sequence",
            "family",
            "median_ate_rmse",
            "median_ate_rmse_associated",
            "median_coverage",
        ],
    )

    method_rows = _method_comparison_rows(
        repeat_summary_rows=repeat_summary_rows,
        per_sequence_rows=per_sequence_rows,
    )
    method_fieldnames = [
        "method",
        "sequence_count",
        "full_mean_of_sequence_medians_ate_rmse",
        "full_mean_of_sequence_medians_ate_rmse_associated",
        "full_mean_of_sequence_medians_coverage",
        "full_median_of_repeat_means_ate_rmse",
        "full_median_of_repeat_means_ate_rmse_associated",
        "full_median_of_repeat_means_coverage",
    ]
    for family in FAMILY_ORDER:
        method_fieldnames.extend(
            [
                f"{family}_mean_of_sequence_medians_ate_rmse",
                f"{family}_mean_of_sequence_medians_ate_rmse_associated",
                f"{family}_mean_of_sequence_medians_coverage",
                f"{family}_median_of_repeat_means_ate_rmse",
                f"{family}_median_of_repeat_means_ate_rmse_associated",
                f"{family}_median_of_repeat_means_coverage",
            ]
        )
    _write_csv(summary_dir / "method_comparison.csv", method_rows, method_fieldnames)
    _write_method_comparison_md(
        output_path=summary_dir / "method_comparison.md",
        method_rows=method_rows,
        per_sequence_rows=per_sequence_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate repeated TUM RGB-D Freiburg1/2/3 benchmark outputs."
    )
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    sequences = [item.strip() for item in str(args.sequences).split(",") if item.strip()]
    aggregate_benchmark(
        benchmark_root=benchmark_root,
        expected_sequences=sequences,
        repeats=int(args.repeats),
    )


if __name__ == "__main__":
    main()
