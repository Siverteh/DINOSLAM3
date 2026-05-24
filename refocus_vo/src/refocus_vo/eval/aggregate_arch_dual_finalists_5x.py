from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ALLOWED_STATUSES = {"ok", "partial_low_coverage", "invalid_trajectory"}
FAMILY_ORDER = ("freiburg1", "freiburg2", "freiburg3")
BASELINE_METHODS = ("dpvo_native", "focus071_best")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _family_for_sequence(sequence: str) -> str:
    seq = str(sequence).strip()
    for family in FAMILY_ORDER:
        if seq.startswith(family + "_"):
            return family
    raise ValueError(f"Unsupported Freiburg sequence: {sequence}")


def _read_frozen_baselines(path: Path, *, expected_sequences: list[str]) -> dict[str, dict[str, float]]:
    rows = _read_csv_rows(path)
    by_method: dict[str, dict[str, float]] = {method: {} for method in BASELINE_METHODS}
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method not in by_method:
            continue
        sequence = str(row.get("sequence", "")).strip()
        value = _safe_float(row.get("median_ate_rmse_associated"))
        if sequence and math.isfinite(value):
            by_method[method][sequence] = value
    for method in BASELINE_METHODS:
        missing = [seq for seq in expected_sequences if seq not in by_method[method]]
        if missing:
            raise ValueError(f"Frozen baseline {method} missing sequences: {missing}")
    return by_method


def _read_historical_method_rows(path: Path) -> dict[str, dict[str, str]]:
    return {
        str(row.get("method", "")).strip(): row
        for row in _read_csv_rows(path)
        if str(row.get("method", "")).strip()
    }


def _read_historical_repeat_summary(path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_method: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv_rows(path):
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        rows_by_method.setdefault(method, []).append(row)
    return rows_by_method


def _validate_repeat_rows(
    rows: list[dict[str, str]],
    *,
    expected_sequences: list[str],
    csv_path: Path,
) -> None:
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if len(rows) != len(expected_sequences):
        raise ValueError(f"{csv_path} has {len(rows)} rows; expected {len(expected_sequences)}")
    if seen != list(expected_sequences):
        raise ValueError(f"{csv_path} sequence order/content mismatch")
    bad = [row for row in rows if str(row.get("status", "")).strip() not in ALLOWED_STATUSES]
    if bad:
        raise ValueError(
            f"{csv_path} contains unsupported status rows: "
            + ", ".join(f"{row.get('sequence')}:{row.get('status')}" for row in bad[:5])
        )


def _wins_against_baseline(
    rows: list[dict[str, str]],
    *,
    baseline_assoc: dict[str, float],
) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        assoc = _safe_float(row.get("ate_rmse_associated"))
        baseline = baseline_assoc[sequence]
        if not math.isfinite(assoc):
            losses += 1
        elif assoc < baseline:
            wins += 1
        elif baseline < assoc:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def _repeat_summary_row(
    *,
    finalist_id: str,
    repeat_id: str,
    rows: list[dict[str, str]],
    baseline_dpvo: dict[str, float],
    baseline_old_focus071: dict[str, float],
) -> dict[str, object]:
    by_family = {
        family: [row for row in rows if _family_for_sequence(str(row.get("sequence", ""))) == family]
        for family in FAMILY_ORDER
    }
    wins_dpvo, losses_dpvo, ties_dpvo = _wins_against_baseline(rows, baseline_assoc=baseline_dpvo)
    wins_old, losses_old, ties_old = _wins_against_baseline(rows, baseline_assoc=baseline_old_focus071)
    output: dict[str, object] = {
        "finalist_id": finalist_id,
        "repeat_id": repeat_id,
        "full_mean_ate_rmse": f"{_mean([_safe_float(row.get('ate_rmse')) for row in rows]):.6f}",
        "full_mean_ate_rmse_associated": f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in rows]):.6f}",
        "full_mean_coverage": f"{_mean([_safe_float(row.get('coverage')) for row in rows]):.6f}",
        "wins_vs_frozen_dpvo_median": wins_dpvo,
        "losses_vs_frozen_dpvo_median": losses_dpvo,
        "ties_vs_frozen_dpvo_median": ties_dpvo,
        "wins_vs_frozen_old_focus071_median": wins_old,
        "losses_vs_frozen_old_focus071_median": losses_old,
        "ties_vs_frozen_old_focus071_median": ties_old,
    }
    for family in FAMILY_ORDER:
        family_rows = by_family[family]
        output[f"{family}_mean_ate_rmse"] = f"{_mean([_safe_float(row.get('ate_rmse')) for row in family_rows]):.6f}"
        output[f"{family}_mean_ate_rmse_associated"] = f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in family_rows]):.6f}"
        output[f"{family}_mean_coverage"] = f"{_mean([_safe_float(row.get('coverage')) for row in family_rows]):.6f}"
    return output


def _per_sequence_medians(
    *,
    finalist_ids: list[str],
    repeat_rows: dict[str, list[list[dict[str, str]]]],
    expected_sequences: list[str],
    baseline_dpvo: dict[str, float],
    baseline_old_focus071: dict[str, float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for finalist_id in finalist_ids:
        rows_by_repeat = repeat_rows[finalist_id]
        for sequence in expected_sequences:
            seq_rows = [
                next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
                for rows in rows_by_repeat
            ]
            assoc = _median([_safe_float(row.get("ate_rmse_associated")) for row in seq_rows])
            baseline_dpvo_assoc = baseline_dpvo[sequence]
            baseline_old_assoc = baseline_old_focus071[sequence]
            output.append(
                {
                    "finalist_id": finalist_id,
                    "sequence": sequence,
                    "family": _family_for_sequence(sequence),
                    "median_ate_rmse": f"{_median([_safe_float(row.get('ate_rmse')) for row in seq_rows]):.6f}",
                    "median_ate_rmse_associated": f"{assoc:.6f}",
                    "median_coverage": f"{_median([_safe_float(row.get('coverage')) for row in seq_rows]):.6f}",
                    "baseline_dpvo_assoc_median": f"{baseline_dpvo_assoc:.6f}",
                    "baseline_old_focus071_assoc_median": f"{baseline_old_assoc:.6f}",
                    "winner_vs_dpvo": (
                        "finalist" if assoc < baseline_dpvo_assoc else "dpvo_native" if baseline_dpvo_assoc < assoc else "tie"
                    ),
                    "winner_vs_old_focus071": (
                        "finalist" if assoc < baseline_old_assoc else "focus071_best" if baseline_old_assoc < assoc else "tie"
                    ),
                }
            )
    return output


def _method_comparison_rows(
    *,
    finalist_ids: list[str],
    repeat_summary_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
    historical_method_rows: dict[str, dict[str, str]],
    historical_repeat_rows: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for finalist_id in finalist_ids:
        finalist_repeat = [row for row in repeat_summary_rows if str(row.get("finalist_id")) == finalist_id]
        finalist_seq = [row for row in per_sequence_rows if str(row.get("finalist_id")) == finalist_id]
        row: dict[str, object] = {
            "finalist_id": finalist_id,
            "repeat_count": len(finalist_repeat),
            "full_mean_of_repeat_means_ate_rmse_associated": f"{_mean([_safe_float(item.get('full_mean_ate_rmse_associated')) for item in finalist_repeat]):.6f}",
            "full_median_of_repeat_means_ate_rmse_associated": f"{_median([_safe_float(item.get('full_mean_ate_rmse_associated')) for item in finalist_repeat]):.6f}",
            "full_mean_of_sequence_medians_ate_rmse_associated": f"{_mean([_safe_float(item.get('median_ate_rmse_associated')) for item in finalist_seq]):.6f}",
            "average_wins_vs_frozen_dpvo_median": f"{_mean([_safe_float(item.get('wins_vs_frozen_dpvo_median')) for item in finalist_repeat]):.6f}",
            "average_wins_vs_frozen_old_focus071_median": f"{_mean([_safe_float(item.get('wins_vs_frozen_old_focus071_median')) for item in finalist_repeat]):.6f}",
            "per_sequence_median_wins_vs_dpvo": sum(1 for item in finalist_seq if item.get("winner_vs_dpvo") == "finalist"),
            "per_sequence_median_wins_vs_old_focus071": sum(1 for item in finalist_seq if item.get("winner_vs_old_focus071") == "finalist"),
        }
        for family in FAMILY_ORDER:
            family_repeat = finalist_repeat
            family_seq = [item for item in finalist_seq if str(item.get("family")) == family]
            row[f"{family}_mean_of_repeat_means_ate_rmse_associated"] = f"{_mean([_safe_float(item.get(f'{family}_mean_ate_rmse_associated')) for item in family_repeat]):.6f}"
            row[f"{family}_median_of_repeat_means_ate_rmse_associated"] = f"{_median([_safe_float(item.get(f'{family}_mean_ate_rmse_associated')) for item in family_repeat]):.6f}"
            row[f"{family}_mean_of_sequence_medians_ate_rmse_associated"] = f"{_mean([_safe_float(item.get('median_ate_rmse_associated')) for item in family_seq]):.6f}"
        dpvo_hist = historical_method_rows.get("dpvo_native", {})
        old_hist = historical_method_rows.get("focus071_best", {})
        row["delta_vs_dpvo_mean_of_sequence_medians"] = (
            f"{_safe_float(row['full_mean_of_sequence_medians_ate_rmse_associated']) - _safe_float(dpvo_hist.get('full_mean_of_sequence_medians_ate_rmse_associated')):+.6f}"
        )
        row["delta_vs_old_focus071_mean_of_sequence_medians"] = (
            f"{_safe_float(row['full_mean_of_sequence_medians_ate_rmse_associated']) - _safe_float(old_hist.get('full_mean_of_sequence_medians_ate_rmse_associated')):+.6f}"
        )
        dpvo_repeat_vals = [_safe_float(item.get("full_mean_ate_rmse_associated")) for item in historical_repeat_rows.get("dpvo_native", [])]
        old_repeat_vals = [_safe_float(item.get("full_mean_ate_rmse_associated")) for item in historical_repeat_rows.get("focus071_best", [])]
        row["delta_vs_dpvo_mean_of_repeat_means"] = (
            f"{_safe_float(row['full_mean_of_repeat_means_ate_rmse_associated']) - _mean(dpvo_repeat_vals):+.6f}"
        )
        row["delta_vs_old_focus071_mean_of_repeat_means"] = (
            f"{_safe_float(row['full_mean_of_repeat_means_ate_rmse_associated']) - _mean(old_repeat_vals):+.6f}"
        )
        output.append(row)
    output.sort(key=lambda item: (_safe_float(item.get("full_mean_of_repeat_means_ate_rmse_associated")), -_safe_float(item.get("average_wins_vs_frozen_dpvo_median"))))
    return output


def _method_comparison_markdown(
    *,
    method_rows: list[dict[str, object]],
    historical_method_rows: dict[str, dict[str, str]],
    historical_repeat_rows: dict[str, list[dict[str, str]]],
) -> str:
    best_by_mean = min(method_rows, key=lambda row: _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated")))
    best_by_wins = max(method_rows, key=lambda row: _safe_float(row.get("average_wins_vs_frozen_dpvo_median")))
    dpvo_hist = historical_method_rows.get("dpvo_native", {})
    old_hist = historical_method_rows.get("focus071_best", {})
    dpvo_repeat_mean = _mean([_safe_float(item.get("full_mean_ate_rmse_associated")) for item in historical_repeat_rows.get("dpvo_native", [])])
    old_repeat_mean = _mean([_safe_float(item.get("full_mean_ate_rmse_associated")) for item in historical_repeat_rows.get("focus071_best", [])])
    lines = [
        "# Dual Finalists Freiburg 5x",
        "",
        "| Finalist | Mean of repeat means | Median of repeat means | Mean of per-sequence medians | Avg wins vs DPVO median | Avg wins vs old Focus071 median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['finalist_id']}` | {row['full_mean_of_repeat_means_ate_rmse_associated']} | "
            f"{row['full_median_of_repeat_means_ate_rmse_associated']} | {row['full_mean_of_sequence_medians_ate_rmse_associated']} | "
            f"{row['average_wins_vs_frozen_dpvo_median']} | {row['average_wins_vs_frozen_old_focus071_median']} |"
        )
    lines.extend(
        [
            "",
            "## Historical Baselines",
            "",
            f"- DPVO mean of repeat means: `{dpvo_repeat_mean:.6f}`",
            f"- Old Focus071 mean of repeat means: `{old_repeat_mean:.6f}`",
            f"- DPVO mean of per-sequence medians: `{_safe_float(dpvo_hist.get('full_mean_of_sequence_medians_ate_rmse_associated')):.6f}`",
            f"- Old Focus071 mean of per-sequence medians: `{_safe_float(old_hist.get('full_mean_of_sequence_medians_ate_rmse_associated')):.6f}`",
            "",
            f"Best by mean of repeat means: `{best_by_mean['finalist_id']}`",
            f"Best by average wins vs frozen DPVO median: `{best_by_wins['finalist_id']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_dual_finalists_benchmark(
    *,
    benchmark_root: Path,
    finalist_ids: list[str],
    expected_sequences: list[str],
    baseline_per_sequence_path: Path,
    historical_method_comparison_path: Path,
    historical_repeat_summary_path: Path,
    summary_dir: Path | None = None,
    repeats: int = 5,
) -> dict[str, Path]:
    summary_root = summary_dir or (benchmark_root / "summary")
    baseline_assoc = _read_frozen_baselines(
        baseline_per_sequence_path,
        expected_sequences=expected_sequences,
    )
    historical_method_rows = _read_historical_method_rows(historical_method_comparison_path)
    historical_repeat_rows = _read_historical_repeat_summary(historical_repeat_summary_path)

    repeat_rows_by_finalist: dict[str, list[list[dict[str, str]]]] = {}
    repeat_summary_rows: list[dict[str, object]] = []
    for finalist_id in finalist_ids:
        finalist_rows: list[list[dict[str, str]]] = []
        for repeat_idx in range(1, int(repeats) + 1):
            repeat_id = f"repeat_{repeat_idx:02d}"
            csv_path = benchmark_root / finalist_id / repeat_id / "dpvo_style_metrics_summary.csv"
            rows = _read_csv_rows(csv_path)
            _validate_repeat_rows(rows, expected_sequences=expected_sequences, csv_path=csv_path)
            finalist_rows.append(rows)
            repeat_summary_rows.append(
                _repeat_summary_row(
                    finalist_id=finalist_id,
                    repeat_id=repeat_id,
                    rows=rows,
                    baseline_dpvo=baseline_assoc["dpvo_native"],
                    baseline_old_focus071=baseline_assoc["focus071_best"],
                )
            )
        repeat_rows_by_finalist[finalist_id] = finalist_rows

    per_sequence_rows = _per_sequence_medians(
        finalist_ids=finalist_ids,
        repeat_rows=repeat_rows_by_finalist,
        expected_sequences=expected_sequences,
        baseline_dpvo=baseline_assoc["dpvo_native"],
        baseline_old_focus071=baseline_assoc["focus071_best"],
    )
    method_rows = _method_comparison_rows(
        finalist_ids=finalist_ids,
        repeat_summary_rows=repeat_summary_rows,
        per_sequence_rows=per_sequence_rows,
        historical_method_rows=historical_method_rows,
        historical_repeat_rows=historical_repeat_rows,
    )

    repeat_summary_path = summary_root / "repeat_summary.csv"
    per_sequence_path = summary_root / "per_sequence_median.csv"
    method_comparison_path = summary_root / "method_comparison.csv"
    method_comparison_md_path = summary_root / "method_comparison.md"

    _write_csv(
        repeat_summary_path,
        repeat_summary_rows,
        [
            "finalist_id",
            "repeat_id",
            "full_mean_ate_rmse",
            "full_mean_ate_rmse_associated",
            "full_mean_coverage",
            "freiburg1_mean_ate_rmse",
            "freiburg1_mean_ate_rmse_associated",
            "freiburg1_mean_coverage",
            "freiburg2_mean_ate_rmse",
            "freiburg2_mean_ate_rmse_associated",
            "freiburg2_mean_coverage",
            "freiburg3_mean_ate_rmse",
            "freiburg3_mean_ate_rmse_associated",
            "freiburg3_mean_coverage",
            "wins_vs_frozen_dpvo_median",
            "losses_vs_frozen_dpvo_median",
            "ties_vs_frozen_dpvo_median",
            "wins_vs_frozen_old_focus071_median",
            "losses_vs_frozen_old_focus071_median",
            "ties_vs_frozen_old_focus071_median",
        ],
    )
    _write_csv(
        per_sequence_path,
        per_sequence_rows,
        [
            "finalist_id",
            "sequence",
            "family",
            "median_ate_rmse",
            "median_ate_rmse_associated",
            "median_coverage",
            "baseline_dpvo_assoc_median",
            "baseline_old_focus071_assoc_median",
            "winner_vs_dpvo",
            "winner_vs_old_focus071",
        ],
    )
    _write_csv(
        method_comparison_path,
        method_rows,
        [
            "finalist_id",
            "repeat_count",
            "full_mean_of_repeat_means_ate_rmse_associated",
            "full_median_of_repeat_means_ate_rmse_associated",
            "full_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg1_mean_of_repeat_means_ate_rmse_associated",
            "freiburg1_median_of_repeat_means_ate_rmse_associated",
            "freiburg1_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg2_mean_of_repeat_means_ate_rmse_associated",
            "freiburg2_median_of_repeat_means_ate_rmse_associated",
            "freiburg2_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg3_mean_of_repeat_means_ate_rmse_associated",
            "freiburg3_median_of_repeat_means_ate_rmse_associated",
            "freiburg3_mean_of_sequence_medians_ate_rmse_associated",
            "average_wins_vs_frozen_dpvo_median",
            "average_wins_vs_frozen_old_focus071_median",
            "per_sequence_median_wins_vs_dpvo",
            "per_sequence_median_wins_vs_old_focus071",
            "delta_vs_dpvo_mean_of_repeat_means",
            "delta_vs_old_focus071_mean_of_repeat_means",
            "delta_vs_dpvo_mean_of_sequence_medians",
            "delta_vs_old_focus071_mean_of_sequence_medians",
        ],
    )
    _write_text(
        method_comparison_md_path,
        _method_comparison_markdown(
            method_rows=method_rows,
            historical_method_rows=historical_method_rows,
            historical_repeat_rows=historical_repeat_rows,
        ),
    )
    return {
        "repeat_summary": repeat_summary_path,
        "per_sequence_median": per_sequence_path,
        "method_comparison": method_comparison_path,
        "method_comparison_md": method_comparison_md_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate a dual-finalist Freiburg 5x benchmark.")
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--finalist-ids", required=True, help="Comma-separated finalist directory names")
    ap.add_argument("--expected-sequences", required=True, help="Comma-separated Freiburg sequence names")
    ap.add_argument("--baseline-per-sequence", required=True)
    ap.add_argument("--historical-method-comparison", required=True)
    ap.add_argument("--historical-repeat-summary", required=True)
    ap.add_argument("--summary-dir", default="")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    outputs = aggregate_dual_finalists_benchmark(
        benchmark_root=Path(args.benchmark_root).expanduser().resolve(),
        finalist_ids=[item.strip() for item in str(args.finalist_ids).split(",") if item.strip()],
        expected_sequences=[item.strip() for item in str(args.expected_sequences).split(",") if item.strip()],
        baseline_per_sequence_path=Path(args.baseline_per_sequence).expanduser().resolve(),
        historical_method_comparison_path=Path(args.historical_method_comparison).expanduser().resolve(),
        historical_repeat_summary_path=Path(args.historical_repeat_summary).expanduser().resolve(),
        summary_dir=(Path(args.summary_dir).expanduser().resolve() if str(args.summary_dir).strip() else None),
        repeats=int(args.repeats),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
