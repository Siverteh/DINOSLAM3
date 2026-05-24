from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ALLOWED_STATUSES = {"ok", "partial_low_coverage", "invalid_trajectory"}
EUROC_SEQUENCE_ORDER = (
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
)
FAMILY_ORDER = ("machine_hall", "vicon_room1", "vicon_room2")


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
    if seq.startswith("MH"):
        return "machine_hall"
    if seq.startswith("V1"):
        return "vicon_room1"
    if seq.startswith("V2"):
        return "vicon_room2"
    raise ValueError(f"Unsupported EuRoC sequence: {sequence}")


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
    bad_rows = [row for row in rows if str(row.get("status", "")).strip() not in ALLOWED_STATUSES]
    if bad_rows:
        raise ValueError(
            f"{csv_path} contains unsupported status rows: "
            + ", ".join(f"{row.get('sequence')}:{row.get('status')}" for row in bad_rows[:5])
        )


def _row_comp_value(row: dict[str, str]) -> float:
    status = str(row.get("status", "")).strip()
    value = _safe_float(row.get("ate_rmse_associated"))
    if status != "ok":
        return math.nan
    return value if math.isfinite(value) else math.nan


def _pairwise_outcome(left_row: dict[str, str], right_row: dict[str, str]) -> str:
    left_value = _row_comp_value(left_row)
    right_value = _row_comp_value(right_row)
    left_ok = math.isfinite(left_value)
    right_ok = math.isfinite(right_value)
    if left_ok and right_ok:
        if left_value < right_value:
            return "left"
        if right_value < left_value:
            return "right"
        return "tie"
    if left_ok and not right_ok:
        return "left"
    if right_ok and not left_ok:
        return "right"
    return "tie"


def _pairwise_counts(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for left_row, right_row in zip(left_rows, right_rows):
        outcome = _pairwise_outcome(left_row, right_row)
        if outcome == "left":
            wins += 1
        elif outcome == "right":
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def _repeat_summary_row(
    *,
    method_id: str,
    repeat_id: str,
    rows: list[dict[str, str]],
    other_rows_by_method: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    family_rows = {
        family: [row for row in rows if _family_for_sequence(str(row.get("sequence", ""))) == family]
        for family in FAMILY_ORDER
    }
    output: dict[str, object] = {
        "method_id": method_id,
        "repeat_id": repeat_id,
        "full_mean_ate_rmse_associated": f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in rows]):.6f}",
        "ok_count": sum(1 for row in rows if str(row.get("status", "")).strip() == "ok"),
        "non_ok_count": sum(1 for row in rows if str(row.get("status", "")).strip() != "ok"),
        "finite_count": sum(1 for row in rows if math.isfinite(_safe_float(row.get("ate_rmse_associated")))),
    }
    for family in FAMILY_ORDER:
        output[f"{family}_mean_ate_rmse_associated"] = f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in family_rows[family]]):.6f}"
    for other_method_id, other_rows in sorted(other_rows_by_method.items()):
        wins, losses, ties = _pairwise_counts(rows, other_rows)
        output[f"wins_vs_{other_method_id}"] = wins
        output[f"losses_vs_{other_method_id}"] = losses
        output[f"ties_vs_{other_method_id}"] = ties
    return output


def _per_sequence_rows(
    *,
    method_ids: list[str],
    repeat_rows: dict[str, list[list[dict[str, str]]]],
    expected_sequences: list[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method_id in method_ids:
        rows_by_repeat = repeat_rows[method_id]
        for sequence in expected_sequences:
            seq_rows = [
                next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
                for rows in rows_by_repeat
            ]
            assoc_median = _median([_safe_float(row.get("ate_rmse_associated")) for row in seq_rows])
            row_out: dict[str, object] = {
                "method_id": method_id,
                "sequence": sequence,
                "family": _family_for_sequence(sequence),
                "median_ate_rmse_associated": f"{assoc_median:.6f}",
                "median_coverage": f"{_median([_safe_float(row.get('coverage')) for row in seq_rows]):.6f}",
                "ok_repeat_count": sum(1 for row in seq_rows if str(row.get("status", "")).strip() == "ok"),
                "non_ok_repeat_count": sum(1 for row in seq_rows if str(row.get("status", "")).strip() != "ok"),
            }
            for other_method_id in method_ids:
                if other_method_id == method_id:
                    continue
                other_seq_rows = [
                    next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
                    for rows in repeat_rows[other_method_id]
                ]
                other_assoc_median = _median([_safe_float(row.get("ate_rmse_associated")) for row in other_seq_rows])
                winner = "tie"
                if math.isfinite(assoc_median) and math.isfinite(other_assoc_median):
                    if assoc_median < other_assoc_median:
                        winner = method_id
                    elif other_assoc_median < assoc_median:
                        winner = other_method_id
                elif math.isfinite(assoc_median):
                    winner = method_id
                elif math.isfinite(other_assoc_median):
                    winner = other_method_id
                row_out[f"winner_vs_{other_method_id}"] = winner
            output.append(row_out)
    return output


def _method_comparison_rows(
    *,
    method_ids: list[str],
    repeat_summary_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method_id in method_ids:
        method_repeat_rows = [row for row in repeat_summary_rows if str(row.get("method_id")) == method_id]
        method_seq_rows = [row for row in per_sequence_rows if str(row.get("method_id")) == method_id]
        row_out: dict[str, object] = {
            "method_id": method_id,
            "repeat_count": len(method_repeat_rows),
            "full_mean_of_repeat_means_ate_rmse_associated": f"{_mean([_safe_float(row.get('full_mean_ate_rmse_associated')) for row in method_repeat_rows]):.6f}",
            "full_median_of_repeat_means_ate_rmse_associated": f"{_median([_safe_float(row.get('full_mean_ate_rmse_associated')) for row in method_repeat_rows]):.6f}",
            "full_mean_of_sequence_medians_ate_rmse_associated": f"{_mean([_safe_float(row.get('median_ate_rmse_associated')) for row in method_seq_rows]):.6f}",
            "average_ok_count": f"{_mean([_safe_float(row.get('ok_count')) for row in method_repeat_rows]):.6f}",
            "average_non_ok_count": f"{_mean([_safe_float(row.get('non_ok_count')) for row in method_repeat_rows]):.6f}",
        }
        pairwise_average_wins: list[float] = []
        pairwise_median_wins_total = 0
        for family in FAMILY_ORDER:
            family_seq_rows = [row for row in method_seq_rows if str(row.get("family")) == family]
            row_out[f"{family}_mean_of_repeat_means_ate_rmse_associated"] = f"{_mean([_safe_float(row.get(f'{family}_mean_ate_rmse_associated')) for row in method_repeat_rows]):.6f}"
            row_out[f"{family}_median_of_repeat_means_ate_rmse_associated"] = f"{_median([_safe_float(row.get(f'{family}_mean_ate_rmse_associated')) for row in method_repeat_rows]):.6f}"
            row_out[f"{family}_mean_of_sequence_medians_ate_rmse_associated"] = f"{_mean([_safe_float(row.get('median_ate_rmse_associated')) for row in family_seq_rows]):.6f}"
        for other_method_id in method_ids:
            if other_method_id == method_id:
                continue
            avg_wins = _mean([_safe_float(row.get(f"wins_vs_{other_method_id}")) for row in method_repeat_rows])
            row_out[f"average_wins_vs_{other_method_id}"] = f"{avg_wins:.6f}"
            row_out[f"average_losses_vs_{other_method_id}"] = f"{_mean([_safe_float(row.get(f'losses_vs_{other_method_id}')) for row in method_repeat_rows]):.6f}"
            row_out[f"per_sequence_median_wins_vs_{other_method_id}"] = sum(
                1 for row in method_seq_rows if str(row.get(f"winner_vs_{other_method_id}")) == method_id
            )
            pairwise_average_wins.append(avg_wins)
            pairwise_median_wins_total += int(row_out[f"per_sequence_median_wins_vs_{other_method_id}"])
        row_out["average_pairwise_wins"] = f"{_mean(pairwise_average_wins):.6f}"
        row_out["per_sequence_median_wins_total"] = pairwise_median_wins_total
        output.append(row_out)
    output.sort(
        key=lambda row: (
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated")),
            -_safe_float(row.get("average_pairwise_wins")),
        )
    )
    return output


def _method_comparison_markdown(method_rows: list[dict[str, object]]) -> str:
    best_by_mean = min(method_rows, key=lambda row: _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated")))
    best_by_wins = max(method_rows, key=lambda row: _safe_float(row.get("average_pairwise_wins")))
    lines = [
        "# EuRoC 3-Method 5x",
        "",
        "| Method | Mean of repeat means | Median of repeat means | Mean of sequence medians | Average pairwise wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method_id']}` | {row['full_mean_of_repeat_means_ate_rmse_associated']} | "
            f"{row['full_median_of_repeat_means_ate_rmse_associated']} | "
            f"{row['full_mean_of_sequence_medians_ate_rmse_associated']} | "
            f"{row['average_pairwise_wins']} |"
        )
    lines.extend(
        [
            "",
            f"Best by mean of repeat means: `{best_by_mean['method_id']}`",
            f"Best by average pairwise wins: `{best_by_wins['method_id']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_euroc_three_method_benchmark(
    *,
    benchmark_root: Path,
    method_ids: list[str],
    expected_sequences: list[str],
    repeats: int,
) -> dict[str, Path]:
    if expected_sequences != list(EUROC_SEQUENCE_ORDER):
        raise ValueError("expected_sequences must exactly match EUROC_SEQUENCE_ORDER")
    repeat_rows: dict[str, list[list[dict[str, str]]]] = {}
    for method_id in method_ids:
        rows_by_repeat: list[list[dict[str, str]]] = []
        for repeat_idx in range(1, int(repeats) + 1):
            csv_path = benchmark_root / method_id / f"repeat_{repeat_idx:02d}" / "dpvo_style_metrics_summary.csv"
            rows = _read_csv_rows(csv_path)
            _validate_repeat_rows(rows, expected_sequences=expected_sequences, csv_path=csv_path)
            rows_by_repeat.append(rows)
        repeat_rows[method_id] = rows_by_repeat

    repeat_summary_rows: list[dict[str, object]] = []
    for repeat_idx in range(1, int(repeats) + 1):
        same_repeat_rows = {
            method_id: repeat_rows[method_id][repeat_idx - 1]
            for method_id in method_ids
        }
        for method_id in method_ids:
            other_rows_by_method = {
                other_method_id: same_repeat_rows[other_method_id]
                for other_method_id in method_ids
                if other_method_id != method_id
            }
            repeat_summary_rows.append(
                _repeat_summary_row(
                    method_id=method_id,
                    repeat_id=f"repeat_{repeat_idx:02d}",
                    rows=same_repeat_rows[method_id],
                    other_rows_by_method=other_rows_by_method,
                )
            )

    per_sequence_rows = _per_sequence_rows(
        method_ids=method_ids,
        repeat_rows=repeat_rows,
        expected_sequences=expected_sequences,
    )
    method_rows = _method_comparison_rows(
        method_ids=method_ids,
        repeat_summary_rows=repeat_summary_rows,
        per_sequence_rows=per_sequence_rows,
    )

    summary_dir = benchmark_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    repeat_fields = [
        "method_id",
        "repeat_id",
        "full_mean_ate_rmse_associated",
        "machine_hall_mean_ate_rmse_associated",
        "vicon_room1_mean_ate_rmse_associated",
        "vicon_room2_mean_ate_rmse_associated",
        "ok_count",
        "non_ok_count",
        "finite_count",
    ]
    for method_id in method_ids:
        for prefix in ("wins", "losses", "ties"):
            for other_method_id in method_ids:
                if other_method_id == method_id:
                    continue
                key = f"{prefix}_vs_{other_method_id}"
                if key not in repeat_fields:
                    repeat_fields.append(key)

    per_sequence_fields = [
        "method_id",
        "sequence",
        "family",
        "median_ate_rmse_associated",
        "median_coverage",
        "ok_repeat_count",
        "non_ok_repeat_count",
    ]
    for method_id in method_ids:
        for other_method_id in method_ids:
            if other_method_id == method_id:
                continue
            key = f"winner_vs_{other_method_id}"
            if key not in per_sequence_fields:
                per_sequence_fields.append(key)

    method_fields = [
        "method_id",
        "repeat_count",
        "full_mean_of_repeat_means_ate_rmse_associated",
        "full_median_of_repeat_means_ate_rmse_associated",
        "full_mean_of_sequence_medians_ate_rmse_associated",
        "machine_hall_mean_of_repeat_means_ate_rmse_associated",
        "machine_hall_median_of_repeat_means_ate_rmse_associated",
        "machine_hall_mean_of_sequence_medians_ate_rmse_associated",
        "vicon_room1_mean_of_repeat_means_ate_rmse_associated",
        "vicon_room1_median_of_repeat_means_ate_rmse_associated",
        "vicon_room1_mean_of_sequence_medians_ate_rmse_associated",
        "vicon_room2_mean_of_repeat_means_ate_rmse_associated",
        "vicon_room2_median_of_repeat_means_ate_rmse_associated",
        "vicon_room2_mean_of_sequence_medians_ate_rmse_associated",
        "average_ok_count",
        "average_non_ok_count",
        "average_pairwise_wins",
        "per_sequence_median_wins_total",
    ]
    for other_method_id in method_ids:
        method_fields.append(f"average_wins_vs_{other_method_id}")
        method_fields.append(f"average_losses_vs_{other_method_id}")
        method_fields.append(f"per_sequence_median_wins_vs_{other_method_id}")
    method_fields = [field for field in method_fields if not field.endswith("_vs_")]

    repeat_path = summary_dir / "repeat_summary.csv"
    per_sequence_path = summary_dir / "per_sequence_median.csv"
    method_path = summary_dir / "method_comparison.csv"
    md_path = summary_dir / "method_comparison.md"
    _write_csv(repeat_path, repeat_summary_rows, repeat_fields)
    _write_csv(per_sequence_path, per_sequence_rows, per_sequence_fields)
    _write_csv(method_path, method_rows, [field for field in method_fields if any(field in row for row in method_rows) or field in {"method_id", "repeat_count"}])
    _write_text(md_path, _method_comparison_markdown(method_rows))
    return {
        "repeat_summary": repeat_path,
        "per_sequence_median": per_sequence_path,
        "method_comparison": method_path,
        "method_comparison_md": md_path,
    }
