from __future__ import annotations

import argparse
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.focus071_tumwin_finalists import (
    ALLOWED_STATUSES,
    _read_csv_rows,
    _run_eval as _run_tum_eval,
    _safe_float,
    _write_csv,
    _write_text,
)
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import (
    _enumerate_freiburg_sequences,
    _gpu_heavy_process_lines,
)


REPEATS = 5
TUM_FRONTEND_MODE = "dino_hybrid"
TUM_STRIDE = 4
TUM_BACKEND_THRESH = 32.0
TUM_IMAGE_HEIGHT = 240
TUM_IMAGE_WIDTH = 320
TUM_DPVO_OPTS = (
    "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 "
    "OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"
)

FAMILY_ORDER = ("freiburg1", "freiburg2", "freiburg3")

FROZEN_MULTISCALE25_METHOD_ID = "multiscale_32x4_v1_hybrid75_25"
FROZEN_DPVO_METHOD_ID = "dpvo_native"
FROZEN_FOCUS071_METHOD_ID = "focus071_best"

DEFAULT_METHOD_IDS = [
    "pure100_focus071_unfreeze1_convgru_v1",
    "pure100_rescue_lowdedupe_v1",
    "multiscale_pure100_semgrid88_v1",
    "pure100_focus071_ctrl_v2",
    "multiscale_pure100_transfer_ctrl_v1",
    "pure100_rescue_gram04_v1",
    "pure100_rescue_register_score_v1",
    "pure100_focus071_randombackfill35_v1",
]

FALLBACK_METHOD_IDS = [
    "pure100_rescue_widepool_v1",
    "pure100_focus071_register_fused_v1",
]

RATIONALE_TAGS = {
    "pure100_focus071_unfreeze1_convgru_v1": "top3_current_best_forced_terminal",
    "pure100_rescue_lowdedupe_v1": "top3_rescue_family_best",
    "multiscale_pure100_semgrid88_v1": "top3_multiscale_family_best",
    "pure100_focus071_ctrl_v2": "focus071_control",
    "multiscale_pure100_transfer_ctrl_v1": "multiscale_control",
    "pure100_rescue_gram04_v1": "gram_anchor_ablation",
    "pure100_rescue_register_score_v1": "register_context_ablation",
    "pure100_focus071_randombackfill35_v1": "sampler_random_backfill_ablation",
    "pure100_rescue_widepool_v1": "fallback_rescue_widepool",
    "pure100_focus071_register_fused_v1": "fallback_focus071_register_fused",
}


@dataclass(frozen=True)
class ResolvedMethod:
    requested_method_id: str
    method_id: str
    checkpoint_path: Path
    config_path: Path
    selection_score: float
    rationale_tag: str
    was_fallback: bool
    replacement_reason: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    return {
        "leaderboard": subtree_root / "runs" / "sweeps" / "dino_dpvo_pure100_tum30_recovery_relaxed_v2" / "leaderboard_dev.csv",
        "dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "multiscale_baseline_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_arch_dual_finalists_5x_v1",
        "dpvo_focus_baseline_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6",
        "output_root": subtree_root / "runs" / "eval" / "dino_dpvo_pure100_tum30_recovery_relaxed_v2_tum_ablation_5x_v1",
    }


def _parse_method_ids(text: str) -> list[str]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one method id")
    return items


def _family_for_sequence(sequence: str) -> str:
    seq = str(sequence).strip()
    for family in FAMILY_ORDER:
        if seq.startswith(family + "_"):
            return family
    raise ValueError(f"Unsupported Freiburg sequence: {sequence}")


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


def _format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def _select_score(row: dict[str, str]) -> float:
    score = _safe_float(row.get("best_pure_tum_score"))
    if math.isfinite(score):
        return score
    return _safe_float(row.get("best_assoc"))


def _validate_leaderboard_candidate(row: dict[str, str]) -> tuple[Path, Path]:
    checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip()).expanduser().resolve()
    config_path = Path(str(row.get("config_path", "")).strip()).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    return checkpoint_path, config_path


def _resolve_shortlist(
    *,
    leaderboard_path: Path,
    requested_method_ids: list[str],
    fallback_method_ids: list[str],
) -> list[ResolvedMethod]:
    rows = _read_csv_rows(leaderboard_path)
    rows_by_run = {
        str(row.get("run_id", "")).strip(): row
        for row in rows
        if str(row.get("run_id", "")).strip()
    }
    selected_method_ids: set[str] = set()
    available_fallbacks = [item for item in fallback_method_ids if item not in requested_method_ids]
    resolved: list[ResolvedMethod] = []

    for requested_method_id in requested_method_ids:
        chosen_method_id = requested_method_id
        chosen_row = rows_by_run.get(chosen_method_id)
        replacement_reason = ""
        was_fallback = False
        validation_error: Exception | None = None
        checkpoint_path: Path | None = None
        config_path: Path | None = None

        if chosen_row is not None:
            try:
                checkpoint_path, config_path = _validate_leaderboard_candidate(chosen_row)
            except Exception as exc:
                validation_error = exc
                chosen_row = None

        while chosen_row is None:
            if not available_fallbacks:
                if validation_error is not None:
                    raise RuntimeError(
                        f"Unable to resolve {requested_method_id}: {validation_error}"
                    ) from validation_error
                raise RuntimeError(
                    f"Unable to resolve {requested_method_id}: missing leaderboard row and no fallbacks remain"
                )
            fallback_method_id = available_fallbacks.pop(0)
            if fallback_method_id in selected_method_ids:
                continue
            fallback_row = rows_by_run.get(fallback_method_id)
            if fallback_row is None:
                continue
            try:
                checkpoint_path, config_path = _validate_leaderboard_candidate(fallback_row)
            except Exception:
                continue
            chosen_method_id = fallback_method_id
            chosen_row = fallback_row
            replacement_reason = (
                str(validation_error) if validation_error is not None else "missing leaderboard candidate"
            )
            was_fallback = True

        if chosen_method_id in selected_method_ids:
            raise RuntimeError(f"Resolved duplicate method in shortlist: {chosen_method_id}")
        assert chosen_row is not None
        assert checkpoint_path is not None
        assert config_path is not None
        selected_method_ids.add(chosen_method_id)
        resolved.append(
            ResolvedMethod(
                requested_method_id=requested_method_id,
                method_id=chosen_method_id,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                selection_score=_select_score(chosen_row),
                rationale_tag=RATIONALE_TAGS.get(chosen_method_id, "ablation"),
                was_fallback=was_fallback,
                replacement_reason=replacement_reason,
            )
        )
    return resolved


def _assert_idle_or_raise() -> None:
    active = _gpu_heavy_process_lines(exclude_pid=None)
    filtered = [
        line
        for line in active
        if "pure100_recovery_tum_ablation_5x" not in line
    ]
    if filtered:
        raise RuntimeError(
            "Refusing to start because another GPU-heavy job is active:\n"
            + "\n".join(filtered[:10])
        )


def _repeat_dir(*, benchmark_root: Path, method_id: str, repeat_idx: int) -> Path:
    return benchmark_root / method_id / f"repeat_{repeat_idx:02d}"


def _validate_repeat_dir(repeat_dir: Path, *, expected_sequences: list[str]) -> None:
    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
    rows = _read_csv_rows(csv_path)
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


def _load_repeat_rows(
    *,
    benchmark_root: Path,
    method_id: str,
    expected_sequences: list[str],
    repeats: int,
) -> list[list[dict[str, str]]]:
    rows_by_repeat: list[list[dict[str, str]]] = []
    for repeat_idx in range(1, int(repeats) + 1):
        repeat_dir = _repeat_dir(benchmark_root=benchmark_root, method_id=method_id, repeat_idx=repeat_idx)
        _validate_repeat_dir(repeat_dir, expected_sequences=expected_sequences)
        rows_by_repeat.append(_read_csv_rows(repeat_dir / "dpvo_style_metrics_summary.csv"))
    return rows_by_repeat


def _wins_against_baseline(
    rows: list[dict[str, str]],
    *,
    baseline_assoc: dict[str, float],
    method_id: str,
    baseline_method_id: str,
) -> tuple[int, int, int]:
    if method_id == baseline_method_id:
        return 0, 0, len(rows)
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


def _baseline_assoc_from_repeat_rows(
    *,
    repeat_rows_by_method: dict[str, list[list[dict[str, str]]]],
    method_id: str,
    expected_sequences: list[str],
) -> dict[str, float]:
    rows_by_repeat = repeat_rows_by_method[method_id]
    output: dict[str, float] = {}
    for sequence in expected_sequences:
        seq_rows = [
            next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
            for rows in rows_by_repeat
        ]
        output[sequence] = _median([_safe_float(row.get("ate_rmse_associated")) for row in seq_rows])
    return output


def _per_sequence_rows(
    *,
    method_ids: list[str],
    method_source: dict[str, str],
    repeat_rows_by_method: dict[str, list[list[dict[str, str]]]],
    expected_sequences: list[str],
    baseline_dpvo_assoc: dict[str, float],
    baseline_multiscale_assoc: dict[str, float],
    baseline_focus071_assoc: dict[str, float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method_id in method_ids:
        rows_by_repeat = repeat_rows_by_method[method_id]
        for sequence in expected_sequences:
            seq_rows = [
                next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
                for rows in rows_by_repeat
            ]
            assoc = _median([_safe_float(row.get("ate_rmse_associated")) for row in seq_rows])
            dpvo_assoc = baseline_dpvo_assoc[sequence]
            multiscale_assoc = baseline_multiscale_assoc[sequence]
            focus071_assoc = baseline_focus071_assoc[sequence]

            def _winner(value: float, baseline_value: float, baseline_method_id: str) -> str:
                if method_id == baseline_method_id:
                    return "tie"
                if value < baseline_value:
                    return method_id
                if baseline_value < value:
                    return baseline_method_id
                return "tie"

            output.append(
                {
                    "method_id": method_id,
                    "method_source": method_source[method_id],
                    "sequence": sequence,
                    "family": _family_for_sequence(sequence),
                    "median_ate_rmse": _format_float(_median([_safe_float(row.get("ate_rmse")) for row in seq_rows])),
                    "median_ate_rmse_associated": _format_float(assoc),
                    "median_coverage": _format_float(_median([_safe_float(row.get("coverage")) for row in seq_rows])),
                    "baseline_dpvo_assoc_median": _format_float(dpvo_assoc),
                    "baseline_multiscale25_assoc_median": _format_float(multiscale_assoc),
                    "baseline_focus071_assoc_median": _format_float(focus071_assoc),
                    "winner_vs_dpvo": _winner(assoc, dpvo_assoc, FROZEN_DPVO_METHOD_ID),
                    "winner_vs_multiscale25": _winner(assoc, multiscale_assoc, FROZEN_MULTISCALE25_METHOD_ID),
                    "winner_vs_focus071": _winner(assoc, focus071_assoc, FROZEN_FOCUS071_METHOD_ID),
                }
            )
    return output


def _repeat_summary_rows(
    *,
    method_ids: list[str],
    method_source: dict[str, str],
    repeat_rows_by_method: dict[str, list[list[dict[str, str]]]],
    baseline_dpvo_assoc: dict[str, float],
    baseline_multiscale_assoc: dict[str, float],
    baseline_focus071_assoc: dict[str, float],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for method_id in method_ids:
        for repeat_idx, rows in enumerate(repeat_rows_by_method[method_id], start=1):
            by_family = {
                family: [row for row in rows if _family_for_sequence(str(row.get("sequence", ""))) == family]
                for family in FAMILY_ORDER
            }
            wins_dpvo, losses_dpvo, ties_dpvo = _wins_against_baseline(
                rows,
                baseline_assoc=baseline_dpvo_assoc,
                method_id=method_id,
                baseline_method_id=FROZEN_DPVO_METHOD_ID,
            )
            wins_multiscale, losses_multiscale, ties_multiscale = _wins_against_baseline(
                rows,
                baseline_assoc=baseline_multiscale_assoc,
                method_id=method_id,
                baseline_method_id=FROZEN_MULTISCALE25_METHOD_ID,
            )
            wins_focus071, losses_focus071, ties_focus071 = _wins_against_baseline(
                rows,
                baseline_assoc=baseline_focus071_assoc,
                method_id=method_id,
                baseline_method_id=FROZEN_FOCUS071_METHOD_ID,
            )
            row_out: dict[str, object] = {
                "method_id": method_id,
                "method_source": method_source[method_id],
                "repeat_id": f"repeat_{repeat_idx:02d}",
                "full_mean_ate_rmse": _format_float(_mean([_safe_float(row.get("ate_rmse")) for row in rows])),
                "full_mean_ate_rmse_associated": _format_float(_mean([_safe_float(row.get("ate_rmse_associated")) for row in rows])),
                "full_mean_coverage": _format_float(_mean([_safe_float(row.get("coverage")) for row in rows])),
                "wins_vs_frozen_dpvo_median": wins_dpvo,
                "losses_vs_frozen_dpvo_median": losses_dpvo,
                "ties_vs_frozen_dpvo_median": ties_dpvo,
                "wins_vs_frozen_multiscale25_median": wins_multiscale,
                "losses_vs_frozen_multiscale25_median": losses_multiscale,
                "ties_vs_frozen_multiscale25_median": ties_multiscale,
                "wins_vs_frozen_focus071_median": wins_focus071,
                "losses_vs_frozen_focus071_median": losses_focus071,
                "ties_vs_frozen_focus071_median": ties_focus071,
            }
            for family in FAMILY_ORDER:
                family_rows = by_family[family]
                row_out[f"{family}_mean_ate_rmse"] = _format_float(
                    _mean([_safe_float(row.get("ate_rmse")) for row in family_rows])
                )
                row_out[f"{family}_mean_ate_rmse_associated"] = _format_float(
                    _mean([_safe_float(row.get("ate_rmse_associated")) for row in family_rows])
                )
                row_out[f"{family}_mean_coverage"] = _format_float(
                    _mean([_safe_float(row.get("coverage")) for row in family_rows])
                )
            rows_out.append(row_out)
    return rows_out


def _method_comparison_rows(
    *,
    method_ids: list[str],
    method_source: dict[str, str],
    repeat_summary_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    method_rows: list[dict[str, object]] = []
    for method_id in method_ids:
        method_repeat = [row for row in repeat_summary_rows if str(row.get("method_id")) == method_id]
        method_seq = [row for row in per_sequence_rows if str(row.get("method_id")) == method_id]
        row: dict[str, object] = {
            "method_id": method_id,
            "method_source": method_source[method_id],
            "repeat_count": len(method_repeat),
            "full_mean_of_repeat_means_ate_rmse_associated": _format_float(
                _mean([_safe_float(item.get("full_mean_ate_rmse_associated")) for item in method_repeat])
            ),
            "full_median_of_repeat_means_ate_rmse_associated": _format_float(
                _median([_safe_float(item.get("full_mean_ate_rmse_associated")) for item in method_repeat])
            ),
            "full_mean_of_sequence_medians_ate_rmse_associated": _format_float(
                _mean([_safe_float(item.get("median_ate_rmse_associated")) for item in method_seq])
            ),
            "average_wins_vs_frozen_dpvo_median": _format_float(
                _mean([_safe_float(item.get("wins_vs_frozen_dpvo_median")) for item in method_repeat])
            ),
            "average_wins_vs_frozen_multiscale25_median": _format_float(
                _mean([_safe_float(item.get("wins_vs_frozen_multiscale25_median")) for item in method_repeat])
            ),
            "average_wins_vs_frozen_focus071_median": _format_float(
                _mean([_safe_float(item.get("wins_vs_frozen_focus071_median")) for item in method_repeat])
            ),
            "per_sequence_median_wins_vs_dpvo": sum(1 for item in method_seq if item.get("winner_vs_dpvo") == method_id),
            "per_sequence_median_wins_vs_multiscale25": sum(1 for item in method_seq if item.get("winner_vs_multiscale25") == method_id),
            "per_sequence_median_wins_vs_focus071": sum(1 for item in method_seq if item.get("winner_vs_focus071") == method_id),
        }
        for family in FAMILY_ORDER:
            family_repeat = method_repeat
            family_seq = [item for item in method_seq if str(item.get("family")) == family]
            row[f"{family}_mean_of_repeat_means_ate_rmse_associated"] = _format_float(
                _mean([_safe_float(item.get(f"{family}_mean_ate_rmse_associated")) for item in family_repeat])
            )
            row[f"{family}_median_of_repeat_means_ate_rmse_associated"] = _format_float(
                _median([_safe_float(item.get(f"{family}_mean_ate_rmse_associated")) for item in family_repeat])
            )
            row[f"{family}_mean_of_sequence_medians_ate_rmse_associated"] = _format_float(
                _mean([_safe_float(item.get("median_ate_rmse_associated")) for item in family_seq])
            )
        method_rows.append(row)

    by_method = {str(row.get("method_id")): row for row in method_rows}
    multiscale_row = by_method[FROZEN_MULTISCALE25_METHOD_ID]
    dpvo_row = by_method[FROZEN_DPVO_METHOD_ID]
    focus071_row = by_method[FROZEN_FOCUS071_METHOD_ID]

    for row in method_rows:
        row["delta_vs_frozen_multiscale25_mean_of_repeat_means"] = _format_float(
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated"))
            - _safe_float(multiscale_row.get("full_mean_of_repeat_means_ate_rmse_associated"))
        )
        row["delta_vs_frozen_multiscale25_mean_of_sequence_medians"] = _format_float(
            _safe_float(row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
            - _safe_float(multiscale_row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
        )
        row["delta_vs_frozen_dpvo_mean_of_repeat_means"] = _format_float(
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated"))
            - _safe_float(dpvo_row.get("full_mean_of_repeat_means_ate_rmse_associated"))
        )
        row["delta_vs_frozen_dpvo_mean_of_sequence_medians"] = _format_float(
            _safe_float(row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
            - _safe_float(dpvo_row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
        )
        row["delta_vs_frozen_focus071_mean_of_repeat_means"] = _format_float(
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated"))
            - _safe_float(focus071_row.get("full_mean_of_repeat_means_ate_rmse_associated"))
        )
        row["delta_vs_frozen_focus071_mean_of_sequence_medians"] = _format_float(
            _safe_float(row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
            - _safe_float(focus071_row.get("full_mean_of_sequence_medians_ate_rmse_associated"))
        )

    method_rows.sort(
        key=lambda item: (
            _safe_float(item.get("full_mean_of_repeat_means_ate_rmse_associated")),
            -_safe_float(item.get("average_wins_vs_frozen_multiscale25_median")),
            -_safe_float(item.get("average_wins_vs_frozen_dpvo_median")),
        )
    )
    return method_rows


def _aggregate_benchmark(
    *,
    benchmark_root: Path,
    selected_methods: list[ResolvedMethod],
    expected_sequences: list[str],
    multiscale_baseline_root: Path,
    dpvo_focus_baseline_root: Path,
    repeats: int,
) -> dict[str, Path]:
    selected_method_ids = [item.method_id for item in selected_methods]
    method_ids = selected_method_ids + [
        FROZEN_MULTISCALE25_METHOD_ID,
        FROZEN_DPVO_METHOD_ID,
        FROZEN_FOCUS071_METHOD_ID,
    ]
    method_source = {
        method_id: "selected" for method_id in selected_method_ids
    }
    method_source[FROZEN_MULTISCALE25_METHOD_ID] = "frozen_baseline"
    method_source[FROZEN_DPVO_METHOD_ID] = "frozen_baseline"
    method_source[FROZEN_FOCUS071_METHOD_ID] = "frozen_baseline"

    roots_by_method = {
        method_id: benchmark_root for method_id in selected_method_ids
    }
    roots_by_method[FROZEN_MULTISCALE25_METHOD_ID] = multiscale_baseline_root
    roots_by_method[FROZEN_DPVO_METHOD_ID] = dpvo_focus_baseline_root
    roots_by_method[FROZEN_FOCUS071_METHOD_ID] = dpvo_focus_baseline_root

    repeat_rows_by_method: dict[str, list[list[dict[str, str]]]] = {}
    for method_id in method_ids:
        repeat_rows_by_method[method_id] = _load_repeat_rows(
            benchmark_root=roots_by_method[method_id],
            method_id=method_id,
            expected_sequences=expected_sequences,
            repeats=repeats,
        )

    baseline_dpvo_assoc = _baseline_assoc_from_repeat_rows(
        repeat_rows_by_method=repeat_rows_by_method,
        method_id=FROZEN_DPVO_METHOD_ID,
        expected_sequences=expected_sequences,
    )
    baseline_multiscale_assoc = _baseline_assoc_from_repeat_rows(
        repeat_rows_by_method=repeat_rows_by_method,
        method_id=FROZEN_MULTISCALE25_METHOD_ID,
        expected_sequences=expected_sequences,
    )
    baseline_focus071_assoc = _baseline_assoc_from_repeat_rows(
        repeat_rows_by_method=repeat_rows_by_method,
        method_id=FROZEN_FOCUS071_METHOD_ID,
        expected_sequences=expected_sequences,
    )

    per_sequence_rows = _per_sequence_rows(
        method_ids=method_ids,
        method_source=method_source,
        repeat_rows_by_method=repeat_rows_by_method,
        expected_sequences=expected_sequences,
        baseline_dpvo_assoc=baseline_dpvo_assoc,
        baseline_multiscale_assoc=baseline_multiscale_assoc,
        baseline_focus071_assoc=baseline_focus071_assoc,
    )
    repeat_summary_rows = _repeat_summary_rows(
        method_ids=method_ids,
        method_source=method_source,
        repeat_rows_by_method=repeat_rows_by_method,
        baseline_dpvo_assoc=baseline_dpvo_assoc,
        baseline_multiscale_assoc=baseline_multiscale_assoc,
        baseline_focus071_assoc=baseline_focus071_assoc,
    )
    method_rows = _method_comparison_rows(
        method_ids=method_ids,
        method_source=method_source,
        repeat_summary_rows=repeat_summary_rows,
        per_sequence_rows=per_sequence_rows,
    )

    summary_dir = benchmark_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    repeat_summary_path = summary_dir / "repeat_summary.csv"
    per_sequence_path = summary_dir / "per_sequence_median.csv"
    method_comparison_path = summary_dir / "method_comparison.csv"
    method_comparison_md_path = summary_dir / "method_comparison.md"

    _write_csv(
        repeat_summary_path,
        repeat_summary_rows,
        [
            "method_id",
            "method_source",
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
            "wins_vs_frozen_multiscale25_median",
            "losses_vs_frozen_multiscale25_median",
            "ties_vs_frozen_multiscale25_median",
            "wins_vs_frozen_focus071_median",
            "losses_vs_frozen_focus071_median",
            "ties_vs_frozen_focus071_median",
        ],
    )
    _write_csv(
        per_sequence_path,
        per_sequence_rows,
        [
            "method_id",
            "method_source",
            "sequence",
            "family",
            "median_ate_rmse",
            "median_ate_rmse_associated",
            "median_coverage",
            "baseline_dpvo_assoc_median",
            "baseline_multiscale25_assoc_median",
            "baseline_focus071_assoc_median",
            "winner_vs_dpvo",
            "winner_vs_multiscale25",
            "winner_vs_focus071",
        ],
    )
    _write_csv(
        method_comparison_path,
        method_rows,
        [
            "method_id",
            "method_source",
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
            "per_sequence_median_wins_vs_dpvo",
            "average_wins_vs_frozen_multiscale25_median",
            "per_sequence_median_wins_vs_multiscale25",
            "average_wins_vs_frozen_focus071_median",
            "per_sequence_median_wins_vs_focus071",
            "delta_vs_frozen_multiscale25_mean_of_repeat_means",
            "delta_vs_frozen_multiscale25_mean_of_sequence_medians",
            "delta_vs_frozen_dpvo_mean_of_repeat_means",
            "delta_vs_frozen_dpvo_mean_of_sequence_medians",
            "delta_vs_frozen_focus071_mean_of_repeat_means",
            "delta_vs_frozen_focus071_mean_of_sequence_medians",
        ],
    )

    lines = [
        "# Pure100 Recovery Full TUM 5x",
        "",
        "| Method | Source | Mean of repeat means | Mean of sequence medians | Avg wins vs DPVO | Avg wins vs Multiscale25 | Delta vs Multiscale25 mean | Delta vs Multiscale25 seq medians |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method_id']}` | `{row['method_source']}` | {row['full_mean_of_repeat_means_ate_rmse_associated']} | "
            f"{row['full_mean_of_sequence_medians_ate_rmse_associated']} | {row['average_wins_vs_frozen_dpvo_median']} | "
            f"{row['average_wins_vs_frozen_multiscale25_median']} | {row['delta_vs_frozen_multiscale25_mean_of_repeat_means']} | "
            f"{row['delta_vs_frozen_multiscale25_mean_of_sequence_medians']} |"
        )
    best_row = min(
        [row for row in method_rows if str(row.get("method_source")) == "selected"],
        key=lambda item: _safe_float(item.get("full_mean_of_repeat_means_ate_rmse_associated")),
    )
    lines.extend(
        [
            "",
            f"Best selected by mean of repeat means: `{best_row['method_id']}`",
            "",
        ]
    )
    _write_text(method_comparison_md_path, "\n".join(lines) + "\n")
    return {
        "repeat_summary": repeat_summary_path,
        "per_sequence_median": per_sequence_path,
        "method_comparison": method_comparison_path,
        "method_comparison_md": method_comparison_md_path,
    }


def _write_shortlist_manifest(output_root: Path, methods: list[ResolvedMethod]) -> None:
    _write_csv(
        output_root / "shortlist_manifest.csv",
        [
            {
                "requested_method_id": item.requested_method_id,
                "method_id": item.method_id,
                "checkpoint_path": str(item.checkpoint_path),
                "config_path": str(item.config_path),
                "selection_score": _format_float(item.selection_score),
                "rationale_tag": item.rationale_tag,
                "was_fallback": int(item.was_fallback),
                "replacement_reason": item.replacement_reason,
            }
            for item in methods
        ],
        [
            "requested_method_id",
            "method_id",
            "checkpoint_path",
            "config_path",
            "selection_score",
            "rationale_tag",
            "was_fallback",
            "replacement_reason",
        ],
    )
    lines = [
        "# Pure100 Recovery TUM Ablation Shortlist",
        "",
        "| Requested | Resolved | Score | Rationale | Fallback |",
        "|---|---|---:|---|---|",
    ]
    for item in methods:
        lines.append(
            f"| `{item.requested_method_id}` | `{item.method_id}` | {_format_float(item.selection_score)} | "
            f"`{item.rationale_tag}` | `{int(item.was_fallback)}` |"
        )
        if item.replacement_reason:
            lines.append(f"|  |  |  | replacement_reason: `{item.replacement_reason}` |  |")
    lines.append("")
    _write_text(output_root / "shortlist_manifest.md", "\n".join(lines) + "\n")


def _print_dry_run(
    *,
    methods: list[ResolvedMethod],
    benchmark_root: Path,
    sequences: list[str],
    repeats: int,
    multiscale_baseline_root: Path,
    dpvo_focus_baseline_root: Path,
) -> None:
    print(f"benchmark_root: {benchmark_root}")
    print(f"repeats: {repeats}")
    print(f"sequence_count: {len(sequences)}")
    print(f"method_count: {len(methods)}")
    print(f"multiscale_baseline_root: {multiscale_baseline_root}")
    print(f"dpvo_focus_baseline_root: {dpvo_focus_baseline_root}")
    print("methods:")
    for item in methods:
        print(
            "  "
            + f"{item.method_id} requested={item.requested_method_id} "
            + f"score={_format_float(item.selection_score)} rationale={item.rationale_tag} "
            + f"fallback={int(item.was_fallback)}"
        )
        print(f"    checkpoint={item.checkpoint_path}")
        print(f"    config={item.config_path}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Run a full-TUM 5x ablation pack for the frozen pure100 recovery shortlist."
    )
    ap.add_argument("--leaderboard", default=str(defaults["leaderboard"]))
    ap.add_argument("--method-ids", default=",".join(DEFAULT_METHOD_IDS))
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--multiscale-baseline-root", default=str(defaults["multiscale_baseline_root"]))
    ap.add_argument("--dpvo-focus-baseline-root", default=str(defaults["dpvo_focus_baseline_root"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    leaderboard_path = Path(args.leaderboard).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    multiscale_baseline_root = Path(args.multiscale_baseline_root).expanduser().resolve()
    dpvo_focus_baseline_root = Path(args.dpvo_focus_baseline_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"TUM dataset root not found: {dataset_root}")
    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not dpvo_weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {dpvo_weights}")
    if not dpvo_config.exists():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")
    if not multiscale_baseline_root.exists():
        raise FileNotFoundError(f"Multiscale baseline root not found: {multiscale_baseline_root}")
    if not dpvo_focus_baseline_root.exists():
        raise FileNotFoundError(f"DPVO/focus baseline root not found: {dpvo_focus_baseline_root}")

    requested_method_ids = _parse_method_ids(args.method_ids)
    resolved_methods = _resolve_shortlist(
        leaderboard_path=leaderboard_path,
        requested_method_ids=requested_method_ids,
        fallback_method_ids=FALLBACK_METHOD_IDS,
    )
    expected_sequences = _enumerate_freiburg_sequences(dataset_root)

    if args.dry_run:
        _print_dry_run(
            methods=resolved_methods,
            benchmark_root=output_root,
            sequences=expected_sequences,
            repeats=int(args.repeats),
            multiscale_baseline_root=multiscale_baseline_root,
            dpvo_focus_baseline_root=dpvo_focus_baseline_root,
        )
        return

    if not args.force:
        _assert_idle_or_raise()

    output_root.mkdir(parents=True, exist_ok=True)
    _write_shortlist_manifest(output_root, resolved_methods)
    _write_text(output_root / "frozen_sequences.txt", "\n".join(expected_sequences) + "\n")

    for method in resolved_methods:
        for repeat_idx in range(1, int(args.repeats) + 1):
            repeat_dir = _repeat_dir(
                benchmark_root=output_root,
                method_id=method.method_id,
                repeat_idx=repeat_idx,
            )
            if repeat_dir.exists():
                try:
                    _validate_repeat_dir(repeat_dir, expected_sequences=expected_sequences)
                    print(f"[recovery_tum_ablation_5x] reusing {method.method_id} repeat_{repeat_idx:02d}")
                    continue
                except Exception:
                    shutil.rmtree(repeat_dir)
            print(f"[recovery_tum_ablation_5x] running {method.method_id} repeat_{repeat_idx:02d}")
            _run_tum_eval(
                repo_root=repo_root,
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                sequences=expected_sequences,
                output_dir=repeat_dir,
                frontend_mode=TUM_FRONTEND_MODE,
                frontend_config=method.config_path,
                checkpoint_path=method.checkpoint_path,
                stride=TUM_STRIDE,
                backend_thresh=TUM_BACKEND_THRESH,
                image_height=TUM_IMAGE_HEIGHT,
                image_width=TUM_IMAGE_WIDTH,
                dpvo_opts=TUM_DPVO_OPTS,
            )
            _validate_repeat_dir(repeat_dir, expected_sequences=expected_sequences)

    outputs = _aggregate_benchmark(
        benchmark_root=output_root,
        selected_methods=resolved_methods,
        expected_sequences=expected_sequences,
        multiscale_baseline_root=multiscale_baseline_root,
        dpvo_focus_baseline_root=dpvo_focus_baseline_root,
        repeats=int(args.repeats),
    )
    for key, path in outputs.items():
        print(f"[recovery_tum_ablation_5x] {key}: {path}")
    print(f"[recovery_tum_ablation_5x] benchmark complete: {output_root}")


if __name__ == "__main__":
    main()
