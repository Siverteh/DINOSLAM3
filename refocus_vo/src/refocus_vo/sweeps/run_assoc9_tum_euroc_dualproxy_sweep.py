from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from refocus_vo.eval.aggregate_euroc_three_method_5x import (
    EUROC_SEQUENCE_ORDER,
    aggregate_euroc_three_method_benchmark,
)
from refocus_vo.eval.euroc_three_method_5x import (
    LockedMethodSpec as EurocLockedMethodSpec,
    _run_eval as _run_euroc_eval,
    _validate_repeat_dir as _validate_euroc_repeat_dir,
)
from refocus_vo.eval.focus071_tumwin_finalists import (
    _read_csv_rows,
    _run_eval as _run_tum_eval,
)
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import _enumerate_freiburg_sequences
from refocus_vo.eval.sweepbest_tum_euroc_5x import _aggregate_tum_single_method, _validate_tum_repeat_dir
from refocus_vo.sweeps.run_assoc9_cross_dataset_sweep import (
    DatasetEvalSpec,
    DatasetSummary,
    EvalMethodSpec,
    _baseline_methods,
    _build_live_proxy_eval_cfg,
    _candidate_methods,
    _dataset_summary_to_row,
    _evaluate_method_group,
    _format_float,
    _load_dataset_specs,
    _load_live_candidate_proxy_summaries,
    _safe_float,
    _verify_gpu_idle,
    _write_csv,
    _write_text,
)
from refocus_vo.sweeps.run_assoc9_sweep import (
    REPO_ROOT,
    _load_manifest,
    _print_dry_run,
    _resolve_path,
    _run_training_sweep,
)


REPEATS = 5
TUM_FIXED_FRONTEND_MODE = "dino_hybrid"
TUM_FIXED_STRIDE = 4
TUM_FIXED_BACKEND_THRESH = 32.0
TUM_FIXED_IMAGE_HEIGHT = 240
TUM_FIXED_IMAGE_WIDTH = 320
TUM_FIXED_DPVO_OPTS = (
    "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 "
    "OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"
)
EUROC_FIXED_FRONTEND_MODE = "dino_hybrid"
EUROC_FIXED_STRIDE = 4
EUROC_FIXED_BACKEND_THRESH = 32.0
EUROC_FIXED_IMAGE_HEIGHT = 240
EUROC_FIXED_IMAGE_WIDTH = 320
EUROC_FIXED_DPVO_OPTS = (
    "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 "
    "OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"
)


def _mean(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(sum(usable) / len(usable))


def _ratio(value: float, reference: float, *, eps: float = 1e-6) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(reference)):
        return math.inf
    return float(value) / max(abs(float(reference)), float(eps))


def _read_tum_dpvo_sequence_baselines(path: Path, *, expected_sequences: list[str]) -> dict[str, float]:
    rows = _read_csv_rows(path)
    output: dict[str, float] = {}
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        value = _safe_float(row.get("baseline_dpvo_assoc_median"))
        if sequence and math.isfinite(value):
            output[sequence] = value
    missing = [sequence for sequence in expected_sequences if sequence not in output]
    if missing:
        raise RuntimeError(f"TUM frozen DPVO baseline missing sequences: {missing}")
    return output


def _read_euroc_dpvo_sequence_baselines(path: Path, *, expected_sequences: list[str]) -> dict[str, float]:
    rows = _read_csv_rows(path)
    output: dict[str, float] = {}
    for row in rows:
        if str(row.get("method_id", "")).strip() != "dpvo_native_matched":
            continue
        sequence = str(row.get("sequence", "")).strip()
        value = _safe_float(row.get("median_ate_rmse_associated"))
        if sequence and math.isfinite(value):
            output[sequence] = value
    missing = [sequence for sequence in expected_sequences if sequence not in output]
    if missing:
        raise RuntimeError(f"EuRoC frozen DPVO baseline missing sequences: {missing}")
    return output


def _count_wins(rows: list[dict[str, str]], *, baseline_assoc: dict[str, float]) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        assoc = _safe_float(row.get("ate_rmse_associated"))
        baseline = _safe_float(baseline_assoc.get(sequence))
        if not math.isfinite(baseline):
            continue
        if not math.isfinite(assoc):
            losses += 1
        elif assoc < baseline:
            wins += 1
        elif baseline < assoc:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def _single_row(path: Path, *, key_field: str, key_value: str) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if str(row.get(key_field, "")).strip() == key_value:
            return row
    raise RuntimeError(f"Could not find {key_value} in {path}")


def _mean_repeat_metric(path: Path, *, method_field: str, method_id: str, metric_field: str) -> float:
    rows = [
        row
        for row in _read_csv_rows(path)
        if str(row.get(method_field, "")).strip() == method_id
    ]
    if not rows:
        raise RuntimeError(f"No rows for {method_id} in {path}")
    return _mean([_safe_float(row.get(metric_field)) for row in rows])


def _build_frozen_sequence_baselines(
    manifest: dict[str, Any],
    *,
    proxy_datasets: dict[str, DatasetEvalSpec],
) -> dict[str, dict[str, float]]:
    proxy_cfg = manifest.get("proxy_validation", {}) or {}
    frozen_cfg = dict(proxy_cfg.get("frozen_baselines", {}) or {})
    tum_path = _resolve_path(str(frozen_cfg["tum_per_sequence"]), REPO_ROOT)
    euroc_path = _resolve_path(str(frozen_cfg["euroc_per_sequence"]), REPO_ROOT)
    return {
        "tum": _read_tum_dpvo_sequence_baselines(tum_path, expected_sequences=list(proxy_datasets["tum"].sequences)),
        "euroc": _read_euroc_dpvo_sequence_baselines(euroc_path, expected_sequences=list(proxy_datasets["euroc"].sequences)),
    }


def _frozen_sequence_baseline_rows(sequence_baselines: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id, mapping in sorted(sequence_baselines.items()):
        for sequence, assoc in sorted(mapping.items()):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "sequence": sequence,
                    "baseline_ate_rmse_associated": _format_float(_safe_float(assoc)),
                }
            )
    return rows


def _rank_dual_proxy_candidates(
    *,
    candidate_methods: list[EvalMethodSpec],
    summaries_by_method: dict[str, dict[str, DatasetSummary]],
    leaderboard_rows: list[dict[str, str]],
    reference_ids: dict[str, str],
    dataset_weights: dict[str, float],
    win_weights: dict[str, float],
    tum_no_regression_multiplier: float,
    tum_pressure_multiplier: float,
    tum_min_wins_vs_dpvo: int,
    euroc_min_wins_vs_dpvo: int,
) -> list[dict[str, object]]:
    row_by_id = {str(row["run_id"]): row for row in leaderboard_rows}
    reference_tum = summaries_by_method[reference_ids["tum"]]["tum"]
    reference_euroc = summaries_by_method[reference_ids["euroc"]]["euroc"]

    rows: list[dict[str, object]] = []
    for method in candidate_methods:
        tum = summaries_by_method[method.method_id]["tum"]
        euroc = summaries_by_method[method.method_id]["euroc"]
        live_row = row_by_id.get(method.method_id, {})
        tum_wins = _safe_float(live_row.get("best_tum_proxy_wins_vs_dpvo"))
        euroc_wins = _safe_float(live_row.get("best_euroc_proxy_wins_vs_dpvo"))
        tum_win_penalty = (
            max(0.0, float(tum.row_count) - float(tum_wins)) / max(float(tum.row_count), 1.0)
            if math.isfinite(tum_wins) and float(tum.row_count) > 0.0
            else math.inf
        )
        euroc_win_penalty = (
            max(0.0, float(euroc.row_count) - float(euroc_wins)) / max(float(euroc.row_count), 1.0)
            if math.isfinite(euroc_wins) and float(euroc.row_count) > 0.0
            else math.inf
        )

        passes_tum_gate = (
            math.isfinite(tum.mean_ate_rmse_associated)
            and tum.failed_count == 0
            and tum.finite_count == tum.row_count
            and tum.mean_ate_rmse_associated
            <= float(tum_no_regression_multiplier) * float(reference_tum.mean_ate_rmse_associated)
            and (
                not math.isfinite(reference_tum.pressure_mean_ate_rmse_associated)
                or tum.pressure_mean_ate_rmse_associated
                <= float(tum_pressure_multiplier) * float(reference_tum.pressure_mean_ate_rmse_associated)
            )
            and math.isfinite(tum_wins)
            and tum_wins >= float(tum_min_wins_vs_dpvo)
        )
        passes_euroc_validity = euroc.failed_count == 0 and euroc.finite_count == euroc.row_count
        beats_dpvo_on_euroc = (
            math.isfinite(euroc.mean_ate_rmse_associated)
            and euroc.mean_ate_rmse_associated < reference_euroc.mean_ate_rmse_associated
        )
        passes_advancement_gate = bool(
            passes_tum_gate
            and passes_euroc_validity
            and beats_dpvo_on_euroc
            and math.isfinite(euroc_wins)
            and euroc_wins >= float(euroc_min_wins_vs_dpvo)
        )

        weighted_ate = (
            float(dataset_weights.get("tum", 0.65))
            * _ratio(tum.mean_ate_rmse_associated, reference_tum.mean_ate_rmse_associated)
            + float(dataset_weights.get("euroc", 0.35))
            * _ratio(euroc.mean_ate_rmse_associated, reference_euroc.mean_ate_rmse_associated)
        )
        weighted_dual_score = weighted_ate
        if float(win_weights.get("tum", 0.0)) > 0.0:
            weighted_dual_score += float(win_weights.get("tum", 0.0)) * tum_win_penalty
        if float(win_weights.get("euroc", 0.0)) > 0.0:
            weighted_dual_score += float(win_weights.get("euroc", 0.0)) * euroc_win_penalty
        weighted_rpe_trans = (
            float(dataset_weights.get("tum", 0.65))
            * _ratio(tum.mean_rpe_trans_rmse, reference_tum.mean_rpe_trans_rmse)
            + float(dataset_weights.get("euroc", 0.35))
            * _ratio(euroc.mean_rpe_trans_rmse, reference_euroc.mean_rpe_trans_rmse)
        )
        weighted_rpe_rot = (
            float(dataset_weights.get("tum", 0.65))
            * _ratio(tum.mean_rpe_rot_rmse, reference_tum.mean_rpe_rot_rmse)
            + float(dataset_weights.get("euroc", 0.35))
            * _ratio(euroc.mean_rpe_rot_rmse, reference_euroc.mean_rpe_rot_rmse)
        )
        weighted_scale = (
            float(dataset_weights.get("tum", 0.65))
            * _ratio(tum.mean_scale_error_abs_log, reference_tum.mean_scale_error_abs_log)
            + float(dataset_weights.get("euroc", 0.35))
            * _ratio(euroc.mean_scale_error_abs_log, reference_euroc.mean_scale_error_abs_log)
        )
        tum_pressure_score = _ratio(
            tum.pressure_mean_ate_rmse_associated,
            reference_tum.pressure_mean_ate_rmse_associated,
        )

        rows.append(
            {
                "proxy_rank": "",
                "method_id": method.method_id,
                "kind": method.kind,
                "checkpoint_path": str(method.checkpoint_path or ""),
                "config_path": str(method.runtime_config_path or ""),
                "passes_tum_gate": int(bool(passes_tum_gate)),
                "passes_euroc_validity_gate": int(bool(passes_euroc_validity)),
                "beats_dpvo_on_euroc": int(bool(beats_dpvo_on_euroc)),
                "passes_advancement_gate": int(bool(passes_advancement_gate)),
                "tum_mean_ate_rmse_associated": _format_float(tum.mean_ate_rmse_associated),
                "tum_pressure_mean_ate_rmse_associated": _format_float(tum.pressure_mean_ate_rmse_associated),
                "tum_proxy_wins_vs_dpvo": _format_float(tum_wins),
                "euroc_mean_ate_rmse_associated": _format_float(euroc.mean_ate_rmse_associated),
                "euroc_proxy_wins_vs_dpvo": _format_float(euroc_wins),
                "weighted_dual_score": _format_float(weighted_dual_score if passes_advancement_gate else math.inf),
                "weighted_rpe_trans_score": _format_float(weighted_rpe_trans if passes_advancement_gate else math.inf),
                "weighted_rpe_rot_score": _format_float(weighted_rpe_rot if passes_advancement_gate else math.inf),
                "weighted_scale_error_abs_log_score": _format_float(weighted_scale if passes_advancement_gate else math.inf),
                "tum_pressure_score": _format_float(tum_pressure_score if passes_advancement_gate else math.inf),
            }
        )

    rows.sort(
        key=lambda row: (
            0 if int(row["passes_advancement_gate"]) else 1,
            _safe_float(row["weighted_dual_score"]),
            -_safe_float(row["euroc_proxy_wins_vs_dpvo"]),
            _safe_float(row["tum_pressure_score"]),
            _safe_float(row["weighted_rpe_trans_score"]),
            _safe_float(row["weighted_rpe_rot_score"]),
            _safe_float(row["weighted_scale_error_abs_log_score"]),
            -_safe_float(row["tum_proxy_wins_vs_dpvo"]),
            str(row["method_id"]),
        )
    )
    rank = 1
    for row in rows:
        if int(row["passes_advancement_gate"]):
            row["proxy_rank"] = rank
            rank += 1
    return rows


def _dual_proxy_markdown(
    *,
    ranked_rows: list[dict[str, object]],
    baseline_summaries: list[dict[str, object]],
    top_k: int,
) -> str:
    lines = [
        "# TUM + EuRoC Dual-Proxy Leaderboard",
        "",
        "| Rank | Method | TUM gate | EuRoC valid | Beats DPVO on EuRoC | Weighted score | TUM wins | EuRoC wins |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in ranked_rows:
        lines.append(
            f"| {row.get('proxy_rank', '')} | `{row['method_id']}` | {row['passes_tum_gate']} | "
            f"{row['passes_euroc_validity_gate']} | {row['beats_dpvo_on_euroc']} | {row['weighted_dual_score']} | "
            f"{row['tum_proxy_wins_vs_dpvo']} | {row['euroc_proxy_wins_vs_dpvo']} |"
        )
    lines.extend(["", f"## Top {top_k} advancing candidates", ""])
    for row in ranked_rows:
        if not int(row["passes_advancement_gate"]):
            continue
        lines.append(
            f"- `{row['method_id']}`: score `{row['weighted_dual_score']}`, "
            f"TUM `{row['tum_mean_ate_rmse_associated']}`, EuRoC `{row['euroc_mean_ate_rmse_associated']}`"
        )
        if len([line for line in lines if line.startswith("- `")]) >= int(top_k):
            break
    lines.extend(
        [
            "",
            "## Fresh proxy baselines",
            "",
            "| Method | Dataset | Mean ATE | Mean RPE(t) | Mean RPE(r) | Mean scale log error |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_summaries:
        lines.append(
            f"| `{row['method_id']}` | `{row['dataset_id']}` | {row['mean_ate_rmse_associated']} | "
            f"{row['mean_rpe_trans_rmse']} | {row['mean_rpe_rot_rmse']} | {row['mean_scale_error_abs_log']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _top_candidate_methods(
    candidate_methods: list[EvalMethodSpec],
    ranked_rows: list[dict[str, object]],
    top_k: int,
) -> list[EvalMethodSpec]:
    by_id = {method.method_id: method for method in candidate_methods}
    output: list[EvalMethodSpec] = []
    for row in ranked_rows:
        if not int(row["passes_advancement_gate"]):
            continue
        output.append(by_id[str(row["method_id"])])
        if len(output) >= int(top_k):
            break
    return output


def _run_candidate_tum_5x(
    *,
    candidate: EvalMethodSpec,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    benchmark_root: Path,
    sequences: list[str],
    repeats: int,
) -> dict[str, Path]:
    for repeat_idx in range(1, int(repeats) + 1):
        repeat_dir = benchmark_root / candidate.method_id / f"repeat_{repeat_idx:02d}"
        if repeat_dir.exists():
            try:
                _validate_tum_repeat_dir(repeat_dir, expected_sequences=sequences)
                continue
            except Exception:
                shutil.rmtree(repeat_dir)
        _run_tum_eval(
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            sequences=sequences,
            output_dir=repeat_dir,
            frontend_mode=TUM_FIXED_FRONTEND_MODE,
            frontend_config=Path(candidate.runtime_config_path or candidate.frontend_config or ""),
            checkpoint_path=Path(candidate.checkpoint_path or candidate.frontend_checkpoint or ""),
            stride=TUM_FIXED_STRIDE,
            backend_thresh=TUM_FIXED_BACKEND_THRESH,
            image_height=TUM_FIXED_IMAGE_HEIGHT,
            image_width=TUM_FIXED_IMAGE_WIDTH,
            dpvo_opts=TUM_FIXED_DPVO_OPTS,
        )
        _validate_tum_repeat_dir(repeat_dir, expected_sequences=sequences)
    return _aggregate_tum_single_method(
        benchmark_root=benchmark_root,
        method_id=candidate.method_id,
        expected_sequences=sequences,
        repeats=repeats,
    )


def _run_candidate_euroc_5x(
    *,
    candidate: EvalMethodSpec,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    benchmark_root: Path,
    sequences: list[str],
    repeats: int,
) -> dict[str, Path]:
    method = EurocLockedMethodSpec(
        method_id=candidate.method_id,
        frontend_mode=EUROC_FIXED_FRONTEND_MODE,
        checkpoint_path=Path(candidate.checkpoint_path or candidate.frontend_checkpoint or ""),
        config_path=Path(candidate.runtime_config_path or candidate.frontend_config or ""),
        repeat01_source_dir=None,
    )
    for repeat_idx in range(1, int(repeats) + 1):
        repeat_dir = benchmark_root / candidate.method_id / f"repeat_{repeat_idx:02d}"
        if repeat_dir.exists():
            try:
                _validate_euroc_repeat_dir(repeat_dir, expected_sequences=sequences)
                continue
            except Exception:
                shutil.rmtree(repeat_dir)
        _run_euroc_eval(
            repo_root=REPO_ROOT,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            output_dir=repeat_dir,
            method=method,
            sequences=sequences,
        )
        _validate_euroc_repeat_dir(repeat_dir, expected_sequences=sequences)
    return aggregate_euroc_three_method_benchmark(
        benchmark_root=benchmark_root,
        method_ids=[candidate.method_id],
        expected_sequences=sequences,
        repeats=repeats,
    )


def _candidate_full_stats(
    *,
    benchmark_root: Path,
    method_id: str,
    baseline_assoc: dict[str, float],
    id_field: str,
    repeats: int,
) -> dict[str, float]:
    repeat_scores: list[float] = []
    repeat_wins: list[float] = []
    repeat_rpe_trans: list[float] = []
    repeat_rpe_rot: list[float] = []
    repeat_scale_log: list[float] = []
    for repeat_idx in range(1, int(repeats) + 1):
        csv_path = benchmark_root / method_id / f"repeat_{repeat_idx:02d}" / "dpvo_style_metrics_summary.csv"
        rows = _read_csv_rows(csv_path)
        repeat_scores.append(_mean([_safe_float(row.get("ate_rmse_associated")) for row in rows]))
        repeat_rpe_trans.append(_mean([_safe_float(row.get("rpe_trans_rmse")) for row in rows]))
        repeat_rpe_rot.append(_mean([_safe_float(row.get("rpe_rot_rmse")) for row in rows]))
        repeat_scale_log.append(_mean([_safe_float(row.get("scale_error_abs_log")) for row in rows]))
        wins, _, _ = _count_wins(rows, baseline_assoc=baseline_assoc)
        repeat_wins.append(float(wins))
    method_row = _single_row(
        benchmark_root / "summary" / "method_comparison.csv",
        key_field=id_field,
        key_value=method_id,
    )
    per_sequence_rows = [
        row
        for row in _read_csv_rows(benchmark_root / "summary" / "per_sequence_median.csv")
        if str(row.get(id_field, "")).strip() == method_id
    ]
    per_sequence_wins = 0
    for row in per_sequence_rows:
        sequence = str(row.get("sequence", "")).strip()
        assoc = _safe_float(row.get("median_ate_rmse_associated"))
        baseline = _safe_float(baseline_assoc.get(sequence))
        if math.isfinite(assoc) and math.isfinite(baseline) and assoc < baseline:
            per_sequence_wins += 1
    return {
        "mean_ate": _safe_float(method_row.get("full_mean_of_repeat_means_ate_rmse_associated")),
        "average_wins": _mean(repeat_wins),
        "per_sequence_wins": float(per_sequence_wins),
        "mean_rpe_trans": _mean(repeat_rpe_trans),
        "mean_rpe_rot": _mean(repeat_rpe_rot),
        "mean_scale_error_abs_log": _mean(repeat_scale_log),
    }


def _stage1_method_rows(
    *,
    candidates: list[EvalMethodSpec],
    base_output_dir: Path,
    manifest: dict[str, Any],
    tum_dpvo_baseline: dict[str, float],
    euroc_dpvo_baseline: dict[str, float],
    stage1_repeats: int,
) -> list[dict[str, object]]:
    frozen_rows = _read_stage1_frozen_rows(manifest)
    full_cfg = manifest.get("full_benchmark", {}) or {}
    reference_cfg = dict(full_cfg.get("references", {}) or {})
    tum_reference_id = str(reference_cfg.get("tum", "current_multiscale_32x4_v1_hybrid75_25"))
    euroc_reference_id = str(reference_cfg.get("euroc", "dpvo_native_matched"))
    frozen_tum_reference_mean = _safe_float(
        frozen_rows[tum_reference_id].get("tum_mean_of_repeat_means_ate_rmse_associated")
    )
    frozen_euroc_reference_mean = _safe_float(
        frozen_rows[euroc_reference_id].get("euroc_mean_of_repeat_means_ate_rmse_associated")
    )
    stage1_cfg = dict(((manifest.get("full_benchmark", {}) or {}).get("stage1_ranking", {}) or {}))
    weights_cfg = dict(stage1_cfg.get("weights", {}) or {})
    tum_ate_weight = float(weights_cfg.get("tum_ate", 0.45))
    euroc_ate_weight = float(weights_cfg.get("euroc_ate", 0.30))
    tum_wins_weight = float(weights_cfg.get("tum_wins", 0.10))
    euroc_wins_weight = float(weights_cfg.get("euroc_wins", 0.15))
    rows: list[dict[str, object]] = list(frozen_rows.values())

    for candidate in candidates:
        tum_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        euroc_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "euroc"
        tum_stats = _candidate_full_stats(
            benchmark_root=tum_root,
            method_id=candidate.method_id,
            baseline_assoc=tum_dpvo_baseline,
            id_field="method_id",
            repeats=stage1_repeats,
        )
        euroc_stats = _candidate_full_stats(
            benchmark_root=euroc_root,
            method_id=candidate.method_id,
            baseline_assoc=euroc_dpvo_baseline,
            id_field="method_id",
            repeats=stage1_repeats,
        )
        stage1_score = (
            tum_ate_weight * _ratio(float(tum_stats["mean_ate"]), frozen_tum_reference_mean)
            + euroc_ate_weight * _ratio(float(euroc_stats["mean_ate"]), frozen_euroc_reference_mean)
            + tum_wins_weight * ((38.0 - float(tum_stats["average_wins"])) / 38.0)
            + euroc_wins_weight * ((11.0 - float(euroc_stats["average_wins"])) / 11.0)
        )
        rows.append(
            {
                "method_id": candidate.method_id,
                "kind": "candidate",
                "tum_mean_of_repeat_means_ate_rmse_associated": _format_float(tum_stats["mean_ate"]),
                "tum_average_wins_vs_dpvo": _format_float(tum_stats["average_wins"]),
                "tum_per_sequence_median_wins_vs_dpvo": int(tum_stats["per_sequence_wins"]),
                "euroc_mean_of_repeat_means_ate_rmse_associated": _format_float(euroc_stats["mean_ate"]),
                "euroc_average_wins_vs_dpvo": _format_float(euroc_stats["average_wins"]),
                "euroc_per_sequence_median_wins_vs_dpvo": int(euroc_stats["per_sequence_wins"]),
                "stage1_score": _format_float(stage1_score),
            }
        )

    rows.sort(
        key=lambda row: (
            0 if str(row.get("kind")) == "candidate" else 1,
            _safe_float(row.get("stage1_score")),
            -_safe_float(row.get("euroc_average_wins_vs_dpvo")),
            _safe_float(row.get("euroc_mean_of_repeat_means_ate_rmse_associated")),
            _safe_float(row.get("tum_mean_of_repeat_means_ate_rmse_associated")),
            -_safe_float(row.get("tum_average_wins_vs_dpvo")),
        )
    )
    return rows


def _stage1_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# TUM + EuRoC Stage 1 Full Benchmark",
        "",
        "| Method | Kind | Stage1 score | TUM mean | TUM wins vs DPVO | EuRoC mean | EuRoC wins vs DPVO |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method_id']}` | {row.get('kind', '')} | {row.get('stage1_score', '')} | "
            f"{row.get('tum_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('tum_average_wins_vs_dpvo', '')} | "
            f"{row.get('euroc_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('euroc_average_wins_vs_dpvo', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _top_stage1_candidates(
    rows: list[dict[str, object]],
    candidate_methods: list[EvalMethodSpec],
    top_k: int,
) -> list[EvalMethodSpec]:
    by_id = {method.method_id: method for method in candidate_methods}
    output: list[EvalMethodSpec] = []
    for row in rows:
        if str(row.get("kind")) != "candidate":
            continue
        method = by_id.get(str(row.get("method_id", "")))
        if method is None:
            continue
        output.append(method)
        if len(output) >= int(top_k):
            break
    return output


def _read_stage1_frozen_rows(manifest: dict[str, Any]) -> dict[str, dict[str, object]]:
    full_cfg = manifest.get("full_benchmark", {}) or {}
    frozen_cfg = dict(full_cfg.get("frozen", {}) or {})
    path = _resolve_path(str(frozen_cfg["stage1_method_comparison"]), REPO_ROOT)
    aliases = dict(frozen_cfg.get("stage1_method_aliases", {}) or {})
    if not aliases:
        aliases = {
            "incumbent_qualitysoft_r78_22": "ms32x4_micro2_anchor16_qualitysoft_r78_22_v1",
            "current_multiscale_32x4_v1_hybrid75_25": "multiscale_32x4_v1_hybrid75_25",
            "dpvo_native_matched": "dpvo_native",
        }

    rows: dict[str, dict[str, object]] = {}
    for method_id, source_method_id in aliases.items():
        source_row = _single_row(path, key_field="method_id", key_value=str(source_method_id))
        rows[str(method_id)] = {
            "method_id": str(method_id),
            "kind": "frozen_baseline",
            "tum_mean_of_repeat_means_ate_rmse_associated": _format_float(
                _safe_float(source_row.get("tum_mean_of_repeat_means_ate_rmse_associated"))
            ),
            "tum_average_wins_vs_dpvo": _format_float(_safe_float(source_row.get("tum_average_wins_vs_dpvo"))),
            "tum_per_sequence_median_wins_vs_dpvo": str(
                int(_safe_float(source_row.get("tum_per_sequence_median_wins_vs_dpvo")))
            ),
            "euroc_mean_of_repeat_means_ate_rmse_associated": _format_float(
                _safe_float(source_row.get("euroc_mean_of_repeat_means_ate_rmse_associated"))
            ),
            "euroc_average_wins_vs_dpvo": _format_float(_safe_float(source_row.get("euroc_average_wins_vs_dpvo"))),
            "euroc_per_sequence_median_wins_vs_dpvo": str(
                int(_safe_float(source_row.get("euroc_per_sequence_median_wins_vs_dpvo")))
            ),
        }
    return rows


def _final_method_rows(
    *,
    candidates: list[EvalMethodSpec],
    base_output_dir: Path,
    manifest: dict[str, Any],
    tum_dpvo_baseline: dict[str, float],
    euroc_dpvo_baseline: dict[str, float],
    repeats: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    frozen_rows = _read_stage1_frozen_rows(manifest)
    full_cfg = manifest.get("full_benchmark", {}) or {}
    reference_cfg = dict(full_cfg.get("references", {}) or {})
    tum_reference_id = str(reference_cfg.get("tum", "incumbent_qualitysoft_r78_22"))
    euroc_reference_id = str(reference_cfg.get("euroc", "dpvo_native_matched"))
    winner_gate = dict(full_cfg.get("winner_gate", {}) or {})
    final_ranking = dict((full_cfg.get("final_ranking", {}) or {}).get("weights", {}) or {})
    tum_avg_wins_min = float(winner_gate.get("tum_average_wins_vs_dpvo_min", 30.0))
    euroc_avg_wins_min = float(winner_gate.get("euroc_average_wins_vs_dpvo_min", 8.0))
    require_tum_mean_le = bool(winner_gate.get("require_tum_mean_le_reference_tum", False))
    require_euroc_mean_lt = bool(winner_gate.get("require_euroc_mean_lt_reference_euroc", True))
    frozen_tum_reference_mean = _safe_float(
        frozen_rows[tum_reference_id].get("tum_mean_of_repeat_means_ate_rmse_associated")
    )
    frozen_euroc_reference_mean = _safe_float(
        frozen_rows[euroc_reference_id].get("euroc_mean_of_repeat_means_ate_rmse_associated")
    )
    rows: list[dict[str, object]] = list(frozen_rows.values())

    for candidate in candidates:
        tum_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        euroc_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "euroc"
        tum_stats = _candidate_full_stats(
            benchmark_root=tum_root,
            method_id=candidate.method_id,
            baseline_assoc=tum_dpvo_baseline,
            id_field="method_id",
            repeats=repeats,
        )
        euroc_stats = _candidate_full_stats(
            benchmark_root=euroc_root,
            method_id=candidate.method_id,
            baseline_assoc=euroc_dpvo_baseline,
            id_field="method_id",
            repeats=repeats,
        )
        tum_mean = float(tum_stats["mean_ate"])
        tum_avg_wins = float(tum_stats["average_wins"])
        tum_perseq_wins = int(tum_stats["per_sequence_wins"])
        euroc_mean = float(euroc_stats["mean_ate"])
        euroc_avg_wins = float(euroc_stats["average_wins"])
        euroc_perseq_wins = int(euroc_stats["per_sequence_wins"])
        meets_tum_win_target = tum_avg_wins >= tum_avg_wins_min
        meets_euroc_win_target = euroc_avg_wins >= euroc_avg_wins_min
        meets_tum_mean_target = (not require_tum_mean_le) or (
            math.isfinite(tum_mean) and math.isfinite(frozen_tum_reference_mean) and tum_mean <= frozen_tum_reference_mean
        )
        beats_dpvo_on_euroc = (not require_euroc_mean_lt) or (
            math.isfinite(euroc_mean) and math.isfinite(frozen_euroc_reference_mean) and euroc_mean < frozen_euroc_reference_mean
        )
        final_eligible = bool(
            meets_tum_win_target
            and meets_euroc_win_target
            and meets_tum_mean_target
            and beats_dpvo_on_euroc
        )
        final_score = (
            float(final_ranking.get("tum_ate", 0.45)) * _ratio(tum_mean, frozen_tum_reference_mean)
            + float(final_ranking.get("euroc_ate", 0.30)) * _ratio(euroc_mean, frozen_euroc_reference_mean)
            + float(final_ranking.get("tum_wins", 0.10)) * ((38.0 - tum_avg_wins) / 38.0)
            + float(final_ranking.get("euroc_wins", 0.15)) * ((11.0 - euroc_avg_wins) / 11.0)
        )
        weighted_rpe_trans = (
            float(final_ranking.get("tum_ate", 0.45)) * float(tum_stats["mean_rpe_trans"])
            + float(final_ranking.get("euroc_ate", 0.30)) * float(euroc_stats["mean_rpe_trans"])
        )
        weighted_rpe_rot = (
            float(final_ranking.get("tum_ate", 0.45)) * float(tum_stats["mean_rpe_rot"])
            + float(final_ranking.get("euroc_ate", 0.30)) * float(euroc_stats["mean_rpe_rot"])
        )
        weighted_scale = (
            float(final_ranking.get("tum_ate", 0.45)) * float(tum_stats["mean_scale_error_abs_log"])
            + float(final_ranking.get("euroc_ate", 0.30)) * float(euroc_stats["mean_scale_error_abs_log"])
        )
        rows.append(
            {
                "method_id": candidate.method_id,
                "kind": "candidate",
                "tum_mean_of_repeat_means_ate_rmse_associated": _format_float(tum_mean),
                "tum_average_wins_vs_dpvo": _format_float(tum_avg_wins),
                "tum_per_sequence_median_wins_vs_dpvo": tum_perseq_wins,
                "euroc_mean_of_repeat_means_ate_rmse_associated": _format_float(euroc_mean),
                "euroc_average_wins_vs_dpvo": _format_float(euroc_avg_wins),
                "euroc_per_sequence_median_wins_vs_dpvo": euroc_perseq_wins,
                "meets_tum_win_target": int(meets_tum_win_target),
                "meets_euroc_win_target": int(meets_euroc_win_target),
                "meets_tum_mean_target": int(meets_tum_mean_target),
                "beats_dpvo_on_euroc": int(beats_dpvo_on_euroc),
                "final_score": _format_float(final_score if final_eligible else math.inf),
                "weighted_rpe_trans_tiebreak": _format_float(weighted_rpe_trans if final_eligible else math.inf),
                "weighted_rpe_rot_tiebreak": _format_float(weighted_rpe_rot if final_eligible else math.inf),
                "weighted_scale_error_abs_log_tiebreak": _format_float(weighted_scale if final_eligible else math.inf),
                "final_eligible": int(final_eligible),
            }
        )

    candidate_rows = [row for row in rows if str(row.get("kind")) == "candidate"]
    eligible = [row for row in candidate_rows if int(row.get("final_eligible", 0))]
    winner: dict[str, object] | None = None
    if eligible:
        eligible.sort(
            key=lambda row: (
                _safe_float(row.get("final_score")),
                -_safe_float(row.get("euroc_per_sequence_median_wins_vs_dpvo")),
                -_safe_float(row.get("tum_per_sequence_median_wins_vs_dpvo")),
                _safe_float(row.get("weighted_rpe_trans_tiebreak")),
                _safe_float(row.get("weighted_rpe_rot_tiebreak")),
            )
        )
        winner = eligible[0]

    rows.sort(
        key=lambda row: (
            0 if str(row.get("kind")) == "candidate" else 1,
            0 if int(row.get("final_eligible", 0)) else 1,
            _safe_float(row.get("final_score")),
            -_safe_float(row.get("euroc_per_sequence_median_wins_vs_dpvo")),
            -_safe_float(row.get("tum_per_sequence_median_wins_vs_dpvo")),
        )
    )
    return rows, winner


def _final_markdown(rows: list[dict[str, object]], winner: dict[str, object] | None) -> str:
    lines = [
        "# TUM + EuRoC Full 5x",
        "",
        f"Winner: `{winner['method_id']}`" if winner is not None else "Winner: none",
        "",
        "| Method | Kind | Final score | TUM mean | TUM avg wins | EuRoC mean | EuRoC avg wins | TUM>=30 | EuRoC>=8 | TUM<=ref | EuRoC<ref | Eligible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method_id']}` | {row.get('kind', '')} | "
            f"{row.get('final_score', '')} | "
            f"{row.get('tum_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('tum_average_wins_vs_dpvo', '')} | "
            f"{row.get('euroc_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('euroc_average_wins_vs_dpvo', '')} | "
            f"{row.get('meets_tum_win_target', '')} | "
            f"{row.get('meets_euroc_win_target', '')} | "
            f"{row.get('meets_tum_mean_target', '')} | "
            f"{row.get('beats_dpvo_on_euroc', '')} | "
            f"{row.get('final_eligible', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TUM+EuRoC live dual-proxy DINO-DPVO sweep.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-runs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest_path = _resolve_path(args.manifest, REPO_ROOT)
    manifest, runs = _load_manifest(manifest_path)
    base_output_dir = (
        _resolve_path(args.base_output_dir, REPO_ROOT)
        if args.base_output_dir
        else (REPO_ROOT / "refocus_vo" / "runs" / "sweeps" / str(manifest.get("name", "tum_euroc_dualproxy_sweep")))
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, base_output_dir / "manifest.yaml")

    if args.dry_run:
        _print_dry_run(manifest, runs[: int(args.limit_runs)] if args.limit_runs is not None else runs, base_output_dir)
        proxy_cfg = manifest.get("proxy_validation", {}) or {}
        for dataset_id, spec in _load_dataset_specs(proxy_cfg).items():
            print(f"proxy_dataset[{dataset_id}]={','.join(spec.sequences)}")
        return

    _verify_gpu_idle(bool(args.force))

    from refocus_vo.sweeps.run_assoc9_cross_dataset_sweep import _runner_paths

    dpvo_root, dpvo_weights, dpvo_config = _runner_paths(manifest)
    proxy_cfg = manifest.get("proxy_validation", {}) or {}
    proxy_datasets = _load_dataset_specs(proxy_cfg)
    baseline_methods = _baseline_methods(manifest)
    references = proxy_cfg.get("references", {}) or {}
    ranking_cfg = dict(proxy_cfg.get("ranking", {}) or {})
    dataset_weights = {str(key): float(value) for key, value in (ranking_cfg.get("weights", {}) or {}).items()}
    win_weights = {str(key): float(value) for key, value in (ranking_cfg.get("win_weights", {}) or {}).items()}
    gate_cfg = dict(proxy_cfg.get("gate", {}) or {})
    tum_no_regression_multiplier = float(gate_cfg.get("tum_no_regression_multiplier", 1.03))
    tum_pressure_multiplier = float(gate_cfg.get("tum_pressure_multiplier", 1.05))
    tum_min_wins_vs_dpvo = int(gate_cfg.get("tum_proxy_min_wins_vs_dpvo", 8) or 8)
    euroc_min_wins_vs_dpvo = int(gate_cfg.get("euroc_proxy_min_wins_vs_dpvo_for_advancement", 6) or 6)
    use_live_training_eval = bool(proxy_cfg.get("use_live_training_eval", True))

    baseline_summaries = _evaluate_method_group(
        python_bin=sys.executable,
        methods=baseline_methods,
        datasets=proxy_datasets,
        stage_root=base_output_dir / "proxy_eval" / "baselines",
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        resume=bool(args.resume),
    )
    frozen_sequence_baselines = _build_frozen_sequence_baselines(manifest, proxy_datasets=proxy_datasets)

    training_manifest = deepcopy(manifest)
    if use_live_training_eval:
        training_manifest.setdefault("sweep", {}).setdefault("config_overrides", {}).setdefault("eval", {})
        training_manifest["sweep"]["config_overrides"]["eval"]["selection_metric"] = "dual_proxy_score"
        training_manifest["sweep"]["config_overrides"]["eval"]["live_proxy"] = _build_live_proxy_eval_cfg(
            proxy_datasets=proxy_datasets,
            baseline_summaries=baseline_summaries,
            reference_ids={"tum": str(references["tum"]), "euroc": str(references["euroc"])},
            dataset_weights=dataset_weights,
            win_weights=win_weights,
            tum_no_regression_multiplier=tum_no_regression_multiplier,
            sequence_assoc_baselines=frozen_sequence_baselines,
            gate_overrides={
                "tum_no_regression_multiplier": tum_no_regression_multiplier,
                "tum_pressure_multiplier": tum_pressure_multiplier,
                "tum_proxy_min_wins_vs_dpvo": tum_min_wins_vs_dpvo,
                "required_valid_datasets": ["euroc"],
            },
        )

    leaderboard_rows = _run_training_sweep(
        training_manifest,
        runs,
        base_output_dir=base_output_dir,
        resume=bool(args.resume),
        limit_runs=args.limit_runs,
    )

    candidate_methods = _candidate_methods(leaderboard_rows)
    if not candidate_methods:
        raise RuntimeError("No completed candidate runs were available for proxy validation")

    candidate_proxy_summaries = _load_live_candidate_proxy_summaries(
        candidate_methods=candidate_methods,
        leaderboard_rows=leaderboard_rows,
        datasets=proxy_datasets,
        base_output_dir=base_output_dir,
    )
    proxy_summaries = dict(baseline_summaries)
    proxy_summaries.update(candidate_proxy_summaries)

    dataset_summary_rows = [
        _dataset_summary_to_row(method, summary)
        for method in baseline_methods
        for summary in proxy_summaries[method.method_id].values()
    ]
    dataset_summary_rows.extend(
        _dataset_summary_to_row(method, summary)
        for method in candidate_methods
        for summary in proxy_summaries[method.method_id].values()
    )
    _write_csv(
        base_output_dir / "proxy_eval" / "dataset_summary.csv",
        dataset_summary_rows,
        [
            "method_id",
            "kind",
            "dataset_id",
            "mean_ate_rmse_associated",
            "mean_rpe_trans_rmse",
            "mean_rpe_rot_rmse",
            "mean_scale_correction",
            "mean_scale_error_abs",
            "mean_scale_error_abs_log",
            "mean_coverage",
            "pressure_mean_ate_rmse_associated",
            "mean_kitti_trans_percent",
            "mean_kitti_rot_deg_per_m",
            "row_count",
            "finite_count",
            "ok_count",
            "non_ok_count",
            "failed_count",
        ],
    )

    ranked_rows = _rank_dual_proxy_candidates(
        candidate_methods=candidate_methods,
        summaries_by_method=proxy_summaries,
        leaderboard_rows=leaderboard_rows,
        reference_ids={"tum": str(references["tum"]), "euroc": str(references["euroc"])},
        dataset_weights=dataset_weights,
        win_weights=win_weights,
        tum_no_regression_multiplier=tum_no_regression_multiplier,
        tum_pressure_multiplier=tum_pressure_multiplier,
        tum_min_wins_vs_dpvo=tum_min_wins_vs_dpvo,
        euroc_min_wins_vs_dpvo=euroc_min_wins_vs_dpvo,
    )
    _write_csv(
        base_output_dir / "proxy_leaderboard.csv",
        ranked_rows,
        [
            "proxy_rank",
            "method_id",
            "kind",
            "checkpoint_path",
            "config_path",
            "passes_tum_gate",
            "passes_euroc_validity_gate",
            "beats_dpvo_on_euroc",
            "passes_advancement_gate",
            "tum_mean_ate_rmse_associated",
            "tum_pressure_mean_ate_rmse_associated",
            "tum_proxy_wins_vs_dpvo",
            "euroc_mean_ate_rmse_associated",
            "euroc_proxy_wins_vs_dpvo",
            "weighted_dual_score",
            "weighted_rpe_trans_score",
            "weighted_rpe_rot_score",
            "weighted_scale_error_abs_log_score",
            "tum_pressure_score",
        ],
    )
    _write_text(
        base_output_dir / "proxy_leaderboard.md",
        _dual_proxy_markdown(
            ranked_rows=ranked_rows,
            baseline_summaries=[row for row in dataset_summary_rows if row["kind"] == "baseline"],
            top_k=int(proxy_cfg.get("top_k", 6) or 6),
        ),
    )

    top_candidates = _top_candidate_methods(
        candidate_methods,
        ranked_rows,
        int(proxy_cfg.get("top_k", 6) or 6),
    )
    _write_csv(
        base_output_dir / "proxy_eval" / "frozen_sequence_baselines.csv",
        _frozen_sequence_baseline_rows(frozen_sequence_baselines),
        ["dataset_id", "sequence", "baseline_ate_rmse_associated"],
    )
    if not top_candidates:
        _write_csv(
            base_output_dir / "finalists" / "full_method_comparison.csv",
            [],
            [
                "method_id",
                "kind",
                "tum_mean_of_repeat_means_ate_rmse_associated",
                "tum_average_wins_vs_dpvo",
                "tum_per_sequence_median_wins_vs_dpvo",
                "euroc_mean_of_repeat_means_ate_rmse_associated",
                "euroc_average_wins_vs_dpvo",
                "euroc_per_sequence_median_wins_vs_dpvo",
                "meets_tum_win_target",
                "meets_euroc_win_target",
                "meets_tum_mean_target",
                "beats_dpvo_on_euroc",
                "final_score",
                "weighted_rpe_trans_tiebreak",
                "weighted_rpe_rot_tiebreak",
                "weighted_scale_error_abs_log_tiebreak",
                "final_eligible",
            ],
        )
        _write_text(
            base_output_dir / "finalists" / "full_method_comparison.md",
            "# TUM + EuRoC Full 5x\n\nNo candidates passed the dual-proxy advancement gate.\n",
        )
        return

    full_cfg = manifest.get("full_benchmark", {}) or {}
    tum_dataset = _resolve_path(str((full_cfg.get("datasets", {}) or {})["tum"]["dataset_root"]), REPO_ROOT)
    euroc_dataset = _resolve_path(str((full_cfg.get("datasets", {}) or {})["euroc"]["dataset_root"]), REPO_ROOT)
    tum_sequences = _enumerate_freiburg_sequences(tum_dataset)
    euroc_sequences = list(EUROC_SEQUENCE_ORDER)
    repeats = int(full_cfg.get("repeats", REPEATS) or REPEATS)
    stage1_repeats = int(full_cfg.get("stage1_repeats", 1) or 1)
    stage2_top_k = int(full_cfg.get("stage2_top_k", 3) or 3)

    for candidate in top_candidates:
        tum_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        euroc_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "euroc"
        _run_candidate_tum_5x(
            candidate=candidate,
            repo_root=REPO_ROOT,
            dataset_root=tum_dataset,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=tum_root,
            sequences=tum_sequences,
            repeats=stage1_repeats,
        )
        _run_candidate_euroc_5x(
            candidate=candidate,
            dataset_root=euroc_dataset,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=euroc_root,
            sequences=euroc_sequences,
            repeats=stage1_repeats,
        )

    stage1_rows = _stage1_method_rows(
        candidates=top_candidates,
        base_output_dir=base_output_dir,
        manifest=manifest,
        tum_dpvo_baseline=frozen_sequence_baselines["tum"],
        euroc_dpvo_baseline=frozen_sequence_baselines["euroc"],
        stage1_repeats=stage1_repeats,
    )
    _write_csv(
        base_output_dir / "finalists" / "stage1_method_comparison.csv",
        stage1_rows,
        [
            "method_id",
            "kind",
            "tum_mean_of_repeat_means_ate_rmse_associated",
            "tum_average_wins_vs_dpvo",
            "tum_per_sequence_median_wins_vs_dpvo",
            "euroc_mean_of_repeat_means_ate_rmse_associated",
            "euroc_average_wins_vs_dpvo",
            "euroc_per_sequence_median_wins_vs_dpvo",
            "stage1_score",
        ],
    )
    _write_text(
        base_output_dir / "finalists" / "stage1_method_comparison.md",
        _stage1_markdown(stage1_rows),
    )

    stage2_candidates = _top_stage1_candidates(stage1_rows, top_candidates, stage2_top_k)
    if not stage2_candidates:
        _write_csv(
            base_output_dir / "finalists" / "full_method_comparison.csv",
            [],
            [
                "method_id",
                "kind",
                "tum_mean_of_repeat_means_ate_rmse_associated",
                "tum_average_wins_vs_dpvo",
                "tum_per_sequence_median_wins_vs_dpvo",
                "euroc_mean_of_repeat_means_ate_rmse_associated",
                "euroc_average_wins_vs_dpvo",
                "euroc_per_sequence_median_wins_vs_dpvo",
                "meets_tum_win_target",
                "meets_euroc_win_target",
                "meets_tum_mean_target",
                "beats_dpvo_on_euroc",
                "final_score",
                "weighted_rpe_trans_tiebreak",
                "weighted_rpe_rot_tiebreak",
                "weighted_scale_error_abs_log_tiebreak",
                "final_eligible",
            ],
        )
        _write_text(
            base_output_dir / "finalists" / "full_method_comparison.md",
            "# TUM + EuRoC Full 5x\n\nNo candidates advanced from the stage-1 full benchmark.\n",
        )
        return

    for candidate in stage2_candidates:
        tum_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        euroc_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "euroc"
        _run_candidate_tum_5x(
            candidate=candidate,
            repo_root=REPO_ROOT,
            dataset_root=tum_dataset,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=tum_root,
            sequences=tum_sequences,
            repeats=repeats,
        )
        _run_candidate_euroc_5x(
            candidate=candidate,
            dataset_root=euroc_dataset,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=euroc_root,
            sequences=euroc_sequences,
            repeats=repeats,
        )

    final_rows, winner = _final_method_rows(
        candidates=stage2_candidates,
        base_output_dir=base_output_dir,
        manifest=manifest,
        tum_dpvo_baseline=frozen_sequence_baselines["tum"],
        euroc_dpvo_baseline=frozen_sequence_baselines["euroc"],
        repeats=repeats,
    )
    _write_csv(
        base_output_dir / "finalists" / "full_method_comparison.csv",
        final_rows,
        [
            "method_id",
            "kind",
            "tum_mean_of_repeat_means_ate_rmse_associated",
            "tum_average_wins_vs_dpvo",
            "tum_per_sequence_median_wins_vs_dpvo",
            "euroc_mean_of_repeat_means_ate_rmse_associated",
            "euroc_average_wins_vs_dpvo",
            "euroc_per_sequence_median_wins_vs_dpvo",
            "meets_tum_win_target",
            "meets_euroc_win_target",
            "meets_tum_mean_target",
            "beats_dpvo_on_euroc",
            "final_score",
            "weighted_rpe_trans_tiebreak",
            "weighted_rpe_rot_tiebreak",
            "weighted_scale_error_abs_log_tiebreak",
            "final_eligible",
        ],
    )
    _write_text(
        base_output_dir / "finalists" / "full_method_comparison.md",
        _final_markdown(final_rows, winner),
    )


if __name__ == "__main__":
    main()
