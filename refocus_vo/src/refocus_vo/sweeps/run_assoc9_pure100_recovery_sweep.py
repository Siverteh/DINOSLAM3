from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import _enumerate_freiburg_sequences
from refocus_vo.sweeps.run_assoc9_cross_dataset_sweep import (
    DatasetEvalSpec,
    DatasetSummary,
    EvalMethodSpec,
    _baseline_methods,
    _dataset_summary_to_row,
    _evaluate_method_group,
    _format_float,
    _load_dataset_specs,
    _load_live_candidate_proxy_summaries,
    _runner_paths,
    _safe_float,
    _verify_gpu_idle,
    _write_csv,
    _write_text,
)
from refocus_vo.sweeps.run_assoc9_sweep import (
    REPO_ROOT,
    _load_manifest,
    _print_dry_run,
    _read_dev_rows,
    _resolve_path,
    _run_training_sweep,
)
from refocus_vo.sweeps.run_assoc9_tum_euroc_dualproxy_sweep import (
    _candidate_full_stats,
    _run_candidate_tum_5x,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _single_row(path: Path, *, key_field: str, key_value: str) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if str(row.get(key_field, "")).strip() == key_value:
            return row
    raise RuntimeError(f"Could not find {key_value} in {path}")


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
        if str(row.get("method", "")).strip() != "dpvo_native":
            continue
        sequence = str(row.get("sequence", "")).strip()
        value = _safe_float(row.get("median_ate_rmse_associated"))
        if sequence and math.isfinite(value):
            output[sequence] = value
    missing = [sequence for sequence in expected_sequences if sequence not in output]
    if missing:
        raise RuntimeError(f"TUM frozen DPVO baseline missing sequences: {missing}")
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


def _build_live_proxy_eval_cfg(
    *,
    proxy_dataset: DatasetEvalSpec,
    baseline_summaries: dict[str, dict[str, DatasetSummary]],
    reference_id: str,
    primary_weights: dict[str, float],
    tum_no_regression_multiplier: float,
    tum_pressure_multiplier: float,
    tum_min_wins_vs_dpvo: int,
    tum_pressure_min_wins_vs_dpvo: int,
    require_full_dino_patch_fraction: bool,
    sequence_assoc_baselines: dict[str, dict[str, float]],
) -> dict[str, object]:
    tum_ref = baseline_summaries[reference_id]["tum"]
    return {
        "enabled": True,
        "pure_tum_weights": {
            "primary_ate": float(primary_weights.get("primary_ate", 0.55)),
            "pressure_ate": float(primary_weights.get("pressure_ate", 0.20)),
            "primary_wins": float(primary_weights.get("primary_wins", 0.15)),
            "pressure_wins": float(primary_weights.get("pressure_wins", 0.10)),
        },
        "gate": {
            "tum_no_regression_multiplier": float(tum_no_regression_multiplier),
            "tum_pressure_multiplier": float(tum_pressure_multiplier),
            "tum_proxy_min_wins_vs_dpvo": int(tum_min_wins_vs_dpvo),
            "tum_pressure_min_wins_vs_dpvo": int(tum_pressure_min_wins_vs_dpvo),
            "required_valid_datasets": [],
            "require_full_dino_patch_fraction": bool(require_full_dino_patch_fraction),
        },
        "references": {
            "tum_mean_ate_rmse_associated": float(tum_ref.mean_ate_rmse_associated),
            "tum_pressure_mean_ate_rmse_associated": float(tum_ref.pressure_mean_ate_rmse_associated),
            "tum_mean_rpe_trans_rmse": float(tum_ref.mean_rpe_trans_rmse),
            "tum_mean_rpe_rot_rmse": float(tum_ref.mean_rpe_rot_rmse),
            "tum_mean_scale_error_abs_log": float(tum_ref.mean_scale_error_abs_log),
            "sequence_assoc_baselines": deepcopy(sequence_assoc_baselines),
        },
        "datasets": {
            "tum": {
                "sequences": list(proxy_dataset.sequences),
                "pressure_sequences": list(proxy_dataset.pressure_sequences),
                "max_dt": float(proxy_dataset.max_dt),
                "missing_penalty_m": float(proxy_dataset.missing_penalty_m),
                "min_coverage_ok": float(proxy_dataset.min_coverage_ok),
                "image_height": int(proxy_dataset.image_height),
                "image_width": int(proxy_dataset.image_width),
                "stride": int(proxy_dataset.stride),
                "backend_thresh": float(proxy_dataset.backend_thresh),
                "dpvo_opts": str(proxy_dataset.dpvo_opts),
                "frontend_mode": "dino_proposals",
                "collect_diagnostics": True,
                "write_patch_diagnostics": True,
            }
        },
    }


def _candidate_methods(leaderboard_rows: list[dict[str, str]]) -> list[EvalMethodSpec]:
    methods: list[EvalMethodSpec] = []
    for row in leaderboard_rows:
        status = str(row.get("status", "")).strip()
        if status not in {"completed", "early_stopped"}:
            continue
        checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip()).expanduser().resolve()
        config_path = Path(str(row.get("config_path", "")).strip()).expanduser().resolve()
        if not checkpoint_path.exists() or not config_path.exists():
            continue
        methods.append(
            EvalMethodSpec(
                method_id=str(row["run_id"]),
                frontend_mode="dino_proposals",
                frontend_config=config_path,
                frontend_checkpoint=checkpoint_path,
                kind="candidate",
                source_run_id=str(row["run_id"]),
                runtime_config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )
    return methods


def _rank_proxy_candidates(
    *,
    candidate_methods: list[EvalMethodSpec],
    summaries_by_method: dict[str, dict[str, DatasetSummary]],
    leaderboard_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    row_by_id = {str(row["run_id"]): row for row in leaderboard_rows}
    rows: list[dict[str, object]] = []
    for method in candidate_methods:
        tum = summaries_by_method[method.method_id]["tum"]
        live_row = row_by_id.get(method.method_id, {})
        score = _safe_float(live_row.get("best_pure_tum_score"))
        tum_wins = _safe_float(live_row.get("best_tum_proxy_wins_vs_dpvo"))
        dino_patch_fraction = _safe_float(live_row.get("best_dino_patch_fraction"))
        passes_gate = (
            math.isfinite(score)
            and math.isfinite(dino_patch_fraction)
            and dino_patch_fraction >= 0.999
        )
        rows.append(
            {
                "proxy_rank": "",
                "method_id": method.method_id,
                "kind": method.kind,
                "checkpoint_path": str(method.checkpoint_path or ""),
                "config_path": str(method.runtime_config_path or ""),
                "passes_proxy_gate": int(bool(passes_gate)),
                "tum_mean_ate_rmse_associated": _format_float(tum.mean_ate_rmse_associated),
                "tum_pressure_mean_ate_rmse_associated": _format_float(tum.pressure_mean_ate_rmse_associated),
                "tum_proxy_wins_vs_dpvo": _format_float(tum_wins),
                "dino_patch_fraction": _format_float(dino_patch_fraction),
                "weighted_pure_tum_score": _format_float(score if passes_gate else math.inf),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if int(row["passes_proxy_gate"]) else 1,
            _safe_float(row["weighted_pure_tum_score"]),
            _safe_float(row["tum_pressure_mean_ate_rmse_associated"]),
            -_safe_float(row["tum_proxy_wins_vs_dpvo"]),
            str(row["method_id"]),
        )
    )
    rank = 1
    for row in rows:
        if int(row["passes_proxy_gate"]):
            row["proxy_rank"] = rank
            rank += 1
    return rows


def _proxy_markdown(*, ranked_rows: list[dict[str, object]], baseline_summaries: list[dict[str, object]], top_k: int) -> str:
    lines = [
        "# Pure-DINO TUM Proxy Leaderboard",
        "",
        "| Rank | Method | Gate | Score | TUM mean | Pressure mean | Wins vs DPVO | DINO frac |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked_rows:
        lines.append(
            f"| {row.get('proxy_rank', '')} | `{row['method_id']}` | {row['passes_proxy_gate']} | "
            f"{row['weighted_pure_tum_score']} | {row['tum_mean_ate_rmse_associated']} | "
            f"{row['tum_pressure_mean_ate_rmse_associated']} | {row['tum_proxy_wins_vs_dpvo']} | "
            f"{row['dino_patch_fraction']} |"
        )
    lines.extend(["", f"## Top {top_k} passing candidates", ""])
    for row in ranked_rows:
        if not int(row["passes_proxy_gate"]):
            continue
        lines.append(
            f"- `{row['method_id']}`: score `{row['weighted_pure_tum_score']}`, "
            f"TUM `{row['tum_mean_ate_rmse_associated']}`, wins `{row['tum_proxy_wins_vs_dpvo']}`"
        )
        if len([line for line in lines if line.startswith("- `")]) >= int(top_k):
            break
    lines.extend(
        [
            "",
            "## Fresh proxy baselines",
            "",
            "| Method | Dataset | Mean ATE | Pressure mean | Coverage |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_summaries:
        lines.append(
            f"| `{row['method_id']}` | `{row['dataset_id']}` | {row['mean_ate_rmse_associated']} | "
            f"{row['pressure_mean_ate_rmse_associated']} | {row['mean_coverage']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _top_candidate_methods(candidate_methods: list[EvalMethodSpec], ranked_rows: list[dict[str, object]], top_k: int) -> list[EvalMethodSpec]:
    by_id = {method.method_id: method for method in candidate_methods}
    output: list[EvalMethodSpec] = []
    for row in ranked_rows:
        if not int(row["passes_proxy_gate"]):
            continue
        output.append(by_id[str(row["method_id"])])
        if len(output) >= int(top_k):
            break
    return output


def _benchmark_stats_from_root(
    *,
    benchmark_root: Path,
    method_id: str,
    id_field: str,
    baseline_assoc: dict[str, float],
    repeats: int,
) -> dict[str, float]:
    summary_row = _single_row(benchmark_root / "summary" / "method_comparison.csv", key_field=id_field, key_value=method_id)
    full_stats = _candidate_full_stats(
        benchmark_root=benchmark_root,
        method_id=method_id,
        baseline_assoc=baseline_assoc,
        id_field=id_field,
        repeats=repeats,
    )
    return {
        "mean_ate": float(full_stats["mean_ate"]),
        "median_ate": _safe_float(summary_row.get("full_median_of_repeat_means_ate_rmse_associated")),
        "sequence_median_ate": _safe_float(summary_row.get("full_mean_of_sequence_medians_ate_rmse_associated")),
        "average_wins": float(full_stats["average_wins"]),
        "per_sequence_wins": float(full_stats["per_sequence_wins"]),
        "mean_rpe_trans": float(full_stats["mean_rpe_trans"]),
        "mean_rpe_rot": float(full_stats["mean_rpe_rot"]),
        "mean_scale_error_abs_log": float(full_stats["mean_scale_error_abs_log"]),
    }


def _frozen_rows(manifest: dict[str, Any], *, tum_dpvo_baseline: dict[str, float]) -> list[dict[str, object]]:
    frozen_cfg = dict(((manifest.get("full_benchmark", {}) or {}).get("frozen", {}) or {}))
    rows: list[dict[str, object]] = []
    specs = [
        {
            "method_id": "target_multiscale_32x4_v1_hybrid75_25",
            "kind": "frozen_baseline",
            "benchmark_root": _resolve_path(str(frozen_cfg["multiscale_benchmark_root"]), REPO_ROOT),
            "id_field": str(frozen_cfg.get("multiscale_id_field", "finalist_id")),
            "source_method_id": str(frozen_cfg.get("multiscale_method_id", "multiscale_32x4_v1_hybrid75_25")),
        },
        {
            "method_id": "dpvo_native",
            "kind": "frozen_baseline",
            "benchmark_root": _resolve_path(str(frozen_cfg["legacy_pure_benchmark_root"]), REPO_ROOT),
            "id_field": str(frozen_cfg.get("legacy_id_field", "method")),
            "source_method_id": str(frozen_cfg.get("dpvo_method_id", "dpvo_native")),
        },
        {
            "method_id": "focus071_best_pure100",
            "kind": "frozen_baseline",
            "benchmark_root": _resolve_path(str(frozen_cfg["legacy_pure_benchmark_root"]), REPO_ROOT),
            "id_field": str(frozen_cfg.get("legacy_id_field", "method")),
            "source_method_id": str(frozen_cfg.get("focus071_method_id", "focus071_best")),
        },
    ]
    for spec in specs:
        stats = _benchmark_stats_from_root(
            benchmark_root=spec["benchmark_root"],
            method_id=spec["source_method_id"],
            id_field=spec["id_field"],
            baseline_assoc=tum_dpvo_baseline,
            repeats=5,
        )
        rows.append(
            {
                "method_id": spec["method_id"],
                "kind": spec["kind"],
                "full_mean_of_repeat_means_ate_rmse_associated": _format_float(stats["mean_ate"]),
                "full_median_of_repeat_means_ate_rmse_associated": _format_float(stats["median_ate"]),
                "full_mean_of_sequence_medians_ate_rmse_associated": _format_float(stats["sequence_median_ate"]),
                "average_wins_vs_dpvo": _format_float(stats["average_wins"]),
                "per_sequence_median_wins_vs_dpvo": int(stats["per_sequence_wins"]),
            }
        )
    return rows


def _stage1_rows(
    *,
    candidates: list[EvalMethodSpec],
    base_output_dir: Path,
    manifest: dict[str, Any],
    tum_dpvo_baseline: dict[str, float],
) -> list[dict[str, object]]:
    full_cfg = manifest.get("full_benchmark", {}) or {}
    ranking_cfg = dict((full_cfg.get("stage1_ranking", {}) or {}).get("weights", {}) or {})
    frozen_rows = _frozen_rows(manifest, tum_dpvo_baseline=tum_dpvo_baseline)
    multiscale_row = next(row for row in frozen_rows if str(row["method_id"]) == "target_multiscale_32x4_v1_hybrid75_25")
    multiscale_mean = _safe_float(multiscale_row["full_mean_of_repeat_means_ate_rmse_associated"])

    rows: list[dict[str, object]] = list(frozen_rows)
    for candidate in candidates:
        benchmark_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        stats = _benchmark_stats_from_root(
            benchmark_root=benchmark_root,
            method_id=candidate.method_id,
            id_field="method_id",
            baseline_assoc=tum_dpvo_baseline,
            repeats=1,
        )
        stage1_score = (
            float(ranking_cfg.get("tum_ate", 0.65)) * _ratio(float(stats["mean_ate"]), multiscale_mean)
            + float(ranking_cfg.get("avg_wins", 0.20)) * ((38.0 - float(stats["average_wins"])) / 38.0)
            + float(ranking_cfg.get("per_sequence_wins", 0.15)) * ((38.0 - float(stats["per_sequence_wins"])) / 38.0)
        )
        rows.append(
            {
                "method_id": candidate.method_id,
                "kind": "candidate",
                "full_mean_of_repeat_means_ate_rmse_associated": _format_float(stats["mean_ate"]),
                "full_median_of_repeat_means_ate_rmse_associated": _format_float(stats["median_ate"]),
                "full_mean_of_sequence_medians_ate_rmse_associated": _format_float(stats["sequence_median_ate"]),
                "average_wins_vs_dpvo": _format_float(stats["average_wins"]),
                "per_sequence_median_wins_vs_dpvo": int(stats["per_sequence_wins"]),
                "stage1_score": _format_float(stage1_score),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if str(row.get("kind")) == "candidate" else 1,
            _safe_float(row.get("stage1_score")),
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated")),
            -_safe_float(row.get("average_wins_vs_dpvo")),
            -_safe_float(row.get("per_sequence_median_wins_vs_dpvo")),
        )
    )
    return rows


def _stage1_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Pure-DINO TUM Stage 1 Full Benchmark",
        "",
        "| Method | Kind | Stage1 score | Mean ATE | Avg wins vs DPVO | Per-seq wins vs DPVO |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method_id']}` | {row.get('kind', '')} | {row.get('stage1_score', '')} | "
            f"{row.get('full_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('average_wins_vs_dpvo', '')} | {row.get('per_sequence_median_wins_vs_dpvo', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _top_stage1_candidates(rows: list[dict[str, object]], candidate_methods: list[EvalMethodSpec], top_k: int) -> list[EvalMethodSpec]:
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


def _final_rows(
    *,
    candidates: list[EvalMethodSpec],
    base_output_dir: Path,
    manifest: dict[str, Any],
    tum_dpvo_baseline: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    full_cfg = manifest.get("full_benchmark", {}) or {}
    final_cfg = dict((full_cfg.get("final_ranking", {}) or {}).get("weights", {}) or {})
    frozen_rows = _frozen_rows(manifest, tum_dpvo_baseline=tum_dpvo_baseline)
    multiscale_row = next(row for row in frozen_rows if str(row["method_id"]) == "target_multiscale_32x4_v1_hybrid75_25")
    multiscale_mean = _safe_float(multiscale_row["full_mean_of_repeat_means_ate_rmse_associated"])

    rows: list[dict[str, object]] = list(frozen_rows)
    winner: dict[str, object] | None = None
    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        benchmark_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        stats = _benchmark_stats_from_root(
            benchmark_root=benchmark_root,
            method_id=candidate.method_id,
            id_field="method_id",
            baseline_assoc=tum_dpvo_baseline,
            repeats=int(full_cfg.get("repeats", 5) or 5),
        )
        meets_wins = float(stats["average_wins"]) >= float(full_cfg.get("winner_gate", {}).get("average_wins_vs_dpvo_min", 30.0))
        meets_mean = math.isfinite(float(stats["mean_ate"])) and float(stats["mean_ate"]) <= float(multiscale_mean)
        eligible = bool(meets_wins and meets_mean)
        final_score = (
            float(final_cfg.get("tum_ate", 0.60)) * _ratio(float(stats["mean_ate"]), multiscale_mean)
            + float(final_cfg.get("avg_wins", 0.20)) * ((38.0 - float(stats["average_wins"])) / 38.0)
            + float(final_cfg.get("per_sequence_wins", 0.10)) * ((38.0 - float(stats["per_sequence_wins"])) / 38.0)
            + float(final_cfg.get("sequence_median_ate", 0.10)) * _ratio(float(stats["sequence_median_ate"]), multiscale_mean)
        )
        candidate_rows.append(
            {
                "method_id": candidate.method_id,
                "kind": "candidate",
                "full_mean_of_repeat_means_ate_rmse_associated": _format_float(stats["mean_ate"]),
                "full_median_of_repeat_means_ate_rmse_associated": _format_float(stats["median_ate"]),
                "full_mean_of_sequence_medians_ate_rmse_associated": _format_float(stats["sequence_median_ate"]),
                "average_wins_vs_dpvo": _format_float(stats["average_wins"]),
                "per_sequence_median_wins_vs_dpvo": int(stats["per_sequence_wins"]),
                "final_score": _format_float(final_score if eligible else math.inf),
                "meets_wins_target": int(meets_wins),
                "meets_mean_target": int(meets_mean),
                "final_eligible": int(eligible),
            }
        )
    eligible_rows = [row for row in candidate_rows if int(row["final_eligible"])]
    if eligible_rows:
        eligible_rows.sort(
            key=lambda row: (
                _safe_float(row["final_score"]),
                _safe_float(row["full_mean_of_repeat_means_ate_rmse_associated"]),
                _safe_float(row["full_median_of_repeat_means_ate_rmse_associated"]),
                -_safe_float(row["average_wins_vs_dpvo"]),
                -_safe_float(row["per_sequence_median_wins_vs_dpvo"]),
                _safe_float(row["full_mean_of_sequence_medians_ate_rmse_associated"]),
            )
        )
        winner = eligible_rows[0]
    rows.extend(candidate_rows)
    rows.sort(
        key=lambda row: (
            0 if str(row.get("kind")) == "candidate" else 1,
            0 if int(row.get("final_eligible", 0)) else 1,
            _safe_float(row.get("final_score")),
            _safe_float(row.get("full_mean_of_repeat_means_ate_rmse_associated")),
            -_safe_float(row.get("average_wins_vs_dpvo")),
        )
    )
    return rows, winner


def _final_markdown(rows: list[dict[str, object]], winner: dict[str, object] | None) -> str:
    lines = [
        "# Pure-DINO TUM Full 5x",
        "",
        f"Winner: `{winner['method_id']}`" if winner is not None else "Winner: none",
        "",
        "| Method | Kind | Final score | Mean ATE | Median repeat-mean ATE | Seq-median ATE | Avg wins vs DPVO | Per-seq wins vs DPVO | Eligible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method_id']}` | {row.get('kind', '')} | {row.get('final_score', '')} | "
            f"{row.get('full_mean_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('full_median_of_repeat_means_ate_rmse_associated', '')} | "
            f"{row.get('full_mean_of_sequence_medians_ate_rmse_associated', '')} | "
            f"{row.get('average_wins_vs_dpvo', '')} | "
            f"{row.get('per_sequence_median_wins_vs_dpvo', '')} | "
            f"{row.get('final_eligible', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pure-DINO TUM recovery sweep.")
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
        else (REPO_ROOT / "refocus_vo" / "runs" / "sweeps" / str(manifest.get("name", "dino_dpvo_pure100_tum30_recovery_v1")))
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

    dpvo_root, dpvo_weights, dpvo_config = _runner_paths(manifest)
    proxy_cfg = manifest.get("proxy_validation", {}) or {}
    proxy_datasets = _load_dataset_specs(proxy_cfg)
    tum_proxy_dataset = proxy_datasets["tum"]
    baseline_methods = _baseline_methods(manifest)
    references = dict(proxy_cfg.get("references", {}) or {})
    ranking_cfg = dict(proxy_cfg.get("ranking", {}) or {})
    gate_cfg = dict(proxy_cfg.get("gate", {}) or {})
    frozen_cfg = dict(proxy_cfg.get("frozen_baselines", {}) or {})

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

    tum_expected_sequences = _enumerate_freiburg_sequences(_resolve_path(manifest["full_benchmark"]["datasets"]["tum"]["dataset_root"], REPO_ROOT))
    tum_dpvo_baseline = _read_tum_dpvo_sequence_baselines(
        _resolve_path(str(frozen_cfg["tum_per_sequence"]), REPO_ROOT),
        expected_sequences=tum_expected_sequences,
    )

    training_manifest = deepcopy(manifest)
    training_manifest.setdefault("sweep", {}).setdefault("config_overrides", {}).setdefault("eval", {})
    training_manifest["sweep"]["config_overrides"]["eval"]["selection_metric"] = "pure_tum_proxy_score"
    training_manifest["sweep"]["config_overrides"]["eval"]["live_proxy"] = _build_live_proxy_eval_cfg(
        proxy_dataset=tum_proxy_dataset,
        baseline_summaries=baseline_summaries,
        reference_id=str(references["tum"]),
        primary_weights=dict(ranking_cfg.get("weights", {}) or {}),
        tum_no_regression_multiplier=float(gate_cfg.get("tum_no_regression_multiplier", 1.10)),
        tum_pressure_multiplier=float(gate_cfg.get("tum_pressure_multiplier", 1.10)),
        tum_min_wins_vs_dpvo=int(gate_cfg.get("tum_proxy_min_wins_vs_dpvo", 6) or 6),
        tum_pressure_min_wins_vs_dpvo=int(gate_cfg.get("tum_pressure_min_wins_vs_dpvo", 0) or 0),
        require_full_dino_patch_fraction=bool(gate_cfg.get("require_full_dino_patch_fraction", True)),
        sequence_assoc_baselines={"tum": tum_dpvo_baseline},
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

    ranked_rows = _rank_proxy_candidates(
        candidate_methods=candidate_methods,
        summaries_by_method=proxy_summaries,
        leaderboard_rows=leaderboard_rows,
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
            "passes_proxy_gate",
            "tum_mean_ate_rmse_associated",
            "tum_pressure_mean_ate_rmse_associated",
            "tum_proxy_wins_vs_dpvo",
            "dino_patch_fraction",
            "weighted_pure_tum_score",
        ],
    )
    _write_text(
        base_output_dir / "proxy_leaderboard.md",
        _proxy_markdown(
            ranked_rows=ranked_rows,
            baseline_summaries=[row for row in dataset_summary_rows if row["kind"] == "baseline"],
            top_k=int(proxy_cfg.get("top_k", 8) or 8),
        ),
    )

    top_candidates = _top_candidate_methods(candidate_methods, ranked_rows, int(proxy_cfg.get("top_k", 8) or 8))
    if not top_candidates:
        _write_csv(
            base_output_dir / "finalists" / "full_method_comparison.csv",
            [],
            [
                "method_id",
                "kind",
                "full_mean_of_repeat_means_ate_rmse_associated",
                "full_median_of_repeat_means_ate_rmse_associated",
                "full_mean_of_sequence_medians_ate_rmse_associated",
                "average_wins_vs_dpvo",
                "per_sequence_median_wins_vs_dpvo",
                "final_score",
                "final_eligible",
            ],
        )
        _write_text(
            base_output_dir / "finalists" / "full_method_comparison.md",
            "# Pure-DINO TUM Full 5x\n\nNo candidates passed the proxy advancement gate.\n",
        )
        return

    full_cfg = manifest.get("full_benchmark", {}) or {}
    tum_dataset_root = _resolve_path(str(full_cfg["datasets"]["tum"]["dataset_root"]), REPO_ROOT)
    stage1_repeats = int(full_cfg.get("stage1_repeats", 1) or 1)
    for candidate in top_candidates:
        tum_root = base_output_dir / "finalists" / "stage1_full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        _run_candidate_tum_5x(
            candidate=candidate,
            repo_root=REPO_ROOT,
            dataset_root=tum_dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=tum_root,
            sequences=tum_expected_sequences,
            repeats=stage1_repeats,
        )

    stage1_rows = _stage1_rows(
        candidates=top_candidates,
        base_output_dir=base_output_dir,
        manifest=manifest,
        tum_dpvo_baseline=tum_dpvo_baseline,
    )
    _write_csv(
        base_output_dir / "finalists" / "stage1_method_comparison.csv",
        stage1_rows,
        [
            "method_id",
            "kind",
            "full_mean_of_repeat_means_ate_rmse_associated",
            "full_median_of_repeat_means_ate_rmse_associated",
            "full_mean_of_sequence_medians_ate_rmse_associated",
            "average_wins_vs_dpvo",
            "per_sequence_median_wins_vs_dpvo",
            "stage1_score",
        ],
    )
    _write_text(base_output_dir / "finalists" / "stage1_method_comparison.md", _stage1_markdown(stage1_rows))

    stage2_candidates = _top_stage1_candidates(stage1_rows, top_candidates, int(full_cfg.get("stage2_top_k", 3) or 3))
    if not stage2_candidates:
        _write_csv(
            base_output_dir / "finalists" / "full_method_comparison.csv",
            [],
            [
                "method_id",
                "kind",
                "full_mean_of_repeat_means_ate_rmse_associated",
                "full_median_of_repeat_means_ate_rmse_associated",
                "full_mean_of_sequence_medians_ate_rmse_associated",
                "average_wins_vs_dpvo",
                "per_sequence_median_wins_vs_dpvo",
                "final_score",
                "final_eligible",
            ],
        )
        _write_text(
            base_output_dir / "finalists" / "full_method_comparison.md",
            "# Pure-DINO TUM Full 5x\n\nNo candidates advanced from the stage-1 benchmark.\n",
        )
        return

    for candidate in stage2_candidates:
        tum_root = base_output_dir / "finalists" / "full_eval" / candidate.method_id / "tum_rgbd_freiburg123"
        _run_candidate_tum_5x(
            candidate=candidate,
            repo_root=REPO_ROOT,
            dataset_root=tum_dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            benchmark_root=tum_root,
            sequences=tum_expected_sequences,
            repeats=int(full_cfg.get("repeats", 5) or 5),
        )

    final_rows, winner = _final_rows(
        candidates=stage2_candidates,
        base_output_dir=base_output_dir,
        manifest=manifest,
        tum_dpvo_baseline=tum_dpvo_baseline,
    )
    _write_csv(
        base_output_dir / "finalists" / "full_method_comparison.csv",
        final_rows,
        [
            "method_id",
            "kind",
            "full_mean_of_repeat_means_ate_rmse_associated",
            "full_median_of_repeat_means_ate_rmse_associated",
            "full_mean_of_sequence_medians_ate_rmse_associated",
            "average_wins_vs_dpvo",
            "per_sequence_median_wins_vs_dpvo",
            "final_score",
            "meets_wins_target",
            "meets_mean_target",
            "final_eligible",
        ],
    )
    _write_text(base_output_dir / "finalists" / "full_method_comparison.md", _final_markdown(final_rows, winner))


if __name__ == "__main__":
    main()
