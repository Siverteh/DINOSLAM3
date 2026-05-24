from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refocus_vo.sweeps.run_assoc9_sweep import (
    REPO_ROOT,
    _load_manifest,
    _print_dry_run,
    _resolve_path,
    _run_training_sweep,
)


ALLOWED_PROXY_STATUSES = {"ok", "partial_low_coverage"}
STATE_FILENAME = "cross_dataset_state.yaml"


@dataclass(frozen=True)
class EvalMethodSpec:
    method_id: str
    frontend_mode: str
    frontend_config: Path | None
    frontend_checkpoint: Path | None
    kind: str
    source_run_id: str | None = None
    runtime_config_path: Path | None = None
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class DatasetEvalSpec:
    dataset_id: str
    module: str
    dataset_root: Path
    sequences: tuple[str, ...]
    max_dt: float
    missing_penalty_m: float
    min_coverage_ok: float
    image_height: int
    image_width: int
    stride: int
    backend_thresh: float
    dpvo_opts: str
    pressure_sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSummary:
    dataset_id: str
    row_count: int
    finite_count: int
    ok_count: int
    non_ok_count: int
    failed_count: int
    mean_ate_rmse_associated: float
    mean_rpe_trans_rmse: float
    mean_rpe_rot_rmse: float
    mean_scale_correction: float
    mean_scale_error_abs: float
    mean_scale_error_abs_log: float
    mean_coverage: float
    pressure_mean_ate_rmse_associated: float
    mean_kitti_trans_percent: float
    mean_kitti_rot_deg_per_m: float


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _mean(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(sum(usable) / len(usable))


def _ratio(value: float, reference: float, *, eps: float = 1e-6) -> float:
    if not math.isfinite(value):
        return math.inf
    if not math.isfinite(reference):
        return math.inf
    denom = max(abs(float(reference)), float(eps))
    return float(value) / denom


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _format_float(value: float) -> str:
    return "NaN" if not math.isfinite(value) else f"{float(value):.6f}"


def _split_dpvo_opts(opts_text: str) -> list[str]:
    tokens = [item for item in shlex.split(str(opts_text)) if item.strip()]
    output: list[str] = []
    for item in tokens:
        if "=" in item:
            key, value = item.split("=", 1)
            output.extend([key, value])
        else:
            output.append(item)
    return output


def _resolve_dataset_spec(dataset_id: str, payload: dict[str, Any]) -> DatasetEvalSpec:
    return DatasetEvalSpec(
        dataset_id=str(dataset_id),
        module=str(payload["module"]),
        dataset_root=_resolve_path(payload["dataset_root"], REPO_ROOT),
        sequences=tuple(str(item) for item in payload["sequences"]),
        max_dt=float(payload.get("max_dt", 0.02)),
        missing_penalty_m=float(payload.get("missing_penalty_m", 3.0)),
        min_coverage_ok=float(payload.get("min_coverage_ok", 0.95)),
        image_height=int(payload["image_height"]),
        image_width=int(payload["image_width"]),
        stride=int(payload["stride"]),
        backend_thresh=float(payload["backend_thresh"]),
        dpvo_opts=str(payload["dpvo_opts"]),
        pressure_sequences=tuple(str(item) for item in payload.get("pressure_sequences", []) or []),
    )


def _load_dataset_specs(section: dict[str, Any]) -> dict[str, DatasetEvalSpec]:
    datasets = section.get("datasets", {}) or {}
    output: dict[str, DatasetEvalSpec] = {}
    for dataset_id, payload in datasets.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset spec for {dataset_id} must be a mapping")
        output[str(dataset_id)] = _resolve_dataset_spec(str(dataset_id), payload)
    return output


def _baseline_methods(manifest: dict[str, Any], *, section_name: str = "proxy_validation") -> list[EvalMethodSpec]:
    payload = manifest.get(section_name, {}) or {}
    methods: list[EvalMethodSpec] = []
    for item in payload.get("baselines", []) or []:
        methods.append(
            EvalMethodSpec(
                method_id=str(item["method_id"]),
                frontend_mode=str(item["frontend_mode"]),
                frontend_config=_resolve_path(item["frontend_config"], REPO_ROOT) if item.get("frontend_config") else None,
                frontend_checkpoint=_resolve_path(item["frontend_checkpoint"], REPO_ROOT) if item.get("frontend_checkpoint") else None,
                kind="baseline",
            )
        )
    return methods


def _candidate_methods(leaderboard_rows: list[dict[str, str]]) -> list[EvalMethodSpec]:
    methods: list[EvalMethodSpec] = []
    for row in leaderboard_rows:
        status = str(row.get("status", "")).strip()
        if status not in {"completed", "early_stopped"}:
            continue
        checkpoint_path = Path(str(row["checkpoint_path"])).expanduser().resolve()
        config_path = Path(str(row["config_path"])).expanduser().resolve()
        if not checkpoint_path.exists() or not config_path.exists():
            continue
        methods.append(
            EvalMethodSpec(
                method_id=str(row["run_id"]),
                frontend_mode="dino_hybrid",
                frontend_config=config_path,
                frontend_checkpoint=checkpoint_path,
                kind="candidate",
                source_run_id=str(row["run_id"]),
                runtime_config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )
    return methods


def _read_sequences(rows: list[dict[str, str]]) -> list[str]:
    return [str(row.get("sequence", "")).strip() for row in rows]


def _valid_existing_result(csv_path: Path, expected_sequences: tuple[str, ...]) -> bool:
    if not csv_path.exists():
        return False
    try:
        rows = _read_csv_rows(csv_path)
    except Exception:
        return False
    return _read_sequences(rows) == list(expected_sequences)


def _evaluation_command(
    *,
    python_bin: str,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    method: EvalMethodSpec,
    dataset: DatasetEvalSpec,
    output_dir: Path,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        dataset.module,
        "--dataset-root",
        str(dataset.dataset_root),
        "--dpvo-root",
        str(dpvo_root),
        "--weights",
        str(dpvo_weights),
        "--config",
        str(dpvo_config),
        "--output-dir",
        str(output_dir),
        "--csv-path",
        str(output_dir / "metrics_summary.csv"),
        "--dpvo-style-csv-path",
        str(output_dir / "dpvo_style_metrics_summary.csv"),
        "--sequences",
        ",".join(dataset.sequences),
        "--max-dt",
        str(dataset.max_dt),
        "--missing-penalty-m",
        str(dataset.missing_penalty_m),
        "--min-coverage-ok",
        str(dataset.min_coverage_ok),
        "--stride",
        str(dataset.stride),
        "--backend-thresh",
        str(dataset.backend_thresh),
        "--image-height",
        str(dataset.image_height),
        "--image-width",
        str(dataset.image_width),
        "--frontend-mode",
        str(method.frontend_mode),
    ]
    if method.frontend_config is not None:
        cmd.extend(["--frontend-config", str(method.frontend_config)])
    if method.frontend_checkpoint is not None:
        cmd.extend(["--frontend-checkpoint", str(method.frontend_checkpoint)])
    opts = _split_dpvo_opts(dataset.dpvo_opts)
    if opts:
        cmd.extend(["--opts", *opts])
    return cmd


def _run_eval(
    *,
    python_bin: str,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    method: EvalMethodSpec,
    dataset: DatasetEvalSpec,
    output_dir: Path,
    resume: bool,
) -> Path:
    csv_path = output_dir / "metrics_summary.csv"
    if resume and _valid_existing_result(csv_path, dataset.sequences):
        return csv_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _evaluation_command(
        python_bin=python_bin,
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        method=method,
        dataset=dataset,
        output_dir=output_dir,
    )
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in cmd) + "\n",
        encoding="utf-8",
    )
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, text=True)
    return csv_path


def _summarize_dataset(csv_path: Path, dataset: DatasetEvalSpec) -> DatasetSummary:
    rows = _read_csv_rows(csv_path)
    sequences = _read_sequences(rows)
    if sequences != list(dataset.sequences):
        raise ValueError(
            f"{csv_path} sequence mismatch: expected {list(dataset.sequences)}, got {sequences}"
        )

    ate_values: list[float] = []
    rpe_t_values: list[float] = []
    rpe_r_values: list[float] = []
    scale_values: list[float] = []
    scale_abs_values: list[float] = []
    scale_log_values: list[float] = []
    coverage_values: list[float] = []
    pressure_values: list[float] = []
    kitti_trans_values: list[float] = []
    kitti_rot_values: list[float] = []
    ok_count = 0
    non_ok_count = 0
    failed_count = 0
    finite_count = 0
    pressure_set = set(dataset.pressure_sequences)

    for row in rows:
        status = str(row.get("status", "")).strip()
        if status == "ok":
            ok_count += 1
        else:
            non_ok_count += 1
        if status not in ALLOWED_PROXY_STATUSES:
            failed_count += 1

        ate = _safe_float(row.get("ate_rmse_associated"))
        rpe_t = _safe_float(row.get("rpe_trans_rmse"))
        rpe_r = _safe_float(row.get("rpe_rot_rmse"))
        scale = _safe_float(row.get("scale_correction"))
        scale_abs = _safe_float(row.get("scale_error_abs"))
        scale_log = _safe_float(row.get("scale_error_abs_log"))
        coverage = _safe_float(row.get("coverage"))
        kitti_trans = _safe_float(row.get("kitti_trans_percent"))
        kitti_rot = _safe_float(row.get("kitti_rot_deg_per_m"))

        if math.isfinite(ate):
            finite_count += 1
            ate_values.append(ate)
            if str(row.get("sequence", "")).strip() in pressure_set:
                pressure_values.append(ate)
        if math.isfinite(rpe_t):
            rpe_t_values.append(rpe_t)
        if math.isfinite(rpe_r):
            rpe_r_values.append(rpe_r)
        if math.isfinite(scale):
            scale_values.append(scale)
        if math.isfinite(scale_abs):
            scale_abs_values.append(scale_abs)
        if math.isfinite(scale_log):
            scale_log_values.append(scale_log)
        if math.isfinite(coverage):
            coverage_values.append(coverage)
        if math.isfinite(kitti_trans):
            kitti_trans_values.append(kitti_trans)
        if math.isfinite(kitti_rot):
            kitti_rot_values.append(kitti_rot)

    return DatasetSummary(
        dataset_id=dataset.dataset_id,
        row_count=len(rows),
        finite_count=finite_count,
        ok_count=ok_count,
        non_ok_count=non_ok_count,
        failed_count=failed_count,
        mean_ate_rmse_associated=_mean(ate_values),
        mean_rpe_trans_rmse=_mean(rpe_t_values),
        mean_rpe_rot_rmse=_mean(rpe_r_values),
        mean_scale_correction=_mean(scale_values),
        mean_scale_error_abs=_mean(scale_abs_values),
        mean_scale_error_abs_log=_mean(scale_log_values),
        mean_coverage=_mean(coverage_values),
        pressure_mean_ate_rmse_associated=_mean(pressure_values),
        mean_kitti_trans_percent=_mean(kitti_trans_values),
        mean_kitti_rot_deg_per_m=_mean(kitti_rot_values),
    )


def _dataset_summary_to_row(method: EvalMethodSpec, dataset: DatasetSummary) -> dict[str, object]:
    return {
        "method_id": method.method_id,
        "kind": method.kind,
        "dataset_id": dataset.dataset_id,
        "mean_ate_rmse_associated": _format_float(dataset.mean_ate_rmse_associated),
        "mean_rpe_trans_rmse": _format_float(dataset.mean_rpe_trans_rmse),
        "mean_rpe_rot_rmse": _format_float(dataset.mean_rpe_rot_rmse),
        "mean_scale_correction": _format_float(dataset.mean_scale_correction),
        "mean_scale_error_abs": _format_float(dataset.mean_scale_error_abs),
        "mean_scale_error_abs_log": _format_float(dataset.mean_scale_error_abs_log),
        "mean_coverage": _format_float(dataset.mean_coverage),
        "pressure_mean_ate_rmse_associated": _format_float(dataset.pressure_mean_ate_rmse_associated),
        "mean_kitti_trans_percent": _format_float(dataset.mean_kitti_trans_percent),
        "mean_kitti_rot_deg_per_m": _format_float(dataset.mean_kitti_rot_deg_per_m),
        "row_count": dataset.row_count,
        "finite_count": dataset.finite_count,
        "ok_count": dataset.ok_count,
        "non_ok_count": dataset.non_ok_count,
        "failed_count": dataset.failed_count,
    }


def _gpu_heavy_process_lines(exclude_pid: int | None = None) -> list[str]:
    patterns = (
        "refocus_vo.train_dino_dpvo_frontend",
        "refocus_vo.sweeps.run_assoc9_sweep",
        "refocus_vo.sweeps.run_assoc9_cross_dataset_sweep",
        "refocus_vo.eval.external_dpvo",
        "refocus_vo.eval.external_dpvo_euroc",
        "refocus_vo.eval.external_dpvo_kitti",
        "scripts/train_dino_dpvo_frontend.sh",
    )
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"],
        text=True,
        capture_output=True,
        check=True,
    )
    output: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, _ppid_text, cmd = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if exclude_pid is not None and pid == exclude_pid:
            continue
        if any(pattern in cmd for pattern in patterns):
            output.append(line)
    return output


def _rank_proxy_candidates(
    *,
    candidate_methods: list[EvalMethodSpec],
    summaries_by_method: dict[str, dict[str, DatasetSummary]],
    reference_ids: dict[str, str],
    dataset_weights: dict[str, float],
    tum_no_regression_multiplier: float,
) -> list[dict[str, object]]:
    reference_tum = summaries_by_method[reference_ids["tum"]]["tum"]
    reference_euroc = summaries_by_method[reference_ids["euroc"]]["euroc"]
    reference_kitti = summaries_by_method[reference_ids["kitti"]]["kitti"]

    rows: list[dict[str, object]] = []
    for method in candidate_methods:
        tum = summaries_by_method[method.method_id]["tum"]
        euroc = summaries_by_method[method.method_id]["euroc"]
        kitti = summaries_by_method[method.method_id]["kitti"]

        passes_tum_gate = (
            math.isfinite(tum.mean_ate_rmse_associated)
            and tum.failed_count == 0
            and tum.finite_count == tum.row_count
            and tum.mean_ate_rmse_associated
            <= float(tum_no_regression_multiplier) * float(reference_tum.mean_ate_rmse_associated)
        )
        passes_transfer_gate = (
            euroc.failed_count == 0
            and euroc.finite_count == euroc.row_count
            and kitti.failed_count == 0
            and kitti.finite_count == kitti.row_count
        )
        passes_proxy_gate = bool(passes_tum_gate and passes_transfer_gate)

        weighted_ate = (
            dataset_weights["tum"] * _ratio(tum.mean_ate_rmse_associated, reference_tum.mean_ate_rmse_associated)
            + dataset_weights["euroc"] * _ratio(euroc.mean_ate_rmse_associated, reference_euroc.mean_ate_rmse_associated)
            + dataset_weights["kitti"] * _ratio(kitti.mean_ate_rmse_associated, reference_kitti.mean_ate_rmse_associated)
        )
        weighted_rpe_trans = (
            dataset_weights["tum"] * _ratio(tum.mean_rpe_trans_rmse, reference_tum.mean_rpe_trans_rmse)
            + dataset_weights["euroc"] * _ratio(euroc.mean_rpe_trans_rmse, reference_euroc.mean_rpe_trans_rmse)
            + dataset_weights["kitti"] * _ratio(kitti.mean_rpe_trans_rmse, reference_kitti.mean_rpe_trans_rmse)
        )
        weighted_rpe_rot = (
            dataset_weights["tum"] * _ratio(tum.mean_rpe_rot_rmse, reference_tum.mean_rpe_rot_rmse)
            + dataset_weights["euroc"] * _ratio(euroc.mean_rpe_rot_rmse, reference_euroc.mean_rpe_rot_rmse)
            + dataset_weights["kitti"] * _ratio(kitti.mean_rpe_rot_rmse, reference_kitti.mean_rpe_rot_rmse)
        )
        weighted_scale = (
            dataset_weights["tum"] * _ratio(tum.mean_scale_error_abs_log, reference_tum.mean_scale_error_abs_log)
            + dataset_weights["euroc"] * _ratio(euroc.mean_scale_error_abs_log, reference_euroc.mean_scale_error_abs_log)
            + dataset_weights["kitti"] * _ratio(kitti.mean_scale_error_abs_log, reference_kitti.mean_scale_error_abs_log)
        )
        tum_pressure = _ratio(
            tum.pressure_mean_ate_rmse_associated,
            reference_tum.pressure_mean_ate_rmse_associated,
        )
        rows.append(
            {
                "method_id": method.method_id,
                "kind": method.kind,
                "checkpoint_path": str(method.checkpoint_path or ""),
                "config_path": str(method.runtime_config_path or ""),
                "passes_tum_gate": int(bool(passes_tum_gate)),
                "passes_transfer_gate": int(bool(passes_transfer_gate)),
                "passes_proxy_gate": int(bool(passes_proxy_gate)),
                "tum_mean_ate_rmse_associated": _format_float(tum.mean_ate_rmse_associated),
                "tum_pressure_mean_ate_rmse_associated": _format_float(tum.pressure_mean_ate_rmse_associated),
                "euroc_mean_ate_rmse_associated": _format_float(euroc.mean_ate_rmse_associated),
                "kitti_mean_ate_rmse_associated": _format_float(kitti.mean_ate_rmse_associated),
                "kitti_mean_kitti_trans_percent": _format_float(kitti.mean_kitti_trans_percent),
                "kitti_mean_kitti_rot_deg_per_m": _format_float(kitti.mean_kitti_rot_deg_per_m),
                "weighted_ate_score": _format_float(weighted_ate if passes_proxy_gate else math.inf),
                "weighted_rpe_trans_score": _format_float(weighted_rpe_trans if passes_proxy_gate else math.inf),
                "weighted_rpe_rot_score": _format_float(weighted_rpe_rot if passes_proxy_gate else math.inf),
                "weighted_scale_error_abs_log_score": _format_float(weighted_scale if passes_proxy_gate else math.inf),
                "tum_pressure_score": _format_float(tum_pressure if passes_proxy_gate else math.inf),
            }
        )

    rows.sort(
        key=lambda row: (
            0 if int(row["passes_proxy_gate"]) else 1,
            _safe_float(row["weighted_ate_score"]),
            _safe_float(row["weighted_rpe_trans_score"]),
            _safe_float(row["weighted_rpe_rot_score"]),
            _safe_float(row["weighted_scale_error_abs_log_score"]),
            _safe_float(row["tum_pressure_score"]),
            str(row["method_id"]),
        )
    )
    rank = 1
    for row in rows:
        row["proxy_rank"] = rank if int(row["passes_proxy_gate"]) else ""
        if int(row["passes_proxy_gate"]):
            rank += 1
    return rows


def _proxy_markdown(
    *,
    ranked_rows: list[dict[str, object]],
    baseline_summaries: list[dict[str, object]],
    top_k: int,
) -> str:
    lines = [
        "# Cross-Dataset Proxy Leaderboard",
        "",
        "## Candidate ranking",
        "",
        "| Rank | Method | TUM gate | Transfer gate | Weighted ATE | Weighted RPE(t) | Weighted RPE(r) | Weighted scale | TUM pressure |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked_rows:
        lines.append(
            f"| {row.get('proxy_rank', '')} | `{row['method_id']}` | {row['passes_tum_gate']} | {row['passes_transfer_gate']} | "
            f"{row['weighted_ate_score']} | {row['weighted_rpe_trans_score']} | {row['weighted_rpe_rot_score']} | "
            f"{row['weighted_scale_error_abs_log_score']} | {row['tum_pressure_score']} |"
        )
    lines.extend(
        [
            "",
            f"## Top {top_k} passing candidates",
            "",
        ]
    )
    passing = [row for row in ranked_rows if int(row["passes_proxy_gate"])]
    for row in passing[:top_k]:
        lines.append(
            f"- `{row['method_id']}`: weighted ATE `{row['weighted_ate_score']}`, "
            f"TUM `{row['tum_mean_ate_rmse_associated']}`, EuRoC `{row['euroc_mean_ate_rmse_associated']}`, "
            f"KITTI `{row['kitti_mean_ate_rmse_associated']}`"
        )
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
        if not int(row["passes_proxy_gate"]):
            continue
        method = by_id[str(row["method_id"])]
        output.append(method)
        if len(output) >= int(top_k):
            break
    return output


def _full_benchmark_rows(
    methods: list[EvalMethodSpec],
    summaries_by_method: dict[str, dict[str, DatasetSummary]],
    *,
    transfer_reference_id: str,
) -> list[dict[str, object]]:
    euroc_ref = summaries_by_method[transfer_reference_id]["euroc"]
    kitti_ref = summaries_by_method[transfer_reference_id]["kitti"]
    rows: list[dict[str, object]] = []
    for method in methods:
        tum = summaries_by_method[method.method_id]["tum"]
        euroc = summaries_by_method[method.method_id]["euroc"]
        kitti = summaries_by_method[method.method_id]["kitti"]
        transfer_score = (
            0.60 * _ratio(euroc.mean_ate_rmse_associated, euroc_ref.mean_ate_rmse_associated)
            + 0.40 * _ratio(kitti.mean_ate_rmse_associated, kitti_ref.mean_ate_rmse_associated)
        )
        rows.append(
            {
                "method_id": method.method_id,
                "kind": method.kind,
                "tum_mean_ate_rmse_associated": _format_float(tum.mean_ate_rmse_associated),
                "tum_mean_rpe_trans_rmse": _format_float(tum.mean_rpe_trans_rmse),
                "tum_mean_rpe_rot_rmse": _format_float(tum.mean_rpe_rot_rmse),
                "tum_mean_scale_error_abs_log": _format_float(tum.mean_scale_error_abs_log),
                "euroc_mean_ate_rmse_associated": _format_float(euroc.mean_ate_rmse_associated),
                "euroc_mean_rpe_trans_rmse": _format_float(euroc.mean_rpe_trans_rmse),
                "euroc_mean_rpe_rot_rmse": _format_float(euroc.mean_rpe_rot_rmse),
                "euroc_mean_scale_error_abs_log": _format_float(euroc.mean_scale_error_abs_log),
                "kitti_mean_ate_rmse_associated": _format_float(kitti.mean_ate_rmse_associated),
                "kitti_mean_rpe_trans_rmse": _format_float(kitti.mean_rpe_trans_rmse),
                "kitti_mean_rpe_rot_rmse": _format_float(kitti.mean_rpe_rot_rmse),
                "kitti_mean_scale_error_abs_log": _format_float(kitti.mean_scale_error_abs_log),
                "kitti_mean_kitti_trans_percent": _format_float(kitti.mean_kitti_trans_percent),
                "kitti_mean_kitti_rot_deg_per_m": _format_float(kitti.mean_kitti_rot_deg_per_m),
                "transfer_ate_score": _format_float(transfer_score),
            }
        )
    rows.sort(key=lambda row: (_safe_float(row["tum_mean_ate_rmse_associated"]), _safe_float(row["transfer_ate_score"])))
    return rows


def _select_full_winner(
    rows: list[dict[str, object]],
    candidate_ids: set[str],
    *,
    tum_tie_abs_threshold: float,
) -> dict[str, object]:
    candidate_rows = [row for row in rows if str(row["method_id"]) in candidate_ids]
    if not candidate_rows:
        raise RuntimeError("No finalist candidates available for full-benchmark winner selection")
    candidate_rows.sort(
        key=lambda row: (
            _safe_float(row["tum_mean_ate_rmse_associated"]),
            _safe_float(row["transfer_ate_score"]),
        )
    )
    best_tum = _safe_float(candidate_rows[0]["tum_mean_ate_rmse_associated"])
    tied = [
        row
        for row in candidate_rows
        if math.isfinite(_safe_float(row["tum_mean_ate_rmse_associated"]))
        and abs(_safe_float(row["tum_mean_ate_rmse_associated"]) - best_tum) <= float(tum_tie_abs_threshold)
    ]
    if len(tied) <= 1:
        return candidate_rows[0]
    tied.sort(key=lambda row: _safe_float(row["transfer_ate_score"]))
    return tied[0]


def _full_benchmark_markdown(
    *,
    rows: list[dict[str, object]],
    winner: dict[str, object],
    candidate_ids: set[str],
) -> str:
    lines = [
        "# Cross-Dataset Full Benchmark",
        "",
        f"Winner: `{winner['method_id']}`",
        "",
        "| Method | Candidate | TUM ATE | EuRoC ATE | KITTI ATE | KITTI trans % | KITTI rot deg/m | Transfer score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method_id']}` | {'yes' if str(row['method_id']) in candidate_ids else 'no'} | "
            f"{row['tum_mean_ate_rmse_associated']} | {row['euroc_mean_ate_rmse_associated']} | "
            f"{row['kitti_mean_ate_rmse_associated']} | {row['kitti_mean_kitti_trans_percent']} | "
            f"{row['kitti_mean_kitti_rot_deg_per_m']} | {row['transfer_ate_score']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _verify_gpu_idle(force: bool) -> None:
    busy = _gpu_heavy_process_lines(exclude_pid=os.getpid())
    if busy and not force:
        preview = "\n".join(busy[:8])
        raise RuntimeError(
            "Refusing to start cross-dataset sweep while other GPU-heavy jobs are active.\n"
            "Re-run with --force if you want to override.\n"
            f"Active processes:\n{preview}"
        )


def _runner_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    runner_cfg = manifest.get("runner", {}) or {}
    dpvo_root = _resolve_path(runner_cfg.get("dpvo_root", "refocus_vo/external/repos/DPVO"), REPO_ROOT)
    dpvo_weights = _resolve_path(
        runner_cfg.get("dpvo_weights_path", str(dpvo_root / "dpvo.pth")),
        REPO_ROOT,
    )
    dpvo_config = _resolve_path(runner_cfg.get("dpvo_config_path", str(dpvo_root / "config" / "default.yaml")), REPO_ROOT)
    return dpvo_root, dpvo_weights, dpvo_config


def _evaluate_method_group(
    *,
    python_bin: str,
    methods: list[EvalMethodSpec],
    datasets: dict[str, DatasetEvalSpec],
    stage_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    resume: bool,
) -> dict[str, dict[str, DatasetSummary]]:
    summaries: dict[str, dict[str, DatasetSummary]] = {}
    dataset_summary_rows: list[dict[str, object]] = []
    for method in methods:
        method_summaries: dict[str, DatasetSummary] = {}
        for dataset_id, dataset in datasets.items():
            output_dir = stage_root / method.method_id / dataset_id
            csv_path = _run_eval(
                python_bin=python_bin,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                method=method,
                dataset=dataset,
                output_dir=output_dir,
                resume=resume,
            )
            summary = _summarize_dataset(csv_path, dataset)
            method_summaries[dataset_id] = summary
            dataset_summary_rows.append(_dataset_summary_to_row(method, summary))
        summaries[method.method_id] = method_summaries

    _write_csv(
        stage_root / "dataset_summary.csv",
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
    return summaries


def _build_live_proxy_eval_cfg(
    *,
    proxy_datasets: dict[str, DatasetEvalSpec],
    baseline_summaries: dict[str, dict[str, DatasetSummary]],
    reference_ids: dict[str, str],
    dataset_weights: dict[str, float],
    win_weights: dict[str, float] | None = None,
    tum_no_regression_multiplier: float,
    sequence_assoc_baselines: dict[str, dict[str, float]] | None = None,
    gate_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    dataset_ids = [str(dataset_id) for dataset_id in proxy_datasets]
    gate_cfg = dict(gate_overrides or {})
    return {
        "enabled": True,
        "weights": {
            dataset_id: float(dataset_weights.get(dataset_id, 0.0))
            for dataset_id in dataset_ids
        },
        "win_weights": {
            dataset_id: float((win_weights or {}).get(dataset_id, 0.0))
            for dataset_id in dataset_ids
        },
        "tum_no_regression_multiplier": float(tum_no_regression_multiplier),
        "gate": {
            "tum_no_regression_multiplier": float(tum_no_regression_multiplier),
            **gate_cfg,
        },
        "references": {
            "tum_mean_ate_rmse_associated": float(
                baseline_summaries[reference_ids["tum"]]["tum"].mean_ate_rmse_associated
            ),
            "tum_pressure_mean_ate_rmse_associated": float(
                baseline_summaries[reference_ids["tum"]]["tum"].pressure_mean_ate_rmse_associated
            ),
            "tum_mean_rpe_trans_rmse": float(
                baseline_summaries[reference_ids["tum"]]["tum"].mean_rpe_trans_rmse
            ),
            "tum_mean_rpe_rot_rmse": float(
                baseline_summaries[reference_ids["tum"]]["tum"].mean_rpe_rot_rmse
            ),
            "tum_mean_scale_error_abs_log": float(
                baseline_summaries[reference_ids["tum"]]["tum"].mean_scale_error_abs_log
            ),
            "euroc_mean_ate_rmse_associated": float(
                baseline_summaries[reference_ids["euroc"]]["euroc"].mean_ate_rmse_associated
            ),
            "euroc_mean_rpe_trans_rmse": float(
                baseline_summaries[reference_ids["euroc"]]["euroc"].mean_rpe_trans_rmse
            ),
            "euroc_mean_rpe_rot_rmse": float(
                baseline_summaries[reference_ids["euroc"]]["euroc"].mean_rpe_rot_rmse
            ),
            "euroc_mean_scale_error_abs_log": float(
                baseline_summaries[reference_ids["euroc"]]["euroc"].mean_scale_error_abs_log
            ),
            **(
                {
                    "kitti_mean_ate_rmse_associated": float(
                        baseline_summaries[reference_ids["kitti"]]["kitti"].mean_ate_rmse_associated
                    ),
                    "kitti_mean_rpe_trans_rmse": float(
                        baseline_summaries[reference_ids["kitti"]]["kitti"].mean_rpe_trans_rmse
                    ),
                    "kitti_mean_rpe_rot_rmse": float(
                        baseline_summaries[reference_ids["kitti"]]["kitti"].mean_rpe_rot_rmse
                    ),
                    "kitti_mean_scale_error_abs_log": float(
                        baseline_summaries[reference_ids["kitti"]]["kitti"].mean_scale_error_abs_log
                    ),
                }
                if "kitti" in dataset_ids and "kitti" in reference_ids
                else {}
            ),
            "sequence_assoc_baselines": deepcopy(sequence_assoc_baselines or {}),
        },
        "datasets": {
            dataset_id: {
                "sequences": list(spec.sequences),
                "pressure_sequences": list(spec.pressure_sequences),
                "max_dt": float(spec.max_dt),
                "missing_penalty_m": float(spec.missing_penalty_m),
                "min_coverage_ok": float(spec.min_coverage_ok),
                "image_height": int(spec.image_height),
                "image_width": int(spec.image_width),
                "stride": int(spec.stride),
                "backend_thresh": float(spec.backend_thresh),
                "dpvo_opts": str(spec.dpvo_opts),
                "frontend_mode": "dino_hybrid",
            }
            for dataset_id, spec in proxy_datasets.items()
        },
    }


def _load_live_candidate_proxy_summaries(
    *,
    candidate_methods: list[EvalMethodSpec],
    leaderboard_rows: list[dict[str, str]],
    datasets: dict[str, DatasetEvalSpec],
    base_output_dir: Path,
) -> dict[str, dict[str, DatasetSummary]]:
    row_by_id = {str(row["run_id"]): row for row in leaderboard_rows}
    summaries: dict[str, dict[str, DatasetSummary]] = {}
    for method in candidate_methods:
        row = row_by_id.get(method.method_id)
        if row is None:
            raise RuntimeError(f"Missing leaderboard row for candidate {method.method_id}")
        try:
            best_step = int(str(row.get("best_step", "")).strip())
        except Exception as exc:
            raise RuntimeError(f"Candidate {method.method_id} is missing a usable best_step") from exc
        step_dir = base_output_dir / "train" / method.method_id / "live_proxy_eval" / f"step_{best_step:06d}"
        method_summaries: dict[str, DatasetSummary] = {}
        for dataset_id, dataset in datasets.items():
            csv_path = step_dir / dataset_id / "metrics_summary.csv"
            if not csv_path.exists():
                raise RuntimeError(f"Missing live proxy metrics for {method.method_id} dataset={dataset_id}: {csv_path}")
            method_summaries[dataset_id] = _summarize_dataset(csv_path, dataset)
        summaries[method.method_id] = method_summaries
    return summaries


def _write_cross_dataset_state(path: Path, payload: dict[str, object]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the cross-dataset tri-proxy DINO-DPVO sweep.")
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
        else (REPO_ROOT / "refocus_vo" / "runs" / "sweeps" / str(manifest.get("name", "cross_dataset_sweep")))
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, base_output_dir / "manifest.yaml")

    if args.dry_run:
        _print_dry_run(manifest, runs[: int(args.limit_runs)] if args.limit_runs is not None else runs, base_output_dir)
        proxy_cfg = manifest.get("proxy_validation", {}) or {}
        full_cfg = manifest.get("full_benchmark", {}) or {}
        print("proxy datasets:")
        for dataset_id, spec in _load_dataset_specs(proxy_cfg).items():
            print(f"  - {dataset_id}: {','.join(spec.sequences)}")
        print("full datasets:")
        for dataset_id, spec in _load_dataset_specs(full_cfg).items():
            print(f"  - {dataset_id}: {','.join(spec.sequences)}")
        print("baseline methods:")
        for method in _baseline_methods(manifest):
            print(f"  - {method.method_id}: mode={method.frontend_mode}")
        return

    _verify_gpu_idle(bool(args.force))

    dpvo_root, dpvo_weights, dpvo_config = _runner_paths(manifest)
    proxy_cfg = manifest.get("proxy_validation", {}) or {}
    proxy_datasets = _load_dataset_specs(proxy_cfg)
    baseline_methods = _baseline_methods(manifest)
    references = proxy_cfg.get("references", {}) or {}
    dataset_weights = {
        str(key): float(value)
        for key, value in ((proxy_cfg.get("ranking", {}) or {}).get("weights", {}) or {}).items()
    }
    tum_no_regression_multiplier = float((proxy_cfg.get("gate", {}) or {}).get("tum_no_regression_multiplier", 1.03))
    use_live_training_eval = bool(proxy_cfg.get("use_live_training_eval", False))

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

    training_manifest = deepcopy(manifest)
    if use_live_training_eval:
        training_manifest.setdefault("sweep", {}).setdefault("config_overrides", {}).setdefault("eval", {})
        training_manifest["sweep"]["config_overrides"]["eval"]["selection_metric"] = "tri_proxy_score"
        training_manifest["sweep"]["config_overrides"]["eval"]["live_proxy"] = _build_live_proxy_eval_cfg(
            proxy_datasets=proxy_datasets,
            baseline_summaries=baseline_summaries,
            reference_ids={
                "tum": str(references["tum"]),
                "euroc": str(references["euroc"]),
                "kitti": str(references["kitti"]),
            },
            dataset_weights=dataset_weights,
            tum_no_regression_multiplier=tum_no_regression_multiplier,
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

    candidate_proxy_summaries = (
        _load_live_candidate_proxy_summaries(
            candidate_methods=candidate_methods,
            leaderboard_rows=leaderboard_rows,
            datasets=proxy_datasets,
            base_output_dir=base_output_dir,
        )
        if use_live_training_eval
        else _evaluate_method_group(
            python_bin=sys.executable,
            methods=candidate_methods,
            datasets=proxy_datasets,
            stage_root=base_output_dir / "proxy_eval" / "candidates",
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            resume=bool(args.resume),
        )
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
        reference_ids={
            "tum": str(references["tum"]),
            "euroc": str(references["euroc"]),
            "kitti": str(references["kitti"]),
        },
        dataset_weights=dataset_weights,
        tum_no_regression_multiplier=tum_no_regression_multiplier,
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
            "passes_transfer_gate",
            "passes_proxy_gate",
            "tum_mean_ate_rmse_associated",
            "tum_pressure_mean_ate_rmse_associated",
            "euroc_mean_ate_rmse_associated",
            "kitti_mean_ate_rmse_associated",
            "kitti_mean_kitti_trans_percent",
            "kitti_mean_kitti_rot_deg_per_m",
            "weighted_ate_score",
            "weighted_rpe_trans_score",
            "weighted_rpe_rot_score",
            "weighted_scale_error_abs_log_score",
            "tum_pressure_score",
        ],
    )
    baseline_rows = [
        _dataset_summary_to_row(
            next(method for method in baseline_methods if method.method_id == method_id),
            summary,
        )
        for method_id, summaries in proxy_summaries.items()
        if any(method.method_id == method_id for method in baseline_methods)
        for summary in summaries.values()
    ]
    _write_text(
        base_output_dir / "proxy_leaderboard.md",
        _proxy_markdown(
            ranked_rows=ranked_rows,
            baseline_summaries=baseline_rows,
            top_k=int(proxy_cfg.get("top_k", 3)),
        ),
    )

    top_candidates = _top_candidate_methods(
        candidate_methods,
        ranked_rows,
        int(proxy_cfg.get("top_k", 3)),
    )
    if len(top_candidates) < int(proxy_cfg.get("top_k", 3)):
        raise RuntimeError("Fewer than the requested number of candidates passed the proxy gates")

    full_cfg = manifest.get("full_benchmark", {}) or {}
    full_datasets = _load_dataset_specs(full_cfg)
    full_baseline_methods = _baseline_methods(manifest, section_name="full_benchmark") or baseline_methods
    full_methods = full_baseline_methods + top_candidates
    full_summaries = _evaluate_method_group(
        python_bin=sys.executable,
        methods=full_methods,
        datasets=full_datasets,
        stage_root=base_output_dir / "finalists" / "full_eval",
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        resume=bool(args.resume),
    )

    transfer_reference_id = str((full_cfg.get("references", {}) or {}).get("transfer", references["euroc"]))
    full_rows = _full_benchmark_rows(
        full_methods,
        full_summaries,
        transfer_reference_id=transfer_reference_id,
    )
    candidate_ids = {method.method_id for method in top_candidates}
    winner = _select_full_winner(
        full_rows,
        candidate_ids,
        tum_tie_abs_threshold=float(full_cfg.get("tum_tie_abs_threshold", 0.005)),
    )
    _write_csv(
        base_output_dir / "finalists" / "full_method_comparison.csv",
        full_rows,
        [
            "method_id",
            "kind",
            "tum_mean_ate_rmse_associated",
            "tum_mean_rpe_trans_rmse",
            "tum_mean_rpe_rot_rmse",
            "tum_mean_scale_error_abs_log",
            "euroc_mean_ate_rmse_associated",
            "euroc_mean_rpe_trans_rmse",
            "euroc_mean_rpe_rot_rmse",
            "euroc_mean_scale_error_abs_log",
            "kitti_mean_ate_rmse_associated",
            "kitti_mean_rpe_trans_rmse",
            "kitti_mean_rpe_rot_rmse",
            "kitti_mean_scale_error_abs_log",
            "kitti_mean_kitti_trans_percent",
            "kitti_mean_kitti_rot_deg_per_m",
            "transfer_ate_score",
        ],
    )
    _write_text(
        base_output_dir / "finalists" / "full_method_comparison.md",
        _full_benchmark_markdown(
            rows=full_rows,
            winner=winner,
            candidate_ids=candidate_ids,
        ),
    )
    _write_cross_dataset_state(
        base_output_dir / STATE_FILENAME,
        {
            "manifest": str(manifest_path),
            "proxy_top_candidates": [method.method_id for method in top_candidates],
            "winner_method_id": str(winner["method_id"]),
            "winner_row": dict(winner),
        },
    )
    print(f"[cross_dataset_sweep] winner={winner['method_id']}")
    print(f"[cross_dataset_sweep] output_root={base_output_dir}")


if __name__ == "__main__":
    main()
