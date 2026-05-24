from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import signal
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUBTREE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = SUBTREE_ROOT.parent
STATE_FILENAME = "state.yaml"

DEV_NUMERIC_FIELDS = [
    "external_mean_ate",
    "external_mean_ate_associated",
    "external_mean_coverage",
    "lowtex_mean_ate",
    "lowtex_mean_ate_associated",
    "lowtex_mean_coverage",
    "best_pure_assoc",
    "best_hybrid_assoc",
    "best_pure_lowtex_assoc",
    "best_hybrid_lowtex_assoc",
    "repeated_patch_fraction",
    "mean_dedupe_radius_used",
    "mean_grid_occupancy",
    "mean_track_age",
    "survival_rate_1",
    "survival_rate_3",
    "survival_rate_5",
    "unstable_motion_proxy_fraction",
    "dino_patch_fraction",
    "tum_proxy_mean_ate",
    "tum_proxy_mean_ate_associated",
    "tum_proxy_mean_coverage",
    "tum_proxy_mean_rpe_trans_rmse",
    "tum_proxy_mean_rpe_rot_rmse",
    "tum_proxy_mean_scale_correction",
    "tum_proxy_mean_scale_error_abs",
    "tum_proxy_mean_scale_error_abs_log",
    "tum_proxy_failed_count",
    "tum_proxy_finite_count",
    "tum_proxy_row_count",
    "tum_proxy_wins_vs_dpvo",
    "tum_pressure_wins_vs_dpvo",
    "tum_pressure_mean_ate_associated",
    "tum_pressure_mean_coverage",
    "euroc_proxy_mean_ate",
    "euroc_proxy_mean_ate_associated",
    "euroc_proxy_mean_coverage",
    "euroc_proxy_mean_rpe_trans_rmse",
    "euroc_proxy_mean_rpe_rot_rmse",
    "euroc_proxy_mean_scale_correction",
    "euroc_proxy_mean_scale_error_abs",
    "euroc_proxy_mean_scale_error_abs_log",
    "euroc_proxy_failed_count",
    "euroc_proxy_finite_count",
    "euroc_proxy_row_count",
    "euroc_proxy_wins_vs_dpvo",
    "kitti_proxy_mean_ate",
    "kitti_proxy_mean_ate_associated",
    "kitti_proxy_mean_coverage",
    "kitti_proxy_mean_rpe_trans_rmse",
    "kitti_proxy_mean_rpe_rot_rmse",
    "kitti_proxy_mean_scale_correction",
    "kitti_proxy_mean_scale_error_abs",
    "kitti_proxy_mean_scale_error_abs_log",
    "kitti_proxy_mean_kitti_trans_percent",
    "kitti_proxy_mean_kitti_rot_deg_per_m",
    "kitti_proxy_failed_count",
    "kitti_proxy_finite_count",
    "kitti_proxy_row_count",
    "live_tri_proxy_score",
    "live_weighted_rpe_trans_score",
    "live_weighted_rpe_rot_score",
    "live_weighted_scale_error_abs_log_score",
    "live_tum_gate_pass",
    "live_transfer_gate_pass",
    "live_dual_proxy_score",
    "live_pure_tum_proxy_score",
    "best_dual_proxy_score",
    "best_tum_proxy_wins_vs_dpvo",
    "best_euroc_proxy_wins_vs_dpvo",
    "dual_selection_passed_gate",
]

LEADERBOARD_EXTRA_FIELDS = [
    "best_pure_tum_score",
    "best_tum_proxy_assoc",
    "best_tum_pressure_assoc",
    "best_euroc_proxy_assoc",
    "best_tum_proxy_wins_vs_dpvo",
    "best_euroc_proxy_wins_vs_dpvo",
    "best_kitti_proxy_assoc",
    "best_dino_patch_fraction",
    "best_dual_score",
    "best_dual_gate",
    "best_tri_score",
    "best_tri_gate",
    "last_pure_tum_score",
    "last_tum_proxy_assoc",
    "last_tum_pressure_assoc",
    "last_euroc_proxy_assoc",
    "last_tum_proxy_wins_vs_dpvo",
    "last_euroc_proxy_wins_vs_dpvo",
    "last_kitti_proxy_assoc",
    "last_dino_patch_fraction",
    "last_dual_score",
    "last_dual_gate",
    "last_tri_score",
    "last_tri_gate",
]


@dataclass
class SweepRunSpec:
    run_id: str
    config_path: Path
    subset_config: Path
    init_checkpoint: Path
    init_mode: str
    expected_eval_mode: str
    stop_thresholds: dict[int, dict[str, float]]
    config_overrides: dict[str, Any]


@dataclass
class PackCandidate:
    candidate_id: str
    source_run_id: str
    kind: str
    checkpoint: Path
    config_path: Path


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return math.nan
    return number


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": _now_timestamp(), "runs": {}}
    payload = _load_yaml(path)
    payload.setdefault("updated_at", _now_timestamp())
    payload.setdefault("runs", {})
    if not isinstance(payload["runs"], dict):
        payload["runs"] = {}
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    payload = deepcopy(payload)
    payload["updated_at"] = _now_timestamp()
    _write_yaml(path, payload)


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[SweepRunSpec]]:
    payload = _load_yaml(path)
    runs = []
    for item in payload.get("runs", []):
        stop_thresholds: dict[int, dict[str, float]] = {}
        for step, value in (item.get("stop_thresholds", {}) or {}).items():
            if isinstance(value, dict):
                threshold_payload = {}
                if value.get("paper_assoc", value.get("primary_assoc")) not in (None, ""):
                    threshold_payload["paper_assoc"] = float(value.get("paper_assoc", value.get("primary_assoc")))
                if value.get("lowtex_assoc", value.get("secondary_assoc")) not in (None, ""):
                    threshold_payload["lowtex_assoc"] = float(value.get("lowtex_assoc", value.get("secondary_assoc")))
                stop_thresholds[int(step)] = threshold_payload
            else:
                stop_thresholds[int(step)] = {"paper_assoc": float(value)}
        runs.append(
            SweepRunSpec(
                run_id=str(item["run_id"]),
                config_path=_resolve_path(item["config"], REPO_ROOT),
                subset_config=_resolve_path(item["subset_config"], REPO_ROOT),
                init_checkpoint=_resolve_path(item["init_checkpoint"], REPO_ROOT),
                init_mode=str(item.get("init_mode", "partial")),
                expected_eval_mode=str(item.get("expected_eval_mode", "pure100")),
                stop_thresholds=stop_thresholds,
                config_overrides=deepcopy(item.get("config_overrides", {}) or {}),
            )
        )
    return payload, runs


def _read_dev_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    if not metrics_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("split", "")).strip() != "dev":
                continue
            parsed = {"step": int(row.get("step", 0) or 0)}
            for key in DEV_NUMERIC_FIELDS:
                parsed[key] = _safe_float(row.get(key))
            parsed["best_mode"] = str(row.get("best_mode", "")).strip() or "pure100"
            parsed["selection_metric"] = str(row.get("selection_metric", "")).strip().lower()
            parsed["selection_passed_gate"] = str(row.get("selection_passed_gate", "")).strip()
            rows.append(parsed)
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def _last_logged_step(metrics_csv: Path) -> int:
    if not metrics_csv.exists():
        return 0
    last_step = 0
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last_step = max(last_step, int(row.get("step", 0) or 0))
            except Exception:
                continue
    return last_step


def _secondary_assoc_guardrail(evaluation_cfg: dict[str, Any] | None) -> float | None:
    payload = evaluation_cfg or {}
    raw = payload.get("secondary_assoc_guardrail")
    if raw in (None, ""):
        return None
    return float(raw)


def _secondary_coverage_gate(evaluation_cfg: dict[str, Any] | None, coverage_gate: float) -> float | None:
    payload = evaluation_cfg or {}
    if "secondary_coverage_gate" not in payload and _secondary_assoc_guardrail(payload) is None:
        return None
    raw = payload.get("secondary_coverage_gate", coverage_gate)
    return float(raw)


def _selection_metric_name(row: dict[str, Any]) -> str:
    return str(row.get("selection_metric", "")).strip().lower()


def _selection_passed_gate(row: dict[str, Any]) -> bool:
    return str(row.get("selection_passed_gate", "")).strip().lower() not in {"", "0", "false", "nan", "none"}


def _row_key(
    row: dict[str, Any],
    coverage_gate: float,
    *,
    secondary_coverage_gate: float | None = None,
    secondary_assoc_guardrail: float | None = None,
) -> tuple[float, ...] | None:
    if _selection_metric_name(row) in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}:
        pure_tum_score = _safe_float(row.get("live_pure_tum_proxy_score"))
        if not (_selection_passed_gate(row) and math.isfinite(pure_tum_score)):
            return None
        tum_wins = _safe_float(row.get("tum_proxy_wins_vs_dpvo"))
        tum_wins_key = -tum_wins if math.isfinite(tum_wins) else math.inf
        pressure_wins = _safe_float(row.get("tum_pressure_wins_vs_dpvo"))
        pressure_wins_key = -pressure_wins if math.isfinite(pressure_wins) else math.inf
        return (
            pure_tum_score,
            _safe_float(row.get("tum_pressure_mean_ate_associated")) if math.isfinite(_safe_float(row.get("tum_pressure_mean_ate_associated"))) else math.inf,
            tum_wins_key,
            pressure_wins_key,
            _safe_float(row.get("mean_track_age")) if math.isfinite(_safe_float(row.get("mean_track_age"))) else math.inf,
            _safe_float(row.get("repeated_patch_fraction")) if math.isfinite(_safe_float(row.get("repeated_patch_fraction"))) else math.inf,
        )
    if _selection_metric_name(row) in {"dual_proxy_score", "live_dual_proxy_score"}:
        dual_score = _safe_float(row.get("live_dual_proxy_score"))
        if not (_selection_passed_gate(row) and math.isfinite(dual_score)):
            return None
        euroc_wins = _safe_float(row.get("euroc_proxy_wins_vs_dpvo"))
        euroc_wins_key = -euroc_wins if math.isfinite(euroc_wins) else math.inf
        return (
            dual_score,
            _safe_float(row.get("tum_pressure_mean_ate_associated")) if math.isfinite(_safe_float(row.get("tum_pressure_mean_ate_associated"))) else math.inf,
            euroc_wins_key,
            _safe_float(row.get("live_weighted_rpe_trans_score")) if math.isfinite(_safe_float(row.get("live_weighted_rpe_trans_score"))) else math.inf,
            _safe_float(row.get("live_weighted_rpe_rot_score")) if math.isfinite(_safe_float(row.get("live_weighted_rpe_rot_score"))) else math.inf,
            _safe_float(row.get("live_weighted_scale_error_abs_log_score")) if math.isfinite(_safe_float(row.get("live_weighted_scale_error_abs_log_score"))) else math.inf,
        )
    if _selection_metric_name(row) in {"tri_proxy_score", "live_tri_proxy_score"}:
        tri_score = _safe_float(row.get("live_tri_proxy_score"))
        if not (_selection_passed_gate(row) and math.isfinite(tri_score)):
            return None
        return (
            tri_score,
            _safe_float(row.get("live_weighted_rpe_trans_score")) if math.isfinite(_safe_float(row.get("live_weighted_rpe_trans_score"))) else math.inf,
            _safe_float(row.get("live_weighted_rpe_rot_score")) if math.isfinite(_safe_float(row.get("live_weighted_rpe_rot_score"))) else math.inf,
            _safe_float(row.get("live_weighted_scale_error_abs_log_score")) if math.isfinite(_safe_float(row.get("live_weighted_scale_error_abs_log_score"))) else math.inf,
            _safe_float(row.get("tum_pressure_mean_ate_associated")) if math.isfinite(_safe_float(row.get("tum_pressure_mean_ate_associated"))) else math.inf,
        )

    assoc = _safe_float(row.get("external_mean_ate_associated"))
    ate = _safe_float(row.get("external_mean_ate"))
    coverage = _safe_float(row.get("external_mean_coverage"))
    if not (math.isfinite(assoc) and math.isfinite(coverage) and coverage >= float(coverage_gate)):
        return None
    ate_key = ate if math.isfinite(ate) else math.inf
    repeated = _safe_float(row.get("repeated_patch_fraction"))
    repeated_key = repeated if math.isfinite(repeated) else math.inf
    if secondary_coverage_gate is None and secondary_assoc_guardrail is None:
        return (assoc, repeated_key, ate_key, -coverage)

    lowtex_assoc = _safe_float(row.get("lowtex_mean_ate_associated"))
    lowtex_coverage = _safe_float(row.get("lowtex_mean_coverage"))
    lowtex_gate = float(secondary_coverage_gate if secondary_coverage_gate is not None else coverage_gate)
    lowtex_guardrail = float(secondary_assoc_guardrail if secondary_assoc_guardrail is not None else math.inf)
    if not (
        math.isfinite(lowtex_assoc)
        and math.isfinite(lowtex_coverage)
        and lowtex_coverage >= lowtex_gate
        and lowtex_assoc <= lowtex_guardrail
    ):
        return None
    return (assoc, lowtex_assoc, ate_key, repeated_key, -coverage, -lowtex_coverage)


def _best_dev_row(
    rows: list[dict[str, Any]],
    coverage_gate: float,
    *,
    max_step: int | None = None,
    secondary_coverage_gate: float | None = None,
    secondary_assoc_guardrail: float | None = None,
) -> dict[str, Any] | None:
    eligible = []
    for row in rows:
        if max_step is not None and int(row["step"]) > int(max_step):
            continue
        key = _row_key(
            row,
            coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
        )
        if key is not None:
            eligible.append((key, row))
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0])
    return eligible[0][1]


def _usable_step_row(
    rows: list[dict[str, Any]],
    step: int,
    *,
    require_secondary: bool = False,
) -> dict[str, Any] | None:
    for row in rows:
        if int(row["step"]) != int(step):
            continue
        if _selection_metric_name(row) in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}:
            if (
                _selection_passed_gate(row)
                and math.isfinite(_safe_float(row.get("tum_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("tum_pressure_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("live_pure_tum_proxy_score")))
                and _safe_float(row.get("dino_patch_fraction")) >= 0.999
            ):
                return row
            continue
        if _selection_metric_name(row) in {"dual_proxy_score", "live_dual_proxy_score"}:
            if (
                _selection_passed_gate(row)
                and math.isfinite(_safe_float(row.get("tum_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("euroc_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("live_dual_proxy_score")))
            ):
                return row
            continue
        if _selection_metric_name(row) in {"tri_proxy_score", "live_tri_proxy_score"}:
            if (
                _selection_passed_gate(row)
                and math.isfinite(_safe_float(row.get("tum_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("euroc_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("kitti_proxy_mean_ate_associated")))
                and math.isfinite(_safe_float(row.get("live_tri_proxy_score")))
            ):
                return row
            continue
        assoc = _safe_float(row.get("external_mean_ate_associated"))
        coverage = _safe_float(row.get("external_mean_coverage"))
        lowtex_assoc = _safe_float(row.get("lowtex_mean_ate_associated"))
        lowtex_coverage = _safe_float(row.get("lowtex_mean_coverage"))
        if math.isfinite(assoc) and math.isfinite(coverage) and (
            not require_secondary or (math.isfinite(lowtex_assoc) and math.isfinite(lowtex_coverage))
        ):
            return row
    return None


def _required_usable_dev_steps(sweep_cfg: dict[str, Any]) -> list[int]:
    raw_steps = sweep_cfg.get("required_usable_dev_steps", [1000])
    if raw_steps in (None, ""):
        raw_steps = [1000]
    steps = {
        int(step)
        for step in raw_steps
        if str(step).strip() and int(step) > 0
    }
    return sorted(steps)


def _latest_dev_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[-1] if rows else None


def _sequence_set(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return {str(value).strip() for value in (values or []) if str(value).strip()}


def _worse_on_both_streak(
    rows: list[dict[str, Any]],
    *,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
) -> int:
    best_paper = math.inf
    best_secondary = math.inf
    streak = 0
    for row in rows:
        paper = _safe_float(row.get("external_mean_ate_associated"))
        paper_cov = _safe_float(row.get("external_mean_coverage"))
        secondary = _safe_float(row.get("lowtex_mean_ate_associated"))
        secondary_cov = _safe_float(row.get("lowtex_mean_coverage"))
        paper_ok = math.isfinite(paper) and math.isfinite(paper_cov) and paper_cov >= float(coverage_gate)
        secondary_ok = (
            math.isfinite(secondary)
            and math.isfinite(secondary_cov)
            and secondary_cov >= float(secondary_coverage_gate if secondary_coverage_gate is not None else coverage_gate)
        )
        if not (paper_ok and secondary_ok):
            continue
        if math.isfinite(best_paper) and math.isfinite(best_secondary) and paper > best_paper and secondary > best_secondary:
            streak += 1
        else:
            streak = 0
        if paper_ok:
            best_paper = min(best_paper, paper)
        if secondary_ok:
            best_secondary = min(best_secondary, secondary)

    return streak


def _serialize_dev_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = {
        "step": int(row.get("step", 0) or 0),
        "best_mode": str(row.get("best_mode", "")).strip() or "pure100",
        "selection_metric": _selection_metric_name(row),
        "selection_passed_gate": str(row.get("selection_passed_gate", "")).strip(),
    }
    for key in DEV_NUMERIC_FIELDS:
        payload[key] = _safe_float(row.get(key))
    return payload


def _checkpoint_path_for_best_mode(best_row: dict[str, Any] | None, output_dir: Path) -> Path:
    mode = str((best_row or {}).get("best_mode", "")).strip().lower()
    if mode == "hybrid":
        hybrid_path = output_dir / "best_hybrid.pt"
        if hybrid_path.exists():
            return hybrid_path
    pure_path = output_dir / "best_pure100.pt"
    if pure_path.exists():
        return pure_path
    return output_dir / "best.pt"


def _row_metric_text(row: dict[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    value = _safe_float(row.get(key))
    return "" if not math.isfinite(value) else f"{value:.6f}"


def _state_entry_template(
    run: SweepRunSpec,
    *,
    output_dir: Path,
    runtime_config_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": "pending",
        "output_dir": str(output_dir),
        "runtime_config_path": str(runtime_config_path),
        "process_pid": None,
        "process_pgid": None,
        "start_time": None,
        "last_update_time": _now_timestamp(),
        "best_dev": None,
        "latest_dev": None,
    }


def _update_state_entry(
    entry: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    status: str,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
    secondary_assoc_guardrail: float | None,
    output_dir: Path,
    runtime_config_path: Path,
    process_pid: int | None,
    process_pgid: int | None,
    start_time: str | None = None,
) -> None:
    entry["status"] = status
    entry["output_dir"] = str(output_dir)
    entry["runtime_config_path"] = str(runtime_config_path)
    entry["process_pid"] = int(process_pid) if process_pid else None
    entry["process_pgid"] = int(process_pgid) if process_pgid else None
    entry["start_time"] = start_time or entry.get("start_time")
    entry["last_update_time"] = _now_timestamp()
    entry["best_dev"] = _serialize_dev_row(
        _best_dev_row(
            rows,
            coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
        )
    )
    entry["latest_dev"] = _serialize_dev_row(_latest_dev_row(rows))


def _state_entry_to_leaderboard_row(run: SweepRunSpec, entry: dict[str, Any]) -> dict[str, str]:
    best_row = entry.get("best_dev") or {}
    latest_row = entry.get("latest_dev") or {}
    checkpoint_path = _checkpoint_path_for_best_mode(best_row, Path(entry.get("output_dir", "")))
    output = {
        "run_id": run.run_id,
        "status": str(entry.get("status", "pending")),
        "expected_eval_mode": run.expected_eval_mode,
        "config_path": str(entry.get("runtime_config_path", run.config_path)),
        "subset_config": str(run.subset_config),
        "init_checkpoint": str(run.init_checkpoint),
        "best_step": "" if not best_row else str(int(best_row.get("step", 0) or 0)),
        "best_assoc": _row_metric_text(best_row, "external_mean_ate_associated"),
        "best_ate": _row_metric_text(best_row, "external_mean_ate"),
        "best_coverage": _row_metric_text(best_row, "external_mean_coverage"),
        "best_lowtex_assoc": _row_metric_text(best_row, "lowtex_mean_ate_associated"),
        "best_lowtex_coverage": _row_metric_text(best_row, "lowtex_mean_coverage"),
        "best_mode": "" if not best_row else str(best_row.get("best_mode", "pure100")),
        "best_pure_assoc": _row_metric_text(best_row, "best_pure_assoc"),
        "best_hybrid_assoc": _row_metric_text(best_row, "best_hybrid_assoc"),
        "last_step": "" if not latest_row else str(int(latest_row.get("step", 0) or 0)),
        "last_assoc": _row_metric_text(latest_row, "external_mean_ate_associated"),
        "last_ate": _row_metric_text(latest_row, "external_mean_ate"),
        "last_coverage": _row_metric_text(latest_row, "external_mean_coverage"),
        "last_lowtex_assoc": _row_metric_text(latest_row, "lowtex_mean_ate_associated"),
        "last_lowtex_coverage": _row_metric_text(latest_row, "lowtex_mean_coverage"),
        "checkpoint_path": str(checkpoint_path),
    }
    output.update(
        {
            "best_pure_tum_score": _row_metric_text(best_row, "live_pure_tum_proxy_score"),
            "best_tum_proxy_assoc": _row_metric_text(best_row, "tum_proxy_mean_ate_associated"),
            "best_tum_pressure_assoc": _row_metric_text(best_row, "tum_pressure_mean_ate_associated"),
            "best_euroc_proxy_assoc": _row_metric_text(best_row, "euroc_proxy_mean_ate_associated"),
            "best_tum_proxy_wins_vs_dpvo": _row_metric_text(best_row, "tum_proxy_wins_vs_dpvo"),
            "best_euroc_proxy_wins_vs_dpvo": _row_metric_text(best_row, "euroc_proxy_wins_vs_dpvo"),
            "best_kitti_proxy_assoc": _row_metric_text(best_row, "kitti_proxy_mean_ate_associated"),
            "best_dino_patch_fraction": _row_metric_text(best_row, "dino_patch_fraction"),
            "best_dual_score": _row_metric_text(best_row, "live_dual_proxy_score"),
            "best_dual_gate": "" if not best_row else str(int(_selection_passed_gate(best_row))),
            "best_tri_score": _row_metric_text(best_row, "live_tri_proxy_score"),
            "best_tri_gate": "" if not best_row else str(int(_selection_passed_gate(best_row))),
            "last_pure_tum_score": _row_metric_text(latest_row, "live_pure_tum_proxy_score"),
            "last_tum_proxy_assoc": _row_metric_text(latest_row, "tum_proxy_mean_ate_associated"),
            "last_tum_pressure_assoc": _row_metric_text(latest_row, "tum_pressure_mean_ate_associated"),
            "last_euroc_proxy_assoc": _row_metric_text(latest_row, "euroc_proxy_mean_ate_associated"),
            "last_tum_proxy_wins_vs_dpvo": _row_metric_text(latest_row, "tum_proxy_wins_vs_dpvo"),
            "last_euroc_proxy_wins_vs_dpvo": _row_metric_text(latest_row, "euroc_proxy_wins_vs_dpvo"),
            "last_kitti_proxy_assoc": _row_metric_text(latest_row, "kitti_proxy_mean_ate_associated"),
            "last_dino_patch_fraction": _row_metric_text(latest_row, "dino_patch_fraction"),
            "last_dual_score": _row_metric_text(latest_row, "live_dual_proxy_score"),
            "last_dual_gate": "" if not latest_row else str(int(_selection_passed_gate(latest_row))),
            "last_tri_score": _row_metric_text(latest_row, "live_tri_proxy_score"),
            "last_tri_gate": "" if not latest_row else str(int(_selection_passed_gate(latest_row))),
        }
    )
    return output


def _leaderboard_row(
    run: SweepRunSpec,
    rows: list[dict[str, Any]],
    status: str,
    coverage_gate: float,
    output_dir: Path,
    *,
    secondary_coverage_gate: float | None = None,
    secondary_assoc_guardrail: float | None = None,
) -> dict[str, str]:
    best_row = _best_dev_row(
        rows,
        coverage_gate,
        secondary_coverage_gate=secondary_coverage_gate,
        secondary_assoc_guardrail=secondary_assoc_guardrail,
    )
    latest_row = _latest_dev_row(rows)
    checkpoint_path = _checkpoint_path_for_best_mode(best_row, output_dir)
    output = {
        "run_id": run.run_id,
        "status": status,
        "expected_eval_mode": run.expected_eval_mode,
        "config_path": str(run.config_path),
        "subset_config": str(run.subset_config),
        "init_checkpoint": str(run.init_checkpoint),
        "best_step": "" if best_row is None else str(int(best_row["step"])),
        "best_assoc": _row_metric_text(best_row, "external_mean_ate_associated"),
        "best_ate": _row_metric_text(best_row, "external_mean_ate"),
        "best_coverage": _row_metric_text(best_row, "external_mean_coverage"),
        "best_lowtex_assoc": _row_metric_text(best_row, "lowtex_mean_ate_associated"),
        "best_lowtex_coverage": _row_metric_text(best_row, "lowtex_mean_coverage"),
        "best_mode": "" if best_row is None else str(best_row.get("best_mode", "pure100")),
        "best_pure_assoc": _row_metric_text(best_row, "best_pure_assoc"),
        "best_hybrid_assoc": _row_metric_text(best_row, "best_hybrid_assoc"),
        "last_step": "" if latest_row is None else str(int(latest_row["step"])),
        "last_assoc": _row_metric_text(latest_row, "external_mean_ate_associated"),
        "last_ate": _row_metric_text(latest_row, "external_mean_ate"),
        "last_coverage": _row_metric_text(latest_row, "external_mean_coverage"),
        "last_lowtex_assoc": _row_metric_text(latest_row, "lowtex_mean_ate_associated"),
        "last_lowtex_coverage": _row_metric_text(latest_row, "lowtex_mean_coverage"),
        "checkpoint_path": str(checkpoint_path),
    }
    output.update(
        {
            "best_pure_tum_score": _row_metric_text(best_row, "live_pure_tum_proxy_score"),
            "best_tum_proxy_assoc": _row_metric_text(best_row, "tum_proxy_mean_ate_associated"),
            "best_tum_pressure_assoc": _row_metric_text(best_row, "tum_pressure_mean_ate_associated"),
            "best_euroc_proxy_assoc": _row_metric_text(best_row, "euroc_proxy_mean_ate_associated"),
            "best_tum_proxy_wins_vs_dpvo": _row_metric_text(best_row, "tum_proxy_wins_vs_dpvo"),
            "best_euroc_proxy_wins_vs_dpvo": _row_metric_text(best_row, "euroc_proxy_wins_vs_dpvo"),
            "best_kitti_proxy_assoc": _row_metric_text(best_row, "kitti_proxy_mean_ate_associated"),
            "best_dino_patch_fraction": _row_metric_text(best_row, "dino_patch_fraction"),
            "best_dual_score": _row_metric_text(best_row, "live_dual_proxy_score"),
            "best_dual_gate": "" if best_row is None else str(int(_selection_passed_gate(best_row))),
            "best_tri_score": _row_metric_text(best_row, "live_tri_proxy_score"),
            "best_tri_gate": "" if best_row is None else str(int(_selection_passed_gate(best_row))),
            "last_pure_tum_score": _row_metric_text(latest_row, "live_pure_tum_proxy_score"),
            "last_tum_proxy_assoc": _row_metric_text(latest_row, "tum_proxy_mean_ate_associated"),
            "last_tum_pressure_assoc": _row_metric_text(latest_row, "tum_pressure_mean_ate_associated"),
            "last_euroc_proxy_assoc": _row_metric_text(latest_row, "euroc_proxy_mean_ate_associated"),
            "last_tum_proxy_wins_vs_dpvo": _row_metric_text(latest_row, "tum_proxy_wins_vs_dpvo"),
            "last_euroc_proxy_wins_vs_dpvo": _row_metric_text(latest_row, "euroc_proxy_wins_vs_dpvo"),
            "last_kitti_proxy_assoc": _row_metric_text(latest_row, "kitti_proxy_mean_ate_associated"),
            "last_dino_patch_fraction": _row_metric_text(latest_row, "dino_patch_fraction"),
            "last_dual_score": _row_metric_text(latest_row, "live_dual_proxy_score"),
            "last_dual_gate": "" if latest_row is None else str(int(_selection_passed_gate(latest_row))),
            "last_tri_score": _row_metric_text(latest_row, "live_tri_proxy_score"),
            "last_tri_gate": "" if latest_row is None else str(int(_selection_passed_gate(latest_row))),
        }
    )
    return output


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_existing_leaderboard(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def _process_table() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2])
        except Exception:
            continue
        rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "cmd": parts[3]})
    return rows


def _find_live_training_process(output_dir: Path) -> dict[str, Any] | None:
    resolved_output = str(output_dir.resolve())
    table = _process_table()
    by_pid = {row["pid"]: row for row in table}
    trainer_rows = [
        row
        for row in table
        if "-m refocus_vo.train_dino_dpvo_frontend" in str(row["cmd"])
        and f"--output-dir {resolved_output}" in str(row["cmd"])
    ]
    if not trainer_rows:
        return None
    trainer_pids = {row["pid"] for row in trainer_rows}
    main_row = next((row for row in trainer_rows if row["ppid"] not in trainer_pids), trainer_rows[0])
    shell_row = by_pid.get(int(main_row["ppid"]))
    if shell_row is not None and "train_dino_dpvo_frontend.sh" in str(shell_row["cmd"]):
        return {
            "process_pid": int(shell_row["pid"]),
            "process_pgid": int(shell_row["pgid"]),
            "trainer_pid": int(main_row["pid"]),
            "trainer_pgid": int(main_row["pgid"]),
            "cmd": str(shell_row["cmd"]),
        }
    return {
        "process_pid": int(main_row["pid"]),
        "process_pgid": int(main_row["pgid"]),
        "trainer_pid": int(main_row["pid"]),
        "trainer_pgid": int(main_row["pgid"]),
        "cmd": str(main_row["cmd"]),
    }


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _terminate_process_group_by_pid(pid: int | None, pgid: int | None, *, grace_seconds: float = 15.0) -> None:
    target_pgid = int(pgid) if pgid else None
    if target_pgid is None and pid is not None and _pid_is_alive(pid):
        try:
            target_pgid = os.getpgid(int(pid))
        except OSError:
            target_pgid = None
    if target_pgid is None:
        return
    try:
        os.killpg(target_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + float(grace_seconds)
    while time.time() < deadline:
        if pid is None or not _pid_is_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.killpg(target_pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _train_env(run: SweepRunSpec, output_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    runner_cfg = manifest.get("runner", {}) or {}
    env = os.environ.copy()
    raw_seed = runner_cfg.get("seed")
    env["DATA_PATH"] = str(_resolve_path(runner_cfg["dataset_root"], REPO_ROOT))
    env["EVAL_DATA_PATH"] = str(_resolve_path(runner_cfg["eval_dataset_root"], REPO_ROOT))
    env["TUM_EVAL_DATA_PATH"] = str(_resolve_path(runner_cfg["tum_dataset_root"], REPO_ROOT))
    if runner_cfg.get("euroc_dataset_root"):
        env["EUROC_EVAL_DATA_PATH"] = str(_resolve_path(runner_cfg["euroc_dataset_root"], REPO_ROOT))
    if runner_cfg.get("kitti_dataset_root"):
        env["KITTI_EVAL_DATA_PATH"] = str(_resolve_path(runner_cfg["kitti_dataset_root"], REPO_ROOT))
    env["SUBSET_CONFIG"] = str(run.subset_config)
    env["DINO_DPVO_CONFIG"] = str(run.config_path)
    env["DPVO_CONFIG_PATH"] = str(_resolve_path(runner_cfg["dpvo_config_path"], REPO_ROOT))
    env["TRAIN_RUN_ID"] = run.run_id
    env["OUTPUT_DIR_OVERRIDE"] = str(output_dir)
    env["INIT_CHECKPOINT"] = str(run.init_checkpoint)
    env["INIT_MODE"] = run.init_mode
    if raw_seed not in (None, ""):
        seed = int(raw_seed)
        env["SEED"] = str(seed)
        env["PYTHONHASHSEED"] = str(seed)
    env["TRAIN_DEVICE"] = str(runner_cfg.get("train_device", "cuda"))
    if bool(runner_cfg.get("deterministic", False)):
        env["DETERMINISTIC"] = "1"
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if bool(runner_cfg.get("legacy_repro", False)):
        env["LEGACY_REPRO"] = "1"
    return env


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_seconds: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _configured_total_steps(config_path: Path) -> int:
    cfg = _load_yaml(config_path)
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    return int(training_cfg.get("train_steps", 0) or 0)
    deadline = time.time() + float(grace_seconds)
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _materialize_pure_pack_config(source_config: Path, output_config: Path, run_label: str) -> Path:
    payload = _load_yaml(source_config)
    payload.setdefault("model", {})
    payload.setdefault("eval", {})
    model_cfg = payload["model"]
    eval_cfg = payload["eval"]
    overrides = deepcopy(eval_cfg.get("pure100_model_overrides", {}))
    if overrides:
        _deep_update(model_cfg, overrides)
    else:
        model_cfg["native_fraction"] = 0.0
        model_cfg["dino_fraction"] = 1.0
        model_cfg.setdefault("enforce_unique_semantic", True)
        model_cfg.setdefault("semantic_backfill_source", "dino")
        model_cfg.setdefault("semantic_grid_rows", 6)
        model_cfg.setdefault("semantic_grid_cols", 8)
        model_cfg.setdefault("max_semantic_per_cell", 2)
        model_cfg.setdefault("semantic_dedupe_schedule_px", [8.0, 6.0, 4.0, 2.0])
    payload["method_id"] = f"{payload.get('method_id', 'dino_dpvo')}_{run_label}"
    payload["feature_type"] = f"{payload.get('feature_type', 'DINO_DPVO')}_{run_label.upper()}"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_config


def _materialize_training_config(
    source_config: Path,
    output_config: Path,
    *,
    manifest_overrides: dict[str, Any],
    run_overrides: dict[str, Any],
    run_label: str,
) -> Path:
    payload = _load_yaml(source_config)
    if manifest_overrides:
        _deep_update(payload, manifest_overrides)
    if run_overrides:
        _deep_update(payload, run_overrides)
    payload["method_id"] = f"{payload.get('method_id', run_label)}_{run_label}"
    payload["feature_type"] = f"{payload.get('feature_type', run_label.upper())}_{run_label.upper()}"
    _write_yaml(output_config, payload)
    return output_config


def _mean_metrics_from_eval_csv(path: Path, *, exclude_sequences: set[str] | None = None) -> dict[str, float]:
    rows = []
    excluded = _sequence_set(exclude_sequences)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if excluded and str(row.get("sequence", "")).strip() in excluded:
                continue
            rows.append(row)
    if not rows:
        return {"assoc": math.nan, "ate": math.nan, "coverage": math.nan}
    assoc = [_safe_float(row.get("ate_rmse_associated")) for row in rows]
    ate = [_safe_float(row.get("ate_rmse")) for row in rows]
    coverage = [_safe_float(row.get("coverage")) for row in rows]
    assoc = [v for v in assoc if math.isfinite(v)]
    ate = [v for v in ate if math.isfinite(v)]
    coverage = [v for v in coverage if math.isfinite(v)]
    return {
        "assoc": (sum(assoc) / len(assoc)) if assoc else math.nan,
        "ate": (sum(ate) / len(ate)) if ate else math.nan,
        "coverage": (sum(coverage) / len(coverage)) if coverage else math.nan,
    }


def _pack_row_key(row: dict[str, str], coverage_gate: float) -> tuple[float, float, float] | None:
    assoc = _safe_float(row.get("pack_assoc"))
    ate = _safe_float(row.get("pack_ate"))
    coverage = _safe_float(row.get("pack_coverage"))
    if not (math.isfinite(assoc) and math.isfinite(coverage) and coverage >= float(coverage_gate)):
        return None
    ate_key = ate if math.isfinite(ate) else math.inf
    return (assoc, ate_key, -coverage)


def _dual_pack_row_key(
    row: dict[str, str],
    *,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
    secondary_assoc_guardrail: float | None,
) -> tuple[float, ...] | None:
    if secondary_coverage_gate is None and secondary_assoc_guardrail is None:
        return _pack_row_key(
            {
                "pack_assoc": row.get("paper_assoc", row.get("pack_assoc", "")),
                "pack_ate": row.get("paper_ate", row.get("pack_ate", "")),
                "pack_coverage": row.get("paper_coverage", row.get("pack_coverage", "")),
            },
            coverage_gate,
        )

    paper_assoc = _safe_float(row.get("paper_assoc"))
    paper_ate = _safe_float(row.get("paper_ate"))
    paper_coverage = _safe_float(row.get("paper_coverage"))
    lowtex_assoc = _safe_float(row.get("lowtex_assoc"))
    lowtex_coverage = _safe_float(row.get("lowtex_coverage"))
    if not (
        math.isfinite(paper_assoc)
        and math.isfinite(paper_coverage)
        and paper_coverage >= float(coverage_gate)
        and math.isfinite(lowtex_assoc)
        and math.isfinite(lowtex_coverage)
        and lowtex_coverage >= float(secondary_coverage_gate if secondary_coverage_gate is not None else coverage_gate)
        and lowtex_assoc <= float(secondary_assoc_guardrail if secondary_assoc_guardrail is not None else math.inf)
    ):
        return None
    paper_ate_key = paper_ate if math.isfinite(paper_ate) else math.inf
    return (paper_assoc, lowtex_assoc, paper_ate_key, -paper_coverage, -lowtex_coverage)


def _write_winner_summary(
    path: Path,
    *,
    winner: dict[str, str] | None,
    baseline_assoc: float,
    baseline_ate: float,
    coverage_gate: float,
    secondary_assoc_guardrail: float | None = None,
    secondary_label: str = "Low-texture",
    lowtex_reference_assoc: float | None = None,
    broad_reference_assoc: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if secondary_assoc_guardrail is None:
        lines = [
            "# Assoc-9 Sweep Winner Summary",
            "",
            f"- Coverage gate: `{coverage_gate:.2f}`",
            f"- Baseline associated ATE: `{baseline_assoc:.6f}`",
            f"- Baseline coverage-aware ATE: `{baseline_ate:.6f}`",
            "",
        ]
        if winner is None:
            lines.extend(
                [
                    "No validated candidate beat the coverage gate and produced a usable official pack score.",
                    "",
                ]
            )
        else:
            pack_assoc = _safe_float(winner.get("pack_assoc"))
            pack_ate = _safe_float(winner.get("pack_ate"))
            lines.extend(
                [
                    f"- Winner: `{winner.get('candidate_id', '')}`",
                    f"- Kind: `{winner.get('kind', '')}`",
                    f"- Official pack associated ATE: `{pack_assoc:.6f}`",
                    f"- Official pack coverage-aware ATE: `{pack_ate:.6f}`",
                    f"- Delta vs baseline associated ATE: `{pack_assoc - baseline_assoc:+.6f}`",
                    f"- Delta vs baseline coverage-aware ATE: `{pack_ate - baseline_ate:+.6f}`",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines = [
        f"# Paper + {secondary_label} Sweep Winner Summary",
        "",
        f"- Paper coverage gate: `{coverage_gate:.2f}`",
        f"- {secondary_label} guardrail: assoc `<= {secondary_assoc_guardrail:.6f}`",
        f"- Paper baseline to beat: `{baseline_assoc:.6f}`",
    ]
    if lowtex_reference_assoc is not None and math.isfinite(lowtex_reference_assoc):
        lines.append(f"- {secondary_label} fixed reference: `{lowtex_reference_assoc:.6f}`")
    if broad_reference_assoc is not None and math.isfinite(broad_reference_assoc):
        lines.append(f"- Broad fixed96 paper reference: `{broad_reference_assoc:.6f}`")
    lines.append("")
    if winner is None:
        lines.extend(
            [
                "No validated candidate beat the paper target while also satisfying the low-texture guardrail.",
                "",
            ]
        )
    else:
        paper_assoc = _safe_float(winner.get("paper_assoc"))
        paper_ate = _safe_float(winner.get("paper_ate"))
        lowtex_assoc = _safe_float(winner.get("lowtex_assoc"))
        lowtex_coverage = _safe_float(winner.get("lowtex_coverage"))
        lines.extend(
            [
                f"- Winner: `{winner.get('candidate_id', '')}`",
                f"- Kind: `{winner.get('kind', '')}`",
                f"- Paper assoc: `{paper_assoc:.6f}`",
                f"- Paper ATE: `{paper_ate:.6f}`",
                f"- {secondary_label} assoc: `{lowtex_assoc:.6f}`",
                f"- {secondary_label} coverage: `{lowtex_coverage:.6f}`",
                f"- Delta vs paper baseline: `{paper_assoc - baseline_assoc:+.6f}`",
                f"- Delta vs {secondary_label.lower()} guardrail: `{lowtex_assoc - secondary_assoc_guardrail:+.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_lowtex_tradeoff(
    path: Path,
    *,
    rows: list[dict[str, str]],
    winner: dict[str, str] | None,
    paper_baseline_assoc: float,
    lowtex_reference_assoc: float | None,
    broad_reference_assoc: float | None,
    lowtex_guardrail: float,
    secondary_label: str = "Low-texture",
) -> None:
    lines = [
        f"# Paper vs {secondary_label} Tradeoff",
        "",
        f"- Paper baseline to beat: `{paper_baseline_assoc:.6f}`",
        f"- {secondary_label} guardrail: `{lowtex_guardrail:.6f}`",
    ]
    if lowtex_reference_assoc is not None and math.isfinite(lowtex_reference_assoc):
        lines.append(f"- Fixed {secondary_label.lower()} reference: `{lowtex_reference_assoc:.6f}`")
    if broad_reference_assoc is not None and math.isfinite(broad_reference_assoc):
        lines.append(f"- Broad fixed96 paper reference: `{broad_reference_assoc:.6f}`")
    lines.extend(["", "| Candidate | Paper Assoc | Lowtex Assoc | Passes Guardrail |", "|---|---:|---:|---|"])
    for row in rows:
        lines.append(
            f"| `{row.get('candidate_id', '')}` | {row.get('paper_assoc', '')} | {row.get('lowtex_assoc', '')} | {row.get('passes_lowtex_guardrail', '')} |"
        )
    lines.append("")
    if winner is None:
        lines.append("No candidate satisfied both objectives in final validation.")
    else:
        lines.append(
            f"Validated winner `{winner.get('candidate_id', '')}` reached paper assoc `{winner.get('paper_assoc', '')}` "
            f"and {secondary_label.lower()} assoc `{winner.get('lowtex_assoc', '')}`."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _top_dev_candidates(
    leaderboard_rows: list[dict[str, str]],
    *,
    coverage_gate: float,
    secondary_coverage_gate: float | None = None,
    secondary_assoc_guardrail: float | None = None,
    limit: int,
) -> list[dict[str, str]]:
    eligible = []
    for row in leaderboard_rows:
        key = _row_key(
            {
                "external_mean_ate_associated": row.get("best_assoc", ""),
                "external_mean_ate": row.get("best_ate", ""),
                "external_mean_coverage": row.get("best_coverage", ""),
                "lowtex_mean_ate_associated": row.get("best_lowtex_assoc", ""),
                "lowtex_mean_coverage": row.get("best_lowtex_coverage", ""),
                "repeated_patch_fraction": "",
            },
            coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
        )
        if key is None:
            continue
        eligible.append((key, row))
    eligible.sort(key=lambda item: item[0])
    return [row for _, row in eligible[: int(limit)]]


def _baseline_pack_candidates(evaluation_cfg: dict[str, Any]) -> list[PackCandidate]:
    candidates: list[PackCandidate] = []
    seen_ids: set[str] = set()

    baseline = evaluation_cfg.get("current_champion", {}) or {}
    if baseline:
        candidate_id = str(baseline.get("run_id", "")).strip()
        checkpoint = str(baseline.get("checkpoint", "")).strip()
        config_path = str(baseline.get("config", "")).strip()
        if candidate_id and checkpoint and config_path and candidate_id not in seen_ids:
            candidates.append(
                PackCandidate(
                    candidate_id=candidate_id,
                    source_run_id=candidate_id,
                    kind=str(baseline.get("kind", "current_broad_champion")),
                    checkpoint=_resolve_path(checkpoint, REPO_ROOT),
                    config_path=_resolve_path(config_path, REPO_ROOT),
                )
            )
            seen_ids.add(candidate_id)

    extra_baselines = evaluation_cfg.get("baseline_candidates", []) or []
    for item in extra_baselines:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("run_id", "")).strip()
        checkpoint = str(item.get("checkpoint", "")).strip()
        config_path = str(item.get("config", "")).strip()
        if not candidate_id or not checkpoint or not config_path or candidate_id in seen_ids:
            continue
        candidates.append(
            PackCandidate(
                candidate_id=candidate_id,
                source_run_id=candidate_id,
                kind=str(item.get("kind", "comparison_baseline")),
                checkpoint=_resolve_path(checkpoint, REPO_ROOT),
                config_path=_resolve_path(config_path, REPO_ROOT),
            )
        )
        seen_ids.add(candidate_id)

    return candidates


def _write_leaderboard_from_state(
    *,
    path: Path,
    runs: list[SweepRunSpec],
    state: dict[str, Any],
    fieldnames: list[str],
) -> list[dict[str, str]]:
    rows = []
    state_runs = state.get("runs", {}) if isinstance(state.get("runs", {}), dict) else {}
    for run in runs:
        entry = state_runs.get(run.run_id) or {}
        rows.append(_state_entry_to_leaderboard_row(run, entry))
    _write_csv(path, rows, fieldnames)
    return rows


def _finalize_existing_run_status(
    *,
    output_dir: Path,
    runtime_config_path: Path,
    rows: list[dict[str, Any]],
) -> str:
    total_steps = _configured_total_steps(runtime_config_path)
    last_logged_step = _last_logged_step(output_dir / "train_metrics.csv")
    if total_steps > 0 and last_logged_step >= total_steps:
        return "completed"
    if rows or output_dir.exists():
        return "interrupted"
    return "failed"


def _monitor_run(
    *,
    run: SweepRunSpec,
    output_dir: Path,
    runtime_config_path: Path,
    metrics_csv: Path,
    proc: subprocess.Popen[str] | None,
    process_pid: int | None,
    process_pgid: int | None,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
    secondary_assoc_guardrail: float | None,
    required_usable_dev_steps: list[int],
    no_improve_patience_steps: int,
    worse_on_both_patience: int,
    min_failure_step: int,
    state: dict[str, Any],
    state_path: Path,
    leaderboard_path: Path,
    fieldnames: list[str],
    all_runs: list[SweepRunSpec],
) -> str:
    enforced_required_steps: set[int] = set()
    enforced_threshold_steps: set[int] = set()
    if proc is None:
        existing_rows = _read_dev_rows(metrics_csv)
        for step in required_usable_dev_steps:
            if any(int(row["step"]) >= int(step) for row in existing_rows):
                enforced_required_steps.add(int(step))
        for step in sorted(run.stop_thresholds):
            if any(int(row["step"]) >= int(step) for row in existing_rows):
                enforced_threshold_steps.add(int(step))

    status = "running"
    run_entry = state["runs"][run.run_id]
    while True:
        alive = proc.poll() is None if proc is not None else _pid_is_alive(process_pid)
        if not alive:
            break
        time.sleep(10.0)
        rows = _read_dev_rows(metrics_csv)
        _update_state_entry(
            run_entry,
            rows=rows,
            status=status,
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
            output_dir=output_dir,
            runtime_config_path=runtime_config_path,
            process_pid=process_pid,
            process_pgid=process_pgid,
        )
        _write_state(state_path, state)
        _write_leaderboard_from_state(path=leaderboard_path, runs=all_runs, state=state, fieldnames=fieldnames)

        for step in required_usable_dev_steps:
            if int(step) in enforced_required_steps:
                continue
            if int(step) < int(min_failure_step):
                continue
            if not any(int(row["step"]) >= int(step) for row in rows):
                continue
            enforced_required_steps.add(int(step))
            if _usable_step_row(rows, int(step), require_secondary=secondary_assoc_guardrail is not None) is None:
                status = "failed"
                if proc is not None:
                    _terminate_process_group(proc)
                else:
                    _terminate_process_group_by_pid(process_pid, process_pgid)
                break
        if status != "running":
            continue

        for step, threshold in sorted(run.stop_thresholds.items()):
            if int(step) in enforced_threshold_steps:
                continue
            if int(step) < int(min_failure_step):
                continue
            if not any(int(row["step"]) >= int(step) for row in rows):
                continue
            enforced_threshold_steps.add(int(step))
            best_row = _best_dev_row(
                rows,
                coverage_gate,
                max_step=step,
                secondary_coverage_gate=secondary_coverage_gate,
                secondary_assoc_guardrail=secondary_assoc_guardrail,
            )
            best_assoc = math.inf if best_row is None else _safe_float(best_row["external_mean_ate_associated"])
            best_lowtex_assoc = math.inf if best_row is None else _safe_float(best_row.get("lowtex_mean_ate_associated"))
            paper_threshold = float(threshold.get("paper_assoc", math.inf))
            lowtex_threshold = float(threshold.get("lowtex_assoc", math.inf))
            if (
                not math.isfinite(best_assoc)
                or best_assoc > paper_threshold
                or (secondary_assoc_guardrail is not None and (not math.isfinite(best_lowtex_assoc) or best_lowtex_assoc > lowtex_threshold))
            ):
                status = "early_stopped"
                if proc is not None:
                    _terminate_process_group(proc)
                else:
                    _terminate_process_group_by_pid(process_pid, process_pgid)
                break
        if status != "running":
            continue

        latest_row = _latest_dev_row(rows)
        latest_step = int(latest_row["step"]) if latest_row is not None else 0
        if (
            worse_on_both_patience > 0
            and latest_step >= int(min_failure_step)
            and _worse_on_both_streak(
                rows,
                coverage_gate=coverage_gate,
                secondary_coverage_gate=secondary_coverage_gate,
            ) >= int(worse_on_both_patience)
        ):
            status = "early_stopped"
            if proc is not None:
                _terminate_process_group(proc)
            else:
                _terminate_process_group_by_pid(process_pid, process_pgid)
            continue

        if no_improve_patience_steps > 0:
            best_row = _best_dev_row(
                rows,
                coverage_gate,
                secondary_coverage_gate=secondary_coverage_gate,
                secondary_assoc_guardrail=secondary_assoc_guardrail,
            )
            if best_row is not None and latest_row is not None:
                best_step = int(best_row["step"])
                latest_step = int(latest_row["step"])
                if latest_step >= int(min_failure_step) and latest_step - best_step >= no_improve_patience_steps:
                    status = "early_stopped"
                    if proc is not None:
                        _terminate_process_group(proc)
                    else:
                        _terminate_process_group_by_pid(process_pid, process_pgid)
                    break

    rows = _read_dev_rows(metrics_csv)
    if proc is not None:
        return_code = proc.wait()
        if status == "running":
            status = "completed" if return_code == 0 else "failed"
    elif status == "running":
        status = _finalize_existing_run_status(output_dir=output_dir, runtime_config_path=runtime_config_path, rows=rows)

    _update_state_entry(
        run_entry,
        rows=rows,
        status=status,
        coverage_gate=coverage_gate,
        secondary_coverage_gate=secondary_coverage_gate,
        secondary_assoc_guardrail=secondary_assoc_guardrail,
        output_dir=output_dir,
        runtime_config_path=runtime_config_path,
        process_pid=None,
        process_pgid=None,
    )
    _write_state(state_path, state)
    _write_leaderboard_from_state(path=leaderboard_path, runs=all_runs, state=state, fieldnames=fieldnames)
    return status


def _run_training_sweep(
    manifest: dict[str, Any],
    runs: list[SweepRunSpec],
    *,
    base_output_dir: Path,
    resume: bool,
    limit_runs: int | None,
) -> list[dict[str, str]]:
    runner_cfg = manifest.get("runner", {}) or {}
    sweep_cfg = manifest.get("sweep", {}) or {}
    evaluation_cfg = manifest.get("evaluation", {}) or {}
    coverage_gate = float(evaluation_cfg.get("coverage_gate", 0.95))
    secondary_coverage_gate = _secondary_coverage_gate(evaluation_cfg, coverage_gate)
    secondary_assoc_guardrail = _secondary_assoc_guardrail(evaluation_cfg)
    train_script = _resolve_path(runner_cfg["train_script"], REPO_ROOT)
    leaderboard_path = base_output_dir / "leaderboard_dev.csv"
    state_path = base_output_dir / STATE_FILENAME
    generated_config_dir = base_output_dir / "generated_train_configs"
    manifest_overrides = deepcopy(sweep_cfg.get("config_overrides", {}) or {})
    required_usable_dev_steps = _required_usable_dev_steps(sweep_cfg)
    no_improve_patience_steps = int(sweep_cfg.get("no_improve_patience_steps", 0) or 0)
    min_failure_step = int(sweep_cfg.get("min_failure_step", 0) or 0)
    legacy_worse_on_both = bool(sweep_cfg.get("stop_if_latest_worse_on_both_metrics", False))
    worse_on_both_patience = int(sweep_cfg.get("worse_on_both_patience", 1 if legacy_worse_on_both else 0) or 0)
    fieldnames = [
        "run_id",
        "status",
        "expected_eval_mode",
        "config_path",
        "subset_config",
        "init_checkpoint",
        "best_step",
        "best_assoc",
        "best_ate",
        "best_coverage",
        "best_lowtex_assoc",
        "best_lowtex_coverage",
        "best_mode",
        "best_pure_assoc",
        "best_hybrid_assoc",
        "last_step",
        "last_assoc",
        "last_ate",
        "last_coverage",
        "last_lowtex_assoc",
        "last_lowtex_coverage",
        "checkpoint_path",
    ]
    fieldnames.extend(LEADERBOARD_EXTRA_FIELDS)
    state = _load_state(state_path) if resume else {"updated_at": _now_timestamp(), "runs": {}}

    selected_runs = runs[: int(limit_runs)] if limit_runs is not None else runs
    selected_run_ids = {run.run_id for run in selected_runs}
    for run in selected_runs:
        output_dir = base_output_dir / "train" / run.run_id
        runtime_config_path = generated_config_dir / f"{run.run_id}.yaml"
        entry = (state.get("runs", {}) or {}).get(run.run_id)
        if entry is None:
            state.setdefault("runs", {})[run.run_id] = _state_entry_template(
                run,
                output_dir=output_dir,
                runtime_config_path=runtime_config_path if runtime_config_path.exists() else run.config_path,
            )
        else:
            entry.setdefault("run_id", run.run_id)
            entry.setdefault("output_dir", str(output_dir))
            if runtime_config_path.exists():
                entry["runtime_config_path"] = str(runtime_config_path)
    _write_state(state_path, state)
    leaderboard_rows = _write_leaderboard_from_state(
        path=leaderboard_path,
        runs=selected_runs,
        state=state,
        fieldnames=fieldnames,
    )

    for run in selected_runs:
        run_entry = state["runs"][run.run_id]
        existing_status = str(run_entry.get("status", "pending"))
        if resume and existing_status in {"completed", "early_stopped", "failed"}:
            continue

        output_dir = base_output_dir / "train" / run.run_id
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() and not resume:
            raise RuntimeError(f"Output directory already exists and --resume was not set: {output_dir}")

        runtime_config_path = generated_config_dir / f"{run.run_id}.yaml"
        if not runtime_config_path.exists() or not resume:
            runtime_config_path = _materialize_training_config(
                run.config_path,
                runtime_config_path,
                manifest_overrides=manifest_overrides,
                run_overrides=run.config_overrides,
                run_label=run.run_id,
            )
        runtime_run = SweepRunSpec(
            run_id=run.run_id,
            config_path=runtime_config_path,
            subset_config=run.subset_config,
            init_checkpoint=run.init_checkpoint,
            init_mode=run.init_mode,
            expected_eval_mode=run.expected_eval_mode,
            stop_thresholds=run.stop_thresholds,
            config_overrides=run.config_overrides,
        )

        metrics_csv = output_dir / "train_metrics.csv"
        rows = _read_dev_rows(metrics_csv)
        live_process = _find_live_training_process(output_dir) if output_dir.exists() else None
        process_pid = None
        process_pgid = None
        if live_process is not None:
            process_pid = int(live_process["process_pid"])
            process_pgid = int(live_process["process_pgid"])

        if resume:
            recorded_pid = run_entry.get("process_pid")
            recorded_pid = int(recorded_pid) if recorded_pid not in (None, "") else None
            recorded_alive = _pid_is_alive(recorded_pid)
            if existing_status == "running":
                if recorded_alive and live_process is not None and recorded_pid in {
                    int(live_process["process_pid"]),
                    int(live_process["trainer_pid"]),
                }:
                    process_pid = int(live_process["process_pid"])
                    process_pgid = int(live_process["process_pgid"])
                    _update_state_entry(
                        run_entry,
                        rows=rows,
                        status="running",
                        coverage_gate=coverage_gate,
                        secondary_coverage_gate=secondary_coverage_gate,
                        secondary_assoc_guardrail=secondary_assoc_guardrail,
                        output_dir=output_dir,
                        runtime_config_path=runtime_config_path,
                        process_pid=process_pid,
                        process_pgid=process_pgid,
                        start_time=str(run_entry.get("start_time") or _now_timestamp()),
                    )
                    _write_state(state_path, state)
                    _write_leaderboard_from_state(
                        path=leaderboard_path,
                        runs=selected_runs,
                        state=state,
                        fieldnames=fieldnames,
                    )
                    _monitor_run(
                        run=runtime_run,
                        output_dir=output_dir,
                        runtime_config_path=runtime_config_path,
                        metrics_csv=metrics_csv,
                        proc=None,
                        process_pid=process_pid,
                        process_pgid=process_pgid,
                        coverage_gate=coverage_gate,
                        secondary_coverage_gate=secondary_coverage_gate,
                        secondary_assoc_guardrail=secondary_assoc_guardrail,
                        required_usable_dev_steps=required_usable_dev_steps,
                        no_improve_patience_steps=no_improve_patience_steps,
                        worse_on_both_patience=worse_on_both_patience,
                        min_failure_step=min_failure_step,
                        state=state,
                        state_path=state_path,
                        leaderboard_path=leaderboard_path,
                        fieldnames=fieldnames,
                        all_runs=selected_runs,
                    )
                    continue
                finalized_status = _finalize_existing_run_status(
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    rows=rows,
                )
                _update_state_entry(
                    run_entry,
                    rows=rows,
                    status=finalized_status,
                    coverage_gate=coverage_gate,
                    secondary_coverage_gate=secondary_coverage_gate,
                    secondary_assoc_guardrail=secondary_assoc_guardrail,
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    process_pid=None,
                    process_pgid=None,
                )
                _write_state(state_path, state)
                _write_leaderboard_from_state(
                    path=leaderboard_path,
                    runs=selected_runs,
                    state=state,
                    fieldnames=fieldnames,
                )
                if finalized_status in {"completed", "early_stopped", "failed", "interrupted"}:
                    continue

            if live_process is not None:
                _update_state_entry(
                    run_entry,
                    rows=rows,
                    status="running",
                    coverage_gate=coverage_gate,
                    secondary_coverage_gate=secondary_coverage_gate,
                    secondary_assoc_guardrail=secondary_assoc_guardrail,
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    process_pid=process_pid,
                    process_pgid=process_pgid,
                    start_time=str(run_entry.get("start_time") or _now_timestamp()),
                )
                _write_state(state_path, state)
                _write_leaderboard_from_state(
                    path=leaderboard_path,
                    runs=selected_runs,
                    state=state,
                    fieldnames=fieldnames,
                )
                _monitor_run(
                    run=runtime_run,
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    metrics_csv=metrics_csv,
                    proc=None,
                    process_pid=process_pid,
                    process_pgid=process_pgid,
                    coverage_gate=coverage_gate,
                    secondary_coverage_gate=secondary_coverage_gate,
                    secondary_assoc_guardrail=secondary_assoc_guardrail,
                    required_usable_dev_steps=required_usable_dev_steps,
                    no_improve_patience_steps=no_improve_patience_steps,
                    worse_on_both_patience=worse_on_both_patience,
                    min_failure_step=min_failure_step,
                    state=state,
                    state_path=state_path,
                    leaderboard_path=leaderboard_path,
                    fieldnames=fieldnames,
                    all_runs=selected_runs,
                )
                continue

            if existing_status == "interrupted":
                _update_state_entry(
                    run_entry,
                    rows=rows,
                    status="interrupted",
                    coverage_gate=coverage_gate,
                    secondary_coverage_gate=secondary_coverage_gate,
                    secondary_assoc_guardrail=secondary_assoc_guardrail,
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    process_pid=None,
                    process_pgid=None,
                )
                _write_state(state_path, state)
                _write_leaderboard_from_state(
                    path=leaderboard_path,
                    runs=selected_runs,
                    state=state,
                    fieldnames=fieldnames,
                )
                continue

            if output_dir.exists() and (rows or metrics_csv.exists()):
                inferred_status = _finalize_existing_run_status(
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    rows=rows,
                )
                _update_state_entry(
                    run_entry,
                    rows=rows,
                    status=inferred_status,
                    coverage_gate=coverage_gate,
                    secondary_coverage_gate=secondary_coverage_gate,
                    secondary_assoc_guardrail=secondary_assoc_guardrail,
                    output_dir=output_dir,
                    runtime_config_path=runtime_config_path,
                    process_pid=None,
                    process_pgid=None,
                )
                _write_state(state_path, state)
                _write_leaderboard_from_state(
                    path=leaderboard_path,
                    runs=selected_runs,
                    state=state,
                    fieldnames=fieldnames,
                )
                if inferred_status in {"completed", "interrupted", "failed", "early_stopped"}:
                    continue

        env = _train_env(runtime_run, output_dir, manifest)
        proc = subprocess.Popen(
            ["bash", str(train_script)],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            start_new_session=True,
        )
        process_pid = int(proc.pid)
        process_pgid = int(proc.pid)
        _update_state_entry(
            run_entry,
            rows=rows,
            status="running",
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
            output_dir=output_dir,
            runtime_config_path=runtime_config_path,
            process_pid=process_pid,
            process_pgid=process_pgid,
            start_time=_now_timestamp(),
        )
        _write_state(state_path, state)
        _write_leaderboard_from_state(
            path=leaderboard_path,
            runs=selected_runs,
            state=state,
            fieldnames=fieldnames,
        )
        _monitor_run(
            run=runtime_run,
            output_dir=output_dir,
            runtime_config_path=runtime_config_path,
            metrics_csv=metrics_csv,
            proc=proc,
            process_pid=process_pid,
            process_pgid=process_pgid,
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
            secondary_assoc_guardrail=secondary_assoc_guardrail,
            required_usable_dev_steps=required_usable_dev_steps,
            no_improve_patience_steps=no_improve_patience_steps,
            worse_on_both_patience=worse_on_both_patience,
            min_failure_step=min_failure_step,
            state=state,
            state_path=state_path,
            leaderboard_path=leaderboard_path,
            fieldnames=fieldnames,
            all_runs=selected_runs,
        )

    return _write_leaderboard_from_state(
        path=leaderboard_path,
        runs=selected_runs,
        state=state,
        fieldnames=fieldnames,
    )


def _validate_candidates(
    manifest: dict[str, Any],
    dev_rows: list[dict[str, str]],
    *,
    base_output_dir: Path,
) -> list[dict[str, str]]:
    evaluation_cfg = manifest.get("evaluation", {}) or {}
    coverage_gate = float(evaluation_cfg.get("coverage_gate", 0.95))
    secondary_coverage_gate = _secondary_coverage_gate(evaluation_cfg, coverage_gate)
    secondary_assoc_guardrail = _secondary_assoc_guardrail(evaluation_cfg)
    top_dev_limit = int(evaluation_cfg.get("top_dev_limit", 4) or 4)
    pack_script = _resolve_path((manifest.get("runner", {}) or {})["pack_script"], REPO_ROOT)
    secondary_pack_script = (
        _resolve_path(evaluation_cfg["secondary_pack_script"], REPO_ROOT)
        if evaluation_cfg.get("secondary_pack_script")
        else None
    )
    secondary_from_primary_exclude_sequences = _sequence_set(
        evaluation_cfg.get("secondary_from_primary_exclude_sequences", [])
    )
    pack_config_dir = base_output_dir / "pack_eval_configs"
    leaderboard_pack_path = base_output_dir / "leaderboard_pack.csv"
    primary_pack_env = {str(key): str(value) for key, value in (evaluation_cfg.get("pack_env", {}) or {}).items()}
    secondary_pack_env = {
        str(key): str(value)
        for key, value in (evaluation_cfg.get("secondary_pack_env", evaluation_cfg.get("pack_env", {})) or {}).items()
    }
    dual_validation = secondary_pack_script is not None or bool(secondary_from_primary_exclude_sequences)

    candidates = []
    for row in _top_dev_candidates(
        dev_rows,
        coverage_gate=coverage_gate,
        secondary_coverage_gate=secondary_coverage_gate,
        secondary_assoc_guardrail=secondary_assoc_guardrail,
        limit=top_dev_limit,
    ):
        candidates.append(
            PackCandidate(
                candidate_id=row["run_id"],
                source_run_id=row["run_id"],
                kind="sweep_run",
                checkpoint=Path(row["checkpoint_path"]).resolve(),
                config_path=Path(row["config_path"]).resolve(),
            )
        )
    candidates.extend(_baseline_pack_candidates(evaluation_cfg))

    pack_rows: list[dict[str, str]] = []
    fieldnames = ["candidate_id", "source_run_id", "kind", "checkpoint", "config_path"]
    if dual_validation:
        fieldnames.extend(
            [
                "paper_assoc",
                "paper_ate",
                "paper_coverage",
                "lowtex_assoc",
                "lowtex_ate",
                "lowtex_coverage",
                "passes_lowtex_guardrail",
            ]
        )
    else:
        fieldnames.extend(["pack_assoc", "pack_ate", "pack_coverage"])

    for candidate in candidates:
        run_id = f"{candidate.candidate_id}_pack"
        pack_cfg_path = _materialize_pure_pack_config(
            candidate.config_path,
            pack_config_dir / f"{candidate.candidate_id}.yaml",
            run_id,
        )
        env = os.environ.copy()
        env["RUNS_ROOT"] = str(base_output_dir)
        env["FRONTEND_MODE"] = "dino_proposals"
        env["FRONTEND_CONFIG"] = str(pack_cfg_path)
        env["CHECKPOINT"] = str(candidate.checkpoint)
        env["RUN_ORB_BASELINE"] = "0"
        env["RUN_DPVO_BASELINE"] = "0"
        paper_run_id = f"{candidate.candidate_id}_paper_pack"
        paper_env = env.copy()
        paper_env.update(primary_pack_env)
        paper_env["PACK_ID"] = paper_run_id
        paper_env["DINO_DPVO_RUN_ID"] = paper_run_id
        subprocess.run(["bash", str(pack_script)], cwd=str(REPO_ROOT), env=paper_env, check=True, text=True)

        paper_metrics_csv = base_output_dir / "eval" / paper_run_id / "metrics_summary.csv"
        paper_metrics = _mean_metrics_from_eval_csv(paper_metrics_csv)
        row = {
            "candidate_id": candidate.candidate_id,
            "source_run_id": candidate.source_run_id,
            "kind": candidate.kind,
            "checkpoint": str(candidate.checkpoint),
            "config_path": str(pack_cfg_path),
        }
        if dual_validation:
            if secondary_from_primary_exclude_sequences:
                lowtex_metrics = _mean_metrics_from_eval_csv(
                    paper_metrics_csv,
                    exclude_sequences=secondary_from_primary_exclude_sequences,
                )
            else:
                lowtex_run_id = f"{candidate.candidate_id}_lowtex_pack"
                lowtex_env = env.copy()
                lowtex_env.update(secondary_pack_env)
                lowtex_env["PACK_ID"] = lowtex_run_id
                lowtex_env["DINO_DPVO_RUN_ID"] = lowtex_run_id
                subprocess.run(["bash", str(secondary_pack_script)], cwd=str(REPO_ROOT), env=lowtex_env, check=True, text=True)
                lowtex_metrics_csv = base_output_dir / "eval" / lowtex_run_id / "metrics_summary.csv"
                lowtex_metrics = _mean_metrics_from_eval_csv(lowtex_metrics_csv)
            passes_lowtex_guardrail = (
                math.isfinite(lowtex_metrics["assoc"])
                and math.isfinite(lowtex_metrics["coverage"])
                and lowtex_metrics["coverage"] >= float(secondary_coverage_gate if secondary_coverage_gate is not None else coverage_gate)
                and lowtex_metrics["assoc"] <= float(secondary_assoc_guardrail if secondary_assoc_guardrail is not None else math.inf)
            )
            row.update(
                {
                    "paper_assoc": f"{paper_metrics['assoc']:.6f}",
                    "paper_ate": f"{paper_metrics['ate']:.6f}",
                    "paper_coverage": f"{paper_metrics['coverage']:.6f}",
                    "lowtex_assoc": f"{lowtex_metrics['assoc']:.6f}",
                    "lowtex_ate": f"{lowtex_metrics['ate']:.6f}",
                    "lowtex_coverage": f"{lowtex_metrics['coverage']:.6f}",
                    "passes_lowtex_guardrail": "1" if passes_lowtex_guardrail else "0",
                }
            )
        else:
            row.update(
                {
                    "pack_assoc": f"{paper_metrics['assoc']:.6f}",
                    "pack_ate": f"{paper_metrics['ate']:.6f}",
                    "pack_coverage": f"{paper_metrics['coverage']:.6f}",
                }
            )
        pack_rows.append(row)
        _write_csv(leaderboard_pack_path, pack_rows, fieldnames)

    def candidate_key(row: dict[str, str]) -> tuple[float, ...]:
        key = _dual_pack_row_key(
            row,
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate if dual_validation else None,
            secondary_assoc_guardrail=secondary_assoc_guardrail if dual_validation else None,
        )
        if key is None:
            return (math.inf, math.inf, math.inf, math.inf, math.inf)
        return key

    pack_rows.sort(key=candidate_key)
    _write_csv(leaderboard_pack_path, pack_rows, fieldnames)

    winner = None
    for row in pack_rows:
        if _dual_pack_row_key(
            row,
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate if dual_validation else None,
            secondary_assoc_guardrail=secondary_assoc_guardrail if dual_validation else None,
        ) is not None:
            winner = row
            break

    _write_winner_summary(
        base_output_dir / "winner_summary.md",
        winner=winner,
        baseline_assoc=float(evaluation_cfg.get("baseline_assoc", 0.068249)),
        baseline_ate=float(evaluation_cfg.get("baseline_ate", 0.377303)),
        coverage_gate=coverage_gate,
        secondary_assoc_guardrail=secondary_assoc_guardrail if dual_validation else None,
        secondary_label=str(evaluation_cfg.get("secondary_label", "Low-texture")),
        lowtex_reference_assoc=_safe_float(evaluation_cfg.get("lowtex_reference_assoc")) if "lowtex_reference_assoc" in evaluation_cfg else None,
        broad_reference_assoc=_safe_float(evaluation_cfg.get("broad_reference_assoc")) if "broad_reference_assoc" in evaluation_cfg else None,
    )
    if dual_validation:
        _write_paper_lowtex_tradeoff(
            base_output_dir / "paper_vs_lowtex_tradeoff.md",
            rows=pack_rows,
            winner=winner,
            paper_baseline_assoc=float(evaluation_cfg.get("baseline_assoc", 0.078713)),
            lowtex_reference_assoc=_safe_float(evaluation_cfg.get("lowtex_reference_assoc")) if "lowtex_reference_assoc" in evaluation_cfg else None,
            broad_reference_assoc=_safe_float(evaluation_cfg.get("broad_reference_assoc")) if "broad_reference_assoc" in evaluation_cfg else None,
            lowtex_guardrail=float(secondary_assoc_guardrail if secondary_assoc_guardrail is not None else math.inf),
            secondary_label=str(evaluation_cfg.get("secondary_label", "Low-texture")),
        )
    return pack_rows


def _print_dry_run(manifest: dict[str, Any], runs: list[SweepRunSpec], base_output_dir: Path) -> None:
    evaluation_cfg = manifest.get("evaluation", {}) or {}
    sweep_cfg = manifest.get("sweep", {}) or {}
    print(f"sweep_name: {manifest.get('name', 'assoc9_sweep')}")
    print(f"base_output_dir: {base_output_dir}")
    if evaluation_cfg.get("sequences"):
        print(f"primary_sequences: {','.join(str(seq) for seq in evaluation_cfg.get('sequences', []))}")
    if sweep_cfg.get("config_overrides", {}).get("eval", {}).get("secondary_eval_sequences"):
        print(
            "secondary_sequences: "
            + ",".join(str(seq) for seq in sweep_cfg.get("config_overrides", {}).get("eval", {}).get("secondary_eval_sequences", []))
        )
    print("runs:")
    for run in runs:
        print(
            f"  - {run.run_id}: config={run.config_path} subset={run.subset_config} "
            f"init={run.init_checkpoint} mode={run.expected_eval_mode}"
        )


def _skip_validation(manifest: dict[str, Any]) -> bool:
    sweep_cfg = manifest.get("sweep", {}) or {}
    evaluation_cfg = manifest.get("evaluation", {}) or {}
    return bool(
        evaluation_cfg.get("skip_validation", False)
        or sweep_cfg.get("skip_validation", False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aggressive assoc-9 sweep.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest, REPO_ROOT)
    manifest, runs = _load_manifest(manifest_path)
    default_base = REPO_ROOT / "refocus_vo" / "runs" / "sweeps" / str(manifest.get("name", "dino_dpvo_assoc9_sweep_v1"))
    base_output_dir = _resolve_path(args.base_output_dir, REPO_ROOT) if args.base_output_dir else default_base
    base_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, base_output_dir / "manifest.yaml")

    if args.dry_run:
        _print_dry_run(manifest, runs[: int(args.limit_runs)] if args.limit_runs is not None else runs, base_output_dir)
        return

    dev_rows = _run_training_sweep(
        manifest,
        runs,
        base_output_dir=base_output_dir,
        resume=bool(args.resume),
        limit_runs=args.limit_runs,
    )
    if _skip_validation(manifest):
        print("[assoc9_sweep] skipping post-sweep validation by manifest request")
        return
    _validate_candidates(manifest, dev_rows, base_output_dir=base_output_dir)


if __name__ == "__main__":
    main()
