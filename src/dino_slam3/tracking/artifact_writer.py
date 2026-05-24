from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_run_manifest(
    *,
    out_dir: str | Path,
    cfg: Dict[str, Any],
    experiment_id: str,
    config_hash: str,
    git_commit: str,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    tracking_cfg = run_cfg.get("tracking", {}) if isinstance(run_cfg.get("tracking", {}), dict) else {}
    artifact_dir = Path(str(tracking_cfg.get("artifact_dir", "artifacts")))
    base_out = Path(out_dir)
    if not base_out.is_absolute():
        base_out = (Path.cwd() / base_out).resolve()
    adir = (base_out / artifact_dir).resolve()
    adir.mkdir(parents=True, exist_ok=True)
    manifest_path = adir / "run_manifest.json"
    sqlite_path = Path(str(tracking_cfg.get("sqlite_path", "runs/experiments.db"))).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = (Path.cwd() / sqlite_path).resolve()

    payload = {
        "created_utc": _now(),
        "experiment_id": str(experiment_id),
        "config_hash": str(config_hash),
        "git_commit": str(git_commit),
        "run_name": str(run_cfg.get("name", "unknown")),
        "prompt_tag": str(run_cfg.get("prompt_tag", "default")),
        "parent_experiment_id": run_cfg.get("parent_experiment_id"),
        "config": cfg,
        "env": dict(env or {}),
        "resolved_paths": {
            "cwd": str(Path.cwd().resolve()),
            "run_out_dir": str(base_out),
            "artifact_dir": str(adir),
            "sqlite_path": str(sqlite_path),
        },
    }

    # Keep this immutable once created, unless explicitly overwritten by config.
    overwrite = bool(tracking_cfg.get("overwrite_manifest", False))
    if manifest_path.exists() and not overwrite:
        return manifest_path
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return manifest_path


def write_semantic_selection_snapshot(
    *,
    out_dir: str | Path,
    epoch: int,
    payload: Dict[str, Any],
) -> Path:
    base_out = Path(out_dir)
    if not base_out.is_absolute():
        base_out = (Path.cwd() / base_out).resolve()
    adir = (base_out / "artifacts" / "selection_snapshots").resolve()
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / f"semantic_selection_epoch_{int(epoch):03d}.json"
    body = dict(payload)
    body.setdefault("created_utc", _now())
    body.setdefault("epoch", int(epoch))
    path.write_text(json.dumps(body, indent=2, sort_keys=False), encoding="utf-8")
    return path
