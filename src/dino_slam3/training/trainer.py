from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext
import os
import csv
import subprocess
import math
import random
import json
import sys
import hashlib
import uuid
import re
from datetime import datetime, timezone
from copy import deepcopy

import numpy as np

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dino_slam3.data.tum_rgbd import TUMRGBDDataset
from dino_slam3.data.manifest_rgbd_pairs import ManifestRGBDPairDataset
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.losses.two_view_loss import compute_losses
from dino_slam3.utils.config import ensure_dir
from dino_slam3.utils.rich_logging import (
    print_epoch_header,
    print_metrics_table,
    print_save_notice,
    print_match_table,
)
from dino_slam3.tracking import (
    ExperimentStore,
    write_run_manifest,
    write_semantic_selection_snapshot,
)
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch
from dino_slam3.geometry.projection import unproject, transform, project

def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _seed_everything(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def _scheduled_loss_cfg(cfg: Dict[str, Any], epoch: int, total_epochs: int) -> Dict[str, Any]:
    base = deepcopy(cfg.get("loss", {}))
    base["__total_epochs"] = int(total_epochs)
    sched = cfg.get("training", {}).get("loss_weight_schedule", None)
    if not isinstance(sched, list) or len(sched) == 0:
        return base

    for rule in sched:
        if not isinstance(rule, dict):
            continue
        start = int(rule.get("epoch_start", rule.get("start_epoch", 1)))
        end = int(rule.get("epoch_end", rule.get("end_epoch", total_epochs)))
        if epoch < start or epoch > end:
            continue
        patch = rule.get("loss", {})
        if isinstance(patch, dict):
            _deep_merge_dict(base, patch)
    base["__total_epochs"] = int(total_epochs)
    return base


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown"


def _config_hash(cfg: Dict[str, Any]) -> str:
    try:
        s = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        s = str(cfg)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _append_run_registry(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    selection_metric_used: str,
    selection_value: float,
    ate_short_mean: float,
    fps: Optional[float] = None,
) -> None:
    run_cfg = cfg.get("run", {})
    registry_path = Path(str(run_cfg.get("registry_path", "runs/_registry.csv"))).expanduser()
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "run_name",
        "config_hash",
        "ckpt_path",
        "git_commit",
        "date_utc",
        "selection_metric_used",
        "selection_value",
        "ate_short_mean",
        "fps",
    ]
    row = {
        "run_name": str(run_cfg.get("name", "unknown")),
        "config_hash": _config_hash(cfg),
        "ckpt_path": str(checkpoint_path.resolve()),
        "git_commit": _git_commit(),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_metric_used": str(selection_metric_used),
        "selection_value": f"{float(selection_value):.6f}" if math.isfinite(float(selection_value)) else "inf",
        "ate_short_mean": f"{float(ate_short_mean):.6f}" if math.isfinite(float(ate_short_mean)) else "NaN",
        "fps": "" if fps is None else f"{float(fps):.3f}",
    }
    write_header = not registry_path.exists()
    with registry_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _resolve_experiment_id(cfg: Dict[str, Any]) -> str:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    exp_id = str(run_cfg.get("experiment_id") or "").strip()
    if exp_id:
        return exp_id
    return str(uuid.uuid4())


def _score_semantic_eval(
    *,
    rows: List[Dict[str, Any]],
    sequences: List[str],
    sequence_weights_cfg: Dict[str, Any],
    missing_penalty: float,
    status_penalties_cfg: Dict[str, Any],
    min_coverage_ok: float,
    primary_ate_field: str = "ate_rmse_associated",
    robust_ate_field: str = "ate_rmse",
) -> Dict[str, Any]:
    seq_weights: Dict[str, float] = {}
    for s in sequences:
        w = _safe_float(sequence_weights_cfg.get(s, 1.0), default=1.0)
        if w <= 0:
            w = 1.0
        seq_weights[str(s)] = float(w)
    wsum = float(sum(seq_weights.values())) if seq_weights else 1.0
    if wsum <= 0:
        wsum = 1.0

    by_seq: Dict[str, Dict[str, Any]] = {
        str(r.get("sequence", "")): r for r in rows if str(r.get("sequence", "")).strip()
    }
    weighted_ok_num = 0.0
    weighted_ok_den = 0.0
    weighted_ok_num_robust = 0.0
    weighted_ok_den_robust = 0.0
    weighted_missing = 0.0
    weighted_status_pen = 0.0
    statuses: Dict[str, int] = {}
    weighted_cov_num = 0.0
    weighted_cov_den = 0.0
    per_sequence: List[Dict[str, Any]] = []
    for seq, w in seq_weights.items():
        row = by_seq.get(seq)
        if row is None:
            weighted_missing += w
            statuses["missing"] = statuses.get("missing", 0) + 1
            per_sequence.append(
                {
                    "sequence": seq,
                    "status": "missing",
                    "weight": w,
                    "ate_rmse": float("nan"),
                    "ate_rmse_associated": float("nan"),
                    "ate_primary": float("nan"),
                    "ate_robust": float("nan"),
                    "coverage": float("nan"),
                }
            )
            continue

        status = str(row.get("status", "")).strip().lower()
        statuses[status] = statuses.get(status, 0) + 1
        ate_primary = _safe_float(row.get(primary_ate_field), default=float("nan"))
        ate_robust = _safe_float(row.get(robust_ate_field), default=float("nan"))
        ate_full = _safe_float(row.get("ate_rmse"), default=float("nan"))
        ate_assoc = _safe_float(row.get("ate_rmse_associated"), default=float("nan"))
        cov = _safe_float(row.get("coverage"), default=float("nan"))
        if math.isfinite(cov):
            weighted_cov_num += w * cov
            weighted_cov_den += w

        if status == "ok" and math.isfinite(ate_primary):
            weighted_ok_num += w * ate_primary
            weighted_ok_den += w
        else:
            weighted_missing += w
            weighted_status_pen += w * _safe_float(status_penalties_cfg.get(status, 0.0), default=0.0)
        if status == "ok" and math.isfinite(ate_robust):
            weighted_ok_num_robust += w * ate_robust
            weighted_ok_den_robust += w

        per_sequence.append(
            {
                "sequence": seq,
                "status": status,
                "weight": w,
                "ate_rmse": ate_full,
                "ate_rmse_associated": ate_assoc,
                "ate_primary": ate_primary,
                "ate_robust": ate_robust,
                "coverage": cov,
            }
        )

    weighted_mean_ok = float("inf")
    if weighted_ok_den > 0:
        weighted_mean_ok = weighted_ok_num / weighted_ok_den
    weighted_mean_ok_robust = float("inf")
    if weighted_ok_den_robust > 0:
        weighted_mean_ok_robust = weighted_ok_num_robust / weighted_ok_den_robust
    weighted_missing_ratio = weighted_missing / wsum
    weighted_status_penalty = weighted_status_pen / wsum
    weighted_penalized_score = weighted_mean_ok
    if math.isfinite(weighted_penalized_score):
        weighted_penalized_score += float(missing_penalty) * float(weighted_missing_ratio)
    else:
        weighted_penalized_score = float(missing_penalty) * float(weighted_missing_ratio)
    weighted_penalized_score += float(weighted_status_penalty)
    weighted_penalized_score_robust = weighted_mean_ok_robust
    if math.isfinite(weighted_penalized_score_robust):
        weighted_penalized_score_robust += float(missing_penalty) * float(weighted_missing_ratio)
    else:
        weighted_penalized_score_robust = float(missing_penalty) * float(weighted_missing_ratio)
    weighted_penalized_score_robust += float(weighted_status_penalty)
    weighted_coverage_mean = float("nan")
    if weighted_cov_den > 0:
        weighted_coverage_mean = weighted_cov_num / weighted_cov_den
    coverage_ok = bool(math.isfinite(weighted_coverage_mean) and weighted_coverage_mean >= float(min_coverage_ok))

    ok_vals = [x["ate_primary"] for x in per_sequence if x.get("status") == "ok" and math.isfinite(_safe_float(x.get("ate_primary")))]
    ok_vals_robust = [x["ate_robust"] for x in per_sequence if x.get("status") == "ok" and math.isfinite(_safe_float(x.get("ate_robust")))]
    mean_ok = float(sum(ok_vals) / len(ok_vals)) if ok_vals else float("inf")
    mean_ok_robust = float(sum(ok_vals_robust) / len(ok_vals_robust)) if ok_vals_robust else float("inf")
    penalized = mean_ok
    penalized_robust = mean_ok_robust
    if weighted_missing > 0:
        if math.isfinite(penalized):
            penalized += float(missing_penalty) * float(weighted_missing_ratio)
        else:
            penalized = float(missing_penalty) * float(weighted_missing_ratio)
        if math.isfinite(penalized_robust):
            penalized_robust += float(missing_penalty) * float(weighted_missing_ratio)
        else:
            penalized_robust = float(missing_penalty) * float(weighted_missing_ratio)
    for st, cnt in statuses.items():
        if st == "ok":
            continue
        p = _safe_float(status_penalties_cfg.get(st, 0.0), default=0.0)
        if p > 0 and len(sequences) > 0:
            penalized += p * (float(cnt) / float(len(sequences)))
            penalized_robust += p * (float(cnt) / float(len(sequences)))

    return {
        "mean_ok": float(mean_ok),
        "penalized_mean": float(penalized),
        "mean_ok_robust": float(mean_ok_robust),
        "penalized_mean_robust": float(penalized_robust),
        "weighted_mean_ok": float(weighted_mean_ok),
        "weighted_penalized_score": float(weighted_penalized_score),
        "weighted_mean_ok_robust": float(weighted_mean_ok_robust),
        "weighted_penalized_score_robust": float(weighted_penalized_score_robust),
        "weighted_missing_ratio": float(weighted_missing_ratio),
        "weighted_status_penalty": float(weighted_status_penalty),
        "weighted_coverage_mean": float(weighted_coverage_mean),
        "coverage_ok": bool(coverage_ok),
        "primary_ate_field": str(primary_ate_field),
        "robust_ate_field": str(robust_ate_field),
        "ok_count": int(sum(1 for x in per_sequence if x.get("status") == "ok" and math.isfinite(_safe_float(x.get("ate_primary"))))),
        "total_count": int(len(sequences)),
        "statuses": statuses,
        "rows": per_sequence,
    }


def _selection_candidate_from_semantic_result(
    selection_metric: str,
    ate_result: Optional[Dict[str, Any]],
) -> Tuple[float, str]:
    metric = str(selection_metric).strip().lower()
    if ate_result is None:
        return float("inf"), metric

    profiles = ate_result.get("profiles", {}) if isinstance(ate_result.get("profiles", {}), dict) else {}
    selected_profile_name = str(ate_result.get("selected_profile", "default"))
    selected_profile = profiles.get(selected_profile_name, ate_result)
    holdout = profiles.get("holdout")
    overfit = profiles.get("overfit")

    if metric == "ate_short_mean":
        return float((selected_profile or {}).get("mean_ok", float("inf"))), metric
    if metric == "ate_short_mean_penalized":
        return float((selected_profile or {}).get("penalized_mean", float("inf"))), metric
    if metric == "weighted_penalized_ate":
        return float((selected_profile or {}).get("weighted_penalized_score", float("inf"))), metric
    if metric == "holdout_weighted_penalized_ate":
        return float((holdout or {}).get("weighted_penalized_score", float("inf"))), metric
    if metric == "overfit_weighted_penalized_ate":
        return float((overfit or {}).get("weighted_penalized_score", float("inf"))), metric
    if metric == "weighted_mean_ate":
        return float((selected_profile or {}).get("weighted_mean_ok", float("inf"))), metric
    if metric == "holdout_weighted_mean_ate":
        return float((holdout or {}).get("weighted_mean_ok", float("inf"))), metric
    if metric == "overfit_weighted_mean_ate":
        return float((overfit or {}).get("weighted_mean_ok", float("inf"))), metric
    if metric == "weighted_mean_ate_robust":
        return float((selected_profile or {}).get("weighted_mean_ok_robust", float("inf"))), metric
    if metric == "holdout_weighted_mean_ate_robust":
        return float((holdout or {}).get("weighted_mean_ok_robust", float("inf"))), metric
    if metric == "overfit_weighted_mean_ate_robust":
        return float((overfit or {}).get("weighted_mean_ok_robust", float("inf"))), metric
    return float("inf"), metric

def _make_loaders(
    cfg: Dict[str, Any],
    epochs: int,
    seed: int,
) -> Tuple[DataLoader, Dict[str, DataLoader], torch.utils.data.Dataset, Dict[str, Any]]:
    dcfg = cfg["dataset"]
    tcfg = cfg["training"]
    assoc = dcfg.get("association", {})
    dataset_root = Path(dcfg.get("root", ".")).expanduser()
    skip_missing = bool(dcfg.get("skip_missing_sequences", True))
    sequence_repeat = dcfg.get("sequence_repeat", {})
    dataset_mode = str(dcfg.get("mode", "tum")).lower()

    external_cfg = dcfg.get("external", {}) if isinstance(dcfg.get("external", {}), dict) else {}
    external_enabled = bool(external_cfg.get("enabled", False))
    if dataset_mode == "auto":
        dataset_mode = "mixed" if external_enabled else "tum"

    if external_enabled and dataset_mode == "tum":
        dataset_mode = "mixed"

    single_cfg = dcfg.get("single_sequence", {}) if isinstance(dcfg.get("single_sequence", {}), dict) else {}
    single_enabled = bool(single_cfg.get("enabled", False))
    teacher_cfg = dcfg.get("teacher", {}) if isinstance(dcfg.get("teacher", {}), dict) else {}
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    teacher_cache_paths = teacher_cfg.get("cache_paths", {}) if isinstance(teacher_cfg.get("cache_paths", {}), dict) else {}
    pair_sampler_cfg = dcfg.get("pair_sampler", {}) if isinstance(dcfg.get("pair_sampler", {}), dict) else {}
    pair_cache_paths = pair_sampler_cfg.get("cache_paths", {}) if isinstance(pair_sampler_cfg.get("cache_paths", {}), dict) else {}
    model_stride = int(cfg.get("model", {}).get("stride", 4))
    single_state: Dict[str, Any] = {
        "enabled": False,
    }

    def _normalize_sequence_name(name: str) -> str:
        s = str(name).strip()
        if s.startswith("rgbd_dataset_"):
            return s
        return f"rgbd_dataset_{s}"

    def _resolve_tum_sequence_name(name: str) -> str:
        desired = _normalize_sequence_name(name)
        if (dataset_root / desired).exists():
            return desired
        alt = str(name).strip()
        if alt and (dataset_root / alt).exists():
            return alt
        if skip_missing:
            raise RuntimeError(
                f"single_sequence.target_sequence '{name}' not found under {dataset_root}. "
                f"Tried '{desired}' and '{alt}'."
            )
        raise FileNotFoundError(f"Sequence folder not found: {dataset_root / desired}")

    def _repeat_count(seq_name: str) -> int:
        try:
            return max(1, int(sequence_repeat.get(seq_name, 1)))
        except Exception:
            return 1

    sequence_sampling_weights_cfg = (
        dcfg.get("sequence_sampling_weights", {})
        if isinstance(dcfg.get("sequence_sampling_weights", {}), dict)
        else {}
    )

    def _sequence_weight_lookup(seq_name: str) -> float:
        seq_full = str(seq_name).strip()
        seq_short = seq_full.replace("rgbd_dataset_", "", 1)
        for k in (seq_full, seq_short):
            if k in sequence_sampling_weights_cfg:
                try:
                    v = float(sequence_sampling_weights_cfg.get(k))
                except Exception:
                    continue
                if math.isfinite(v) and v > 0:
                    return float(v)
        return 1.0

    def _resolve_teacher_cache_path(seq_name: str) -> Any:
        # Per-sequence teacher cache map:
        # 1) full name key (rgbd_dataset_*)
        # 2) short name key (freiburg*)
        # 3) fallback scalar cache_path
        if isinstance(teacher_cache_paths, dict):
            full_key = str(seq_name).strip()
            short_key = full_key.replace("rgbd_dataset_", "", 1)
            for k in (full_key, short_key):
                v = teacher_cache_paths.get(k)
                if v not in (None, ""):
                    return v
        return teacher_cfg.get("cache_path")

    def _resolve_pair_cache_path(seq_name: str) -> Any:
        if isinstance(pair_cache_paths, dict):
            full_key = str(seq_name).strip()
            short_key = full_key.replace("rgbd_dataset_", "", 1)
            for k in (full_key, short_key):
                v = pair_cache_paths.get(k)
                if v not in (None, ""):
                    return v
        v = pair_sampler_cfg.get("cache_path")
        if isinstance(v, str) and ("{sequence}" in v or "{short_sequence}" in v):
            full_key = str(seq_name).strip()
            short_key = full_key.replace("rgbd_dataset_", "", 1)
            return v.replace("{sequence}", full_key).replace("{short_sequence}", short_key)
        return v

    def _manifest_root() -> str | Path:
        return (
            external_cfg.get("root")
            or external_cfg.get("external_root")
            or dcfg.get("external_root")
            or dcfg.get("root")
        )

    def _train_manifests() -> List[str]:
        out = external_cfg.get("train_manifests")
        if not out:
            out = dcfg.get("train_manifests", [])
        return [str(x) for x in out or []]

    def _val_manifests() -> List[str]:
        out = external_cfg.get("val_manifests")
        if not out:
            out = dcfg.get("val_manifests", [])
        return [str(x) for x in out or []]

    def _build_tum_dataset(
        seqs: List[str],
        is_train: bool,
        spacing_min: int,
        spacing_max: int,
        frame_windows: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        ds_list: List[torch.utils.data.Dataset] = []
        for s in seqs:
            seq_dir = dataset_root / s
            if not seq_dir.exists():
                if skip_missing:
                    print(f"[train] WARNING: skipping missing sequence: {seq_dir}")
                    continue
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            try:
                window = (frame_windows or {}).get(str(s))
                frame_start_idx = int(window[0]) if window is not None else 0
                frame_end_idx = int(window[1]) if window is not None else None
                pair_sampler_seq = None
                if is_train:
                    pair_sampler_seq = dict(pair_sampler_cfg)
                    pair_cache_path_seq = _resolve_pair_cache_path(s)
                    if pair_cache_path_seq not in (None, ""):
                        pair_sampler_seq["cache_path"] = pair_cache_path_seq
                ds = TUMRGBDDataset(
                    dataset_root=dataset_root,
                    sequence=s,
                    frame_spacing_min=int(spacing_min),
                    frame_spacing_max=int(spacing_max),
                    frame_start_idx=frame_start_idx,
                    frame_end_idx=frame_end_idx,
                    max_frames=dcfg.get("max_frames"),
                    pad_to=int(dcfg.get("pad_to", 16)),
                    is_train=is_train,
                    augmentation=dcfg.get("augmentation"),
                    max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
                    max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
                    cache_in_memory=bool(dcfg.get("cache_in_memory", False)),
                    cache_to_disk=bool(dcfg.get("cache_to_disk", False)),
                    cache_dir=dcfg.get("cache_dir"),
                    pair_sampler=pair_sampler_seq,
                    total_epochs=int(epochs),
                    teacher_enabled=teacher_enabled,
                    teacher_type=str(teacher_cfg.get("type", "orb2")),
                    teacher_cache_path=_resolve_teacher_cache_path(s),
                    teacher_max_features=int(teacher_cfg.get("max_features", 1000)),
                    teacher_stride=int(teacher_cfg.get("stride", model_stride)),
                    teacher_dilate_radius=int(teacher_cfg.get("dilate_radius", 1)),
                )
            except Exception as exc:
                if skip_missing:
                    print(f"[train] WARNING: skipping sequence {s}: {exc}")
                    continue
                raise
            rep = _repeat_count(s) if is_train else 1
            for _ in range(rep):
                ds_list.append(ds)
        return ds_list

    def _build_manifest_dataset(
        is_train: bool,
        manifest_paths: List[str],
        split_name: str,
    ) -> Optional[torch.utils.data.Dataset]:
        if not manifest_paths:
            return None
        return ManifestRGBDPairDataset(
            manifests=manifest_paths,
            dataset_root=_manifest_root(),
            pad_to=int(dcfg.get("pad_to", 16)),
            depth_scale=float(external_cfg.get("depth_scale", dcfg.get("external_depth_scale", 1000.0))),
            is_train=is_train,
            augmentation=dcfg.get("augmentation"),
            pair_quality=external_cfg.get("pair_quality", dcfg.get("pair_quality", {})),
            pair_mining=dcfg.get("pair_mining", {}) if is_train else {},
            split_name=split_name,
        )

    def _build_concat(ds_list: List[torch.utils.data.Dataset], split: str):
        if len(ds_list) == 0:
            raise RuntimeError(
                f"No valid datasets available for split='{split}'. "
                "Check dataset config (sequences/manifests) and paths."
            )
        if len(ds_list) == 1:
            return ds_list[0]
        return ConcatDataset(ds_list)

    single_target_seq: Optional[str] = None
    train_frame_window: Optional[Tuple[int, int]] = None
    holdout_frame_window: Optional[Tuple[int, int]] = None
    split_mode = str(single_cfg.get("split_mode", "dual")).strip().lower()
    if split_mode not in {"dual", "overfit_only", "split_only"}:
        split_mode = "dual"
    if single_enabled:
        if dataset_mode not in {"tum", "mixed"}:
            raise RuntimeError(
                "dataset.single_sequence.enabled=true requires dataset.mode to include TUM data."
            )
        target_raw = str(single_cfg.get("target_sequence", "freiburg1_desk")).strip()
        single_target_seq = _resolve_tum_sequence_name(target_raw)

        probe_spacing_min = 1
        probe_spacing_max = 1
        probe_ds = TUMRGBDDataset(
            dataset_root=dataset_root,
            sequence=single_target_seq,
            frame_spacing_min=probe_spacing_min,
            frame_spacing_max=probe_spacing_max,
            frame_start_idx=0,
            frame_end_idx=None,
            max_frames=dcfg.get("max_frames"),
            pad_to=int(dcfg.get("pad_to", 16)),
            is_train=False,
            augmentation=None,
            max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
            max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
            cache_in_memory=False,
            cache_to_disk=False,
            cache_dir=None,
            pair_sampler=None,
            total_epochs=1,
        )
        total_frames = int(len(getattr(probe_ds, "frames", [])))
        if total_frames <= 0:
            raise RuntimeError(f"single_sequence probe found no frames for {single_target_seq}")

        train_ratio = float(single_cfg.get("train_ratio", 0.70))
        if not math.isfinite(train_ratio):
            train_ratio = 0.70
        if split_mode == "overfit_only":
            # Overfit mode can intentionally use the full sequence.
            train_ratio = min(max(train_ratio, 0.05), 1.0)
        else:
            train_ratio = min(max(train_ratio, 0.05), 0.95)

        train_max_delta = int(dcfg.get("frame_spacing_max", 4))
        pair_cfg = dcfg.get("pair_sampler", {}) if isinstance(dcfg.get("pair_sampler", {}), dict) else {}
        for k in ("short_range", "medium_range", "hard_range"):
            rv = pair_cfg.get(k)
            if isinstance(rv, (list, tuple)) and len(rv) == 2:
                try:
                    train_max_delta = max(train_max_delta, int(rv[1]))
                except Exception:
                    pass

        val_cfg = cfg.get("validation", {}).get("splits", {})
        val_max_delta = int(dcfg.get("frame_spacing_max", 4))
        if isinstance(val_cfg, dict):
            for _split_name, scfg in val_cfg.items():
                if not isinstance(scfg, dict):
                    continue
                if not bool(scfg.get("enabled", True)):
                    continue
                try:
                    val_max_delta = max(val_max_delta, int(scfg.get("frame_spacing_max", val_max_delta)))
                except Exception:
                    pass

        min_train_frames = int(single_cfg.get("min_train_frames", 0))
        min_train_frames = max(min_train_frames, 1 + int(train_max_delta))
        min_holdout_frames = int(single_cfg.get("min_holdout_frames", 0))
        min_holdout_frames = max(min_holdout_frames, 1 + int(val_max_delta))
        if split_mode == "overfit_only":
            min_holdout_frames = 0

        if total_frames < (min_train_frames + min_holdout_frames):
            raise RuntimeError(
                f"single_sequence split infeasible for {single_target_seq}: total_frames={total_frames}, "
                f"min_train_frames={min_train_frames}, min_holdout_frames={min_holdout_frames}"
            )

        if split_mode == "overfit_only":
            train_end = int(total_frames)
        else:
            train_end = int(round(float(total_frames) * float(train_ratio)))
            train_end = max(min_train_frames, train_end)
            max_train_end = total_frames - min_holdout_frames
            train_end = min(max_train_end, train_end)

        train_frame_window = (0, int(train_end))
        if split_mode == "overfit_only":
            holdout_frame_window = train_frame_window
        else:
            holdout_frame_window = (int(train_end), int(total_frames))

        single_state = {
            "enabled": True,
            "target_sequence": single_target_seq,
            "target_sequence_short": single_target_seq.replace("rgbd_dataset_", "", 1),
            "split_mode": split_mode,
            "train_ratio": float(train_ratio),
            "total_frames": int(total_frames),
            "train_window": {"start_idx": int(train_frame_window[0]), "end_idx": int(train_frame_window[1])},
            "holdout_window": {"start_idx": int(holdout_frame_window[0]), "end_idx": int(holdout_frame_window[1])},
        }
        print(
            "[train] single-sequence mode "
            f"target={single_target_seq} split_mode={split_mode} "
            f"train_window=[{train_frame_window[0]}:{train_frame_window[1]}] "
            f"holdout_window=[{holdout_frame_window[0]}:{holdout_frame_window[1]}] total_frames={total_frames}"
        )

        if dataset_mode == "mixed":
            print("[train] single-sequence mode: manifest branch disabled; using TUM-only data for this run.")

    train_ds_list: List[torch.utils.data.Dataset] = []
    if dataset_mode in {"tum", "mixed"}:
        if single_enabled and single_target_seq is not None:
            train_seqs = [single_target_seq]
            train_windows = {single_target_seq: train_frame_window} if train_frame_window is not None else None
        else:
            train_seqs = list(dcfg.get("train_sequences", []))
            train_windows = None
        train_ds_list.extend(
            _build_tum_dataset(
                seqs=train_seqs,
                is_train=True,
                spacing_min=int(dcfg.get("frame_spacing_min", 1)),
                spacing_max=int(dcfg.get("frame_spacing_max", 4)),
                frame_windows=train_windows,
            )
        )
    if dataset_mode in {"manifest", "mixed"} and not single_enabled:
        m = _build_manifest_dataset(
            is_train=True,
            manifest_paths=_train_manifests(),
            split_name="train",
        )
        if m is not None:
            train_ds_list.append(m)
    train_ds = _build_concat(train_ds_list, split="train")

    val_loaders_cfg = cfg.get("validation", {}).get("splits", {})
    if not isinstance(val_loaders_cfg, dict) or len(val_loaders_cfg) == 0:
        val_loaders_cfg = {
            "short": {
                "frame_spacing_min": 1,
                "frame_spacing_max": 2,
                "sequences": list(dcfg.get("val_sequences", [])),
            },
            "hard": {
                "frame_spacing_min": 4,
                "frame_spacing_max": 8,
                "sequences": list(dcfg.get("val_sequences", [])),
            },
        }

    val_datasets: Dict[str, torch.utils.data.Dataset] = {}
    for split_name, scfg in val_loaders_cfg.items():
        if not bool(scfg.get("enabled", True)):
            continue
        ds_list: List[torch.utils.data.Dataset] = []
        if dataset_mode in {"tum", "mixed"}:
            if single_enabled and single_target_seq is not None:
                seqs = [single_target_seq]
                val_windows = {single_target_seq: holdout_frame_window} if holdout_frame_window is not None else None
            else:
                seqs = list(scfg.get("sequences", dcfg.get("val_sequences", [])))
                val_windows = None
            ds_list.extend(
                _build_tum_dataset(
                    seqs=seqs,
                    is_train=False,
                    spacing_min=int(scfg.get("frame_spacing_min", dcfg.get("frame_spacing_min", 1))),
                    spacing_max=int(scfg.get("frame_spacing_max", dcfg.get("frame_spacing_max", 4))),
                    frame_windows=val_windows,
                )
            )
        if dataset_mode in {"manifest", "mixed"} and not single_enabled:
            manifests = list(scfg.get("manifests", _val_manifests()))
            m = _build_manifest_dataset(
                is_train=False,
                manifest_paths=manifests,
                split_name=f"val_{split_name}",
            )
            if m is not None:
                ds_list.append(m)
        if len(ds_list) == 0:
            print(f"[train] WARNING: skipping empty validation split '{split_name}'")
            continue
        val_datasets[str(split_name)] = _build_concat(ds_list, split=f"val:{split_name}")

    num_workers = int(tcfg.get("num_workers", 8))
    if num_workers < 0:
        num_workers = max(1, min(64, (os.cpu_count() or 8) // 2))
    pair_sampler_cfg_runtime = dcfg.get("pair_sampler", {}) if isinstance(dcfg.get("pair_sampler", {}), dict) else {}
    pair_sched = pair_sampler_cfg_runtime.get("schedule")
    pair_dist_sched = pair_sampler_cfg_runtime.get("pair_distance_schedule")
    persistent = (num_workers > 0) and not (isinstance(pair_sched, dict) or isinstance(pair_dist_sched, dict))

    def _worker_init_fn(worker_id: int) -> None:
        wseed = int(seed) + int(worker_id) * 97 + 13
        random.seed(wseed)
        np.random.seed(wseed % (2**32 - 1))
        torch.manual_seed(wseed)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seed))
    train_sampler = None
    if isinstance(train_ds, ConcatDataset) and sequence_sampling_weights_cfg:
        sample_weights: List[float] = []
        for child in train_ds.datasets:
            n_child = int(len(child))
            if n_child <= 0:
                continue
            seq_name = getattr(child, "sequence", "")
            w_seq = _sequence_weight_lookup(str(seq_name))
            per_sample = max(1.0e-8, float(w_seq) / float(n_child))
            sample_weights.extend([per_sample] * n_child)
        if len(sample_weights) == int(len(train_ds)) and len(sample_weights) > 0:
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=int(len(sample_weights)),
                replacement=True,
                generator=loader_gen,
            )
            print(
                "[train] sequence_sampling_weights active: "
                + ", ".join(
                    f"{k}={v}" for k, v in sorted(sequence_sampling_weights_cfg.items(), key=lambda x: str(x[0]))
                )
            )

    common_loader_kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=loader_gen,
    )
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = int(tcfg.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loaders = {
        name: DataLoader(
            ds,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )
        for name, ds in val_datasets.items()
    }
    return train_loader, val_loaders, train_ds, single_state

def _build_model(cfg: Dict[str, Any]) -> LocalFeatureNet:
    m = cfg["model"]
    heads_cfg = m.get("heads", {}) if isinstance(m.get("heads", {}), dict) else {}
    fusion_cfg = m.get("fusion", {}) if isinstance(m.get("fusion", {}), dict) else {}
    pseudo_cfg = m.get("pseudo_object", {}) if isinstance(m.get("pseudo_object", {}), dict) else {}
    backbone_tuning_cfg = m.get("backbone_tuning", {}) if isinstance(m.get("backbone_tuning", {}), dict) else {}
    backbone_tuning_mode = str(backbone_tuning_cfg.get("mode", "")).strip().lower()
    if backbone_tuning_mode not in {"frozen", "full", "last_block"}:
        backbone_tuning_mode = "frozen" if bool(m.get("freeze_backbone", True)) else "full"
    head_channels = heads_cfg.get("head_channels", None)
    if head_channels is not None:
        head_channels = int(head_channels)
    return LocalFeatureNet(
        dinov3_name=m["dinov3"]["name_or_path"],
        patch_size=int(m.get("patch_size", 16)),
        descriptor_dim=int(heads_cfg.get("descriptor_dim", 256)),
        fine_channels=int(m["fine_cnn"].get("channels", 96)),
        fine_blocks=int(m["fine_cnn"].get("num_blocks", 8)),
        freeze_backbone=(backbone_tuning_mode == "frozen"),
        use_offset=bool(heads_cfg.get("offset", {}).get("enabled", True)),
        use_reliability=bool(heads_cfg.get("reliability", {}).get("enabled", True)),
        dinov3_dtype=str(m["dinov3"].get("dtype", "bf16")),
        head_variant=str(heads_cfg.get("variant", "v1")),
        head_channels=head_channels,
        head_tower_depth=int(heads_cfg.get("tower_depth", 2)),
        head_norm=str(heads_cfg.get("norm", "group")),
        head_act=str(heads_cfg.get("act", "silu")),
        detector_logit_scale=float(heads_cfg.get("detector_logit_scale", 1.0)),
        reliability_logit_scale=float(heads_cfg.get("reliability_logit_scale", 1.0)),
        desc_relgate_detach=bool(heads_cfg.get("desc_relgate_detach", False)),
        fusion_variant=str(fusion_cfg.get("variant", "v1")),
        fusion_channels=int(fusion_cfg.get("channels", 256)),
        pseudo_object_enabled=bool(pseudo_cfg.get("enabled", False)),
        pseudo_object_k=int(pseudo_cfg.get("k", 5)),
        pseudo_object_temperature=float(pseudo_cfg.get("temperature", 10.0)),
    )


def _configure_backbone_tuning(model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    mcfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    tuning_cfg = mcfg.get("backbone_tuning", {}) if isinstance(mcfg.get("backbone_tuning", {}), dict) else {}
    mode = str(tuning_cfg.get("mode", "")).strip().lower()
    if mode not in {"frozen", "full", "last_block"}:
        mode = "frozen" if bool(mcfg.get("freeze_backbone", True)) else "full"

    backbone = getattr(model, "backbone", None)
    core = getattr(backbone, "model", None)
    if core is None:
        print(f"[train] WARNING: backbone tuning mode '{mode}' ignored (model.backbone.model missing)")
        return

    named_params = list(core.named_parameters())
    if len(named_params) == 0:
        print(f"[train] WARNING: backbone tuning mode '{mode}' ignored (no backbone params)")
        return

    if mode == "frozen":
        for _name, p in named_params:
            p.requires_grad_(False)
        print("[train] backbone_tuning.mode=frozen")
        return

    if mode == "full":
        for _name, p in named_params:
            p.requires_grad_(True)
        print("[train] backbone_tuning.mode=full")
        return

    # mode == "last_block"
    patterns = (
        re.compile(r"(?:^|\.)(?:blocks)\.(\d+)\."),
        re.compile(r"(?:^|\.)(?:layers)\.(\d+)\."),
        re.compile(r"(?:^|\.)(?:encoder\.layer)\.(\d+)\."),
    )

    def _layer_index(name: str) -> Optional[int]:
        for pat in patterns:
            m = pat.search(name)
            if m is not None:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    max_idx = -1
    for n, _p in named_params:
        idx = _layer_index(n)
        if idx is not None:
            max_idx = max(max_idx, int(idx))

    for _name, p in named_params:
        p.requires_grad_(False)

    unfrozen = 0
    if max_idx >= 0:
        for n, p in named_params:
            idx = _layer_index(n)
            if idx == max_idx:
                p.requires_grad_(True)
                unfrozen += int(p.numel())
            elif idx is None and ("norm" in n.lower() or "ln_" in n.lower()):
                p.requires_grad_(True)
                unfrozen += int(p.numel())
    else:
        # Fallback if block naming is unavailable: unfreeze the last 10% params.
        cut = int(max(0, round(0.90 * len(named_params))))
        for i, (_n, p) in enumerate(named_params):
            if i >= cut:
                p.requires_grad_(True)
                unfrozen += int(p.numel())

    print(
        "[train] backbone_tuning.mode=last_block "
        f"(detected_max_block={max_idx}, trainable_backbone_params={unfrozen})"
    )


def _apply_trainable_config(model: torch.nn.Module, tcfg: Dict[str, Any]) -> None:
    trainable_prefixes = tcfg.get("trainable_param_prefixes")
    freeze_prefixes = tcfg.get("freeze_param_prefixes")

    def _norm_prefixes(v):
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]

    trainable_prefixes = _norm_prefixes(trainable_prefixes)
    freeze_prefixes = _norm_prefixes(freeze_prefixes)

    if trainable_prefixes:
        for name, p in model.named_parameters():
            p.requires_grad_(any(name.startswith(pref) for pref in trainable_prefixes))
        print(f"[train] trainable_param_prefixes={trainable_prefixes}")
    elif freeze_prefixes:
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in freeze_prefixes):
                p.requires_grad_(False)
        print(f"[train] freeze_param_prefixes={freeze_prefixes}")

    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"[train] trainable params: {trainable}/{total}")


def _iter_leaf_datasets(ds):
    if isinstance(ds, ConcatDataset):
        for child in ds.datasets:
            yield from _iter_leaf_datasets(child)
    else:
        yield ds


def _set_dataset_epoch(ds, epoch: int, epochs: int) -> None:
    for leaf in _iter_leaf_datasets(ds):
        fn = getattr(leaf, "set_epoch", None)
        if callable(fn):
            fn(epoch, epochs)


def _enforce_train_modes(
    model: torch.nn.Module,
    freeze_non_trainable_modules: bool,
    freeze_bn_running_stats: bool,
) -> None:
    if not freeze_non_trainable_modules:
        return

    bn_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if len(params) == 0:
            continue
        any_trainable = any(bool(p.requires_grad) for p in params)
        if not any_trainable:
            module.eval()
            if freeze_bn_running_stats and isinstance(module, bn_types):
                module.track_running_stats = True


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu")


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Invalid checkpoint: could not find model state_dict")
    return state


def _load_model_weights(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(ckpt_path)
    state = _extract_model_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(
        f"[train] loaded model from {ckpt_path} "
        f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[train] missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[train] unexpected keys (first 10): {unexpected[:10]}")
    return ckpt

@torch.no_grad()
def _val_diagnostics(cfg: Dict[str, Any], batch: Dict[str, torch.Tensor], out1, out2) -> Dict[str, float]:
    import numpy as np
    import cv2
    import torch.nn.functional as F

    def _as_bK(bK: torch.Tensor) -> torch.Tensor:
        return bK.unsqueeze(0) if bK.dim() == 2 else bK

    def _as_bT(bT: torch.Tensor) -> torch.Tensor:
        return bT.unsqueeze(0) if bT.dim() == 2 else bT

    def _mutual_nn_matches(d1: torch.Tensor, d2: torch.Tensor):
        if d1.numel() == 0 or d2.numel() == 0:
            return None, None
        sim = d1 @ d2.t()
        nn12 = sim.argmax(dim=1)
        nn21 = sim.argmax(dim=0)
        ids = torch.arange(sim.shape[0], device=sim.device)
        mutual = (nn21[nn12] == ids)
        idx1 = ids[mutual]
        idx2 = nn12[mutual]
        if idx1.numel() == 0:
            return None, None
        return idx1, idx2

    def _sample_depth_bilinear(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # depth: (1,1,H,W), xy: (N,2)
        H, W = depth.shape[-2:]
        x = xy[:, 0]
        y = xy[:, 1]
        gx = (x / (W - 1)) * 2 - 1
        gy = (y / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        z = F.grid_sample(depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return z.view(-1)

    def _pnp_ransac_inliers(
        xy1: torch.Tensor,
        xy2: torch.Tensor,
        depth1: torch.Tensor,
        K: torch.Tensor,
        reproj_px: float = 3.0,
        z_min_m: float = 0.10,
        max_iters: int = 2000,
        conf: float = 0.999,
    ) -> Tuple[int, float]:
        if xy1.numel() == 0:
            return 0, 0.0
        z = _sample_depth_bilinear(depth1, xy1)
        valid = torch.isfinite(z) & (z > z_min_m)
        if valid.sum().item() < 6:
            return 0, 0.0
        xy1v = xy1[valid]
        xy2v = xy2[valid]
        zv = z[valid]

        fx = float(K[0, 0].item())
        fy = float(K[1, 1].item())
        cx = float(K[0, 2].item())
        cy = float(K[1, 2].item())

        X = (xy1v[:, 0] - cx) / fx * zv
        Y = (xy1v[:, 1] - cy) / fy * zv
        Z = zv
        pts3d = torch.stack([X, Y, Z], dim=-1)

        obj = pts3d.detach().cpu().numpy().astype(np.float32)
        img = xy2v.detach().cpu().numpy().astype(np.float32)
        Kcv = K.detach().cpu().numpy().astype(np.float64)

        if obj.shape[0] < 6:
            return 0, 0.0

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=Kcv,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(reproj_px),
            iterationsCount=int(max_iters),
            confidence=float(conf),
        )
        if not ok or inliers is None:
            return 0, 0.0
        ninl = int(inliers.shape[0])
        ratio = ninl / float(obj.shape[0])
        return ninl, float(ratio)

    stride = int(cfg["model"].get("stride", 4))
    det = cfg["model"]["heads"]["detector"]
    loss_cfg = cfg.get("loss", {})
    use_rel_score = bool(cfg.get("inference", {}).get("use_reliability_in_score", False))
    inlier_px = float(loss_cfg.get("diag_inlier_px", 3.0))
    pnp_px = float(loss_cfg.get("diag_pnp_px", 3.0))
    z_min = float(loss_cfg.get("z_min_m", 0.10))

    k1 = extract_keypoints_torch(
        out1.heatmap, out1.desc, out1.offset, out1.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        valid_mask_img=batch.get("valid_depth1", None),
        use_reliability_in_score=use_rel_score,
        adaptive_tiling=bool(det.get("adaptive_tiling", False)),
        adaptive_k_min=int(det.get("adaptive_k_min", 1)),
        adaptive_k_max=det.get("adaptive_k_max"),
    )
    k2 = extract_keypoints_torch(
        out2.heatmap, out2.desc, out2.offset, out2.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        valid_mask_img=batch.get("valid_depth2", None),
        use_reliability_in_score=use_rel_score,
        adaptive_tiling=bool(det.get("adaptive_tiling", False)),
        adaptive_k_min=int(det.get("adaptive_k_min", 1)),
        adaptive_k_max=det.get("adaptive_k_max"),
    )

    # only first sample
    d1 = k1.desc[0]
    d2 = k2.desc[0]
    xy1_all = k1.xy_img[0]
    xy2_all = k2.xy_img[0]

    if d1.numel() == 0 or d2.numel() == 0:
        return {
            "kpts1": float(d1.shape[0] if d1.dim() > 0 else 0),
            "kpts2": float(d2.shape[0] if d2.dim() > 0 else 0),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    idx1, idx2 = _mutual_nn_matches(d1, d2)
    if idx1 is None:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    idx1 = idx1.long()
    idx2 = idx2.long()
    xy1m = xy1_all[idx1].view(1, -1, 2)
    xy2m = xy2_all[idx2].view(1, -1, 2)

    mcount = int(xy1m.shape[1])
    if mcount == 0:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    depth1 = batch["depth1"][0:1]  # (1,1,H,W)
    Kb = _as_bK(batch["K"])[0:1]
    T21 = _as_bT(batch["relative_pose"])[0:1]

    # GT reprojection
    pts1 = unproject(depth1, Kb, xy1m)
    pts2 = transform(T21, pts1)
    xy2_gt = project(pts2, Kb)

    H, W = depth1.shape[-2:]
    xg, yg = xy2_gt[..., 0], xy2_gt[..., 1]
    inb = (xg >= 0) & (xg <= (W - 1)) & (yg >= 0) & (yg <= (H - 1)) & torch.isfinite(xy2_gt).all(dim=-1)
    valid_ratio = float(inb.float().mean().item())

    err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)[0]
    err_v = err[inb[0]]
    if err_v.numel() == 0:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": float(mcount),
            "valid_match_ratio": valid_ratio,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    inl = err_v < inlier_px
    inlier_count = int(inl.sum().item())
    inlier_rate = float(inl.float().mean().item())
    mean_all = float(err_v.mean().item())
    mean_inl = float(err_v[inl].mean().item()) if inl.any() else 0.0
    med_inl = float(err_v[inl].median().item()) if inl.any() else 0.0

    # PnP-RANSAC on CPU (OpenCV)
    ninl_pnp, rate_pnp = _pnp_ransac_inliers(
        xy1=xy1m[0],
        xy2=xy2m[0],
        depth1=depth1,
        K=Kb[0],
        reproj_px=pnp_px,
        z_min_m=z_min,
    )

    return {
        "kpts1": float(d1.shape[0]),
        "kpts2": float(d2.shape[0]),
        "matches": float(mcount),
        "valid_match_ratio": valid_ratio,

        f"inliers@{inlier_px:.0f}px": float(inlier_count),
        f"inlier_rate@{inlier_px:.0f}px": float(inlier_rate),

        f"pnp_inliers@{pnp_px:.0f}px": float(ninl_pnp),
        f"pnp_inlier_rate@{pnp_px:.0f}px": float(rate_pnp),

        "mean_reproj_err": mean_all,
        "mean_reproj_err_inliers": mean_inl,
        "median_reproj_err_inliers": med_inl,
    }


def train(cfg: Dict[str, Any]) -> None:
    device = _device(cfg)
    tcfg = cfg["training"]
    seed = int(tcfg.get("seed", 42))
    cudnn_deterministic = bool(tcfg.get("cudnn_deterministic", False))
    cudnn_benchmark = bool(tcfg.get("cudnn_benchmark", not cudnn_deterministic))
    _seed_everything(seed=seed, deterministic=cudnn_deterministic, cudnn_benchmark=cudnn_benchmark)

    sharing_strategy = str(tcfg.get("sharing_strategy", "")).strip().lower()
    if sharing_strategy:
        try:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
            print(f"[train] torch multiprocessing sharing_strategy={sharing_strategy}")
        except Exception as exc:
            print(f"[train] WARNING: failed to set sharing_strategy={sharing_strategy}: {exc}")

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    experiment_id = _resolve_experiment_id(cfg)
    run_cfg["experiment_id"] = experiment_id
    cfg["run"] = run_cfg
    out_dir = ensure_dir(Path(run_cfg["out_dir"]) / run_cfg["name"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    tracking_cfg = run_cfg.get("tracking", {}) if isinstance(run_cfg.get("tracking", {}), dict) else {}
    exp_store: Optional[ExperimentStore] = None
    try:
        sqlite_path = tracking_cfg.get("sqlite_path", "runs/experiments.db")
        exp_store = ExperimentStore(sqlite_path)
        exp_store.upsert_experiment(
            experiment_id=experiment_id,
            run_name=str(run_cfg.get("name", "unknown")),
            prompt_tag=str(run_cfg.get("prompt_tag", "default")),
            parent_id=(None if run_cfg.get("parent_experiment_id") in (None, "") else str(run_cfg.get("parent_experiment_id"))),
            git_commit=_git_commit(),
            config_hash=_config_hash(cfg),
            notes=(None if run_cfg.get("ablation_tag") in (None, "") else str(run_cfg.get("ablation_tag"))),
        )
        if bool(tracking_cfg.get("save_manifest_json", True)):
            env_manifest = {
                "CKPT": str(os.environ.get("CKPT", "")),
                "SEQUENCES": str(os.environ.get("SEQUENCES", "")),
                "PYTHON_BIN": str(os.environ.get("PYTHON_BIN", "")),
            }
            manifest_path = write_run_manifest(
                out_dir=out_dir,
                cfg=cfg,
                experiment_id=experiment_id,
                config_hash=_config_hash(cfg),
                git_commit=_git_commit(),
                env=env_manifest,
            )
            exp_store.log_artifact(experiment_id=experiment_id, artifact_type="run_manifest", path=manifest_path)
    except Exception as exc:
        exp_store = None
        print(f"[train] WARNING: experiment tracking disabled due to error: {exc}")

    epochs = int(tcfg["epochs"])
    train_loader, val_loaders, train_ds, single_seq_state = _make_loaders(cfg, epochs=epochs, seed=seed)
    model = _build_model(cfg).to(device)
    _configure_backbone_tuning(model, cfg)
    channels_last = bool(tcfg.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    _apply_trainable_config(model, tcfg)
    freeze_non_trainable_modules = bool(tcfg.get("freeze_non_trainable_modules", True))
    freeze_bn_running_stats = bool(tcfg.get("freeze_bn_running_stats", True))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters after applying training.{trainable,freeze}_param_prefixes.")

    opt = AdamW(
        trainable_params,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tcfg.get("lr_min", 1e-6)))
    start_epoch = 1

    init_ckpt = tcfg.get("init_checkpoint")
    resume_ckpt = tcfg.get("resume_checkpoint")
    init_strict = bool(tcfg.get("init_strict", False))
    if init_ckpt and resume_ckpt:
        raise ValueError("Use only one of training.init_checkpoint or training.resume_checkpoint.")

    selection_metric = str(
        run_cfg.get(
            "primary_selection_metric",
            run_cfg.get("selection_metric", "val_short_loss_total"),
        )
    ).lower()
    allow_selection_fallback = bool(run_cfg.get("allow_selection_fallback", False))
    secondary_guard_metric = str(run_cfg.get("secondary_guard_metric", f"val_hard_inlier_rate@3px"))
    metric_guard_cfg = tcfg.get("metric_guard", {}) if isinstance(tcfg.get("metric_guard", {}), dict) else {}
    hard_regression_max = float(metric_guard_cfg.get("hard_inlier_regression_max", 0.10))
    best_selection = float("inf")
    best_selection_epoch = 0
    guard_metric_l = secondary_guard_metric.lower()
    guard_higher_better = not any(tok in guard_metric_l for tok in ("loss", "err", "rmse", "mean"))
    best_guard_value = -float("inf") if guard_higher_better else float("inf")
    best_rel_score = float("inf")
    best_val = float("inf")
    rel_only = list(tcfg.get("trainable_param_prefixes") or []) == ["heads.rel"]
    if resume_ckpt:
        ckpt = _load_model_weights(model, resume_ckpt, strict=init_strict)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            sch.load_state_dict(ckpt["scheduler"])
        else:
            done_epochs = int(ckpt.get("epoch", 0))
            for _ in range(done_epochs):
                sch.step()
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        best_selection = float(ckpt.get("best_selection", best_selection))
        best_selection_epoch = int(ckpt.get("best_selection_epoch", best_selection_epoch))
        print(f"[train] resuming from epoch {start_epoch}")
    elif init_ckpt:
        _load_model_weights(model, init_ckpt, strict=init_strict)

    eager_model = model
    compile_active = False
    if bool(tcfg.get("torch_compile", False)):
        compile_mode = str(tcfg.get("compile_mode", "max-autotune"))
        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=bool(tcfg.get("compile_fullgraph", False)),
                dynamic=bool(tcfg.get("compile_dynamic", False)),
            )
            compile_active = True
            print(f"[train] torch.compile enabled (mode={compile_mode})")
        except Exception as exc:
            print(f"[train] WARNING: torch.compile failed, continuing without it: {exc}")

    stride = int(cfg["model"].get("stride", 4))
    use_amp = bool(tcfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(tcfg.get("amp_dtype", "bf16")).lower() in ("bf16", "bfloat16") else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 0.0) or 0.0)
    loss_fp32 = bool(tcfg.get("loss_fp32", True))
    enable_validation = bool(tcfg.get("enable_validation", True))
    val_every = max(1, int(tcfg.get("val_every", 1)))
    inlier_px = float(cfg.get("loss", {}).get("diag_inlier_px", 3.0))
    inlier_key = f"inlier_rate@{inlier_px:.0f}px"

    ate_cfg = cfg.get("validation", {}).get("semantic_eval", {})
    ate_eval_every = int(cfg.get("validation", {}).get("ate_eval_every", 0) or 0)
    early_cfg = tcfg.get("early_stop", {}) if isinstance(tcfg.get("early_stop", {}), dict) else {}
    early_stop_enabled = bool(early_cfg.get("enabled", False))
    early_stop_min_epochs = int(early_cfg.get("min_epochs", 6))
    early_stop_patience = int(early_cfg.get("patience", 4))
    early_stop_min_delta = float(early_cfg.get("min_delta", 0.0))
    stop_training = False
    stage_name = str(run_cfg.get("name", "unknown"))
    if selection_metric in {
        "ate_short_mean",
        "ate_short_mean_penalized",
        "weighted_penalized_ate",
        "holdout_weighted_penalized_ate",
        "overfit_weighted_penalized_ate",
        "weighted_mean_ate",
        "holdout_weighted_mean_ate",
        "overfit_weighted_mean_ate",
        "weighted_mean_ate_robust",
        "holdout_weighted_mean_ate_robust",
        "overfit_weighted_mean_ate_robust",
    } and ate_eval_every <= 0:
        print("[train] WARNING: selection metric requires validation.ate_eval_every>0; falling back to val_short_loss_total.")
        selection_metric = "val_short_loss_total"

    dcfg_runtime = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    dataset_root_runtime = Path(str(dcfg_runtime.get("root", "."))).expanduser()

    def _normalize_seq_short(name: str) -> str:
        s = str(name).strip()
        if s.startswith("rgbd_dataset_"):
            s = s[len("rgbd_dataset_") :]
        return s

    def _ensure_sequence_associations(seq_short: str) -> Path:
        seq_dir = (dataset_root_runtime / f"rgbd_dataset_{seq_short}").resolve()
        assoc_file = seq_dir / "associations.txt"
        if assoc_file.exists():
            return assoc_file
        rgb_txt = seq_dir / "rgb.txt"
        depth_txt = seq_dir / "depth.txt"
        associate_script = (Path.cwd() / "pyslam_integration" / "scripts" / "associate.py").resolve()
        if not associate_script.exists():
            raise FileNotFoundError(f"associate.py not found at {associate_script}")
        if not rgb_txt.exists() or not depth_txt.exists():
            raise FileNotFoundError(
                f"Cannot build associations for {seq_short}: missing rgb/depth txt under {seq_dir}"
            )
        proc = subprocess.run(
            [sys.executable, str(associate_script), str(rgb_txt), str(depth_txt), "--output", str(assoc_file)],
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not assoc_file.exists():
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            raise RuntimeError(f"Failed generating associations for {seq_short}: rc={proc.returncode}\n{tail}")
        return assoc_file

    def _build_association_window(
        *,
        sequence_short: str,
        start_idx: int,
        end_idx: int,
        out_path: Path,
    ) -> Path:
        assoc_src = _ensure_sequence_associations(sequence_short)
        builder = (Path.cwd() / "scripts" / "build_associations_window.py").resolve()
        if not builder.exists():
            raise FileNotFoundError(f"build_associations_window.py not found: {builder}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(builder),
                "--input",
                str(assoc_src),
                "--output",
                str(out_path),
                "--start-idx",
                str(int(start_idx)),
                "--end-idx",
                str(int(end_idx)),
            ],
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not out_path.exists():
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            raise RuntimeError(
                f"Failed building associations window for {sequence_short}: rc={proc.returncode}\n{tail}"
            )
        return out_path.resolve()

    def _load_scored_csv(
        *,
        csv_path: Path,
        sequences: List[str],
        sequence_weights_cfg: Dict[str, Any],
        missing_penalty: float,
        status_penalties_cfg: Dict[str, Any],
        min_coverage_ok: float,
        primary_ate_field: str,
        robust_ate_field: str,
    ) -> Dict[str, Any]:
        rows_raw: List[Dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows_raw.append(
                    {
                        "sequence": str(r.get("sequence", "")).strip(),
                        "status": str(r.get("status", "")).strip().lower(),
                        "ate_rmse": _safe_float(r.get("ate_rmse"), default=float("nan")),
                        "ate_rmse_associated": _safe_float(r.get("ate_rmse_associated"), default=float("nan")),
                        "rpe_trans_rmse": _safe_float(r.get("rpe_trans_rmse"), default=float("nan")),
                        "rpe_rot_rmse": _safe_float(r.get("rpe_rot_rmse"), default=float("nan")),
                        "coverage": _safe_float(r.get("coverage"), default=float("nan")),
                    }
                )
        return _score_semantic_eval(
            rows=rows_raw,
            sequences=sequences,
            sequence_weights_cfg=sequence_weights_cfg,
            missing_penalty=missing_penalty,
            status_penalties_cfg=status_penalties_cfg,
            min_coverage_ok=min_coverage_ok,
            primary_ate_field=primary_ate_field,
            robust_ate_field=robust_ate_field,
        )

    def _run_ate_eval(epoch_ckpt: Path, epoch: int) -> Dict[str, Any] | None:
        if ate_eval_every <= 0:
            return None
        strict_preflight = bool(ate_cfg.get("strict_preflight", False))
        script = Path(str(ate_cfg.get("script", "pyslam_integration/run_semantic.sh"))).expanduser()
        if not script.is_absolute():
            script = (Path.cwd() / script).resolve()
        if not script.exists():
            print(f"[train] WARNING: semantic eval script not found: {script}")
            return None
        default_sequences = cfg.get("validation", {}).get(
            "short_eval_sequences",
            ["freiburg1_desk", "freiburg1_plant", "freiburg1_room"],
        )
        seq_cfg = ate_cfg.get("sequences", default_sequences)
        if isinstance(seq_cfg, str):
            sequences = [x.strip() for x in seq_cfg.split(",") if x.strip()]
        else:
            sequences = [str(s) for s in seq_cfg]
        if len(sequences) == 0:
            print("[train] WARNING: semantic eval requested but sequence list is empty")
            return None
        sequence_weights_cfg = ate_cfg.get("sequence_weights", {}) if isinstance(ate_cfg.get("sequence_weights", {}), dict) else {}
        status_penalties_cfg = ate_cfg.get("status_penalties", {}) if isinstance(ate_cfg.get("status_penalties", {}), dict) else {}
        missing_penalty = float(ate_cfg.get("missing_penalty", 3.0))
        min_coverage_ok = float(ate_cfg.get("min_coverage_ok", 0.95))
        primary_ate_field = str(ate_cfg.get("primary_ate_field", "ate_rmse_associated"))
        robust_ate_field = str(ate_cfg.get("robust_ate_field", "ate_rmse"))
        selection_target = str(ate_cfg.get("selection_target", "holdout")).strip().lower()
        if selection_target not in {"holdout", "overfit"}:
            selection_target = "holdout"

        eval_profiles_cfg = ate_cfg.get("eval_profiles", None)
        if isinstance(eval_profiles_cfg, str):
            eval_profiles = [x.strip().lower() for x in eval_profiles_cfg.split(",") if x.strip()]
        elif isinstance(eval_profiles_cfg, list):
            eval_profiles = [str(x).strip().lower() for x in eval_profiles_cfg if str(x).strip()]
        else:
            eval_profiles = []
        if not eval_profiles:
            if bool(single_seq_state.get("enabled", False)):
                mode = str(single_seq_state.get("split_mode", "dual")).strip().lower()
                if mode == "overfit_only":
                    eval_profiles = ["overfit"]
                elif mode == "split_only":
                    eval_profiles = ["holdout"]
                else:
                    eval_profiles = ["overfit", "holdout"]
            else:
                eval_profiles = ["default"]

        eval_root = ensure_dir((out_dir / "semantic_eval" / f"epoch_{epoch:03d}").resolve())
        abs_ckpt = epoch_ckpt.expanduser().resolve()
        if not abs_ckpt.exists():
            print(f"[train] WARNING: semantic eval ckpt does not exist: {abs_ckpt}")
            return None

        if strict_preflight:
            try:
                eval_root.mkdir(parents=True, exist_ok=True)
                touch = eval_root / ".preflight_write_test"
                touch.write_text("ok", encoding="utf-8")
                touch.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[train] WARNING: semantic eval preflight failed for output dir {eval_root}: {exc}")
                return None

        try:
            profile_results: Dict[str, Dict[str, Any]] = {}
            single_enabled_local = bool(single_seq_state.get("enabled", False))
            total_frames = int(single_seq_state.get("total_frames", 0)) if single_enabled_local else 0
            train_win = single_seq_state.get("train_window", {}) if single_enabled_local else {}
            holdout_win = single_seq_state.get("holdout_window", {}) if single_enabled_local else {}
            seq_short_single = str(single_seq_state.get("target_sequence_short", "")).strip()

            for profile in eval_profiles:
                profile_name = str(profile).strip().lower()
                profile_root = ensure_dir((eval_root / profile_name).resolve())
                csv_path = (profile_root / "metrics_summary.csv").resolve()

                env = dict(os.environ)
                env["CKPT"] = str(abs_ckpt)
                env["SEQUENCES"] = ",".join(sequences)
                env["USE_LOOP_CLOSING"] = str(ate_cfg.get("use_loop_closing", 1))
                disable_timeouts = str(os.environ.get("DISABLE_TIMEOUT", "0")).strip() == "1"
                run_timeout_seconds = int(ate_cfg.get("run_timeout_seconds", 1200) or 1200)
                if disable_timeouts:
                    run_timeout_seconds = 0
                env["RUN_TIMEOUT_SECONDS"] = str(run_timeout_seconds)
                env["STRICT_PREFLIGHT"] = "1" if strict_preflight else "0"
                env["MISSING_PENALTY_METERS"] = str(missing_penalty)
                env["MIN_COVERAGE_OK"] = str(min_coverage_ok)
                if "pyslam_use_cpp_core" in ate_cfg:
                    env["PYSLAM_USE_CPP_CORE"] = str(ate_cfg.get("pyslam_use_cpp_core"))
                match_ratio_test = ate_cfg.get("match_ratio_test", "")
                if str(match_ratio_test).strip() != "":
                    env["DINOSLAM3_MATCH_RATIO_TEST"] = str(match_ratio_test)
                max_desc_dist = ate_cfg.get("max_descriptor_distance", "")
                if str(max_desc_dist).strip() != "":
                    env["DINOSLAM3_MAX_DESC_DIST"] = str(max_desc_dist)
                env["RESULTS_DIR_OVERRIDE"] = str(profile_root)
                env["CSV_PATH_OVERRIDE"] = str(csv_path)

                assoc_override_map: Dict[str, str] = {}
                if single_enabled_local and seq_short_single and total_frames > 0 and profile_name in {"overfit", "holdout"}:
                    if profile_name == "overfit":
                        start_idx = int(train_win.get("start_idx", 0))
                        end_idx = int(train_win.get("end_idx", 0))
                    else:
                        start_idx = int(holdout_win.get("start_idx", 0))
                        end_idx = int(holdout_win.get("end_idx", 0))
                    assoc_out = (profile_root / "associations" / f"{seq_short_single}_{profile_name}.txt").resolve()
                    assoc_path = _build_association_window(
                        sequence_short=seq_short_single,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        out_path=assoc_out,
                    )
                    assoc_override_map[seq_short_single] = str(assoc_path)
                if assoc_override_map:
                    env["ASSOCIATIONS_OVERRIDE_JSON"] = json.dumps(assoc_override_map)

                proc = subprocess.run(
                    [str(script)],
                    cwd=str(Path.cwd()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=(
                        None
                        if disable_timeouts
                        else int(ate_cfg.get("wall_timeout_seconds", 7200) or 7200)
                    ),
                )
                if proc.returncode != 0:
                    print(f"[train] WARNING: semantic eval ({profile_name}) failed (rc={proc.returncode})")
                    tail = "\n".join(proc.stdout.splitlines()[-20:]) if proc.stdout else ""
                    if tail:
                        print(tail)
                    return None
                if not csv_path.exists():
                    print(f"[train] WARNING: semantic eval CSV missing ({profile_name}): {csv_path}")
                    return None

                scored = _load_scored_csv(
                    csv_path=csv_path,
                    sequences=sequences,
                    sequence_weights_cfg=sequence_weights_cfg,
                    missing_penalty=missing_penalty,
                    status_penalties_cfg=status_penalties_cfg,
                    min_coverage_ok=min_coverage_ok,
                    primary_ate_field=primary_ate_field,
                    robust_ate_field=robust_ate_field,
                )
                if strict_preflight and int(scored.get("total_count", 0)) <= 0:
                    print(f"[train] WARNING: semantic eval strict mode got empty score rows ({profile_name})")
                    return None
                scored["csv_path"] = str(csv_path)
                scored["checkpoint"] = str(abs_ckpt)
                scored["profile"] = profile_name
                scored["sequence_weights"] = {k: float(v) for k, v in sequence_weights_cfg.items()} if sequence_weights_cfg else {}
                profile_results[profile_name] = scored

            selected_profile = selection_target
            if selected_profile not in profile_results:
                if "holdout" in profile_results:
                    selected_profile = "holdout"
                elif "overfit" in profile_results:
                    selected_profile = "overfit"
                elif profile_results:
                    selected_profile = next(iter(profile_results.keys()))
                else:
                    return None
            selected = dict(profile_results[selected_profile])
            overfit_score = _safe_float(
                profile_results.get("overfit", {}).get("weighted_penalized_score"),
                default=float("nan"),
            )
            holdout_score = _safe_float(
                profile_results.get("holdout", {}).get("weighted_penalized_score"),
                default=float("nan"),
            )
            generalization_gap = float("nan")
            if math.isfinite(overfit_score) and math.isfinite(holdout_score):
                generalization_gap = holdout_score - overfit_score

            selected["profiles"] = profile_results
            selected["selected_profile"] = selected_profile
            selected["selection_target"] = selection_target
            selected["generalization_gap"] = generalization_gap
            return selected
        except Exception as exc:
            print(f"[train] WARNING: semantic eval exception: {exc}")
            return None

    for epoch in range(start_epoch, epochs + 1):
        lr = opt.param_groups[0]["lr"]
        print_epoch_header(epoch, epochs, lr)
        _set_dataset_epoch(train_ds, epoch=epoch, epochs=epochs)
        loss_cfg_epoch = _scheduled_loss_cfg(cfg, epoch=epoch, total_epochs=epochs)

        # train
        model.train()
        _enforce_train_modes(model, freeze_non_trainable_modules, freeze_bn_running_stats)
        train_m = {k: 0.0 for k in ["loss_total","loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity","valid_ratio","occlusion_ratio"]}
        n_train = 0

        pbar = tqdm(train_loader, desc="train", leave=False)
        for it, batch in enumerate(pbar, start=1):
            _enforce_train_modes(model, freeze_non_trainable_modules, freeze_bn_running_stats)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            if channels_last:
                batch["rgb1"] = batch["rgb1"].contiguous(memory_format=torch.channels_last)
                batch["rgb2"] = batch["rgb2"].contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            try:
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    out1 = model(batch["rgb1"], depth=batch.get("depth1"))
                    out2 = model(batch["rgb2"], depth=batch.get("depth2"))
            except RuntimeError as exc:
                if compile_active and "CUBLAS_STATUS_INVALID_VALUE" in str(exc):
                    print("[train] WARNING: torch.compile hit CUBLAS_STATUS_INVALID_VALUE; falling back to eager mode.")
                    model = eager_model
                    compile_active = False
                    with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                        out1 = model(batch["rgb1"], depth=batch.get("depth1"))
                        out2 = model(batch["rgb2"], depth=batch.get("depth2"))
                else:
                    raise

            loss_ctx = torch.autocast("cuda", enabled=False) if (device.type == "cuda" and loss_fp32) else nullcontext()
            with loss_ctx:
                losses, stats = compute_losses(batch, out1, out2, cfg=loss_cfg_epoch, epoch=epoch, stride=stride)
                loss = losses["loss_total"]

            if use_scaler:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()

            bs = batch["rgb1"].shape[0]
            n_train += bs
            train_m["loss_total"] += float(loss.detach().cpu()) * bs
            for k in ["loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity"]:
                train_m[k] += float(losses.get(k, torch.tensor(0.0)).detach().cpu()) * bs
            train_m["valid_ratio"] += float(stats.valid_ratio) * bs
            train_m["occlusion_ratio"] += float(stats.occlusion_ratio) * bs

            if it % int(cfg["training"].get("log_every", 50)) == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{train_m['loss_total']/max(n_train,1):.3f}",
                        "valid%": f"{100.0*(train_m['valid_ratio']/max(n_train,1)):.1f}",
                        "occ%": f"{100.0*(train_m['occlusion_ratio']/max(n_train,1)):.1f}",
                    }
                )

        for k in train_m:
            train_m[k] /= max(n_train, 1)

        print_metrics_table("Train", train_m)
        val_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        ate_result = None
        do_val = enable_validation and ((epoch % val_every == 0) or (epoch == epochs))
        if do_val:
            model.eval()
            pnp_px = float(loss_cfg_epoch.get("diag_pnp_px", 3.0))
            diag_keys = [
                "kpts1","kpts2","matches","valid_match_ratio",
                f"inliers@{inlier_px:.0f}px",
                f"inlier_rate@{inlier_px:.0f}px",
                f"pnp_inliers@{pnp_px:.0f}px",
                f"pnp_inlier_rate@{pnp_px:.0f}px",
                "mean_reproj_err","mean_reproj_err_inliers","median_reproj_err_inliers"
            ]

            with torch.no_grad():
                for split_name, vloader in val_loaders.items():
                    val_m = {k: 0.0 for k in train_m}
                    diag_acc = {k: 0.0 for k in diag_keys}
                    n_val = 0
                    diag_batches = 0

                    for batch in tqdm(vloader, desc=f"val:{split_name}", leave=False):
                        for k, v in batch.items():
                            if torch.is_tensor(v):
                                batch[k] = v.to(device, non_blocking=True)
                        if channels_last:
                            batch["rgb1"] = batch["rgb1"].contiguous(memory_format=torch.channels_last)
                            batch["rgb2"] = batch["rgb2"].contiguous(memory_format=torch.channels_last)

                        with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                            try:
                                out1 = model(batch["rgb1"], depth=batch.get("depth1"))
                                out2 = model(batch["rgb2"], depth=batch.get("depth2"))
                            except RuntimeError as exc:
                                if compile_active and "CUBLAS_STATUS_INVALID_VALUE" in str(exc):
                                    print("[val] WARNING: torch.compile hit CUBLAS_STATUS_INVALID_VALUE; falling back to eager mode.")
                                    model = eager_model
                                    compile_active = False
                                    out1 = model(batch["rgb1"], depth=batch.get("depth1"))
                                    out2 = model(batch["rgb2"], depth=batch.get("depth2"))
                                else:
                                    raise

                        loss_ctx = torch.autocast("cuda", enabled=False) if (device.type == "cuda" and loss_fp32) else nullcontext()
                        with loss_ctx:
                            losses, stats = compute_losses(batch, out1, out2, cfg=loss_cfg_epoch, epoch=epoch, stride=stride)

                        bs = batch["rgb1"].shape[0]
                        n_val += bs
                        val_m["loss_total"] += float(losses["loss_total"].detach().cpu()) * bs
                        for k in ["loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity"]:
                            val_m[k] += float(losses.get(k, torch.tensor(0.0)).detach().cpu()) * bs
                        val_m["valid_ratio"] += float(stats.valid_ratio) * bs
                        val_m["occlusion_ratio"] += float(stats.occlusion_ratio) * bs

                        if diag_batches < int(cfg["training"].get("diag_batches", 3)):
                            d = _val_diagnostics(cfg, batch, out1, out2)
                            for kk in diag_acc:
                                diag_acc[kk] += float(d.get(kk, 0.0))
                            diag_batches += 1

                    for k in val_m:
                        val_m[k] /= max(n_val, 1)
                    diag = {k: v / max(diag_batches, 1) for k, v in diag_acc.items()}
                    val_results[split_name] = {"metrics": val_m, "diag": diag}
                    print_metrics_table(f"Val ({split_name})", val_m)
                    if int(cfg["training"].get("diag_batches", 3)) > 0:
                        print_match_table(f"Val diagnostics ({split_name})", diag)

            if ate_eval_every > 0 and (epoch % ate_eval_every == 0):
                eval_ckpt_dir = ensure_dir(out_dir / "semantic_eval" / "checkpoints")
                tmp_ckpt = eval_ckpt_dir / f"epoch_{epoch:03d}.pt"
                model_to_eval = eager_model if compile_active else model
                torch.save(
                    {
                        "model": model_to_eval.state_dict(),
                        "config": cfg,
                        "single_sequence_state": single_seq_state,
                    },
                    tmp_ckpt,
                )
                ate_result = _run_ate_eval(tmp_ckpt, epoch=epoch)
                if ate_result is not None:
                    mean_ok = float(ate_result.get("mean_ok", float("inf")))
                    penalized = float(ate_result.get("penalized_mean", mean_ok))
                    weighted = float(ate_result.get("weighted_penalized_score", penalized))
                    weighted_robust = float(ate_result.get("weighted_penalized_score_robust", float("inf")))
                    ok_count = int(ate_result.get("ok_count", 0))
                    total_count = int(ate_result.get("total_count", 0))
                    selected_profile = str(ate_result.get("selected_profile", "default"))
                    gap = _safe_float(ate_result.get("generalization_gap"), default=float("nan"))
                    profile_scores = ate_result.get("profiles", {}) if isinstance(ate_result.get("profiles", {}), dict) else {}
                    overfit_w = _safe_float(profile_scores.get("overfit", {}).get("weighted_penalized_score"), default=float("nan"))
                    holdout_w = _safe_float(profile_scores.get("holdout", {}).get("weighted_penalized_score"), default=float("nan"))
                    print(
                        "[val] semantic ATE "
                        f"profile={selected_profile} "
                        f"mean_ok={mean_ok:.6f} "
                        f"penalized={penalized:.6f} "
                        f"weighted={weighted:.6f} "
                        f"weighted_robust={weighted_robust:.6f} "
                        f"ok={ok_count}/{total_count} "
                        f"overfit={overfit_w:.6f} holdout={holdout_w:.6f} gap={gap:.6f}"
                    )
        else:
            print(f"[train] skipped validation at epoch {epoch} (val_every={val_every})")

        sch.step()

        model_to_save = eager_model if compile_active else model

        # save
        if bool(tcfg.get("save_every_epoch", True)):
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_to_save.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "best_val_loss": float(best_val),
                    "config": cfg,
                    "single_sequence_state": single_seq_state,
                },
                ckpt_path,
            )
            print_save_notice(str(ckpt_path), "epoch")
            if exp_store is not None:
                try:
                    exp_store.log_checkpoint(
                        experiment_id=experiment_id,
                        stage=stage_name,
                        epoch=epoch,
                        path=ckpt_path,
                        selected_flag=False,
                        selection_score=None,
                    )
                except Exception as exc:
                    print(f"[train] WARNING: failed to log epoch checkpoint to experiment store: {exc}")

        if val_results:
            short_name = "short" if "short" in val_results else next(iter(val_results.keys()))
            hard_name = "hard" if "hard" in val_results else short_name
            short_metrics = val_results[short_name]["metrics"]
            hard_diag = val_results[hard_name]["diag"]

            def _lookup_metric(name: str, default: float = float("nan")) -> float:
                n = str(name).strip()
                if not n:
                    return float(default)
                lower = n.lower()
                split = short_name
                key = n
                if lower.startswith("val_short_"):
                    split = short_name
                    key = n[len("val_short_") :]
                elif lower.startswith("val_hard_"):
                    split = hard_name
                    key = n[len("val_hard_") :]
                elif lower.startswith("short_"):
                    split = short_name
                    key = n[len("short_") :]
                elif lower.startswith("hard_"):
                    split = hard_name
                    key = n[len("hard_") :]

                pool = val_results.get(split, {})
                metrics = pool.get("metrics", {})
                diag = pool.get("diag", {})
                if key in metrics:
                    return float(metrics[key])
                if key in diag:
                    return float(diag[key])
                return float(default)

            guard_value = _lookup_metric(
                secondary_guard_metric,
                default=float(hard_diag.get(inlier_key, 0.0)),
            )
            prev_best_guard = best_guard_value
            if math.isfinite(guard_value):
                if guard_higher_better:
                    best_guard_value = max(best_guard_value, guard_value)
                else:
                    best_guard_value = min(best_guard_value, guard_value)

            hard_gate_ok = True
            if math.isfinite(guard_value):
                if guard_higher_better and math.isfinite(prev_best_guard):
                    hard_gate_ok = guard_value >= (1.0 - hard_regression_max) * prev_best_guard
                elif (not guard_higher_better) and math.isfinite(prev_best_guard):
                    hard_gate_ok = guard_value <= (1.0 + hard_regression_max) * prev_best_guard

            if rel_only:
                rel_score = float(short_metrics.get("loss_rel", float("inf")))
                if rel_score < best_rel_score:
                    best_rel_score = rel_score
                    payload = {
                        "epoch": epoch,
                        "model": model_to_save.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sch.state_dict(),
                        "best_val_loss": float(best_val),
                        "best_rel_loss": float(best_rel_score),
                        "config": cfg,
                        "single_sequence_state": single_seq_state,
                    }
                    rel_path = ckpt_dir / "rel_best.pt"
                    best_path = ckpt_dir / "best.pt"
                    torch.save(payload, rel_path)
                    torch.save(payload, best_path)
                    print_save_notice(str(rel_path), f"new best val_loss_rel={best_rel_score:.6f}")
                    print_save_notice(str(best_path), f"new best val_loss_rel={best_rel_score:.6f}")
                    _append_run_registry(
                        cfg=cfg,
                        checkpoint_path=rel_path,
                        selection_metric_used="val_loss_rel",
                        selection_value=float(best_rel_score),
                        ate_short_mean=float("nan"),
                    )
                    if exp_store is not None:
                        try:
                            exp_store.log_checkpoint(
                                experiment_id=experiment_id,
                                stage=stage_name,
                                epoch=epoch,
                                path=rel_path,
                                selected_flag=True,
                                selection_score=float(best_rel_score),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="val_loss_rel_best",
                                metric_value=float(best_rel_score),
                            )
                            exp_store.log_artifact(
                                experiment_id=experiment_id,
                                artifact_type="rel_best_checkpoint",
                                path=rel_path,
                            )
                        except Exception as exc:
                            print(f"[train] WARNING: failed to log rel-only checkpoint to experiment store: {exc}")
                continue

            candidate_name = selection_metric
            candidate = float("inf")
            ate_profiles = ate_result.get("profiles", {}) if (ate_result is not None and isinstance(ate_result.get("profiles", {}), dict)) else {}
            selected_profile_name = str(ate_result.get("selected_profile", "default")) if ate_result is not None else "default"
            selected_profile_result = ate_profiles.get(selected_profile_name, ate_result) if ate_result is not None else None
            holdout_profile_result = ate_profiles.get("holdout")
            overfit_profile_result = ate_profiles.get("overfit")

            if selection_metric in {
                "ate_short_mean",
                "ate_short_mean_penalized",
                "weighted_penalized_ate",
                "holdout_weighted_penalized_ate",
                "overfit_weighted_penalized_ate",
                "weighted_mean_ate",
                "holdout_weighted_mean_ate",
                "overfit_weighted_mean_ate",
                "weighted_mean_ate_robust",
                "holdout_weighted_mean_ate_robust",
                "overfit_weighted_mean_ate_robust",
            }:
                if ate_result is not None:
                    candidate, candidate_name = _selection_candidate_from_semantic_result(selection_metric, ate_result)
                elif allow_selection_fallback:
                    candidate = float(short_metrics.get("loss_total", float("inf")))
                    candidate_name = "val_short_loss_total(fallback)"
            elif selection_metric == "val_short_loss_total":
                candidate = float(short_metrics.get("loss_total", float("inf")))
            elif selection_metric.startswith("val_") or selection_metric.startswith("short_") or selection_metric.startswith("hard_"):
                candidate = _lookup_metric(selection_metric, default=float("inf"))
            else:
                candidate = float(short_metrics.get("loss_total", float("inf")))

            coverage_gate_ok = True
            if ate_result is not None and bool(ate_cfg.get("enforce_coverage_gate", True)):
                gate_profile = holdout_profile_result or selected_profile_result or ate_result
                coverage_gate_ok = bool((gate_profile or {}).get("coverage_ok", True))
            robust_gate_ok = True
            if ate_result is not None and bool(ate_cfg.get("enforce_robustness_gate", False)):
                gate_profile = holdout_profile_result or selected_profile_result or ate_result
                robust_score = float((gate_profile or {}).get("weighted_penalized_score_robust", float("inf")))
                robust_gate_max = float(ate_cfg.get("robustness_gate_max", float("inf")))
                robust_gate_ok = bool(math.isfinite(robust_score) and robust_score <= robust_gate_max)

            if ate_result is not None:
                try:
                    snapshot_payload = {
                        "experiment_id": experiment_id,
                        "run_name": stage_name,
                        "selection_metric": selection_metric,
                        "selection_metric_used": candidate_name,
                        "candidate": float(candidate) if math.isfinite(candidate) else float("inf"),
                        "hard_gate_ok": bool(hard_gate_ok),
                        "coverage_gate_ok": bool(coverage_gate_ok),
                        "robust_gate_ok": bool(robust_gate_ok),
                        "ate_eval": ate_result,
                    }
                    snap_path = write_semantic_selection_snapshot(
                        out_dir=out_dir,
                        epoch=epoch,
                        payload=snapshot_payload,
                    )
                    if exp_store is not None:
                        exp_store.log_artifact(
                            experiment_id=experiment_id,
                            artifact_type="semantic_selection_snapshot",
                            path=snap_path,
                        )
                        if ate_profiles:
                            for profile_name, profile_data in ate_profiles.items():
                                exp_store.log_sequence_metrics(
                                    experiment_id=experiment_id,
                                    stage=f"{stage_name}:semantic_eval_epoch_{int(epoch):03d}:{profile_name}",
                                    rows=list(profile_data.get("rows", [])),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name=f"semantic_eval_epoch_{int(epoch):03d}_{profile_name}_weighted_mean_ate",
                                    metric_value=float(profile_data.get("weighted_mean_ok", float("nan"))),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name=f"semantic_eval_epoch_{int(epoch):03d}_{profile_name}_weighted_penalized_ate",
                                    metric_value=float(profile_data.get("weighted_penalized_score", float("nan"))),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name=f"semantic_eval_epoch_{int(epoch):03d}_{profile_name}_weighted_mean_ate_robust",
                                    metric_value=float(profile_data.get("weighted_mean_ok_robust", float("nan"))),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name=f"semantic_eval_epoch_{int(epoch):03d}_{profile_name}_weighted_penalized_ate_robust",
                                    metric_value=float(profile_data.get("weighted_penalized_score_robust", float("nan"))),
                                )
                                for st, cnt in dict(profile_data.get("statuses", {})).items():
                                    exp_store.log_aggregate(
                                        experiment_id=experiment_id,
                                        metric_name=f"semantic_eval_epoch_{int(epoch):03d}_{profile_name}_status_count_{st}",
                                        metric_value=float(cnt),
                                    )
                        else:
                            exp_store.log_sequence_metrics(
                                experiment_id=experiment_id,
                                stage=f"{stage_name}:semantic_eval_epoch_{int(epoch):03d}",
                                rows=list(ate_result.get("rows", [])),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name=f"semantic_eval_epoch_{int(epoch):03d}_weighted_mean_ate",
                                metric_value=float(ate_result.get("weighted_mean_ok", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name=f"semantic_eval_epoch_{int(epoch):03d}_weighted_penalized_ate",
                                metric_value=float(ate_result.get("weighted_penalized_score", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name=f"semantic_eval_epoch_{int(epoch):03d}_weighted_mean_ate_robust",
                                metric_value=float(ate_result.get("weighted_mean_ok_robust", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name=f"semantic_eval_epoch_{int(epoch):03d}_weighted_penalized_ate_robust",
                                metric_value=float(ate_result.get("weighted_penalized_score_robust", float("nan"))),
                            )
                            for st, cnt in dict(ate_result.get("statuses", {})).items():
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name=f"semantic_eval_epoch_{int(epoch):03d}_status_count_{st}",
                                    metric_value=float(cnt),
                                )
                        exp_store.log_aggregate(
                            experiment_id=experiment_id,
                            metric_name=f"semantic_eval_epoch_{int(epoch):03d}_generalization_gap",
                            metric_value=float(ate_result.get("generalization_gap", float("nan"))),
                        )
                except Exception as exc:
                    print(f"[train] WARNING: failed to write semantic selection snapshot: {exc}")

            improved = bool(
                math.isfinite(candidate)
                and (candidate + float(early_stop_min_delta) < best_selection)
                and hard_gate_ok
                and coverage_gate_ok
                and robust_gate_ok
            )
            if improved:
                best_selection = candidate
                best_selection_epoch = int(epoch)
                best_val = float(short_metrics.get("loss_total", best_val))
                selection_used = candidate_name
                ate_short = float("nan")
                if ate_result is not None:
                    ate_short = float((selected_profile_result or ate_result).get("penalized_mean", (selected_profile_result or ate_result).get("mean_ok", float("nan"))))
                payload = {
                    "epoch": epoch,
                    "model": model_to_save.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "best_val_loss": float(best_val),
                    "best_selection": float(best_selection),
                    "best_selection_epoch": int(best_selection_epoch),
                    "config": cfg,
                    "selection_metric": selection_metric,
                    "selection_metric_used": selection_used,
                    "secondary_guard_metric": secondary_guard_metric,
                    "secondary_guard_value": float(guard_value),
                    "ate_eval": ate_result,
                    "single_sequence_state": single_seq_state,
                }
                best_path = ckpt_dir / "best.pt"
                geom_best_path = ckpt_dir / "geom_best.pt"
                torch.save(payload, best_path)
                torch.save(payload, geom_best_path)
                print_save_notice(str(best_path), f"new best {selection_used}={best_selection:.6f}")
                print_save_notice(str(geom_best_path), f"new best {selection_used}={best_selection:.6f}")
                _append_run_registry(
                    cfg=cfg,
                    checkpoint_path=geom_best_path,
                    selection_metric_used=selection_used,
                    selection_value=float(best_selection),
                    ate_short_mean=float(ate_short),
                )
                if exp_store is not None:
                    try:
                        exp_store.log_checkpoint(
                            experiment_id=experiment_id,
                            stage=stage_name,
                            epoch=epoch,
                            path=geom_best_path,
                            selected_flag=True,
                            selection_score=float(best_selection),
                        )
                        exp_store.log_aggregate(
                            experiment_id=experiment_id,
                            metric_name="best_selection_score",
                            metric_value=float(best_selection),
                        )
                        exp_store.log_artifact(
                            experiment_id=experiment_id,
                            artifact_type="geom_best_checkpoint",
                            path=geom_best_path,
                        )
                        if ate_result is not None:
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="best_weighted_mean_ate",
                                metric_value=float(ate_result.get("weighted_mean_ok", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="best_weighted_penalized_ate",
                                metric_value=float(ate_result.get("weighted_penalized_score", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="best_weighted_mean_ate_robust",
                                metric_value=float(ate_result.get("weighted_mean_ok_robust", float("nan"))),
                            )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="best_weighted_penalized_ate_robust",
                                metric_value=float(ate_result.get("weighted_penalized_score_robust", float("nan"))),
                            )
                            if overfit_profile_result is not None:
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name="semantic_overfit_weighted_mean_ate",
                                    metric_value=float(overfit_profile_result.get("weighted_mean_ok", float("nan"))),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name="semantic_overfit_weighted_penalized_ate",
                                    metric_value=float(overfit_profile_result.get("weighted_penalized_score", float("nan"))),
                                )
                            if holdout_profile_result is not None:
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name="semantic_holdout_weighted_mean_ate",
                                    metric_value=float(holdout_profile_result.get("weighted_mean_ok", float("nan"))),
                                )
                                exp_store.log_aggregate(
                                    experiment_id=experiment_id,
                                    metric_name="semantic_holdout_weighted_penalized_ate",
                                    metric_value=float(holdout_profile_result.get("weighted_penalized_score", float("nan"))),
                                )
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name="semantic_generalization_gap",
                                metric_value=float(ate_result.get("generalization_gap", float("nan"))),
                            )
                    except Exception as exc:
                        print(f"[train] WARNING: failed to log selected checkpoint to experiment store: {exc}")

            if early_stop_enabled and do_val and epoch >= early_stop_min_epochs:
                epochs_since_best = int(epoch - best_selection_epoch) if best_selection_epoch > 0 else int(epoch)
                if epochs_since_best >= early_stop_patience:
                    print(
                        f"[train] early stopping at epoch {epoch} "
                        f"(best_selection_epoch={best_selection_epoch}, "
                        f"patience={early_stop_patience}, metric={selection_metric})"
                    )
                    stop_training = True

        if stop_training:
            break
