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
import hashlib
import uuid
from datetime import datetime, timezone
from copy import deepcopy

import numpy as np

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
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
                    "coverage": float("nan"),
                }
            )
            continue

        status = str(row.get("status", "")).strip().lower()
        statuses[status] = statuses.get(status, 0) + 1
        ate = _safe_float(row.get("ate_rmse"), default=float("nan"))
        cov = _safe_float(row.get("coverage"), default=float("nan"))
        if math.isfinite(cov):
            weighted_cov_num += w * cov
            weighted_cov_den += w

        if status == "ok" and math.isfinite(ate):
            weighted_ok_num += w * ate
            weighted_ok_den += w
        else:
            weighted_missing += w
            weighted_status_pen += w * _safe_float(status_penalties_cfg.get(status, 0.0), default=0.0)

        per_sequence.append(
            {
                "sequence": seq,
                "status": status,
                "weight": w,
                "ate_rmse": ate,
                "coverage": cov,
            }
        )

    weighted_mean_ok = float("inf")
    if weighted_ok_den > 0:
        weighted_mean_ok = weighted_ok_num / weighted_ok_den
    weighted_missing_ratio = weighted_missing / wsum
    weighted_status_penalty = weighted_status_pen / wsum
    weighted_penalized_score = weighted_mean_ok
    if math.isfinite(weighted_penalized_score):
        weighted_penalized_score += float(missing_penalty) * float(weighted_missing_ratio)
    else:
        weighted_penalized_score = float(missing_penalty) * float(weighted_missing_ratio)
    weighted_penalized_score += float(weighted_status_penalty)
    weighted_coverage_mean = float("nan")
    if weighted_cov_den > 0:
        weighted_coverage_mean = weighted_cov_num / weighted_cov_den
    coverage_ok = bool(math.isfinite(weighted_coverage_mean) and weighted_coverage_mean >= float(min_coverage_ok))

    ok_vals = [x["ate_rmse"] for x in per_sequence if x.get("status") == "ok" and math.isfinite(_safe_float(x.get("ate_rmse")))]
    mean_ok = float(sum(ok_vals) / len(ok_vals)) if ok_vals else float("inf")
    penalized = mean_ok
    if weighted_missing > 0:
        if math.isfinite(penalized):
            penalized += float(missing_penalty) * float(weighted_missing_ratio)
        else:
            penalized = float(missing_penalty) * float(weighted_missing_ratio)
    for st, cnt in statuses.items():
        if st == "ok":
            continue
        p = _safe_float(status_penalties_cfg.get(st, 0.0), default=0.0)
        if p > 0 and len(sequences) > 0:
            penalized += p * (float(cnt) / float(len(sequences)))

    return {
        "mean_ok": float(mean_ok),
        "penalized_mean": float(penalized),
        "weighted_mean_ok": float(weighted_mean_ok),
        "weighted_penalized_score": float(weighted_penalized_score),
        "weighted_missing_ratio": float(weighted_missing_ratio),
        "weighted_status_penalty": float(weighted_status_penalty),
        "weighted_coverage_mean": float(weighted_coverage_mean),
        "coverage_ok": bool(coverage_ok),
        "ok_count": int(sum(1 for x in per_sequence if x.get("status") == "ok" and math.isfinite(_safe_float(x.get("ate_rmse"))))),
        "total_count": int(len(sequences)),
        "statuses": statuses,
        "rows": per_sequence,
    }

def _make_loaders(
    cfg: Dict[str, Any],
    epochs: int,
    seed: int,
) -> Tuple[DataLoader, Dict[str, DataLoader], torch.utils.data.Dataset]:
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

    def _repeat_count(seq_name: str) -> int:
        try:
            return max(1, int(sequence_repeat.get(seq_name, 1)))
        except Exception:
            return 1

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
                ds = TUMRGBDDataset(
                    dataset_root=dataset_root,
                    sequence=s,
                    frame_spacing_min=int(spacing_min),
                    frame_spacing_max=int(spacing_max),
                    max_frames=dcfg.get("max_frames"),
                    pad_to=int(dcfg.get("pad_to", 16)),
                    is_train=is_train,
                    augmentation=dcfg.get("augmentation"),
                    max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
                    max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
                    cache_in_memory=bool(dcfg.get("cache_in_memory", False)),
                    cache_to_disk=bool(dcfg.get("cache_to_disk", False)),
                    cache_dir=dcfg.get("cache_dir"),
                    pair_sampler=dcfg.get("pair_sampler") if is_train else None,
                    total_epochs=int(epochs),
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

    train_ds_list: List[torch.utils.data.Dataset] = []
    if dataset_mode in {"tum", "mixed"}:
        train_seqs = list(dcfg.get("train_sequences", []))
        train_ds_list.extend(
            _build_tum_dataset(
                seqs=train_seqs,
                is_train=True,
                spacing_min=int(dcfg.get("frame_spacing_min", 1)),
                spacing_max=int(dcfg.get("frame_spacing_max", 4)),
            )
        )
    if dataset_mode in {"manifest", "mixed"}:
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
            seqs = list(scfg.get("sequences", dcfg.get("val_sequences", [])))
            ds_list.extend(
                _build_tum_dataset(
                    seqs=seqs,
                    is_train=False,
                    spacing_min=int(scfg.get("frame_spacing_min", dcfg.get("frame_spacing_min", 1))),
                    spacing_max=int(scfg.get("frame_spacing_max", dcfg.get("frame_spacing_max", 4))),
                )
            )
        if dataset_mode in {"manifest", "mixed"}:
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
    pair_sched = dcfg.get("pair_sampler", {}).get("schedule")
    persistent = (num_workers > 0) and not isinstance(pair_sched, dict)

    def _worker_init_fn(worker_id: int) -> None:
        wseed = int(seed) + int(worker_id) * 97 + 13
        random.seed(wseed)
        np.random.seed(wseed % (2**32 - 1))
        torch.manual_seed(wseed)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seed))
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
        shuffle=True,
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
    return train_loader, val_loaders, train_ds

def _build_model(cfg: Dict[str, Any]) -> LocalFeatureNet:
    m = cfg["model"]
    return LocalFeatureNet(
        dinov3_name=m["dinov3"]["name_or_path"],
        patch_size=int(m.get("patch_size", 16)),
        descriptor_dim=int(m["heads"].get("descriptor_dim", 256)),
        fine_channels=int(m["fine_cnn"].get("channels", 96)),
        fine_blocks=int(m["fine_cnn"].get("num_blocks", 8)),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
        use_offset=bool(m["heads"].get("offset", {}).get("enabled", True)),
        use_reliability=bool(m["heads"].get("reliability", {}).get("enabled", True)),
        dinov3_dtype=str(m["dinov3"].get("dtype", "bf16")),
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
    train_loader, val_loaders, train_ds = _make_loaders(cfg, epochs=epochs, seed=seed)
    model = _build_model(cfg).to(device)
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
    stage_name = str(run_cfg.get("name", "unknown"))
    if selection_metric in {"ate_short_mean", "ate_short_mean_penalized", "weighted_penalized_ate"} and ate_eval_every <= 0:
        print("[train] WARNING: selection metric requires validation.ate_eval_every>0; falling back to val_short_loss_total.")
        selection_metric = "val_short_loss_total"

    def _run_ate_eval(epoch_ckpt: Path, epoch: int) -> Dict[str, Any] | None:
        if ate_eval_every <= 0:
            return None
        strict_preflight = bool(ate_cfg.get("strict_preflight", False))
        selection_mode = str(ate_cfg.get("selection_mode", "weighted_penalized_ate")).strip().lower()
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

        eval_root = ensure_dir((out_dir / "semantic_eval" / f"epoch_{epoch:03d}").resolve())
        csv_path = (eval_root / "metrics_summary.csv").resolve()
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

        env = dict(os.environ)
        env["CKPT"] = str(abs_ckpt)
        env["SEQUENCES"] = ",".join(sequences)
        env["USE_LOOP_CLOSING"] = str(ate_cfg.get("use_loop_closing", 1))
        env["RUN_TIMEOUT_SECONDS"] = str(ate_cfg.get("run_timeout_seconds", 1200))
        env["STRICT_PREFLIGHT"] = "1" if strict_preflight else "0"
        env["MISSING_PENALTY_METERS"] = str(missing_penalty)
        env["MIN_COVERAGE_OK"] = str(min_coverage_ok)
        env["PYSLAM_USE_CPP_CORE"] = str(ate_cfg.get("pyslam_use_cpp_core", 0))
        env["RESULTS_DIR_OVERRIDE"] = str(eval_root)
        env["CSV_PATH_OVERRIDE"] = str(csv_path)
        try:
            proc = subprocess.run(
                [str(script)],
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(ate_cfg.get("wall_timeout_seconds", 7200)),
            )
            if proc.returncode != 0:
                print(f"[train] WARNING: semantic eval failed (rc={proc.returncode})")
                tail = "\n".join(proc.stdout.splitlines()[-20:]) if proc.stdout else ""
                if tail:
                    print(tail)
                return None
            if not csv_path.exists():
                print(f"[train] WARNING: semantic eval CSV missing: {csv_path}")
                return None

            rows_raw: List[Dict[str, Any]] = []
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows_raw.append(
                        {
                            "sequence": str(r.get("sequence", "")).strip(),
                            "status": str(r.get("status", "")).strip().lower(),
                            "ate_rmse": _safe_float(r.get("ate_rmse"), default=float("nan")),
                            "rpe_trans_rmse": _safe_float(r.get("rpe_trans_rmse"), default=float("nan")),
                            "rpe_rot_rmse": _safe_float(r.get("rpe_rot_rmse"), default=float("nan")),
                            "coverage": _safe_float(r.get("coverage"), default=float("nan")),
                        }
                    )
            if strict_preflight and len(rows_raw) == 0:
                print(f"[train] WARNING: semantic eval preflight strict mode failed: empty CSV {csv_path}")
                return None

            scored = _score_semantic_eval(
                rows=rows_raw,
                sequences=sequences,
                sequence_weights_cfg=sequence_weights_cfg,
                missing_penalty=missing_penalty,
                status_penalties_cfg=status_penalties_cfg,
                min_coverage_ok=min_coverage_ok,
            )
            scored["csv_path"] = str(csv_path)
            scored["checkpoint"] = str(abs_ckpt)
            scored["selection_mode"] = selection_mode
            scored["sequence_weights"] = {k: float(v) for k, v in sequence_weights_cfg.items()} if sequence_weights_cfg else {}
            return scored
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
                    out1 = model(batch["rgb1"])
                    out2 = model(batch["rgb2"])
            except RuntimeError as exc:
                if compile_active and "CUBLAS_STATUS_INVALID_VALUE" in str(exc):
                    print("[train] WARNING: torch.compile hit CUBLAS_STATUS_INVALID_VALUE; falling back to eager mode.")
                    model = eager_model
                    compile_active = False
                    with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                        out1 = model(batch["rgb1"])
                        out2 = model(batch["rgb2"])
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
                                out1 = model(batch["rgb1"])
                                out2 = model(batch["rgb2"])
                            except RuntimeError as exc:
                                if compile_active and "CUBLAS_STATUS_INVALID_VALUE" in str(exc):
                                    print("[val] WARNING: torch.compile hit CUBLAS_STATUS_INVALID_VALUE; falling back to eager mode.")
                                    model = eager_model
                                    compile_active = False
                                    out1 = model(batch["rgb1"])
                                    out2 = model(batch["rgb2"])
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
                torch.save({"model": model_to_eval.state_dict(), "config": cfg}, tmp_ckpt)
                ate_result = _run_ate_eval(tmp_ckpt, epoch=epoch)
                if ate_result is not None:
                    mean_ok = float(ate_result.get("mean_ok", float("inf")))
                    penalized = float(ate_result.get("penalized_mean", mean_ok))
                    weighted = float(ate_result.get("weighted_penalized_score", penalized))
                    ok_count = int(ate_result.get("ok_count", 0))
                    total_count = int(ate_result.get("total_count", 0))
                    print(
                        "[val] semantic ATE "
                        f"mean_ok={mean_ok:.6f} "
                        f"penalized={penalized:.6f} "
                        f"weighted={weighted:.6f} "
                        f"ok={ok_count}/{total_count}"
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
            if selection_metric in {"ate_short_mean", "ate_short_mean_penalized", "weighted_penalized_ate"}:
                if ate_result is not None:
                    if selection_metric == "ate_short_mean":
                        candidate = float(ate_result.get("mean_ok", float("inf")))
                    elif selection_metric == "weighted_penalized_ate":
                        candidate = float(ate_result.get("weighted_penalized_score", float("inf")))
                    else:
                        candidate = float(ate_result.get("penalized_mean", float("inf")))
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
                coverage_gate_ok = bool(ate_result.get("coverage_ok", True))

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
                        exp_store.log_sequence_metrics(
                            experiment_id=experiment_id,
                            stage=f"{stage_name}:semantic_eval_epoch_{int(epoch):03d}",
                            rows=list(ate_result.get("rows", [])),
                        )
                        exp_store.log_aggregate(
                            experiment_id=experiment_id,
                            metric_name=f"semantic_eval_epoch_{int(epoch):03d}_weighted_penalized_ate",
                            metric_value=float(ate_result.get("weighted_penalized_score", float("nan"))),
                        )
                        for st, cnt in dict(ate_result.get("statuses", {})).items():
                            exp_store.log_aggregate(
                                experiment_id=experiment_id,
                                metric_name=f"semantic_eval_epoch_{int(epoch):03d}_status_count_{st}",
                                metric_value=float(cnt),
                            )
                except Exception as exc:
                    print(f"[train] WARNING: failed to write semantic selection snapshot: {exc}")

            if math.isfinite(candidate) and candidate < best_selection and hard_gate_ok and coverage_gate_ok:
                best_selection = candidate
                best_val = float(short_metrics.get("loss_total", best_val))
                selection_used = candidate_name
                ate_short = float("nan")
                if ate_result is not None:
                    ate_short = float(ate_result.get("penalized_mean", ate_result.get("mean_ok", float("nan"))))
                payload = {
                    "epoch": epoch,
                    "model": model_to_save.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "best_val_loss": float(best_val),
                    "best_selection": float(best_selection),
                    "config": cfg,
                    "selection_metric": selection_metric,
                    "selection_metric_used": selection_used,
                    "secondary_guard_metric": secondary_guard_metric,
                    "secondary_guard_value": float(guard_value),
                    "ate_eval": ate_result,
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
                                metric_name="best_weighted_penalized_ate",
                                metric_value=float(ate_result.get("weighted_penalized_score", float("nan"))),
                            )
                    except Exception as exc:
                        print(f"[train] WARNING: failed to log selected checkpoint to experiment store: {exc}")
