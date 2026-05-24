from __future__ import annotations

import argparse
import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from refocus_vo.data import TartanAirWindowDataset
from refocus_vo.dino_dpvo import (
    DinoDPVOConfig,
    DinoProposalFrontend,
    dense_gradient_offset_targets,
    load_dino_dpvo_config,
)
from refocus_vo.dino_dpvo.diagnostics import init_diagnostics_outputs
from refocus_vo.dino_dpvo.frontend import build_dino_dpvo_frontend, load_matching_state_dict
from refocus_vo.eval.external_dpvo import evaluate_sequence as evaluate_dpvo_tum_sequence
from refocus_vo.eval.external_dpvo_euroc import evaluate_sequence as evaluate_dpvo_euroc_sequence
from refocus_vo.eval.external_dpvo_kitti import evaluate_sequence as evaluate_dpvo_kitti_sequence
from refocus_vo.eval.external_dpvo_tartanair import _select_eval_sequences, evaluate_sequence as evaluate_dpvo_tartanair_sequence
from refocus_vo.patchgraph.supervision import build_teacher_scores_from_fused, project_patch_centers, relative_pose_matrix


METRICS_HEADER = [
    "step",
    "split",
    "loss",
    "selector_bce",
    "staticness_bce",
    "offset_l1",
    "descriptor_reg",
    "coverage_kl",
    "proposal_repulsion",
    "gram_anchor_loss",
    "mean_teacher_score",
    "mean_track_survival",
    "mean_quality",
    "external_mean_ate",
    "external_mean_ate_associated",
    "external_mean_coverage",
    "mean_unique_semantic_count_before_repeat",
    "repeated_patch_fraction",
    "mean_dedupe_radius_used",
    "mean_grid_occupancy",
    "mean_track_age",
    "survival_rate_1",
    "survival_rate_3",
    "survival_rate_5",
    "unstable_motion_proxy_fraction",
    "dino_patch_fraction",
    "pure100_mean_ate",
    "pure100_mean_ate_associated",
    "pure100_mean_coverage",
    "lowtex_mean_ate",
    "lowtex_mean_ate_associated",
    "lowtex_mean_coverage",
    "hybrid_mean_ate",
    "hybrid_mean_ate_associated",
    "hybrid_mean_coverage",
    "hybrid_lowtex_mean_ate",
    "hybrid_lowtex_mean_ate_associated",
    "hybrid_lowtex_mean_coverage",
    "best_mode",
    "best_pure_assoc",
    "best_hybrid_assoc",
    "best_pure_lowtex_assoc",
    "best_hybrid_lowtex_assoc",
    "selection_metric",
    "selection_passed_gate",
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


def _seed_data_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(
    dataset: TartanAirWindowDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    seed: int,
    legacy_repro: bool = False,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if not legacy_repro:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["worker_init_fn"] = _seed_data_worker
        kwargs["generator"] = generator
    return DataLoader(**kwargs)


def _set_reproducibility(seed: int, *, deterministic: bool, legacy_repro: bool = False) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not legacy_repro:
        random.seed(seed)
        np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _capture_rng_state() -> dict[str, object]:
    payload: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return payload


def _fmt_metric(value) -> str:
    try:
        num = float(value)
    except Exception:
        return "" if value is None else str(value)
    return "NaN" if not math.isfinite(num) else f"{num:.6f}"


def _ratio(value: float, reference: float, *, eps: float = 1e-6) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(reference)):
        return math.inf
    return float(value) / max(abs(float(reference)), float(eps))


def _mean(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(sum(usable) / len(usable))


def _weighted_ratio_score(
    *,
    dataset_ids: list[str],
    weights: dict[str, float],
    value_lookup,
    reference_lookup,
) -> float:
    total = 0.0
    used = False
    for dataset_id in dataset_ids:
        weight = float(weights.get(dataset_id, 0.0))
        if weight <= 0.0:
            continue
        total += weight * _ratio(float(value_lookup(dataset_id)), float(reference_lookup(dataset_id)))
        used = True
    return float(total) if used else math.nan


def _weighted_component_score(
    *,
    item_ids: list[str],
    weights: dict[str, float],
    value_lookup,
) -> float:
    total = 0.0
    used = False
    for item_id in item_ids:
        weight = float(weights.get(item_id, 0.0))
        if weight <= 0.0:
            continue
        value = float(value_lookup(item_id))
        if not math.isfinite(value):
            return math.inf
        total += weight * value
        used = True
    return float(total) if used else 0.0


def _sequence_assoc_baseline_map(raw: object, *, dataset_id: str) -> dict[str, float]:
    payload = raw if isinstance(raw, dict) else {}
    dataset_payload = payload.get(dataset_id, {}) if isinstance(payload, dict) else {}
    if not isinstance(dataset_payload, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in dataset_payload.items():
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            output[str(key).strip()] = number
    return output


def _wins_against_sequence_baseline(
    rows: list[dict[str, object]],
    *,
    baseline_assoc: dict[str, float],
) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        assoc = float(row.get("ate_rmse_associated", math.nan))
        baseline = float(baseline_assoc.get(sequence, math.nan))
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


def _compute_sampled_similarity_loss(
    student_fused: torch.Tensor,
    teacher_fused: torch.Tensor,
    *,
    sample_tokens: int,
    downsample_stride: int,
) -> torch.Tensor:
    if student_fused.shape != teacher_fused.shape:
        raise ValueError(
            f"Student/teacher fused feature shapes must match, got "
            f"{tuple(student_fused.shape)} vs {tuple(teacher_fused.shape)}"
        )
    stride = max(1, int(downsample_stride))
    student_tokens = student_fused[:, :, :, ::stride, ::stride]
    teacher_tokens = teacher_fused[:, :, :, ::stride, ::stride]

    student_tokens = student_tokens.permute(0, 1, 3, 4, 2).reshape(-1, student_tokens.shape[2])
    teacher_tokens = teacher_tokens.permute(0, 1, 3, 4, 2).reshape(-1, teacher_tokens.shape[2]).detach()
    token_count = int(student_tokens.shape[0])
    if token_count <= 1:
        return student_fused.new_zeros(())

    target_count = max(2, int(sample_tokens))
    if token_count > target_count:
        indices = torch.randperm(token_count, device=student_fused.device)[:target_count]
        student_tokens = student_tokens[indices]
        teacher_tokens = teacher_tokens[indices]

    student_tokens = F.normalize(student_tokens.float(), dim=-1, eps=1e-6)
    teacher_tokens = F.normalize(teacher_tokens.float(), dim=-1, eps=1e-6)
    student_sim = student_tokens @ student_tokens.transpose(0, 1)
    teacher_sim = teacher_tokens @ teacher_tokens.transpose(0, 1)
    return F.mse_loss(student_sim, teacher_sim)


def _gram_anchor_teacher_fused(
    teacher: DinoProposalFrontend,
    images: torch.Tensor,
) -> torch.Tensor:
    encoded = teacher._encode_backbone(images)
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def _append_metrics_row(path: Path, step: int, split: str, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(step),
                split,
                _fmt_metric(metrics.get("loss", math.nan)),
                _fmt_metric(metrics.get("selector_bce", math.nan)),
                _fmt_metric(metrics.get("staticness_bce", math.nan)),
                _fmt_metric(metrics.get("offset_l1", math.nan)),
                _fmt_metric(metrics.get("descriptor_reg", math.nan)),
                _fmt_metric(metrics.get("coverage_kl", math.nan)),
                _fmt_metric(metrics.get("proposal_repulsion", math.nan)),
                _fmt_metric(metrics.get("gram_anchor_loss", math.nan)),
                _fmt_metric(metrics.get("mean_teacher_score", math.nan)),
                _fmt_metric(metrics.get("mean_track_survival", math.nan)),
                _fmt_metric(metrics.get("mean_quality", math.nan)),
                _fmt_metric(metrics.get("external_mean_ate", math.nan)),
                _fmt_metric(metrics.get("external_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("external_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("mean_unique_semantic_count_before_repeat", math.nan)),
                _fmt_metric(metrics.get("repeated_patch_fraction", math.nan)),
                _fmt_metric(metrics.get("mean_dedupe_radius_used", math.nan)),
                _fmt_metric(metrics.get("mean_grid_occupancy", math.nan)),
                _fmt_metric(metrics.get("mean_track_age", math.nan)),
                _fmt_metric(metrics.get("survival_rate_1", math.nan)),
                _fmt_metric(metrics.get("survival_rate_3", math.nan)),
                _fmt_metric(metrics.get("survival_rate_5", math.nan)),
                _fmt_metric(metrics.get("unstable_motion_proxy_fraction", math.nan)),
                _fmt_metric(metrics.get("dino_patch_fraction", math.nan)),
                _fmt_metric(metrics.get("pure100_mean_ate", math.nan)),
                _fmt_metric(metrics.get("pure100_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("pure100_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("lowtex_mean_ate", math.nan)),
                _fmt_metric(metrics.get("lowtex_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("lowtex_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("hybrid_mean_ate", math.nan)),
                _fmt_metric(metrics.get("hybrid_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("hybrid_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("hybrid_lowtex_mean_ate", math.nan)),
                _fmt_metric(metrics.get("hybrid_lowtex_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("hybrid_lowtex_mean_coverage", math.nan)),
                str(metrics.get("best_mode", "")),
                _fmt_metric(metrics.get("best_pure_assoc", math.nan)),
                _fmt_metric(metrics.get("best_hybrid_assoc", math.nan)),
                _fmt_metric(metrics.get("best_pure_lowtex_assoc", math.nan)),
                _fmt_metric(metrics.get("best_hybrid_lowtex_assoc", math.nan)),
                str(metrics.get("selection_metric", "")),
                str(metrics.get("selection_passed_gate", "")),
                _fmt_metric(metrics.get("tum_proxy_mean_ate", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_rpe_trans_rmse", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_rpe_rot_rmse", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_scale_correction", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_scale_error_abs", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_mean_scale_error_abs_log", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_failed_count", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_finite_count", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_row_count", math.nan)),
                _fmt_metric(metrics.get("tum_proxy_wins_vs_dpvo", math.nan)),
                _fmt_metric(metrics.get("tum_pressure_wins_vs_dpvo", math.nan)),
                _fmt_metric(metrics.get("tum_pressure_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("tum_pressure_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_ate", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_rpe_trans_rmse", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_rpe_rot_rmse", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_scale_correction", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_scale_error_abs", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_mean_scale_error_abs_log", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_failed_count", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_finite_count", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_row_count", math.nan)),
                _fmt_metric(metrics.get("euroc_proxy_wins_vs_dpvo", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_ate", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_ate_associated", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_coverage", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_rpe_trans_rmse", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_rpe_rot_rmse", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_scale_correction", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_scale_error_abs", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_scale_error_abs_log", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_kitti_trans_percent", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_mean_kitti_rot_deg_per_m", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_failed_count", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_finite_count", math.nan)),
                _fmt_metric(metrics.get("kitti_proxy_row_count", math.nan)),
                _fmt_metric(metrics.get("live_tri_proxy_score", math.nan)),
                _fmt_metric(metrics.get("live_weighted_rpe_trans_score", math.nan)),
                _fmt_metric(metrics.get("live_weighted_rpe_rot_score", math.nan)),
                _fmt_metric(metrics.get("live_weighted_scale_error_abs_log_score", math.nan)),
                _fmt_metric(metrics.get("live_tum_gate_pass", math.nan)),
                _fmt_metric(metrics.get("live_transfer_gate_pass", math.nan)),
                _fmt_metric(metrics.get("live_dual_proxy_score", math.nan)),
                _fmt_metric(metrics.get("live_pure_tum_proxy_score", math.nan)),
                _fmt_metric(metrics.get("best_dual_proxy_score", math.nan)),
                _fmt_metric(metrics.get("best_tum_proxy_wins_vs_dpvo", math.nan)),
                _fmt_metric(metrics.get("best_euroc_proxy_wins_vs_dpvo", math.nan)),
                _fmt_metric(metrics.get("dual_selection_passed_gate", math.nan)),
            ]
        )


def _compute_track_survival_targets(
    model: DinoProposalFrontend,
    batch: dict[str, torch.Tensor],
    *,
    horizon: int,
    occlusion_tol_m: float = 0.15,
) -> torch.Tensor:
    depths = batch["depths"].to(model.device)
    poses = batch["poses"].to(model.device)
    intrinsics = batch["intrinsics"].to(model.device)
    b, t, h, w = depths.shape
    patch_depths = []
    patch_valids = []
    for ti in range(t):
        d, valid = model.teacher.pool_depth(depths[:, ti])
        patch_depths.append(d)
        patch_valids.append(valid)
    patch_depths = torch.stack(patch_depths, dim=1)
    patch_valids = torch.stack(patch_valids, dim=1)

    ht = patch_depths.shape[-2]
    wt = patch_depths.shape[-1]
    centers_u, centers_v = model.teacher.patch_centers(ht, wt, model.device)
    centers = torch.stack([centers_u, centers_v], dim=-1).reshape(-1, 2)
    centers = centers.unsqueeze(0).expand(b, -1, -1)

    targets = torch.zeros((b, t, ht * wt), device=model.device, dtype=torch.float32)
    counts = torch.zeros_like(targets)
    for bi in range(b):
        K = intrinsics[bi]
        for ti in range(t):
            src_depth = patch_depths[bi, ti].reshape(-1)
            src_valid = patch_valids[bi, ti].reshape(-1)
            for hj in range(1, max(1, int(horizon)) + 1):
                tj = ti + hj
                if tj >= t:
                    break
                rel = relative_pose_matrix(poses[bi, ti], poses[bi, tj]).to(model.device)
                proj_xy, proj_z = project_patch_centers(centers[bi], src_depth, rel, K)
                in_bounds = (
                    src_valid
                    & (proj_z > 1e-6)
                    & (proj_xy[:, 0] >= 0.0)
                    & (proj_xy[:, 0] <= float(w - 1))
                    & (proj_xy[:, 1] >= 0.0)
                    & (proj_xy[:, 1] <= float(h - 1))
                )
                sampled_tgt_depth = model.teacher.sample_scalar(
                    depths[bi, tj].unsqueeze(0),
                    proj_xy[:, 0].unsqueeze(0),
                    proj_xy[:, 1].unsqueeze(0),
                )[0]
                consistent = in_bounds & ((sampled_tgt_depth - proj_z).abs() <= float(occlusion_tol_m))
                targets[bi, ti] += consistent.float()
                counts[bi, ti] += src_valid.float()

    survival = targets / counts.clamp_min(1.0)
    return survival.reshape(b, t, ht, wt)


def _compute_coverage_regularizer(
    selector_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    grid_rows: int,
    grid_cols: int,
    uniform_mix: float,
) -> torch.Tensor:
    if selector_logits.dim() != 4 or teacher_scores.dim() != 4:
        raise ValueError("selector_logits and teacher_scores must both have shape (B,T,H,W)")

    bt = selector_logits.shape[0] * selector_logits.shape[1]
    ht = selector_logits.shape[2]
    wt = selector_logits.shape[3]
    if bt == 0 or ht == 0 or wt == 0:
        return selector_logits.new_zeros(())

    selector_map = torch.sigmoid(selector_logits).reshape(bt, 1, ht, wt)
    teacher_map = teacher_scores.reshape(bt, 1, ht, wt).detach()

    selector_cells = F.adaptive_max_pool2d(selector_map, (int(grid_rows), int(grid_cols))).flatten(1)
    teacher_cells = F.adaptive_max_pool2d(teacher_map, (int(grid_rows), int(grid_cols))).flatten(1)

    selector_dist = selector_cells / selector_cells.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_dist = teacher_cells / teacher_cells.sum(dim=1, keepdim=True).clamp_min(1e-6)

    num_cells = selector_dist.shape[1]
    uniform = torch.full_like(selector_dist, 1.0 / max(num_cells, 1))
    mix = float(max(0.0, min(1.0, uniform_mix)))
    target_dist = ((1.0 - mix) * teacher_dist) + (mix * uniform)
    target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)

    return F.kl_div(selector_dist.clamp_min(1e-6).log(), target_dist, reduction="batchmean")


def _compute_proposal_repulsion_loss(
    output,
    *,
    radius_px: float,
    top_k: int,
) -> torch.Tensor:
    radius_px = float(radius_px)
    if radius_px <= 0.0 or int(top_k) <= 1:
        ref = output.selector_logits
        return ref.new_zeros(())

    losses = []
    for frame_outputs in output.observations:
        for frame_output in frame_outputs:
            pixel_xy = frame_output.proposal.pixel_xy
            quality = frame_output.qualities
            if pixel_xy.numel() == 0 or pixel_xy.shape[0] < 2:
                continue
            k = min(int(top_k), int(pixel_xy.shape[0]))
            if k < 2:
                continue
            ranking = torch.argsort(quality, descending=True)[:k]
            selected_xy = pixel_xy[ranking]
            pairwise = torch.cdist(selected_xy, selected_xy)
            mask = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
            penalties = F.relu(float(radius_px) - pairwise[mask]) / float(radius_px)
            if penalties.numel() > 0:
                losses.append(penalties.mean())

    if not losses:
        ref = output.selector_logits
        return ref.new_zeros(())
    return torch.stack(losses).mean()


def _parse_dpvo_opts(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item for item in value.strip().split() if item]
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    else:
        return []

    opts: list[str] = []
    for item in raw_items:
        if "=" in item:
            key, raw_value = item.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key and raw_value:
                opts.extend([key, raw_value])
        else:
            opts.append(item)
    return opts


def _compute_frontend_losses(
    model: DinoProposalFrontend,
    output,
    batch: dict[str, torch.Tensor],
    cfg: DinoDPVOConfig,
    *,
    step: int,
    gram_anchor_teacher: DinoProposalFrontend | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_scores = build_teacher_scores_from_fused(output.fused, batch, model.teacher)
    static_targets = _compute_track_survival_targets(
        model,
        batch,
        horizon=int(cfg.model.get("track_survival_horizon", 2)),
    )
    offset_targets = dense_gradient_offset_targets(batch["images"].to(model.device), model.patch_size)
    coverage_loss = _compute_coverage_regularizer(
        output.selector_logits,
        teacher_scores,
        grid_rows=int(cfg.model.get("coverage_grid_rows", cfg.model.get("hybrid_grid_rows", 6))),
        grid_cols=int(cfg.model.get("coverage_grid_cols", cfg.model.get("hybrid_grid_cols", 8))),
        uniform_mix=float(cfg.losses.get("coverage_uniform_mix", 0.15)),
    )

    selector_loss = F.binary_cross_entropy_with_logits(output.selector_logits, teacher_scores)
    static_loss = F.binary_cross_entropy_with_logits(output.staticness_logits, static_targets)

    offset_losses = []
    descriptor_regs = []
    mean_quality = []
    for bi, frame_outputs in enumerate(output.observations):
        for ti, frame_output in enumerate(frame_outputs):
            if frame_output.proposal.patch_indices.numel() > 0:
                dense_target = offset_targets[bi, ti].reshape(-1, 2)
                target_offsets = dense_target[frame_output.proposal.patch_indices]
                offset_losses.append(F.l1_loss(frame_output.proposal.offset_xy, target_offsets))
            if frame_output.descriptor_bias is not None:
                descriptor_regs.append(frame_output.descriptor_bias.square().mean())
            mean_quality.append(frame_output.qualities.mean())

    offset_loss = torch.stack(offset_losses).mean() if offset_losses else torch.zeros((), device=model.device)
    descriptor_reg = torch.stack(descriptor_regs).mean() if descriptor_regs else torch.zeros((), device=model.device)
    mean_quality_value = torch.stack(mean_quality).mean() if mean_quality else torch.zeros((), device=model.device)
    proposal_repulsion = _compute_proposal_repulsion_loss(
        output,
        radius_px=float(cfg.model.get("proposal_repulsion_radius_px", cfg.model.get("dedupe_radius_px", 8.0))),
        top_k=int(cfg.model.get("proposal_repulsion_topk", cfg.model.get("patch_budget", 96))),
    )
    gram_anchor_loss = torch.zeros((), device=model.device)
    gram_anchor_weight = float(cfg.losses.get("gram_anchor", 0.0))
    gram_anchor_start_step = int(cfg.training.get("gram_anchor_start_step", 0) or 0)
    if gram_anchor_teacher is not None and gram_anchor_weight > 0.0 and int(step) >= max(0, gram_anchor_start_step):
        with torch.inference_mode():
            teacher_fused = _gram_anchor_teacher_fused(gram_anchor_teacher, batch["images"].to(model.device))
        gram_anchor_loss = _compute_sampled_similarity_loss(
            output.fused,
            teacher_fused,
            sample_tokens=int(cfg.training.get("gram_anchor_sample_tokens", 96) or 96),
            downsample_stride=int(cfg.training.get("gram_anchor_downsample_stride", 2) or 2),
        )

    loss_weights = cfg.losses
    loss = (
        float(loss_weights.get("selector_bce", 1.0)) * selector_loss
        + float(loss_weights.get("staticness_bce", 0.5)) * static_loss
        + float(loss_weights.get("offset_l1", 0.25)) * offset_loss
        + float(loss_weights.get("descriptor_reg", 0.01)) * descriptor_reg
        + float(loss_weights.get("coverage_kl", 0.0)) * coverage_loss
        + float(loss_weights.get("proposal_repulsion", 0.0)) * proposal_repulsion
        + gram_anchor_weight * gram_anchor_loss
    )
    metrics = {
        "loss": float(loss.item()),
        "selector_bce": float(selector_loss.item()),
        "staticness_bce": float(static_loss.item()),
        "offset_l1": float(offset_loss.item()),
        "descriptor_reg": float(descriptor_reg.item()),
        "coverage_kl": float(coverage_loss.item()),
        "proposal_repulsion": float(proposal_repulsion.item()),
        "gram_anchor_loss": float(gram_anchor_loss.item()),
        "mean_teacher_score": float(teacher_scores.mean().item()),
        "mean_track_survival": float(static_targets.mean().item()),
        "mean_quality": float(mean_quality_value.item()),
    }
    return loss, metrics


@torch.no_grad()
def _evaluate_dev_loss(
    model: DinoProposalFrontend,
    loader: DataLoader,
    cfg: DinoDPVOConfig,
    *,
    max_batches: int = 8,
    gram_anchor_teacher: DinoProposalFrontend | None = None,
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        output = model(batch["images"].to(model.device))
        _, metrics = _compute_frontend_losses(
            model,
            output,
            batch,
            cfg,
            step=0,
            gram_anchor_teacher=gram_anchor_teacher,
        )
        for key, value in metrics.items():
            agg[key] = agg.get(key, 0.0) + float(value)
        count += 1
        if count >= int(max_batches):
            break
    if count == 0:
        return {"loss": math.nan}
    return {k: v / count for k, v in agg.items()}


@torch.no_grad()
def _evaluate_external_ate(
    model: DinoProposalFrontend,
    cfg: DinoDPVOConfig,
    *,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    step: int,
) -> dict[str, float]:
    sequences, _ = _select_eval_sequences(dataset_root)
    max_sequences = int(cfg.training.get("max_eval_sequences", 4))
    sequences = sequences[:max_sequences]
    if not sequences:
        return {"external_mean_ate": math.nan, "external_mean_coverage": math.nan}

    eval_dir = output_dir / "dev_eval" / f"step_{step:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_list = []
    assoc_metrics_list = []
    coverage_list = []
    model.eval()
    dpvo_opts = _parse_dpvo_opts(cfg.eval.get("dpvo_opts", []))
    for seq in sequences:
        status, metrics = evaluate_dpvo_tartanair_sequence(
            sequence=seq,
            dpvo_root=dpvo_root,
            weights=dpvo_weights,
            config_path=dpvo_config,
            output_dir=eval_dir,
            max_dt=float(cfg.eval.get("max_dt", 0.02)),
            missing_penalty_m=float(cfg.eval.get("missing_penalty_m", 3.0)),
            min_coverage_ok=float(cfg.eval.get("coverage_gate", 0.99)),
            stride=int(cfg.eval.get("stride", 1)),
            backend_thresh=float(cfg.eval.get("backend_thresh", 18.0)),
            viz=False,
            opts=dpvo_opts,
            image_height=int(cfg.model.get("image_size", [240, 320])[0]),
            image_width=int(cfg.model.get("image_size", [240, 320])[1]),
            frontend_mode=str(cfg.eval.get("frontend_mode", "dino_proposals")),
            frontend_cfg=cfg,
            frontend=model,
        )
        if metrics is None:
            continue
        ate = float(metrics.get("ate_rmse", math.nan))
        ate_assoc = float(metrics.get("ate_rmse_associated", math.nan))
        coverage = float(metrics.get("coverage", math.nan))
        if math.isfinite(ate):
            metrics_list.append(ate)
        if math.isfinite(ate_assoc):
            assoc_metrics_list.append(ate_assoc)
        if math.isfinite(coverage):
            coverage_list.append(coverage)
    return {
        "external_mean_ate": (sum(metrics_list) / len(metrics_list)) if metrics_list else math.nan,
        "external_mean_ate_associated": (sum(assoc_metrics_list) / len(assoc_metrics_list)) if assoc_metrics_list else math.nan,
        "external_mean_coverage": (sum(coverage_list) / len(coverage_list)) if coverage_list else math.nan,
    }


def _deep_update(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)


def _cfg_with_overrides(
    cfg: DinoDPVOConfig,
    *,
    method_suffix: str,
    feature_suffix: str,
    frontend_mode: str,
    model_overrides: dict | None = None,
    eval_overrides: dict | None = None,
) -> DinoDPVOConfig:
    raw = deepcopy(cfg.raw)
    raw["method_id"] = f"{cfg.method_id}_{method_suffix}"
    raw["feature_type"] = f"{cfg.feature_type}_{feature_suffix}"
    raw.setdefault("model", {})
    raw.setdefault("eval", {})
    if model_overrides:
        _deep_update(raw["model"], model_overrides)
    if eval_overrides:
        _deep_update(raw["eval"], eval_overrides)
    raw["eval"]["frontend_mode"] = str(frontend_mode)
    return DinoDPVOConfig(
        method_id=str(raw["method_id"]),
        feature_type=str(raw["feature_type"]),
        raw=raw,
    )


def _mean_external_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    ate = [float(row.get("ate_rmse", math.nan)) for row in rows if math.isfinite(float(row.get("ate_rmse", math.nan)))]
    assoc = [
        float(row.get("ate_rmse_associated", math.nan))
        for row in rows
        if math.isfinite(float(row.get("ate_rmse_associated", math.nan)))
    ]
    coverage = [float(row.get("coverage", math.nan)) for row in rows if math.isfinite(float(row.get("coverage", math.nan)))]
    return {
        "external_mean_ate": (sum(ate) / len(ate)) if ate else math.nan,
        "external_mean_ate_associated": (sum(assoc) / len(assoc)) if assoc else math.nan,
        "external_mean_coverage": (sum(coverage) / len(coverage)) if coverage else math.nan,
    }


def _mean_csv_metric(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except Exception:
            value = math.nan
        if math.isfinite(value):
            values.append(value)
    return (sum(values) / len(values)) if values else math.nan


def _read_external_summary_metrics(
    summary_path: Path,
    *,
    exclude_sequences: set[str] | None = None,
) -> dict[str, float]:
    if not summary_path.exists():
        return {
            "external_mean_ate": math.nan,
            "external_mean_ate_associated": math.nan,
            "external_mean_coverage": math.nan,
        }

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    excluded = {str(seq) for seq in (exclude_sequences or set())}
    if excluded:
        rows = [row for row in rows if str(row.get("sequence", "")).strip() not in excluded]

    return {
        "external_mean_ate": _mean_csv_metric(rows, "ate_rmse"),
        "external_mean_ate_associated": _mean_csv_metric(rows, "ate_rmse_associated"),
        "external_mean_coverage": _mean_csv_metric(rows, "coverage"),
    }


def _mean_jsonl_boolean(path: Path, key: str) -> float:
    if not path.exists():
        return math.nan
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            values.append(1.0 if bool(row.get(key, False)) else 0.0)
    return (sum(values) / len(values)) if values else math.nan


def _read_diagnostics_summary_metrics(
    summary_path: Path,
    *,
    patch_diagnostics_path: Path | None = None,
) -> dict[str, float]:
    if not summary_path.exists():
        return {
            "mean_unique_semantic_count_before_repeat": math.nan,
            "repeated_patch_fraction": math.nan,
            "mean_dedupe_radius_used": math.nan,
            "mean_grid_occupancy": math.nan,
            "mean_track_age": math.nan,
            "survival_rate_1": math.nan,
            "survival_rate_3": math.nan,
            "survival_rate_5": math.nan,
            "unstable_motion_proxy_fraction": math.nan,
            "dino_patch_fraction": math.nan,
        }

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        "mean_unique_semantic_count_before_repeat": _mean_csv_metric(rows, "mean_unique_semantic_count_before_repeat"),
        "repeated_patch_fraction": _mean_csv_metric(rows, "repeated_patch_fraction"),
        "mean_dedupe_radius_used": _mean_csv_metric(rows, "mean_dedupe_radius_used"),
        "mean_grid_occupancy": _mean_csv_metric(rows, "mean_grid_occupancy"),
        "mean_track_age": _mean_csv_metric(rows, "mean_track_age"),
        "survival_rate_1": _mean_csv_metric(rows, "survival_rate_1"),
        "survival_rate_3": _mean_csv_metric(rows, "survival_rate_3"),
        "survival_rate_5": _mean_csv_metric(rows, "survival_rate_5"),
        "unstable_motion_proxy_fraction": _mean_jsonl_boolean(
            patch_diagnostics_path,
            "unstable_motion_proxy",
        ) if patch_diagnostics_path is not None else math.nan,
        "dino_patch_fraction": _mean_csv_metric(rows, "dino_patch_fraction"),
    }


LIVE_PROXY_STATUSES_OK = {"ok", "partial_low_coverage"}
LIVE_PROXY_SUMMARY_HEADER = [
    "sequence",
    "status",
    "ate_rmse",
    "ate_rmse_associated",
    "rpe_trans_rmse",
    "rpe_rot_rmse",
    "scale_correction",
    "scale_error_abs",
    "scale_error_abs_log",
    "coverage",
    "kitti_trans_percent",
    "kitti_rot_deg_per_m",
]


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "nan", "none"}


def _write_live_proxy_summary_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(LIVE_PROXY_SUMMARY_HEADER)


def _append_live_proxy_summary_row(path: Path, *, sequence: str, status: str, metrics: dict[str, float] | None) -> None:
    payload = dict(metrics or {})
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            [
                sequence,
                status,
                _fmt_metric(payload.get("ate_rmse", math.nan)),
                _fmt_metric(payload.get("ate_rmse_associated", math.nan)),
                _fmt_metric(payload.get("rpe_trans_rmse", math.nan)),
                _fmt_metric(payload.get("rpe_rot_rmse", math.nan)),
                _fmt_metric(payload.get("scale_correction", math.nan)),
                _fmt_metric(payload.get("scale_error_abs", math.nan)),
                _fmt_metric(payload.get("scale_error_abs_log", math.nan)),
                _fmt_metric(payload.get("coverage", math.nan)),
                _fmt_metric(payload.get("kitti_trans_percent", math.nan)),
                _fmt_metric(payload.get("kitti_rot_deg_per_m", math.nan)),
            ]
        )


def _mean_metric_values(rows: list[dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except Exception:
            value = math.nan
        if math.isfinite(value):
            values.append(value)
    return (sum(values) / len(values)) if values else math.nan


def _summarize_live_proxy_rows(
    rows: list[dict[str, object]],
    *,
    pressure_sequences: list[str] | tuple[str, ...] | None = None,
    baseline_assoc: dict[str, float] | None = None,
) -> dict[str, float]:
    pressure_set = {str(seq).strip() for seq in (pressure_sequences or []) if str(seq).strip()}
    pressure_values: list[float] = []
    pressure_rows: list[dict[str, object]] = []
    ok_count = 0
    non_ok_count = 0
    failed_count = 0
    finite_count = 0
    for row in rows:
        status = str(row.get("status", "")).strip()
        if status == "ok":
            ok_count += 1
        else:
            non_ok_count += 1
        if status not in LIVE_PROXY_STATUSES_OK:
            failed_count += 1
        ate_assoc = float(row.get("ate_rmse_associated", math.nan))
        if math.isfinite(ate_assoc):
            finite_count += 1
            if str(row.get("sequence", "")).strip() in pressure_set:
                pressure_values.append(ate_assoc)
                pressure_rows.append(row)

    pressure_wins, pressure_losses, pressure_ties = _wins_against_sequence_baseline(
        pressure_rows,
        baseline_assoc=baseline_assoc or {},
    )
    return {
        "row_count": float(len(rows)),
        "finite_count": float(finite_count),
        "ok_count": float(ok_count),
        "non_ok_count": float(non_ok_count),
        "failed_count": float(failed_count),
        "external_mean_ate": _mean_metric_values(rows, "ate_rmse"),
        "external_mean_ate_associated": _mean_metric_values(rows, "ate_rmse_associated"),
        "external_mean_coverage": _mean_metric_values(rows, "coverage"),
        "mean_rpe_trans_rmse": _mean_metric_values(rows, "rpe_trans_rmse"),
        "mean_rpe_rot_rmse": _mean_metric_values(rows, "rpe_rot_rmse"),
        "mean_scale_correction": _mean_metric_values(rows, "scale_correction"),
        "mean_scale_error_abs": _mean_metric_values(rows, "scale_error_abs"),
        "mean_scale_error_abs_log": _mean_metric_values(rows, "scale_error_abs_log"),
        "pressure_mean_ate_associated": (sum(pressure_values) / len(pressure_values)) if pressure_values else math.nan,
        "pressure_row_count": float(len(pressure_rows)),
        "pressure_wins_vs_reference": float(pressure_wins),
        "pressure_losses_vs_reference": float(pressure_losses),
        "pressure_ties_vs_reference": float(pressure_ties),
        "mean_kitti_trans_percent": _mean_metric_values(rows, "kitti_trans_percent"),
        "mean_kitti_rot_deg_per_m": _mean_metric_values(rows, "kitti_rot_deg_per_m"),
        "mean_grid_occupancy": _mean_metric_values(rows, "mean_grid_occupancy"),
        "mean_track_age": _mean_metric_values(rows, "mean_track_age"),
        "survival_rate_1": _mean_metric_values(rows, "survival_rate_1"),
        "survival_rate_3": _mean_metric_values(rows, "survival_rate_3"),
        "survival_rate_5": _mean_metric_values(rows, "survival_rate_5"),
        "unstable_motion_proxy_fraction": _mean_metric_values(rows, "unstable_motion_proxy_fraction"),
        "dino_patch_fraction": _mean_metric_values(rows, "dino_patch_fraction"),
    }


@torch.no_grad()
def _evaluate_live_proxy_dataset(
    model: DinoProposalFrontend,
    cfg: DinoDPVOConfig,
    *,
    dataset_id: str,
    dataset_root: Path | None,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    step: int,
    spec: dict[str, object],
    sequence_assoc_baselines: dict[str, float] | None = None,
) -> dict[str, float]:
    if dataset_root is None or not dataset_root.exists():
        return {
            "row_count": math.nan,
            "finite_count": math.nan,
            "ok_count": math.nan,
            "non_ok_count": math.nan,
            "failed_count": math.nan,
            "external_mean_ate": math.nan,
            "external_mean_ate_associated": math.nan,
            "external_mean_coverage": math.nan,
            "mean_rpe_trans_rmse": math.nan,
            "mean_rpe_rot_rmse": math.nan,
            "mean_scale_correction": math.nan,
            "mean_scale_error_abs": math.nan,
            "mean_scale_error_abs_log": math.nan,
            "pressure_mean_ate_associated": math.nan,
            "mean_kitti_trans_percent": math.nan,
            "mean_kitti_rot_deg_per_m": math.nan,
            "wins_vs_reference": math.nan,
            "losses_vs_reference": math.nan,
            "ties_vs_reference": math.nan,
        }

    sequences = [str(item).strip() for item in (spec.get("sequences", []) or []) if str(item).strip()]
    if not sequences:
        return {
            "row_count": 0.0,
            "finite_count": 0.0,
            "ok_count": 0.0,
            "non_ok_count": 0.0,
            "failed_count": 0.0,
            "external_mean_ate": math.nan,
            "external_mean_ate_associated": math.nan,
            "external_mean_coverage": math.nan,
            "mean_rpe_trans_rmse": math.nan,
            "mean_rpe_rot_rmse": math.nan,
            "mean_scale_correction": math.nan,
            "mean_scale_error_abs": math.nan,
            "mean_scale_error_abs_log": math.nan,
            "pressure_mean_ate_associated": math.nan,
            "mean_kitti_trans_percent": math.nan,
            "mean_kitti_rot_deg_per_m": math.nan,
            "wins_vs_reference": math.nan,
            "losses_vs_reference": math.nan,
            "ties_vs_reference": math.nan,
        }

    eval_dir = output_dir / "live_proxy_eval" / f"step_{step:06d}" / str(dataset_id)
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "metrics_summary.csv"
    _write_live_proxy_summary_header(csv_path)
    collect_diagnostics = bool(spec.get("collect_diagnostics", False))
    write_patch_diagnostics = bool(spec.get("write_patch_diagnostics", collect_diagnostics))

    rows: list[dict[str, object]] = []
    frontend_mode = str(spec.get("frontend_mode", "dino_hybrid"))
    opts = _parse_dpvo_opts(str(spec.get("dpvo_opts", "")))
    max_dt = float(spec.get("max_dt", 0.02))
    missing_penalty_m = float(spec.get("missing_penalty_m", 3.0))
    min_coverage_ok = float(spec.get("min_coverage_ok", 0.95))
    stride = int(spec.get("stride", 4))
    backend_thresh = float(spec.get("backend_thresh", 32.0))
    image_height = int(spec.get("image_height", cfg.model.get("image_size", [240, 320])[0]))
    image_width = int(spec.get("image_width", cfg.model.get("image_size", [240, 320])[1]))
    pressure_sequences = [str(item).strip() for item in (spec.get("pressure_sequences", []) or []) if str(item).strip()]

    model.eval()
    for sequence in sequences:
        if str(dataset_id) == "tum":
            diagnostics_summary_path = eval_dir / f"{sequence}_diagnostics_summary.csv" if collect_diagnostics else None
            patch_diagnostics_path = (
                eval_dir / f"{sequence}_patch_diagnostics.jsonl"
                if collect_diagnostics and write_patch_diagnostics
                else None
            )
            if collect_diagnostics and diagnostics_summary_path is not None:
                init_diagnostics_outputs(diagnostics_summary_path, patch_diagnostics_path)
            status, metrics = evaluate_dpvo_tum_sequence(
                dataset_root=dataset_root,
                sequence=sequence,
                dpvo_root=dpvo_root,
                weights=dpvo_weights,
                config_path=dpvo_config,
                output_dir=eval_dir,
                max_dt=max_dt,
                missing_penalty_m=missing_penalty_m,
                min_coverage_ok=min_coverage_ok,
                stride=stride,
                backend_thresh=backend_thresh,
                viz=False,
                opts=opts,
                target_height=image_height,
                target_width=image_width,
                frontend_mode=frontend_mode,
                frontend_cfg=cfg,
                frontend=model,
                collect_diagnostics=collect_diagnostics,
                diagnostics_summary_path=diagnostics_summary_path,
                patch_diagnostics_path=patch_diagnostics_path,
                feature_type=cfg.feature_type,
                write_plots=False,
            )
        elif str(dataset_id) == "euroc":
            status, metrics = evaluate_dpvo_euroc_sequence(
                dataset_root=dataset_root,
                groundtruth_root=None,
                sequence=sequence,
                dpvo_root=dpvo_root,
                weights=dpvo_weights,
                config_path=dpvo_config,
                output_dir=eval_dir,
                max_dt=max_dt,
                missing_penalty_m=missing_penalty_m,
                min_coverage_ok=min_coverage_ok,
                stride=stride,
                backend_thresh=backend_thresh,
                viz=False,
                opts=opts,
                image_height=image_height,
                image_width=image_width,
                frontend_mode=frontend_mode,
                frontend_cfg=cfg,
                frontend=model,
                feature_type=cfg.feature_type,
                calib_path=None,
            )
        elif str(dataset_id) == "kitti":
            status, metrics = evaluate_dpvo_kitti_sequence(
                dataset_root=dataset_root,
                sequence=sequence,
                dpvo_root=dpvo_root,
                weights=dpvo_weights,
                config_path=dpvo_config,
                output_dir=eval_dir,
                max_dt=max_dt,
                missing_penalty_m=missing_penalty_m,
                min_coverage_ok=min_coverage_ok,
                stride=stride,
                backend_thresh=backend_thresh,
                viz=False,
                opts=opts,
                image_height=image_height,
                image_width=image_width,
                frontend_mode=frontend_mode,
                frontend_cfg=cfg,
                frontend=model,
                feature_type=cfg.feature_type,
            )
        else:
            raise ValueError(f"Unsupported live proxy dataset: {dataset_id}")

        row_payload = dict(metrics or {})
        row_payload["sequence"] = sequence
        row_payload["status"] = str(status)
        if str(dataset_id) == "tum" and collect_diagnostics:
            row_payload.update(
                _read_diagnostics_summary_metrics(
                    eval_dir / f"{sequence}_diagnostics_summary.csv",
                    patch_diagnostics_path=(
                        eval_dir / f"{sequence}_patch_diagnostics.jsonl"
                        if write_patch_diagnostics
                        else None
                    ),
                )
            )
        rows.append(row_payload)
        _append_live_proxy_summary_row(csv_path, sequence=sequence, status=str(status), metrics=metrics)

    summary = _summarize_live_proxy_rows(
        rows,
        pressure_sequences=pressure_sequences,
        baseline_assoc=sequence_assoc_baselines or {},
    )
    wins, losses, ties = _wins_against_sequence_baseline(
        rows,
        baseline_assoc=sequence_assoc_baselines or {},
    )
    summary["wins_vs_reference"] = float(wins)
    summary["losses_vs_reference"] = float(losses)
    summary["ties_vs_reference"] = float(ties)
    return summary


def _empty_live_proxy_summary() -> dict[str, float]:
    return {
        "row_count": math.nan,
        "finite_count": math.nan,
        "ok_count": math.nan,
        "non_ok_count": math.nan,
        "failed_count": math.nan,
        "external_mean_ate": math.nan,
        "external_mean_ate_associated": math.nan,
        "external_mean_coverage": math.nan,
        "mean_rpe_trans_rmse": math.nan,
        "mean_rpe_rot_rmse": math.nan,
        "mean_scale_correction": math.nan,
        "mean_scale_error_abs": math.nan,
        "mean_scale_error_abs_log": math.nan,
        "pressure_mean_ate_associated": math.nan,
        "pressure_row_count": math.nan,
        "pressure_wins_vs_reference": math.nan,
        "pressure_losses_vs_reference": math.nan,
        "pressure_ties_vs_reference": math.nan,
        "mean_kitti_trans_percent": math.nan,
        "mean_kitti_rot_deg_per_m": math.nan,
        "mean_grid_occupancy": math.nan,
        "mean_track_age": math.nan,
        "survival_rate_1": math.nan,
        "survival_rate_3": math.nan,
        "survival_rate_5": math.nan,
        "unstable_motion_proxy_fraction": math.nan,
        "dino_patch_fraction": math.nan,
        "wins_vs_reference": math.nan,
        "losses_vs_reference": math.nan,
        "ties_vs_reference": math.nan,
    }


@torch.no_grad()
def _evaluate_live_proxy_selection(
    model: DinoProposalFrontend,
    cfg: DinoDPVOConfig,
    *,
    tum_dataset_root: Path | None,
    euroc_dataset_root: Path | None,
    kitti_dataset_root: Path | None,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    step: int,
) -> dict[str, float]:
    live_cfg = dict(cfg.eval.get("live_proxy", {}) or {})
    datasets_cfg = dict(live_cfg.get("datasets", {}) or {})
    references = dict(live_cfg.get("references", {}) or {})
    weights = {str(key): float(value) for key, value in (live_cfg.get("weights", {}) or {}).items()}
    win_weights = {str(key): float(value) for key, value in (live_cfg.get("win_weights", {}) or {}).items()}
    gate_cfg = dict(live_cfg.get("gate", {}) or {})
    tum_no_regression_multiplier = float(
        gate_cfg.get("tum_no_regression_multiplier", live_cfg.get("tum_no_regression_multiplier", 1.03))
    )
    tum_pressure_multiplier = float(gate_cfg.get("tum_pressure_multiplier", 1.0e9))
    tum_min_wins_vs_dpvo = int(gate_cfg.get("tum_proxy_min_wins_vs_dpvo", 0) or 0)
    required_valid_datasets = [
        str(item).strip()
        for item in (gate_cfg.get("required_valid_datasets", []) or [])
        if str(item).strip()
    ]
    if not required_valid_datasets:
        required_valid_datasets = [
            dataset_id
            for dataset_id in datasets_cfg
            if dataset_id != "tum"
        ]

    dataset_roots = {
        "tum": tum_dataset_root,
        "euroc": euroc_dataset_root,
        "kitti": kitti_dataset_root,
    }
    sequence_assoc_baselines = dict(references.get("sequence_assoc_baselines", {}) or {})
    dataset_summaries: dict[str, dict[str, float]] = {}
    for dataset_id in datasets_cfg:
        dataset_summaries[str(dataset_id)] = _evaluate_live_proxy_dataset(
            model,
            cfg,
            dataset_id=str(dataset_id),
            dataset_root=dataset_roots.get(str(dataset_id)),
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            output_dir=output_dir,
            step=step,
            spec=dict(datasets_cfg.get(str(dataset_id), {}) or {}),
            sequence_assoc_baselines=_sequence_assoc_baseline_map(sequence_assoc_baselines, dataset_id=str(dataset_id)),
        )

    def _dataset(dataset_id: str) -> dict[str, float]:
        return dict(dataset_summaries.get(dataset_id, _empty_live_proxy_summary()))

    active_weighted_datasets = [
        dataset_id
        for dataset_id in datasets_cfg
        if float(weights.get(str(dataset_id), 0.0)) > 0.0
    ]
    weighted_ate = _weighted_ratio_score(
        dataset_ids=active_weighted_datasets,
        weights=weights,
        value_lookup=lambda dataset_id: _dataset(dataset_id).get("external_mean_ate_associated", math.nan),
        reference_lookup=lambda dataset_id: references.get(f"{dataset_id}_mean_ate_rmse_associated", math.nan),
    )
    weighted_rpe_trans = _weighted_ratio_score(
        dataset_ids=active_weighted_datasets,
        weights=weights,
        value_lookup=lambda dataset_id: _dataset(dataset_id).get("mean_rpe_trans_rmse", math.nan),
        reference_lookup=lambda dataset_id: references.get(f"{dataset_id}_mean_rpe_trans_rmse", math.nan),
    )
    weighted_rpe_rot = _weighted_ratio_score(
        dataset_ids=active_weighted_datasets,
        weights=weights,
        value_lookup=lambda dataset_id: _dataset(dataset_id).get("mean_rpe_rot_rmse", math.nan),
        reference_lookup=lambda dataset_id: references.get(f"{dataset_id}_mean_rpe_rot_rmse", math.nan),
    )
    weighted_scale = _weighted_ratio_score(
        dataset_ids=active_weighted_datasets,
        weights=weights,
        value_lookup=lambda dataset_id: _dataset(dataset_id).get("mean_scale_error_abs_log", math.nan),
        reference_lookup=lambda dataset_id: references.get(f"{dataset_id}_mean_scale_error_abs_log", math.nan),
    )
    active_win_datasets = [
        dataset_id
        for dataset_id in datasets_cfg
        if float(win_weights.get(str(dataset_id), 0.0)) > 0.0
    ]
    weighted_win_penalty = _weighted_component_score(
        item_ids=active_win_datasets,
        weights=win_weights,
        value_lookup=lambda dataset_id: (
            (
                max(0.0, float(_dataset(dataset_id).get("row_count", math.nan)) - float(_dataset(dataset_id).get("wins_vs_reference", math.nan)))
                / max(float(_dataset(dataset_id).get("row_count", math.nan)), 1.0)
            )
            if math.isfinite(float(_dataset(dataset_id).get("row_count", math.nan)))
            and float(_dataset(dataset_id).get("row_count", math.nan)) > 0.0
            and math.isfinite(float(_dataset(dataset_id).get("wins_vs_reference", math.nan)))
            else math.inf
        ),
    )
    weighted_dual_score = float(weighted_ate + weighted_win_penalty) if math.isfinite(weighted_ate) else math.inf

    tum = _dataset("tum")
    euroc = _dataset("euroc")
    kitti = _dataset("kitti")
    selection_metric = str(cfg.eval.get("selection_metric", "associated_ate")).strip().lower()
    pure_tum_weights = dict(live_cfg.get("pure_tum_weights", {}) or {})
    primary_ref = float(references.get("tum_mean_ate_rmse_associated", math.nan))
    pressure_ref = float(references.get("tum_pressure_mean_ate_rmse_associated", math.nan))
    primary_row_count = float(tum.get("row_count", math.nan))
    pressure_row_count = float(tum.get("pressure_row_count", math.nan))
    primary_wins = float(tum.get("wins_vs_reference", math.nan))
    pressure_wins = float(tum.get("pressure_wins_vs_reference", math.nan))
    primary_win_penalty = (
        max(0.0, primary_row_count - primary_wins) / max(primary_row_count, 1.0)
        if math.isfinite(primary_row_count) and primary_row_count > 0.0 and math.isfinite(primary_wins)
        else math.inf
    )
    pressure_win_penalty = (
        max(0.0, pressure_row_count - pressure_wins) / max(pressure_row_count, 1.0)
        if math.isfinite(pressure_row_count) and pressure_row_count > 0.0 and math.isfinite(pressure_wins)
        else math.inf
    )
    pure_tum_score = (
        float(pure_tum_weights.get("primary_ate", 0.55))
        * _ratio(float(tum.get("external_mean_ate_associated", math.nan)), primary_ref)
        + float(pure_tum_weights.get("pressure_ate", 0.20))
        * _ratio(float(tum.get("pressure_mean_ate_associated", math.nan)), pressure_ref)
        + float(pure_tum_weights.get("primary_wins", 0.15)) * primary_win_penalty
        + float(pure_tum_weights.get("pressure_wins", 0.10)) * pressure_win_penalty
    )
    tum_pressure_score = _ratio(
        float(tum.get("pressure_mean_ate_associated", math.nan)),
        float(references.get("tum_pressure_mean_ate_rmse_associated", math.nan)),
    )
    tum_gate_pass = (
        math.isfinite(float(tum.get("external_mean_ate_associated", math.nan)))
        and float(tum.get("failed_count", math.nan)) == 0.0
        and float(tum.get("finite_count", math.nan)) == float(tum.get("row_count", math.nan))
        and float(tum.get("external_mean_ate_associated", math.nan))
        <= float(tum_no_regression_multiplier) * float(references.get("tum_mean_ate_rmse_associated", math.nan))
        and (
            not math.isfinite(float(references.get("tum_pressure_mean_ate_rmse_associated", math.nan)))
            or float(tum.get("pressure_mean_ate_associated", math.nan))
            <= float(tum_pressure_multiplier) * float(references.get("tum_pressure_mean_ate_rmse_associated", math.nan))
        )
        and float(tum.get("wins_vs_reference", 0.0)) >= float(tum_min_wins_vs_dpvo)
    )
    require_full_dino_patch_fraction = bool(gate_cfg.get("require_full_dino_patch_fraction", False))
    full_dino_gate_pass = (
        not require_full_dino_patch_fraction
        or (
            math.isfinite(float(tum.get("dino_patch_fraction", math.nan)))
            and float(tum.get("dino_patch_fraction", math.nan)) >= 0.999
        )
    )

    def _dataset_is_valid(dataset_id: str) -> bool:
        summary = _dataset(dataset_id)
        return (
            float(summary.get("failed_count", math.nan)) == 0.0
            and float(summary.get("finite_count", math.nan)) == float(summary.get("row_count", math.nan))
        )

    transfer_gate_pass = all(_dataset_is_valid(dataset_id) for dataset_id in required_valid_datasets)
    tum_pressure_min_wins = int(gate_cfg.get("tum_pressure_min_wins_vs_dpvo", 0) or 0)
    tum_pressure_gate_pass = (
        not math.isfinite(pressure_row_count)
        or pressure_row_count <= 0.0
        or float(tum.get("pressure_wins_vs_reference", math.nan)) >= float(tum_pressure_min_wins)
    )
    selection_passed_gate = int(
        bool(
            tum_gate_pass
            and tum_pressure_gate_pass
            and transfer_gate_pass
            and full_dino_gate_pass
            and (
                math.isfinite(pure_tum_score)
                if selection_metric in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}
                else math.isfinite(weighted_dual_score)
            )
        )
    )

    output = {
        "tum_proxy_mean_ate": float(tum.get("external_mean_ate", math.nan)),
        "tum_proxy_mean_ate_associated": float(tum.get("external_mean_ate_associated", math.nan)),
        "tum_proxy_mean_coverage": float(tum.get("external_mean_coverage", math.nan)),
        "tum_proxy_mean_rpe_trans_rmse": float(tum.get("mean_rpe_trans_rmse", math.nan)),
        "tum_proxy_mean_rpe_rot_rmse": float(tum.get("mean_rpe_rot_rmse", math.nan)),
        "tum_proxy_mean_scale_correction": float(tum.get("mean_scale_correction", math.nan)),
        "tum_proxy_mean_scale_error_abs": float(tum.get("mean_scale_error_abs", math.nan)),
        "tum_proxy_mean_scale_error_abs_log": float(tum.get("mean_scale_error_abs_log", math.nan)),
        "tum_proxy_failed_count": float(tum.get("failed_count", math.nan)),
        "tum_proxy_finite_count": float(tum.get("finite_count", math.nan)),
        "tum_proxy_row_count": float(tum.get("row_count", math.nan)),
        "tum_proxy_wins_vs_dpvo": float(tum.get("wins_vs_reference", math.nan)),
        "tum_pressure_wins_vs_dpvo": float(tum.get("pressure_wins_vs_reference", math.nan)),
        "tum_pressure_mean_ate_associated": float(tum.get("pressure_mean_ate_associated", math.nan)),
        "tum_pressure_mean_coverage": float(tum.get("external_mean_coverage", math.nan)),
        "mean_grid_occupancy": float(tum.get("mean_grid_occupancy", math.nan)),
        "mean_track_age": float(tum.get("mean_track_age", math.nan)),
        "survival_rate_1": float(tum.get("survival_rate_1", math.nan)),
        "survival_rate_3": float(tum.get("survival_rate_3", math.nan)),
        "survival_rate_5": float(tum.get("survival_rate_5", math.nan)),
        "unstable_motion_proxy_fraction": float(tum.get("unstable_motion_proxy_fraction", math.nan)),
        "dino_patch_fraction": float(tum.get("dino_patch_fraction", math.nan)),
        "euroc_proxy_mean_ate": float(euroc.get("external_mean_ate", math.nan)),
        "euroc_proxy_mean_ate_associated": float(euroc.get("external_mean_ate_associated", math.nan)),
        "euroc_proxy_mean_coverage": float(euroc.get("external_mean_coverage", math.nan)),
        "euroc_proxy_mean_rpe_trans_rmse": float(euroc.get("mean_rpe_trans_rmse", math.nan)),
        "euroc_proxy_mean_rpe_rot_rmse": float(euroc.get("mean_rpe_rot_rmse", math.nan)),
        "euroc_proxy_mean_scale_correction": float(euroc.get("mean_scale_correction", math.nan)),
        "euroc_proxy_mean_scale_error_abs": float(euroc.get("mean_scale_error_abs", math.nan)),
        "euroc_proxy_mean_scale_error_abs_log": float(euroc.get("mean_scale_error_abs_log", math.nan)),
        "euroc_proxy_failed_count": float(euroc.get("failed_count", math.nan)),
        "euroc_proxy_finite_count": float(euroc.get("finite_count", math.nan)),
        "euroc_proxy_row_count": float(euroc.get("row_count", math.nan)),
        "euroc_proxy_wins_vs_dpvo": float(euroc.get("wins_vs_reference", math.nan)),
        "kitti_proxy_mean_ate": float(kitti.get("external_mean_ate", math.nan)),
        "kitti_proxy_mean_ate_associated": float(kitti.get("external_mean_ate_associated", math.nan)),
        "kitti_proxy_mean_coverage": float(kitti.get("external_mean_coverage", math.nan)),
        "kitti_proxy_mean_rpe_trans_rmse": float(kitti.get("mean_rpe_trans_rmse", math.nan)),
        "kitti_proxy_mean_rpe_rot_rmse": float(kitti.get("mean_rpe_rot_rmse", math.nan)),
        "kitti_proxy_mean_scale_correction": float(kitti.get("mean_scale_correction", math.nan)),
        "kitti_proxy_mean_scale_error_abs": float(kitti.get("mean_scale_error_abs", math.nan)),
        "kitti_proxy_mean_scale_error_abs_log": float(kitti.get("mean_scale_error_abs_log", math.nan)),
        "kitti_proxy_mean_kitti_trans_percent": float(kitti.get("mean_kitti_trans_percent", math.nan)),
        "kitti_proxy_mean_kitti_rot_deg_per_m": float(kitti.get("mean_kitti_rot_deg_per_m", math.nan)),
        "kitti_proxy_failed_count": float(kitti.get("failed_count", math.nan)),
        "kitti_proxy_finite_count": float(kitti.get("finite_count", math.nan)),
        "kitti_proxy_row_count": float(kitti.get("row_count", math.nan)),
        "live_weighted_rpe_trans_score": float(weighted_rpe_trans),
        "live_weighted_rpe_rot_score": float(weighted_rpe_rot),
        "live_weighted_scale_error_abs_log_score": float(weighted_scale),
        "live_tum_gate_pass": int(bool(tum_gate_pass)),
        "live_transfer_gate_pass": int(bool(transfer_gate_pass)),
        "selection_passed_gate": int(selection_passed_gate),
    }
    if selection_metric in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}:
        output.update(
            {
                "live_pure_tum_proxy_score": float(pure_tum_score),
                "live_dual_proxy_score": math.nan,
                "live_tri_proxy_score": math.nan,
                "best_dual_proxy_score": math.nan,
                "best_tum_proxy_wins_vs_dpvo": float(tum.get("wins_vs_reference", math.nan)),
                "best_euroc_proxy_wins_vs_dpvo": math.nan,
                "dual_selection_passed_gate": int(selection_passed_gate),
            }
        )
    elif selection_metric in {"dual_proxy_score", "live_dual_proxy_score"}:
        output.update(
            {
                "live_dual_proxy_score": float(weighted_dual_score),
                "live_pure_tum_proxy_score": math.nan,
                "best_dual_proxy_score": float(weighted_dual_score),
                "best_tum_proxy_wins_vs_dpvo": float(tum.get("wins_vs_reference", math.nan)),
                "best_euroc_proxy_wins_vs_dpvo": float(euroc.get("wins_vs_reference", math.nan)),
                "dual_selection_passed_gate": int(selection_passed_gate),
                "live_tri_proxy_score": math.nan,
            }
        )
    else:
        output.update(
            {
                "live_tri_proxy_score": float(weighted_ate),
                "live_dual_proxy_score": math.nan,
                "live_pure_tum_proxy_score": math.nan,
                "best_dual_proxy_score": math.nan,
                "best_tum_proxy_wins_vs_dpvo": math.nan,
                "best_euroc_proxy_wins_vs_dpvo": math.nan,
                "dual_selection_passed_gate": int(selection_passed_gate),
            }
        )
    return output


@torch.no_grad()
def _evaluate_external_tum_ate(
    model: DinoProposalFrontend,
    cfg: DinoDPVOConfig,
    *,
    dataset_root: Path | None,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    step: int,
    sequences: list[str],
    frontend_mode: str,
    run_tag: str,
    model_overrides: dict | None = None,
    eval_overrides: dict | None = None,
    collect_summary: bool = False,
) -> dict[str, float]:
    if dataset_root is None or not dataset_root.exists():
        return {
            "external_mean_ate": math.nan,
            "external_mean_ate_associated": math.nan,
            "external_mean_coverage": math.nan,
        }
    if not sequences:
        return {
            "external_mean_ate": math.nan,
            "external_mean_ate_associated": math.nan,
            "external_mean_coverage": math.nan,
        }

    runtime_cfg = _cfg_with_overrides(
        cfg,
        method_suffix=run_tag,
        feature_suffix=run_tag.upper(),
        frontend_mode=frontend_mode,
        model_overrides=model_overrides,
        eval_overrides=eval_overrides,
    )
    eval_dir = output_dir / "dev_eval" / f"step_{step:06d}" / run_tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    collect_diagnostics = bool(runtime_cfg.eval.get("collect_dev_diagnostics", runtime_cfg.model.get("collect_diagnostics", False)))
    write_patch_diagnostics = bool(runtime_cfg.eval.get("write_patch_diagnostics", collect_diagnostics))
    write_plots = bool(runtime_cfg.eval.get("write_dev_plots", True))
    should_collect_summary = bool(collect_diagnostics or collect_summary)
    diagnostics_summary_path = eval_dir / "diagnostics_summary.csv"
    patch_diagnostics_path = eval_dir / "patch_diagnostics.jsonl" if write_patch_diagnostics else None
    if should_collect_summary:
        init_diagnostics_outputs(diagnostics_summary_path, patch_diagnostics_path)

    metrics_rows: list[dict[str, float]] = []
    model.eval()
    dpvo_opts = _parse_dpvo_opts(runtime_cfg.eval.get("dpvo_opts", []))
    for seq in sequences:
        status, metrics = evaluate_dpvo_tum_sequence(
            dataset_root=dataset_root,
            sequence=seq,
            dpvo_root=dpvo_root,
            weights=dpvo_weights,
            config_path=dpvo_config,
            output_dir=eval_dir,
            max_dt=float(runtime_cfg.eval.get("max_dt", 0.02)),
            missing_penalty_m=float(runtime_cfg.eval.get("missing_penalty_m", 3.0)),
            min_coverage_ok=float(runtime_cfg.eval.get("coverage_gate", 0.99)),
            stride=int(runtime_cfg.eval.get("stride", 1)),
            backend_thresh=float(runtime_cfg.eval.get("backend_thresh", 18.0)),
            viz=False,
            opts=dpvo_opts,
            target_height=int(runtime_cfg.model.get("image_size", [240, 320])[0]),
            target_width=int(runtime_cfg.model.get("image_size", [240, 320])[1]),
            frontend_mode=str(frontend_mode),
            frontend_cfg=runtime_cfg,
            frontend=model,
            collect_diagnostics=should_collect_summary,
            diagnostics_summary_path=diagnostics_summary_path if should_collect_summary else None,
            patch_diagnostics_path=patch_diagnostics_path if should_collect_summary else None,
            feature_type=runtime_cfg.feature_type,
            write_plots=write_plots,
        )
        if metrics is not None and str(status) != "skipped_missing_sequence":
            metrics_rows.append(metrics)

    aggregate = _mean_external_metrics(metrics_rows)
    if should_collect_summary:
        aggregate.update(
            _read_diagnostics_summary_metrics(
                diagnostics_summary_path,
                patch_diagnostics_path=patch_diagnostics_path,
            )
        )
    return aggregate


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    mode = str(selection_metric).strip().lower()
    if mode in {"associated_ate", "ate_associated", "dpvo_style_ate"}:
        return float(metrics.get("external_mean_ate_associated", math.nan))
    if mode in {"coverage_aware_ate", "ate", "ate_rmse"}:
        return float(metrics.get("external_mean_ate", math.nan))
    if mode in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}:
        return float(metrics.get("live_pure_tum_proxy_score", math.nan))
    if mode in {"dual_proxy_score", "live_dual_proxy_score"}:
        return float(metrics.get("live_dual_proxy_score", math.nan))
    if mode in {"tri_proxy_score", "live_tri_proxy_score"}:
        return float(metrics.get("live_tri_proxy_score", math.nan))
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def _secondary_eval_enabled(cfg: DinoDPVOConfig) -> bool:
    sequences = cfg.eval.get("secondary_eval_sequences", []) or []
    derive_exclude = cfg.eval.get("secondary_from_primary_exclude_sequences", []) or []
    return bool(sequences or derive_exclude)


def _selection_mode(cfg: DinoDPVOConfig) -> str:
    return str(cfg.eval.get("selection_mode", "pure_only")).strip().lower()


def _mode_metric_bundle(dev_metrics: dict[str, float], mode: str) -> dict[str, float]:
    mode_key = str(mode).strip().lower()
    if mode_key == "hybrid":
        return {
            "ate": float(dev_metrics.get("hybrid_mean_ate", math.nan)),
            "assoc": float(dev_metrics.get("hybrid_mean_ate_associated", math.nan)),
            "coverage": float(dev_metrics.get("hybrid_mean_coverage", math.nan)),
            "lowtex_ate": float(dev_metrics.get("hybrid_lowtex_mean_ate", math.nan)),
            "lowtex_assoc": float(dev_metrics.get("hybrid_lowtex_mean_ate_associated", math.nan)),
            "lowtex_coverage": float(dev_metrics.get("hybrid_lowtex_mean_coverage", math.nan)),
        }
    return {
        "ate": float(dev_metrics.get("pure100_mean_ate", dev_metrics.get("external_mean_ate", math.nan))),
        "assoc": float(dev_metrics.get("pure100_mean_ate_associated", dev_metrics.get("external_mean_ate_associated", math.nan))),
        "coverage": float(dev_metrics.get("pure100_mean_coverage", dev_metrics.get("external_mean_coverage", math.nan))),
        "lowtex_ate": float(dev_metrics.get("lowtex_mean_ate", math.nan)),
        "lowtex_assoc": float(dev_metrics.get("lowtex_mean_ate_associated", math.nan)),
        "lowtex_coverage": float(dev_metrics.get("lowtex_mean_coverage", math.nan)),
    }


def _winner_mode_from_metrics(dev_metrics: dict[str, float], cfg: DinoDPVOConfig) -> str:
    selection_mode = _selection_mode(cfg)
    if selection_mode == "hybrid_only":
        return "hybrid"
    if selection_mode != "best_of_pure_hybrid":
        return "pure100"

    pure = _mode_metric_bundle(dev_metrics, "pure100")
    hybrid = _mode_metric_bundle(dev_metrics, "hybrid")
    coverage_gate = float(cfg.eval.get("coverage_gate", 0.99))
    secondary_required = _secondary_eval_enabled(cfg)
    secondary_coverage_gate = float(cfg.eval.get("secondary_coverage_gate", coverage_gate))

    def _valid(bundle: dict[str, float]) -> bool:
        primary_ok = math.isfinite(bundle["assoc"]) and math.isfinite(bundle["coverage"]) and bundle["coverage"] >= coverage_gate
        if not primary_ok:
            return False
        if not secondary_required:
            return True
        return (
            math.isfinite(bundle["lowtex_assoc"])
            and math.isfinite(bundle["lowtex_coverage"])
            and bundle["lowtex_coverage"] >= secondary_coverage_gate
        )

    pure_valid = _valid(pure)
    hybrid_valid = _valid(hybrid)
    if pure_valid and not hybrid_valid:
        return "pure100"
    if hybrid_valid and not pure_valid:
        return "hybrid"
    if not pure_valid and not hybrid_valid:
        return "pure100"

    pure_key = (
        pure["assoc"],
        pure["lowtex_assoc"] if math.isfinite(pure["lowtex_assoc"]) else math.inf,
        pure["ate"] if math.isfinite(pure["ate"]) else math.inf,
        0,
    )
    hybrid_key = (
        hybrid["assoc"],
        hybrid["lowtex_assoc"] if math.isfinite(hybrid["lowtex_assoc"]) else math.inf,
        hybrid["ate"] if math.isfinite(hybrid["ate"]) else math.inf,
        1,
    )
    return "pure100" if pure_key <= hybrid_key else "hybrid"


def _apply_winner_mode_metrics(dev_metrics: dict[str, float], cfg: DinoDPVOConfig) -> None:
    winner_mode = _winner_mode_from_metrics(dev_metrics, cfg)
    winner = _mode_metric_bundle(dev_metrics, winner_mode)
    dev_metrics["best_mode"] = winner_mode
    dev_metrics["best_pure_assoc"] = float(dev_metrics.get("pure100_mean_ate_associated", math.nan))
    dev_metrics["best_hybrid_assoc"] = float(dev_metrics.get("hybrid_mean_ate_associated", math.nan))
    dev_metrics["best_pure_lowtex_assoc"] = float(dev_metrics.get("lowtex_mean_ate_associated", math.nan))
    dev_metrics["best_hybrid_lowtex_assoc"] = float(dev_metrics.get("hybrid_lowtex_mean_ate_associated", math.nan))
    dev_metrics["external_mean_ate"] = winner["ate"]
    dev_metrics["external_mean_ate_associated"] = winner["assoc"]
    dev_metrics["external_mean_coverage"] = winner["coverage"]
    dev_metrics["lowtex_mean_ate"] = winner["lowtex_ate"]
    dev_metrics["lowtex_mean_ate_associated"] = winner["lowtex_assoc"]
    dev_metrics["lowtex_mean_coverage"] = winner["lowtex_coverage"]


def _checkpoint_selection_key(
    metrics: dict[str, float],
    cfg: DinoDPVOConfig,
    selection_metric: str,
) -> tuple[tuple[float, ...] | None, bool]:
    if str(selection_metric).strip().lower() in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}:
        score = _selection_score(metrics, selection_metric)
        if not (_truthy(metrics.get("selection_passed_gate")) and math.isfinite(score)):
            return None, False
        tum_wins_key = -float(metrics.get("tum_proxy_wins_vs_dpvo", math.nan))
        if not math.isfinite(tum_wins_key):
            tum_wins_key = math.inf
        pressure_wins_key = -float(metrics.get("tum_pressure_wins_vs_dpvo", math.nan))
        if not math.isfinite(pressure_wins_key):
            pressure_wins_key = math.inf
        return (
            float(score),
            float(metrics.get("tum_pressure_mean_ate_associated", math.inf)),
            tum_wins_key,
            pressure_wins_key,
            float(metrics.get("mean_track_age", math.inf)),
            float(metrics.get("repeated_patch_fraction", math.inf)),
        ), True
    if str(selection_metric).strip().lower() in {"dual_proxy_score", "live_dual_proxy_score"}:
        score = _selection_score(metrics, selection_metric)
        if not (_truthy(metrics.get("selection_passed_gate")) and math.isfinite(score)):
            return None, False
        euroc_wins_key = -float(metrics.get("euroc_proxy_wins_vs_dpvo", math.nan))
        if not math.isfinite(euroc_wins_key):
            euroc_wins_key = math.inf
        return (
            float(score),
            float(metrics.get("tum_pressure_mean_ate_associated", math.inf)),
            euroc_wins_key,
            float(metrics.get("live_weighted_rpe_trans_score", math.inf)),
            float(metrics.get("live_weighted_rpe_rot_score", math.inf)),
            float(metrics.get("live_weighted_scale_error_abs_log_score", math.inf)),
        ), True
    if str(selection_metric).strip().lower() in {"tri_proxy_score", "live_tri_proxy_score"}:
        score = _selection_score(metrics, selection_metric)
        if not (_truthy(metrics.get("selection_passed_gate")) and math.isfinite(score)):
            return None, False
        return (
            float(score),
            float(metrics.get("live_weighted_rpe_trans_score", math.inf)),
            float(metrics.get("live_weighted_rpe_rot_score", math.inf)),
            float(metrics.get("live_weighted_scale_error_abs_log_score", math.inf)),
            float(metrics.get("tum_pressure_mean_ate_associated", math.inf)),
        ), True

    coverage_gate = float(cfg.eval.get("coverage_gate", 0.99))
    primary_cov = float(metrics.get("external_mean_coverage", math.nan))
    score = _selection_score(metrics, selection_metric)
    if not (math.isfinite(score) and math.isfinite(primary_cov) and primary_cov >= coverage_gate):
        return None, False

    primary_ate = float(metrics.get("external_mean_ate", math.nan))
    repeated_patch_fraction = float(metrics.get("repeated_patch_fraction", math.nan))
    repeated_key = repeated_patch_fraction if math.isfinite(repeated_patch_fraction) else math.inf
    primary_ate_key = primary_ate if math.isfinite(primary_ate) else math.inf

    if not _secondary_eval_enabled(cfg):
        return (float(score), repeated_key, primary_ate_key), True

    secondary_cov = float(metrics.get("lowtex_mean_coverage", math.nan))
    secondary_assoc = float(metrics.get("lowtex_mean_ate_associated", math.nan))
    secondary_coverage_gate = float(cfg.eval.get("secondary_coverage_gate", coverage_gate))
    secondary_assoc_guardrail = float(cfg.eval.get("secondary_assoc_guardrail", math.inf))
    if not (
        math.isfinite(secondary_cov)
        and secondary_cov >= secondary_coverage_gate
        and math.isfinite(secondary_assoc)
        and secondary_assoc <= secondary_assoc_guardrail
    ):
        return None, False
    return (float(score), secondary_assoc, primary_ate_key, repeated_key), True


def _checkpoint_candidate_key(
    metrics: dict[str, float],
    selection_metric: str,
    *,
    tie_breakers: tuple[str, ...] = (),
) -> tuple[float, ...] | None:
    score = _selection_score(metrics, selection_metric)
    if not math.isfinite(score):
        return None

    key = [float(score)]
    for tie_breaker in tie_breakers:
        try:
            value = float(metrics.get(tie_breaker, math.nan))
        except Exception:
            value = math.nan
        key.append(value if math.isfinite(value) else math.inf)
    return tuple(key)


def _build_optimizer(model: DinoProposalFrontend, cfg: DinoDPVOConfig) -> torch.optim.Optimizer:
    base_lr = float(cfg.training.get("learning_rate", 2e-4))
    weight_decay = float(cfg.training.get("weight_decay", 1e-6))
    backbone_lr = float(cfg.training.get("dino_backbone_lr", base_lr))

    backbone_param_ids: set[int] = set()
    if hasattr(model, "backbone"):
        backbone_param_ids = {id(param) for param in model.backbone.parameters() if param.requires_grad}

    backbone_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_param_ids:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters found when building optimizer")

    return torch.optim.AdamW(param_groups)


def _lerp(start: float, end: float, alpha: float) -> float:
    alpha = float(max(0.0, min(1.0, alpha)))
    return (1.0 - alpha) * float(start) + alpha * float(end)


def _lr_scale_for_step(step: int, total_steps: int, schedule_cfg: dict[str, object] | None) -> float:
    schedule_cfg = dict(schedule_cfg or {})
    schedule_type = str(schedule_cfg.get("type", "constant")).strip().lower()
    if schedule_type in {"", "constant", "none"}:
        return 1.0

    total_steps = max(int(total_steps), 1)
    step = max(1, min(int(step), total_steps))
    progress = float(step - 1) / float(max(total_steps - 1, 1))
    min_scale = float(schedule_cfg.get("min_lr_scale", 0.1))
    min_scale = max(0.0, min_scale)

    if schedule_type == "linear_decay":
        return _lerp(1.0, min_scale, progress)

    if schedule_type == "cosine_decay":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_scale) + (1.0 - float(min_scale)) * cosine

    if schedule_type == "warmup_cosine":
        warmup_steps = max(1, int(schedule_cfg.get("warmup_steps", max(1, total_steps // 6))))
        warmup_start = float(schedule_cfg.get("warmup_start_scale", 0.25))
        if step <= warmup_steps:
            warmup_progress = float(step - 1) / float(max(warmup_steps - 1, 1))
            return _lerp(warmup_start, 1.0, warmup_progress)
        decay_progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, decay_progress))))
        return float(min_scale) + (1.0 - float(min_scale)) * cosine

    if schedule_type == "step_decay":
        step_size = max(1, int(schedule_cfg.get("step_size_steps", max(1, total_steps // 4))))
        gamma = float(schedule_cfg.get("gamma", 0.5))
        exponent = int((step - 1) // step_size)
        return max(min_scale, float(gamma) ** exponent)

    if schedule_type == "triangular":
        peak_step = max(1, min(int(schedule_cfg.get("peak_step", max(1, total_steps // 3))), total_steps))
        start_scale = float(schedule_cfg.get("start_scale", 0.5))
        peak_scale = float(schedule_cfg.get("peak_scale", 1.5))
        end_scale = float(schedule_cfg.get("end_scale", min_scale))
        if step <= peak_step:
            rise_progress = float(step - 1) / float(max(peak_step - 1, 1))
            return _lerp(start_scale, peak_scale, rise_progress)
        fall_progress = float(step - peak_step) / float(max(total_steps - peak_step, 1))
        return _lerp(peak_scale, end_scale, fall_progress)

    if schedule_type == "cosine_restart":
        cycle_steps = max(1, int(schedule_cfg.get("cycle_steps", max(1, total_steps // 2))))
        cycle_step = (step - 1) % cycle_steps
        cycle_progress = float(cycle_step) / float(max(cycle_steps - 1, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        return float(min_scale) + (1.0 - float(min_scale)) * cosine

    raise ValueError(f"Unsupported lr_schedule.type: {schedule_type}")


def _apply_lr_scale(
    optimizer: torch.optim.Optimizer,
    *,
    initial_lrs: list[float],
    scale: float,
) -> None:
    scale = max(0.0, float(scale))
    for group, base_lr in zip(optimizer.param_groups, initial_lrs):
        group["lr"] = float(base_lr) * scale


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the DINO-guided DPVO frontend on TartanAir.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--eval-dataset-root", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset-config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--dpvo-weights", required=True)
    ap.add_argument("--dpvo-config", required=True)
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--max-dev-windows", type=int, default=None)
    ap.add_argument("--tum-dataset-root", default=None)
    ap.add_argument("--euroc-dataset-root", default=None)
    ap.add_argument("--kitti-dataset-root", default=None)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--init-mode", choices=("strict", "partial"), default="strict")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--legacy-repro", action="store_true")
    args = ap.parse_args()

    cfg = load_dino_dpvo_config(args.config)
    import yaml

    subset_cfg_path = Path(args.subset_config).expanduser().resolve() if args.subset_config else (
        Path(__file__).resolve().parents[2] / "configs" / "tartanair_subset_v1.yaml"
    )
    subset_cfg = yaml.safe_load(subset_cfg_path.read_text(encoding="utf-8")) or {}
    subset_block = subset_cfg.get("subset", {})
    window_block = subset_cfg.get("windowing", {})

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "train_metrics.csv"
    default_tum_root = Path(__file__).resolve().parents[3] / "src" / "dino_slam3" / "data" / "tum_rgbd"
    tum_dataset_root = (
        Path(args.tum_dataset_root).expanduser().resolve()
        if args.tum_dataset_root
        else (default_tum_root if default_tum_root.exists() else None)
    )
    euroc_dataset_root = Path(args.euroc_dataset_root).expanduser().resolve() if args.euroc_dataset_root else None
    kitti_dataset_root = Path(args.kitti_dataset_root).expanduser().resolve() if args.kitti_dataset_root else None

    device = str(args.device or cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(args.seed if args.seed is not None else cfg.training.get("seed", 13))
    deterministic = bool(args.deterministic or cfg.training.get("deterministic", False))
    legacy_repro = bool(args.legacy_repro or cfg.training.get("legacy_repro", False))
    cfg.raw.setdefault("training", {})["device"] = device
    cfg.raw["training"]["seed"] = seed
    cfg.raw["training"]["deterministic"] = deterministic
    cfg.raw["training"]["legacy_repro"] = legacy_repro
    cfg.raw.setdefault("eval", {})
    if not legacy_repro:
        cfg.raw["eval"].setdefault("seed", seed)
    cfg.raw["eval"].setdefault("deterministic", deterministic)
    cfg.raw["eval"].setdefault("seed_runtime", not legacy_repro)
    _set_reproducibility(seed, deterministic=deterministic, legacy_repro=legacy_repro)

    live_proxy_cfg = dict(cfg.eval.get("live_proxy", {}) or {})
    live_proxy_enabled = bool(live_proxy_cfg.get("enabled", False))
    if live_proxy_enabled:
        for dataset_id, dataset_spec in (live_proxy_cfg.get("datasets", {}) or {}).items():
            sequences = dataset_spec.get("sequences", []) or []
            if not sequences:
                continue
            if str(dataset_id) == "tum" and (tum_dataset_root is None or not tum_dataset_root.exists()):
                raise FileNotFoundError("Live proxy evaluation requested, but TUM dataset root is unavailable")
            if str(dataset_id) == "euroc" and (euroc_dataset_root is None or not euroc_dataset_root.exists()):
                raise FileNotFoundError("Live proxy evaluation requested, but EuRoC dataset root is unavailable")
            if str(dataset_id) == "kitti" and (kitti_dataset_root is None or not kitti_dataset_root.exists()):
                raise FileNotFoundError("Live proxy evaluation requested, but KITTI dataset root is unavailable")

    image_size = tuple(int(v) for v in cfg.model.get("image_size", [240, 320]))
    train_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="train",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=int(window_block.get("n_frames", 4)),
        image_size=image_size,
        max_windows=int(args.max_train_windows or window_block.get("train_windows", 80000)),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=seed,
    )
    dev_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="dev",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=int(window_block.get("n_frames", 4)),
        image_size=image_size,
        max_windows=int(args.max_dev_windows or (window_block.get("train_windows", 80000) + window_block.get("dev_windows", 8000))),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=seed,
    )

    batch_size = int(cfg.training.get("batch_size", 4))
    num_workers = int(cfg.training.get("num_workers", 4))
    train_loader = _make_loader(train_ds, batch_size, num_workers, True, seed=seed, legacy_repro=legacy_repro)
    dev_loader = _make_loader(
        dev_ds,
        batch_size,
        max(0, num_workers // 2),
        False,
        seed=seed + 1,
        legacy_repro=legacy_repro,
    )

    model = build_dino_dpvo_frontend(cfg)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location="cpu")
        if args.init_mode == "partial":
            info = load_matching_state_dict(model, payload["state_dict"])
            print(f"Partially loaded initialization checkpoint: {init_path} ({info})")
        else:
            model.load_state_dict(payload["state_dict"], strict=True)
            print(f"Loaded initialization checkpoint: {init_path}")
    else:
        init_path = None

    gram_anchor_weight = float(cfg.losses.get("gram_anchor", 0.0))
    gram_anchor_teacher: DinoProposalFrontend | None = None
    if gram_anchor_weight > 0.0:
        if init_path is None:
            raise ValueError("losses.gram_anchor > 0 requires --init-checkpoint so the frozen teacher comes from the run seed")
        gram_anchor_teacher = deepcopy(model)
        gram_anchor_teacher.eval()
        for param in gram_anchor_teacher.parameters():
            param.requires_grad_(False)
    optimizer = _build_optimizer(model, cfg)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    lr_schedule_cfg = dict(cfg.training.get("lr_schedule", {}) or {})

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_HEADER)

    total_steps = int(cfg.training.get("train_steps", 10000))
    eval_every = int(cfg.training.get("eval_every", 1000))
    grad_clip = float(cfg.training.get("grad_clip", 5.0))
    best_external_key: tuple[float, ...] | None = None
    best_hybrid_external_key: tuple[float, ...] | None = None
    selection_metric = str(cfg.eval.get("selection_metric", "associated_ate"))
    selection_mode = _selection_mode(cfg)
    hybrid_only = selection_mode == "hybrid_only"
    save_best_pure100 = bool(cfg.eval.get("save_best_pure100", False))
    save_best_hybrid = bool(cfg.eval.get("save_best_hybrid", False) or selection_mode == "best_of_pure_hybrid")
    run_hybrid_dev_eval = bool(cfg.eval.get("run_hybrid_dev_eval", False) or selection_mode == "best_of_pure_hybrid")
    best_saved = False
    dpvo_root_path = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights_path = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config_path = Path(args.dpvo_config).expanduser().resolve()

    train_iter = iter(train_loader)
    for step in range(1, total_steps + 1):
        _apply_lr_scale(
            optimizer,
            initial_lrs=initial_lrs,
            scale=_lr_scale_for_step(step, total_steps, lr_schedule_cfg),
        )
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        batch["images"] = batch["images"].to(model.device)
        output = model(batch["images"])
        loss, metrics = _compute_frontend_losses(
            model,
            output,
            batch,
            cfg,
            step=step,
            gram_anchor_teacher=gram_anchor_teacher,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        _append_metrics_row(metrics_csv, step, "train", metrics)
        if step % int(cfg.training.get("log_every", 50)) == 0:
            print(
                f"[step {step}/{total_steps}] loss={metrics['loss']:.4f} "
                f"selector={metrics['selector_bce']:.4f} static={metrics['staticness_bce']:.4f} "
                f"offset={metrics['offset_l1']:.4f} coverage={metrics['coverage_kl']:.4f} "
                f"repulsion={metrics['proposal_repulsion']:.4f} "
                f"quality={metrics['mean_quality']:.3f}"
            )

        if step % eval_every != 0 and step != total_steps:
            continue

        ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
        torch.save(
            {
                "config_path": str(Path(args.config).expanduser().resolve()),
                "subset_config_path": str(subset_cfg_path),
                "state_dict": model.state_dict(),
                "step": step,
                "seed": seed,
                "deterministic": deterministic,
                "legacy_repro": legacy_repro,
                "rng_state": _capture_rng_state(),
            },
            ckpt_path,
        )

        dev_metrics = _evaluate_dev_loss(
            model,
            dev_loader,
            cfg,
            gram_anchor_teacher=gram_anchor_teacher,
        )
        dev_metrics.update(
            {
                "lowtex_mean_ate": math.nan,
                "lowtex_mean_ate_associated": math.nan,
                "lowtex_mean_coverage": math.nan,
            }
        )
        if live_proxy_enabled:
            live_metrics = _evaluate_live_proxy_selection(
                model,
                cfg,
                tum_dataset_root=tum_dataset_root,
                euroc_dataset_root=euroc_dataset_root,
                kitti_dataset_root=kitti_dataset_root,
                dpvo_root=dpvo_root_path,
                dpvo_weights=dpvo_weights_path,
                dpvo_config=dpvo_config_path,
                output_dir=output_dir,
                step=step,
            )
            best_mode = "hybrid" if hybrid_only else "pure100"
            primary_ate = live_metrics.get("tum_proxy_mean_ate", math.nan)
            primary_assoc = live_metrics.get("tum_proxy_mean_ate_associated", math.nan)
            primary_cov = live_metrics.get("tum_proxy_mean_coverage", math.nan)
            pressure_assoc = live_metrics.get("tum_pressure_mean_ate_associated", math.nan)
            pressure_cov = live_metrics.get("tum_pressure_mean_coverage", math.nan)
            dev_metrics.update(live_metrics)
            dev_metrics.update(
                {
                    "external_mean_ate": primary_ate,
                    "external_mean_ate_associated": primary_assoc,
                    "external_mean_coverage": primary_cov,
                    "lowtex_mean_ate": pressure_assoc,
                    "lowtex_mean_ate_associated": pressure_assoc,
                    "lowtex_mean_coverage": pressure_cov,
                    "pure100_mean_ate": primary_ate if not hybrid_only else math.nan,
                    "pure100_mean_ate_associated": primary_assoc if not hybrid_only else math.nan,
                    "pure100_mean_coverage": primary_cov if not hybrid_only else math.nan,
                    "hybrid_mean_ate": primary_ate if hybrid_only else math.nan,
                    "hybrid_mean_ate_associated": primary_assoc if hybrid_only else math.nan,
                    "hybrid_mean_coverage": primary_cov if hybrid_only else math.nan,
                    "hybrid_lowtex_mean_ate": pressure_assoc if hybrid_only else math.nan,
                    "hybrid_lowtex_mean_ate_associated": pressure_assoc if hybrid_only else math.nan,
                    "hybrid_lowtex_mean_coverage": pressure_cov if hybrid_only else math.nan,
                    "best_mode": best_mode,
                    "best_pure_assoc": primary_assoc if not hybrid_only else math.nan,
                    "best_hybrid_assoc": primary_assoc if hybrid_only else math.nan,
                    "best_pure_lowtex_assoc": pressure_assoc if not hybrid_only else math.nan,
                    "best_hybrid_lowtex_assoc": pressure_assoc if hybrid_only else math.nan,
                    "selection_metric": selection_metric,
                    "selection_passed_gate": int(live_metrics.get("selection_passed_gate", 0)),
                }
            )
        elif save_best_pure100 or save_best_hybrid:
            primary_sequences = [str(s) for s in cfg.eval.get("primary_eval_sequences", [])]
            pure100_sequences = [str(s) for s in cfg.eval.get("pure100_eval_sequences", primary_sequences)]
            pure100_frontend_mode = str(cfg.eval.get("pure100_frontend_mode", "dino_proposals"))
            pure100_model_overrides = dict(cfg.eval.get("pure100_model_overrides", {}))
            pure100_eval_overrides = dict(cfg.eval.get("pure100_eval_overrides", {}))
            secondary_from_primary_exclude = {
                str(seq)
                for seq in (cfg.eval.get("secondary_from_primary_exclude_sequences", []) or [])
                if str(seq).strip()
            }
            run_pure100_dev_eval = bool(save_best_pure100 or not hybrid_only)
            pure100_metrics = {
                "external_mean_ate": math.nan,
                "external_mean_ate_associated": math.nan,
                "external_mean_coverage": math.nan,
                "mean_unique_semantic_count_before_repeat": math.nan,
                "repeated_patch_fraction": math.nan,
                "mean_dedupe_radius_used": math.nan,
            }
            if run_pure100_dev_eval:
                pure100_metrics = _evaluate_external_tum_ate(
                    model,
                    cfg,
                    dataset_root=tum_dataset_root,
                    dpvo_root=dpvo_root_path,
                    dpvo_weights=dpvo_weights_path,
                    dpvo_config=dpvo_config_path,
                    output_dir=output_dir,
                    step=step,
                    sequences=pure100_sequences,
                    frontend_mode=pure100_frontend_mode,
                    run_tag="pure100",
                    model_overrides=pure100_model_overrides,
                    eval_overrides=pure100_eval_overrides,
                    collect_summary=bool(secondary_from_primary_exclude),
                )
                if _secondary_eval_enabled(cfg):
                    if secondary_from_primary_exclude:
                        secondary_metrics = _read_external_summary_metrics(
                            output_dir / "dev_eval" / f"step_{step:06d}" / "pure100" / "diagnostics_summary.csv",
                            exclude_sequences=secondary_from_primary_exclude,
                        )
                    else:
                        secondary_sequences = [str(s) for s in cfg.eval.get("secondary_eval_sequences", [])]
                        secondary_frontend_mode = str(cfg.eval.get("secondary_frontend_mode", pure100_frontend_mode))
                        secondary_model_overrides = dict(cfg.eval.get("secondary_model_overrides", pure100_model_overrides))
                        secondary_eval_overrides = dict(cfg.eval.get("secondary_eval_overrides", pure100_eval_overrides))
                        secondary_metrics = _evaluate_external_tum_ate(
                            model,
                            cfg,
                            dataset_root=tum_dataset_root,
                            dpvo_root=dpvo_root_path,
                            dpvo_weights=dpvo_weights_path,
                            dpvo_config=dpvo_config_path,
                            output_dir=output_dir,
                            step=step,
                            sequences=secondary_sequences,
                            frontend_mode=secondary_frontend_mode,
                            run_tag="lowtex",
                            model_overrides=secondary_model_overrides,
                            eval_overrides=secondary_eval_overrides,
                        )
                    dev_metrics.update(
                        {
                            "lowtex_mean_ate": secondary_metrics["external_mean_ate"],
                            "lowtex_mean_ate_associated": secondary_metrics["external_mean_ate_associated"],
                            "lowtex_mean_coverage": secondary_metrics["external_mean_coverage"],
                        }
                    )
            hybrid_metrics = {
                "external_mean_ate": math.nan,
                "external_mean_ate_associated": math.nan,
                "external_mean_coverage": math.nan,
            }
            hybrid_secondary_metrics = {
                "external_mean_ate": math.nan,
                "external_mean_ate_associated": math.nan,
                "external_mean_coverage": math.nan,
            }
            if save_best_hybrid or run_hybrid_dev_eval:
                hybrid_sequences = [str(s) for s in cfg.eval.get("hybrid_eval_sequences", primary_sequences)]
                hybrid_frontend_mode = str(cfg.eval.get("hybrid_frontend_mode", "dino_hybrid"))
                hybrid_model_overrides = dict(cfg.eval.get("hybrid_model_overrides", {}))
                hybrid_eval_overrides = dict(cfg.eval.get("hybrid_eval_overrides", {}))
                hybrid_metrics = _evaluate_external_tum_ate(
                    model,
                    cfg,
                    dataset_root=tum_dataset_root,
                    dpvo_root=dpvo_root_path,
                    dpvo_weights=dpvo_weights_path,
                    dpvo_config=dpvo_config_path,
                    output_dir=output_dir,
                    step=step,
                    sequences=hybrid_sequences,
                    frontend_mode=hybrid_frontend_mode,
                    run_tag="hybrid90_10",
                    model_overrides=hybrid_model_overrides,
                    eval_overrides=hybrid_eval_overrides,
                )
                if _secondary_eval_enabled(cfg):
                    if secondary_from_primary_exclude:
                        hybrid_secondary_metrics = _read_external_summary_metrics(
                            output_dir / "dev_eval" / f"step_{step:06d}" / "hybrid90_10" / "diagnostics_summary.csv",
                            exclude_sequences=secondary_from_primary_exclude,
                        )
                    else:
                        secondary_sequences = [str(s) for s in cfg.eval.get("secondary_eval_sequences", [])]
                        hybrid_secondary_metrics = _evaluate_external_tum_ate(
                            model,
                            cfg,
                            dataset_root=tum_dataset_root,
                            dpvo_root=dpvo_root_path,
                            dpvo_weights=dpvo_weights_path,
                            dpvo_config=dpvo_config_path,
                            output_dir=output_dir,
                            step=step,
                            sequences=secondary_sequences,
                            frontend_mode=hybrid_frontend_mode,
                            run_tag="hybrid_lowtex",
                            model_overrides=hybrid_model_overrides,
                            eval_overrides=hybrid_eval_overrides,
                        )
            dev_metrics.update(
                {
                    "external_mean_ate": pure100_metrics["external_mean_ate"],
                    "external_mean_ate_associated": pure100_metrics["external_mean_ate_associated"],
                    "external_mean_coverage": pure100_metrics["external_mean_coverage"],
                    "mean_unique_semantic_count_before_repeat": pure100_metrics.get("mean_unique_semantic_count_before_repeat", math.nan),
                    "repeated_patch_fraction": pure100_metrics.get("repeated_patch_fraction", math.nan),
                    "mean_dedupe_radius_used": pure100_metrics.get("mean_dedupe_radius_used", math.nan),
                    "pure100_mean_ate": pure100_metrics["external_mean_ate"],
                    "pure100_mean_ate_associated": pure100_metrics["external_mean_ate_associated"],
                    "pure100_mean_coverage": pure100_metrics["external_mean_coverage"],
                    "hybrid_mean_ate": hybrid_metrics["external_mean_ate"],
                    "hybrid_mean_ate_associated": hybrid_metrics["external_mean_ate_associated"],
                    "hybrid_mean_coverage": hybrid_metrics["external_mean_coverage"],
                    "hybrid_lowtex_mean_ate": hybrid_secondary_metrics["external_mean_ate"],
                    "hybrid_lowtex_mean_ate_associated": hybrid_secondary_metrics["external_mean_ate_associated"],
                    "hybrid_lowtex_mean_coverage": hybrid_secondary_metrics["external_mean_coverage"],
                }
            )
            if selection_mode in {"best_of_pure_hybrid", "hybrid_only"}:
                _apply_winner_mode_metrics(dev_metrics, cfg)
            else:
                dev_metrics.update(
                    {
                        "best_mode": "pure100",
                        "best_pure_assoc": pure100_metrics["external_mean_ate_associated"],
                        "best_hybrid_assoc": hybrid_metrics["external_mean_ate_associated"],
                        "best_pure_lowtex_assoc": dev_metrics.get("lowtex_mean_ate_associated", math.nan),
                        "best_hybrid_lowtex_assoc": hybrid_secondary_metrics["external_mean_ate_associated"],
                    }
                )
        else:
            external_metrics = _evaluate_external_ate(
                model,
                cfg,
                dataset_root=Path(args.eval_dataset_root or args.dataset_root).expanduser().resolve(),
                dpvo_root=dpvo_root_path,
                dpvo_weights=dpvo_weights_path,
                dpvo_config=dpvo_config_path,
                output_dir=output_dir,
                step=step,
            )
            dev_metrics.update(external_metrics)
            dev_metrics.update(
                {
                    "best_mode": "pure100",
                    "best_pure_assoc": dev_metrics.get("external_mean_ate_associated", math.nan),
                    "best_hybrid_assoc": math.nan,
                    "best_pure_lowtex_assoc": dev_metrics.get("lowtex_mean_ate_associated", math.nan),
                    "best_hybrid_lowtex_assoc": math.nan,
                    "hybrid_lowtex_mean_ate": math.nan,
                    "hybrid_lowtex_mean_ate_associated": math.nan,
                    "hybrid_lowtex_mean_coverage": math.nan,
                }
            )

        ext_ate = float(dev_metrics.get("external_mean_ate", math.nan))
        ext_ate_assoc = float(dev_metrics.get("external_mean_ate_associated", math.nan))
        ext_cov = float(dev_metrics.get("external_mean_coverage", math.nan))
        pure_key, passed_gate = _checkpoint_selection_key(dev_metrics, cfg, selection_metric)
        dev_metrics["selection_metric"] = selection_metric
        dev_metrics["selection_passed_gate"] = int(bool(passed_gate))
        _append_metrics_row(metrics_csv, step, "dev", dev_metrics)

        if passed_gate and pure_key is not None and (best_external_key is None or pure_key < best_external_key):
            best_external_key = pure_key
            best_saved = True
            payload = {
                "config_path": str(Path(args.config).expanduser().resolve()),
                "subset_config_path": str(subset_cfg_path),
                "state_dict": model.state_dict(),
                "step": step,
                "seed": seed,
                "deterministic": deterministic,
                "legacy_repro": legacy_repro,
                "rng_state": _capture_rng_state(),
                "dev_metrics": dev_metrics,
            }
            torch.save(payload, output_dir / "best.pt")
            if save_best_pure100:
                torch.save(payload, output_dir / "best_pure100.pt")
        if save_best_hybrid:
            hybrid_proxy = dict(dev_metrics) if live_proxy_enabled else {
                "external_mean_ate": float(dev_metrics.get("hybrid_mean_ate", math.nan)),
                "external_mean_ate_associated": float(dev_metrics.get("hybrid_mean_ate_associated", math.nan)),
                "external_mean_coverage": float(dev_metrics.get("hybrid_mean_coverage", math.nan)),
                "lowtex_mean_ate": float(dev_metrics.get("hybrid_lowtex_mean_ate", math.nan)),
                "lowtex_mean_ate_associated": float(dev_metrics.get("hybrid_lowtex_mean_ate_associated", math.nan)),
                "lowtex_mean_coverage": float(dev_metrics.get("hybrid_lowtex_mean_coverage", math.nan)),
            }
            hybrid_key, hybrid_passed_gate = _checkpoint_selection_key(hybrid_proxy, cfg, selection_metric)
            if hybrid_passed_gate and hybrid_key is not None and (
                best_hybrid_external_key is None or hybrid_key < best_hybrid_external_key
            ):
                best_hybrid_external_key = hybrid_key
                torch.save(
                    {
                        "config_path": str(Path(args.config).expanduser().resolve()),
                        "subset_config_path": str(subset_cfg_path),
                        "state_dict": model.state_dict(),
                        "step": step,
                        "seed": seed,
                        "deterministic": deterministic,
                        "legacy_repro": legacy_repro,
                        "dev_metrics": dev_metrics,
                    },
                    output_dir / "best_hybrid.pt",
                )
        print(
            f"[dev {step}] loss={dev_metrics.get('loss', math.nan):.4f} "
            f"mode={dev_metrics.get('best_mode', 'pure100')} "
            f"ext_ate={ext_ate:.4f} "
            f"ext_ate_assoc={ext_ate_assoc:.4f} "
            f"ext_cov={dev_metrics.get('external_mean_coverage', math.nan):.4f} "
            f"lowtex_assoc={dev_metrics.get('lowtex_mean_ate_associated', math.nan):.4f} "
            f"hyb_assoc={dev_metrics.get('hybrid_mean_ate_associated', math.nan):.4f}"
            + (
                ""
                if not live_proxy_enabled
                else (
                    f" tum_proxy_assoc={dev_metrics.get('tum_proxy_mean_ate_associated', math.nan):.4f}"
                    f" tum_pressure_assoc={dev_metrics.get('tum_pressure_mean_ate_associated', math.nan):.4f}"
                    f" euroc_proxy_assoc={dev_metrics.get('euroc_proxy_mean_ate_associated', math.nan):.4f}"
                    + (
                        ""
                        if not math.isfinite(float(dev_metrics.get("kitti_proxy_mean_ate_associated", math.nan)))
                        else f" kitti_proxy_assoc={dev_metrics.get('kitti_proxy_mean_ate_associated', math.nan):.4f}"
                    )
                    + (
                        f" pure_tum_score={dev_metrics.get('live_pure_tum_proxy_score', math.nan):.4f}"
                        if str(selection_metric).strip().lower() in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}
                        else ""
                    )
                    + (
                        f" dual_score={dev_metrics.get('live_dual_proxy_score', math.nan):.4f}"
                        if str(selection_metric).strip().lower() in {"dual_proxy_score", "live_dual_proxy_score"}
                        else f" tri_score={dev_metrics.get('live_tri_proxy_score', math.nan):.4f}"
                    )
                    + (
                        f" tum_wins={int(dev_metrics.get('tum_proxy_wins_vs_dpvo', math.nan))}"
                        if math.isfinite(float(dev_metrics.get("tum_proxy_wins_vs_dpvo", math.nan)))
                        else ""
                    )
                    + (
                        f" tum_pressure_wins={int(dev_metrics.get('tum_pressure_wins_vs_dpvo', math.nan))}"
                        if math.isfinite(float(dev_metrics.get("tum_pressure_wins_vs_dpvo", math.nan)))
                        else ""
                    )
                    + (
                        f" euroc_wins={int(dev_metrics.get('euroc_proxy_wins_vs_dpvo', math.nan))}"
                        if math.isfinite(float(dev_metrics.get("euroc_proxy_wins_vs_dpvo", math.nan)))
                        else ""
                    )
                    + (
                        f" pure_tum_gate={int(bool(dev_metrics.get('selection_passed_gate', 0)))}"
                        if str(selection_metric).strip().lower() in {"pure_tum_proxy_score", "live_pure_tum_proxy_score"}
                        else ""
                    )
                    + (
                        f" dual_gate={int(bool(dev_metrics.get('selection_passed_gate', 0)))}"
                        if str(selection_metric).strip().lower() in {"dual_proxy_score", "live_dual_proxy_score"}
                        else f" tri_gate={int(bool(dev_metrics.get('selection_passed_gate', 0)))}"
                    )
                )
            )
        )

    if not best_saved:
        payload = {
            "config_path": str(Path(args.config).expanduser().resolve()),
            "subset_config_path": str(subset_cfg_path),
            "state_dict": model.state_dict(),
            "step": total_steps,
            "seed": seed,
            "deterministic": deterministic,
            "legacy_repro": legacy_repro,
            "rng_state": _capture_rng_state(),
        }
        torch.save(payload, output_dir / "best.pt")
        if save_best_pure100:
            torch.save(payload, output_dir / "best_pure100.pt")
        if save_best_hybrid:
            torch.save(payload, output_dir / "best_hybrid.pt")

    print(f"Training complete. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
