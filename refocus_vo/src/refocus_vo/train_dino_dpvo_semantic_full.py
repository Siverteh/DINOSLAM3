from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import yaml

from dpvo.lietorch import SE3

from refocus_vo.data import TartanAirWindowDataset
from refocus_vo.dino_dpvo import DinoDPVOConfig, dense_gradient_offset_targets, load_dino_dpvo_config
from refocus_vo.dino_dpvo.frontend import load_matching_state_dict
from refocus_vo.dino_dpvo.semantic_vonet import DinoSemanticVONet
from refocus_vo.patchgraph.supervision import build_teacher_scores_from_fused, project_patch_centers, relative_pose_matrix


def _make_loader(dataset: TartanAirWindowDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def _append_metrics_row(path: Path, step: int, split: str, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(step),
                split,
                str(metrics.get("stage_name", "")),
                str(metrics.get("dpvo_init_mode", "")),
                f"{float(metrics.get('loss', math.nan)):.6f}",
                f"{float(metrics.get('dpvo_loss', math.nan)):.6f}",
                f"{float(metrics.get('flow_loss', math.nan)):.6f}",
                f"{float(metrics.get('pose_loss', math.nan)):.6f}",
                f"{float(metrics.get('effective_pose_weight', math.nan)):.6f}",
                f"{float(metrics.get('structure_only_flag', math.nan)):.6f}",
                f"{float(metrics.get('selector_bce', math.nan)):.6f}",
                f"{float(metrics.get('staticness_bce', math.nan)):.6f}",
                f"{float(metrics.get('offset_l1', math.nan)):.6f}",
                f"{float(metrics.get('coverage_kl', math.nan)):.6f}",
                f"{float(metrics.get('mean_teacher_score', math.nan)):.6f}",
                f"{float(metrics.get('mean_track_survival', math.nan)):.6f}",
                f"{float(metrics.get('mean_quality', math.nan)):.6f}",
                f"{float(metrics.get('semantic_fraction_target', math.nan)):.6f}",
                f"{float(metrics.get('semantic_fraction_realized', math.nan)):.6f}",
                f"{float(metrics.get('native_fraction_realized', math.nan)):.6f}",
                f"{float(metrics.get('pose_trans_err', math.nan)):.6f}",
                f"{float(metrics.get('pose_rot_err', math.nan)):.6f}",
            ]
        )


def kabsch_umeyama(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    n, _ = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    var_a = torch.mean((A - EA).norm(dim=1) ** 2)
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)
    del U, VT
    return var_a / torch.trace(torch.diag(D))


def _effective_pose_weight(cfg: DinoDPVOConfig, step: int) -> float:
    target = float(cfg.training.get("pose_weight_target", cfg.losses.get("pose_weight", 10.0)))
    start = int(cfg.training.get("pose_ramp_start_step", cfg.training.get("warmup_structure_only_steps", 3000)))
    end = int(cfg.training.get("pose_ramp_end_step", 12000))
    if int(step) <= start:
        return 0.0
    if int(step) >= end:
        return target
    alpha = float(int(step) - start) / max(float(end - start), 1.0)
    return target * alpha


def _semantic_fraction_target(
    cfg: DinoDPVOConfig,
    step: int,
    *,
    hold_fraction: float | None = None,
) -> float:
    if hold_fraction is not None:
        return float(hold_fraction)

    start = float(cfg.training.get("semantic_fraction_start", 0.25))
    mid = float(cfg.training.get("semantic_fraction_mid", 0.40))
    end = float(cfg.training.get("semantic_fraction_end", 0.50))
    ramp_1_end = int(cfg.training.get("mix_ramp_1_end_step", 5000))
    ramp_2_end = int(cfg.training.get("mix_ramp_2_end_step", 15000))
    hold_step = int(cfg.training.get("mix_hold_step", 22000))

    if int(step) <= ramp_1_end:
        return start
    if int(step) <= ramp_2_end:
        alpha = float(int(step) - ramp_1_end) / max(float(ramp_2_end - ramp_1_end), 1.0)
        return start + (mid - start) * alpha
    if int(step) <= hold_step:
        alpha = float(int(step) - ramp_2_end) / max(float(hold_step - ramp_2_end), 1.0)
        return mid + (end - mid) * alpha
    return end


def _build_optimizer(model: DinoSemanticVONet, cfg: DinoDPVOConfig) -> tuple[torch.optim.Optimizer, list[float]]:
    main_lr = float(cfg.training.get("learning_rate", 8e-5))
    backbone_lr = float(cfg.training.get("dino_backbone_lr", main_lr))
    weight_decay = float(cfg.training.get("weight_decay", 1e-6))

    backbone_model = model.patchify.semantic.backbone.model
    if backbone_model is None:
        raise RuntimeError("DINO backbone must be loaded before optimizer construction.")

    backbone_ids = {id(p) for p in backbone_model.parameters() if p.requires_grad}
    backbone_params = [p for p in backbone_model.parameters() if p.requires_grad]
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

    param_groups: list[dict[str, object]]
    if backbone_params:
        param_groups = [
            {"params": main_params, "lr": main_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
        max_lrs = [main_lr, backbone_lr]
    else:
        param_groups = [{"params": main_params, "lr": main_lr}]
        max_lrs = [main_lr]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer, max_lrs


def _load_dpvo_init_weights(
    model: DinoSemanticVONet,
    payload: dict[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, int]:
    def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

    mode = str(mode).lower()
    if mode == "none":
        return {"loaded": 0, "missing": 0, "unexpected": 0, "skipped": len(payload)}
    if mode == "update_only":
        best_info: dict[str, int] | None = None
        for prefix in ("module.update.", "update."):
            info = load_matching_state_dict(model.update, payload, prefix=prefix)
            if best_info is None or int(info["loaded"]) > int(best_info["loaded"]):
                best_info = info
            if int(info["loaded"]) > 0:
                return info
        return best_info or {"loaded": 0, "missing": 0, "unexpected": 0, "skipped": len(payload)}
    if mode == "full":
        info = load_matching_state_dict(model, payload)
        if int(info["loaded"]) == 0 and any(key.startswith("module.") for key in payload):
            info = load_matching_state_dict(model, _strip_prefix(payload, "module."))
        return info
    raise ValueError(f"Unsupported DPVO init mode: {mode}")


def _compute_track_survival_targets(
    model: DinoSemanticVONet,
    batch: dict[str, torch.Tensor],
    *,
    horizon: int,
    occlusion_tol_m: float = 0.15,
) -> torch.Tensor:
    teacher = model.patchify.semantic.teacher
    depths = batch["depths"].to(model.patchify.device)
    poses = batch["poses"].to(model.patchify.device)
    intrinsics = batch["intrinsics"].to(model.patchify.device)
    b, t, h, w = depths.shape
    patch_depths = []
    patch_valids = []
    for ti in range(t):
        d, valid = teacher.pool_depth(depths[:, ti])
        patch_depths.append(d)
        patch_valids.append(valid)
    patch_depths = torch.stack(patch_depths, dim=1)
    patch_valids = torch.stack(patch_valids, dim=1)

    ht = patch_depths.shape[-2]
    wt = patch_depths.shape[-1]
    centers_u, centers_v = teacher.patch_centers(ht, wt, model.patchify.device)
    centers = torch.stack([centers_u, centers_v], dim=-1).reshape(-1, 2)
    centers = centers.unsqueeze(0).expand(b, -1, -1)

    targets = torch.zeros((b, t, ht * wt), device=model.patchify.device, dtype=torch.float32)
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
                rel = relative_pose_matrix(poses[bi, ti], poses[bi, tj]).to(model.patchify.device)
                proj_xy, proj_z = project_patch_centers(centers[bi], src_depth, rel, K)
                in_bounds = (
                    src_valid
                    & (proj_z > 1e-6)
                    & (proj_xy[:, 0] >= 0.0)
                    & (proj_xy[:, 0] <= float(w - 1))
                    & (proj_xy[:, 1] >= 0.0)
                    & (proj_xy[:, 1] <= float(h - 1))
                )
                sampled_tgt_depth = teacher.sample_scalar(
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
    bt = selector_logits.shape[0] * selector_logits.shape[1]
    ht = selector_logits.shape[2]
    wt = selector_logits.shape[3]
    selector_map = torch.sigmoid(selector_logits).reshape(bt, 1, ht, wt)
    teacher_map = teacher_scores.reshape(bt, 1, ht, wt).detach()

    selector_cells = F.adaptive_max_pool2d(selector_map, (int(grid_rows), int(grid_cols))).flatten(1)
    teacher_cells = F.adaptive_max_pool2d(teacher_map, (int(grid_rows), int(grid_cols))).flatten(1)
    selector_dist = selector_cells / selector_cells.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_dist = teacher_cells / teacher_cells.sum(dim=1, keepdim=True).clamp_min(1e-6)
    uniform = torch.full_like(selector_dist, 1.0 / max(int(selector_dist.shape[1]), 1))
    mix = float(max(0.0, min(1.0, uniform_mix)))
    target_dist = ((1.0 - mix) * teacher_dist) + (mix * uniform)
    target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(selector_dist.clamp_min(1e-6).log(), target_dist, reduction="batchmean")


def _compute_semantic_aux_losses(
    model: DinoSemanticVONet,
    batch: dict[str, torch.Tensor],
    cfg: DinoDPVOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_output = model.patchify.last_output
    if semantic_output is None:
        raise RuntimeError("Semantic patchifier did not produce cached output for auxiliary losses.")

    teacher_scores = build_teacher_scores_from_fused(semantic_output.fused, batch, model.patchify.semantic.teacher)
    static_targets = _compute_track_survival_targets(
        model,
        batch,
        horizon=int(cfg.model.get("track_survival_horizon", 2)),
    )
    offset_targets = dense_gradient_offset_targets(batch["images"].to(model.patchify.device), model.patchify.semantic.patch_size)
    coverage_loss = _compute_coverage_regularizer(
        semantic_output.selector_logits,
        teacher_scores,
        grid_rows=int(cfg.model.get("coverage_grid_rows", cfg.model.get("hybrid_grid_rows", 6))),
        grid_cols=int(cfg.model.get("coverage_grid_cols", cfg.model.get("hybrid_grid_cols", 8))),
        uniform_mix=float(cfg.losses.get("coverage_uniform_mix", 0.15)),
    )

    selector_loss = F.binary_cross_entropy_with_logits(semantic_output.selector_logits, teacher_scores)
    static_loss = F.binary_cross_entropy_with_logits(semantic_output.staticness_logits, static_targets)

    offset_losses = []
    mean_quality = []
    for bi, frame_outputs in enumerate(semantic_output.observations):
        for ti, frame_output in enumerate(frame_outputs):
            if frame_output.proposal.patch_indices.numel() > 0:
                dense_target = offset_targets[bi, ti].reshape(-1, 2)
                target_offsets = dense_target[frame_output.proposal.patch_indices]
                offset_losses.append(F.l1_loss(frame_output.proposal.offset_xy, target_offsets))
            mean_quality.append(frame_output.qualities.mean())
    offset_loss = torch.stack(offset_losses).mean() if offset_losses else torch.zeros((), device=model.patchify.device)
    mean_quality_value = torch.stack(mean_quality).mean() if mean_quality else torch.zeros((), device=model.patchify.device)

    loss_weights = cfg.losses
    total = (
        float(loss_weights.get("selector_bce", 0.0)) * selector_loss
        + float(loss_weights.get("staticness_bce", 0.0)) * static_loss
        + float(loss_weights.get("offset_l1", 0.0)) * offset_loss
        + float(loss_weights.get("coverage_kl", 0.0)) * coverage_loss
    )
    metrics = {
        "selector_bce": float(selector_loss.item()),
        "staticness_bce": float(static_loss.item()),
        "offset_l1": float(offset_loss.item()),
        "coverage_kl": float(coverage_loss.item()),
        "mean_teacher_score": float(teacher_scores.mean().item()),
        "mean_track_survival": float(static_targets.mean().item()),
        "mean_quality": float(mean_quality_value.item()),
    }
    return total, metrics


def _compute_dpvo_training_loss(
    traj: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any, torch.Tensor]],
    *,
    patch_size: int,
    flow_weight: float,
    effective_pose_weight: float,
    structure_only: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=device)
    flow_terms = []
    pose_terms = []
    tr_metric = torch.zeros((), device=device)
    ro_metric = torch.zeros((), device=device)
    kl_metric = torch.zeros((), device=device)

    for i, (valid, coords_est, coords_gt, P1, P2, kl) in enumerate(traj):
        e = (coords_est - coords_gt).norm(dim=-1)
        e = e.reshape(-1, patch_size ** 2)[(valid > 0.5).reshape(-1)].min(dim=-1).values
        flow_mean = e.mean() if e.numel() > 0 else torch.zeros((), device=device)
        flow_terms.append(flow_mean)
        loss = loss + (float(flow_weight) * flow_mean)

        N = P1.shape[1]
        ii, jj = torch.meshgrid(
            torch.arange(N, device=device),
            torch.arange(N, device=device),
            indexing="ij",
        )
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        keep = ii != jj
        ii = ii[keep]
        jj = jj[keep]

        P1_inv = P1.inv()
        P2_inv = P2.inv()
        t1 = P1_inv.matrix()[..., :3, 3]
        t2 = P2_inv.matrix()[..., :3, 3]
        s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
        P1_scaled = P1_inv.scale(s.view(1, 1))

        dP = P1_scaled[:, ii].inv() * P1_scaled[:, jj]
        dG = P2_inv[:, ii].inv() * P2_inv[:, jj]
        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1).mean()
        ro = e1[..., 3:6].norm(dim=-1).mean()
        tr_metric = tr.detach()
        ro_metric = ro.detach()
        pose_term = tr + ro
        if (not structure_only) and i >= 2 and float(effective_pose_weight) > 0.0:
            pose_terms.append(pose_term)
            loss = loss + (float(effective_pose_weight) * pose_term)
        kl_metric = kl if isinstance(kl, torch.Tensor) else torch.as_tensor(float(kl), device=device)

    loss = loss + kl_metric
    flow_loss = torch.stack(flow_terms).mean() if flow_terms else torch.zeros((), device=device)
    pose_loss = torch.stack(pose_terms).mean() if pose_terms else torch.zeros((), device=device)
    metrics = {
        "dpvo_loss": float(loss.item()),
        "flow_loss": float(flow_loss.item()),
        "pose_loss": float(pose_loss.item()),
        "effective_pose_weight": float(effective_pose_weight),
        "structure_only_flag": float(1.0 if structure_only else 0.0),
        "pose_trans_err": float(tr_metric.item()),
        "pose_rot_err": float(ro_metric.item()),
    }
    return loss, metrics


@torch.no_grad()
def _evaluate_dev_loss(
    model: DinoSemanticVONet,
    loader: DataLoader,
    cfg: DinoDPVOConfig,
    *,
    step: int,
    semantic_fraction: float,
    native_fraction: float,
    stage_name: str,
    dpvo_init_mode: str,
    max_batches: int = 4,
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    device = model.patchify.device
    structure_only = int(step) <= int(cfg.training.get("warmup_structure_only_steps", 3000))
    effective_pose_weight = _effective_pose_weight(cfg, int(step))
    for batch in loader:
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        poses = SE3(batch["poses"].to(device)).inv()
        disps = torch.where(depths > 1e-6, depths.reciprocal(), torch.zeros_like(depths))
        traj, patch_output = model(
            images * 255.0,
            poses,
            disps,
            intrinsics,
            STEPS=int(cfg.training.get("dpvo_steps", 18)),
            structure_only=structure_only,
            frontend_mode="dino_hybrid",
            native_fraction=float(native_fraction),
            dino_fraction=float(semantic_fraction),
        )
        dpvo_loss, dpvo_metrics = _compute_dpvo_training_loss(
            traj,
            patch_size=model.P,
            flow_weight=float(cfg.losses.get("flow_weight", 0.1)),
            effective_pose_weight=float(effective_pose_weight),
            structure_only=structure_only,
            device=device,
        )
        aux_loss, aux_metrics = _compute_semantic_aux_losses(model, batch, cfg)
        total = dpvo_loss + aux_loss
        metrics = {
            "loss": float(total.item()),
            "stage_name": str(stage_name),
            "dpvo_init_mode": str(dpvo_init_mode),
            "semantic_fraction_target": float(semantic_fraction),
            "semantic_fraction_realized": float(patch_output.semantic_fraction_realized),
            "native_fraction_realized": float(patch_output.native_fraction_realized),
            **dpvo_metrics,
            **aux_metrics,
        }
        for key, value in metrics.items():
            if isinstance(value, str):
                agg[key] = value
            else:
                agg[key] = agg.get(key, 0.0) + float(value)
        count += 1
        if count >= int(max_batches):
            break
    if count == 0:
        return {"loss": math.nan}
    out: dict[str, float | str] = {}
    for key, value in agg.items():
        if isinstance(value, str):
            out[key] = value
        else:
            out[key] = value / count
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimental full DINO-semantic DPVO training.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset-config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--max-dev-windows", type=int, default=None)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--init-mode", choices=("strict", "partial"), default="strict")
    ap.add_argument("--frontend-init-checkpoint", default=None)
    ap.add_argument("--frontend-init-mode", choices=("strict", "partial"), default="partial")
    ap.add_argument("--dpvo-init-weights", default=None)
    ap.add_argument("--dpvo-init-mode", choices=("none", "update_only", "full"), default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_dino_dpvo_config(args.config)
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

    device = str(args.device or cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.raw.setdefault("training", {})["device"] = device
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    image_size = tuple(int(v) for v in cfg.model.get("image_size", [240, 320]))
    n_frames = int(cfg.training.get("n_frames", 15))
    train_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="train",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=n_frames,
        image_size=image_size,
        max_windows=int(args.max_train_windows or window_block.get("train_windows", 80000)),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=int(args.seed),
    )
    dev_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="dev",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=n_frames,
        image_size=image_size,
        max_windows=int(args.max_dev_windows or (window_block.get("train_windows", 80000) + window_block.get("dev_windows", 8000))),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=int(args.seed),
    )

    batch_size = int(cfg.training.get("batch_size", 1))
    num_workers = int(cfg.training.get("num_workers", 4))
    train_loader = _make_loader(train_ds, batch_size, num_workers, True)
    dev_loader = _make_loader(dev_ds, batch_size, max(0, num_workers // 2), False)

    model = DinoSemanticVONet(
        dino_name_or_path=str(cfg.model.get("dino_name_or_path")),
        dino_layers=tuple(int(v) for v in cfg.model.get("dino_layers", [6, 11])),
        dino_dtype=str(cfg.model.get("dino_dtype", "bf16")),
        image_size=image_size,
        dino_patch_size=int(cfg.model.get("dino_patch_size", 16)),
        dpvo_patch_size=int(cfg.model.get("dpvo_patch_size", 3)),
        semantic_candidate_pool=int(cfg.model.get("semantic_candidate_pool", 128)),
        semantic_patch_budget=int(cfg.model.get("semantic_patch_budget", 80)),
        max_nodes_per_object_ratio=float(cfg.model.get("max_nodes_per_object_ratio", 0.20)),
        k_mutual_neighbors=int(cfg.model.get("k_mutual_neighbors", 4)),
        local_patch_dim=int(cfg.model.get("local_patch_dim", 64)),
        dpvo_dim=int(cfg.model.get("dpvo_dim", 384)),
        corr_dim=int(cfg.model.get("corr_dim", 128)),
        static_score_weight=float(cfg.model.get("static_score_weight", 0.35)),
        quality_floor=float(cfg.model.get("quality_floor", 0.05)),
        use_offset_refinement=bool(cfg.model.get("use_offset_refinement", True)),
        enable_gradient_branch=bool(cfg.model.get("enable_gradient_branch", False)),
        gradient_branch_dim=int(cfg.model.get("gradient_branch_dim", 32)),
        dino_unfreeze_blocks=int(cfg.model.get("dino_unfreeze_blocks", 0)),
        hybrid_grid_rows=int(cfg.model.get("hybrid_grid_rows", 6)),
        hybrid_grid_cols=int(cfg.model.get("hybrid_grid_cols", 8)),
        max_semantic_per_cell=int(cfg.model.get("max_semantic_per_cell", 1)),
        dedupe_radius_px=float(cfg.model.get("dedupe_radius_px", 8.0)),
        device=device,
    ).to(torch.device(device))

    if args.init_checkpoint:
        init_ckpt = Path(args.init_checkpoint).expanduser().resolve()
        payload = torch.load(init_ckpt, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if args.init_mode == "partial":
            info = load_matching_state_dict(model, state_dict)
            print(f"Partially loaded semantic-full init from {init_ckpt} ({info})")
        else:
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded semantic-full init from {init_ckpt}")

    dpvo_init_mode = str(args.dpvo_init_mode or cfg.training.get("dpvo_init_mode", "full")).lower()
    if args.dpvo_init_weights and dpvo_init_mode != "none":
        dpvo_path = Path(args.dpvo_init_weights).expanduser().resolve()
        payload = torch.load(dpvo_path, map_location="cpu")
        info = _load_dpvo_init_weights(model, payload, mode=dpvo_init_mode)
        print(f"Loaded DPVO init weights from {dpvo_path} with mode={dpvo_init_mode} ({info})")

    if args.frontend_init_checkpoint:
        frontend_ckpt = Path(args.frontend_init_checkpoint).expanduser().resolve()
        payload = torch.load(frontend_ckpt, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if args.frontend_init_mode == "partial":
            info = load_matching_state_dict(model.patchify.semantic, state_dict)
            print(f"Partially loaded DINO semantic frontend init from {frontend_ckpt} ({info})")
        else:
            model.patchify.semantic.load_state_dict(state_dict, strict=True)
            print(f"Loaded DINO semantic frontend init from {frontend_ckpt}")

    optimizer, max_lrs = _build_optimizer(model, cfg)
    total_steps = int(cfg.training.get("train_steps", 240000))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0],
        total_steps=total_steps,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "split",
                "stage_name",
                "dpvo_init_mode",
                "loss",
                "dpvo_loss",
                "flow_loss",
                "pose_loss",
                "effective_pose_weight",
                "structure_only_flag",
                "selector_bce",
                "staticness_bce",
                "offset_l1",
                "coverage_kl",
                "mean_teacher_score",
                "mean_track_survival",
                "mean_quality",
                "semantic_fraction_target",
                "semantic_fraction_realized",
                "native_fraction_realized",
                "pose_trans_err",
                "pose_rot_err",
            ]
        )

    eval_every = int(cfg.training.get("eval_every", 1000))
    grad_clip = float(cfg.training.get("grad_clip", 10.0))
    warmup_structure_only_steps = int(cfg.training.get("warmup_structure_only_steps", 1000))
    best_dev = float("inf")
    stage_name = str(cfg.training.get("stage_name", cfg.method_id))
    gate_reference_dpvo_loss: float | None = None
    semantic_fraction_hold: float | None = None
    gate_checked = False

    train_iter = iter(train_loader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        device_obj = model.patchify.device
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        intrinsics = batch["intrinsics"].to(device_obj)
        poses = SE3(batch["poses"].to(device_obj)).inv()
        disps = torch.where(depths > 1e-6, depths.reciprocal(), torch.zeros_like(depths))
        structure_only = step <= warmup_structure_only_steps
        effective_pose_weight = _effective_pose_weight(cfg, step)
        semantic_fraction_target = _semantic_fraction_target(cfg, step, hold_fraction=semantic_fraction_hold)
        native_fraction_target = max(0.0, 1.0 - float(semantic_fraction_target))

        traj, patch_output = model(
            images * 255.0,
            poses,
            disps,
            intrinsics,
            STEPS=int(cfg.training.get("dpvo_steps", 18)),
            structure_only=structure_only,
            frontend_mode="dino_hybrid",
            native_fraction=float(native_fraction_target),
            dino_fraction=float(semantic_fraction_target),
        )
        dpvo_loss, dpvo_metrics = _compute_dpvo_training_loss(
            traj,
            patch_size=model.P,
            flow_weight=float(cfg.losses.get("flow_weight", 0.1)),
            effective_pose_weight=float(effective_pose_weight),
            structure_only=structure_only,
            device=device_obj,
        )
        aux_loss, aux_metrics = _compute_semantic_aux_losses(model, batch, cfg)
        loss = dpvo_loss + aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        train_metrics = {
            "loss": float(loss.item()),
            "stage_name": stage_name,
            "dpvo_init_mode": dpvo_init_mode,
            "semantic_fraction_target": float(semantic_fraction_target),
            "semantic_fraction_realized": float(patch_output.semantic_fraction_realized),
            "native_fraction_realized": float(patch_output.native_fraction_realized),
            **dpvo_metrics,
            **aux_metrics,
        }
        _append_metrics_row(metrics_csv, step, "train", train_metrics)

        if step % int(cfg.training.get("log_every", 50)) == 0:
            print(
                f"[step {step}/{total_steps}] loss={train_metrics['loss']:.4f} "
                f"dpvo={train_metrics['dpvo_loss']:.4f} flow={train_metrics['flow_loss']:.4f} "
                f"pose={train_metrics['pose_loss']:.4f} sel={train_metrics['selector_bce']:.4f} "
                f"static={train_metrics['staticness_bce']:.4f} cov={train_metrics['coverage_kl']:.4f} "
                f"pose_w={train_metrics['effective_pose_weight']:.4f} sem={train_metrics['semantic_fraction_realized']:.3f}"
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
            },
            ckpt_path,
        )

        dev_metrics = _evaluate_dev_loss(
            model,
            dev_loader,
            cfg,
            step=step,
            semantic_fraction=float(semantic_fraction_target),
            native_fraction=float(native_fraction_target),
            stage_name=stage_name,
            dpvo_init_mode=dpvo_init_mode,
            max_batches=int(cfg.training.get("max_eval_batches", 4)),
        )
        _append_metrics_row(metrics_csv, step, "dev", dev_metrics)

        if step == int(cfg.training.get("mix_ramp_1_end_step", 5000)):
            gate_reference_dpvo_loss = float(dev_metrics.get("dpvo_loss", math.nan))
        gate_step = int(cfg.training.get("pose_ramp_end_step", 12000))
        gate_improvement = float(cfg.training.get("stage2_stability_gate_improvement", 0.10))
        apply_stability_gate = bool(cfg.training.get("apply_stability_gate", False))
        if apply_stability_gate and (not gate_checked) and step >= gate_step and gate_reference_dpvo_loss is not None:
            current_dpvo_loss = float(dev_metrics.get("dpvo_loss", math.nan))
            if (not math.isfinite(current_dpvo_loss)) or current_dpvo_loss > gate_reference_dpvo_loss * (1.0 - gate_improvement):
                semantic_fraction_hold = float(semantic_fraction_target)
                print(
                    f"[gate {step}] holding semantic fraction at {semantic_fraction_hold:.3f} "
                    f"because dev dpvo_loss={current_dpvo_loss:.4f} did not improve enough from {gate_reference_dpvo_loss:.4f}"
                )
            gate_checked = True

        dev_loss = float(dev_metrics.get("loss", math.nan))
        if math.isfinite(dev_loss) and dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(
                {
                    "config_path": str(Path(args.config).expanduser().resolve()),
                    "subset_config_path": str(subset_cfg_path),
                    "state_dict": model.state_dict(),
                    "step": step,
                    "dev_metrics": dev_metrics,
                },
                output_dir / "best.pt",
            )
        print(
            f"[dev {step}] loss={dev_loss:.4f} dpvo={dev_metrics.get('dpvo_loss', math.nan):.4f} "
            f"pose_w={dev_metrics.get('effective_pose_weight', math.nan):.4f} "
            f"sem={dev_metrics.get('semantic_fraction_realized', math.nan):.3f}"
        )

    if not (output_dir / "best.pt").exists():
        torch.save(
            {
                "config_path": str(Path(args.config).expanduser().resolve()),
                "subset_config_path": str(subset_cfg_path),
                "state_dict": model.state_dict(),
                "step": total_steps,
            },
            output_dir / "best.pt",
        )

    print(f"Experimental semantic full training complete. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
