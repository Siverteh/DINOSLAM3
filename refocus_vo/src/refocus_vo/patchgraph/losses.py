from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import WindowPrediction
from .supervision import build_teacher_scores_from_fused, project_patch_centers, relative_pose_matrix, rotvec_from_matrix
from .teacher import DinoPatchTeacher


def compute_patchgraph_losses(
    prediction: WindowPrediction,
    batch: dict[str, Any],
    teacher: DinoPatchTeacher,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    device = prediction.fused.device
    depths = batch["depths"].to(device)
    poses = batch["poses"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    teacher_scores = prediction.teacher_scores
    if teacher_scores is None:
        teacher_scores = build_teacher_scores_from_fused(prediction.fused, batch, teacher)

    selector_loss = F.binary_cross_entropy_with_logits(
        prediction.selector_logits.squeeze(2),
        teacher_scores.detach(),
    )

    persistence_loss = torch.zeros((), device=device)
    edge_loss = torch.zeros((), device=device)
    confidence_loss = torch.zeros((), device=device)
    offset_loss = torch.zeros((), device=device)
    rotation_loss = torch.zeros((), device=device)
    translation_dir_loss = torch.zeros((), device=device)
    translation_scale_loss = torch.zeros((), device=device)
    coverage_loss = torch.zeros((), device=device)
    semantic_loss = torch.zeros((), device=device)
    mean_selected_teacher_score = torch.zeros((), device=device)
    pair_count = 0
    pose_count = 0
    teacher_patch_count = 0
    coverage_count = 0
    total_matches = 0

    for bi, seq_preds in enumerate(prediction.frame_predictions):
        for ti, frame_obs in enumerate(prediction.observations[bi]):
            obj_ids = frame_obs.proposal.object_ids
            if obj_ids.numel() == 0:
                continue
            uniq, counts = torch.unique(obj_ids, return_counts=True)
            ratios = counts.float() / counts.float().sum().clamp_min(1.0)
            coverage_loss = coverage_loss + torch.clamp(ratios.max() - teacher.max_nodes_per_object_ratio, min=0.0)
            coverage_count += 1

            selected_teacher = teacher_scores[bi, ti].reshape(-1)
            patch_scores = selected_teacher[frame_obs.proposal.patch_indices]
            mean_selected_teacher_score = mean_selected_teacher_score + patch_scores.mean()
            teacher_patch_count += 1

        for frame_pred in seq_preds:
            ti = int(frame_pred.frame_idx)
            gt_adj_rel = relative_pose_matrix(poses[bi, ti - 1], poses[bi, ti]).to(device)
            gt_rot = rotvec_from_matrix(gt_adj_rel[:3, :3]).to(device)
            pred_rot = frame_pred.pose_vec[:3]
            pred_t = frame_pred.pose_vec[3:]
            gt_t = gt_adj_rel[:3, 3]
            valid_depth = depths[bi, ti - 1][depths[bi, ti - 1] > 1e-6]
            if valid_depth.numel() == 0:
                depth_scale = torch.as_tensor(1.0, device=device, dtype=torch.float32)
            else:
                depth_scale = valid_depth.median()
            gt_t_norm = gt_t / depth_scale.clamp_min(1e-3)
            rotation_loss = rotation_loss + F.l1_loss(pred_rot, gt_rot)
            translation_dir_loss = translation_dir_loss + F.l1_loss(
                F.normalize(pred_t.unsqueeze(0), dim=1, eps=1e-6),
                F.normalize(gt_t_norm.unsqueeze(0), dim=1, eps=1e-6),
            )
            translation_scale_loss = translation_scale_loss + F.l1_loss(
                pred_t.norm().unsqueeze(0),
                gt_t_norm.norm().unsqueeze(0),
            )
            pose_count += 1

            tgt_obs = prediction.observations[bi][ti]
            patch_half = float(teacher.patch_size) * 0.5
            for pair in frame_pred.incoming_pairs:
                src_obs = prediction.observations[bi][int(pair.src_frame_idx)]
                rel = relative_pose_matrix(poses[bi, int(pair.src_frame_idx)], poses[bi, ti]).to(device)

                depth_patch, depth_valid = teacher.pool_depth(depths[bi, int(pair.src_frame_idx)].unsqueeze(0))
                depth_vals = depth_patch[0].reshape(-1)[src_obs.proposal.patch_indices]
                valid = depth_valid[0].reshape(-1)[src_obs.proposal.patch_indices]

                projected_xy, projected_z = project_patch_centers(
                    src_obs.proposal.pixel_xy,
                    depth_vals.clamp_min(1e-3),
                    rel,
                    intrinsics[bi],
                )
                valid = valid & (projected_z > 1e-6)

                if pair.src_indices.numel() == 0:
                    pair_count += 1
                    continue

                matched_proj = projected_xy[pair.src_indices]
                matched_tgt_xy = tgt_obs.proposal.pixel_xy[pair.tgt_indices]
                dist = torch.norm(matched_tgt_xy - matched_proj, dim=1)
                edge_target = (dist <= float(teacher.patch_size)).float() * valid[pair.src_indices].float()

                persistence_loss = persistence_loss + dist[edge_target > 0.5].mean() if torch.any(edge_target > 0.5) else persistence_loss
                edge_loss = edge_loss + F.binary_cross_entropy_with_logits(pair.confidence_logits, edge_target)
                confidence_loss = confidence_loss + F.binary_cross_entropy_with_logits(pair.confidence_logits, edge_target)
                semantic_loss = semantic_loss + (1.0 - pair.similarity[edge_target > 0.5].mean()) if torch.any(edge_target > 0.5) else semantic_loss
                total_matches += int(pair.src_indices.numel())

                if torch.any(edge_target > 0.5):
                    coarse_tgt_xy = tgt_obs.proposal.coarse_pixel_xy[pair.tgt_indices[edge_target > 0.5]]
                    gt_offset = (matched_proj[edge_target > 0.5] - coarse_tgt_xy).clamp(-patch_half, patch_half)
                    pred_offset = tgt_obs.proposal.offset_xy[pair.tgt_indices[edge_target > 0.5]]
                    offset_loss = offset_loss + F.l1_loss(pred_offset, gt_offset)

                pair_count += 1

    pair_denom = max(1, pair_count)
    pose_denom = max(1, pose_count)
    teacher_denom = max(1, teacher_patch_count)
    persistence_loss = persistence_loss / pair_denom
    edge_loss = edge_loss / pair_denom
    confidence_loss = confidence_loss / pair_denom
    offset_loss = offset_loss / pair_denom
    rotation_loss = rotation_loss / pose_denom
    translation_dir_loss = translation_dir_loss / pose_denom
    translation_scale_loss = translation_scale_loss / pose_denom
    coverage_loss = coverage_loss / max(1, coverage_count)
    semantic_loss = semantic_loss / pair_denom
    mean_selected_teacher_score = mean_selected_teacher_score / teacher_denom

    total = (
        loss_weights.get("selector_bce", 1.0) * selector_loss
        + loss_weights.get("persistence_l1", 0.25) * persistence_loss
        + loss_weights.get("edge_valid_bce", 0.5) * edge_loss
        + loss_weights.get("confidence_bce", 0.5) * confidence_loss
        + loss_weights.get("offset_l1", 0.0) * offset_loss
        + loss_weights.get("rotation_l1", 2.0) * rotation_loss
        + loss_weights.get("translation_dir_l1", 1.0) * translation_dir_loss
        + loss_weights.get("translation_scale_l1", 0.5) * translation_scale_loss
        + loss_weights.get("coverage_reg", 0.1) * coverage_loss
        + loss_weights.get("semantic_consistency", 0.1) * semantic_loss
    )

    metrics = {
        "loss": float(total.detach().cpu().item()),
        "selector_bce": float(selector_loss.detach().cpu().item()),
        "persistence_l1": float(persistence_loss.detach().cpu().item()),
        "edge_valid_bce": float(edge_loss.detach().cpu().item()),
        "confidence_bce": float(confidence_loss.detach().cpu().item()),
        "offset_l1": float(offset_loss.detach().cpu().item()),
        "rotation_l1": float(rotation_loss.detach().cpu().item()),
        "translation_dir_l1": float(translation_dir_loss.detach().cpu().item()),
        "translation_scale_l1": float(translation_scale_loss.detach().cpu().item()),
        "coverage_reg": float(coverage_loss.detach().cpu().item()),
        "semantic_consistency": float(semantic_loss.detach().cpu().item()),
        "mean_selected_teacher_score": float(mean_selected_teacher_score.detach().cpu().item()),
        "matches_per_pair": float(float(total_matches) / float(max(1, pair_count))),
    }
    return total, metrics, teacher_scores.detach()
