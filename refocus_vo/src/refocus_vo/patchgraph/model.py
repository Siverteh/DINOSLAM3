from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from refocus_vo.backbones.dinov3 import DinoTokens, DinoV3Backbone

from .supervision import build_teacher_scores_from_fused
from .teacher import DinoPatchTeacher, PseudoObjectPatchProposal


@dataclass
class FrameObservation:
    fused: torch.Tensor
    local_map: torch.Tensor
    selector_logits: torch.Tensor
    patch_score: torch.Tensor
    object_ids: torch.Tensor
    proposal: PseudoObjectPatchProposal


@dataclass
class PairPrediction:
    src_frame_idx: int
    tgt_frame_idx: int
    lag: int
    src_indices: torch.Tensor
    tgt_indices: torch.Tensor
    similarity: torch.Tensor
    confidence_logits: torch.Tensor
    edge_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    pose_vec: torch.Tensor


@dataclass
class TargetFramePrediction:
    frame_idx: int
    incoming_pairs: list[PairPrediction]
    pose_vec: torch.Tensor
    hidden_state: torch.Tensor | None = None


@dataclass
class WindowPrediction:
    fused: torch.Tensor
    local_map: torch.Tensor
    selector_logits: torch.Tensor
    observations: list[list[FrameObservation]]
    frame_predictions: list[list[TargetFramePrediction]]
    teacher_scores: torch.Tensor | None = None


class SmallLocalEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoPatchGraphVO(nn.Module):
    def __init__(
        self,
        *,
        dino_name_or_path: str,
        dino_layers: Sequence[int] = (6, 11),
        dino_dtype: str = "bf16",
        patch_size: int = 16,
        num_patches: int = 64,
        max_nodes_per_object_ratio: float = 0.20,
        k_mutual_neighbors: int = 4,
        dino_hidden_dim: int = 192,
        local_patch_dim: int = 64,
        edge_dim: int = 192,
        graph_hidden_dim: int = 256,
        min_match_cosine: float = 0.40,
        max_history: int = 3,
        lag_embedding_dim: int = 16,
        enable_offset_refinement: bool = False,
        use_multiframe_graph: bool = False,
        device: str | None = None,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.layer_indices = tuple(int(v) for v in dino_layers)
        self.num_patches = int(num_patches)
        self.min_match_cosine = float(min_match_cosine)
        self.max_history = max(1, int(max_history))
        self.enable_offset_refinement = bool(enable_offset_refinement)
        self.use_multiframe_graph = bool(use_multiframe_graph)
        self.device_name = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.backbone = DinoV3Backbone(
            name_or_path=str(dino_name_or_path),
            patch_size=self.patch_size,
            freeze=True,
            dtype=str(dino_dtype),
        )
        self.backbone.load()

        self.teacher = DinoPatchTeacher(
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            max_nodes_per_object_ratio=float(max_nodes_per_object_ratio),
            k_mutual_neighbors=int(k_mutual_neighbors),
        )

        embed_dim = int(self.backbone.embed_dim or 384)
        self.selector_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )
        self.local_encoder = SmallLocalEncoder(local_patch_dim)
        self.desc_head = nn.Conv2d(embed_dim + local_patch_dim, dino_hidden_dim, kernel_size=1)
        self.offset_head = MLP(dino_hidden_dim + local_patch_dim, max(64, local_patch_dim), 2)
        edge_in_dim = (2 * dino_hidden_dim) + (2 * local_patch_dim) + 6
        self.edge_mlp = MLP(edge_in_dim, edge_dim, edge_dim)
        self.conf_head = nn.Linear(edge_dim, 1)
        self.pair_pose_head = MLP(edge_dim, graph_hidden_dim, 6)
        self.lag_embedding = nn.Embedding(self.max_history + 1, int(lag_embedding_dim))
        self.frame_updater = nn.GRUCell(edge_dim + int(lag_embedding_dim), graph_hidden_dim)
        self.frame_pose_head = MLP(graph_hidden_dim, graph_hidden_dim, 6)

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def _encode_backbone(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        b, t, _, h, w = images.shape
        flat = images.reshape(b * t, 3, h, w)
        with torch.inference_mode():
            out: DinoTokens = self.backbone(
                flat,
                return_hidden_states=True,
                hidden_state_indices=self.layer_indices,
            )
        hidden_states = dict(out.hidden_states or {})
        fused = self.teacher.fuse_layers(hidden_states, self.layer_indices)
        fused = fused.reshape(b, t, fused.shape[1], fused.shape[2], fused.shape[3])
        reshaped_hidden = {
            idx: feat.reshape(b, t, feat.shape[1], feat.shape[2], feat.shape[3])
            for idx, feat in hidden_states.items()
        }
        return fused, reshaped_hidden

    def _sample_feature_vectors(
        self,
        feature_map: torch.Tensor,
        pixel_xy: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        if pixel_xy.numel() == 0:
            return torch.zeros((0, feature_map.shape[1]), device=feature_map.device, dtype=feature_map.dtype)
        grid_x = (pixel_xy[:, 0] / max(image_width - 1, 1)) * 2.0 - 1.0
        grid_y = (pixel_xy[:, 1] / max(image_height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
        return sampled[:, :, :, 0].permute(0, 2, 1).reshape(-1, feature_map.shape[1])

    def _refine_proposal(
        self,
        proposal: PseudoObjectPatchProposal,
        *,
        fine_local_map: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> PseudoObjectPatchProposal:
        sampled_local = self._sample_feature_vectors(
            fine_local_map,
            proposal.coarse_pixel_xy,
            image_height=image_height,
            image_width=image_width,
        )
        proposal.local_features = sampled_local
        if not self.enable_offset_refinement or proposal.descriptors.numel() == 0:
            proposal.pixel_xy = proposal.coarse_pixel_xy
            proposal.offset_xy = torch.zeros_like(proposal.coarse_pixel_xy)
            return proposal

        offset_in = torch.cat([proposal.descriptors, sampled_local], dim=1)
        offset_xy = torch.tanh(self.offset_head(offset_in)) * (float(self.patch_size) * 0.5)
        refined_xy = proposal.coarse_pixel_xy + offset_xy
        refined_xy = torch.stack(
            [
                refined_xy[:, 0].clamp(0.0, float(image_width - 1)),
                refined_xy[:, 1].clamp(0.0, float(image_height - 1)),
            ],
            dim=1,
        )
        proposal.pixel_xy = refined_xy
        proposal.offset_xy = refined_xy - proposal.coarse_pixel_xy
        return proposal

    def _build_observations(
        self,
        *,
        images: torch.Tensor,
        fused: torch.Tensor,
        teacher_scores: torch.Tensor | None = None,
        use_teacher_for_selection: bool = False,
    ) -> tuple[list[list[FrameObservation]], torch.Tensor, torch.Tensor]:
        b, t, _, image_h, image_w = images.shape
        flat_images = images.reshape(b * t, 3, image_h, image_w)
        local_fine_map = self.local_encoder(flat_images)
        local_map = F.interpolate(local_fine_map, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        local_map = local_map.reshape(b, t, local_map.shape[1], local_map.shape[2], local_map.shape[3])
        local_fine_map = local_fine_map.reshape(
            b,
            t,
            local_fine_map.shape[1],
            local_fine_map.shape[2],
            local_fine_map.shape[3],
        )

        selector_logits = self.selector_head(fused.reshape(b * t, fused.shape[2], fused.shape[3], fused.shape[4]))
        selector_logits = selector_logits.reshape(b, t, 1, fused.shape[-2], fused.shape[-1])

        observations: list[list[FrameObservation]] = []
        for bi in range(b):
            frame_obs: list[FrameObservation] = []
            for ti in range(t):
                fused_bt = fused[bi, ti].unsqueeze(0)
                local_bt = local_map[bi, ti].unsqueeze(0)
                object_ids = self.teacher.build_pseudo_objects(fused_bt)[0]
                using_teacher_scores = teacher_scores is not None and use_teacher_for_selection
                if using_teacher_scores:
                    patch_score = teacher_scores[bi, ti]
                else:
                    patch_score = torch.sigmoid(selector_logits[bi, ti, 0])
                desc_map = self.desc_head(torch.cat([fused_bt, local_bt], dim=1))
                proposal = self.teacher.select_patches(
                    fused=F.normalize(desc_map, dim=1, eps=1e-6),
                    local_features=local_bt,
                    patch_score=patch_score.unsqueeze(0),
                    object_ids=object_ids.unsqueeze(0),
                    selector_logits=None if using_teacher_scores else selector_logits[bi, ti],
                    num_patches=self.num_patches,
                )[0]
                proposal = self._refine_proposal(
                    proposal,
                    fine_local_map=local_fine_map[bi, ti].unsqueeze(0),
                    image_height=image_h,
                    image_width=image_w,
                )
                frame_obs.append(
                    FrameObservation(
                        fused=fused[bi, ti],
                        local_map=local_map[bi, ti],
                        selector_logits=selector_logits[bi, ti, 0],
                        patch_score=patch_score,
                        object_ids=object_ids,
                        proposal=proposal,
                    )
                )
            observations.append(frame_obs)
        return observations, selector_logits, local_map

    def _match_proposals(
        self,
        src: PseudoObjectPatchProposal,
        tgt: PseudoObjectPatchProposal,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if src.descriptors.numel() == 0 or tgt.descriptors.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=src.descriptors.device)
            return empty, empty, empty.float()
        src_desc = F.normalize(src.descriptors, dim=1, eps=1e-6)
        tgt_desc = F.normalize(tgt.descriptors, dim=1, eps=1e-6)
        sim = src_desc @ tgt_desc.t()
        src_best = sim.argmax(dim=1)
        tgt_best = sim.argmax(dim=0)
        src_idx = torch.arange(sim.shape[0], device=sim.device)
        mutual = tgt_best[src_best] == src_idx
        mutual = mutual & (sim[src_idx, src_best] >= self.min_match_cosine)
        return src_idx[mutual], src_best[mutual], sim[src_idx[mutual], src_best[mutual]]

    def _pool_pair_embedding(self, edge_emb: torch.Tensor, conf_logits: torch.Tensor) -> torch.Tensor:
        if edge_emb.numel() == 0:
            return torch.zeros((self.edge_mlp.net[-1].out_features,), device=self.device)
        conf = torch.sigmoid(conf_logits).unsqueeze(1)
        pooled = (conf * edge_emb).sum(dim=0) / conf.sum(dim=0).clamp_min(1e-6)
        return pooled

    def _pair_prediction(
        self,
        src_obs: FrameObservation,
        tgt_obs: FrameObservation,
        *,
        src_frame_idx: int,
        tgt_frame_idx: int,
        lag: int,
    ) -> PairPrediction:
        src_idx, tgt_idx, similarity = self._match_proposals(src_obs.proposal, tgt_obs.proposal)
        if src_idx.numel() == 0:
            edge_emb = torch.zeros((0, self.edge_mlp.net[-1].out_features), device=src_obs.fused.device)
            pooled = torch.zeros((self.edge_mlp.net[-1].out_features,), device=src_obs.fused.device)
            return PairPrediction(
                src_frame_idx=int(src_frame_idx),
                tgt_frame_idx=int(tgt_frame_idx),
                lag=int(lag),
                src_indices=src_idx,
                tgt_indices=tgt_idx,
                similarity=similarity,
                confidence_logits=torch.zeros((0,), device=src_obs.fused.device),
                edge_embeddings=edge_emb,
                pooled_embedding=pooled,
                pose_vec=self.pair_pose_head(pooled.unsqueeze(0)).squeeze(0),
            )

        src_desc = src_obs.proposal.descriptors[src_idx]
        tgt_desc = tgt_obs.proposal.descriptors[tgt_idx]
        src_local = src_obs.proposal.local_features[src_idx]
        tgt_local = tgt_obs.proposal.local_features[tgt_idx]
        delta_xy = (tgt_obs.proposal.pixel_xy[tgt_idx] - src_obs.proposal.pixel_xy[src_idx]) / 320.0
        same_object = (src_obs.proposal.object_ids[src_idx] == tgt_obs.proposal.object_ids[tgt_idx]).float().unsqueeze(1)
        edge_in = torch.cat(
            [
                src_desc,
                tgt_desc,
                src_local,
                tgt_local,
                delta_xy,
                similarity.unsqueeze(1),
                same_object,
                src_obs.proposal.scores[src_idx].unsqueeze(1),
                tgt_obs.proposal.scores[tgt_idx].unsqueeze(1),
            ],
            dim=1,
        )
        edge_emb = self.edge_mlp(edge_in)
        conf_logits = self.conf_head(edge_emb).squeeze(1)
        pooled = self._pool_pair_embedding(edge_emb, conf_logits)
        pose_vec = self.pair_pose_head(pooled.unsqueeze(0)).squeeze(0)
        return PairPrediction(
            src_frame_idx=int(src_frame_idx),
            tgt_frame_idx=int(tgt_frame_idx),
            lag=int(lag),
            src_indices=src_idx,
            tgt_indices=tgt_idx,
            similarity=similarity,
            confidence_logits=conf_logits,
            edge_embeddings=edge_emb,
            pooled_embedding=pooled,
            pose_vec=pose_vec,
        )

    def _predict_frame_pose(
        self,
        incoming_pairs: list[PairPrediction],
        prev_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_multiframe_graph:
            if incoming_pairs:
                return incoming_pairs[-1].pose_vec, prev_hidden
            return torch.zeros((6,), device=self.device), prev_hidden

        if prev_hidden is None:
            prev_hidden = torch.zeros((self.frame_updater.hidden_size,), device=self.device)
        if incoming_pairs:
            incoming = []
            for pair in incoming_pairs:
                lag_idx = min(max(int(pair.lag), 1), self.max_history)
                lag_emb = self.lag_embedding(torch.as_tensor(lag_idx, device=self.device))
                incoming.append(torch.cat([pair.pooled_embedding, lag_emb], dim=0))
            incoming_summary = torch.stack(incoming, dim=0).mean(dim=0)
        else:
            incoming_summary = torch.zeros((self.edge_mlp.net[-1].out_features + self.lag_embedding.embedding_dim,), device=self.device)
        hidden = self.frame_updater(incoming_summary.unsqueeze(0), prev_hidden.unsqueeze(0)).squeeze(0)
        pose_vec = self.frame_pose_head(hidden.unsqueeze(0)).squeeze(0)
        return pose_vec, hidden

    def predict_target_from_history(
        self,
        history_observations: Sequence[FrameObservation],
        cur_obs: FrameObservation,
        *,
        prev_hidden: torch.Tensor | None = None,
    ) -> tuple[list[PairPrediction], torch.Tensor, torch.Tensor | None]:
        if not history_observations:
            return [], torch.zeros((6,), device=self.device), prev_hidden
        history_len = len(history_observations)
        keep = self.max_history if self.use_multiframe_graph else 1
        start = max(0, history_len - keep)
        incoming_pairs: list[PairPrediction] = []
        for src_frame_idx in range(start, history_len):
            lag = history_len - src_frame_idx
            incoming_pairs.append(
                self._pair_prediction(
                    history_observations[src_frame_idx],
                    cur_obs,
                    src_frame_idx=src_frame_idx,
                    tgt_frame_idx=history_len,
                    lag=lag,
                )
            )
        pose_vec, hidden = self._predict_frame_pose(incoming_pairs, prev_hidden)
        return incoming_pairs, pose_vec, hidden

    def forward(
        self,
        images: torch.Tensor,
        *,
        teacher_scores: torch.Tensor | None = None,
        teacher: DinoPatchTeacher | None = None,
        depths: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        use_teacher_for_selection: bool = False,
    ) -> WindowPrediction:
        images = images.to(self.device)
        fused, _ = self._encode_backbone(images)
        computed_teacher_scores = teacher_scores
        if (
            computed_teacher_scores is None
            and use_teacher_for_selection
            and teacher is not None
            and depths is not None
            and poses is not None
            and intrinsics is not None
        ):
            computed_teacher_scores = build_teacher_scores_from_fused(
                fused,
                {
                    "images": images,
                    "depths": depths.to(self.device),
                    "poses": poses.to(self.device),
                    "intrinsics": intrinsics.to(self.device),
                },
                teacher,
            )

        observations, selector_logits, local_map = self._build_observations(
            images=images,
            fused=fused,
            teacher_scores=computed_teacher_scores,
            use_teacher_for_selection=use_teacher_for_selection,
        )

        frame_predictions: list[list[TargetFramePrediction]] = []
        for bi in range(images.shape[0]):
            seq_preds: list[TargetFramePrediction] = []
            prev_hidden = None
            for ti in range(1, images.shape[1]):
                history = observations[bi][:ti]
                incoming_pairs, pose_vec, prev_hidden = self.predict_target_from_history(
                    history,
                    observations[bi][ti],
                    prev_hidden=prev_hidden,
                )
                seq_preds.append(
                    TargetFramePrediction(
                        frame_idx=ti,
                        incoming_pairs=incoming_pairs,
                        pose_vec=pose_vec,
                        hidden_state=None if prev_hidden is None else prev_hidden.clone(),
                    )
                )
            frame_predictions.append(seq_preds)

        return WindowPrediction(
            fused=fused,
            local_map=local_map,
            selector_logits=selector_logits,
            observations=observations,
            frame_predictions=frame_predictions,
            teacher_scores=computed_teacher_scores,
        )

    def infer_single_frame(self, image: torch.Tensor) -> FrameObservation:
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:
            image = image.unsqueeze(1)
        pred = self.forward(image, teacher_scores=None, use_teacher_for_selection=False)
        return pred.observations[0][0]
