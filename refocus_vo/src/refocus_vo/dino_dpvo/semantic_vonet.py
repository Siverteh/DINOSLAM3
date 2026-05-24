from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpvo import altcorr
from dpvo import projective_ops as pops
from dpvo.ba import BA
from dpvo.lietorch import SE3
from dpvo.net import CorrBlock, DIM, Update, autocast
from dpvo.utils import coords_grid_with_index, flatmeshgrid, set_depth

from refocus_vo.dino_dpvo.adapter import SOURCE_DINO, build_dpvo_patch_input
from refocus_vo.dino_dpvo.config import DinoDPVOConfig, load_dino_dpvo_config
from refocus_vo.dino_dpvo.frontend import DinoDPVOBatchOutput, DinoProposalFrontend


def _repeat_to_budget(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.shape[0] >= count:
        return value[:count]
    if value.shape[0] == 0:
        raise ValueError("Cannot expand an empty proposal set to the requested patch budget.")
    repeat = (int(count) + value.shape[0] - 1) // value.shape[0]
    tiled = value.repeat((repeat,) + (1,) * (value.dim() - 1))
    return tiled[:count]


@dataclass
class DinoSemanticPatchifierOutput:
    fmap: torch.Tensor
    gmap: torch.Tensor
    imap: torch.Tensor
    patches: torch.Tensor
    frame_index: torch.Tensor
    coords: torch.Tensor
    semantic: DinoDPVOBatchOutput
    patch_metadata: list[dict[str, torch.Tensor]]
    semantic_fraction_realized: float
    native_fraction_realized: float


class DinoSemanticPatchifier(nn.Module):
    def __init__(
        self,
        *,
        dino_name_or_path: str,
        dino_layers: Sequence[int] = (6, 11),
        dino_dtype: str = "bf16",
        image_size: Sequence[int] = (240, 320),
        dino_patch_size: int = 16,
        dpvo_patch_size: int = 3,
        semantic_candidate_pool: int = 128,
        semantic_patch_budget: int = 80,
        max_nodes_per_object_ratio: float = 0.20,
        k_mutual_neighbors: int = 4,
        local_patch_dim: int = 64,
        dpvo_dim: int = DIM,
        corr_dim: int = 128,
        static_score_weight: float = 0.35,
        quality_floor: float = 0.05,
        use_offset_refinement: bool = True,
        enable_gradient_branch: bool = False,
        gradient_branch_dim: int = 32,
        dino_unfreeze_blocks: int = 0,
        hybrid_grid_rows: int = 6,
        hybrid_grid_cols: int = 8,
        max_semantic_per_cell: int = 1,
        dedupe_radius_px: float = 8.0,
        patch_input_config: dict[str, Any] | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.semantic_patch_budget = int(semantic_patch_budget)
        self.dpvo_patch_size = int(dpvo_patch_size)
        self.hybrid_grid_rows = int(hybrid_grid_rows)
        self.hybrid_grid_cols = int(hybrid_grid_cols)
        self.max_semantic_per_cell = int(max_semantic_per_cell)
        self.dedupe_radius_px = float(dedupe_radius_px)
        self.patch_input_config = dict(patch_input_config or {})
        self.device_name = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.semantic = DinoProposalFrontend(
            dino_name_or_path=str(dino_name_or_path),
            dino_layers=tuple(int(v) for v in dino_layers),
            dino_dtype=str(dino_dtype),
            image_size=tuple(int(v) for v in image_size),
            patch_size=int(dino_patch_size),
            candidate_pool=int(semantic_candidate_pool),
            patch_budget=int(semantic_patch_budget),
            max_nodes_per_object_ratio=float(max_nodes_per_object_ratio),
            k_mutual_neighbors=int(k_mutual_neighbors),
            local_patch_dim=int(local_patch_dim),
            dpvo_dim=int(dpvo_dim),
            use_offset_refinement=bool(use_offset_refinement),
            use_descriptor_bias=False,
            static_score_weight=float(static_score_weight),
            quality_floor=float(quality_floor),
            enable_gradient_branch=bool(enable_gradient_branch),
            gradient_branch_dim=int(gradient_branch_dim),
            dino_unfreeze_blocks=int(dino_unfreeze_blocks),
            device=self.device_name,
        )

        embed_dim = int(self.semantic.backbone.embed_dim or dpvo_dim)
        combined_dim = embed_dim + int(local_patch_dim)
        self.fmap_proj = nn.Sequential(
            nn.Conv2d(combined_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, int(corr_dim), kernel_size=1),
        )
        self.imap_proj = nn.Sequential(
            nn.Conv2d(combined_dim, int(dpvo_dim), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dpvo_dim), int(dpvo_dim), kernel_size=1),
        )

        self.last_output: DinoDPVOBatchOutput | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def _build_semantic_output(self, rgb01: torch.Tensor) -> DinoDPVOBatchOutput:
        b, t, _, image_h, image_w = rgb01.shape
        fused, register_context = self.semantic._encode_backbone(rgb01)
        local_feats = self.semantic._encode_local_features(rgb01, target_hw=(fused.shape[-2], fused.shape[-1]))
        local_map = local_feats["local_map"]
        local_fine = local_feats["local_fine"]
        offset_fine = local_feats["offset_fine"]

        fused_flat = fused.reshape(b * t, fused.shape[2], fused.shape[3], fused.shape[4])
        selector_logits = self.semantic.selector_head(fused_flat).reshape(b, t, 1, fused.shape[-2], fused.shape[-1])
        static_in = torch.cat(
            [
                fused_flat,
                local_map.reshape(b * t, local_map.shape[2], local_map.shape[3], local_map.shape[4]),
            ],
            dim=1,
        )
        staticness_logits = self.semantic.staticness_head(static_in).reshape(b, t, 1, fused.shape[-2], fused.shape[-1])
        gradient_score = self.semantic._gradient_score(rgb01)

        observations: list[list[Any]] = []
        for bi in range(b):
            frame_outputs = []
            for ti in range(t):
                frame_outputs.append(
                    self.semantic._prepare_frame_output(
                        fused_bt=fused[bi, ti].unsqueeze(0),
                        local_bt=local_map[bi, ti].unsqueeze(0),
                        local_fine_bt=local_fine[bi, ti].unsqueeze(0),
                        offset_bt=offset_fine[bi, ti].unsqueeze(0),
                        selector_bt=selector_logits[bi, ti].unsqueeze(0),
                        staticness_bt=staticness_logits[bi, ti].unsqueeze(0),
                        gradient_bt=gradient_score[bi, ti],
                        register_context_bt=(
                            None if register_context is None else register_context[bi, ti].unsqueeze(0)
                        ),
                        image_height=image_h,
                        image_width=image_w,
                    )
                )
            observations.append(frame_outputs)

        return DinoDPVOBatchOutput(
            fused=fused,
            selector_logits=selector_logits[:, :, 0],
            staticness_logits=staticness_logits[:, :, 0],
            gradient_score=gradient_score,
            observations=observations,
            register_context=register_context,
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        patches_per_image: int | None = None,
        disps: torch.Tensor | None = None,
        frontend_mode: str = "dino_proposals",
        native_fraction: float = 0.0,
        dino_fraction: float = 1.0,
    ) -> DinoSemanticPatchifierOutput:
        if images.dim() != 5:
            raise ValueError(f"Expected images with shape (B,T,3,H,W), got {tuple(images.shape)}")
        b, t, _, image_h, image_w = images.shape
        if b != 1:
            raise ValueError("Experimental DINO semantic DPVO training currently supports batch_size=1 only.")

        rgb01 = ((images + 0.5) * 0.5).clamp(0.0, 1.0)
        semantic_output = self._build_semantic_output(rgb01)
        self.last_output = semantic_output

        fused = semantic_output.fused
        fused_flat = fused.reshape(b * t, fused.shape[2], fused.shape[3], fused.shape[4])
        h4 = int(image_h) // 4
        w4 = int(image_w) // 4
        local_feats = self.semantic._encode_local_features(rgb01, target_hw=(fused.shape[-2], fused.shape[-1]))
        local_fine = local_feats["local_fine"].reshape(b * t, local_feats["local_fine"].shape[2], local_feats["local_fine"].shape[3], local_feats["local_fine"].shape[4])
        local_mid = local_feats["local_mid"].reshape(b * t, local_feats["local_mid"].shape[2], local_feats["local_mid"].shape[3], local_feats["local_mid"].shape[4])
        fused_mid = F.interpolate(fused_flat, size=(h4, w4), mode="bilinear", align_corners=False)
        combined_mid = torch.cat([fused_mid, local_mid], dim=1)

        fmap = (self.fmap_proj(combined_mid) / 4.0).reshape(b, t, -1, h4, w4)
        imap_dense = (self.imap_proj(combined_mid) / 4.0).reshape(b, t, -1, h4, w4)

        patch_budget = int(self.semantic_patch_budget if patches_per_image is None else patches_per_image)
        coords_per_frame = []
        patch_metadata: list[dict[str, torch.Tensor]] = []
        total_semantic = 0.0
        total_native = 0.0
        runtime_state: dict[str, Any] = {}
        for frame_output in semantic_output.observations[0]:
            patch_input = build_dpvo_patch_input(
                frame_output,
                patch_budget=patch_budget,
                frontend_mode=frontend_mode,
                dpvo_res=4,
                image_height=image_h,
                image_width=image_w,
                config={
                    "native_fraction": float(native_fraction),
                    "dino_fraction": float(dino_fraction),
                    "static_score_weight": float(self.semantic.static_score_weight),
                    "hybrid_grid_rows": int(self.hybrid_grid_rows),
                    "hybrid_grid_cols": int(self.hybrid_grid_cols),
                    "max_dino_per_cell": int(self.max_semantic_per_cell),
                    "dedupe_radius_px": float(self.dedupe_radius_px),
                    **self.patch_input_config,
                },
                runtime_state=runtime_state,
            )
            coords = patch_input["external_coords"][0].to(device=fused.device, dtype=torch.float32)
            coords_per_frame.append(coords)
            metadata = patch_input["patch_metadata"]
            patch_metadata.append(metadata)
            sources = metadata["source_labels"]
            if int(sources.numel()) > 0:
                total_semantic += float((sources == int(SOURCE_DINO)).float().mean().item())
                total_native += float((sources != int(SOURCE_DINO)).float().mean().item())
        coords = torch.stack(coords_per_frame, dim=0)

        gmap = altcorr.patchify(fmap[0], coords, self.dpvo_patch_size // 2).view(b, -1, fmap.shape[2], self.dpvo_patch_size, self.dpvo_patch_size)
        imap = altcorr.patchify(imap_dense[0], coords, 0).view(b, -1, DIM, 1, 1)

        if disps is None:
            disps = torch.ones((b, t, h4, w4), device=fmap.device, dtype=torch.float32)
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, self.dpvo_patch_size // 2).view(
            b,
            -1,
            3,
            self.dpvo_patch_size,
            self.dpvo_patch_size,
        )

        frame_index = torch.arange(t, device=fmap.device).view(t, 1).repeat(1, patch_budget).reshape(-1)
        return DinoSemanticPatchifierOutput(
            fmap=fmap,
            gmap=gmap,
            imap=imap,
            patches=patches,
            frame_index=frame_index,
            coords=coords,
            semantic=semantic_output,
            patch_metadata=patch_metadata,
            semantic_fraction_realized=(total_semantic / max(len(patch_metadata), 1)),
            native_fraction_realized=(total_native / max(len(patch_metadata), 1)),
        )


class DinoSemanticVONet(nn.Module):
    def __init__(
        self,
        *,
        dino_name_or_path: str,
        dino_layers: Sequence[int] = (6, 11),
        dino_dtype: str = "bf16",
        image_size: Sequence[int] = (240, 320),
        dino_patch_size: int = 16,
        dpvo_patch_size: int = 3,
        semantic_candidate_pool: int = 128,
        semantic_patch_budget: int = 80,
        max_nodes_per_object_ratio: float = 0.20,
        k_mutual_neighbors: int = 4,
        local_patch_dim: int = 64,
        dpvo_dim: int = DIM,
        corr_dim: int = 128,
        static_score_weight: float = 0.35,
        quality_floor: float = 0.05,
        use_offset_refinement: bool = True,
        enable_gradient_branch: bool = False,
        gradient_branch_dim: int = 32,
        dino_unfreeze_blocks: int = 0,
        hybrid_grid_rows: int = 6,
        hybrid_grid_cols: int = 8,
        max_semantic_per_cell: int = 1,
        dedupe_radius_px: float = 8.0,
        patch_input_config: dict[str, Any] | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.P = int(dpvo_patch_size)
        self.patchify = DinoSemanticPatchifier(
            dino_name_or_path=str(dino_name_or_path),
            dino_layers=tuple(int(v) for v in dino_layers),
            dino_dtype=str(dino_dtype),
            image_size=tuple(int(v) for v in image_size),
            dino_patch_size=int(dino_patch_size),
            dpvo_patch_size=int(dpvo_patch_size),
            semantic_candidate_pool=int(semantic_candidate_pool),
            semantic_patch_budget=int(semantic_patch_budget),
            max_nodes_per_object_ratio=float(max_nodes_per_object_ratio),
            k_mutual_neighbors=int(k_mutual_neighbors),
            local_patch_dim=int(local_patch_dim),
            dpvo_dim=int(dpvo_dim),
            corr_dim=int(corr_dim),
            static_score_weight=float(static_score_weight),
            quality_floor=float(quality_floor),
            use_offset_refinement=bool(use_offset_refinement),
            enable_gradient_branch=bool(enable_gradient_branch),
            gradient_branch_dim=int(gradient_branch_dim),
            dino_unfreeze_blocks=int(dino_unfreeze_blocks),
            hybrid_grid_rows=int(hybrid_grid_rows),
            hybrid_grid_cols=int(hybrid_grid_cols),
            max_semantic_per_cell=int(max_semantic_per_cell),
            dedupe_radius_px=float(dedupe_radius_px),
            patch_input_config=patch_input_config,
            device=device,
        )
        self.update = Update(self.P)
        self.DIM = DIM
        self.RES = 4

    @autocast(enabled=False)
    def forward(
        self,
        images,
        poses,
        disps,
        intrinsics,
        M=1024,
        STEPS=12,
        P=1,
        structure_only=False,
        rescale=False,
        frontend_mode: str = "dino_proposals",
        native_fraction: float = 0.0,
        dino_fraction: float = 1.0,
    ):
        del M, P, rescale
        images = 2 * (images / 255.0) - 0.5
        if intrinsics.dim() == 2:
            intrinsics = intrinsics[:, None, :].expand(-1, images.shape[1], -1)
        intrinsics = intrinsics / 4.0
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()

        patch_output = self.patchify(
            images,
            disps=disps,
            frontend_mode=frontend_mode,
            native_fraction=float(native_fraction),
            dino_fraction=float(dino_fraction),
        )
        fmap = patch_output.fmap
        gmap = patch_output.gmap
        imap = patch_output.imap
        patches = patch_output.patches
        ix = patch_output.frame_index

        corr_fn = CorrBlock(fmap, gmap)

        b, _, _, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p // 2, p // 2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0, 8, device=images.device), indexing="ij")
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device=images.device, dtype=torch.float32)

        Gs = SE3.IdentityLike(poses)
        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]

        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only:
                    Gs.data[:, n] = Gs.data[:, n - 1]
                kk1, jj1 = flatmeshgrid(torch.where(ix < n)[0], torch.arange(n, n + 1, device=images.device), indexing="ij")
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n + 1, device=images.device), indexing="ij")

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device=images.device)
                net = torch.cat([net1, net], dim=1)

                if torch.rand((), device=images.device).item() < 0.1:
                    keep = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[keep]
                    jj = jj[keep]
                    kk = kk[keep]
                    net = net[:, keep]

                patches[:, ix == n, 2] = torch.median(patches[:, (ix == n - 1) | (ix == n - 2), 2])

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:, kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[..., p // 2, p // 2, :] + delta

            for _ in range(2):
                Gs, patches = BA(
                    Gs,
                    patches,
                    intrinsics,
                    target,
                    weight,
                    lmbda,
                    ii,
                    jj,
                    kk,
                    bounds,
                    ep=10,
                    fixedp=1,
                    structure_only=structure_only,
                )

            kl = torch.as_tensor(0.0, device=images.device)
            dij = (ii - jj).abs()
            keep = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[keep], jj[keep], kk[keep])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[keep], jj[keep], kk[keep], jacobian=True)
            traj.append((valid, coords, coords_gt, Gs[:, :n], Ps[:, :n], kl))

        return traj, patch_output


def build_dino_semantic_vonet(cfg: DinoDPVOConfig) -> DinoSemanticVONet:
    model_cfg = cfg.model
    return DinoSemanticVONet(
        dino_name_or_path=str(model_cfg.get("dino_name_or_path")),
        dino_layers=tuple(int(v) for v in model_cfg.get("dino_layers", [6, 11])),
        dino_dtype=str(model_cfg.get("dino_dtype", "bf16")),
        image_size=tuple(int(v) for v in model_cfg.get("image_size", [240, 320])),
        dino_patch_size=int(model_cfg.get("dino_patch_size", 16)),
        dpvo_patch_size=int(model_cfg.get("dpvo_patch_size", 3)),
        semantic_candidate_pool=int(model_cfg.get("semantic_candidate_pool", 128)),
        semantic_patch_budget=int(model_cfg.get("semantic_patch_budget", 80)),
        max_nodes_per_object_ratio=float(model_cfg.get("max_nodes_per_object_ratio", 0.20)),
        k_mutual_neighbors=int(model_cfg.get("k_mutual_neighbors", 4)),
        local_patch_dim=int(model_cfg.get("local_patch_dim", 64)),
        dpvo_dim=int(model_cfg.get("dpvo_dim", 384)),
        corr_dim=int(model_cfg.get("corr_dim", 128)),
        static_score_weight=float(model_cfg.get("static_score_weight", 0.35)),
        quality_floor=float(model_cfg.get("quality_floor", 0.05)),
        use_offset_refinement=bool(model_cfg.get("use_offset_refinement", True)),
        enable_gradient_branch=bool(model_cfg.get("enable_gradient_branch", False)),
        gradient_branch_dim=int(model_cfg.get("gradient_branch_dim", 32)),
        dino_unfreeze_blocks=int(model_cfg.get("dino_unfreeze_blocks", 0)),
        hybrid_grid_rows=int(model_cfg.get("hybrid_grid_rows", 6)),
        hybrid_grid_cols=int(model_cfg.get("hybrid_grid_cols", 8)),
        max_semantic_per_cell=int(model_cfg.get("max_semantic_per_cell", 1)),
        dedupe_radius_px=float(model_cfg.get("dedupe_radius_px", 8.0)),
        patch_input_config=dict(model_cfg),
        device=str(cfg.training.get("device", "cuda")),
    ).to(torch.device(str(cfg.training.get("device", "cuda"))))


def load_dino_semantic_vonet_checkpoint(
    checkpoint: str | Path,
    *,
    config: str | Path | None = None,
    device: str = "cuda",
) -> tuple[DinoDPVOConfig, DinoSemanticVONet]:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = load_dino_dpvo_config(config or payload["config_path"])
    cfg.raw.setdefault("training", {})["device"] = str(device)
    model = build_dino_semantic_vonet(cfg)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return cfg, model
