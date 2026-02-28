from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_slam3.utils.config import load_config
from dino_slam3.data.tum_rgbd import TUMRGBDDataset
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch
from dino_slam3.geometry.projection import unproject, transform, project


def _mutual_nn_matches(d1: torch.Tensor, d2: torch.Tensor):
    """Mutual NN matching (no ratio test). Returns (idx1, idx2) or (None, None)."""
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


def _as_bK(bK: torch.Tensor) -> torch.Tensor:
    return bK.unsqueeze(0) if bK.dim() == 2 else bK


def _as_bT(bT: torch.Tensor) -> torch.Tensor:
    return bT.unsqueeze(0) if bT.dim() == 2 else bT


def _sample_depth_bilinear(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    depth: (1,1,H,W) float
    xy: (N,2) pixel coords (x,y)
    returns: (N,) depth sampled bilinearly
    """
    assert depth.dim() == 4 and depth.shape[0] == 1 and depth.shape[1] == 1
    H, W = depth.shape[-2:]
    x = xy[:, 0]
    y = xy[:, 1]
    gx = (x / (W - 1)) * 2 - 1
    gy = (y / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)  # (1,N,1,2)
    z = F.grid_sample(depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return z.view(-1)


def _pnp_ransac_inliers(
    xy1: torch.Tensor,         # (N,2) pixel coords in img1
    xy2: torch.Tensor,         # (N,2) pixel coords in img2
    depth1: torch.Tensor,      # (1,1,H,W) meters
    K: torch.Tensor,           # (3,3)
    reproj_px: float = 3.0,
    z_min_m: float = 0.10,
    max_iters: int = 2000,
    conf: float = 0.999,
) -> Tuple[int, float, bool]:
    """
    Runs PnP-RANSAC using 3D from depth1 and 2D in frame2.
    Returns (num_inliers, inlier_ratio_over_used_correspondences, success).
    """
    if xy1.numel() == 0:
        return 0, 0.0, False

    z = _sample_depth_bilinear(depth1, xy1)  # (N,)
    valid = torch.isfinite(z) & (z > z_min_m)
    if valid.sum().item() < 6:
        return 0, 0.0, False

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
    pts3d = torch.stack([X, Y, Z], dim=-1)  # (M,3)

    obj = pts3d.detach().cpu().numpy().astype(np.float32)
    img = xy2v.detach().cpu().numpy().astype(np.float32)
    Kcv = K.detach().cpu().numpy().astype(np.float64)

    if obj.shape[0] < 6:
        return 0, 0.0, False

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
        return 0, 0.0, False

    ninl = int(inliers.shape[0])
    ratio = ninl / float(obj.shape[0])
    return ninl, float(ratio), True


@dataclass
class SeqStats:
    sequence: str
    pairs: int
    mean_matches: float
    med_matches: float

    mean_gt_inl: float
    med_gt_inl: float
    mean_gt_rate: float
    med_gt_rate: float

    mean_pnp_inl: float
    med_pnp_inl: float
    mean_pnp_rate: float
    med_pnp_rate: float

    pnp_success_rate: float


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else 0.0


@torch.no_grad()
def eval_sequence(
    *,
    cfg: Dict,
    model: LocalFeatureNet,
    device: torch.device,
    sequence: str,
    num_batches: int,
    inlier_px: float,
    pnp_reproj_px: float,
    z_min_m: float,
    num_workers: int,
) -> Optional[SeqStats]:
    dcfg = cfg["dataset"]
    assoc = dcfg.get("association", {})

    ds = TUMRGBDDataset(
        dataset_root=dcfg["root"],
        sequence=sequence,
        frame_spacing_min=1,
        frame_spacing_max=1,
        is_train=False,
        pad_to=int(dcfg.get("pad_to", 16)),
        augmentation=None,
        max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
        max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
        max_frames=dcfg.get("max_frames"),
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    mcfg = cfg["model"]
    det = mcfg["heads"]["detector"]
    stride = int(mcfg.get("stride", 4))
    use_rel_score = bool(cfg.get("inference", {}).get("use_reliability_in_score", False))

    match_counts: List[int] = []
    gt_inlier_counts: List[int] = []
    gt_inlier_rates: List[float] = []
    pnp_inlier_counts: List[int] = []
    pnp_inlier_rates: List[float] = []
    pnp_success: List[int] = []

    iters = min(num_batches, len(ds))
    for it, batch in enumerate(tqdm(dl, total=iters, desc=f"{sequence}", leave=False), start=1):
        if it > num_batches:
            break

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        out1 = model(batch["rgb1"])
        out2 = model(batch["rgb2"])

        k1 = extract_keypoints_torch(
            out1.heatmap, out1.desc, out1.offset, out1.reliability,
            stride=stride,
            nms_radius=int(det.get("nms_radius", 4)),
            tile_size=int(det.get("tile_size", 16)),
            k_per_tile=int(det.get("k_per_tile", 8)),
            max_keypoints=int(det.get("max_keypoints", 1024)),
            valid_mask_img=batch.get("valid_depth1", None),
            use_reliability_in_score=use_rel_score,
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
        )

        d1 = k1.desc[0]
        d2 = k2.desc[0]
        xy1_all = k1.xy_img[0]
        xy2_all = k2.xy_img[0]

        idx1, idx2 = _mutual_nn_matches(d1, d2)
        if idx1 is None:
            continue

        idx1 = idx1.long()
        idx2 = idx2.long()

        xy1m = xy1_all[idx1]
        xy2m = xy2_all[idx2]
        M = int(xy1m.shape[0])
        match_counts.append(M)

        # --- GT reprojection inliers ---
        depth1 = batch["depth1"]  # (1,1,H,W)
        bK = _as_bK(batch["K"])
        bT = _as_bT(batch["relative_pose"])

        pts1 = unproject(depth1, bK, xy1m.view(1, -1, 2))
        pts2 = transform(bT, pts1)
        xy2_gt = project(pts2, bK)[0]

        err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)
        inl = err < float(inlier_px)
        gt_cnt = int(inl.sum().item())
        gt_inlier_counts.append(gt_cnt)
        gt_inlier_rates.append(gt_cnt / float(max(M, 1)))

        # --- PnP-RANSAC inliers ---
        K = bK[0]
        ninl, ratio, ok = _pnp_ransac_inliers(
            xy1=xy1m,
            xy2=xy2m,
            depth1=depth1,
            K=K,
            reproj_px=float(pnp_reproj_px),
            z_min_m=float(z_min_m),
        )
        pnp_success.append(1 if ok else 0)
        pnp_inlier_counts.append(int(ninl))
        pnp_inlier_rates.append(float(ratio))

    if not match_counts:
        return None

    return SeqStats(
        sequence=sequence,
        pairs=len(match_counts),
        mean_matches=_mean([float(x) for x in match_counts]),
        med_matches=_median([float(x) for x in match_counts]),

        mean_gt_inl=_mean([float(x) for x in gt_inlier_counts]),
        med_gt_inl=_median([float(x) for x in gt_inlier_counts]),
        mean_gt_rate=_mean(gt_inlier_rates),
        med_gt_rate=_median(gt_inlier_rates),

        mean_pnp_inl=_mean([float(x) for x in pnp_inlier_counts]),
        med_pnp_inl=_median([float(x) for x in pnp_inlier_counts]),
        mean_pnp_rate=_mean(pnp_inlier_rates),
        med_pnp_rate=_median(pnp_inlier_rates),

        pnp_success_rate=_mean([float(x) for x in pnp_success]),
    )


def _print_stats(s: SeqStats, inlier_px: float, pnp_px: float) -> None:
    print(f"Sequence: {s.sequence}")
    print(f"Pairs evaluated: {s.pairs}")
    print(f"Mean matches: {s.mean_matches:.1f} | Median matches: {s.med_matches:.1f}")
    print(f"GT inliers (@{inlier_px:.1f}px): mean {s.mean_gt_inl:.1f} | median {s.med_gt_inl:.1f}")
    print(f"GT inlier_rate (@{inlier_px:.1f}px): mean {s.mean_gt_rate:.3f} | median {s.med_gt_rate:.3f}")
    print(f"PnP-RANSAC inliers (@{pnp_px:.1f}px): mean {s.mean_pnp_inl:.1f} | median {s.med_pnp_inl:.1f}")
    print(f"PnP-RANSAC inlier_rate (@{pnp_px:.1f}px): mean {s.mean_pnp_rate:.3f} | median {s.med_pnp_rate:.3f}")
    print(f"PnP success rate: {s.pnp_success_rate:.3f}")
    print("-" * 80)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/eval/viz.yaml")
    ap.add_argument("--ckpt", type=str, default="runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt")

    # either pass many --sequence, or use --all to read dataset.sequences from config
    ap.add_argument("--sequence", type=str, action="append", default=None,
                    help="Can be passed multiple times: --sequence a --sequence b ...")
    ap.add_argument("--all", action="store_true",
                    help="Evaluate all sequences listed in cfg.dataset.sequences")

    ap.add_argument("--num_batches", type=int, default=50)
    ap.add_argument("--inlier_px", type=float, default=3.0)
    ap.add_argument("--pnp_reproj_px", type=float, default=3.0)
    ap.add_argument("--z_min_m", type=float, default=0.10)
    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pick sequences
    seqs: List[str] = []
    if args.sequence:
        seqs = list(args.sequence)
    elif args.all:
        seqs = list(cfg.get("dataset", {}).get("sequences", []))
        if not seqs:
            raise ValueError("cfg.dataset.sequences is empty. Either add sequences or pass --sequence.")
    else:
        raise ValueError("Provide --sequence (can repeat) or use --all to read cfg.dataset.sequences.")

    # build model once
    mcfg = cfg["model"]
    model = LocalFeatureNet(
        dinov3_name=mcfg["dinov3"]["name_or_path"],
        patch_size=int(mcfg.get("patch_size", 16)),
        descriptor_dim=int(mcfg["heads"].get("descriptor_dim", 256)),
        fine_channels=int(mcfg["fine_cnn"].get("channels", 96)),
        fine_blocks=int(mcfg["fine_cnn"].get("num_blocks", 8)),
        freeze_backbone=True,
        use_offset=bool(mcfg["heads"].get("offset", {}).get("enabled", True)),
        use_reliability=bool(mcfg["heads"].get("reliability", {}).get("enabled", True)),
        dinov3_dtype=str(mcfg["dinov3"].get("dtype", "bf16")),
    ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    all_stats: List[SeqStats] = []
    for s in seqs:
        st = eval_sequence(
            cfg=cfg,
            model=model,
            device=device,
            sequence=s,
            num_batches=int(args.num_batches),
            inlier_px=float(args.inlier_px),
            pnp_reproj_px=float(args.pnp_reproj_px),
            z_min_m=float(args.z_min_m),
            num_workers=int(args.num_workers),
        )
        if st is None:
            print(f"Sequence: {s} -> no valid pairs")
            print("-" * 80)
            continue
        _print_stats(st, float(args.inlier_px), float(args.pnp_reproj_px))
        all_stats.append(st)

    if not all_stats:
        print("No valid results across sequences.")
        return

    # simple overall aggregation (mean of per-sequence means)
    print("OVERALL (mean over sequences)")
    print(f"Sequences evaluated: {len(all_stats)}")
    print(f"Mean matches: {_mean([s.mean_matches for s in all_stats]):.1f}")
    print(f"GT inlier_rate (@{args.inlier_px:.1f}px): {_mean([s.mean_gt_rate for s in all_stats]):.3f}")
    print(f"PnP inlier_rate (@{args.pnp_reproj_px:.1f}px): {_mean([s.mean_pnp_rate for s in all_stats]):.3f}")
    print(f"PnP success rate: {_mean([s.pnp_success_rate for s in all_stats]):.3f}")


if __name__ == "__main__":
    main()
