from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dino_slam3.utils.config import load_config
from dino_slam3.data.tum_rgbd import TUMRGBDDataset
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch
from dino_slam3.geometry.projection import unproject, transform, project


def _unnorm_rgb_to_bgr_u8(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) normalized with ImageNet mean/std.
    returns BGR uint8 image.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1) * 255.0
    rgb = y.byte().permute(1, 2, 0).cpu().numpy()
    return rgb[..., ::-1].copy()


def _mutual_nn(d1: torch.Tensor, d2: torch.Tensor):
    """
    d1: (N,D), d2:(M,D)
    returns idx1(K,), idx2(K,) mutual NN matches
    """
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


def _draw_matches_custom(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    pts1: np.ndarray,   # (K,2) float
    pts2: np.ndarray,   # (K,2) float
    is_inlier: np.ndarray,  # (K,) bool
    max_lines: int,
    radius: int = 3,
    thickness: int = 1,
) -> np.ndarray:
    """
    Build a side-by-side canvas and draw per-match colored lines.
    """
    h1, w1 = img1_bgr.shape[:2]
    h2, w2 = img2_bgr.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_bgr
    canvas[:h2, w1:w1 + w2] = img2_bgr

    K = min(len(pts1), max_lines)
    for i in range(K):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        x1i, y1i = int(round(x1)), int(round(y1))
        x2i, y2i = int(round(x2)) + w1, int(round(y2))

        if bool(is_inlier[i]):
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red

        cv2.circle(canvas, (x1i, y1i), radius, color, -1)
        cv2.circle(canvas, (x2i, y2i), radius, color, -1)
        cv2.line(canvas, (x1i, y1i), (x2i, y2i), color, thickness, lineType=cv2.LINE_AA)

    return canvas


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--ckpt", default="runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt", type=str)
    ap.add_argument("--sequence", type=str, default=None)
    ap.add_argument("--out", default="runs/viz/matches_inliers.png")
    ap.add_argument("--max_lines", type=int, default=250)
    ap.add_argument("--inlier_px", type=float, default=3.0)
    ap.add_argument("--only_inliers", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset --------
    dcfg = cfg["dataset"]
    assoc = dcfg.get("association", {})
    seq = args.sequence or (dcfg["sequences"][0] if "sequences" in dcfg else dcfg["val_sequences"][0])

    ds = TUMRGBDDataset(
        dataset_root=dcfg["root"],
        sequence=seq,
        frame_spacing_min=int(dcfg.get("frame_spacing_min", dcfg.get("frame_spacing", 1))),
        frame_spacing_max=int(dcfg.get("frame_spacing_max", dcfg.get("frame_spacing", 1))),
        is_train=False,
        pad_to=int(dcfg.get("pad_to", 16)),
        augmentation=None,
        max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
        max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
        max_frames=dcfg.get("max_frames"),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    batch = next(iter(dl))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)

    # -------- model --------
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

    out1 = model(batch["rgb1"])
    out2 = model(batch["rgb2"])

    det = mcfg["heads"]["detector"]
    stride = int(mcfg.get("stride", 4))

    k1 = extract_keypoints_torch(
        out1.heatmap, out1.desc, out1.offset, out1.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        valid_mask_img=batch.get("valid_depth1", None),
    )
    k2 = extract_keypoints_torch(
        out2.heatmap, out2.desc, out2.offset, out2.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        valid_mask_img=batch.get("valid_depth2", None),
    )

    d1 = k1.desc[0]
    d2 = k2.desc[0]
    xy1_all = k1.xy_img[0]  # (N,2)
    xy2_all = k2.xy_img[0]  # (M,2)

    idx1, idx2 = _mutual_nn(d1, d2)
    if idx1 is None:
        print("No matches.")
        return

    # sort matches by similarity, best first
    sim = (d1[idx1] * d2[idx2]).sum(dim=-1)
    order = torch.argsort(sim, descending=True)
    idx1 = idx1[order]
    idx2 = idx2[order]

    # compute GT reprojection error for matches
    xy1m = xy1_all[idx1].unsqueeze(0)  # (1,K,2)
    xy2m = xy2_all[idx2].unsqueeze(0)  # (1,K,2)

    pts1 = unproject(batch["depth1"], batch["K"], xy1m)
    pts2 = transform(batch["relative_pose"], pts1)
    xy2_gt = project(pts2, batch["K"])

    err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)[0]  # (K,)
    inlier = (err < float(args.inlier_px))

    # optionally filter to only inliers
    if args.only_inliers:
        keep = inlier
        idx1 = idx1[keep]
        idx2 = idx2[keep]
        err = err[keep]
        inlier = inlier[keep]

    K = int(idx1.numel())
    if K == 0:
        print("No inliers (after filtering).")
        return

    # stats
    inlier_ratio = float(inlier.float().mean().item())
    err_np = err.detach().cpu().numpy()
    print(f"matches: {K}  inlier@{args.inlier_px:.1f}px: {inlier_ratio:.3f}  "
          f"mean_err: {err_np.mean():.3f}px  median_err: {np.median(err_np):.3f}px")

    # images
    img1 = _unnorm_rgb_to_bgr_u8(batch["rgb1"][0])
    img2 = _unnorm_rgb_to_bgr_u8(batch["rgb2"][0])

    # prepare drawing arrays
    pts1 = xy1_all[idx1].detach().cpu().numpy()
    pts2 = xy2_all[idx2].detach().cpu().numpy()
    inl = inlier.detach().cpu().numpy().astype(bool)

    vis = _draw_matches_custom(
        img1, img2,
        pts1, pts2,
        inl,
        max_lines=int(args.max_lines),
        radius=3,
        thickness=1,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
