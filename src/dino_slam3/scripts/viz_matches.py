from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dino_slam3.utils.config import load_config
from dino_slam3.data.tum_rgbd import TUMRGBDDataset, CameraIntrinsics
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints import extract_keypoints_and_descriptors


def _unnorm(x: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std = np.array([0.229, 0.224, 0.225])[None, None, :]
    x = x.transpose(1, 2, 0)
    x = (x * std + mean) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out", default="runs/viz_matches.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    cam = cfg["camera"]
    intr = CameraIntrinsics(float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"]))

    seq = dcfg["sequences"][0]
    ds = TUMRGBDDataset(
        dcfg["root"],
        seq,
        intr,
        frame_spacing=int(dcfg.get("frame_spacing", 1)),
        is_train=False,
        max_association_dt=float(dcfg.get("max_association_dt", 0.02)),
        pad_to=int(dcfg.get("pad_to", 16)),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    mcfg = cfg["model"]
    model = LocalFeatureNet(patch_size=int(mcfg.get("patch_size", 16)), descriptor_dim=int(mcfg.get("descriptor_dim", 128)))
    ckpt = torch.load(mcfg["checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch = next(iter(dl))
    rgb1 = batch["rgb1"].to(device)
    rgb2 = batch["rgb2"].to(device)

    out1 = model(rgb1)
    out2 = model(rgb2)

    stride = int(cfg["model"].get("stride", 4))
    det = cfg["model"]["heads"]["detector"]
    k1 = extract_keypoints_and_descriptors(
        out1.heatmap, out1.desc, out1.offset, out1.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
    )[0]
    k2 = extract_keypoints_and_descriptors(
        out2.heatmap, out2.desc, out2.offset, out2.reliability,
        stride=stride,
        nms_radius=int(det.get("nms_radius", 4)),
        max_keypoints=int(det.get("max_keypoints", 1024)),
        tile_size=int(det.get("tile_size", 16)),
        k_per_tile=int(det.get("k_per_tile", 8)),
    )[0]

    # draw keypoints
    img1 = _unnorm(rgb1[0].detach().cpu().numpy())[..., ::-1].copy()
    img2 = _unnorm(rgb2[0].detach().cpu().numpy())[..., ::-1].copy()

    for (x, y) in k1.kpts[:500]:
        cv2.circle(img1, (int(x), int(y)), 2, (0, 255, 0), -1)
    for (x, y) in k2.kpts[:500]:
        cv2.circle(img2, (int(x), int(y)), 2, (0, 255, 0), -1)

    # match (cross-check)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(k1.desc, k2.desc)
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    cvk1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in k1.kpts]
    cvk2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in k2.kpts]
    vis = cv2.drawMatches(img1, cvk1, img2, cvk2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
