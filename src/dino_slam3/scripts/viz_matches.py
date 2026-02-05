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

def draw_matches(img1, k1, img2, k2, matches, max_draw=100):
    m = matches[:max_draw]
    out = cv2.drawMatches(img1, k1, img2, k2, m, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return out

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out", default="runs/viz.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    cam = cfg["camera"]
    intr = CameraIntrinsics(float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"]))

    seq = dcfg["sequences"][0]
    ds = TUMRGBDDataset(dcfg["root"], seq, intr, input_size=int(dcfg["input_size"]), frame_spacing=int(dcfg["frame_spacing"]), is_train=False)
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

    stride = 4
    kps1 = extract_keypoints_and_descriptors(out1.heatmap, out1.desc, out1.offset, out1.reliability, stride=stride, nms_radius=4, max_keypoints=800)[0]
    kps2 = extract_keypoints_and_descriptors(out2.heatmap, out2.desc, out2.offset, out2.reliability, stride=stride, nms_radius=4, max_keypoints=800)[0]

    # BF match
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    d1 = kps1.desc
    d2 = kps2.desc
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Convert tensors to uint8 images for viz
    def unnorm(x):
        mean = np.array([0.485,0.456,0.406])[None,None,:]
        std = np.array([0.229,0.224,0.225])[None,None,:]
        x = x.transpose(1,2,0)
        x = (x*std + mean)*255.0
        return np.clip(x, 0, 255).astype(np.uint8)

    img1 = unnorm(rgb1[0].detach().cpu().numpy())
    img2 = unnorm(rgb2[0].detach().cpu().numpy())
    img1_bgr = img1[..., ::-1].copy()
    img2_bgr = img2[..., ::-1].copy()

    cvk1 = [cv2.KeyPoint(float(x), float(y), 1) for x,y in kps1.kpts]
    cvk2 = [cv2.KeyPoint(float(x), float(y), 1) for x,y in kps2.kpts]

    vis = draw_matches(img1_bgr, cvk1, img2_bgr, cvk2, matches, max_draw=150)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
