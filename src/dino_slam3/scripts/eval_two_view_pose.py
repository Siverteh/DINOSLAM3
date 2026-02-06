from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from dino_slam3.utils.config import load_config
from dino_slam3.data.tum_rgbd import TUMRGBDDataset, CameraIntrinsics
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints import extract_keypoints_and_descriptors


def _K_np(K: torch.Tensor) -> np.ndarray:
    if K.dim() == 3:
        K = K[0]
    return K.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--pairs", type=int, default=50)
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

    stride = int(cfg["model"].get("stride", 4))
    det = cfg["model"]["heads"]["detector"]

    inlier_rates = []
    pnp_success = 0
    rot_errs = []
    trans_errs = []

    for i, batch in enumerate(dl):
        if i >= args.pairs:
            break
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        out1 = model(batch["rgb1"])
        out2 = model(batch["rgb2"])

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

        if k1.kpts.shape[0] < 20 or k2.kpts.shape[0] < 20:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(k1.desc, k2.desc)
        if len(matches) < 20:
            continue

        # inlier by GT reprojection using depth1+pose
        depth1 = batch["depth1"][0, 0].detach().cpu().numpy()
        K = _K_np(batch["K"])
        T21 = batch["relative_pose"][0].detach().cpu().numpy()

        pts3d = []
        pts2d = []
        errs = []
        for m in matches[:300]:
            x1, y1 = k1.kpts[m.queryIdx]
            x2, y2 = k2.kpts[m.trainIdx]
            xi = int(round(x1))
            yi = int(round(y1))
            if yi < 0 or yi >= depth1.shape[0] or xi < 0 or xi >= depth1.shape[1]:
                continue
            z = float(depth1[yi, xi])
            if z <= 0.0:
                continue

            # unproject
            X = (x1 - K[0, 2]) * z / K[0, 0]
            Y = (y1 - K[1, 2]) * z / K[1, 1]
            P1 = np.array([X, Y, z, 1.0], dtype=np.float64)
            P2 = (T21 @ P1)[:3]
            if P2[2] <= 1e-6:
                continue
            u = K[0, 0] * (P2[0] / P2[2]) + K[0, 2]
            v = K[1, 1] * (P2[1] / P2[2]) + K[1, 2]
            e = float(np.hypot(u - x2, v - y2))
            errs.append(e)

            pts3d.append(P2.astype(np.float64))   # 3D in cam2 frame
            pts2d.append([x2, y2])

        if len(errs) < 20:
            continue

        inl = np.mean(np.array(errs) < 3.0)
        inlier_rates.append(float(inl))

        # PnP in cam2: use (3D in cam2, 2D in cam2) is degenerate -> use 3D in cam1 with 2D in cam2 instead
        pts3d_cam1 = []
        pts2d_cam2 = []
        for m in matches[:300]:
            x1, y1 = k1.kpts[m.queryIdx]
            x2, y2 = k2.kpts[m.trainIdx]
            xi = int(round(x1))
            yi = int(round(y1))
            z = float(depth1[yi, xi]) if (0 <= yi < depth1.shape[0] and 0 <= xi < depth1.shape[1]) else 0.0
            if z <= 0.0:
                continue
            X = (x1 - K[0, 2]) * z / K[0, 0]
            Y = (y1 - K[1, 2]) * z / K[1, 1]
            pts3d_cam1.append([X, Y, z])
            pts2d_cam2.append([x2, y2])

        if len(pts3d_cam1) >= 30:
            pts3d_cam1 = np.asarray(pts3d_cam1, np.float64)
            pts2d_cam2 = np.asarray(pts2d_cam2, np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_cam1, pts2d_cam2, K, None,
                iterationsCount=200,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                pnp_success += 1
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.reshape(3)

                # compare to GT T21 (cam1->cam2)
                R_gt = T21[:3, :3]
                t_gt = T21[:3, 3]

                dR = R @ R_gt.T
                ang = np.arccos(np.clip((np.trace(dR) - 1) * 0.5, -1.0, 1.0))
                rot_errs.append(float(ang))
                trans_errs.append(float(np.linalg.norm(t - t_gt)))

    print(f"Pairs evaluated: {len(inlier_rates)}")
    if inlier_rates:
        print(f"Inlier rate @3px: mean={np.mean(inlier_rates):.3f}  median={np.median(inlier_rates):.3f}")
    print(f"PnP success: {pnp_success}/{len(inlier_rates)}")
    if rot_errs:
        print(f"Rot err (rad): mean={np.mean(rot_errs):.4f}")
        print(f"Trans err (m): mean={np.mean(trans_errs):.4f}")


if __name__ == "__main__":
    main()
