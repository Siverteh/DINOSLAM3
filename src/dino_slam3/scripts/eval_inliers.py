from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_slam3.utils.config import load_config
from dino_slam3.data.tum_rgbd import TUMRGBDDataset
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch
from dino_slam3.geometry.projection import unproject, transform, project


def _mutual_nn_matches(d1: torch.Tensor, d2: torch.Tensor):
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sequence", type=str, required=True)
    ap.add_argument("--num_batches", type=int, default=50)
    ap.add_argument("--inlier_px", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = cfg["dataset"]
    assoc = dcfg.get("association", {})

    ds = TUMRGBDDataset(
        dataset_root=dcfg["root"],
        sequence=args.sequence,
        frame_spacing_min=1,
        frame_spacing_max=1,
        is_train=False,
        pad_to=int(dcfg.get("pad_to", 16)),
        augmentation=None,
        max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
        max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
        max_frames=dcfg.get("max_frames"),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

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

    det = mcfg["heads"]["detector"]
    stride = int(mcfg.get("stride", 4))

    inlier_rates = []
    match_counts = []

    for it, batch in enumerate(tqdm(dl), start=1):
        if it > args.num_batches:
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
        xy1 = k1.xy_img[0:1]
        xy2 = k2.xy_img[0:1]

        idx1, idx2 = _mutual_nn_matches(d1, d2)
        if idx1 is None:
            continue

        match_counts.append(float(idx1.numel()))

        xy1m = xy1[:, idx1, :]
        xy2m = xy2[:, idx2, :]

        pts1 = unproject(batch["depth1"], batch["K"], xy1m)
        pts2 = transform(batch["relative_pose"], pts1)
        xy2_gt = project(pts2, batch["K"])

        err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)[0]
        inlier = (err < float(args.inlier_px)).float().mean().item()
        inlier_rates.append(inlier)

    if not inlier_rates:
        print("No valid pairs.")
        return

    mean_inlier = sum(inlier_rates) / len(inlier_rates)
    mean_matches = sum(match_counts) / len(match_counts) if match_counts else 0.0
    print(f"Sequence: {args.sequence}")
    print(f"Pairs evaluated: {len(inlier_rates)}")
    print(f"Mean matches: {mean_matches:.1f}")
    print(f"Mean inlier_rate@{args.inlier_px:.1f}px: {mean_inlier:.3f}")


if __name__ == "__main__":
    main()
