from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dino_slam3.data.tum_rgbd import TUMRGBDDataset, CameraIntrinsics
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.losses.geometry_supervision import compute_losses
from dino_slam3.utils.rich_logging import (
    print_epoch_header,
    print_metrics_table,
    print_save_notice,
    print_match_table,
)
from dino_slam3.utils.config import ensure_dir
from dino_slam3.slam.keypoints import extract_keypoints_and_descriptors


def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _make_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg["dataset"]
    cam = cfg["camera"]
    intr = CameraIntrinsics(
        fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam["cx"]), cy=float(cam["cy"])
    )

    def build(seqs, is_train: bool):
        ds_list = []
        for s in seqs:
            ds_list.append(
                TUMRGBDDataset(
                    dataset_root=dcfg["root"],
                    sequence=s,
                    intrinsics=intr,
                    frame_spacing=int(dcfg.get("frame_spacing", 1)),
                    max_frames=dcfg.get("max_frames"),
                    is_train=is_train,
                    augmentation=dcfg.get("augmentation"),
                    max_association_dt=float(dcfg.get("max_association_dt", 0.02)),
                    pad_to=int(dcfg.get("pad_to", 16)),
                )
            )
        return ConcatDataset(ds_list)

    train_ds = build(dcfg["train_sequences"], True)
    val_ds = build(dcfg["val_sequences"], False)

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def _build_model(cfg: Dict[str, Any]) -> LocalFeatureNet:
    m = cfg["model"]
    fine = m.get("fine_cnn", {})
    heads = m.get("heads", {})
    net = LocalFeatureNet(
        patch_size=int(m.get("patch_size", 16)),
        descriptor_dim=int(m.get("descriptor_dim", 128)),
        fine_channels=int(fine.get("channels", 64)),
        fine_blocks=int(fine.get("num_blocks", 6)),
        use_offset=bool(heads.get("offset", {}).get("enabled", True)),
        use_reliability=bool(heads.get("reliability", {}).get("enabled", True)),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
    )
    return net


@torch.no_grad()
def _quick_match_diagnostics(cfg: Dict[str, Any], batch: Dict[str, torch.Tensor], out1, out2) -> Dict[str, Any]:
    """
    Diagnostics:
      - keypoint counts
      - mutual NN match count
      - ratio/margin filtering
      - robust geometric inlier rate using GT depth+pose reprojection
      - reprojection error computed only on valid projections
    """
    stride = int(cfg["model"].get("stride", 4))
    det_cfg = cfg["model"]["heads"]["detector"]
    diag_cfg = cfg.get("training", {})
    margin = float(diag_cfg.get("diag_margin", 0.08))  # cosine margin gate (works like ratio test)

    # Depth gating for keypoints (critical)
    vm1 = batch.get("valid_depth1", None)
    vm2 = batch.get("valid_depth2", None)

    kps1 = extract_keypoints_and_descriptors(
        out1.heatmap, out1.desc, out1.offset, out1.reliability,
        stride=stride,
        nms_radius=int(det_cfg.get("nms_radius", 4)),
        max_keypoints=int(det_cfg.get("max_keypoints", 1024)),
        tile_size=int(det_cfg.get("tile_size", 16)),
        k_per_tile=int(det_cfg.get("k_per_tile", 8)),
        valid_mask_img=vm1,
    )[0]
    kps2 = extract_keypoints_and_descriptors(
        out2.heatmap, out2.desc, out2.offset, out2.reliability,
        stride=stride,
        nms_radius=int(det_cfg.get("nms_radius", 4)),
        max_keypoints=int(det_cfg.get("max_keypoints", 1024)),
        tile_size=int(det_cfg.get("tile_size", 16)),
        k_per_tile=int(det_cfg.get("k_per_tile", 8)),
        valid_mask_img=vm2,
    )[0]

    n1 = int(kps1.kpts.shape[0])
    n2 = int(kps2.kpts.shape[0])
    if n1 == 0 or n2 == 0:
        return {
            "kpts1": n1, "kpts2": n2, "matches": 0,
            "valid_match_ratio": 0.0,
            "inlier_rate@3px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
        }

    # mutual NN matches + margin filter
    d1 = torch.from_numpy(kps1.desc).to(out1.desc.device).float()
    d2 = torch.from_numpy(kps2.desc).to(out1.desc.device).float()

    sim = d1 @ d2.t()  # cosine similarity
    # top2 for margin gate
    top2_vals, top2_idx = torch.topk(sim, k=2, dim=1)
    nn12 = top2_idx[:, 0]
    second = top2_idx[:, 1]
    keep12 = top2_vals[:, 0] > (top2_vals[:, 1] + margin)

    nn21 = sim.argmax(dim=0)
    ids = torch.arange(n1, device=sim.device)
    mutual = (nn21[nn12] == ids)
    keep = mutual & keep12

    idx1 = ids[keep]
    idx2 = nn12[keep]
    mcount = int(idx1.numel())
    if mcount == 0:
        return {
            "kpts1": n1, "kpts2": n2, "matches": 0,
            "valid_match_ratio": 0.0,
            "inlier_rate@3px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
        }

    # Use first element only for geometry check
    depth1 = batch["depth1"][0:1]  # (1,1,H,W)
    H, W = int(depth1.shape[-2]), int(depth1.shape[-1])

    K = batch["K"]
    if K.dim() == 2:
        K = K.unsqueeze(0)
    else:
        K = K[0:1]
    T21 = batch["relative_pose"][0:1]

    xy1 = torch.from_numpy(kps1.kpts[idx1.cpu().numpy()]).to(sim.device).unsqueeze(0)  # (1,M,2)
    xy2 = torch.from_numpy(kps2.kpts[idx2.cpu().numpy()]).to(sim.device).unsqueeze(0)  # (1,M,2)

    from dino_slam3.geometry.projection import unproject, transform, project

    pts1 = unproject(depth1, K, xy1)     # (1,M,3)
    z1 = pts1[..., 2]
    valid_z1 = z1 > 1e-4

    pts2 = transform(T21, pts1)
    z2 = pts2[..., 2]
    valid_z2 = z2 > 1e-4

    xy2_gt = project(pts2, K)           # (1,M,2)

    finite = torch.isfinite(xy2_gt).all(dim=-1)
    xg, yg = xy2_gt[..., 0], xy2_gt[..., 1]
    inb = (xg >= 0) & (xg <= (W - 1)) & (yg >= 0) & (yg <= (H - 1))

    valid = (valid_z1 & valid_z2 & finite & inb)[0]  # (M,)
    valid_ratio = float(valid.float().mean().item())

    if valid.sum() == 0:
        return {
            "kpts1": n1,
            "kpts2": n2,
            "matches": mcount,
            "valid_match_ratio": valid_ratio,
            "inlier_rate@3px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
        }

    err = torch.norm(xy2_gt - xy2, dim=-1)[0]  # (M,)
    err_v = err[valid]

    inl = err_v < 3.0
    inlier_rate = float(inl.float().mean().item())

    mean_valid = float(err_v.mean().item())
    mean_inl = float(err_v[inl].mean().item()) if inl.any() else 0.0

    return {
        "kpts1": n1,
        "kpts2": n2,
        "matches": mcount,
        "valid_match_ratio": valid_ratio,
        "inlier_rate@3px": inlier_rate,
        "mean_reproj_err": mean_valid,               # <-- printed
        "mean_reproj_err_inliers": mean_inl,         # <-- printed
    }


def train(cfg: Dict[str, Any]) -> None:
    device = _device(cfg)
    train_loader, val_loader = _make_loaders(cfg)

    model = _build_model(cfg).to(device)

    # speed knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["training"]["epochs"])
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=float(cfg["training"].get("lr_min", 1e-6)))

    out_dir = ensure_dir(Path(cfg["run"]["out_dir"]) / cfg["run"]["name"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    stride = int(cfg["model"].get("stride", 4))
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_name = str(cfg["training"].get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ["bf16", "bfloat16"] else torch.float16

    # IMPORTANT: scaler only for fp16. For bf16 it should be disabled.
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        lr = opt.param_groups[0]["lr"]
        print_epoch_header(epoch, epochs, lr)

        # ---------------- train ----------------
        model.train()
        train_m = {
            "loss_total": 0.0,
            "loss_desc": 0.0,
            "loss_det": 0.0,
            "loss_sparse": 0.0,
            "loss_offset": 0.0,
            "loss_rel": 0.0,
            "valid_ratio": 0.0,
        }
        n_train = 0

        pbar = tqdm(train_loader, desc="train", leave=False)
        for it, batch in enumerate(pbar, start=1):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                out1 = model(batch["rgb1"])
                out2 = model(batch["rgb2"])

            # IMPORTANT: loss/geometry in fp32 (stability)
            with torch.autocast("cuda", enabled=False):
                losses, stats = compute_losses(batch, out1, out2, stride=stride, cfg=cfg["loss"])
                loss = losses["loss_total"]

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = batch["rgb1"].shape[0]
            n_train += bs
            for k in ["loss_total", "loss_desc", "loss_det", "loss_sparse", "loss_offset", "loss_rel"]:
                train_m[k] += float(losses[k].detach().cpu()) * bs
            train_m["valid_ratio"] += float(stats.valid_ratio) * bs

            if it % int(cfg["training"].get("log_every", 50)) == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{train_m['loss_total']/max(n_train,1):.3f}",
                        "valid%": f"{100.0*(train_m['valid_ratio']/max(n_train,1)):.1f}",
                    }
                )

        for k in train_m:
            train_m[k] /= max(n_train, 1)

        # ---------------- val ----------------
        model.eval()
        val_m = {k: 0.0 for k in train_m}
        n_val = 0
        diag_accum = {
            "kpts1": 0.0,
            "kpts2": 0.0,
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            "inlier_rate@3px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
        }
        diag_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    out1 = model(batch["rgb1"])
                    out2 = model(batch["rgb2"])

                with torch.autocast("cuda", enabled=False):
                    losses, stats = compute_losses(batch, out1, out2, stride=stride, cfg=cfg["loss"])

                bs = batch["rgb1"].shape[0]
                n_val += bs
                for k in ["loss_total", "loss_desc", "loss_det", "loss_sparse", "loss_offset", "loss_rel"]:
                    val_m[k] += float(losses[k].detach().cpu()) * bs
                val_m["valid_ratio"] += float(stats.valid_ratio) * bs

                if diag_batches < int(cfg["training"].get("diag_batches", 3)):
                    d = _quick_match_diagnostics(cfg, batch, out1, out2)
                    for kk in diag_accum:
                        diag_accum[kk] += float(d.get(kk, 0.0))
                    diag_batches += 1

        for k in val_m:
            val_m[k] /= max(n_val, 1)

        diag = {k: (v / max(diag_batches, 1)) for k, v in diag_accum.items()}

        print_metrics_table("Train", train_m)
        print_metrics_table("Val", val_m)
        print_match_table("Val diagnostics", diag)

        sch.step()

        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )
        print_save_notice(str(ckpt_path), "epoch")

        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                },
                best_path,
            )
            print_save_notice(str(best_path), f"new best val loss_total={best_val:.4f}")
