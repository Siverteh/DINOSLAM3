from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dino_slam3.data.tum_rgbd import TUMRGBDDataset
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.losses.two_view_loss import compute_losses
from dino_slam3.utils.config import ensure_dir
from dino_slam3.utils.rich_logging import (
    print_epoch_header,
    print_metrics_table,
    print_save_notice,
    print_match_table,
)
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch
from dino_slam3.geometry.projection import unproject, transform, project

def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)

def _make_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg["dataset"]
    assoc = dcfg.get("association", {})

    def build(seqs, is_train: bool):
        ds_list = []
        for s in seqs:
            ds_list.append(
                TUMRGBDDataset(
                    dataset_root=dcfg["root"],
                    sequence=s,
                    frame_spacing_min=int(dcfg.get("frame_spacing_min", 1)),
                    frame_spacing_max=int(dcfg.get("frame_spacing_max", 4)),
                    max_frames=dcfg.get("max_frames"),
                    pad_to=int(dcfg.get("pad_to", 16)),
                    is_train=is_train,
                    augmentation=dcfg.get("augmentation"),
                    max_rgb_depth_dt=float(assoc.get("max_rgb_depth_dt", 0.02)),
                    max_rgb_gt_dt=float(assoc.get("max_rgb_gt_dt", 0.02)),
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
        num_workers=int(tcfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return train_loader, val_loader

def _build_model(cfg: Dict[str, Any]) -> LocalFeatureNet:
    m = cfg["model"]
    return LocalFeatureNet(
        dinov3_name=m["dinov3"]["name_or_path"],
        patch_size=int(m.get("patch_size", 16)),
        descriptor_dim=int(m["heads"].get("descriptor_dim", 256)),
        fine_channels=int(m["fine_cnn"].get("channels", 96)),
        fine_blocks=int(m["fine_cnn"].get("num_blocks", 8)),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
        use_offset=bool(m["heads"].get("offset", {}).get("enabled", True)),
        use_reliability=bool(m["heads"].get("reliability", {}).get("enabled", True)),
        dinov3_dtype=str(m["dinov3"].get("dtype", "bf16")),
    )

@torch.no_grad()
def _val_diagnostics(cfg: Dict[str, Any], batch: Dict[str, torch.Tensor], out1, out2) -> Dict[str, float]:
    stride = int(cfg["model"].get("stride", 4))
    det = cfg["model"]["heads"]["detector"]

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

    B = batch["rgb1"].shape[0]
    # Evaluate only first sample for simplicity
    xy1 = k1.xy_img[0:1]
    xy2 = k2.xy_img[0:1]
    d1 = k1.desc[0]
    d2 = k2.desc[0]

    if d1.numel() == 0 or d2.numel() == 0:
        return {"kpts1": 0.0, "kpts2": 0.0, "matches": 0.0, "valid_match_ratio": 0.0, "inlier_rate@3px": 0.0,
                "mean_reproj_err": 0.0, "mean_reproj_err_inliers": 0.0, "median_reproj_err_inliers": 0.0}

    sim = (d1 @ d2.t())
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    ids = torch.arange(sim.shape[0], device=sim.device)
    mutual = (nn21[nn12] == ids)
    idx1 = ids[mutual]
    idx2 = nn12[mutual]

    mcount = int(idx1.numel())
    if mcount == 0:
        return {"kpts1": float(d1.shape[0]), "kpts2": float(d2.shape[0]), "matches": 0.0, "valid_match_ratio": 0.0,
                "inlier_rate@3px": 0.0, "mean_reproj_err": 0.0, "mean_reproj_err_inliers": 0.0, "median_reproj_err_inliers": 0.0}

    depth1 = batch["depth1"][0:1]
    K = batch["K"]
    if K.dim() == 2:
        K = K.unsqueeze(0)
    T21 = batch["relative_pose"]
    if T21.dim() == 2:
        T21 = T21.unsqueeze(0)

    xy1m = xy1[:, idx1, :]
    xy2m = xy2[:, idx2, :]

    pts1 = unproject(depth1, K[0:1], xy1m)
    pts2 = transform(T21[0:1], pts1)
    xy2_gt = project(pts2, K[0:1])

    H, W = depth1.shape[-2:]
    xg, yg = xy2_gt[..., 0], xy2_gt[..., 1]
    inb = (xg >= 0) & (xg <= (W - 1)) & (yg >= 0) & (yg <= (H - 1)) & torch.isfinite(xy2_gt).all(dim=-1)
    valid_ratio = float(inb.float().mean().item())

    err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)[0]
    err_v = err[inb[0]]
    if err_v.numel() == 0:
        return {"kpts1": float(d1.shape[0]), "kpts2": float(d2.shape[0]), "matches": float(mcount),
                "valid_match_ratio": valid_ratio, "inlier_rate@3px": 0.0,
                "mean_reproj_err": 0.0, "mean_reproj_err_inliers": 0.0, "median_reproj_err_inliers": 0.0}

    inl = err_v < 3.0
    inlier_rate = float(inl.float().mean().item())
    mean_all = float(err_v.mean().item())
    mean_inl = float(err_v[inl].mean().item()) if inl.any() else 0.0
    med_inl = float(err_v[inl].median().item()) if inl.any() else 0.0

    return {
        "kpts1": float(d1.shape[0]),
        "kpts2": float(d2.shape[0]),
        "matches": float(mcount),
        "valid_match_ratio": valid_ratio,
        "inlier_rate@3px": inlier_rate,
        "mean_reproj_err": mean_all,
        "mean_reproj_err_inliers": mean_inl,
        "median_reproj_err_inliers": med_inl,
    }

def train(cfg: Dict[str, Any]) -> None:
    device = _device(cfg)
    train_loader, val_loader = _make_loaders(cfg)
    model = _build_model(cfg).to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))

    epochs = int(cfg["training"]["epochs"])
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=float(cfg["training"].get("lr_min", 1e-6)))

    out_dir = ensure_dir(Path(cfg["run"]["out_dir"]) / cfg["run"]["name"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    stride = int(cfg["model"].get("stride", 4))
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg["training"].get("amp_dtype", "bf16")).lower() in ("bf16", "bfloat16") else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        lr = opt.param_groups[0]["lr"]
        print_epoch_header(epoch, epochs, lr)

        # train
        model.train()
        train_m = {k: 0.0 for k in ["loss_total","loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity","valid_ratio"]}
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

            with torch.autocast("cuda", enabled=False):
                losses, stats = compute_losses(batch, out1, out2, cfg=cfg["loss"], epoch=epoch, stride=stride)
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
            train_m["loss_total"] += float(loss.detach().cpu()) * bs
            for k in ["loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity"]:
                train_m[k] += float(losses.get(k, torch.tensor(0.0)).detach().cpu()) * bs
            train_m["valid_ratio"] += float(stats.valid_ratio) * bs

            if it % int(cfg["training"].get("log_every", 50)) == 0:
                pbar.set_postfix({"loss": f"{train_m['loss_total']/max(n_train,1):.3f}", "valid%": f"{100.0*(train_m['valid_ratio']/max(n_train,1)):.1f}"})

        for k in train_m:
            train_m[k] /= max(n_train, 1)

        # val
        model.eval()
        val_m = {k: 0.0 for k in train_m}
        n_val = 0
        diag_acc = {k: 0.0 for k in ["kpts1","kpts2","matches","valid_match_ratio","inlier_rate@3px","mean_reproj_err","mean_reproj_err_inliers","median_reproj_err_inliers"]}
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
                    losses, stats = compute_losses(batch, out1, out2, cfg=cfg["loss"], epoch=epoch, stride=stride)

                bs = batch["rgb1"].shape[0]
                n_val += bs
                val_m["loss_total"] += float(losses["loss_total"].detach().cpu()) * bs
                for k in ["loss_desc","loss_repeat","loss_rel","loss_refine","loss_pose","loss_sparsity"]:
                    val_m[k] += float(losses.get(k, torch.tensor(0.0)).detach().cpu()) * bs
                val_m["valid_ratio"] += float(stats.valid_ratio) * bs

                if diag_batches < int(cfg["training"].get("diag_batches", 3)):
                    d = _val_diagnostics(cfg, batch, out1, out2)
                    for kk in diag_acc:
                        diag_acc[kk] += float(d.get(kk, 0.0))
                    diag_batches += 1

        for k in val_m:
            val_m[k] /= max(n_val, 1)
        diag = {k: v / max(diag_batches, 1) for k, v in diag_acc.items()}

        print_metrics_table("Train", train_m)
        print_metrics_table("Val", val_m)
        print_match_table("Val diagnostics", diag)

        sch.step()

        # save
        if bool(cfg["training"].get("save_every_epoch", True)):
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg}, ckpt_path)
            print_save_notice(str(ckpt_path), "epoch")

        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            best_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg}, best_path)
            print_save_notice(str(best_path), f"new best val loss_total={best_val:.4f}")
