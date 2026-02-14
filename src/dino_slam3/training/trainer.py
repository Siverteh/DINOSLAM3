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
    dataset_root = Path(dcfg["root"]).expanduser()
    skip_missing = bool(dcfg.get("skip_missing_sequences", True))
    sequence_repeat = dcfg.get("sequence_repeat", {})

    def _repeat_count(seq_name: str) -> int:
        try:
            return max(1, int(sequence_repeat.get(seq_name, 1)))
        except Exception:
            return 1

    def build(seqs, is_train: bool):
        ds_list = []
        for s in seqs:
            seq_dir = dataset_root / s
            if not seq_dir.exists():
                if skip_missing:
                    print(f"[train] WARNING: skipping missing sequence: {seq_dir}")
                    continue
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            try:
                ds = TUMRGBDDataset(
                    dataset_root=dataset_root,
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
            except Exception as exc:
                if skip_missing:
                    print(f"[train] WARNING: skipping sequence {s}: {exc}")
                    continue
                raise

            rep = _repeat_count(s) if is_train else 1
            for _ in range(rep):
                ds_list.append(ds)

        if len(ds_list) == 0:
            split = "train" if is_train else "val"
            raise RuntimeError(
                f"No valid sequences available for split='{split}'. "
                "Check dataset.root, sequence names, and skip_missing_sequences."
            )
        return ConcatDataset(ds_list)

    train_ds = build(dcfg["train_sequences"], True)
    val_ds = build(dcfg["val_sequences"], False)

    tcfg = cfg["training"]
    num_workers = int(tcfg.get("num_workers", 8))
    common_loader_kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = int(tcfg.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
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


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu")


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Invalid checkpoint: could not find model state_dict")
    return state


def _load_model_weights(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(ckpt_path)
    state = _extract_model_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(
        f"[train] loaded model from {ckpt_path} "
        f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[train] missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[train] unexpected keys (first 10): {unexpected[:10]}")
    return ckpt

@torch.no_grad()
def _val_diagnostics(cfg: Dict[str, Any], batch: Dict[str, torch.Tensor], out1, out2) -> Dict[str, float]:
    import numpy as np
    import cv2
    import torch.nn.functional as F

    def _as_bK(bK: torch.Tensor) -> torch.Tensor:
        return bK.unsqueeze(0) if bK.dim() == 2 else bK

    def _as_bT(bT: torch.Tensor) -> torch.Tensor:
        return bT.unsqueeze(0) if bT.dim() == 2 else bT

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

    def _sample_depth_bilinear(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # depth: (1,1,H,W), xy: (N,2)
        H, W = depth.shape[-2:]
        x = xy[:, 0]
        y = xy[:, 1]
        gx = (x / (W - 1)) * 2 - 1
        gy = (y / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        z = F.grid_sample(depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return z.view(-1)

    def _pnp_ransac_inliers(
        xy1: torch.Tensor,
        xy2: torch.Tensor,
        depth1: torch.Tensor,
        K: torch.Tensor,
        reproj_px: float = 3.0,
        z_min_m: float = 0.10,
        max_iters: int = 2000,
        conf: float = 0.999,
    ) -> Tuple[int, float]:
        if xy1.numel() == 0:
            return 0, 0.0
        z = _sample_depth_bilinear(depth1, xy1)
        valid = torch.isfinite(z) & (z > z_min_m)
        if valid.sum().item() < 6:
            return 0, 0.0
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
        pts3d = torch.stack([X, Y, Z], dim=-1)

        obj = pts3d.detach().cpu().numpy().astype(np.float32)
        img = xy2v.detach().cpu().numpy().astype(np.float32)
        Kcv = K.detach().cpu().numpy().astype(np.float64)

        if obj.shape[0] < 6:
            return 0, 0.0

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
            return 0, 0.0
        ninl = int(inliers.shape[0])
        ratio = ninl / float(obj.shape[0])
        return ninl, float(ratio)

    stride = int(cfg["model"].get("stride", 4))
    det = cfg["model"]["heads"]["detector"]
    loss_cfg = cfg.get("loss", {})
    inlier_px = float(loss_cfg.get("diag_inlier_px", 3.0))
    pnp_px = float(loss_cfg.get("diag_pnp_px", 3.0))
    z_min = float(loss_cfg.get("z_min_m", 0.10))

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

    # only first sample
    d1 = k1.desc[0]
    d2 = k2.desc[0]
    xy1_all = k1.xy_img[0]
    xy2_all = k2.xy_img[0]

    if d1.numel() == 0 or d2.numel() == 0:
        return {
            "kpts1": float(d1.shape[0] if d1.dim() > 0 else 0),
            "kpts2": float(d2.shape[0] if d2.dim() > 0 else 0),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    idx1, idx2 = _mutual_nn_matches(d1, d2)
    if idx1 is None:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    idx1 = idx1.long()
    idx2 = idx2.long()
    xy1m = xy1_all[idx1].view(1, -1, 2)
    xy2m = xy2_all[idx2].view(1, -1, 2)

    mcount = int(xy1m.shape[1])
    if mcount == 0:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": 0.0,
            "valid_match_ratio": 0.0,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    depth1 = batch["depth1"][0:1]  # (1,1,H,W)
    Kb = _as_bK(batch["K"])[0:1]
    T21 = _as_bT(batch["relative_pose"])[0:1]

    # GT reprojection
    pts1 = unproject(depth1, Kb, xy1m)
    pts2 = transform(T21, pts1)
    xy2_gt = project(pts2, Kb)

    H, W = depth1.shape[-2:]
    xg, yg = xy2_gt[..., 0], xy2_gt[..., 1]
    inb = (xg >= 0) & (xg <= (W - 1)) & (yg >= 0) & (yg <= (H - 1)) & torch.isfinite(xy2_gt).all(dim=-1)
    valid_ratio = float(inb.float().mean().item())

    err = torch.linalg.norm(xy2_gt - xy2m, dim=-1)[0]
    err_v = err[inb[0]]
    if err_v.numel() == 0:
        return {
            "kpts1": float(d1.shape[0]),
            "kpts2": float(d2.shape[0]),
            "matches": float(mcount),
            "valid_match_ratio": valid_ratio,
            f"inlier_rate@{inlier_px:.0f}px": 0.0,
            f"inliers@{inlier_px:.0f}px": 0.0,
            f"pnp_inliers@{pnp_px:.0f}px": 0.0,
            f"pnp_inlier_rate@{pnp_px:.0f}px": 0.0,
            "mean_reproj_err": 0.0,
            "mean_reproj_err_inliers": 0.0,
            "median_reproj_err_inliers": 0.0,
        }

    inl = err_v < inlier_px
    inlier_count = int(inl.sum().item())
    inlier_rate = float(inl.float().mean().item())
    mean_all = float(err_v.mean().item())
    mean_inl = float(err_v[inl].mean().item()) if inl.any() else 0.0
    med_inl = float(err_v[inl].median().item()) if inl.any() else 0.0

    # PnP-RANSAC on CPU (OpenCV)
    ninl_pnp, rate_pnp = _pnp_ransac_inliers(
        xy1=xy1m[0],
        xy2=xy2m[0],
        depth1=depth1,
        K=Kb[0],
        reproj_px=pnp_px,
        z_min_m=z_min,
    )

    return {
        "kpts1": float(d1.shape[0]),
        "kpts2": float(d2.shape[0]),
        "matches": float(mcount),
        "valid_match_ratio": valid_ratio,

        f"inliers@{inlier_px:.0f}px": float(inlier_count),
        f"inlier_rate@{inlier_px:.0f}px": float(inlier_rate),

        f"pnp_inliers@{pnp_px:.0f}px": float(ninl_pnp),
        f"pnp_inlier_rate@{pnp_px:.0f}px": float(rate_pnp),

        "mean_reproj_err": mean_all,
        "mean_reproj_err_inliers": mean_inl,
        "median_reproj_err_inliers": med_inl,
    }


def train(cfg: Dict[str, Any]) -> None:
    device = _device(cfg)
    tcfg = cfg["training"]
    out_dir = ensure_dir(Path(cfg["run"]["out_dir"]) / cfg["run"]["name"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    train_loader, val_loader = _make_loaders(cfg)
    model = _build_model(cfg).to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    epochs = int(tcfg["epochs"])
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tcfg.get("lr_min", 1e-6)))
    start_epoch = 1

    init_ckpt = tcfg.get("init_checkpoint")
    resume_ckpt = tcfg.get("resume_checkpoint")
    init_strict = bool(tcfg.get("init_strict", False))
    if init_ckpt and resume_ckpt:
        raise ValueError("Use only one of training.init_checkpoint or training.resume_checkpoint.")

    best_val = float("inf")
    if resume_ckpt:
        ckpt = _load_model_weights(model, resume_ckpt, strict=init_strict)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            sch.load_state_dict(ckpt["scheduler"])
        else:
            done_epochs = int(ckpt.get("epoch", 0))
            for _ in range(done_epochs):
                sch.step()
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"[train] resuming from epoch {start_epoch}")
    elif init_ckpt:
        _load_model_weights(model, init_ckpt, strict=init_strict)

    stride = int(cfg["model"].get("stride", 4))
    use_amp = bool(tcfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(tcfg.get("amp_dtype", "bf16")).lower() in ("bf16", "bfloat16") else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 0.0) or 0.0)

    for epoch in range(start_epoch, epochs + 1):
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
                if grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
        inlier_px = float(cfg.get("loss", {}).get("diag_inlier_px", 3.0))
        pnp_px = float(cfg.get("loss", {}).get("diag_pnp_px", 3.0))

        diag_keys = [
            "kpts1","kpts2","matches","valid_match_ratio",
            f"inliers@{inlier_px:.0f}px",
            f"inlier_rate@{inlier_px:.0f}px",
            f"pnp_inliers@{pnp_px:.0f}px",
            f"pnp_inlier_rate@{pnp_px:.0f}px",
            "mean_reproj_err","mean_reproj_err_inliers","median_reproj_err_inliers"
        ]
        diag_acc = {k: 0.0 for k in diag_keys}

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
        if bool(tcfg.get("save_every_epoch", True)):
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "best_val_loss": float(best_val),
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
                    "scheduler": sch.state_dict(),
                    "best_val_loss": float(best_val),
                    "config": cfg,
                },
                best_path,
            )
            print_save_notice(str(best_path), f"new best val loss_total={best_val:.4f}")
